# === Final: Sub-table + Summary + SQL Reasoning + 저장 (예시 프롬프트 반영) ===
import re
import os
import csv
import json
import sqlite3
import pandas as pd
from io import StringIO
from openai import OpenAI

from utils.preprocess import *
from utils.prompt_wtq import *
from subtable_extractor_euclid import ColumnSimilarityAnalyzer

# OpenAI client
client = OpenAI()
analyzer = ColumnSimilarityAnalyzer()

# ---------------------- 공통 함수 ----------------------
def extract_sql_only(text: str) -> str:
    text = re.sub(r"```sql", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    lines = text.splitlines()
    sql_lines = []
    capture = False
    for line in lines:
        if line.strip().lower().startswith("sql:"):
            # SQL: 뒤에 같은 줄에 있으면 추출
            part = line.split("SQL:", 1)[1].strip()
            if part:
                sql_lines.append(part)
            capture = True
            continue
        if capture:
            # SQL: 뒤에 오는 줄들도 다 포함
            sql_lines.append(line.strip())
    return " ".join(sql_lines).strip()

def extract_numeric_columns(summary_text: str):
    numeric_cols = []
    # Column Analysis 블록만 자르기
    if "**Column Analysis:**" in summary_text:
        col_block = summary_text.split("**Column Analysis:**")[1]
        if "**Outlier Row Analysis:**" in col_block:
            col_block = col_block.split("**Outlier Row Analysis:**")[0]

        # 정규식으로 Column + Data type 매칭
        pattern = r"\d+\.\s+(.*?)\n\s+- Data type:\s+(\w+)"
        matches = re.findall(pattern, col_block)

        for col_name, dtype in matches:
            if dtype.lower() == "numeric":
                numeric_cols.append(col_name.strip())

    return numeric_cols

def parse_answer(response: str) -> str:
    """LLM 응답에서 Answer 부분만 추출"""
    output_ans = response
    try:
        output_ans = response.split("Answer:")[1]
    except:
        output_ans = "" + response
    match = re.search(r'(The|the) answer is ([^\.]+)\.$', output_ans)
    if match:
        output_ans = match.group(2).strip('"')
    return output_ans.strip().lower()


def call_gpt_table_summary(table_markdown: str, outliers: list, model: str = "gpt-3.5-turbo") -> str:
    prompt = f"""
You are a strict table analysis assistant.
Here is a partial preview of a table in markdown format:

{table_markdown}

Additionally, the following ROW INDICES were identified as OUTLIERS (atypical rows) by statistical similarity analysis:
{json.dumps(outliers, ensure_ascii=False)}

**Important Rules:**
- Not all candidate outlier rows are truly special. Some are just ordinary data rows.
- Do not treat empty or missing cells as distinctive values. They are simple missing data and must not be treated as distinctive outliers.

**Your task:**
1. For every column, describe:
   - Data type (numeric, categorical, etc.)
   - 4 representative example values
   - Role of the column (identifier, descriptor, measure, etc.)

2. Outlier Row Analysis:
- For each candidate row, explicitly mention the exact values that make it distinctive (e.g., “National Cup = Semifinals”, “Reg. Season = 5th?”).
- If these values are clearly different from the majority of the table, explain why this makes the row an outlier.
- If the values are ambiguous markers like "?" or "5th?" but are ordinary within the context of the table, state explicitly that they are ordinary values and not special.
- If the row is consistent with ordinary entries, state that it is an ordinary row and requires no special handling.
- Focus only on candidate rows; do not discuss rows outside the candidate list.
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# ---------------------- SQL Reasoning ----------------------
def tabsqlify_wtq(T, title, tab_col, question, full_table, summary, log_path=None, gold_answer=None):
    """col 기반 → sql reasoning → fallback, 모든 과정 로그 저장"""
    response = ""
    output_ans = ""
    linear_table = ""
    result_sql = pd.DataFrame()

    # === SQLite 메모리 DB 연결 ===
    conn = sqlite3.connect(":memory:")
    T.to_sql("T", conn, index=False, if_exists="replace")

    # === 로그 준비 ===
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        flog = open(log_path, "w", encoding="utf-8")
    else:
        flog = None

    def log(msg):
        print(msg)
        if flog:
            flog.write(msg + "\n")

    if flog:
        flog.write("=== Table Summary ===\n")
        flog.write(summary + "\n\n")
        flog.write("=== Full Table Preview ===\n")
        flog.write(T.to_markdown(index=False) + "\n\n")
        flog.flush()

    # 질문 기반 최종 SQL 생성
    prompt_sql = gen_table_decom_prompt(title, tab_col, question, full_table, summary = summary)
    sql_final = get_sql_3(prompt_sql)
    log(f"Generated Reasoning SQL: {sql_final}")
    sql_final = extract_sql_only(sql_final) 
    log(f"Generated Final SQL: {sql_final}")

    try:
        result_sql = pd.read_sql_query(sql_final, conn)
        log(f"SQL 실행 결과 셀 수: {result_sql.size}")
        if not result_sql.empty:
            log("=== SQL 실행 결과 미리보기 ===")
            log(result_sql.to_markdown(index=False))
    except Exception as e:
        log(f"[Error] Final SQL 실행 실패: {e}")
        result_sql = pd.DataFrame()

    if not result_sql.empty:
        if result_sql.isnull().all().all():  # 모든 값이 NaN
            log("[Warning] SQL 결과가 전부 None/NaN → Fallback으로 전환")
            result_sql = pd.DataFrame()  # 강제로 fallback
        else:
            # 정상 결과 있으면 reasoning 단계로 진행
            linear_table = table_linearization(result_sql, style='pipe')
            reasoning_prompt = generate_sql_answer_prompt(title, sql_final, linear_table, question)
            log(f"Reasoning Prompt:\n{reasoning_prompt}")

            response = get_answer(reasoning_prompt)
            output_ans = parse_answer(response)
            log(f"Prediction: {output_ans}")
            log(f"[Gold Answer] {gold_answer}\n")

            if flog: flog.close()
            conn.close()

            return sql_final, result_sql, response, output_ans, linear_table

    log("[Fallback] SQL 결과가 없으므로 Full Table 기반 추론으로 전환")
    sql = "select * from T"
    result = T.copy() # copy full table
    linear_table = table_linearization(result, style='pipe')
    prompt_ans = gen_full_table_prompt(title, tab_col, linear_table, question)

    log(f"[Fallback] Prompt SQL:\n{prompt_ans}")
    response = get_answer(prompt_ans)
    log(f"[Fallback] Raw Response: {response}")

    output_ans = parse_answer(response)
    log(f"[Fallback] Prediction: {output_ans}")
    log(f"[Gold Answer] {gold_answer}\n")

    if flog: flog.close()
    return sql, result, response, output_ans, linear_table


# ---------------------- 메인 실행 ----------------------
if __name__ == "__main__":
    path = 'datasets/wtq.jsonl'
    start = 174
    end = start + 1   # ✅ 여러 개 돌리려면 조정

    table_ids = list(range(start, end))
    base_output = "outputs_one"
    subtable_dir = os.path.join(base_output, "subtables")
    summary_dir = os.path.join(base_output, "summary_log")

    os.makedirs(base_output, exist_ok=True)
    os.makedirs(subtable_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    correct = 0
    t_samples = 0

    with open(path, encoding='utf-8') as f1, \
         open(os.path.join(base_output, 'wtq_results.jsonl'), 'a', encoding='utf-8') as fw, \
         open(os.path.join(base_output, 'wtq_results.csv'), 'a', newline='', encoding='utf-8') as fcsv:

        writer = csv.writer(fcsv)
        header = ['id', 'question', 'answer', 'prediction', 'sql', 'response',
                  'summary', 'r_num_cell', 't_num_cell']
        writer.writerow(header)

        for i, l in enumerate(f1):
            if i in table_ids:
                dic = json.loads(l)
                idx = dic['id']
                title = dic['title']
                question = dic['question']
                answer = ','.join(dic['answer']).lower()
                table_id = dic['table_id']

                print(f"\n=== ID: {idx}, Q: {question}, Gold: {answer} ===")

                # === 1) Sub-table 캐싱 ===
                subtable_path = os.path.join(subtable_dir, f"{table_id.replace('/','_')}.tsv")
                outliers = []

                if os.path.exists(subtable_path):
                    print(f"[Cache Hit] Using cached subtable: {subtable_path}")
                    with open(subtable_path, "r", encoding="utf-8") as f:
                        result_lines = f.read().splitlines()
                else:
                    print(f"[Cache Miss] Creating subtable for {table_id}")
                    result_lines, selected_rows, outliers = analyzer.process_jsonl_record(
                        jsonl_path=path,
                        index=i,
                        log_file_path=os.path.join(subtable_dir, f"{table_id.replace('/','_')}_log.txt")
                    )
                    with open(subtable_path, "w", encoding="utf-8") as f:
                        for line in result_lines:
                            f.write(line + "\n")

                # DataFrame 변환 (마크다운용)
                csv_content = "\n".join([result_lines[0]] + result_lines[2:])
                subtable_df = pd.read_csv(StringIO(csv_content), sep="\t")
                preview_table = subtable_df.to_markdown(index=False)

                # === 2) 요약 캐싱 ===
                summary_path = os.path.join(summary_dir, f"{table_id.replace('/','_')}_summary.txt")
                if os.path.exists(summary_path):
                    print(f"[Cache Hit] Using cached summary: {summary_path}")
                    with open(summary_path, "r", encoding="utf-8") as f:
                        summary = f.read().strip()
                else:
                    print(f"[Cache Miss] Creating summary for {table_id}")
                    summary = call_gpt_table_summary(preview_table, outliers)
                    with open(summary_path, "w", encoding="utf-8") as f:
                        f.write(summary)

                print("\n[Summary]\n", summary)

                # === 3) SQL Reasoning ===
                T = dict2df([dic['table']['header']] + dic['table']['rows'])
                T = T.assign(row_number=range(len(T)))
                row_number = T.pop('row_number')
                T.insert(0, 'row_number', row_number)

                # --- 여기서 numeric 컬럼만 후처리 ---
                numeric_cols = extract_numeric_columns(summary)

                def normalize_numeric(x):
                    if x is None or str(x).lower() in ["nan", "none", ""]:
                        return None
                    s = str(x).strip()
                    s = re.sub(r"[£$€,]", "", s)   # 화폐기호 제거
                    match = re.search(r"-?\d+(\.\d+)?", s)
                    if match:
                        try:
                            val = float(match.group())
                            return int(val) if val.is_integer() else val
                        except:
                            return s
                    return s

                for col in T.columns:
                    if col.lower() in [c.lower() for c in numeric_cols]:
                        T[col] = T[col].map(normalize_numeric)

                tab_col = ", ".join(T.columns)
                conn_tmp = sqlite3.connect(":memory:")
                T.to_sql("T", conn_tmp, index=False, if_exists="replace")
                full_table = table_linearization(T, style='pipe')

                sql, result, response, output_ans, linear_table = tabsqlify_wtq(
                    T, title, tab_col, question, full_table, summary,
                    log_path=os.path.join("outputs_one/sql_logs", f"{i}.txt"),  # ✅ idx만 사용
                    gold_answer=answer
                )

                # === 평가 ===
                output_ans = output_ans.lower()
                if output_ans.strip() == answer or \
                   output_ans.strip().find(answer) != -1 or \
                   answer.strip().find(output_ans.strip()) != -1:
                    correct += 1

                t_samples += 1
                acc = correct / (t_samples + 0.0001)
                print(f"\n[Prediction] {output_ans} | [Gold] {answer} | Acc={acc:.4f}")

                # === 저장 ===
                tmp = {
                    'idx': i,   # ✅ 실행 시점 인덱스 (start 기준)
                    'question_id': idx,   # ✅ 원래 JSON의 "id" (예: "nu-165")
                    'question': question,
                    'response': response,
                    'prediction': output_ans,
                    'answer': answer,
                    'table_id': table_id
                }
                fw.write(json.dumps(tmp) + '\n')

                data = [idx, question, answer, output_ans.strip(), sql, response,
                        summary, result.size, T.size]
                writer.writerow(data)

    print(f"\n✅ Final Accuracy: {correct}/{t_samples} ({correct / (t_samples + 0.0001):.4f})")
        