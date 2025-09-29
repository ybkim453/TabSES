import time
import os
from openai import OpenAI
import tiktoken

# ---------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY", None),  # this is also the default, it can be omitted
)

p_wtq_full = """
The provided summary is the primary structure for interpreting the Full Table.  
Always start by carefully examining the summary to identify which columns, data types, and special cases are relevant to the question.  
Then, use the full table to confirm and extract the exact values based on the hints from the summary.  
Your reasoning must explicitly combine evidence from both the summary and the full table before giving the final answer.  
Final output must follow the format:
Final Answer: AnswerName1, AnswerName2...

How many cyclists in the top 10 were French?
To answer this question, I will use both the summary and the full table.

From the summary:
- The **Cyclist** column is text and includes both name and nationality in parentheses, e.g., "Stéphane Goubert (FRA)".
- Special / Atypical Rows: none were listed, so every row in the preview represents an individual cyclist.
- Preprocessing Artifacts: none critical except "s.t." in the Time column, which stands for "same time". This does not affect nationality counting.

From the full table:
- The cyclists with "(FRA)" are:
  1. Stéphane Goubert (Rank 8, FRA)
  2. David Moncoutié (Rank 10, FRA)

Thus, by combining the summary’s description (Cyclist column contains nationality; no atypical rows interfere)  
with the explicit entries in the full table, we can conclude there are 2 French cyclists in the top 10.

Final Answer: Stéphane Goubert, David Moncoutié
"""

p_sql_answer_wtq = """
You are a strict SQL reasoning assistant. 
Your task is to return the final answer **directly from the SQL result table**.

Rules:
- Do not generate new SQL or mention schema/summary.
- Always write a short reasoning step before the Final Answer.
- If one cell is returned: briefly confirm it is the only result.
- If multiple rows/columns: reason step by step to filter or choose.
- Ensure the final answer format is only "Final Answer: AnswerName1, AnswerName2..." form, no other form. And ensure the final answer is a number or entity names, as short as possible, without any explanation.

Examples:

Table_title: Olympic Events
SQL: select "event" from T where cast("year" as integer) = 2008

event
100m Sprint
Long Jump
Marathon
Swimming

Question: how many events were held in 2008?
Answer: Let's think step by step.  
There are 4 rows returned for 2008, each representing one event.  
Final Answer: 4


Table_title: 2013-14 Toros Mexico season
SQL: select "opponent", "date" from T where "date" > '2014-11-10' order by "date" asc

opponent          | date
Turlock Express   | 2014-11-14

Question: who did the team play after the las vegas legends on november 10?
Answer: Let's think step by step.  
The next opponent listed after November 10 is Turlock Express.  
Final Answer: Turlock Express
"""

p_sql_wtq_with_summary = """
You will receive the **full table, the question, and the summary**.
Generate SQL with a detailed explanation and the final query.  
Exclude irrelevant rows (e.g., "Total", "Average", "World", "N/A", "?", "—") if the summary marks them as such.  

**Important Rule:**
- When using COUNT, you must **never use DISTINCT**.
- Whenever a calculation (e.g., SUM, AVG, MAX, MIN, arithmetic comparisons) is required, always CAST the column values to INTEGER or FLOAT in SQL.  
  Example: `SUM(CAST("Total Wins" AS INTEGER))`, `SUM(CAST("Total Wins" AS FLOAT))`, 
- Never use `HAVING` for conditions on zero aggregrates
- All missing values in the table are stored as NULL
- When filtering categorical values that might appear as part of a longer string 
  (e.g., "No playoff" vs "Champion (no playoff)"), always use a `LIKE '%value%'` condition instead of `=`.

Always **read the summary first** and start with an explanation line beginning with 'Explanation:'.  
- If the summary labels a row as "ordinary", you must not exclude it.  
- If the summary marks a row as an outlier and it is relevant to the question, explicitly exclude or handle it in your SQL.  
- In the explanation, explicitly mention which columns and values from the summary are relevant to answer the question.  
- You must use the provided summary carefully: if the summary specifies data types (e.g., Numeric, Categorical) or shows example values, use this information to decide how to construct conditions or comparisons in SQL.   
- Outlier notes in the summary must also be considered: explain whether they should be excluded from the SQL.  
- Never ignore the summary: every reasoning step must be grounded in it.  

**If the summary does not mention a column or entity explicitly, but the question specifies a clear condition (e.g., a player name, a competition, a year), you must use that condition directly in the SQL filter based on the most relevant column from the full table preview.**

After the explanation, on the next line, output the SQL query prefixed with 'SQL:'.  
- Always wrap column names in double quotes (e.g., SELECT "column_name" FROM T).  
- The SQL must directly follow from the reasoning in the Explanation section.  

If the SQL alone cannot fully answer the question, state in the Explanation how the summary information should be used alongside SQL execution to derive the final answer.  

Your final output must always include **both** the Explanation and the SQL, in that order.

Examples :
SQLite table properties:

Table: List of Olympic Events (row_number, year, sport, event, athlete, medal)

Q: how many events were held in 2008?
Explanation: The summary indicates that the "year" column is numeric, and the "event" column is categorical with each row representing a single Olympic event.
To answer the question, we count the total number of rows where "year" = 2008.
It is important to note that we must count all rows, not just distinct events, because each row already corresponds to one event. Using DISTINCT here would undercount the total.
SQL: select count("event") from T where cast("year" as integer) = 2008;

SQLite table properties:

Table: List of storms on the Great Lakes(row_number,ship,type_of_vessel,lake,location,lives_lost)

Q: how many more ships were wrecked in lake huron than in erie?
Explanation: The summary indicates that the **"lake" column** contains categorical values such as Lake Huron and Lake Erie.  
To answer the question, we count the number of rows where "lake" = Lake Huron and subtract the number of rows where "lake" = Lake Erie.  
No placeholder or aggregate rows (e.g., "Total", "—", "N/A") are present in the summary, so all rows are valid for counting.  
SQL: select ((select count("ship") from T where "lake" like '%lake huron%') - (select count("ship") from T where "lake" like '%lake erie%'))

SQLite table properties:

Table: List of The Citadel Bulldogs football seasons (row_number, season, conference, head coach, total wins, total losses, total ties, conference wins, conference losses, conference ties, conference standing, postseason result)

Q: how many total wins did the citadel bulldogs have before 1908?
Explanation: The summary specifies that the "season" column is Numeric and can be used to filter rows. 
The "total wins" column is also Numeric, but outlier rows (e.g., “Totals: 105 Seasons …”) must be excluded because they represent cumulative statistics rather than a single season. 
To compute the answer, we filter rows where season ≤ 1907 and sum the values in "total wins". 
Since numeric columns may contain non-numeric placeholders in other contexts, we explicitly cast "total wins" as INTEGER.
SQL:
select sum(cast("total wins" as integer)) as total_wins from T where cast("season" as integer) <= 1907;


SQLite table properties:

Table: List of Olympic medal counts (row_number, country, gold, silver, bronze, total)

Q: what is the total number of medals awarded in the olympics?
Explanation: The summary explicitly marks the row "World Total" as an aggregate row.
Since the question asks for the total number of medals across all countries, we should use the aggregate row directly instead of summing all individual country rows (to avoid double counting).
We select the value from the "total" column where "country" = 'World Total'".
SQL:
select cast("total" as integer) as global_total_medals from T where "country" = 'World Total';
"""

# ---------------------------------------------------------------

def truncate_tokens(prompt,  max_length) -> str:
    """Truncates a text string based on max number of tokens."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    encoded_string = encoding.encode(prompt)
    num_tokens = len(encoded_string)

    if num_tokens > max_length:
        prompt = encoding.decode(encoded_string[:max_length])
        print('truncated -->  ', num_tokens)
    return prompt

def get_completion(prompt, model="gpt-3.5-turbo", temperature=0.2, n=1):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        n=n,
        stream=False,
        max_tokens=4096,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["Table:", "\n\n\n"]
    )
    return response.choices[0].message.content

# -------------------------------------------------------------------------
def gen_table_decom_prompt(title, tab_col, question, full_table, summary=None):
    prompt = "" + p_sql_wtq_with_summary

    prompt += "\nSQLite table properties:\n\n"
    prompt += "Table: " + title + " (" + str(tab_col) + ")" + "\n\n"

    # ✅ Full Table Preview 추가
    prompt += "Full Table Preview:\n"
    prompt += truncate_tokens(full_table, max_length=15000) + "\n\n"

    if summary:  # ✅ 요약문이 있으면 반드시 포함
        prompt += "Structured table summary:\n" + summary + "\n\n"

    prompt += "Q: " + question + "\n"
    prompt += "Explanation:"
    prompt += "SQL:"
    return prompt

def generate_sql_answer_prompt(title, sql, result_table, question):
    prompt = p_sql_answer_wtq
    prompt += "\nTable_title: " + title
    prompt += "\nSQL: " + sql
    # ✅ SQL 실행 결과 테이블 제공
    prompt += "\n\nSQL Execution Result:\n" + result_table + "\n"
    prompt += "\nQuestion: " + question
    prompt += "\nA: To find the answer to this question, let’s think step by step."
    return prompt

def get_sql_3(prompt):
    response = None
    while response is None:
        try:
            response = get_completion(prompt, temperature=0)
        except:
            time.sleep(2)
            pass
    return response

def gen_full_table_prompt(title, tab_col, table, question):
    table = truncate_tokens(table, max_length=15000)

    prompt = p_wtq_full
    prompt += "Table: " + title + " (" + str(tab_col) + ")" + "\n\n"
    prompt += table + "\nQuestion: " + question
    prompt += "\nA: To find the answer to this question, let’s think step by step."

    return prompt

def get_answer(promt):
    response = None
    while response is None:
        try:

            response = get_completion(promt, temperature=0.7)
            # print('Generated ans------>: ', response)
        except:
            # print('sleep')
            time.sleep(2)
            pass

    return response

# --------------------------------------------------------------------------