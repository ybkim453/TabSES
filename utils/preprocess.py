# Normalizer & Preprocess
import pandas as pd

## Adapted from Binder and Dater paper 

def dict2df(table):
    """딕셔너리를 DataFrame으로 변환 (utils.preprocess에서)"""
    header, rows = table[0], table[1:]

    seen = {}
    unique_header = []
    for i, col in enumerate(header):
        col_name = str(col).strip().strip('"').strip("'")
        if not col_name or col_name.lower() == "nan":
            col_name = f"Unnamed: {i}" # 빈 header일 때 Unnamed : i

        if col_name not in seen:
            seen[col_name] = 0
            unique_header.append(col_name)
        else:
            seen[col_name] += 1
            unique_header.append(f"{col_name}_{seen[col_name]}") # 중복된 헤더일때 이름_i

    df = pd.DataFrame(data=rows, columns=unique_header)
    
    for col in df.columns:
    # 모든 값에 대해 콤마 제거
        df[col] = df[col].astype(str).str.replace(",", "", regex=False)

        # 빈 문자열이나 'nan' 같은 건 결측 처리
        df[col] = df[col].replace({"": None, "nan": None, "NaN": None})

        # 숫자로 변환 가능한 값은 숫자로
        df[col] = pd.to_numeric(df[col], errors="ignore")

        if df[col].dtype == object and df[col].str.contains(r"\d{1,2}\s+\w+", na=False).any():
            df[col] = pd.to_datetime(df[col], errors="ignore", dayfirst=True)

    return df

def table_linearization(table: pd.DataFrame, style: str = 'pipe'):
    """테이블을 pipe 형태 문자열로 변환 (utils.preprocess에서)"""
    linear_table = ''
    if style == 'pipe':
        header = ' | '.join(table.columns) + '\n'
        linear_table += header
        rows = table.values.tolist()
        for row_idx, row in enumerate(rows):
            line = ' | '.join(str(v) for v in row)
            if row_idx != len(rows) - 1:
                line += '\n'
            linear_table += line
    return linear_table

def convert_df_type(df):
    """DataFrame 타입 변환 (utils.normalizer에서)"""
    # 원본은 복잡하니까 간단 버전만
    return df