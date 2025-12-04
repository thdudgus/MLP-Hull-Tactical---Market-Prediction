import yfinance as yf
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. 데이터 다운로드
# ---------------------------------------------------------
tickers = {
    '^GSPC': 'Close',       # P8
    '^IRX': 'Rate_Short',   # I1
    '^TNX': 'Rate_Long',    # I2
    'KRW=X': 'ExRate',      # E1
    '^VIX': 'VIX'           # S1
}

print("데이터 다운로드 및 정제 중...")
df_raw = yf.download(list(tickers.keys()), start="1900-01-01", progress=False)

# 데이터 정리
df = pd.DataFrame()
for ticker, name in tickers.items():
    if name == 'Close':
        df['Close'] = df_raw['Close'][ticker]
        df['Volume'] = df_raw['Volume'][ticker]
    else:
        df[name] = df_raw['Close'][ticker]

df = df.fillna(method='ffill').dropna()

# ---------------------------------------------------------
# 2. Target & Feature 계산
# ---------------------------------------------------------
# Target
df['forward_returns'] = df['Close'].shift(-1) / df['Close'] - 1
df['risk_free_rate'] = df['Rate_Short'] / 100 / 252

rolling_mean = df['forward_returns'].rolling(window=1260).mean()
raw_excess = df['forward_returns'] - rolling_mean

def perform_winsorization(series, n_mad=4):
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0: return series
    upper = median + n_mad * mad
    lower = median - n_mad * mad
    return series.clip(lower=lower, upper=upper)

df['market_forward_excess_returns'] = perform_winsorization(raw_excess)

# Feature Template (0으로 초기화)
required_columns = [
    'date',
    'D1','D2','D3','D4','D5','D6','D7','D8','D9',
    'E1','E10','E11','E12','E13','E14','E15','E16','E17','E18','E19',
    'E2','E20','E3','E4','E5','E6','E7','E8','E9',
    'I1','I2','I3','I4','I5','I6','I7','I8','I9',
    'M1','M10','M11','M12','M13','M14','M15','M16','M17','M18',
    'M2','M3','M4','M5','M6','M7','M8','M9',
    'P1','P10','P11','P12','P13','P2','P3','P4','P5','P6','P7','P8','P9',
    'S1','S10','S11','S12','S2','S3','S4','S5','S6','S7','S8','S9',
    'V1','V10','V11','V12','V13','V2','V3','V4','V5','V6','V7','V8','V9',
    'forward_returns','risk_free_rate','market_forward_excess_returns'
]

final_df = pd.DataFrame(0.0, index=df.index, columns=required_columns)

# 값 매핑
final_df['forward_returns'] = df['forward_returns']
final_df['risk_free_rate'] = df['risk_free_rate']
final_df['market_forward_excess_returns'] = df['market_forward_excess_returns']

final_df['I1'] = df['Rate_Short']
final_df['I2'] = df['Rate_Long']
final_df['I3'] = df['Rate_Long'] - df['Rate_Short'] # U1 (Yield Spread)

final_df['P8'] = df['Close']
final_df['P1'] = df['Close'] / df['Close'].rolling(5).mean()

final_df['M11'] = df['Volume']
final_df['M5'] = df['Close'] - df['Close'].shift(5) # MOM_5
avg_rate = (df['Rate_Long'] + df['Rate_Short']) / 2
final_df['M12'] = df['Volume'] / (avg_rate + 1e-9)  # U2

final_df['V1'] = df['Close'].rolling(20).std()      # VOL_20
final_df['S1'] = df['VIX']
final_df['E1'] = df['ExRate']

# ---------------------------------------------------------
# 3. 데이터 정리 및 date_id 생성 (핵심 변경 사항)
# ---------------------------------------------------------
final_df = final_df.dropna()

final_df['date'] = final_df.index.strftime('%Y-%m-%d')


# D1~D9도 날짜 관련이므로 예시로 정수화
final_df['D1'] = final_df.index.dayofweek

# 결과 확인
print("=== 상위 5개 데이터 (date_id 확인) ===")
print(final_df[['date', 'P8', 'forward_returns']].head())

print("\n=== 하위 5개 데이터 ===")
print(final_df[['date', 'P8', 'forward_returns']].tail())

# 타입 확인
print("\n=== date_id 데이터 타입 ===")
print(final_df['date'].dtype)

# 저장
final_df.to_csv("formatted_train.csv", index=False)