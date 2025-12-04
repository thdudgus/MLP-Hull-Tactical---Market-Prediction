import pandas as pd
import yfinance as yf
import numpy as np

# 1. train.csv 로드
df_train = pd.read_csv('train.csv')
print("Train data loaded.")

# 2. S&P 500 실제 데이터 다운로드 (1990년부터 넉넉하게)
# ^GSPC: S&P 500 지수 티커
print("Downloading S&P 500 data...")
sp500 = yf.download('^GSPC', start='1990-01-01', end='2024-12-31', progress=False)

# 'Close' 컬럼이 MultiIndex인 경우 처리
if isinstance(sp500.columns, pd.MultiIndex):
    sp500_close = sp500['Close']['^GSPC']
else:
    sp500_close = sp500['Close']

# 수익률 계산 (train의 forward_returns와 비교하기 위함)
# forward_returns는 보통 (다음날 종가 - 오늘 종가) / 오늘 종가
sp500_returns = sp500_close.pct_change().shift(-1)

# 3. 매칭 알고리즘: 상관관계가 가장 높은 시작일 찾기
# train 데이터의 앞부분 500일 패턴을 사용하여 스캔
scan_window = 500
train_pattern = df_train['forward_returns'].iloc[:scan_window].fillna(0).values

max_corr = -1
best_start_date = None

# S&P 500 데이터 전체를 훑으면서 가장 패턴이 비슷한 구간 찾기
# (속도를 위해 최근 35년 데이터만 스캔)
for i in range(len(sp500_returns) - scan_window):
    market_pattern = sp500_returns.iloc[i : i+scan_window].fillna(0).values
    
    # 상관계수 계산
    corr = np.corrcoef(train_pattern, market_pattern)[0, 1]
    
    if corr > max_corr:
        max_corr = corr
        best_start_date = sp500_returns.index[i]
        best_idx = i

print(f"✅ 찾은 시작 날짜: {best_start_date.date()}")
print(f"✅ 상관계수: {max_corr:.4f} (1.0에 가까울수록 완벽 일치)")

# 4. 날짜 매핑 및 외부 데이터 병합 예시
if best_start_date:
    # S&P 500의 해당 날짜부터 데이터를 잘라옴
    matched_dates = sp500.index[best_idx : best_idx + len(df_train)]
    
    # 길이가 다를 수 있으므로(휴일 처리 차이 등) 최소 길이로 맞춤
    min_len = min(len(matched_dates), len(df_train))
    df_train = df_train.iloc[:min_len].copy()
    df_train['date'] = matched_dates[:min_len]
    
    # 결과 확인
    print("\n[매핑 결과]")
    print(df_train[['date_id', 'date', 'forward_returns']].head())
    
    # (팁) 이제 이 'date'를 기준으로 VIX, 유가 등을 merge 하면 됩니다!
    df_train.to_csv("train_with_date.csv", index=False)