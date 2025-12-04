import pandas as pd

# 1. 파일 불러오기 (실제 파일 경로에 맞게 수정하세요)
# 예시: train.csv와 target.csv가 있다고 가정
df_train = pd.read_csv("train_with_date.csv")
df_target = pd.read_csv("eth_train.csv")

# 공백 제거 (혹시 모를 컬럼명 공백 에러 방지)
df_train.columns = df_train.columns.str.strip()
df_target.columns = df_target.columns.str.strip()

# -------------------------------------------------------
# 2. 날짜 매핑 및 병합 (Inner Join)
# -------------------------------------------------------
# df_train에서 'date'와 'date_id' 컬럼만 추출하여 매핑 테이블로 사용합니다.
# how='inner' 옵션을 사용하면:
#   1. 두 데이터프레임 모두에 'date'가 존재하는 행만 남깁니다. (자동으로 없는 날짜 삭제)
#   2. df_target에 'date_id' 정보가 옆에 붙습니다.
df_merged = pd.merge(df_target, df_train[['date', 'date_id']], on='date', how='inner')

# -------------------------------------------------------
# 3. 컬럼 순서 정리 (date_id를 맨 앞으로 보내기)
# -------------------------------------------------------
# train.csv 처럼 date_id를 가장 첫 번째 컬럼으로 이동시킵니다.
cols = ['date_id'] + [c for c in df_merged.columns if c != 'date_id' and c != 'date']
df_merged = df_merged[cols]

# -------------------------------------------------------
# 4. 결과 확인 및 저장
# -------------------------------------------------------
print(f"원본 Target 행 개수: {len(df_target)}")
print(f"매칭 후 Target 행 개수: {len(df_merged)} (삭제된 행: {len(df_target) - len(df_merged)})")

print("\n=== 결과 데이터 미리보기 ===")
print(df_merged[['date_id', 'forward_returns']].head())

# 파일 저장
df_merged.to_csv("target_with_date_id.csv", index=False)