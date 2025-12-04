import pandas as pd
import numpy as np

def create_train_test_split(input_file='nasdaq_train.csv'):
    print(f"📂 '{input_file}' 로딩 중...")
    df = pd.read_csv(input_file)
    
    # ---------------------------------------------------------
    # 1. Date 컬럼 처리 (String -> Datetime)
    # ---------------------------------------------------------
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        # 날짜순 정렬 (date_id 생성을 위해 필수)
        df = df.sort_values('date').reset_index(drop=True)
    else:
        print("❌ Error: 'date' 컬럼이 없습니다. (예: 2025-12-03)")
        return

    # ---------------------------------------------------------
    # 2. date_id 생성 (없으면 생성, 있으면 그대로 사용)
    # ---------------------------------------------------------
    if 'date_id' not in df.columns:
        print("⚠️ 'date_id' 컬럼이 없어 날짜 순서대로 새로 생성합니다 (0, 1, 2...).")
        # 날짜별로 고유한 ID 부여 (같은 날짜면 같은 ID)
        df['date_id'] = df['date'].factorize()[0]
    
    # date_id는 반드시 정수형이어야 함
    df['date_id'] = df['date_id'].astype(int)

    # ---------------------------------------------------------
    # 3. Lagged Feature 생성 (어제 데이터)
    # ---------------------------------------------------------
    print("⚙️ Lagged Features (어제 데이터) 생성 중...")
    df['lagged_forward_returns'] = df['forward_returns'].shift(1)
    df['lagged_risk_free_rate'] = df['risk_free_rate'].shift(1)
    df['lagged_market_forward_excess_returns'] = df['market_forward_excess_returns'].shift(1)
    
    # 첫 행(이전 데이터 없음) 제거
    df = df.dropna().reset_index(drop=True)

    # ---------------------------------------------------------
    # 4. 2025년 기준 Train / Test 분리
    # ---------------------------------------------------------
    split_date = pd.Timestamp("2025-01-01")
    
    # Train: 2025년 미만
    train_split = df[df['date'] < split_date].copy()
    
    # Test: 2025년 이상
    test_split = df[df['date'] >= split_date].copy()
    
    print(f"✂️ 분리 완료: Train({len(train_split)} rows) / Test({len(test_split)} rows)")

    if len(test_split) == 0:
        print("❌ Error: 2025년 이후 데이터가 없습니다.")
        return

    # ---------------------------------------------------------
    # 5. 저장 (new_test에 date_id 필수 포함)
    # ---------------------------------------------------------
    
    # [Train 저장]
    # Train은 모든 컬럼 유지 (date만 제외, date_id는 유지)
    train_cols = [c for c in df.columns if c != 'date']
    train_split[train_cols].to_csv("new_train.csv", index=False)
    
    # [Test 저장]
    test_split['is_scored'] = True
    
    # 제거할 컬럼: 정답지(Target) + 날짜 문자열(date)
    # date_id는 Target이 아니므로 제거하면 안 됨!
    drop_cols = ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns', 'date']
    
    # 남길 컬럼 리스트
    test_cols = [c for c in test_split.columns if c not in drop_cols]
    
    # ★ 핵심: date_id를 맨 앞으로 이동
    if 'date_id' in test_cols:
        test_cols.insert(0, test_cols.pop(test_cols.index('date_id')))
    else:
        # 혹시라도 빠졌으면 강제 추가
        print("⚠️ date_id가 컬럼 리스트에서 누락되어 복구합니다.")
        test_cols.insert(0, 'date_id')

    # 최종 저장
    test_split[test_cols].to_csv("new_test.csv", index=False)
    
    print("💾 저장 완료!")
    print(f"   👉 new_train.csv: {len(train_split)} 행")
    print(f"   👉 new_test.csv : {len(test_split)} 행")
    print(f"      (Test 컬럼 확인: {test_cols[:3]} ... 포함)")

if __name__ == "__main__":
    create_train_test_split('btc_train.csv')