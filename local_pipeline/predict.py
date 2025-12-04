import polars as pl 
import pandas as pd
import numpy as np
from EvaluationMetric import score
from TrainDiffusion import predict

def predict(test: pl.DataFrame):
    """Replace this function with your inference code.
    You can return either a Pandas or Polars dataframe, though Polars is recommended for performance.
    Each batch of predictions (except the very first) must be returned within 5 minutes of the batch features being provided.
    """
    row_pd = test.to_pandas()

    X_row = row_pd[feature_cols]

    pred_forward = rf.predict(X_row)
    
    # 2. 벡터 연산으로 계산 (모든 행에 대해 동시에 적용)
    # 1 + pred * 50 계산 후 0.0과 2.0 사이로 자름(clip)
    pos = np.clip(1 + pred_forward * 50, 0.0, 2.0)
    
    # 3. 배열(Array) 자체를 반환
    return pos

def main():
    test = pl.read_csv("new_test.csv")

    pred_values = predict(test)

    submission = test.select(["date_id"]).with_columns(
        pl.Series(name="prediction", values=pred_values)
    )
    submission.write_csv("submission.csv")

    solution = pd.read_csv("solution.csv")

    test_date_ids = test["date_id"].to_list()
    solution_filtered = solution[solution["date_id"].isin(test_date_ids)].copy()

    solution_filtered = solution_filtered.sort_values("date_id").reset_index(drop=True)
    submission_pd = submission.to_pandas().sort_values("date_id").reset_index(drop=True)

    print(score(solution_filtered, submission_pd))

if __name__ == "__main__":
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

    train_df = pd.read_csv("new_train.csv")
    EXCLUDE_COLS = ["date", "date_id", "forward_returns", "risk_free_rate", "market_forward_excess_returns"]
    feature_cols = [c for c in train_df.columns if c not in EXCLUDE_COLS]

    X_train = train_df[feature_cols]
    y_train = train_df["forward_returns"]

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    main()
