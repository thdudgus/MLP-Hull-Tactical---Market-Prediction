import pandas as pd
from sklearn.ensemble import RandomForestRegressor

train_df = pd.read_csv("train.csv")
EXCLUDE_COLS = ["forward_returns", "risk_free_rate", "market_forward_excess_returns"]
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

