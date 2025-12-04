import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Sequence

class PreProcessing(DataLoader):
    """
    Minimal DataLoader wrapper for train.csv preprocessing.
    - Sorts by date_id
    - Builds lagged targets (forward_returns, market_forward_excess_returns, risk_free_rate)
    - Imputes remaining NaNs with column means
    - Exposes TensorDataset of (features, target)
    """

    def __init__(
        self,
        path: str,
        target_col: str = "forward_returns",
        exclude_cols: Optional[Sequence[str]] = None,
        batch_size: int = 256,
        shuffle: bool = True,
        num_workers: int = 0,
        dropna_on_lags: bool = True,
    ) -> None:
        self.raw_df = pd.read_csv(path)

        df = self.raw_df.sort_values("date_id").reset_index(drop=True)
        df["lagged_forward_returns"] = df[target_col].shift(1)
        if "market_forward_excess_returns" in df.columns:
            df["lagged_market_forward_excess_returns"] = df["market_forward_excess_returns"].shift(1)
        if "risk_free_rate" in df.columns:
            df["lagged_risk_free_rate"] = df["risk_free_rate"].shift(1)

        if dropna_on_lags:
            lag_cols = [c for c in ["lagged_forward_returns", "lagged_market_forward_excess_returns", "lagged_risk_free_rate"] if c in df.columns]
            df = df.dropna(subset=lag_cols).reset_index(drop=True)

        if exclude_cols is None:
            exclude_cols = ["date_id", target_col]
        self.feature_cols: List[str] = [c for c in df.columns if c not in exclude_cols]
        self.target_col = target_col
        self.df = df

        X = df[self.feature_cols].to_numpy(dtype=np.float32)
        y = df[target_col].to_numpy(dtype=np.float32)

        # Impute numeric NaNs with column means to avoid breaking downstream models.
        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        nan_rows, nan_cols = np.where(np.isnan(X))
        if nan_rows.size:
            X[nan_rows, nan_cols] = col_means[nan_cols]

        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
