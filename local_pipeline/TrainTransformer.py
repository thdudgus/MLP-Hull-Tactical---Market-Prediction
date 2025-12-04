# TrainTransformer.py
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from EvaluationMetric import MAX_INVESTMENT, MIN_INVESTMENT
from preprocessing import read_train_with_lag

MODEL_PATH = "transformer_model.pt"
TARGET_COL = "forward_returns"
EXCLUDE_COLS = ["date_id", TARGET_COL, "risk_free_rate", "market_forward_excess_returns"]
BASE_POSITION = 1.0  # neutral weight
LEVERAGE_SCALE = 15.0  # how aggressively we map return forecasts to positions

class SimpleTransformerRegressor(nn.Module):
    def __init__(self, input_dim: int,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        # seq_len=1이지만 형태를 맞추기 위해 pos embedding 하나 둠
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):  # x: (batch, 1, input_dim)
        h = self.input_proj(x) + self.pos_embedding  # (B,1,d_model)
        h = self.encoder(h)                          # (B,1,d_model)
        out = self.fc_out(h[:, -1, :])               # (B,1) -> (B,)
        return out.squeeze(-1)


def _load_checkpoint(device: torch.device, model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}. Run TrainTransformer.py to train and save the model."
        )
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    feature_cols = checkpoint["feature_cols"]
    scaler_mean = checkpoint["scaler_mean"]
    scaler_scale = checkpoint["scaler_scale"]
    model_kwargs = checkpoint.get("model_kwargs", {"input_dim": len(feature_cols)})

    model = SimpleTransformerRegressor(**model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, feature_cols, scaler_mean, scaler_scale


def _prepare_features(df: pd.DataFrame, feature_cols, scaler_mean, scaler_scale):
    """Select, impute, and scale features in a consistent order."""
    X = df[feature_cols].to_numpy(dtype=np.float32)

    # Fill any NaNs with training means to avoid propagating NaNs through the model.
    for i in range(X.shape[1]):
        col = X[:, i]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            col[nan_mask] = scaler_mean[i]
            X[:, i] = col

    X_scaled = (X - scaler_mean) / scaler_scale
    return X_scaled


@lru_cache(maxsize=1)
def _get_inference_artifacts(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, feature_cols, scaler_mean, scaler_scale = _load_checkpoint(device, model_path=model_path)
    return device, model, feature_cols, scaler_mean, scaler_scale


def predict(df, model_path: str = MODEL_PATH):
    """
    Inference helper used by predict.py and the Kaggle runtime.
    Accepts a pandas or polars DataFrame with feature columns.
    Returns a numpy array of positions clipped to [MIN_INVESTMENT, MAX_INVESTMENT].
    """
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()
    except ImportError:
        pass

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas or polars DataFrame.")

    device, model, feature_cols, scaler_mean, scaler_scale = _get_inference_artifacts(model_path)
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    X_scaled = _prepare_features(df, feature_cols, scaler_mean, scaler_scale)

    # Optional risk-free adjustment before sizing
    rf = df["risk_free_rate"].to_numpy(dtype=np.float32) if "risk_free_rate" in df.columns else np.zeros(len(df), dtype=np.float32)

    X_tensor = torch.from_numpy(X_scaled).unsqueeze(1).to(device)  # (N,1,F)

    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()

    # Map predicted forward returns to positions; center at BASE_POSITION
    preds = preds.reshape(-1, 1)
    rf = rf.reshape(-1, 1)
    sized = BASE_POSITION + LEVERAGE_SCALE * (preds - rf)
    sized = sized.reshape(-1)

    sized = np.nan_to_num(sized, nan=BASE_POSITION, posinf=MAX_INVESTMENT, neginf=MIN_INVESTMENT)
    sized = np.clip(sized, MIN_INVESTMENT, MAX_INVESTMENT)
    return sized


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = read_train_with_lag("train.csv")
    
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    print("Num features:", len(feature_cols))

    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Transformer는 (batch, seq_len, feature_dim) 형태를 기대하므로 seq_len=1로 reshape
    X_tensor = torch.from_numpy(X_scaled).unsqueeze(1)  # (N,1,F)
    y_tensor = torch.from_numpy(y)                      # (N,)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = SimpleTransformerRegressor(input_dim=X.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10  # 일단 짧게 돌려보고 늘리면 됨
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{epochs} - train MSE: {avg_loss:.6f}")

    # 모델 + feature_cols + scaler 파라미터 저장
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "model_kwargs": {
            "input_dim": X.shape[1],
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.1,
        },
    }

    torch.save(checkpoint, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
