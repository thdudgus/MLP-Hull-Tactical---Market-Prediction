import os
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from DiffusionModel import (
    DenoisingDataset,
    TrainingConfig as DMTrainingConfig,
    UNet1D,
    gaussian_smooth,
    set_seed,
    train_model,
)
from EvaluationMetric import MAX_INVESTMENT, MIN_INVESTMENT
from preprocessing import PreProcessing

# Training file path
TRAIN_FILE_PATH = "new_train.csv"

# Position sizing hyperparams
BASE_POSITION = 1.0
BASE_K = 5.0  # scaling before dividing by volatility
MIN_VOL = 1e-6

# Optional adjustments using lagged signals available at inference time
LAG_WEIGHT = 0.25  # weight on lagged_forward_returns added to drift
SIGMA_FLOOR = 1e-4
SIGMA_CEIL = 0.5
MODEL_PATH = "diffusion_denoiser.pt"
DENOISER_WEIGHT = 2.0  # weight for denoised trend signal in sizing


def load_preprocessed_train(
    train_path: str = TRAIN_FILE_PATH,
    batch_size: int = 1024,
    shuffle: bool = False,
    dropna_on_lags: bool = False,
) -> PreProcessing:
    """
    Convenience loader so TrainDiffusion uses the same preprocessing pipeline as the rest of the project.
    """
    return PreProcessing(
        path=train_path,
        batch_size=batch_size,
        shuffle=shuffle,
        dropna_on_lags=dropna_on_lags,
    )


@dataclass
class DiffusionRunConfig:
    train_path: str = TRAIN_FILE_PATH
    seq_len: int = 64
    smoothing_kernel: int = 21
    smoothing_sigma: float = 3.0
    batch_size: int = 1024
    epochs: int = 128
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: Optional[str] = MODEL_PATH
    seed: int = 42


def _build_denoising_loader(cfg: DiffusionRunConfig) -> DataLoader:
    """
    Create a denoising DataLoader from real forward_returns by smoothing them as target.
    """
    df = pd.read_csv(cfg.train_path)
    if "forward_returns" not in df.columns:
        raise ValueError("forward_returns column missing in training data.")

    df = df.sort_values("date_id").reset_index(drop=True)
    returns = df["forward_returns"].to_numpy(dtype=np.float32)
    valid = ~np.isnan(returns)
    if not valid.any():
        raise ValueError("No valid forward_returns for diffusion training.")
    fill_value = float(np.nanmean(returns[valid]))
    returns = np.nan_to_num(returns, nan=fill_value)

    target = gaussian_smooth(returns, kernel_size=cfg.smoothing_kernel, sigma=cfg.smoothing_sigma)
    dataset = DenoisingDataset(noisy=returns, target=target, seq_len=cfg.seq_len)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    return loader


def train_diffusion(cfg: DiffusionRunConfig) -> UNet1D:
    """
    Train UNet1D denoiser on forward_returns using DiffusionModel utilities.
    """
    set_seed(cfg.seed)
    loader = _build_denoising_loader(cfg)

    model = UNet1D()
    dm_cfg = DMTrainingConfig(
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        lr=cfg.lr,
        device=cfg.device,
    )
    print(f"Training diffusion denoiser on {len(loader.dataset)} windows (device={cfg.device})")
    train_model(model, loader, dm_cfg)

    if cfg.save_path:
        config_to_save = asdict(cfg)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config_to_save,
            },
            cfg.save_path,
        )
        print(f"Saved diffusion checkpoint to {cfg.save_path}")

    return model


def _load_denoiser(path: str = MODEL_PATH) -> Optional[tuple[UNet1D, int]]:
    """Load trained diffusion denoiser checkpoint if available."""
    if not os.path.exists(path):
        return None

    # Legacy checkpoints were saved when DiffusionRunConfig lived in __main__,
    # so expose it there to keep unpickling happy.
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "DiffusionRunConfig"):
        setattr(main_mod, "DiffusionRunConfig", DiffusionRunConfig)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = None
    last_err = None
    load_kwargs = [
        {"weights_only": True},
        {},
        {"weights_only": False},
    ]
    for kwargs in load_kwargs:
        try:
            checkpoint = torch.load(path, map_location=device, **kwargs)
            break
        except TypeError as e:
            # weights_only not supported on older torch
            last_err = e
            continue
        except Exception as e:
            last_err = e
            continue
    if checkpoint is None:
        print(f"Warning: failed to load denoiser from {path} ({last_err}); skipping denoiser.")
        return None
    model = UNet1D()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    cfg = checkpoint.get("config", None)
    seq_len = 64
    if cfg is not None:
        if isinstance(cfg, dict):
            seq_len = cfg.get("seq_len", seq_len)
        else:
            seq_len = getattr(cfg, "seq_len", seq_len)

    return model, seq_len


def _denoise_trend(model: UNet1D, series: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Apply denoiser over sliding windows; return per-timestep denoised last values.
    """
    if len(series) < seq_len:
        return np.zeros_like(series, dtype=np.float32)

    windows = []
    for i in range(seq_len - 1, len(series)):
        windows.append(series[i - seq_len + 1 : i + 1])

    device = next(model.parameters()).device
    x = torch.from_numpy(np.stack(windows).astype(np.float32)).unsqueeze(1).to(device)
    with torch.no_grad():
        out = model(x).squeeze(1).cpu().numpy()  # (N, seq_len)

    trend = np.zeros_like(series, dtype=np.float32)
    for idx, i in enumerate(range(seq_len - 1, len(series))):
        trend[i] = out[idx, -1]
    # propagate first available trend backwards
    trend[: seq_len - 1] = trend[seq_len - 1]
    return trend


def calibrate_sde(train_path: str = TRAIN_FILE_PATH):
    """
    Estimate drift (mu) and volatility (sigma) from forward_returns.
    Uses the shared PreProcessing pipeline for consistency.
    """
    loader = load_preprocessed_train(train_path=train_path, batch_size=512, shuffle=False, dropna_on_lags=False)
    df = loader.df
    if "forward_returns" not in df.columns:
        raise ValueError("forward_returns column missing in training data.")

    returns = df["forward_returns"].to_numpy(dtype=np.float32)
    returns = returns[~np.isnan(returns)]
    if len(returns) == 0:
        raise ValueError("No valid forward_returns for calibration.")

    sigma = float(np.std(returns, ddof=1))
    mu_hat = float(np.mean(returns))
    return mu_hat, sigma


def _ensure_lagged_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged columns if they are missing but source columns exist.
    Keeps the original order to preserve alignment with outputs.
    """
    need_copy = any(
        [
            ("lagged_forward_returns" not in df.columns and "forward_returns" in df.columns),
            ("lagged_market_forward_excess_returns" not in df.columns and "market_forward_excess_returns" in df.columns),
            ("lagged_risk_free_rate" not in df.columns and "risk_free_rate" in df.columns),
        ]
    )
    if not need_copy:
        return df

    df = df.copy()
    if "forward_returns" in df.columns and "lagged_forward_returns" not in df.columns:
        df["lagged_forward_returns"] = df["forward_returns"].shift(1)
    if "market_forward_excess_returns" in df.columns and "lagged_market_forward_excess_returns" not in df.columns:
        df["lagged_market_forward_excess_returns"] = df["market_forward_excess_returns"].shift(1)
    if "risk_free_rate" in df.columns and "lagged_risk_free_rate" not in df.columns:
        df["lagged_risk_free_rate"] = df["risk_free_rate"].shift(1)
    return df


@lru_cache(maxsize=1)
def _load_or_default(train_path: str = TRAIN_FILE_PATH):
    if os.path.exists(train_path):
        try:
            mu, sigma = calibrate_sde(train_path)
        except Exception as e:
            print(f"Calibration failed ({e}); using neutral defaults.")
            mu, sigma = 0.0, 0.05
    else:
        print("Training file not found; using neutral defaults.")
        mu, sigma = 0.0, 0.05
    return mu, sigma


def predict(df, train_path: str = TRAIN_FILE_PATH, denoiser_path: str = MODEL_PATH):
    """
    Vectorized inference for Kaggle runtime.
    Accepts pandas or polars DataFrame and returns positions clipped to [MIN_INVESTMENT, MAX_INVESTMENT].
    """
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()
    except ImportError:
        pass

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas or polars DataFrame.")

    df = _ensure_lagged_columns(df)

    mu_hat, sigma_hat = _load_or_default(train_path)
    denoiser = _load_denoiser(denoiser_path)
    denoised_trend = None
    seq_len = None
    if denoiser is not None and "lagged_forward_returns" in df.columns:
        model, seq_len = denoiser
        series = df["lagged_forward_returns"].to_numpy(dtype=np.float32)
        series = np.nan_to_num(series, nan=0.0)
        denoised_trend = _denoise_trend(model, series, seq_len)
        # Center to remove global bias; only relative signal is added.
        denoised_trend = denoised_trend - float(np.mean(denoised_trend))

    rf = df["risk_free_rate"].to_numpy(dtype=np.float32) if "risk_free_rate" in df.columns else np.zeros(len(df), dtype=np.float32)
    lag_fwd = df["lagged_forward_returns"].to_numpy(dtype=np.float32) if "lagged_forward_returns" in df.columns else np.zeros(len(df), dtype=np.float32)
    lag_mkt = df["lagged_market_forward_excess_returns"].to_numpy(dtype=np.float32) if "lagged_market_forward_excess_returns" in df.columns else np.zeros(len(df), dtype=np.float32)

    # Drift adjustment: blend global mean with recent lagged returns and denoised trend if available
    local_mu = mu_hat + LAG_WEIGHT * lag_fwd
    if denoised_trend is not None:
        local_mu += DENOISER_WEIGHT * denoised_trend

    # Volatility proxy: combine global sigma with absolute lagged returns/market excess
    lag_vol_proxy = np.maximum(np.abs(lag_fwd), np.abs(lag_mkt))
    lag_vol_proxy = np.clip(lag_vol_proxy, SIGMA_FLOOR, SIGMA_CEIL)
    sigma_eff = np.maximum(sigma_hat, lag_vol_proxy)

    scale = BASE_K / np.maximum(sigma_eff, MIN_VOL)
    expected_excess = local_mu - rf
    positions = BASE_POSITION + scale * expected_excess
    positions = np.nan_to_num(positions, nan=BASE_POSITION)
    positions = np.clip(positions, MIN_INVESTMENT, MAX_INVESTMENT)
    return positions


if __name__ == "__main__":
    cfg = DiffusionRunConfig()
    train_diffusion(cfg)
    mu, sigma = _load_or_default(cfg.train_path)
    print(f"Calibrated drift={mu:.6f}, volatility={sigma:.6f}")
