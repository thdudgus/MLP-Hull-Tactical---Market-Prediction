import math
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_synthetic_returns(
    n_steps: int = 2000,
    trend_scale: float = 5e-4,
    noise_scale: float = 8e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a noisy market-like return series with a hidden trend."""
    t = np.linspace(0, 1, n_steps, dtype=np.float32)
    slow_cycle = np.sin(2 * math.pi * t * 2)  # slow oscillation
    fast_cycle = 0.35 * np.sin(2 * math.pi * t * 15)  # faster bumps
    drift = 0.1 * np.cumsum(np.random.randn(n_steps)).astype(np.float32) / n_steps
    hidden_trend = trend_scale * (slow_cycle + fast_cycle) + trend_scale * drift
    noise = np.random.normal(loc=0.0, scale=noise_scale, size=n_steps).astype(np.float32)
    noisy_returns = hidden_trend + noise
    return hidden_trend, noisy_returns


def gaussian_smooth(series: np.ndarray, kernel_size: int = 15, sigma: float = 2.5) -> np.ndarray:
    """Apply a centered Gaussian filter using numpy (avoids extra deps)."""
    half = (kernel_size - 1) / 2
    kernel_positions = np.arange(kernel_size) - half
    kernel = np.exp(-0.5 * (kernel_positions / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(series, kernel, mode="same")


class DenoisingDataset(Dataset):
    """Pairs noisy windows with their smoothed targets."""

    def __init__(self, noisy: np.ndarray, target: np.ndarray, seq_len: int):
        assert len(noisy) == len(target), "Noisy and target series must align"
        self.noisy = noisy.astype(np.float32)
        self.target = target.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.noisy) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.noisy[idx : idx + self.seq_len]
        y = self.target[idx : idx + self.seq_len]
        x_tensor = torch.from_numpy(x).unsqueeze(0)  # (1, seq_len)
        y_tensor = torch.from_numpy(y).unsqueeze(0)
        return x_tensor, y_tensor

# --- NEW: Positional Encoding ---
class PositionalEncoding(nn.Module):
    """Adds sinusoidal positional encoding to the input."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # (Max_Len, 1, D_Model) -> 브로드캐스팅을 위해 배치 차원 추가
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Seq_Len, Batch, D_Model)
        return x + self.pe[:x.size(0)]

# --- NEW: Transformer Block replacing ConvBlock ---
class TransformerBlock(nn.Module):
    """
    Wraps nn.TransformerEncoderLayer to work with (Batch, Channel, Length) data.
    """
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        # d_model이 nhead로 나누어 떨어져야 함을 보장하기 위해 체크하거나 패딩할 수 있으나,
        # 여기서는 U-Net 채널 설계 시 64의 배수이므로 보통 안전함.
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout,
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # 입출력 차원 맞추기용 (Skip connection 등)
        self.norm = nn.GroupNorm(min(8, d_model), d_model) # 안정적인 학습을 위한 정규화

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (Batch, Channel, Seq_Len)
        B, C, L = x.shape
        
        # 1. Permute for Transformer: (Seq_Len, Batch, Channel)
        x_t = x.permute(2, 0, 1)
        
        # 2. Add Positional Encoding
        x_t = self.pos_encoder(x_t)
        
        # 3. Apply Transformer
        out_t = self.transformer_encoder(x_t)
        
        # 4. Permute back: (Batch, Channel, Seq_Len)
        out = out_t.permute(1, 2, 0)
        
        # 5. Residual + Norm (Transformer 내부에도 있지만, 블록 전체의 안정성을 위해 추가)
        return self.norm(out + x)

class ConvBlock(nn.Module):
    """
    Two-layer conv block with residual skip; keeps GroupNorm for small batches.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        padding: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        groups = min(8, out_ch)  # keep GroupNorm stable for small channel counts
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.gn1 = nn.GroupNorm(groups, out_ch)
        self.act1 = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding)
        self.gn2 = nn.GroupNorm(groups, out_ch)
        self.act2 = nn.SiLU()
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.act2(x)
        return x + residual


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.block = ConvBlock(out_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Align lengths in case of odd-sized inputs
        if x.shape[-1] != skip.shape[-1]:
            diff = skip.shape[-1] - x.shape[-1]
            skip = skip[..., diff // 2 : diff // 2 + x.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class UNet1D(nn.Module):
    """
    U-Net structure with Transformer Blocks for feature extraction.
    """
    def __init__(self, base_ch: int = 64, dropout: float = 0.1):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        # Input Embedding & First Block
        self.inc = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=3, padding=1), # 채널 확장을 위한 초기 Conv
            nn.SiLU()
        )
        self.enc1 = TransformerBlock(c1, nhead=4, dropout=dropout)
        self.down1 = Down(c1, c2)
        
        self.enc2 = TransformerBlock(c2, nhead=8, dropout=dropout)
        self.down2 = Down(c2, c3)
        
        self.enc3 = TransformerBlock(c3, nhead=8, dropout=dropout)
        self.down3 = Down(c3, c4)
        
        # Bottleneck (Deepest features)
        self.enc4 = TransformerBlock(c4, nhead=8, dropout=dropout)

        # Decoder Path
        self.up1 = Up(c4, c3, c3, dropout=dropout)
        self.up2 = Up(c3, c2, c2, dropout=dropout)
        self.up3 = Up(c2, c1, c1, dropout=dropout)
        
        self.out_conv = nn.Conv1d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.inc(x)
        
        s1 = self.enc1(x0)
        d1 = self.down1(s1)

        s2 = self.enc2(d1)
        d2 = self.down2(s2)

        s3 = self.enc3(d2)
        d3 = self.down3(s3)

        bottleneck = self.enc4(d3)

        u1 = self.up1(bottleneck, s3)
        u2 = self.up2(u1, s2)
        u3 = self.up3(u2, s1)
        
        out = self.out_conv(u3)
        return out


@dataclass
class TrainingConfig:
    seq_len: int = 64
    batch_size: int = 1024
    epochs: int = 128
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(model: nn.Module, loader: DataLoader, cfg: TrainingConfig) -> None:
    model.to(cfg.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    model.train()
    for epoch in range(cfg.epochs):
        running_loss = 0.0
        for noisy, target in loader:
            noisy = noisy.to(cfg.device)
            target = target.to(cfg.device)

            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * noisy.size(0)

        avg_loss = running_loss / len(loader.dataset)
        print(f"Epoch {epoch + 1}/{cfg.epochs} - MSE: {avg_loss:.6f}")


def inference_signal(model: nn.Module, series: np.ndarray, seq_len: int, device: str) -> Tuple[np.ndarray, str]:
    model.eval()
    window = torch.tensor(series[-seq_len:], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        denoised = model(window).squeeze().cpu().numpy()
    signal = "Buy Signal" if denoised[-1] > 0 else "Sell Signal"
    return denoised, signal


def main() -> None:
    set_seed()
    cfg = TrainingConfig()

    hidden_trend, noisy_returns = generate_synthetic_returns()
    smoothed_target = gaussian_smooth(noisy_returns, kernel_size=21, sigma=3.0)

    dataset = DenoisingDataset(noisy_returns, smoothed_target, seq_len=cfg.seq_len)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    model = UNet1D()
    print(f"Training on {len(dataset)} windows of length {cfg.seq_len} (device={cfg.device})")
    train_model(model, loader, cfg)

    denoised, signal = inference_signal(model, noisy_returns, cfg.seq_len, cfg.device)
    print(f"Inference: last-point trend estimate = {denoised[-1]:.6f} -> {signal}")
    print("Sample of last denoised window (first 10 values):")
    print(denoised[:100])


if __name__ == "__main__":
    main()
