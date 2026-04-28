import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ── Model definition ─────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        se = x.mean(dim=-1)
        se = self.fc(se).unsqueeze(-1)
        return x * se


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True),
            nn.BatchNorm1d(out_channels),
        )
        self.se = SEBlock(out_channels, reduction=8)
        self.proj = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        return out + self.proj(x)


class AttentionPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(256, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.attn(x)


class ECGAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.ModuleList(
            [
                ResidualConvBlock(12, 32, 11),
                ResidualConvBlock(32, 64, 7),
                ResidualConvBlock(64, 128, 5),
                ResidualConvBlock(128, 256, 3),
                ResidualConvBlock(256, 256, 3),
            ]
        )
        self.pool = AttentionPool()
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 5),
        )

    def forward(self, x):
        for block in self.features:
            x = block(x)
        attention = self.pool(x)
        attention = torch.softmax(attention, dim=-1)
        x = (x * attention).sum(dim=-1)
        x = self.classifier(x)
        return x


# ── Load model checkpoint ───────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ecg_attention_calibrated.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

model = ECGAttentionModel().to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# Mean & std (IMPORTANT for normalization)
mean = np.array(checkpoint.get("mean", checkpoint.get("train_mean")), dtype=np.float32)
std = np.array(checkpoint.get("std", checkpoint.get("train_std")), dtype=np.float32)

# Class mapping → Risk levels
CLASS_TO_LEVEL = {
    0: "Low",     # NORM
    1: "High",    # MI
    2: "Medium",  # STTC
    3: "Medium",  # CD
    4: "High"     # HYP
}


# ── Preprocess ECG signal ───────────────────────────
def preprocess(ecg_signal):
    ecg = np.array(ecg_signal, dtype=np.float32)

    if ecg.ndim == 1:
        raise ValueError("ECG input must be 2D or 3D with 12 channels, not a 1D vector.")

    if ecg.ndim == 2:
        if ecg.shape[0] != 12:
            raise ValueError("ECG input must have 12 channels on axis 0 (shape [12, length]).")
        ecg = np.expand_dims(ecg, axis=0)

    if ecg.ndim != 3 or ecg.shape[1] != 12:
        raise ValueError("ECG input must have shape [12, length] or [batch, 12, length].")

    ecg = (ecg - mean) / (std + 1e-8)
    ecg_tensor = torch.tensor(ecg, dtype=torch.float32).to(device)
    return ecg_tensor


# ── FINAL PREDICT FUNCTION ──────────────────────────
def predict(ecg_signal):
    """
    ecg_signal: list or numpy array with shape [12, length]
    """

    ecg_tensor = preprocess(ecg_signal)

    with torch.no_grad():
        outputs = model(ecg_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    label = int(np.argmax(probs))

    return {
        "level": CLASS_TO_LEVEL[label],
        "score": float(probs[label])
    }


# ── TEST ───────────────────────────────────────────
if __name__ == "__main__":
    # Dummy ECG signal (replace later with real input)
    sample_signal = np.random.randn(12, 1000)

    result = predict(sample_signal)
    print(result)