# models/waveform_encoder.py
import torch.nn as nn


class WaveformEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 48, 256), nn.ReLU(), nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)
