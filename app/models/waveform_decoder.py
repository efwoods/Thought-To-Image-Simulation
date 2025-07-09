# models/waveform_decoder.py
import torch.nn as nn


class WaveformDecoder(nn.Module):
    def __init__(self, latent_dim=128, channels=1, length=768):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, length),
        )

    def forward(self, z):
        return self.fc(z).view(-1, 16, 48)
