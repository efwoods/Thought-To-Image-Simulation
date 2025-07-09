# models/image_decoder.py
import torch
import torch.nn as nn
import torchvision.models as models


class ImageDecoder(nn.Module):
    def __init__(self, latent_dim=128, out_channels=3, skip_connections=False):
        super().__init__()
        self.skip_connections = skip_connections

        self.fc = nn.Sequential(nn.Linear(latent_dim, 8 * 8 * 256), nn.ReLU())
        if self.skip_connections:
            # When using skip connections, input channels are larger due to concatenation
            self.up1 = nn.Sequential(
                nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1),  # 8→16
                nn.ReLU(),
            )
            self.up2 = nn.Sequential(
                nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1),  # 16→32
                nn.ReLU(),
            )
            self.up3 = nn.Sequential(
                nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),  # 32→64
                nn.Sigmoid(),
            )
        else:
            # Without skip connections, channels are smaller
            self.up1 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 8→16
                nn.ReLU(),
            )
            self.up2 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16→32
                nn.ReLU(),
            )
            self.up3 = nn.Sequential(
                nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1),  # 32→64
                nn.Sigmoid(),
            )

    def forward(self, z, skip_connections=None):
        # Initial projection from latent vector
        x = self.fc(z).view(-1, 256, 8, 8)  # Initialize x from latent z
        # print(f"x initial shape: {x.shape}")
        # print(f"skip_connections shapes: {[s.shape for s in skip_connections]}")
        if skip_connections:
            # Ensure skip[0] matches x
            skip0 = skip_connections[0]
            if skip0.shape[2:] != x.shape[2:]:
                skip0 = torch.nn.functional.interpolate(
                    skip0, size=x.shape[2:], mode="nearest"
                )
            x = torch.cat([x, skip0], dim=1)  # +256 → 512
            x = self.up1(x)

            # up1 output shape is probably (B, 256, 16, 16), match with skip1
            skip1 = skip_connections[1]
            if skip1.shape[2:] != x.shape[2:]:
                skip1 = torch.nn.functional.interpolate(
                    skip1, size=x.shape[2:], mode="nearest"
                )
            x = torch.cat([x, skip1], dim=1)  # +128 → 256
            x = self.up2(x)

            # up2 output shape is probably (B, 128, 32, 32), match with skip2
            skip2 = skip_connections[2]
            if skip2.shape[2:] != x.shape[2:]:
                skip2 = torch.nn.functional.interpolate(
                    skip2, size=x.shape[2:], mode="nearest"
                )
            x = torch.cat([x, skip2], dim=1)  # +64 → 128
            x = self.up3(x)
        else:
            # No skip connections path
            x = self.up1(x)  # (B, 128, 16, 16)
            x = self.up2(x)  # (B, 64, 32, 32)
            x = self.up3(x)  # (B, out_channels, 64, 64)

        return x
