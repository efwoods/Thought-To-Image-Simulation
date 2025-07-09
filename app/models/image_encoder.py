# models/image_encoder
import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        base = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )  # This was trained on 1.2 million images to extract textures, shapes, & semantic features
        # Decompose layers for access to intermediate features
        self.conv1 = base.conv1  # output: 64@112x112 (after stride=2)
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool  # output: 64@56x56

        self.layer1 = base.layer1  # 64@56x56
        self.layer2 = base.layer2  # 128@28x28
        self.layer3 = base.layer3  # 256@14x14
        self.layer4 = base.layer4  # 512@7x7

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        skip0 = x  # 64@112x112 (low-level edges/textures)

        x = self.maxpool(x)
        x = self.layer1(x)
        skip1 = x  # 64@56x56 (basic-shapes)

        x = self.layer2(x)
        skip2 = x  # 128@28x28 (parts of objects)

        x = self.layer3(x)
        skip3 = x  # 256@14x14 (semantic parts of an image)

        x = self.layer4(x)  # 512@7x7
        pooled = self.avgpool(x)
        flat = self.flatten(pooled)
        latent = self.fc(flat)

        # Select which skip features to pass to decoder
        # Adjust sizes as necessary for your decoder
        # Here, pass three skip tensors (skip3, skip2, skip1) for your decoder's expected dims
        return latent, (skip3, skip2, skip1)
