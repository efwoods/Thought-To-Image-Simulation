# service/transform.py
import torch
import json
import base64
from io import BytesIO
from PIL import Image
from service.image_encoder_loader import load_models, get_image_resize_transform
from core.config import settings

image_encoder, waveform_decoder, waveform_encoder = load_models()
image_resize_transform = get_image_resize_transform()


# Load normalization stats
import json

with open(settings.NORMALIZATION_CONFIG, "r") as f:
    electrode_stats = json.load(f)

# Convert to tensors for batch use
means = (
    torch.tensor([electrode_stats[str(i)]["mean"] for i in range(len(electrode_stats))])
    .float()
    .to(settings.DEVICE)
)
stds = (
    torch.tensor([electrode_stats[str(i)]["std"] for i in range(len(electrode_stats))])
    .float()
    .to(settings.DEVICE)
)


def preprocess_image_from_websocket(message):
    request = json.loads(message)
    img_data = request["image_base64"]
    image_bytes = base64.b64decode(img_data.split(",")[1])
    image_tensor = (
        image_resize_transform(Image.open(BytesIO(image_bytes)))
        .unsqueeze(0)
        .to(settings.DEVICE)
    )
    return image_tensor, request


@torch.no_grad()
def transform_image_to_waveform_latents(image_tensor):
    image_latent, skip_connections = image_encoder(image_tensor)
    synthetic_waveform = waveform_decoder(image_latent)

    # Normalize the waveform using the global means and std
    synthetic_waveform = (synthetic_waveform - means[None, :, None]) / stds[
        None, :, None
    ]

    waveform_latent = waveform_encoder(synthetic_waveform)
    # waveform_latent is (batch_size = number_of_images, 128)
    return waveform_latent, skip_connections
