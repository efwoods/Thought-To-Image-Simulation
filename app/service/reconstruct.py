# service/transform.py
import torch
import json
import base64
from io import BytesIO
from PIL import Image
from service.decoder_loader import load_models, get_image_resize_transform
from core.config import settings

image_decoder = load_models()
image_resize_transform = get_image_resize_transform()


def preprocess_image_from_websocket(message):
    request = json.loads(message)

    # Parse waveform_latent and convert to tensor
    waveform_latent = torch.tensor(
        request["payload"], dtype=torch.float32, device=settings.DEVICE
    ).unsqueeze(0)

    return waveform_latent, request


@torch.no_grad()
def reconstruct_image_from_waveform_latents(waveform_latent):
    reconstructed_image = image_decoder(waveform_latent)
    return reconstructed_image
