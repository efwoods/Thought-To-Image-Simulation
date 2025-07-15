# service/transform.py
import torch
import json
import base64
from io import BytesIO
from PIL import Image
from service.decoder_loader import load_models, get_image_resize_transform
from core.config import settings
import zlib

image_decoder = load_models()
image_resize_transform = get_image_resize_transform()


def preprocess_image_from_websocket(message):
    request = json.loads(message)

    # Parse waveform_latent and convert to tensor
    waveform_latent = torch.tensor(
        request["payload"], dtype=torch.float32, device=settings.DEVICE
    ).unsqueeze(0)

    synthetic_waveform = request["synthetic_waveform"]
    img_data = request["img_data"]
    image_bytes = base64.b64decode(img_data.split(",")[1])

    return waveform_latent, request, synthetic_waveform, image_bytes


@torch.no_grad()
def reconstruct_image_from_waveform_latents(waveform_latent, skip_connections=None):
    reconstructed_image = image_decoder(
        waveform_latent, skip_connections=skip_connections
    )
    return reconstructed_image


def decode_and_decompress_tensor(encoded_str: str) -> torch.Tensor:
    compressed = base64.b64decode(encoded_str)
    json_bytes = zlib.decompress(compressed)
    tensor_data = json.loads(json_bytes.decode("utf-8"))
    return torch.tensor(tensor_data, dtype=torch.float32, device=settings.DEVICE)
