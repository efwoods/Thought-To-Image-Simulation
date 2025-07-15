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


def decode_and_decompress_tensor(encoded_str_base64: str) -> torch.Tensor:
    compressed = base64.b64decode(encoded_str_base64)
    json_bytes = zlib.decompress(compressed)
    tensor_data = json.loads(json_bytes.decode("utf-8"))
    return torch.tensor(tensor_data, dtype=torch.float32, device=settings.DEVICE)


def decompress_skip_connections(encoded_skip_connections_base64):
    # Decode from base64 string
    compressed = base64.b64decode(encoded_skip_connections_base64.encode("utf-8"))
    # Decompress with zlib
    skip_json = zlib.decompress(compressed).decode("utf-8")
    # Parse JSON string back to nested lists
    skip_data = json.loads(skip_json)
    # Convert nested lists back to tensors
    skip_connections = [
        torch.tensor(skip_connection_data) for skip_connection_data in skip_data
    ]
    return skip_connections
