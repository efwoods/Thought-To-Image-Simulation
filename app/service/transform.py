# service/transform.py
import torch
import json
import base64
from io import BytesIO
from PIL import Image
from service.image_encoder_loader import load_models, get_image_resize_transform
from core.config import settings
from core.logging import logger
import zlib

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
standard_deviations = (
    torch.tensor([electrode_stats[str(i)]["std"] for i in range(len(electrode_stats))])
    .float()
    .to(settings.DEVICE)
)


def decode_image(base64_data):
    header, encoded = base64_data.split(",", 1)
    return Image.open(BytesIO(base64.b64decode(encoded)))


def preprocess_image_from_websocket(message):
    request = json.loads(message)
    image_base64 = request["payload"]["image_base64"]
    pil_image = decode_image(image_base64)
    request_metadata = request["metadata"]

    image_tensor = image_resize_transform(pil_image).unsqueeze(0).to(settings.DEVICE)
    return image_tensor, request, image_base64, request_metadata


@torch.no_grad()
def transform_image_to_waveform_latents(image_tensor):
    image_latent, skip_connections = image_encoder(image_tensor)
    logger.info(f"type(skip_connections): {type(skip_connections)}")
    synthetic_waveform = waveform_decoder(image_latent)

    # Normalize the waveform using the global means and std
    synthetic_waveform = (
        synthetic_waveform - means[None, :, None]
    ) / standard_deviations[None, :, None]

    waveform_latent = waveform_encoder(synthetic_waveform)
    # waveform_latent is (batch_size = number_of_images, 128)
    return waveform_latent, skip_connections, synthetic_waveform


def compress_skip_connections(skip_connections):
    # Convert tensors to nested lists
    skip_data = [
        skip_connection.detach().cpu().tolist() for skip_connection in skip_connections
    ]

    # Convert to JSON string (text)
    skip_json = json.dumps(skip_data)

    # Compress JSON string with zlib
    compressed = zlib.compress(skip_json.encode("utf-8"))

    # Encode to base64 string for safe WebSocket transmission
    encoded_compressed_skip_connections = base64.b64encode(compressed).decode("utf-8")

    return encoded_compressed_skip_connections


def compress_and_encode_tensor(tensor: torch.Tensor) -> str:
    # Step 1: Convert tensor to nested Python list
    tensor_list = tensor.detach().cpu().tolist()

    # Step 2: Serialize list to JSON string
    json_str = json.dumps(tensor_list)

    # Step 3: Compress JSON string bytes with zlib
    compressed_bytes = zlib.compress(json_str.encode("utf-8"))

    # Step 4: Encode compressed bytes as base64 string
    base64_str = base64.b64encode(compressed_bytes).decode("utf-8")

    return base64_str
