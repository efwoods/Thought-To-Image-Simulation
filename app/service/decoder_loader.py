# models/loader.py
import torch
from torchvision import transforms
from models.image_decoder import ImageDecoder
from core.config import settings


def load_models():

    image_decoder = ImageDecoder(
        latent_dim=settings.LATENT_DIM,
        skip_connections=True,  # Preventing the use of skip connections to decrease latency
    ).to(settings.DEVICE)

    if torch.__version__ >= "2.0.0":
        image_decoder = torch.compile(image_decoder)

    image_decoder.load_state_dict(
        torch.load(settings.IMAGE_DECODER_PATH, map_location=settings.DEVICE)
    )

    image_decoder.eval()

    return image_decoder


def get_image_resize_transform():
    return transforms.Compose(
        [
            transforms.Resize(
                (settings.RESIZED_IMAGE_SIZE, settings.RESIZED_IMAGE_SIZE)
            ),
        ]
    )
