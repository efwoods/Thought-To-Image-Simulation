# models/loader.py
import torch
from torchvision import transforms
from models.image_encoder import ImageEncoder
from models.waveform_decoder import WaveformDecoder
from models.waveform_encoder import WaveformEncoder
from core.config import settings


def load_models():

    image_encoder = ImageEncoder(latent_dim=settings.LATENT_DIM).to(settings.DEVICE)
    waveform_decoder = WaveformDecoder(latent_dim=settings.LATENT_DIM).to(
        settings.DEVICE
    )
    waveform_encoder = WaveformEncoder(latent_dim=settings.LATENT_DIM).to(
        settings.DEVICE
    )

    if torch.__version__ >= "2.0.0":
        image_encoder = torch.compile(image_encoder)
        waveform_decoder = torch.compile(waveform_decoder)
        waveform_encoder = torch.compile(waveform_encoder)

    image_encoder.load_state_dict(
        torch.load(settings.IMAGE_ENCODER_PATH, map_location=settings.DEVICE)
    )
    waveform_decoder.load_state_dict(
        torch.load(settings.WAVEFORM_DECODER_PATH, map_location=settings.DEVICE)
    )
    waveform_encoder.load_state_dict(
        torch.load(settings.WAVEFORM_ENCODER_PATH, map_location=settings.DEVICE)
    )

    image_encoder.eval()
    waveform_decoder.eval()
    waveform_encoder.eval()

    return image_encoder, waveform_decoder, waveform_encoder


def get_image_resize_transform():
    return transforms.Compose(
        [
            transforms.Resize(
                (settings.RESIZED_IMAGE_SIZE, settings.RESIZED_IMAGE_SIZE)
            ),
            transforms.ToTensor(),
        ]
    )
