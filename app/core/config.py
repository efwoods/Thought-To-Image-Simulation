from pydantic_settings import BaseSettings
import torch
from typing import Optional


class Settings(BaseSettings):

    # Redis (Docker-managed)
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_PASSWORD: str
    THOUGHT_TO_IMAGE_REDIS_KEY: str

    # Torch
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Models
    IMAGE_DECODER_PATH: str
    RESIZED_IMAGE_SIZE: int
    LATENT_DIM: int

    # Image -> Image_Latent -> Waveform -> Waveform_Latent
    STIMULUS_IMAGES_DIR: str
    USE_PERCEPTUAL_LOSS: bool
    NORMALIZATION_CONFIG: str
    IMAGE_ENCODER_PATH: str
    WAVEFORM_ENCODER_PATH: str
    WAVEFORM_DECODER_PATH: str

    WAVEFORM_DICT_PATH: str
    IMAGE_DICT_PATH: str
    TEST_METADATA_PATH: str

    # GITHUB NGROK CONFIG
    GITHUB_TOKEN: str
    GITHUB_GIST_ID: str

    NGROK_URL: Optional[str] = None

    @property
    def GIST_API_URL(self) -> str:
        return f"https://api.github.com/gists/{self.GITHUB_GIST_ID}"

    @property
    def ROOT_URI(self) -> str:
        return self.NGROK_URL + "/thought-to-image-simulation-api"

    @property
    def RESIZE_DIM(self) -> tuple:
        return (224, 224) if self.USE_PERCEPTUAL_LOSS else (64, 64)

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
