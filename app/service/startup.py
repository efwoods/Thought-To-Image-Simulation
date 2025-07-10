import os
import re
import requests
from core.config import settings
import pickle
from torchvision import transforms

from data.dataset import ImageWaveformDataset

wavform_dict = None
image_paths = None
test_mega = None
dataset = None
test_indices = None


# -----------------------------
# Extract NGROK URL
# -----------------------------
def extract_ngrok_url(text: str) -> str:
    """
    Extract the ngrok URL from raw gist text using regex
    """
    match = re.search(r"https://[\w\-]+\.ngrok-free\.app", text)
    if match:
        return match.group(0).replace("https", "wss")
    raise ValueError("No ngrok URL found in gist")


# -----------------------------
# Fetch NGROK URL
# -----------------------------
def fetch_ngrok_url():

    if not settings.GITHUB_TOKEN:
        print("GitHub token is missing in environment variable `GITHUB_TOKEN`.")
        return

    headers = {
        "Authorization": f"token {settings.GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

    try:
        response = requests.get(settings.GIST_API_URL, headers=headers, timeout=10)
        response.raise_for_status()
        gist_data = response.json()

        # Assuming there's only one file in the Gist, extract its content
        file_content = next(iter(gist_data["files"].values()))["content"]

        # Extract ngrok URL
        settings.NGROK_URL = extract_ngrok_url(file_content)

        print(f"[Startup] Ngrok URL loaded: {settings.NGROK_URL}")

    except Exception as e:
        print(f"[Startup] Failed to fetch or parse Gist: {e}")


# -----------------------------
# Load Dataset
# -----------------------------
def load_dataset():
    global waveform_dict
    global image_paths
    global test_mega
    global dataset
    global test_indices

    # -----------------------------
    # Transform
    # -----------------------------
    image_resize_transform = transforms.Compose(
        [
            transforms.Resize(settings.RESIZE_DIM),
        ]
    )

    with open(settings.WAVFORM_DICT_PATH, "rb") as f:
        waveform_dict = pickle.load(f)
    with open(settings.IMAGE_DICT_PATH, "rb") as f:
        image_paths = pickle.load(f)
    with open(settings.TEST_METADATA_PATH, "rb") as f:
        test_meta = pickle.load(f)
    test_indices = test_meta["indices"]
    dataset = ImageWaveformDataset(waveform_dict, image_paths, image_resize_transform)
