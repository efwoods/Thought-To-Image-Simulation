from PIL import Image
from io import BytesIO
import base64
import websockets
import json
from service.startup import test_indices, dataset
from core.logging import logger
from core.config import settings


# -----------------------------
# Convert PIL Image Path → Base64
# -----------------------------
def encode_image_to_base64(path):
    image = Image.open(path)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base64_img}"


# -----------------------------
# Convert PIL Image → Base64
# -----------------------------
def pil_image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base64_img}"


# -----------------------------
# WebSocket Sender
# -----------------------------
async def send_image(session_id: str, image: Image.Image):
    image_base64 = pil_image_to_base64(image)
    payload = {
        "type": "simulate",
        "session_id": session_id,
        "image_base64": image_base64,
    }
    try:
        async with websockets.connect(
            settings.ROOT_URI + "/simulate/ws/simulate-image-to-waveform-latent"
        ) as websocket:
            await websocket.send(json.dumps(payload))
            response = await websocket.recv()
            print(f"[{session_id}] Response: {response}")
            return response
    except Exception as e:
        print(f"[ERROR] {session_id}: {e}")
        return json.dumps({"error": str(e)})


# Test Image Simulation
# -----------------------------
async def simulate_all_images(session_id: str, index: int):
    idx = test_indices[index % len(test_indices)]
    pil_image, _ = dataset[idx]
    logger.info(f"type(image_tensor):{type(pil_image)}")
    response = await send_image(session_id, pil_image)
    return response
