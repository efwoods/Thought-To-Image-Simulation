# api/routes.py

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from core.monitoring import metrics
from core.config import settings
from core.logging import logger
import json
import torch
import base64
import io
import websockets
from torchvision import transforms
from io import BytesIO

from core.config import settings
from core.monitoring import metrics
from core.logging import logger
import asyncio

from redis.asyncio import Redis  # requires `redis>=4.2.0`


from service.reconstruct import (
    preprocess_image_from_websocket,
    reconstruct_image_from_waveform_latents,
)

router = APIRouter()


@router.websocket("/ws/reconstruct-image-from-waveform-latent")
async def simulate(websocket: WebSocket):
    redis_client = websocket.app.state.redis

    await websocket.accept()
    try:
        async for message in websocket:
            waveform_latent, request = preprocess_image_from_websocket(message)

            reconstructed_image = reconstruct_image_from_waveform_latents(
                waveform_latent
            )

            # Convert to base64 to send back to client
            image_pil = transforms.ToPILImage()(reconstructed_image.squeeze().cpu())
            buf = BytesIO()
            image_pil.save(buf, format="PNG")
            image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            # Respond to simulation client
            await websocket.send_json(
                {
                    "status": "success",
                }
            )

            # forward to frontend if available (place in Redis Cache)
            # settings.THOUGHT_TO_IMAGE_REDIS_KEY
            redis_key = f"reconstructed:{settings.THOUGHT_TO_IMAGE_REDIS_KEY}"
            redis_value = json.dumps(
                {
                    "type": "reconstructed_image",
                    "session_id": request.get("session_id", "anonymous"),
                    "image_base64": f"data:image/png;base64,{image_base64}",
                }
            )
            redis_client.set(
                redis_key,
                redis_value,
                ex=600,
            )
            metrics.visual_thoughts_rendered.inc()

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.exception("WebSocket error in image reconstruction:")
        metrics.websocket_errors.inc()


router = APIRouter()


@router.websocket("/ws/test")
async def simulate(websocket: WebSocket):
    redis_client: Redis = websocket.app.state.redis
    logger.info(f"[ImageSimulation] Redis client type: {type(redis_client)}")
    assert isinstance(
        redis_client, Redis
    ), "Incorrect Redis client type: expected async Redis"
    await websocket.accept()

    try:
        while True:
            message = await websocket.receive_text()
            request = json.loads(message)

            if request.get("type") == "test":
                logger.info(f"[ImageSimulation] Received test payload: {request}")

                # Set the Redis key where the *relay* will write the final result
                redis_key = f"reconstructed:{settings.THOUGHT_TO_IMAGE_REDIS_KEY}"
                redis_value = json.dumps(request)

                # Store the original request (for frontend or later traceability)
                await redis_client.set(redis_key, redis_value, ex=600)

                # Optional: Wait for the relay to write its response back into Redis
                logger.info(
                    f"[ImageSimulation] Waiting for response on key: {redis_key}"
                )
                result = None
                for attempt in range(60):  # 60 x 0.5s = 30s timeout
                    val = await redis_client.get(redis_key)
                    if val:
                        result = json.loads(val)
                        break
                    await asyncio.sleep(0.5)  # avoid busy-wait

                if result is None:
                    await websocket.send_json(
                        {
                            "status": "timeout",
                            "message": f"No relay response found in Redis at {redis_key}",
                        }
                    )
                else:
                    await websocket.send_json(
                        {
                            "status": "forwarded",
                            "redis_key": redis_key,
                            "relay_response": result,
                        }
                    )
                    metrics.visual_thoughts_rendered.inc()

            else:
                await websocket.send_json(
                    {"status": "error", "message": "Unsupported message type"}
                )

    except Exception as e:
        logger.error(f"[ImageSimulation] WebSocket error: {e}")
        await websocket.close()


@router.get("/ws-info", tags=["Reconstruct"])
async def reconstructed_image_info():
    """
    Returns metadata and schema for the /ws/reconstruct-image-from-waveform-latent WebSocket endpoint.
    """
    return {
        "endpoint": "/ws/reconstruct-image-from-waveform-latent",
        "full_url": "ws://localhost:8000/relay-waveform-latent-to-image-reconstruction-api/ws/reconstruct-image-from-waveform-latent",
        "protocol": "WebSocket",
        "description": (
            "Reconstructs a full-resolution image from waveform_latent and skip_connections. "
            "Accepts serialized latents over WebSocket and sends back a base64-encoded image. "
            "Image is also stored in Redis cache using the session ID or predetermined key."
        ),
        "input_format": {
            "type": "waveform_latent",
            "session_id": "string (optional, used to identify cached image)",
            "payload": "[float list representing waveform latent]",
            "skip_connections": "serialized PyTorch skip connection object (binary, base64-encoded or torch.save buffer)",
        },
        "output_format": {
            "status": "success",
            "type": "reconstructed_image",
            "session_id": "copied from input or 'anonymous'",
            "image_base64": "data:image/png;base64,...",
        },
        "redis_cache": {
            "key_pattern": "reconstructed:{session_id}",
            "value_format": {
                "type": "reconstructed_image",
                "session_id": "{session_id}",
                "image_base64": "data:image/png;base64,...",
            },
            "expiration_seconds": 600,
        },
    }
