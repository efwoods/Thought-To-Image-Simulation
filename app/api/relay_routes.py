# api/relay_routes.py

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
from PIL import Image
import datetime

from redis.asyncio import Redis  # requires `redis>=4.2.0`

from service.reconstruct import (
    preprocess_image_from_websocket,
    reconstruct_image_from_waveform_latents,
    decode_and_decompress_tensor,
    decompress_skip_connections,
)

router = APIRouter()

# api/relay_routes.py
from service.websocket_manager import websocket_manager


@router.websocket("/ws/frontend/{user_id}")
async def frontend_websocket(websocket: WebSocket, user_id: str):
    await websocket_manager.connect_frontend(websocket, user_id)
    try:
        while True:
            # Keep connection alive and handle any frontend messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_manager.disconnect_frontend(user_id)
    except Exception as e:
        logger.exception(f"Frontend WebSocket error: {e}")
        websocket_manager.disconnect_frontend(user_id)


@router.websocket("/ws/reconstruct-image-from-waveform-latent")
async def reconstruct(websocket: WebSocket):
    await websocket_manager.connect_reconstruction(websocket)
    try:
        while True:
            message = await websocket.receive_text()
            request = json.loads(message)
            payload = request["payload"]

            # Get user_id from request
            user_id = request["metadata"]["user_id"]

            waveform_latent = decode_and_decompress_tensor(payload["waveform_latent"])
            synthetic_waveform = decode_and_decompress_tensor(
                payload["synthetic_waveform"]
            )
            skip_connections = decompress_skip_connections(payload["skip_connections"])

            # Optional: if skip_connections is a list of tensors (like from UNet), handle as list
            if (
                isinstance(skip_connections, torch.Tensor)
                and skip_connections.ndim == 3
            ):
                skip_connections = [
                    t.unsqueeze(0).to(settings.DEVICE) for t in skip_connections
                ]

            # Reconstruct the image using decoder
            reconstructed_image_tensor = reconstruct_image_from_waveform_latents(
                waveform_latent, skip_connections
            )

            # Convert tensor to PIL Image
            reconstructed_image_tensor = (
                reconstructed_image_tensor.squeeze(0).cpu().clamp(0, 1)
            )
            image_pil = Image.fromarray(
                (reconstructed_image_tensor.permute(1, 2, 0).numpy() * 255).astype(
                    "uint8"
                )
            )

            # Encode image to base64 PNG
            buf = BytesIO()
            image_pil.save(buf, format="PNG")
            image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            previous_metadata = request["metadata"]
            timestamp = datetime.datetime.utcnow().isoformat() + "Z"
            metadata = {
                **previous_metadata,
                "type": "reconstructed_image",
                "timestamp": timestamp,
                "origin": "reconstruct-image-from-waveform-latent",
            }

            # Send to frontend
            if user_id:
                message_data = {
                    "type": "reconstructed_image",
                    "status": "success",
                    "metadata": metadata,
                    "image_data": image_base64,
                    "image_format": "png",
                }

                success = await websocket_manager.send_to_frontend(
                    user_id, message_data
                )

                # Send acknowledgment back to the reconstruction service
                await websocket.send_json(
                    {
                        "status": "success" if success else "partial_success",
                        "message": "Image reconstructed"
                        + (" and forwarded" if success else " but failed to forward"),
                        "metadata": metadata,
                    }
                )
            else:
                await websocket.send_json(
                    {"status": "error", "message": "No user_id provided for forwarding"}
                )

    except WebSocketDisconnect:
        websocket_manager.disconnect_reconstruction(websocket)
    except Exception as e:
        logger.exception(f"Reconstruction error: {e}")
        await websocket.send_json({"status": "error", "message": str(e)})
        websocket_manager.disconnect_reconstruction(websocket)
        await websocket.close()


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
    Provides metadata and expected schema for WebSocket routes used in image reconstruction.
    """
    return {
        "websocket_routes": [
            {
                "route": "/ws/reconstruct-image-from-waveform-latent",
                "description": "Accepts a latent waveform representation and returns a reconstructed image via base64.",
                "message_format": {
                    "waveform_latent": "<base64 or numerical array>",
                    "session_id": "<optional string identifier>",
                    "synthetic_waveform": "<optional boolean>",
                },
                "response_format": {
                    "status": "success",
                    "type": "reconstructed_image",
                    "session_id": "<string>",
                    "image_base64": "data:image/png;base64,<image>",
                },
                "backend_flow": {
                    "storage": "Stores image in Redis using key: reconstructed:<settings.THOUGHT_TO_IMAGE_REDIS_KEY>",
                    "expiration": "600 seconds",
                    "metrics_updated": ["visual_thoughts_rendered"],
                },
            },
            {
                "route": "/ws/test",
                "description": "Sends a test payload and waits for a relay (or frontend) to write the final result back into Redis.",
                "message_format": {
                    "type": "test",
                    "session_id": "<string>",
                    "example_payload": "<any additional fields>",
                },
                "response_format": {
                    "status": "timeout or forwarded",
                    "redis_key": "<Redis key used>",
                    "relay_response": "<payload returned from Redis>",
                },
                "relay_flow": {
                    "wait_time": "30 seconds max (60 attempts x 0.5s)",
                    "storage": "Checks Redis key: reconstructed:<settings.THOUGHT_TO_IMAGE_REDIS_KEY>",
                    "metrics_updated": ["visual_thoughts_rendered"],
                },
            },
        ],
        "dependencies": {
            "redis": "Redis async client (`redis.asyncio.Redis`, requires `redis>=4.2.0`)",
            "image_processing": "torchvision.transforms, PIL.Image",
            "base64_encoding": "Used to send image preview in base64 format",
        },
    }
