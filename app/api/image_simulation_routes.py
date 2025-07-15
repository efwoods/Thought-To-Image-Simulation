# api/image_simulation_routes.py

from fastapi import APIRouter
from fastapi import WebSocket, WebSocketDisconnect
from core.monitoring import metrics
from core.config import settings
from core.logging import logger
import json
import io
import torch
import websockets

from core.config import settings

from service.transform import (
    preprocess_image_from_websocket,
    transform_image_to_waveform_latents,
    compress_skip_connections,
    compress_and_encode_tensor,
)
import datetime

router = APIRouter()


@router.websocket("/ws/simulate-image-to-waveform-latent")
async def simulate(websocket: WebSocket):
    await websocket.accept()

    logger.info(
        "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX simulate-image-to-waveform-latent TRY LOOP"
    )

    try:
        while True:
            message = await websocket.receive_text()

            image_tensor, request, image_base64, request_metadata = (
                preprocess_image_from_websocket(message)
            )

            waveform_latent, skip_connections, synthetic_waveform = (
                transform_image_to_waveform_latents(image_tensor)
            )

            encoded_compressed_skip_connections = compress_skip_connections(
                skip_connections
            )

            logger.info("POST SKIP_CONNECTION CREATION")

            base64_str_compressed_synthetic_waveform = compress_and_encode_tensor(
                synthetic_waveform
            )
            compressed_encoded_waveform_latent_base64 = compress_and_encode_tensor(
                waveform_latent
            )

            timestamp = datetime.datetime.utcnow().isoformat() + "Z"
            metadata = {
                **request_metadata,
                "type": "test",
                "timestamp": timestamp,
                "origin": "simulate-image-to-waveform-latent",
            }

            message = {
                "payload": {
                    "waveform_latent": compressed_encoded_waveform_latent_base64,
                    "synthetic_waveform": base64_str_compressed_synthetic_waveform,
                    "image_base64": image_base64,
                    "skip_connections": encoded_compressed_skip_connections,
                },
                "metadata": metadata,
            }

            # Forward pipeline
            async with websockets.connect(
                settings.ROOT_URI
                + "/reconstruct/ws/reconstruct-image-from-waveform-latent"
            ) as relay_ws:
                await relay_ws.send(json.dumps(message))

            # Respond back
            await websocket.send_json(
                {"status": "success", "metadata": metadata},
            )
            metrics.visual_thoughts_simulated.inc()

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.exception(f"Unhandled error in WebSocket handler: {e}")
        metrics.websocket_errors.inc()


@router.websocket("/ws/test")
async def simulate(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            message = await websocket.receive_text()
            request = json.loads(message)

            if request.get("type") == "test":
                logger.info(f"[ImageSimulation] Received test payload: {request}")
                logger.info(
                    f"settings.ROOT_URI + /reconstruct/ws/test: {settings.ROOT_URI}"
                    + "/reconstruct/ws/test"
                )
                logger.info(f"json.dumps(request): {json.dumps(request)}")
                # Forward to the relay WebSocket
                try:
                    async with websockets.connect(
                        settings.ROOT_URI + "/reconstruct/ws/test"
                    ) as relay_ws:
                        await relay_ws.send(json.dumps(request))
                        relay_response = await relay_ws.recv()
                        relay_data = json.loads(relay_response)
                        logger.info(f"[ImageSimulation] Relay responded: {relay_data}")
                except Exception as relay_error:
                    await websocket.send_json(
                        {
                            "status": "error",
                            "message": f"Relay failed: {str(relay_error)}",
                        }
                    )
                    return

                # Respond back to the sender (webcam-to-websocket-simulation)
                await websocket.send_json(
                    {"status": "forwarded", "relay_response": relay_data}
                )

            else:
                await websocket.send_json(
                    {"status": "error", "message": "Unsupported message type"}
                )

    except Exception as e:
        print(f"[ImageSimulation] WebSocket error: {e}")
        await websocket.close()


@router.get("/ws-info", tags=["Simulate"])
async def websocket_info():
    """
    Provides metadata and message schema for the WebSocket routes related to image-to-waveform-latent simulation.
    """
    return {
        "websocket_routes": [
            {
                "route": "/ws/simulate-image-to-waveform-latent",
                "description": (
                    "Receives an image from the client, transforms it into a latent waveform "
                    "representation (shape: [batch_size, 128]), and forwards it to the relay "
                    "at /reconstruct/ws/reconstruct-image-from-waveform-latent."
                ),
                "message_format": {
                    "type": "image_upload",
                    "session_id": "<string>",
                    "image_base64": "data:image/png;base64,<image>",
                },
                "response_format": {"status": "success", "latents": "[float[128]]"},
                "relay_behavior": {
                    "forwarded_payload": {
                        "type": "waveform_latent",
                        "session_id": "<string>",
                        "payload": "[float[128]]",
                        "synthetic_waveform": "<bool>",
                        "img_data": "<original image data>",
                    },
                    "destination": "/reconstruct/ws/reconstruct-image-from-waveform-latent",
                    "connection": "Outgoing WebSocket using `websockets.connect()`",
                },
                "backend_flow": {
                    "transforms": [
                        "preprocess_image_from_websocket (message -> torch.Tensor)",
                        "transform_image_to_waveform_latents (image_tensor -> latent)",
                    ],
                    "metrics_updated": ["visual_thoughts_simulated"],
                },
            },
            {
                "route": "/ws/test",
                "description": (
                    "Echo test endpoint that forwards a test payload to "
                    "/reconstruct/ws/test and returns the relay response. "
                    "Useful for verifying relay server connectivity."
                ),
                "message_format": {
                    "type": "test",
                    "session_id": "<string>",
                    "any_additional_fields": "Any",
                },
                "response_format": {
                    "status": "forwarded | error",
                    "relay_response": "Echoed or simulated relay response",
                },
                "relay_flow": {
                    "forwarded_to": "/reconstruct/ws/test",
                    "method": "websockets.connect(...) + send(json)",
                },
            },
        ],
        "dependencies": {
            "torch": "Used for image tensor transformation",
            "websockets": "Used for client-server WebSocket relay",
            "base64": "Input images must be base64-encoded PNGs",
            "metrics": {
                "visual_thoughts_simulated": "Increments when image->waveform latent completed",
                "websocket_errors": "Increments on failure",
            },
        },
    }
