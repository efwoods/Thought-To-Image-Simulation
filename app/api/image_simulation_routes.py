# api/routes.py

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
)

router = APIRouter()


@router.websocket("/ws/simulate-image-to-waveform-latent")
async def simulate(websocket: WebSocket):
    await websocket.accept()

    try:
        async for message in websocket:
            image_tensor, request = preprocess_image_from_websocket(message)
            waveform_latent, skip_connections = transform_image_to_waveform_latents(
                image_tensor
            )
            payload = {
                "type": "waveform_latent",  # (batch_size = number_of_images, 128)
                "session_id": request.get("session_id", "anonymous"),
                "payload": waveform_latent.squeeze().cpu().tolist(),
            }

            # Forward to latents to relay
            async with websockets.connect(
                settings.ROOT_URI + "/ws/reconstruct-image-from-waveform-latent"
            ) as relay_ws:
                await relay_ws.send(json.dumps(payload))

            # Optional: Send response back to client
            await websocket.send_json(
                {"status": "success", "latents": payload["payload"]}
            )
            metrics.visual_thoughts_simulated.inc()

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        print(f"Error: {e}")
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
    return {
        "endpoint": {
            "/simulate/ws/simulate-image-to-waveform-latent",
            "/simulate/ws/test",
        },
        "full_url": {
            "ws://***.ngrok-free.app/thought-to-image-simulation-api/simulate/ws/simulate-image-to-waveform-latent",
            "ws://***.ngrok-free.app/thought-to-image-simulation-api/simulate/ws/test",
        },
        "protocol": "WebSocket",
        "description": "Real-time simulation of image → synthetic waveform → waveform latent.",
        "input_format": {
            "type": "simulate",
            "session_id": "string (optional)",
            "image_base64": "data:image/png;base64,...",
        },
        "output_format": {
            "type": "waveform_latent",
            "session_id": "copied from input",
            "payload": "[float list representing latent]",
        },
    }
