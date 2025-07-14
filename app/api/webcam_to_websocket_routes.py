import os
import re
import json
import pickle
import base64
from io import BytesIO
from contextlib import asynccontextmanager
from pydantic import BaseModel
import datetime
from fastapi import FastAPI, APIRouter
from PIL import Image
import websockets
from dotenv import load_dotenv
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

import uvicorn
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from fastapi import Request
from core.logging import logger
from core.config import settings

from contextlib import asynccontextmanager

import requests

from models.SimulationRequest import SimulationRequest
from service.webcam_to_websocket_service import (
    pil_image_to_base64,
    send_image,
    simulate_all_images,
)
import pickle

IMAGE_PATHS_FLAT_LIST = "/data/image_paths_dict_flat_list.pkl"

simulation_image_index_test_full_pipeline = 0
process_thought_to_image_index = 0
router = APIRouter()


@router.post("/test/full-pipeline")
async def test_pipeline(payload: SimulationRequest):
    global simulation_image_index_test_full_pipeline
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    message = {
        "type": "test",
        "timestamp": timestamp,
        "origin": "webcam-to-websocket-simulation",
        "simulation_image_index": simulation_image_index_test_full_pipeline,
        "payload": payload.json(),
    }
    logger.info(
        f"settings.ROOT_URI + /simulate/ws/test: {settings.ROOT_URI}"
        + "/simulate/ws/test"
    )
    logger.info(f"simulation_image_index: {simulation_image_index_test_full_pipeline}")

    try:
        async with websockets.connect(
            settings.ROOT_URI + "/simulate/ws/test"
        ) as websocket:
            await websocket.send(json.dumps(message))
            response = await websocket.recv()
            logger.info(f"[{timestamp}] Response: {response}")
            simulation_image_index += 1
            return response
    except Exception as e:
        logger.error(f"[ERROR] {timestamp}: {e}")
        return json.dumps({"error": str(e)})


@router.post("/enable-thought-to-image")
async def process_thought_to_image(payload: SimulationRequest):
    """
    Every time this endpoint is called, sample images will be synthesized into waveforms and reconstructed into images.
    The return is the original image, the synthesized neural waveform, and the reconstructed image.
    """
    # Simulate Test Images
    global process_thought_to_image_index
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    with open(IMAGE_PATHS_FLAT_LIST, "rb") as f:
        sample_stimulus_image_path_list = pickle.load(f)

    message = {
        "type": "test",
        "timestamp": timestamp,
        "origin": "webcam-to-websocket-simulation",
        "process_thought_to_image_index": process_thought_to_image_index,
        "payload": payload.json(),
    }
    logger.info(
        f"settings.ROOT_URI + /simulate/ws/simulate-image-to-waveform-latent: {settings.ROOT_URI}"
        + "/simulate/ws/simulate-image-to-waveform-latent"
    )
    logger.info(f"process_thought_to_image_index: {process_thought_to_image_index}")

    try:
        async with websockets.connect(
            settings.ROOT_URI + "/simulate/ws/simulate-image-to-waveform-latent"
        ) as websocket:
            await websocket.send(json.dumps(message))
            response = await websocket.recv()
            logger.info(f"[{timestamp}] Response: {response}")
            process_thought_to_image_index += 1
            return response
    except Exception as e:
        logger.error(f"[ERROR] {timestamp}: {e}")
        return json.dumps({"error": str(e)})


@router.post("/simulate-test-images")
async def simulate_test_images(payload: SimulationRequest):
    # Simulate Test Images
    return {
        "response": "success",
        "description": "This accepts an image and returns the synthetic waveform, the synthetic waveform image, and the reconstructed image.",
    }


@router.post("/simulate-webcam-stream")
async def simulate_test_images(payload: SimulationRequest):
    # Simulate Webcam Images
    return {
        "response": "success",
        "description": "this will enable accept a webcame stream of images and return the associated synthetic neural waveform and the reconstructed image.",
    }


@router.post("/simulate-test-waveform")
async def simulate_test_images(payload: SimulationRequest):
    # Simulate Test Waveform Logic
    return {
        "response": "success",
        "description": "This will accept a synthetic or raw neural waveform and return the reconstructed image.",
    }
