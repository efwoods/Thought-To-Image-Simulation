import os
import re
import json
import pickle
import base64
from io import BytesIO
from contextlib import asynccontextmanager
from pydantic import BaseModel
import datetime
from fastapi import FastAPI
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

from contextlib import asynccontextmanager

import requests


# This file defines the complete MVP architecture for Neural Nexus: Simulation API, Relay API, and Frontend structure.

# ============================
# SIMULATION API - WebSocket Server
# ============================
# Accepts image, encodes it, decodes to waveform, sends waveform_latent to Relay API

# app/main.py
import asyncio
import websockets
import torch
import json
import base64
import numpy as np
from torchvision import transforms
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from core.config import settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

import uvicorn
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from fastapi import Request

from contextlib import asynccontextmanager

# Configurations & Metrics
from core.config import settings
from core.monitoring import metrics
from core.logging import logger

# app/main.py
import asyncio
import websockets
import torch
import json
import base64
import numpy as np
from torchvision import transforms
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from core.config import settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

import uvicorn
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from fastapi import Request

from contextlib import asynccontextmanager

# Configurations & Metrics
from core.config import settings
from core.monitoring import metrics
from core.logging import logger
from service.startup import fetch_ngrok_url


from redis.asyncio import Redis


# Configurations & Metrics
# from core.config import settings
# from core.monitoring import metrics

# API Routes
# from api.routes import router
# API Routes
from api.relay_routes import router as relay_router
from api.image_simulation_routes import router as images_simulation_router
from api.webcam_to_websocket_routes import router as webcam_to_websocket_router

from data.dataset import ImageWaveformDataset


# -----------------------------
# FastAPI App
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Webcam-to-Websocket
    print("[Startup] FastAPI simulation sender is live")

    fetch_ngrok_url()
    logger.info(f"ngrok_url set: {settings.NGROK_URL}")
    # Relay-WS-Thought-to-image
    # Startup: initialize Whisper model
    load_dotenv()
    redis_client = Redis(
        host=settings.REDIS_HOST,  # e.g. "localhost" or a container name like "redis"
        port=settings.REDIS_PORT,  # usually 6379
        password=settings.REDIS_PASSWORD,
        decode_responses=True,  # Automatically decode UTF-8 strings
    )

    app.state.redis = redis_client  # make available globally

    yield
    print("[Shutdown] Shutting down sender...")


app = FastAPI(
    title="Thought-to-Image-Simulation-API",
    root_path="/thought-to-image-simulation-api",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(relay_router, prefix="/reconstruct", tags=["Reconstruct"])
app.include_router(images_simulation_router, prefix="/simulate", tags=["Simulate"])
app.include_router(
    webcam_to_websocket_router, prefix="/initialize", tags=["Initialize"]
)


@app.get("/")
async def root(request: Request):
    return RedirectResponse(url=request.scope.get("root_path", "") + "/docs")


@app.get("/health")
async def health():
    # metrics.health_requests.inc()
    return {"status": "healthy"}


@app.router.get("/metrics")
def metrics_endpoint():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.FASTAPI_PORT, log_level="debug")
