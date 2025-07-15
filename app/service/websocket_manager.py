# core/websocket_manager.py
from typing import Dict, Set
from fastapi import WebSocket
import json
import asyncio
from core.logging import logger


class WebSocketManager:
    def __init__(self):
        # Store active frontend connections by user_id
        self.frontend_connections: Dict[str, WebSocket] = {}
        # Store active reconstruction connections
        self.reconstruction_connections: Set[WebSocket] = set()

    async def connect_frontend(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.frontend_connections[user_id] = websocket
        logger.info(f"Frontend WebSocket connected for user: {user_id}")

    def disconnect_frontend(self, user_id: str):
        if user_id in self.frontend_connections:
            del self.frontend_connections[user_id]
            logger.info(f"Frontend WebSocket disconnected for user: {user_id}")

    async def connect_reconstruction(self, websocket: WebSocket):
        await websocket.accept()
        self.reconstruction_connections.add(websocket)
        logger.info("Reconstruction WebSocket connected")

    def disconnect_reconstruction(self, websocket: WebSocket):
        self.reconstruction_connections.discard(websocket)
        logger.info("Reconstruction WebSocket disconnected")

    async def send_to_frontend(self, user_id: str, message: dict):
        if user_id in self.frontend_connections:
            try:
                await self.frontend_connections[user_id].send_text(json.dumps(message))
                logger.info(f"Message sent to frontend for user: {user_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to send message to frontend: {e}")
                # Remove broken connection
                self.disconnect_frontend(user_id)
                return False
        else:
            logger.warning(f"No frontend connection found for user: {user_id}")
            return False


# Global instance
websocket_manager = WebSocketManager()
