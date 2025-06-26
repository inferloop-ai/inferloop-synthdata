"""
WebSocket Management Module

Provides unified WebSocket management for all Inferloop services.
Supports connection management, broadcasting, and real-time communication.
"""

import asyncio
import json
import uuid
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime
import logging

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState


class WebSocketConnection:
    """Represents a WebSocket connection"""
    
    def __init__(self, websocket: WebSocket, connection_id: str, user_id: Optional[str] = None):
        self.websocket = websocket
        self.connection_id = connection_id
        self.user_id = user_id
        self.created_at = datetime.utcnow()
        self.last_ping = datetime.utcnow()
        self.metadata: Dict[str, Any] = {}
        self.subscriptions: Set[str] = set()
    
    async def send_json(self, data: Dict[str, Any]):
        """Send JSON data to client"""
        if self.websocket.client_state == WebSocketState.CONNECTED:
            await self.websocket.send_json(data)
    
    async def send_text(self, text: str):
        """Send text data to client"""
        if self.websocket.client_state == WebSocketState.CONNECTED:
            await self.websocket.send_text(text)
    
    async def send_bytes(self, data: bytes):
        """Send binary data to client"""
        if self.websocket.client_state == WebSocketState.CONNECTED:
            await self.websocket.send_bytes(data)
    
    def is_connected(self) -> bool:
        """Check if connection is still active"""
        return self.websocket.client_state == WebSocketState.CONNECTED
    
    async def close(self, code: int = 1000, reason: str = ""):
        """Close the connection"""
        if self.is_connected():
            await self.websocket.close(code=code, reason=reason)


class WebSocketManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self.room_connections: Dict[str, Set[str]] = {}  # room_name -> connection_ids
        self.message_handlers: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)
    
    async def connect(self, connection_id: str, websocket: WebSocket, user_id: Optional[str] = None) -> WebSocketConnection:
        """Accept and register a new WebSocket connection"""
        connection = WebSocketConnection(websocket, connection_id, user_id)
        
        # Store connection
        self.connections[connection_id] = connection
        
        # Index by user_id if provided
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            self.user_connections[user_id].add(connection_id)
        
        self.logger.info(f"WebSocket connection {connection_id} connected for user {user_id}")
        
        return connection
    
    def disconnect(self, connection_id: str):
        """Disconnect and remove a WebSocket connection"""
        if connection_id in self.connections:
            connection = self.connections[connection_id]
            
            # Remove from user index
            if connection.user_id and connection.user_id in self.user_connections:
                self.user_connections[connection.user_id].discard(connection_id)
                if not self.user_connections[connection.user_id]:
                    del self.user_connections[connection.user_id]
            
            # Remove from room subscriptions
            for room_name in list(connection.subscriptions):
                self.leave_room(connection_id, room_name)
            
            # Remove connection
            del self.connections[connection_id]
            
            self.logger.info(f"WebSocket connection {connection_id} disconnected")
    
    def get_connection(self, connection_id: str) -> Optional[WebSocketConnection]:
        """Get connection by ID"""
        return self.connections.get(connection_id)
    
    def get_user_connections(self, user_id: str) -> List[WebSocketConnection]:
        """Get all connections for a user"""
        connection_ids = self.user_connections.get(user_id, set())
        return [self.connections[conn_id] for conn_id in connection_ids if conn_id in self.connections]
    
    def join_room(self, connection_id: str, room_name: str):
        """Add connection to a room"""
        if connection_id in self.connections:
            if room_name not in self.room_connections:
                self.room_connections[room_name] = set()
            
            self.room_connections[room_name].add(connection_id)
            self.connections[connection_id].subscriptions.add(room_name)
            
            self.logger.debug(f"Connection {connection_id} joined room {room_name}")
    
    def leave_room(self, connection_id: str, room_name: str):
        """Remove connection from a room"""
        if room_name in self.room_connections:
            self.room_connections[room_name].discard(connection_id)
            if not self.room_connections[room_name]:
                del self.room_connections[room_name]
        
        if connection_id in self.connections:
            self.connections[connection_id].subscriptions.discard(room_name)
            
        self.logger.debug(f"Connection {connection_id} left room {room_name}")
    
    async def send_to_connection(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """Send message to specific connection"""
        connection = self.get_connection(connection_id)
        if connection and connection.is_connected():
            try:
                await connection.send_json(message)
                return True
            except Exception as e:
                self.logger.error(f"Failed to send message to connection {connection_id}: {e}")
                self.disconnect(connection_id)
        return False
    
    async def send_to_user(self, user_id: str, message: Dict[str, Any]) -> int:
        """Send message to all connections for a user"""
        connections = self.get_user_connections(user_id)
        sent_count = 0
        
        for connection in connections:
            if await self.send_to_connection(connection.connection_id, message):
                sent_count += 1
        
        return sent_count
    
    async def broadcast_to_room(self, room_name: str, message: Dict[str, Any], exclude_connection: Optional[str] = None) -> int:
        """Broadcast message to all connections in a room"""
        if room_name not in self.room_connections:
            return 0
        
        connection_ids = self.room_connections[room_name].copy()
        sent_count = 0
        
        for connection_id in connection_ids:
            if connection_id != exclude_connection:
                if await self.send_to_connection(connection_id, message):
                    sent_count += 1
        
        return sent_count
    
    async def broadcast_to_all(self, message: Dict[str, Any], exclude_connection: Optional[str] = None) -> int:
        """Broadcast message to all connections"""
        sent_count = 0
        
        for connection_id in list(self.connections.keys()):
            if connection_id != exclude_connection:
                if await self.send_to_connection(connection_id, message):
                    sent_count += 1
        
        return sent_count
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register a message handler for a specific message type"""
        self.message_handlers[message_type] = handler
    
    async def handle_message(self, connection_id: str, message: Dict[str, Any]):
        """Handle incoming message from connection"""
        message_type = message.get('type')
        
        if message_type in self.message_handlers:
            try:
                await self.message_handlers[message_type](connection_id, message)
            except Exception as e:
                self.logger.error(f"Error handling message type {message_type}: {e}")
                
                # Send error response
                error_response = {
                    'type': 'error',
                    'error': f'Failed to handle message: {str(e)}',
                    'original_message_type': message_type
                }
                await self.send_to_connection(connection_id, error_response)
        else:
            self.logger.warning(f"Unknown message type: {message_type}")
            
            # Send error response
            error_response = {
                'type': 'error',
                'error': f'Unknown message type: {message_type}'
            }
            await self.send_to_connection(connection_id, error_response)
    
    async def ping_all_connections(self):
        """Send ping to all connections to check if they're alive"""
        ping_message = {
            'type': 'ping',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        disconnected_connections = []
        
        for connection_id, connection in self.connections.items():
            try:
                if connection.is_connected():
                    await connection.send_json(ping_message)
                    connection.last_ping = datetime.utcnow()
                else:
                    disconnected_connections.append(connection_id)
            except Exception as e:
                self.logger.error(f"Failed to ping connection {connection_id}: {e}")
                disconnected_connections.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected_connections:
            self.disconnect(connection_id)
    
    async def cleanup_stale_connections(self, max_age_minutes: int = 60):
        """Clean up connections that haven't been pinged recently"""
        cutoff_time = datetime.utcnow().timestamp() - (max_age_minutes * 60)
        stale_connections = []
        
        for connection_id, connection in self.connections.items():
            if connection.last_ping.timestamp() < cutoff_time:
                stale_connections.append(connection_id)
        
        for connection_id in stale_connections:
            self.logger.info(f"Cleaning up stale connection {connection_id}")
            connection = self.connections[connection_id]
            await connection.close(code=1001, reason="Connection stale")
            self.disconnect(connection_id)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            'total_connections': len(self.connections),
            'unique_users': len(self.user_connections),
            'total_rooms': len(self.room_connections),
            'connections_by_user': {
                user_id: len(conn_ids) 
                for user_id, conn_ids in self.user_connections.items()
            },
            'connections_by_room': {
                room_name: len(conn_ids)
                for room_name, conn_ids in self.room_connections.items()
            }
        }
    
    async def close_all(self):
        """Close all connections"""
        for connection in list(self.connections.values()):
            if connection.is_connected():
                await connection.close(code=1001, reason="Server shutdown")
        
        self.connections.clear()
        self.user_connections.clear()
        self.room_connections.clear()


class StreamingManager:
    """Manages streaming data over WebSocket connections"""
    
    def __init__(self, websocket_manager: WebSocketManager):
        self.ws_manager = websocket_manager
        self.active_streams: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    async def start_stream(self, connection_id: str, stream_id: str, stream_type: str, metadata: Dict[str, Any] = None) -> bool:
        """Start a new stream"""
        connection = self.ws_manager.get_connection(connection_id)
        if not connection or not connection.is_connected():
            return False
        
        self.active_streams[stream_id] = {
            'connection_id': connection_id,
            'stream_type': stream_type,
            'metadata': metadata or {},
            'started_at': datetime.utcnow(),
            'chunk_count': 0
        }
        
        # Send stream start message
        start_message = {
            'type': 'stream_start',
            'stream_id': stream_id,
            'stream_type': stream_type,
            'metadata': metadata
        }
        
        await self.ws_manager.send_to_connection(connection_id, start_message)
        
        self.logger.info(f"Started stream {stream_id} for connection {connection_id}")
        return True
    
    async def send_stream_chunk(self, stream_id: str, chunk_data: Any, chunk_type: str = "data") -> bool:
        """Send a chunk of data to the stream"""
        if stream_id not in self.active_streams:
            return False
        
        stream_info = self.active_streams[stream_id]
        connection_id = stream_info['connection_id']
        
        chunk_message = {
            'type': 'stream_chunk',
            'stream_id': stream_id,
            'chunk_type': chunk_type,
            'data': chunk_data,
            'chunk_index': stream_info['chunk_count']
        }
        
        success = await self.ws_manager.send_to_connection(connection_id, chunk_message)
        
        if success:
            stream_info['chunk_count'] += 1
        
        return success
    
    async def end_stream(self, stream_id: str, reason: str = "completed") -> bool:
        """End a stream"""
        if stream_id not in self.active_streams:
            return False
        
        stream_info = self.active_streams[stream_id]
        connection_id = stream_info['connection_id']
        
        end_message = {
            'type': 'stream_end',
            'stream_id': stream_id,
            'reason': reason,
            'total_chunks': stream_info['chunk_count'],
            'duration_seconds': (datetime.utcnow() - stream_info['started_at']).total_seconds()
        }
        
        await self.ws_manager.send_to_connection(connection_id, end_message)
        
        del self.active_streams[stream_id]
        
        self.logger.info(f"Ended stream {stream_id} with reason: {reason}")
        return True
    
    def get_active_streams(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active streams"""
        return self.active_streams.copy()
    
    async def cleanup_orphaned_streams(self):
        """Clean up streams for disconnected connections"""
        orphaned_streams = []
        
        for stream_id, stream_info in self.active_streams.items():
            connection_id = stream_info['connection_id']
            connection = self.ws_manager.get_connection(connection_id)
            
            if not connection or not connection.is_connected():
                orphaned_streams.append(stream_id)
        
        for stream_id in orphaned_streams:
            del self.active_streams[stream_id]
            self.logger.info(f"Cleaned up orphaned stream {stream_id}")


# Built-in message handlers
class DefaultMessageHandlers:
    """Default message handlers for common WebSocket operations"""
    
    def __init__(self, websocket_manager: WebSocketManager):
        self.ws_manager = websocket_manager
    
    async def handle_ping(self, connection_id: str, message: Dict[str, Any]):
        """Handle ping message"""
        pong_message = {
            'type': 'pong',
            'timestamp': datetime.utcnow().isoformat()
        }
        await self.ws_manager.send_to_connection(connection_id, pong_message)
    
    async def handle_join_room(self, connection_id: str, message: Dict[str, Any]):
        """Handle join room message"""
        room_name = message.get('room_name')
        if room_name:
            self.ws_manager.join_room(connection_id, room_name)
            
            response = {
                'type': 'room_joined',
                'room_name': room_name
            }
            await self.ws_manager.send_to_connection(connection_id, response)
    
    async def handle_leave_room(self, connection_id: str, message: Dict[str, Any]):
        """Handle leave room message"""
        room_name = message.get('room_name')
        if room_name:
            self.ws_manager.leave_room(connection_id, room_name)
            
            response = {
                'type': 'room_left',
                'room_name': room_name
            }
            await self.ws_manager.send_to_connection(connection_id, response)
    
    async def handle_broadcast(self, connection_id: str, message: Dict[str, Any]):
        """Handle broadcast message to room"""
        room_name = message.get('room_name')
        broadcast_data = message.get('data')
        
        if room_name and broadcast_data:
            broadcast_message = {
                'type': 'broadcast',
                'room_name': room_name,
                'data': broadcast_data,
                'from_connection': connection_id
            }
            
            await self.ws_manager.broadcast_to_room(room_name, broadcast_message, exclude_connection=connection_id)


def setup_websocket_manager() -> WebSocketManager:
    """Setup WebSocket manager with default handlers"""
    ws_manager = WebSocketManager()
    handlers = DefaultMessageHandlers(ws_manager)
    
    # Register default handlers
    ws_manager.register_message_handler('ping', handlers.handle_ping)
    ws_manager.register_message_handler('join_room', handlers.handle_join_room)
    ws_manager.register_message_handler('leave_room', handlers.handle_leave_room)
    ws_manager.register_message_handler('broadcast', handlers.handle_broadcast)
    
    return ws_manager


# Background task for connection maintenance
async def websocket_maintenance_task(ws_manager: WebSocketManager, interval_seconds: int = 60):
    """Background task for WebSocket connection maintenance"""
    while True:
        try:
            await ws_manager.ping_all_connections()
            await ws_manager.cleanup_stale_connections()
        except Exception as e:
            logging.error(f"Error in WebSocket maintenance task: {e}")
        
        await asyncio.sleep(interval_seconds)


# Utility functions
def generate_connection_id() -> str:
    """Generate a unique connection ID"""
    return f"ws_{uuid.uuid4().hex[:12]}"


def generate_stream_id() -> str:
    """Generate a unique stream ID"""
    return f"stream_{uuid.uuid4().hex[:12]}"