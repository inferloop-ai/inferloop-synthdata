#!/usr/bin/env python3
"""
WebSocket API for real-time communication in structured documents synthetic data system.

Provides WebSocket endpoints for real-time updates, progress tracking,
and interactive document generation workflows.

Features:
- Real-time generation progress updates
- Live document validation feedback
- Interactive quality metrics streaming
- Multi-room support for collaborative workflows
- Authentication and authorization
- Rate limiting and connection management
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import websockets
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed, WebSocketException

from ...core import get_logger, get_config
from ...core.exceptions import ValidationError, ProcessingError, AuthenticationError
from ..storage import DatabaseStorage, CacheManager
from ...privacy import create_privacy_engine
from ...quality import create_quality_engine
from ...generation import create_layout_engine


logger = get_logger(__name__)
config = get_config()


class ConnectionManager:
    """Manages WebSocket connections and rooms"""
    
    def __init__(self):
        self.connections: Dict[str, WebSocketServerProtocol] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self.rooms: Dict[str, Set[str]] = {}  # room_id -> connection_ids
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
    async def add_connection(self, connection_id: str, websocket: WebSocketServerProtocol, 
                           user_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a new WebSocket connection"""
        self.connections[connection_id] = websocket
        
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
        
        self.connection_metadata[connection_id] = {
            'user_id': user_id,
            'connected_at': datetime.utcnow(),
            'last_activity': datetime.utcnow(),
            **(metadata or {})
        }
        
        logger.info(f"Connection {connection_id} added for user {user_id}")
    
    async def remove_connection(self, connection_id: str):
        """Remove a WebSocket connection"""
        if connection_id not in self.connections:
            return
            
        metadata = self.connection_metadata.get(connection_id, {})
        user_id = metadata.get('user_id')
        
        # Remove from user connections
        if user_id and user_id in self.user_connections:
            self.user_connections[user_id].discard(connection_id)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        # Remove from all rooms
        for room_id, connections in self.rooms.items():
            connections.discard(connection_id)
        
        # Clean up empty rooms
        self.rooms = {k: v for k, v in self.rooms.items() if v}
        
        # Remove connection data
        del self.connections[connection_id]
        if connection_id in self.connection_metadata:
            del self.connection_metadata[connection_id]
            
        logger.info(f"Connection {connection_id} removed")
    
    async def join_room(self, connection_id: str, room_id: str):
        """Add connection to a room"""
        if room_id not in self.rooms:
            self.rooms[room_id] = set()
        self.rooms[room_id].add(connection_id)
        
        logger.info(f"Connection {connection_id} joined room {room_id}")
    
    async def leave_room(self, connection_id: str, room_id: str):
        """Remove connection from a room"""
        if room_id in self.rooms:
            self.rooms[room_id].discard(connection_id)
            if not self.rooms[room_id]:
                del self.rooms[room_id]
                
        logger.info(f"Connection {connection_id} left room {room_id}")
    
    async def send_to_connection(self, connection_id: str, message: Dict[str, Any]):
        """Send message to a specific connection"""
        if connection_id not in self.connections:
            return False
            
        try:
            websocket = self.connections[connection_id]
            await websocket.send(json.dumps(message))
            
            # Update last activity
            if connection_id in self.connection_metadata:
                self.connection_metadata[connection_id]['last_activity'] = datetime.utcnow()
                
            return True
        except (ConnectionClosed, WebSocketException):
            await self.remove_connection(connection_id)
            return False
    
    async def send_to_user(self, user_id: str, message: Dict[str, Any]):
        """Send message to all connections of a user"""
        if user_id not in self.user_connections:
            return 0
            
        sent_count = 0
        for connection_id in list(self.user_connections[user_id]):
            if await self.send_to_connection(connection_id, message):
                sent_count += 1
                
        return sent_count
    
    async def send_to_room(self, room_id: str, message: Dict[str, Any], exclude: Optional[str] = None):
        """Send message to all connections in a room"""
        if room_id not in self.rooms:
            return 0
            
        sent_count = 0
        for connection_id in list(self.rooms[room_id]):
            if connection_id != exclude:
                if await self.send_to_connection(connection_id, message):
                    sent_count += 1
                    
        return sent_count
    
    async def broadcast(self, message: Dict[str, Any], exclude: Optional[str] = None):
        """Broadcast message to all connections"""
        sent_count = 0
        for connection_id in list(self.connections.keys()):
            if connection_id != exclude:
                if await self.send_to_connection(connection_id, message):
                    sent_count += 1
                    
        return sent_count
    
    def get_connection_count(self) -> int:
        """Get total number of active connections"""
        return len(self.connections)
    
    def get_user_connections(self, user_id: str) -> List[str]:
        """Get all connection IDs for a user"""
        return list(self.user_connections.get(user_id, set()))
    
    def get_room_connections(self, room_id: str) -> List[str]:
        """Get all connection IDs in a room"""
        return list(self.rooms.get(room_id, set()))


class WebSocketAPI:
    """Main WebSocket API class"""
    
    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.storage = DatabaseStorage()
        self.cache = CacheManager()
        self.privacy_engine = create_privacy_engine()
        self.quality_engine = create_quality_engine()
        self.layout_engine = create_layout_engine()
        self.rate_limiter = {}  # connection_id -> {last_request: timestamp, count: int}
        
    async def authenticate_connection(self, websocket: WebSocketServerProtocol, 
                                    auth_token: str) -> Optional[Dict[str, Any]]:
        """Authenticate WebSocket connection"""
        try:
            # Validate auth token (implement your auth logic)
            user_data = await self._validate_auth_token(auth_token)
            return user_data
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return None
    
    async def _validate_auth_token(self, token: str) -> Dict[str, Any]:
        """Validate authentication token"""
        # Placeholder implementation - replace with your auth system
        # This could validate JWT tokens, API keys, etc.
        if not token or len(token) < 10:
            raise AuthenticationError("Invalid token")
            
        # Mock user data - replace with real validation
        return {
            'id': 'user_123',
            'username': 'test_user',
            'email': 'test@example.com',
            'role': 'user'
        }
    
    async def handle_rate_limiting(self, connection_id: str) -> bool:
        """Check and update rate limiting for connection"""
        current_time = time.time()
        
        if connection_id not in self.rate_limiter:
            self.rate_limiter[connection_id] = {
                'last_request': current_time,
                'count': 1,
                'window_start': current_time
            }
            return True
        
        rate_data = self.rate_limiter[connection_id]
        
        # Reset counter if window expired (1 minute window)
        if current_time - rate_data['window_start'] > 60:
            rate_data['count'] = 1
            rate_data['window_start'] = current_time
        else:
            rate_data['count'] += 1
        
        rate_data['last_request'] = current_time
        
        # Allow 100 requests per minute
        return rate_data['count'] <= 100
    
    async def handle_message(self, connection_id: str, message: Dict[str, Any]):
        """Handle incoming WebSocket message"""
        try:
            # Rate limiting check
            if not await self.handle_rate_limiting(connection_id):
                await self.send_error(connection_id, "rate_limit_exceeded", "Rate limit exceeded")
                return
            
            message_type = message.get('type')
            data = message.get('data', {})
            
            # Route message to appropriate handler
            if message_type == 'ping':
                await self.handle_ping(connection_id)
            elif message_type == 'join_room':
                await self.handle_join_room(connection_id, data)
            elif message_type == 'leave_room':
                await self.handle_leave_room(connection_id, data)
            elif message_type == 'start_generation':
                await self.handle_start_generation(connection_id, data)
            elif message_type == 'validate_document':
                await self.handle_validate_document(connection_id, data)
            elif message_type == 'get_quality_metrics':
                await self.handle_get_quality_metrics(connection_id, data)
            elif message_type == 'subscribe_progress':
                await self.handle_subscribe_progress(connection_id, data)
            else:
                await self.send_error(connection_id, "unknown_message_type", f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self.send_error(connection_id, "internal_error", "Internal server error")
    
    async def handle_ping(self, connection_id: str):
        """Handle ping message"""
        await self.send_message(connection_id, {
            'type': 'pong',
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def handle_join_room(self, connection_id: str, data: Dict[str, Any]):
        """Handle join room request"""
        room_id = data.get('room_id')
        if not room_id:
            await self.send_error(connection_id, "missing_room_id", "Room ID is required")
            return
            
        await self.connection_manager.join_room(connection_id, room_id)
        await self.send_message(connection_id, {
            'type': 'room_joined',
            'data': {'room_id': room_id}
        })
    
    async def handle_leave_room(self, connection_id: str, data: Dict[str, Any]):
        """Handle leave room request"""
        room_id = data.get('room_id')
        if not room_id:
            await self.send_error(connection_id, "missing_room_id", "Room ID is required")
            return
            
        await self.connection_manager.leave_room(connection_id, room_id)
        await self.send_message(connection_id, {
            'type': 'room_left',
            'data': {'room_id': room_id}
        })
    
    async def handle_start_generation(self, connection_id: str, data: Dict[str, Any]):
        """Handle document generation request"""
        try:
            config = data.get('config', {})
            request_id = str(uuid4())
            
            # Get user from connection metadata
            metadata = self.connection_manager.connection_metadata.get(connection_id, {})
            user_id = metadata.get('user_id')
            
            if not user_id:
                await self.send_error(connection_id, "authentication_required", "User not authenticated")
                return
            
            # Start generation process
            await self.storage.store_generation_request({
                'id': request_id,
                'user_id': user_id,
                'config': config,
                'status': 'pending',
                'progress': 0.0,
                'created_at': datetime.utcnow()
            })
            
            # Start background generation
            asyncio.create_task(self._generate_documents_background(request_id, config, connection_id))
            
            await self.send_message(connection_id, {
                'type': 'generation_started',
                'data': {'request_id': request_id}
            })
            
        except Exception as e:
            logger.error(f"Error starting generation: {e}")
            await self.send_error(connection_id, "generation_failed", str(e))
    
    async def handle_validate_document(self, connection_id: str, data: Dict[str, Any]):
        """Handle document validation request"""
        try:
            document_id = data.get('document_id')
            if not document_id:
                await self.send_error(connection_id, "missing_document_id", "Document ID is required")
                return
            
            # Perform validation
            result = await self.quality_engine.validate_document(
                document_id,
                rules=data.get('validation_rules', []),
                privacy_check=data.get('privacy_check', True)
            )
            
            await self.send_message(connection_id, {
                'type': 'validation_result',
                'data': {
                    'document_id': document_id,
                    'result': result,
                    'timestamp': datetime.utcnow().isoformat()
                }
            })
            
        except Exception as e:
            logger.error(f"Error validating document: {e}")
            await self.send_error(connection_id, "validation_failed", str(e))
    
    async def handle_get_quality_metrics(self, connection_id: str, data: Dict[str, Any]):
        """Handle quality metrics request"""
        try:
            document_id = data.get('document_id')
            if not document_id:
                await self.send_error(connection_id, "missing_document_id", "Document ID is required")
                return
            
            # Get quality metrics
            metrics = await self.quality_engine.get_metrics(document_id)
            
            await self.send_message(connection_id, {
                'type': 'quality_metrics',
                'data': {
                    'document_id': document_id,
                    'metrics': metrics,
                    'timestamp': datetime.utcnow().isoformat()
                }
            })
            
        except Exception as e:
            logger.error(f"Error getting quality metrics: {e}")
            await self.send_error(connection_id, "metrics_failed", str(e))
    
    async def handle_subscribe_progress(self, connection_id: str, data: Dict[str, Any]):
        """Handle progress subscription request"""
        request_id = data.get('request_id')
        if not request_id:
            await self.send_error(connection_id, "missing_request_id", "Request ID is required")
            return
        
        # Join room for this request to receive progress updates
        await self.connection_manager.join_room(connection_id, f"progress_{request_id}")
        
        await self.send_message(connection_id, {
            'type': 'progress_subscribed',
            'data': {'request_id': request_id}
        })
    
    async def _generate_documents_background(self, request_id: str, config: Dict[str, Any], connection_id: str):
        """Background document generation with real-time updates"""
        try:
            # Send initial progress
            await self.send_progress_update(request_id, "initializing", 0.1, "Initializing generation...")
            await asyncio.sleep(0.5)  # Small delay for demo purposes
            
            count = config.get('count', 10)
            domain = config.get('domain', 'general')
            format_type = config.get('format', 'docx')
            
            documents = []
            for i in range(count):
                # Generate document
                doc = await self.layout_engine.generate_document(
                    domain=domain,
                    format=format_type,
                    privacy_level=config.get('privacy_level', 'medium')
                )
                
                # Store document
                doc_id = await self.storage.store_document(doc)
                documents.append(doc_id)
                
                # Send progress update
                progress = 0.1 + (0.8 * (i + 1) / count)
                await self.send_progress_update(
                    request_id, "generating", progress, f"Generated {i+1}/{count} documents"
                )
                
                await asyncio.sleep(0.1)  # Small delay between documents
            
            # Final validation
            await self.send_progress_update(request_id, "validating", 0.95, "Validating generated documents...")
            await asyncio.sleep(0.5)
            
            # Complete generation
            await self.storage.update_generation_request(request_id, {
                'status': 'completed',
                'progress': 1.0,
                'completed_at': datetime.utcnow(),
                'result': {'document_ids': documents}
            })
            
            await self.send_progress_update(request_id, "completed", 1.0, "Generation completed successfully")
            
        except Exception as e:
            logger.error(f"Error in background generation: {e}")
            await self.storage.update_generation_request(request_id, {
                'status': 'failed',
                'error_message': str(e),
                'completed_at': datetime.utcnow()
            })
            await self.send_progress_update(request_id, "failed", 0.0, f"Generation failed: {str(e)}")
    
    async def send_progress_update(self, request_id: str, stage: str, progress: float, message: str):
        """Send progress update to all subscribers"""
        progress_data = {
            'type': 'progress_update',
            'data': {
                'request_id': request_id,
                'stage': stage,
                'progress': progress,
                'message': message,
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        
        # Send to progress room
        await self.connection_manager.send_to_room(f"progress_{request_id}", progress_data)
    
    async def send_message(self, connection_id: str, message: Dict[str, Any]):
        """Send message to a connection"""
        return await self.connection_manager.send_to_connection(connection_id, message)
    
    async def send_error(self, connection_id: str, error_code: str, error_message: str):
        """Send error message to a connection"""
        await self.send_message(connection_id, {
            'type': 'error',
            'data': {
                'error_code': error_code,
                'error_message': error_message,
                'timestamp': datetime.utcnow().isoformat()
            }
        })
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection"""
        connection_id = str(uuid4())
        user_data = None
        
        try:
            # Wait for authentication message
            auth_message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
            auth_data = json.loads(auth_message)
            
            if auth_data.get('type') != 'authenticate':
                await websocket.send(json.dumps({
                    'type': 'error',
                    'data': {
                        'error_code': 'authentication_required',
                        'error_message': 'Authentication required as first message'
                    }
                }))
                return
            
            # Authenticate connection
            auth_token = auth_data.get('data', {}).get('token')
            user_data = await self.authenticate_connection(websocket, auth_token)
            
            if not user_data:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'data': {
                        'error_code': 'authentication_failed',
                        'error_message': 'Authentication failed'
                    }
                }))
                return
            
            # Add connection to manager
            await self.connection_manager.add_connection(connection_id, websocket, user_data['id'])
            
            # Send authentication success
            await websocket.send(json.dumps({
                'type': 'authenticated',
                'data': {
                    'connection_id': connection_id,
                    'user': user_data
                }
            }))
            
            logger.info(f"WebSocket connection established: {connection_id} for user {user_data['id']}")
            
            # Handle messages
            async for message_raw in websocket:
                try:
                    message = json.loads(message_raw)
                    await self.handle_message(connection_id, message)
                except json.JSONDecodeError:
                    await self.send_error(connection_id, "invalid_json", "Invalid JSON message")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    await self.send_error(connection_id, "message_error", str(e))
        
        except asyncio.TimeoutError:
            logger.warning(f"Authentication timeout for connection {connection_id}")
        except ConnectionClosed:
            logger.info(f"Connection {connection_id} closed by client")
        except Exception as e:
            logger.error(f"Error in WebSocket connection {connection_id}: {e}")
        finally:
            await self.connection_manager.remove_connection(connection_id)
            logger.info(f"WebSocket connection {connection_id} cleaned up")


# Global API instance
api_instance = WebSocketAPI()


# WebSocket server factory
def create_websocket_server(host: str = "localhost", port: int = 8765):
    """Create WebSocket server"""
    return websockets.serve(
        api_instance.handle_connection,
        host,
        port,
        ping_interval=30,
        ping_timeout=10,
        close_timeout=10,
        max_size=1024 * 1024,  # 1MB max message size
        compression=None
    )


# Factory function
def create_websocket_api() -> WebSocketAPI:
    """Create WebSocket API instance"""
    return WebSocketAPI()


__all__ = [
    'WebSocketAPI',
    'ConnectionManager',
    'create_websocket_server',
    'create_websocket_api'
]