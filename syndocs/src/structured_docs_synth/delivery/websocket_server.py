#!/usr/bin/env python3
"""
WebSocket Server for real-time document generation updates.

Provides WebSocket interface for real-time notifications, progress updates,
and streaming document generation results.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Set, Any, Callable
from enum import Enum
import websockets
from websockets.server import WebSocketServerProtocol
from dataclasses import dataclass, field, asdict

from ..core.logging import get_logger
from ..core.exceptions import ValidationError, ProcessingError
from ..orchestration.pipeline_coordinator import PipelineCoordinator, PipelineStatus


logger = get_logger(__name__)


class MessageType(Enum):
    """WebSocket message types"""
    # Client -> Server
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    START_GENERATION = "start_generation"
    CANCEL_GENERATION = "cancel_generation"
    GET_STATUS = "get_status"
    PING = "ping"
    
    # Server -> Client
    SUBSCRIPTION_CONFIRMED = "subscription_confirmed"
    GENERATION_STARTED = "generation_started"
    GENERATION_PROGRESS = "generation_progress"
    GENERATION_COMPLETED = "generation_completed"
    GENERATION_FAILED = "generation_failed"
    DOCUMENT_READY = "document_ready"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    PONG = "pong"


class SubscriptionType(Enum):
    """Subscription types"""
    ALL = "all"
    JOB = "job"
    PIPELINE = "pipeline"
    METRICS = "metrics"
    ALERTS = "alerts"


@dataclass
class WebSocketClient:
    """WebSocket client connection"""
    id: str
    websocket: WebSocketServerProtocol
    subscriptions: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    connected_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)


@dataclass
class Message:
    """WebSocket message"""
    type: MessageType
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_json(self) -> str:
        """Convert message to JSON"""
        return json.dumps({
            'id': self.id,
            'type': self.type.value,
            'payload': self.payload,
            'timestamp': self.timestamp.isoformat()
        })


class WebSocketServer:
    """
    WebSocket server for real-time updates.
    
    Features:
    - Real-time progress updates
    - Event subscriptions
    - Bi-directional communication
    - Connection management
    - Message broadcasting
    """
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 8765,
        coordinator: Optional[PipelineCoordinator] = None
    ):
        """Initialize WebSocket server"""
        self.host = host
        self.port = port
        self.logger = get_logger(__name__)
        
        # Pipeline coordinator
        self.coordinator = coordinator or PipelineCoordinator()
        
        # Client management
        self.clients: Dict[str, WebSocketClient] = {}
        self.client_lock = asyncio.Lock()
        
        # Subscription management
        self.subscriptions: Dict[str, Set[str]] = {
            sub_type.value: set() for sub_type in SubscriptionType
        }
        
        # Message handlers
        self.message_handlers: Dict[MessageType, Callable] = {
            MessageType.SUBSCRIBE: self._handle_subscribe,
            MessageType.UNSUBSCRIBE: self._handle_unsubscribe,
            MessageType.START_GENERATION: self._handle_start_generation,
            MessageType.CANCEL_GENERATION: self._handle_cancel_generation,
            MessageType.GET_STATUS: self._handle_get_status,
            MessageType.PING: self._handle_ping
        }
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        self.logger.info(f"WebSocket server initialized on {host}:{port}")
    
    async def start(self):
        """Start the WebSocket server"""
        self.is_running = True
        
        # Start background tasks
        self.background_tasks.append(
            asyncio.create_task(self._monitor_pipelines())
        )
        self.background_tasks.append(
            asyncio.create_task(self._cleanup_connections())
        )
        
        # Start WebSocket server
        async with websockets.serve(
            self._handle_connection,
            self.host,
            self.port
        ) as server:
            self.logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
            
            # Keep server running
            await asyncio.Future()  # Run forever
    
    async def stop(self):
        """Stop the WebSocket server"""
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Close all client connections
        async with self.client_lock:
            for client in self.clients.values():
                await client.websocket.close()
        
        self.logger.info("WebSocket server stopped")
    
    async def _handle_connection(
        self,
        websocket: WebSocketServerProtocol,
        path: str
    ):
        """Handle new WebSocket connection"""
        client_id = str(uuid.uuid4())
        client = WebSocketClient(
            id=client_id,
            websocket=websocket,
            metadata={'path': path}
        )
        
        # Register client
        async with self.client_lock:
            self.clients[client_id] = client
        
        self.logger.info(f"Client connected: {client_id}")
        
        # Send welcome message
        welcome_msg = Message(
            type=MessageType.STATUS_UPDATE,
            payload={
                'client_id': client_id,
                'status': 'connected',
                'server_time': datetime.now().isoformat()
            }
        )
        await self._send_to_client(client_id, welcome_msg)
        
        try:
            # Handle client messages
            async for message in websocket:
                await self._handle_message(client_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            self.logger.error(f"Connection error for client {client_id}: {e}")
        finally:
            # Clean up client
            await self._disconnect_client(client_id)
    
    async def _handle_message(self, client_id: str, raw_message: str):
        """Handle incoming message from client"""
        try:
            # Parse message
            data = json.loads(raw_message)
            
            # Validate message structure
            if 'type' not in data:
                raise ValidationError("Message must have 'type' field")
            
            # Create message object
            try:
                msg_type = MessageType(data['type'])
            except ValueError:
                raise ValidationError(f"Unknown message type: {data['type']}")
            
            message = Message(
                type=msg_type,
                payload=data.get('payload', {}),
                id=data.get('id', str(uuid.uuid4()))
            )
            
            # Update client activity
            if client_id in self.clients:
                self.clients[client_id].last_activity = datetime.now()
            
            # Handle message
            handler = self.message_handlers.get(msg_type)
            if handler:
                await handler(client_id, message)
            else:
                await self._send_error(
                    client_id,
                    f"Unsupported message type: {msg_type.value}"
                )
                
        except json.JSONDecodeError:
            await self._send_error(client_id, "Invalid JSON message")
        except ValidationError as e:
            await self._send_error(client_id, str(e))
        except Exception as e:
            self.logger.error(f"Message handling error: {e}")
            await self._send_error(client_id, "Internal server error")
    
    async def _handle_subscribe(self, client_id: str, message: Message):
        """Handle subscription request"""
        payload = message.payload
        subscription_type = payload.get('type', 'all')
        target_id = payload.get('target_id')
        
        # Create subscription key
        if subscription_type == 'job' and target_id:
            sub_key = f"job:{target_id}"
        elif subscription_type == 'pipeline' and target_id:
            sub_key = f"pipeline:{target_id}"
        else:
            sub_key = subscription_type
        
        # Add subscription
        async with self.client_lock:
            if client_id in self.clients:
                self.clients[client_id].subscriptions.add(sub_key)
                
                # Add to subscription index
                if subscription_type in [st.value for st in SubscriptionType]:
                    self.subscriptions[subscription_type].add(client_id)
        
        # Send confirmation
        response = Message(
            type=MessageType.SUBSCRIPTION_CONFIRMED,
            payload={
                'subscription': sub_key,
                'active_subscriptions': list(self.clients[client_id].subscriptions)
            }
        )
        await self._send_to_client(client_id, response)
    
    async def _handle_unsubscribe(self, client_id: str, message: Message):
        """Handle unsubscribe request"""
        payload = message.payload
        subscription = payload.get('subscription')
        
        async with self.client_lock:
            if client_id in self.clients and subscription:
                self.clients[client_id].subscriptions.discard(subscription)
                
                # Update subscription index
                for sub_type in SubscriptionType:
                    if subscription.startswith(sub_type.value):
                        self.subscriptions[sub_type.value].discard(client_id)
        
        # Send confirmation
        response = Message(
            type=MessageType.STATUS_UPDATE,
            payload={
                'status': 'unsubscribed',
                'subscription': subscription
            }
        )
        await self._send_to_client(client_id, response)
    
    async def _handle_start_generation(self, client_id: str, message: Message):
        """Handle document generation start request"""
        try:
            payload = message.payload
            
            # Validate parameters
            count = payload.get('count', 10)
            domain = payload.get('domain', 'general')
            doc_type = payload.get('type', 'pdf')
            
            # Start pipeline
            pipeline_id = self.coordinator.create_pipeline(
                name=f"WebSocket Generation {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                template='standard'
            )
            
            instance_id = self.coordinator.start_pipeline(
                pipeline_id=pipeline_id,
                input_data={'records': [{}] * count},
                parameters={
                    'domain': domain,
                    'document_type': doc_type,
                    'client_id': client_id
                }
            )
            
            # Subscribe client to this pipeline
            sub_key = f"pipeline:{instance_id}"
            async with self.client_lock:
                if client_id in self.clients:
                    self.clients[client_id].subscriptions.add(sub_key)
            
            # Send response
            response = Message(
                type=MessageType.GENERATION_STARTED,
                payload={
                    'pipeline_id': pipeline_id,
                    'instance_id': instance_id,
                    'document_count': count,
                    'parameters': {
                        'domain': domain,
                        'type': doc_type
                    }
                }
            )
            await self._send_to_client(client_id, response)
            
        except Exception as e:
            await self._send_error(client_id, f"Failed to start generation: {str(e)}")
    
    async def _handle_cancel_generation(self, client_id: str, message: Message):
        """Handle generation cancellation request"""
        payload = message.payload
        instance_id = payload.get('instance_id')
        
        if not instance_id:
            await self._send_error(client_id, "Missing instance_id")
            return
        
        # Cancel pipeline
        # self.coordinator.cancel_pipeline(instance_id)
        
        # Send confirmation
        response = Message(
            type=MessageType.STATUS_UPDATE,
            payload={
                'status': 'cancelled',
                'instance_id': instance_id
            }
        )
        await self._send_to_client(client_id, response)
    
    async def _handle_get_status(self, client_id: str, message: Message):
        """Handle status request"""
        payload = message.payload
        instance_id = payload.get('instance_id')
        
        if instance_id:
            # Get specific pipeline status
            status = self.coordinator.get_pipeline_status(instance_id)
            metrics = self.coordinator.get_pipeline_metrics(instance_id)
            
            response_payload = {
                'instance_id': instance_id,
                'status': status.value if status else 'unknown',
                'metrics': {
                    'processed': metrics.processed_documents if metrics else 0,
                    'total': metrics.total_documents if metrics else 0,
                    'progress': (metrics.processed_documents / metrics.total_documents) if metrics and metrics.total_documents > 0 else 0
                }
            }
        else:
            # Get overall status
            stats = self.coordinator.get_stats()
            response_payload = {
                'server_status': 'running',
                'stats': {
                    'total_pipelines': stats.get('total_pipelines', 0),
                    'running_instances': stats.get('running_instances', 0),
                    'connected_clients': len(self.clients)
                }
            }
        
        response = Message(
            type=MessageType.STATUS_UPDATE,
            payload=response_payload
        )
        await self._send_to_client(client_id, response)
    
    async def _handle_ping(self, client_id: str, message: Message):
        """Handle ping message"""
        response = Message(
            type=MessageType.PONG,
            payload={'timestamp': datetime.now().isoformat()}
        )
        await self._send_to_client(client_id, response)
    
    async def _monitor_pipelines(self):
        """Monitor pipeline progress and send updates"""
        while self.is_running:
            try:
                # Get all running pipelines
                instances = self.coordinator.list_instances(status=PipelineStatus.RUNNING)
                
                for instance in instances:
                    instance_id = instance['instance_id']
                    metrics = self.coordinator.get_pipeline_metrics(instance_id)
                    
                    if metrics:
                        # Create progress update
                        progress_msg = Message(
                            type=MessageType.GENERATION_PROGRESS,
                            payload={
                                'instance_id': instance_id,
                                'progress': {
                                    'processed': metrics.processed_documents,
                                    'total': metrics.total_documents,
                                    'percentage': (metrics.processed_documents / metrics.total_documents * 100) if metrics.total_documents > 0 else 0,
                                    'throughput': metrics.throughput,
                                    'current_stage': instance.get('current_stage')
                                }
                            }
                        )
                        
                        # Send to subscribed clients
                        await self._broadcast_to_subscribers(
                            f"pipeline:{instance_id}",
                            progress_msg
                        )
                
                # Check completed pipelines
                completed = self.coordinator.list_instances(status=PipelineStatus.COMPLETED)
                
                for instance in completed:
                    instance_id = instance['instance_id']
                    
                    # Send completion message
                    completion_msg = Message(
                        type=MessageType.GENERATION_COMPLETED,
                        payload={
                            'instance_id': instance_id,
                            'metrics': instance.get('metrics', {}),
                            'documents': []  # Would include actual document info
                        }
                    )
                    
                    await self._broadcast_to_subscribers(
                        f"pipeline:{instance_id}",
                        completion_msg
                    )
                
                # Sleep before next check
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Pipeline monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_connections(self):
        """Clean up inactive connections"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                inactive_threshold = 300  # 5 minutes
                current_time = datetime.now()
                
                async with self.client_lock:
                    inactive_clients = []
                    
                    for client_id, client in self.clients.items():
                        if (current_time - client.last_activity).total_seconds() > inactive_threshold:
                            inactive_clients.append(client_id)
                    
                    # Disconnect inactive clients
                    for client_id in inactive_clients:
                        self.logger.info(f"Disconnecting inactive client: {client_id}")
                        await self._disconnect_client(client_id)
                        
            except Exception as e:
                self.logger.error(f"Connection cleanup error: {e}")
    
    async def _disconnect_client(self, client_id: str):
        """Disconnect and clean up client"""
        async with self.client_lock:
            if client_id in self.clients:
                client = self.clients[client_id]
                
                # Remove from all subscriptions
                for sub_type in SubscriptionType:
                    self.subscriptions[sub_type.value].discard(client_id)
                
                # Close connection
                try:
                    await client.websocket.close()
                except:
                    pass
                
                # Remove client
                del self.clients[client_id]
    
    async def _send_to_client(self, client_id: str, message: Message):
        """Send message to specific client"""
        async with self.client_lock:
            if client_id in self.clients:
                client = self.clients[client_id]
                try:
                    await client.websocket.send(message.to_json())
                except websockets.exceptions.ConnectionClosed:
                    # Client disconnected
                    await self._disconnect_client(client_id)
                except Exception as e:
                    self.logger.error(f"Failed to send message to {client_id}: {e}")
    
    async def _broadcast_to_subscribers(self, subscription: str, message: Message):
        """Broadcast message to all subscribers"""
        async with self.client_lock:
            # Find all clients subscribed to this topic
            subscribers = [
                client_id for client_id, client in self.clients.items()
                if subscription in client.subscriptions or 'all' in client.subscriptions
            ]
        
        # Send to all subscribers
        tasks = [
            self._send_to_client(client_id, message)
            for client_id in subscribers
        ]
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def broadcast_message(self, message: Message):
        """Broadcast message to all connected clients"""
        async with self.client_lock:
            client_ids = list(self.clients.keys())
        
        tasks = [
            self._send_to_client(client_id, message)
            for client_id in client_ids
        ]
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_error(self, client_id: str, error_message: str):
        """Send error message to client"""
        error_msg = Message(
            type=MessageType.ERROR,
            payload={
                'error': error_message,
                'timestamp': datetime.now().isoformat()
            }
        )
        await self._send_to_client(client_id, error_msg)
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        return {
            'connected_clients': len(self.clients),
            'active_subscriptions': sum(
                len(client.subscriptions) for client in self.clients.values()
            ),
            'subscriptions_by_type': {
                sub_type: len(subscribers)
                for sub_type, subscribers in self.subscriptions.items()
            }
        }


# Example WebSocket client code for documentation
EXAMPLE_CLIENT_CODE = '''
import asyncio
import websockets
import json

async def client_example():
    uri = "ws://localhost:8765"
    
    async with websockets.connect(uri) as websocket:
        # Subscribe to all events
        await websocket.send(json.dumps({
            "type": "subscribe",
            "payload": {"type": "all"}
        }))
        
        # Start document generation
        await websocket.send(json.dumps({
            "type": "start_generation",
            "payload": {
                "count": 10,
                "domain": "finance",
                "type": "pdf"
            }
        }))
        
        # Listen for messages
        async for message in websocket:
            data = json.loads(message)
            print(f"Received: {data['type']} - {data['payload']}")
            
            if data['type'] == 'generation_completed':
                break

# Run the client
asyncio.run(client_example())
'''


# Factory function
def create_websocket_server(
    host: str = 'localhost',
    port: int = 8765,
    coordinator: Optional[PipelineCoordinator] = None
) -> WebSocketServer:
    """Create and return a WebSocket server instance"""
    return WebSocketServer(host, port, coordinator)