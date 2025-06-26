"""
Global Inference Network for TextNLP
Distributed inference system for coordinating multiple edge deployments and cloud resources
"""

import asyncio
import json
import logging
import time
import uuid
import hashlib
import aiohttp
import websockets
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import yaml
import redis
import sqlite3
from contextlib import asynccontextmanager
import ssl
import jwt
from datetime import datetime, timedelta

try:
    import kubernetes
    from kubernetes import client, config
    HAS_KUBERNETES = True
except ImportError:
    HAS_KUBERNETES = False

try:
    import consul
    HAS_CONSUL = True
except ImportError:
    HAS_CONSUL = False

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the inference network"""
    EDGE_DEVICE = "edge_device"
    CLOUD_INSTANCE = "cloud_instance"
    HYBRID_NODE = "hybrid_node"
    COORDINATOR = "coordinator"
    LOAD_BALANCER = "load_balancer"


class NodeStatus(Enum):
    """Status of nodes in the network"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class RoutingStrategy(Enum):
    """Routing strategies for inference requests"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    GEOGRAPHIC = "geographic"
    LATENCY_BASED = "latency_based"
    COST_OPTIMIZED = "cost_optimized"
    INTELLIGENT = "intelligent"


class RequestPriority(Enum):
    """Priority levels for inference requests"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class NodeCapabilities:
    """Capabilities and specifications of a network node"""
    node_id: str
    node_type: NodeType
    cpu_cores: int
    memory_gb: float
    gpu_available: bool = False
    gpu_memory_gb: float = 0.0
    storage_gb: float = 0.0
    max_concurrent_requests: int = 10
    supported_models: List[str] = field(default_factory=list)
    specialized_hardware: List[str] = field(default_factory=list)  # TPU, NPU, etc.
    power_budget_watts: float = 100.0
    cost_per_hour: float = 0.0
    geographic_region: str = "unknown"
    network_bandwidth_mbps: float = 1000.0


@dataclass
class NodeMetrics:
    """Real-time metrics for a network node"""
    node_id: str
    timestamp: float
    status: NodeStatus
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float] = None
    active_requests: int = 0
    requests_per_second: float = 0.0
    average_latency_ms: float = 0.0
    error_rate: float = 0.0
    temperature_celsius: Optional[float] = None
    power_consumption_watts: Optional[float] = None
    queue_length: int = 0
    model_load_factor: float = 1.0  # Relative model loading efficiency


@dataclass
class InferenceRequest:
    """Inference request in the global network"""
    request_id: str
    user_id: str
    prompt: str
    model_name: str
    priority: RequestPriority = RequestPriority.NORMAL
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    streaming: bool = False
    timeout_seconds: float = 30.0
    preferred_region: Optional[str] = None
    cost_budget: Optional[float] = None
    latency_requirement_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class InferenceResponse:
    """Response from inference in the global network"""
    request_id: str
    generated_text: str
    success: bool
    node_id: str
    processing_time_ms: float
    queue_time_ms: float
    total_latency_ms: float
    tokens_generated: int
    cost: float = 0.0
    model_used: str = ""
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NetworkConfig:
    """Configuration for the global inference network"""
    # Network topology
    coordinator_nodes: List[str] = field(default_factory=list)
    load_balancer_nodes: List[str] = field(default_factory=list)
    default_routing_strategy: RoutingStrategy = RoutingStrategy.INTELLIGENT
    
    # Service discovery
    use_service_discovery: bool = True
    consul_host: str = "localhost"
    consul_port: int = 8500
    kubernetes_namespace: str = "default"
    
    # Load balancing
    health_check_interval: int = 30
    max_retries: int = 3
    circuit_breaker_threshold: float = 0.5
    
    # Security
    enable_authentication: bool = True
    jwt_secret: str = "change-me-in-production"
    enable_encryption: bool = True
    cert_file: Optional[str] = None
    key_file: Optional[str] = None
    
    # Performance
    connection_pool_size: int = 100
    request_timeout: float = 60.0
    max_concurrent_requests: int = 1000
    
    # Monitoring
    metrics_retention_hours: int = 168  # 1 week
    enable_detailed_logging: bool = False
    
    # Auto-scaling
    enable_auto_scaling: bool = True
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    min_nodes: int = 2
    max_nodes: int = 20


class NetworkNode:
    """Represents a node in the global inference network"""
    
    def __init__(self, capabilities: NodeCapabilities, config: NetworkConfig):
        self.capabilities = capabilities
        self.config = config
        self.metrics = NodeMetrics(
            node_id=capabilities.node_id,
            timestamp=time.time(),
            status=NodeStatus.OFFLINE,
            cpu_usage=0.0,
            memory_usage=0.0
        )
        self.last_heartbeat = 0.0
        self.active_connections = set()
        self.request_queue = asyncio.Queue()
        self.is_running = False
        
        # Performance tracking
        self.request_history = []
        self.latency_history = []
        
    async def start(self):
        """Start the network node"""
        self.is_running = True
        self.metrics.status = NodeStatus.HEALTHY
        
        # Start background tasks
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._metrics_collection_loop())
        asyncio.create_task(self._request_processing_loop())
        
        logger.info(f"Node {self.capabilities.node_id} started")
    
    async def stop(self):
        """Stop the network node"""
        self.is_running = False
        self.metrics.status = NodeStatus.OFFLINE
        
        # Close active connections
        for connection in self.active_connections:
            if hasattr(connection, 'close'):
                await connection.close()
        
        logger.info(f"Node {self.capabilities.node_id} stopped")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat signals"""
        while self.is_running:
            try:
                self.last_heartbeat = time.time()
                await self._send_heartbeat()
                await asyncio.sleep(self.config.health_check_interval)
            except Exception as e:
                logger.error(f"Heartbeat error for node {self.capabilities.node_id}: {e}")
                await asyncio.sleep(5)
    
    async def _send_heartbeat(self):
        """Send heartbeat to coordinator"""
        # Implementation would send heartbeat to coordinator nodes
        pass
    
    async def _metrics_collection_loop(self):
        """Collect and update node metrics"""
        while self.is_running:
            try:
                await self._update_metrics()
                await asyncio.sleep(10)  # Update metrics every 10 seconds
            except Exception as e:
                logger.error(f"Metrics collection error for node {self.capabilities.node_id}: {e}")
                await asyncio.sleep(30)
    
    async def _update_metrics(self):
        """Update node metrics"""
        import psutil
        
        # Update basic system metrics
        self.metrics.timestamp = time.time()
        self.metrics.cpu_usage = psutil.cpu_percent(interval=1)
        self.metrics.memory_usage = psutil.virtual_memory().percent
        self.metrics.active_requests = self.request_queue.qsize()
        
        # Update GPU metrics if available
        if self.capabilities.gpu_available:
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.metrics.gpu_usage = gpus[0].load * 100
            except:
                pass
        
        # Calculate derived metrics
        if self.request_history:
            recent_requests = [r for r in self.request_history if time.time() - r < 60]
            self.metrics.requests_per_second = len(recent_requests) / 60.0
        
        if self.latency_history:
            recent_latencies = [l for l in self.latency_history if time.time() - l[0] < 300]
            if recent_latencies:
                self.metrics.average_latency_ms = np.mean([l[1] for l in recent_latencies])
        
        # Determine health status
        self._update_health_status()
    
    def _update_health_status(self):
        """Update node health status based on metrics"""
        if (self.metrics.cpu_usage > 95 or 
            self.metrics.memory_usage > 95 or
            self.metrics.error_rate > 0.5):
            self.metrics.status = NodeStatus.UNHEALTHY
        elif (self.metrics.cpu_usage > 80 or 
              self.metrics.memory_usage > 80 or
              self.metrics.error_rate > 0.1):
            self.metrics.status = NodeStatus.DEGRADED
        else:
            self.metrics.status = NodeStatus.HEALTHY
    
    async def _request_processing_loop(self):
        """Process incoming inference requests"""
        while self.is_running:
            try:
                # Get request from queue with timeout
                try:
                    request = await asyncio.wait_for(
                        self.request_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process the request
                response = await self._process_request(request)
                
                # Record metrics
                self.request_history.append(time.time())
                self.latency_history.append((time.time(), response.total_latency_ms))
                
                # Cleanup old history
                cutoff = time.time() - 3600  # Keep 1 hour of history
                self.request_history = [r for r in self.request_history if r > cutoff]
                self.latency_history = [l for l in self.latency_history if l[0] > cutoff]
                
            except Exception as e:
                logger.error(f"Request processing error for node {self.capabilities.node_id}: {e}")
    
    async def _process_request(self, request: InferenceRequest) -> InferenceResponse:
        """Process an inference request"""
        start_time = time.time()
        
        try:
            # Simulate model inference (replace with actual inference)
            await asyncio.sleep(0.1)  # Simulate processing time
            
            generated_text = f"Generated response for: {request.prompt[:50]}..."
            
            processing_time = (time.time() - start_time) * 1000
            
            return InferenceResponse(
                request_id=request.request_id,
                generated_text=generated_text,
                success=True,
                node_id=self.capabilities.node_id,
                processing_time_ms=processing_time,
                queue_time_ms=0.0,  # Would track actual queue time
                total_latency_ms=processing_time,
                tokens_generated=len(generated_text.split()),
                model_used=request.model_name
            )
            
        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            return InferenceResponse(
                request_id=request.request_id,
                generated_text="",
                success=False,
                node_id=self.capabilities.node_id,
                processing_time_ms=(time.time() - start_time) * 1000,
                queue_time_ms=0.0,
                total_latency_ms=(time.time() - start_time) * 1000,
                tokens_generated=0,
                error_message=str(e)
            )
    
    def get_load_score(self) -> float:
        """Calculate load score for routing decisions"""
        # Normalize metrics to 0-1 scale
        cpu_factor = self.metrics.cpu_usage / 100.0
        memory_factor = self.metrics.memory_usage / 100.0
        queue_factor = min(self.metrics.queue_length / 10.0, 1.0)
        
        # Weighted average (CPU and memory are most important)
        load_score = (cpu_factor * 0.4 + memory_factor * 0.4 + queue_factor * 0.2)
        
        return min(load_score, 1.0)


class IntelligentRouter:
    """Intelligent routing system for the global network"""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.nodes: Dict[str, NetworkNode] = {}
        self.routing_history = []
        self.model_to_nodes: Dict[str, List[str]] = {}
        
    def register_node(self, node: NetworkNode):
        """Register a node with the router"""
        self.nodes[node.capabilities.node_id] = node
        
        # Update model mapping
        for model in node.capabilities.supported_models:
            if model not in self.model_to_nodes:
                self.model_to_nodes[model] = []
            if node.capabilities.node_id not in self.model_to_nodes[model]:
                self.model_to_nodes[model].append(node.capabilities.node_id)
        
        logger.info(f"Registered node {node.capabilities.node_id} with {len(node.capabilities.supported_models)} models")
    
    def unregister_node(self, node_id: str):
        """Unregister a node from the router"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            
            # Remove from model mapping
            for model in node.capabilities.supported_models:
                if model in self.model_to_nodes:
                    self.model_to_nodes[model] = [
                        nid for nid in self.model_to_nodes[model] if nid != node_id
                    ]
            
            del self.nodes[node_id]
            logger.info(f"Unregistered node {node_id}")
    
    async def route_request(self, request: InferenceRequest) -> Optional[str]:
        """Route request to the best available node"""
        
        # Filter nodes that support the requested model
        candidate_nodes = []
        if request.model_name in self.model_to_nodes:
            for node_id in self.model_to_nodes[request.model_name]:
                if node_id in self.nodes:
                    node = self.nodes[node_id]
                    if node.metrics.status in [NodeStatus.HEALTHY, NodeStatus.DEGRADED]:
                        candidate_nodes.append(node)
        
        if not candidate_nodes:
            logger.warning(f"No available nodes for model {request.model_name}")
            return None
        
        # Apply routing strategy
        selected_node = await self._apply_routing_strategy(
            request, candidate_nodes, self.config.default_routing_strategy
        )
        
        if selected_node:
            # Record routing decision
            self.routing_history.append({
                'timestamp': time.time(),
                'request_id': request.request_id,
                'selected_node': selected_node.capabilities.node_id,
                'strategy': self.config.default_routing_strategy.value,
                'candidate_count': len(candidate_nodes)
            })
            
            return selected_node.capabilities.node_id
        
        return None
    
    async def _apply_routing_strategy(self, request: InferenceRequest, 
                                    candidate_nodes: List[NetworkNode],
                                    strategy: RoutingStrategy) -> Optional[NetworkNode]:
        """Apply the specified routing strategy"""
        
        if not candidate_nodes:
            return None
        
        if strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_routing(candidate_nodes)
        
        elif strategy == RoutingStrategy.LEAST_LOADED:
            return self._least_loaded_routing(candidate_nodes)
        
        elif strategy == RoutingStrategy.LATENCY_BASED:
            return self._latency_based_routing(request, candidate_nodes)
        
        elif strategy == RoutingStrategy.COST_OPTIMIZED:
            return self._cost_optimized_routing(request, candidate_nodes)
        
        elif strategy == RoutingStrategy.GEOGRAPHIC:
            return self._geographic_routing(request, candidate_nodes)
        
        elif strategy == RoutingStrategy.INTELLIGENT:
            return await self._intelligent_routing(request, candidate_nodes)
        
        else:
            # Default to least loaded
            return self._least_loaded_routing(candidate_nodes)
    
    def _round_robin_routing(self, candidate_nodes: List[NetworkNode]) -> NetworkNode:
        """Simple round-robin routing"""
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
        
        selected = candidate_nodes[self._round_robin_index % len(candidate_nodes)]
        self._round_robin_index += 1
        
        return selected
    
    def _least_loaded_routing(self, candidate_nodes: List[NetworkNode]) -> NetworkNode:
        """Route to the least loaded node"""
        return min(candidate_nodes, key=lambda node: node.get_load_score())
    
    def _latency_based_routing(self, request: InferenceRequest, 
                             candidate_nodes: List[NetworkNode]) -> NetworkNode:
        """Route based on expected latency"""
        # Score nodes based on average latency
        def latency_score(node):
            base_latency = node.metrics.average_latency_ms
            load_factor = node.get_load_score()
            # Estimate latency increase due to load
            estimated_latency = base_latency * (1 + load_factor)
            return estimated_latency
        
        return min(candidate_nodes, key=latency_score)
    
    def _cost_optimized_routing(self, request: InferenceRequest,
                              candidate_nodes: List[NetworkNode]) -> NetworkNode:
        """Route to minimize cost"""
        # Filter by cost budget if specified
        if request.cost_budget:
            affordable_nodes = [
                node for node in candidate_nodes 
                if node.capabilities.cost_per_hour <= request.cost_budget
            ]
            if affordable_nodes:
                candidate_nodes = affordable_nodes
        
        return min(candidate_nodes, key=lambda node: node.capabilities.cost_per_hour)
    
    def _geographic_routing(self, request: InferenceRequest,
                          candidate_nodes: List[NetworkNode]) -> NetworkNode:
        """Route based on geographic preference"""
        if request.preferred_region:
            # Prefer nodes in the requested region
            regional_nodes = [
                node for node in candidate_nodes
                if node.capabilities.geographic_region == request.preferred_region
            ]
            if regional_nodes:
                return self._least_loaded_routing(regional_nodes)
        
        # Fall back to least loaded
        return self._least_loaded_routing(candidate_nodes)
    
    async def _intelligent_routing(self, request: InferenceRequest,
                                 candidate_nodes: List[NetworkNode]) -> NetworkNode:
        """Intelligent routing using multiple factors"""
        
        def score_node(node: NetworkNode) -> float:
            score = 0.0
            
            # Load factor (lower is better)
            load_score = node.get_load_score()
            score += (1 - load_score) * 0.3
            
            # Latency factor (lower is better)
            if node.metrics.average_latency_ms > 0:
                latency_score = 1 / (1 + node.metrics.average_latency_ms / 1000)
                score += latency_score * 0.3
            
            # Cost factor (lower is better, if budget specified)
            if request.cost_budget and node.capabilities.cost_per_hour > 0:
                cost_score = request.cost_budget / node.capabilities.cost_per_hour
                score += min(cost_score, 1.0) * 0.2
            else:
                score += 0.2  # Neutral score if no cost consideration
            
            # Geographic preference
            if (request.preferred_region and 
                node.capabilities.geographic_region == request.preferred_region):
                score += 0.1
            
            # Hardware capability match
            if request.model_name in node.capabilities.supported_models:
                score += 0.1
            
            return score
        
        # Score all nodes and return the best one
        scored_nodes = [(node, score_node(node)) for node in candidate_nodes]
        best_node, best_score = max(scored_nodes, key=lambda x: x[1])
        
        return best_node


class GlobalNetworkCoordinator:
    """Central coordinator for the global inference network"""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.router = IntelligentRouter(config)
        self.nodes: Dict[str, NetworkNode] = {}
        self.request_cache = {}
        self.response_cache = {}
        self.is_running = False
        
        # Service discovery
        self.service_discovery = None
        if config.use_service_discovery:
            self._setup_service_discovery()
        
        # Metrics storage
        self.metrics_db = None
        self._setup_metrics_storage()
        
        # WebSocket connections for real-time communication
        self.websocket_connections: Set[websockets.WebSocketServerProtocol] = set()
    
    def _setup_service_discovery(self):
        """Setup service discovery (Consul, Kubernetes, etc.)"""
        if HAS_CONSUL:
            try:
                self.service_discovery = consul.Consul(
                    host=self.config.consul_host,
                    port=self.config.consul_port
                )
                logger.info("Consul service discovery initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Consul: {e}")
        
        # Could also setup Kubernetes service discovery here
    
    def _setup_metrics_storage(self):
        """Setup metrics storage database"""
        try:
            self.metrics_db = sqlite3.connect(":memory:")  # In-memory for demo
            self.metrics_db.execute("""
                CREATE TABLE node_metrics (
                    timestamp REAL,
                    node_id TEXT,
                    cpu_usage REAL,
                    memory_usage REAL,
                    gpu_usage REAL,
                    active_requests INTEGER,
                    requests_per_second REAL,
                    average_latency_ms REAL,
                    error_rate REAL
                )
            """)
            self.metrics_db.execute("""
                CREATE TABLE request_logs (
                    timestamp REAL,
                    request_id TEXT,
                    user_id TEXT,
                    model_name TEXT,
                    node_id TEXT,
                    processing_time_ms REAL,
                    success BOOLEAN,
                    priority TEXT
                )
            """)
            logger.info("Metrics storage initialized")
        except Exception as e:
            logger.error(f"Failed to setup metrics storage: {e}")
    
    async def start(self):
        """Start the global network coordinator"""
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self._node_discovery_loop())
        asyncio.create_task(self._health_monitoring_loop())
        asyncio.create_task(self._metrics_collection_loop())
        asyncio.create_task(self._auto_scaling_loop())
        
        logger.info("Global network coordinator started")
    
    async def stop(self):
        """Stop the global network coordinator"""
        self.is_running = False
        
        # Stop all nodes
        for node in self.nodes.values():
            await node.stop()
        
        # Close database
        if self.metrics_db:
            self.metrics_db.close()
        
        logger.info("Global network coordinator stopped")
    
    async def register_node(self, capabilities: NodeCapabilities) -> bool:
        """Register a new node in the network"""
        try:
            node = NetworkNode(capabilities, self.config)
            self.nodes[capabilities.node_id] = node
            self.router.register_node(node)
            
            await node.start()
            
            # Register with service discovery
            if self.service_discovery:
                self._register_with_service_discovery(capabilities)
            
            logger.info(f"Node {capabilities.node_id} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register node {capabilities.node_id}: {e}")
            return False
    
    async def unregister_node(self, node_id: str) -> bool:
        """Unregister a node from the network"""
        try:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                await node.stop()
                
                self.router.unregister_node(node_id)
                del self.nodes[node_id]
                
                # Unregister from service discovery
                if self.service_discovery:
                    self._unregister_from_service_discovery(node_id)
                
                logger.info(f"Node {node_id} unregistered successfully")
                return True
            else:
                logger.warning(f"Node {node_id} not found for unregistration")
                return False
                
        except Exception as e:
            logger.error(f"Failed to unregister node {node_id}: {e}")
            return False
    
    async def process_request(self, request: InferenceRequest) -> InferenceResponse:
        """Process an inference request through the network"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            if cache_key in self.response_cache:
                cached_response = self.response_cache[cache_key]
                cached_response.request_id = request.request_id  # Update request ID
                return cached_response
            
            # Route request to appropriate node
            selected_node_id = await self.router.route_request(request)
            
            if not selected_node_id:
                return InferenceResponse(
                    request_id=request.request_id,
                    generated_text="",
                    success=False,
                    node_id="",
                    processing_time_ms=0,
                    queue_time_ms=0,
                    total_latency_ms=(time.time() - start_time) * 1000,
                    tokens_generated=0,
                    error_message="No available nodes for the requested model"
                )
            
            # Send request to selected node
            selected_node = self.nodes[selected_node_id]
            await selected_node.request_queue.put(request)
            
            # Wait for response (simplified - would use proper callback mechanism)
            response = await selected_node._process_request(request)
            
            # Update response with total latency
            response.total_latency_ms = (time.time() - start_time) * 1000
            
            # Cache successful responses
            if response.success:
                self.response_cache[cache_key] = response
            
            # Log request
            self._log_request(request, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            return InferenceResponse(
                request_id=request.request_id,
                generated_text="",
                success=False,
                node_id="",
                processing_time_ms=0,
                queue_time_ms=0,
                total_latency_ms=(time.time() - start_time) * 1000,
                tokens_generated=0,
                error_message=str(e)
            )
    
    def _generate_cache_key(self, request: InferenceRequest) -> str:
        """Generate cache key for request"""
        cache_data = {
            "prompt": request.prompt,
            "model_name": request.model_name,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _log_request(self, request: InferenceRequest, response: InferenceResponse):
        """Log request for analytics and monitoring"""
        if self.metrics_db:
            try:
                self.metrics_db.execute("""
                    INSERT INTO request_logs 
                    (timestamp, request_id, user_id, model_name, node_id, 
                     processing_time_ms, success, priority)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    time.time(),
                    request.request_id,
                    request.user_id,
                    request.model_name,
                    response.node_id,
                    response.processing_time_ms,
                    response.success,
                    request.priority.value
                ))
                self.metrics_db.commit()
            except Exception as e:
                logger.error(f"Failed to log request: {e}")
    
    async def _node_discovery_loop(self):
        """Discover and monitor nodes in the network"""
        while self.is_running:
            try:
                if self.service_discovery:
                    # Discover nodes from service registry
                    discovered_nodes = self._discover_nodes()
                    
                    # Register new nodes
                    for node_info in discovered_nodes:
                        if node_info['node_id'] not in self.nodes:
                            capabilities = self._parse_node_capabilities(node_info)
                            await self.register_node(capabilities)
                
                await asyncio.sleep(60)  # Discover every minute
                
            except Exception as e:
                logger.error(f"Node discovery error: {e}")
                await asyncio.sleep(300)  # Wait longer on error
    
    async def _health_monitoring_loop(self):
        """Monitor health of all nodes"""
        while self.is_running:
            try:
                unhealthy_nodes = []
                
                for node_id, node in self.nodes.items():
                    # Check if node is responsive
                    if time.time() - node.last_heartbeat > self.config.health_check_interval * 2:
                        node.metrics.status = NodeStatus.OFFLINE
                        unhealthy_nodes.append(node_id)
                
                # Remove unresponsive nodes
                for node_id in unhealthy_nodes:
                    await self.unregister_node(node_id)
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_collection_loop(self):
        """Collect and store metrics from all nodes"""
        while self.is_running:
            try:
                for node in self.nodes.values():
                    if self.metrics_db:
                        # Store node metrics
                        self.metrics_db.execute("""
                            INSERT INTO node_metrics 
                            (timestamp, node_id, cpu_usage, memory_usage, gpu_usage,
                             active_requests, requests_per_second, average_latency_ms, error_rate)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            node.metrics.timestamp,
                            node.metrics.node_id,
                            node.metrics.cpu_usage,
                            node.metrics.memory_usage,
                            node.metrics.gpu_usage,
                            node.metrics.active_requests,
                            node.metrics.requests_per_second,
                            node.metrics.average_latency_ms,
                            node.metrics.error_rate
                        ))
                
                if self.metrics_db:
                    self.metrics_db.commit()
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(60)
    
    async def _auto_scaling_loop(self):
        """Auto-scaling logic for the network"""
        if not self.config.enable_auto_scaling:
            return
        
        while self.is_running:
            try:
                # Calculate overall network load
                total_load = 0.0
                healthy_nodes = 0
                
                for node in self.nodes.values():
                    if node.metrics.status == NodeStatus.HEALTHY:
                        total_load += node.get_load_score()
                        healthy_nodes += 1
                
                if healthy_nodes > 0:
                    average_load = total_load / healthy_nodes
                    
                    # Scale up if load is high
                    if (average_load > self.config.scale_up_threshold and 
                        len(self.nodes) < self.config.max_nodes):
                        await self._scale_up()
                    
                    # Scale down if load is low
                    elif (average_load < self.config.scale_down_threshold and
                          len(self.nodes) > self.config.min_nodes):
                        await self._scale_down()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")
                await asyncio.sleep(600)
    
    async def _scale_up(self):
        """Scale up the network by adding new nodes"""
        logger.info("Triggering scale-up operation")
        # Implementation would depend on deployment platform
        # (Kubernetes, Docker Swarm, cloud auto-scaling groups, etc.)
    
    async def _scale_down(self):
        """Scale down the network by removing underutilized nodes"""
        logger.info("Triggering scale-down operation")
        # Find least loaded node and gracefully remove it
        least_loaded_node = min(
            self.nodes.values(),
            key=lambda node: node.get_load_score()
        )
        await self.unregister_node(least_loaded_node.capabilities.node_id)
    
    def _register_with_service_discovery(self, capabilities: NodeCapabilities):
        """Register node with service discovery"""
        if self.service_discovery:
            try:
                service_info = {
                    'ID': capabilities.node_id,
                    'Name': f'textnlp-node-{capabilities.node_type.value}',
                    'Tags': [
                        capabilities.node_type.value,
                        capabilities.geographic_region,
                        *capabilities.supported_models
                    ],
                    'Address': '0.0.0.0',  # Would be actual IP
                    'Port': 8080,
                    'Check': {
                        'HTTP': f'http://0.0.0.0:8080/health',
                        'Interval': '30s'
                    }
                }
                self.service_discovery.agent.service.register(**service_info)
            except Exception as e:
                logger.error(f"Service discovery registration failed: {e}")
    
    def _unregister_from_service_discovery(self, node_id: str):
        """Unregister node from service discovery"""
        if self.service_discovery:
            try:
                self.service_discovery.agent.service.deregister(node_id)
            except Exception as e:
                logger.error(f"Service discovery deregistration failed: {e}")
    
    def _discover_nodes(self) -> List[Dict]:
        """Discover nodes from service registry"""
        if not self.service_discovery:
            return []
        
        try:
            services = self.service_discovery.health.service(
                'textnlp-node', 
                passing=True
            )[1]
            
            discovered_nodes = []
            for service in services:
                node_info = {
                    'node_id': service['Service']['ID'],
                    'address': service['Service']['Address'],
                    'port': service['Service']['Port'],
                    'tags': service['Service']['Tags']
                }
                discovered_nodes.append(node_info)
            
            return discovered_nodes
            
        except Exception as e:
            logger.error(f"Node discovery failed: {e}")
            return []
    
    def _parse_node_capabilities(self, node_info: Dict) -> NodeCapabilities:
        """Parse node capabilities from discovery info"""
        # This would parse actual capability information
        # For now, create basic capabilities
        return NodeCapabilities(
            node_id=node_info['node_id'],
            node_type=NodeType.EDGE_DEVICE,  # Would parse from tags
            cpu_cores=4,
            memory_gb=8.0,
            supported_models=['gpt2'],  # Would parse from tags
            geographic_region='us-east-1'  # Would parse from tags
        )
    
    async def get_network_status(self) -> Dict[str, Any]:
        """Get comprehensive network status"""
        healthy_nodes = sum(1 for node in self.nodes.values() 
                           if node.metrics.status == NodeStatus.HEALTHY)
        
        total_load = sum(node.get_load_score() for node in self.nodes.values())
        average_load = total_load / len(self.nodes) if self.nodes else 0
        
        return {
            'total_nodes': len(self.nodes),
            'healthy_nodes': healthy_nodes,
            'average_load': average_load,
            'supported_models': list(self.router.model_to_nodes.keys()),
            'cache_size': len(self.response_cache),
            'uptime': time.time() - getattr(self, 'start_time', time.time())
        }


# Example usage and testing
async def example_usage():
    """Example usage of the global inference network"""
    
    # Create network configuration
    config = NetworkConfig(
        default_routing_strategy=RoutingStrategy.INTELLIGENT,
        enable_auto_scaling=True,
        max_nodes=10
    )
    
    # Create and start coordinator
    coordinator = GlobalNetworkCoordinator(config)
    await coordinator.start()
    
    try:
        # Register some example nodes
        edge_node_1 = NodeCapabilities(
            node_id="edge-1",
            node_type=NodeType.EDGE_DEVICE,
            cpu_cores=4,
            memory_gb=8.0,
            supported_models=["gpt2", "distilgpt2"],
            geographic_region="us-east-1"
        )
        
        cloud_node_1 = NodeCapabilities(
            node_id="cloud-1",
            node_type=NodeType.CLOUD_INSTANCE,
            cpu_cores=16,
            memory_gb=64.0,
            gpu_available=True,
            gpu_memory_gb=16.0,
            supported_models=["gpt2", "gpt-j", "llama"],
            geographic_region="us-east-1",
            cost_per_hour=2.50
        )
        
        await coordinator.register_node(edge_node_1)
        await coordinator.register_node(cloud_node_1)
        
        # Process some test requests
        test_requests = [
            InferenceRequest(
                request_id=str(uuid.uuid4()),
                user_id="user_1",
                prompt="Write a story about space exploration",
                model_name="gpt2",
                priority=RequestPriority.NORMAL
            ),
            InferenceRequest(
                request_id=str(uuid.uuid4()),
                user_id="user_2",
                prompt="Explain quantum computing",
                model_name="gpt-j",
                priority=RequestPriority.HIGH,
                cost_budget=1.00
            )
        ]
        
        print("Processing test requests...")
        for request in test_requests:
            response = await coordinator.process_request(request)
            print(f"Request {request.request_id}: {'SUCCESS' if response.success else 'FAILED'}")
            print(f"  Node: {response.node_id}")
            print(f"  Latency: {response.total_latency_ms:.1f}ms")
            if not response.success:
                print(f"  Error: {response.error_message}")
        
        # Get network status
        status = await coordinator.get_network_status()
        print(f"\nNetwork Status:")
        print(f"  Total nodes: {status['total_nodes']}")
        print(f"  Healthy nodes: {status['healthy_nodes']}")
        print(f"  Average load: {status['average_load']:.2f}")
        print(f"  Supported models: {status['supported_models']}")
        
    finally:
        await coordinator.stop()


if __name__ == "__main__":
    # Run the example
    asyncio.run(example_usage())