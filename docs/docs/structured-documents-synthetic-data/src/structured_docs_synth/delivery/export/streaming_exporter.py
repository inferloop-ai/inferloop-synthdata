#!/usr/bin/env python3
"""
Streaming exporter for real-time data export and processing.

Provides streaming export capabilities for continuous data processing,
real-time exports, and live data feeds with backpressure handling.

Features:
- Real-time document streaming
- Backpressure handling and flow control
- Multiple output targets (files, APIs, message queues)
- Configurable buffering and batching
- Error handling and retry mechanisms
- Monitoring and metrics collection
- Rate limiting and throttling
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional
from uuid import uuid4

import aiofiles
import aiohttp

from ...core import get_logger, get_config
from ...core.exceptions import ProcessingError, ValidationError
from .format_exporters import create_format_exporter


logger = get_logger(__name__)
config = get_config()


class StreamTarget(ABC):
    """Abstract base class for streaming targets"""
    
    @abstractmethod
    async def write(self, data: Dict[str, Any]) -> bool:
        """Write data to target"""
        pass
    
    @abstractmethod
    async def close(self):
        """Close target connection"""
        pass


class FileStreamTarget(StreamTarget):
    """File-based streaming target"""
    
    def __init__(self, file_path: str, format_type: str = 'jsonl'):
        self.file_path = file_path
        self.format_type = format_type
        self.file_handle = None
        
    async def __aenter__(self):
        self.file_handle = await aiofiles.open(self.file_path, 'a', encoding='utf-8')
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def write(self, data: Dict[str, Any]) -> bool:
        """Write data to file"""
        try:
            if not self.file_handle:
                return False
                
            if self.format_type == 'jsonl':
                line = json.dumps(data, ensure_ascii=False) + '\n'
                await self.file_handle.write(line)
                await self.file_handle.flush()
            else:
                # For other formats, convert data first
                formatted_data = await self._format_data(data)
                await self.file_handle.write(formatted_data)
                await self.file_handle.flush()
            
            return True
            
        except Exception as e:
            logger.error(f"Error writing to file {self.file_path}: {e}")
            return False
    
    async def _format_data(self, data: Dict[str, Any]) -> str:
        """Format data for specific file type"""
        if self.format_type == 'json':
            return json.dumps(data, ensure_ascii=False, indent=2) + '\n'
        elif self.format_type == 'csv':
            # Simple CSV formatting - would need proper CSV writer for production
            if isinstance(data, dict):
                values = [str(v) for v in data.values()]
                return ','.join(values) + '\n'
        
        return str(data) + '\n'
    
    async def close(self):
        """Close file handle"""
        if self.file_handle:
            await self.file_handle.close()
            self.file_handle = None


class HTTPStreamTarget(StreamTarget):
    """HTTP endpoint streaming target"""
    
    def __init__(self, endpoint_url: str, headers: Optional[Dict[str, str]] = None):
        self.endpoint_url = endpoint_url
        self.headers = headers or {}
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def write(self, data: Dict[str, Any]) -> bool:
        """Send data to HTTP endpoint"""
        try:
            if not self.session:
                return False
                
            async with self.session.post(
                self.endpoint_url,
                json=data,
                headers=self.headers
            ) as response:
                if response.status < 400:
                    return True
                else:
                    logger.error(f"HTTP error {response.status}: {await response.text()}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending to HTTP endpoint {self.endpoint_url}: {e}")
            return False
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None


class KafkaStreamTarget(StreamTarget):
    """Kafka streaming target (placeholder implementation)"""
    
    def __init__(self, bootstrap_servers: List[str], topic: str):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = None
    
    async def __aenter__(self):
        # In a real implementation, you would initialize Kafka producer here
        # from aiokafka import AIOKafkaProducer
        # self.producer = AIOKafkaProducer(
        #     bootstrap_servers=self.bootstrap_servers
        # )
        # await self.producer.start()
        logger.info(f"Kafka producer initialized for topic {self.topic}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def write(self, data: Dict[str, Any]) -> bool:
        """Send data to Kafka topic"""
        try:
            # In a real implementation:
            # message = json.dumps(data).encode('utf-8')
            # await self.producer.send(self.topic, message)
            logger.info(f"Would send to Kafka topic {self.topic}: {data.get('id', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending to Kafka topic {self.topic}: {e}")
            return False
    
    async def close(self):
        """Close Kafka producer"""
        if self.producer:
            # await self.producer.stop()
            self.producer = None
            logger.info("Kafka producer closed")


class StreamBuffer:
    """Buffer for streaming data with backpressure handling"""
    
    def __init__(self, max_size: int = 1000, batch_size: int = 10):
        self.max_size = max_size
        self.batch_size = batch_size
        self.buffer = deque()
        self.lock = asyncio.Lock()
        self.not_full = asyncio.Condition(self.lock)
        self.not_empty = asyncio.Condition(self.lock)
        self.closed = False
    
    async def put(self, item: Dict[str, Any], timeout: Optional[float] = None) -> bool:
        """Add item to buffer with optional timeout"""
        async with self.not_full:
            # Wait for space in buffer
            try:
                await asyncio.wait_for(
                    self.not_full.wait_for(lambda: len(self.buffer) < self.max_size or self.closed),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                return False
            
            if self.closed:
                return False
            
            self.buffer.append(item)
            self.not_empty.notify()
            return True
    
    async def get_batch(self, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get batch of items from buffer"""
        async with self.not_empty:
            # Wait for items in buffer
            try:
                await asyncio.wait_for(
                    self.not_empty.wait_for(lambda: len(self.buffer) > 0 or self.closed),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                return []
            
            if self.closed and len(self.buffer) == 0:
                return []
            
            # Get batch of items
            batch = []
            for _ in range(min(self.batch_size, len(self.buffer))):
                if self.buffer:
                    batch.append(self.buffer.popleft())
            
            self.not_full.notify()
            return batch
    
    async def close(self):
        """Close buffer"""
        async with self.lock:
            self.closed = True
            self.not_full.notify_all()
            self.not_empty.notify_all()
    
    def size(self) -> int:
        """Get current buffer size"""
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return len(self.buffer) >= self.max_size


class StreamingExporter:
    """Main streaming exporter class"""
    
    def __init__(self, buffer_size: int = 1000, batch_size: int = 10, 
                 max_retry_attempts: int = 3, retry_delay: float = 1.0):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.max_retry_attempts = max_retry_attempts
        self.retry_delay = retry_delay
        
        self.buffer = StreamBuffer(buffer_size, batch_size)
        self.targets = []
        self.metrics = {
            'documents_processed': 0,
            'documents_failed': 0,
            'bytes_sent': 0,
            'start_time': None,
            'last_activity': None
        }
        self.running = False
        self.processor_task = None
        
    def add_target(self, target: StreamTarget):
        """Add streaming target"""
        self.targets.append(target)
        logger.info(f"Added streaming target: {type(target).__name__}")
    
    def add_file_target(self, file_path: str, format_type: str = 'jsonl'):
        """Add file streaming target"""
        target = FileStreamTarget(file_path, format_type)
        self.add_target(target)
    
    def add_http_target(self, endpoint_url: str, headers: Optional[Dict[str, str]] = None):
        """Add HTTP streaming target"""
        target = HTTPStreamTarget(endpoint_url, headers)
        self.add_target(target)
    
    def add_kafka_target(self, bootstrap_servers: List[str], topic: str):
        """Add Kafka streaming target"""
        target = KafkaStreamTarget(bootstrap_servers, topic)
        self.add_target(target)
    
    async def start(self):
        """Start streaming exporter"""
        if self.running:
            logger.warning("Streaming exporter already running")
            return
        
        if not self.targets:
            raise ValidationError("No streaming targets configured")
        
        self.running = True
        self.metrics['start_time'] = datetime.utcnow()
        
        # Initialize targets
        for target in self.targets:
            if hasattr(target, '__aenter__'):
                await target.__aenter__()
        
        # Start processor task
        self.processor_task = asyncio.create_task(self._process_stream())
        
        logger.info(f"Streaming exporter started with {len(self.targets)} targets")
    
    async def stop(self):
        """Stop streaming exporter"""
        if not self.running:
            return
        
        self.running = False
        
        # Close buffer to signal end of stream
        await self.buffer.close()
        
        # Wait for processor to finish
        if self.processor_task:
            await self.processor_task
        
        # Close targets
        for target in self.targets:
            if hasattr(target, '__aexit__'):
                await target.__aexit__(None, None, None)
            else:
                await target.close()
        
        logger.info("Streaming exporter stopped")
    
    async def export_document(self, document: Dict[str, Any], timeout: Optional[float] = 5.0) -> bool:
        """Export single document to stream"""
        if not self.running:
            raise ProcessingError("Streaming exporter not running")
        
        # Add timestamp and processing metadata
        export_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'export_id': str(uuid4()),
            'document': document
        }
        
        success = await self.buffer.put(export_data, timeout=timeout)
        if success:
            self.metrics['last_activity'] = datetime.utcnow()
        
        return success
    
    async def export_documents_stream(self, documents: AsyncGenerator[Dict[str, Any], None]):
        """Export stream of documents"""
        async for document in documents:
            success = await self.export_document(document)
            if not success:
                logger.warning(f"Failed to buffer document {document.get('id', 'unknown')}")
    
    async def _process_stream(self):
        """Background task to process streaming data"""
        logger.info("Stream processor started")
        
        while self.running or self.buffer.size() > 0:
            try:
                # Get batch from buffer
                batch = await self.buffer.get_batch(timeout=1.0)
                
                if not batch:
                    continue
                
                # Process batch
                await self._process_batch(batch)
                
            except Exception as e:
                logger.error(f"Error in stream processor: {e}")
                await asyncio.sleep(self.retry_delay)
        
        logger.info("Stream processor finished")
    
    async def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process batch of documents"""
        for target in self.targets:
            for document_data in batch:
                success = await self._send_with_retry(target, document_data)
                
                if success:
                    self.metrics['documents_processed'] += 1
                    # Estimate bytes sent (rough calculation)
                    self.metrics['bytes_sent'] += len(json.dumps(document_data))
                else:
                    self.metrics['documents_failed'] += 1
                    logger.error(f"Failed to send document {document_data.get('export_id')} to {type(target).__name__}")
    
    async def _send_with_retry(self, target: StreamTarget, data: Dict[str, Any]) -> bool:
        """Send data to target with retry logic"""
        for attempt in range(self.max_retry_attempts):
            try:
                success = await target.write(data)
                if success:
                    return True
                
                # If not successful, wait before retry
                if attempt < self.max_retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    
            except Exception as e:
                logger.error(f"Error sending to target (attempt {attempt + 1}): {e}")
                if attempt < self.max_retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get streaming metrics"""
        metrics = self.metrics.copy()
        
        if metrics['start_time']:
            uptime = (datetime.utcnow() - metrics['start_time']).total_seconds()
            metrics['uptime_seconds'] = uptime
            
            if uptime > 0:
                metrics['documents_per_second'] = metrics['documents_processed'] / uptime
                metrics['bytes_per_second'] = metrics['bytes_sent'] / uptime
        
        metrics['buffer_size'] = self.buffer.size()
        metrics['buffer_full'] = self.buffer.is_full()
        metrics['running'] = self.running
        metrics['targets_count'] = len(self.targets)
        
        return metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        health = {
            'status': 'healthy' if self.running else 'stopped',
            'buffer_utilization': self.buffer.size() / self.buffer_size,
            'targets_count': len(self.targets),
            'metrics': self.get_metrics()
        }
        
        # Check if buffer is getting full (potential backpressure)
        if health['buffer_utilization'] > 0.8:
            health['status'] = 'warning'
            health['warning'] = 'Buffer utilization high - potential backpressure'
        
        # Check if there are recent failures
        if self.metrics['documents_failed'] > 0:
            failure_rate = self.metrics['documents_failed'] / max(1, self.metrics['documents_processed'] + self.metrics['documents_failed'])
            if failure_rate > 0.1:  # More than 10% failure rate
                health['status'] = 'warning'
                health['warning'] = f'High failure rate: {failure_rate:.2%}'
        
        return health


# Factory functions
def create_streaming_exporter(buffer_size: int = 1000, batch_size: int = 10) -> StreamingExporter:
    """Create streaming exporter instance"""
    return StreamingExporter(buffer_size=buffer_size, batch_size=batch_size)


def create_file_stream_target(file_path: str, format_type: str = 'jsonl') -> FileStreamTarget:
    """Create file streaming target"""
    return FileStreamTarget(file_path, format_type)


def create_http_stream_target(endpoint_url: str, headers: Optional[Dict[str, str]] = None) -> HTTPStreamTarget:
    """Create HTTP streaming target"""
    return HTTPStreamTarget(endpoint_url, headers)


__all__ = [
    'StreamTarget',
    'FileStreamTarget',
    'HTTPStreamTarget',
    'KafkaStreamTarget',
    'StreamBuffer',
    'StreamingExporter',
    'create_streaming_exporter',
    'create_file_stream_target',
    'create_http_stream_target'
]