#!/usr/bin/env python3
"""
Kafka Consumer for streaming data ingestion
"""

import json
import time
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib
import threading

from pydantic import BaseModel, Field

from ...core.config import get_config
from ...core.logging import get_logger
from ...core.exceptions import ProcessingError, ValidationError


class ConsumerMode(Enum):
    """Kafka consumer modes"""
    AUTO_COMMIT = "auto_commit"
    MANUAL_COMMIT = "manual_commit"
    BATCH_COMMIT = "batch_commit"
    TRANSACTIONAL = "transactional"


class MessageFormat(Enum):
    """Expected message formats"""
    JSON = "json"
    AVRO = "avro"
    PROTOBUF = "protobuf"
    TEXT = "text"
    BINARY = "binary"


@dataclass
class ConsumedMessage:
    """Consumed message from Kafka"""
    topic: str
    partition: int
    offset: int
    key: Optional[str]
    value: Any
    timestamp: datetime
    headers: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    message_hash: str = ""
    processing_status: str = "pending"


class KafkaConsumerConfig(BaseModel):
    """Kafka consumer configuration"""
    
    # Connection settings
    bootstrap_servers: List[str] = Field(
        default=["localhost:9092"],
        description="Kafka bootstrap servers"
    )
    client_id: str = Field("synthetic-data-consumer", description="Client ID")
    group_id: str = Field("synthetic-data-group", description="Consumer group ID")
    
    # Consumer settings
    auto_offset_reset: str = Field("latest", description="Auto offset reset policy")
    enable_auto_commit: bool = Field(True, description="Enable auto commit")
    auto_commit_interval_ms: int = Field(1000, description="Auto commit interval")
    max_poll_records: int = Field(500, description="Maximum records per poll")
    session_timeout_ms: int = Field(30000, description="Session timeout")
    heartbeat_interval_ms: int = Field(3000, description="Heartbeat interval")
    
    # Message processing
    message_format: MessageFormat = Field(MessageFormat.JSON, description="Expected message format")
    max_message_size: int = Field(1048576, description="Maximum message size in bytes")
    decode_errors: str = Field("ignore", description="How to handle decode errors")
    
    # Performance
    fetch_min_bytes: int = Field(1, description="Minimum bytes to fetch")
    fetch_max_wait_ms: int = Field(500, description="Maximum wait time for fetch")
    consumer_timeout_ms: int = Field(1000, description="Consumer timeout")
    
    # Error handling
    max_retries: int = Field(3, description="Maximum retry attempts")
    retry_backoff_ms: int = Field(1000, description="Retry backoff time")
    skip_invalid_messages: bool = Field(True, description="Skip invalid messages")
    
    # Security (placeholder for production use)
    security_protocol: str = Field("PLAINTEXT", description="Security protocol")
    sasl_mechanism: Optional[str] = Field(None, description="SASL mechanism")
    sasl_username: Optional[str] = Field(None, description="SASL username")
    sasl_password: Optional[str] = Field(None, description="SASL password")
    
    # Monitoring
    enable_metrics: bool = Field(True, description="Enable consumption metrics")
    log_message_content: bool = Field(False, description="Log message content (debug)")


class KafkaConsumer:
    """
    Kafka Consumer for streaming data ingestion
    
    Consumes messages from Kafka topics with support for various formats,
    error handling, metrics, and callback-based processing.
    
    Note: This is a mock implementation. In production, you would use
    kafka-python or confluent-kafka libraries.
    """
    
    def __init__(self, config: Optional[KafkaConsumerConfig] = None):
        self.logger = get_logger(__name__)
        self.config = config or KafkaConsumerConfig()
        
        # State management
        self.subscribed_topics: List[str] = []
        self.consumed_messages: List[ConsumedMessage] = []
        self.message_hashes: set = set()
        
        # Consumer state
        self.is_consuming = False
        self.consumer_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.message_callbacks: List[Callable[[ConsumedMessage], None]] = []
        self.error_callbacks: List[Callable[[str, Exception], None]] = []
        self.batch_callbacks: List[Callable[[List[ConsumedMessage]], None]] = []
        
        # Metrics
        self.metrics = {
            "total_messages": 0,
            "successful_messages": 0,
            "failed_messages": 0,
            "bytes_consumed": 0,
            "start_time": None,
            "last_message_time": None
        }
        
        # Mock consumer (in production, initialize kafka-python consumer)
        self.mock_consumer = None
        self._initialize_mock_consumer()
        
        self.logger.info(f"Kafka Consumer initialized for group: {self.config.group_id}")
    
    def _initialize_mock_consumer(self):
        """Initialize mock consumer (replace with real Kafka consumer in production)"""
        # In production:
        # from kafka import KafkaConsumer as PyKafkaConsumer
        # self.consumer = PyKafkaConsumer(
        #     bootstrap_servers=self.config.bootstrap_servers,
        #     group_id=self.config.group_id,
        #     auto_offset_reset=self.config.auto_offset_reset,
        #     enable_auto_commit=self.config.enable_auto_commit,
        #     value_deserializer=self._get_deserializer()
        # )
        
        self.mock_consumer = {
            "connected": True,
            "subscribed_topics": [],
            "message_queue": []
        }
    
    def subscribe(self, topics: Union[str, List[str]]):
        """Subscribe to Kafka topics"""
        if isinstance(topics, str):
            topics = [topics]
        
        self.subscribed_topics.extend(topics)
        
        # Mock subscription
        self.mock_consumer["subscribed_topics"].extend(topics)
        
        # In production:
        # self.consumer.subscribe(topics)
        
        self.logger.info(f"Subscribed to topics: {topics}")
    
    def add_message_callback(self, callback: Callable[[ConsumedMessage], None]):
        """Add callback for individual messages"""
        self.message_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[str, Exception], None]):
        """Add callback for errors"""
        self.error_callbacks.append(callback)
    
    def add_batch_callback(self, callback: Callable[[List[ConsumedMessage]], None]):
        """Add callback for message batches"""
        self.batch_callbacks.append(callback)
    
    def start_consuming(self, duration: Optional[float] = None) -> List[ConsumedMessage]:
        """Start consuming messages"""
        if self.is_consuming:
            raise ValueError("Consumer is already running")
        
        self.is_consuming = True
        self.metrics["start_time"] = time.time()
        
        try:
            if duration:
                # Consume for specified duration
                return self._consume_with_timeout(duration)
            else:
                # Start background consumption
                self._start_background_consumption()
                return []
                
        except Exception as e:
            self.is_consuming = False
            self.logger.error(f"Failed to start consuming: {e}")
            raise ProcessingError(f"Kafka consumption error: {e}")
    
    def _consume_with_timeout(self, duration: float) -> List[ConsumedMessage]:
        """Consume messages for specified duration"""
        start_time = time.time()
        messages = []
        
        self.logger.info(f"Starting Kafka consumption for {duration} seconds")
        
        while (time.time() - start_time) < duration and self.is_consuming:
            try:
                # Poll for messages (mock implementation)
                batch = self._poll_messages()
                
                if batch:
                    messages.extend(batch)
                    self._process_message_batch(batch)
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                self._handle_error("consumption", e)
                if not self.config.skip_invalid_messages:
                    raise
        
        self.is_consuming = False
        self.logger.info(f"Consumption completed. Processed {len(messages)} messages")
        return messages
    
    def _start_background_consumption(self):
        """Start background consumption thread"""
        def consume_loop():
            while self.is_consuming:
                try:
                    batch = self._poll_messages()
                    if batch:
                        self._process_message_batch(batch)
                    time.sleep(0.1)
                except Exception as e:
                    self._handle_error("background_consumption", e)
        
        self.consumer_thread = threading.Thread(target=consume_loop, daemon=True)
        self.consumer_thread.start()
        self.logger.info("Started background Kafka consumption")
    
    def _poll_messages(self) -> List[ConsumedMessage]:
        """Poll for new messages (mock implementation)"""
        # Mock message generation for testing
        # In production, this would be:
        # message_batch = self.consumer.poll(timeout_ms=self.config.consumer_timeout_ms)
        
        if not self.subscribed_topics:
            return []
        
        # Generate mock messages occasionally
        import random
        if random.random() < 0.1:  # 10% chance of new messages
            mock_messages = self._generate_mock_messages()
            return mock_messages
        
        return []
    
    def _generate_mock_messages(self) -> List[ConsumedMessage]:
        """Generate mock messages for testing"""
        import random
        
        messages = []
        num_messages = random.randint(1, 5)
        
        for i in range(num_messages):
            topic = random.choice(self.subscribed_topics)
            
            # Generate mock message data
            mock_data = {
                "id": f"msg_{int(time.time())}_{i}",
                "timestamp": datetime.now().isoformat(),
                "content": f"Sample document content {i}",
                "metadata": {
                    "source": "mock_producer",
                    "type": "document"
                }
            }
            
            message = ConsumedMessage(
                topic=topic,
                partition=random.randint(0, 3),
                offset=random.randint(1000, 9999),
                key=f"key_{i}",
                value=mock_data,
                timestamp=datetime.now(),
                headers={"content-type": "application/json"},
                metadata={"mock": True}
            )
            
            # Calculate message hash
            message_str = json.dumps(mock_data, sort_keys=True)
            message.message_hash = hashlib.md5(message_str.encode()).hexdigest()
            
            messages.append(message)
        
        return messages
    
    def _process_message_batch(self, messages: List[ConsumedMessage]):
        """Process a batch of messages"""
        processed_messages = []
        
        for message in messages:
            try:
                # Process individual message
                processed_message = self._process_message(message)
                if processed_message:
                    processed_messages.append(processed_message)
                    
            except Exception as e:
                self._handle_error(f"message_processing_{message.topic}", e)
                message.processing_status = "failed"
                
                if not self.config.skip_invalid_messages:
                    raise
        
        # Store processed messages
        self.consumed_messages.extend(processed_messages)
        
        # Trigger batch callbacks
        if processed_messages:
            for callback in self.batch_callbacks:
                try:
                    callback(processed_messages)
                except Exception as e:
                    self.logger.warning(f"Batch callback failed: {e}")
    
    def _process_message(self, message: ConsumedMessage) -> Optional[ConsumedMessage]:
        """Process individual message"""
        try:
            # Deserialize message based on format
            if self.config.message_format == MessageFormat.JSON:
                # Message value should already be deserialized
                if isinstance(message.value, str):
                    message.value = json.loads(message.value)
            
            # Check for duplicates
            if message.message_hash in self.message_hashes:
                self.logger.debug(f"Duplicate message {message.message_hash}, skipping")
                return None
            
            self.message_hashes.add(message.message_hash)
            
            # Validate message size
            message_size = len(json.dumps(message.value))
            if message_size > self.config.max_message_size:
                raise ValidationError(f"Message too large: {message_size} bytes")
            
            # Update metrics
            self.metrics["total_messages"] += 1
            self.metrics["successful_messages"] += 1
            self.metrics["bytes_consumed"] += message_size
            self.metrics["last_message_time"] = time.time()
            
            # Set processing status
            message.processing_status = "processed"
            
            # Log message content if enabled
            if self.config.log_message_content:
                self.logger.debug(f"Processed message from {message.topic}: {message.value}")
            
            # Trigger message callbacks
            for callback in self.message_callbacks:
                try:
                    callback(message)
                except Exception as e:
                    self.logger.warning(f"Message callback failed: {e}")
            
            return message
            
        except Exception as e:
            self.metrics["failed_messages"] += 1
            message.processing_status = "failed"
            raise ProcessingError(f"Message processing failed: {e}")
    
    def _handle_error(self, context: str, error: Exception):
        """Handle consumption errors"""
        self.logger.error(f"Error in {context}: {error}")
        
        for callback in self.error_callbacks:
            try:
                callback(context, error)
            except Exception as e:
                self.logger.warning(f"Error callback failed: {e}")
    
    def stop_consuming(self):
        """Stop consuming messages"""
        self.is_consuming = False
        
        if self.consumer_thread and self.consumer_thread.is_alive():
            self.consumer_thread.join(timeout=5.0)
        
        # In production:
        # if self.consumer:
        #     self.consumer.close()
        
        self.logger.info("Kafka consumption stopped")
    
    def commit_offsets(self, message: Optional[ConsumedMessage] = None):
        """Manually commit offsets"""
        # In production:
        # if message:
        #     self.consumer.commit({TopicPartition(message.topic, message.partition): message.offset + 1})
        # else:
        #     self.consumer.commit()
        
        self.logger.debug("Offsets committed")
    
    def seek_to_beginning(self, topic: str, partition: int):
        """Seek to beginning of partition"""
        # In production:
        # from kafka import TopicPartition
        # tp = TopicPartition(topic, partition)
        # self.consumer.seek_to_beginning(tp)
        
        self.logger.info(f"Seeking to beginning of {topic}:{partition}")
    
    def seek_to_end(self, topic: str, partition: int):
        """Seek to end of partition"""
        # In production:
        # from kafka import TopicPartition
        # tp = TopicPartition(topic, partition)
        # self.consumer.seek_to_end(tp)
        
        self.logger.info(f"Seeking to end of {topic}:{partition}")
    
    def get_consumption_metrics(self) -> Dict[str, Any]:
        """Get consumption metrics"""
        current_time = time.time()
        duration = current_time - self.metrics["start_time"] if self.metrics["start_time"] else 0
        
        return {
            "total_messages": self.metrics["total_messages"],
            "successful_messages": self.metrics["successful_messages"],
            "failed_messages": self.metrics["failed_messages"],
            "success_rate": (
                self.metrics["successful_messages"] / self.metrics["total_messages"]
                if self.metrics["total_messages"] > 0 else 0
            ),
            "bytes_consumed": self.metrics["bytes_consumed"],
            "consumption_duration": duration,
            "messages_per_second": (
                self.metrics["total_messages"] / duration
                if duration > 0 else 0
            ),
            "subscribed_topics": self.subscribed_topics,
            "unique_messages": len(self.message_hashes),
            "is_consuming": self.is_consuming
        }
    
    def export_consumed_messages(self, format: str = "json") -> List[Dict[str, Any]]:
        """Export consumed messages"""
        exported_data = []
        
        for message in self.consumed_messages:
            exported_item = {
                "topic": message.topic,
                "partition": message.partition,
                "offset": message.offset,
                "key": message.key,
                "value": message.value,
                "timestamp": message.timestamp.isoformat(),
                "processing_status": message.processing_status,
                "message_hash": message.message_hash
            }
            
            if message.headers:
                exported_item["headers"] = message.headers
            
            if message.metadata:
                exported_item["metadata"] = message.metadata
            
            exported_data.append(exported_item)
        
        return exported_data
    
    def clear_consumed_messages(self):
        """Clear consumed messages history"""
        self.consumed_messages.clear()
        self.message_hashes.clear()
        self.logger.info("Consumed messages history cleared")


# Factory function
def create_kafka_consumer(**config_kwargs) -> KafkaConsumer:
    """Factory function to create Kafka consumer"""
    config = KafkaConsumerConfig(**config_kwargs)
    return KafkaConsumer(config)