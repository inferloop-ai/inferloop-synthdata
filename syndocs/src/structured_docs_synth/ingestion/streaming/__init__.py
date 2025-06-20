#!/usr/bin/env python3
"""
Streaming Data Ingestion Module

Provides real-time data ingestion capabilities including API polling,
Kafka consumption, and webhook handling for streaming document data.
"""

# Import main components
from .api_poller import (
    APIPoller, APIPollerConfig, PolledData,
    APIEndpoint, PollingMode, APIAuthType,
    create_api_poller
)

from .kafka_consumer import (
    KafkaConsumer, KafkaConsumerConfig, ConsumedMessage,
    ConsumerMode, MessageFormat,
    create_kafka_consumer
)

from .webhook_handler import (
    WebhookHandler, WebhookHandlerConfig, WebhookEvent,
    WebhookSecurity, WebhookMethod,
    create_webhook_handler
)

# Factory functions for streaming processing
def create_streaming_ingestion_pipeline(**config_kwargs):
    """Create a complete streaming ingestion pipeline"""
    api_config = config_kwargs.get('api_poller', {})
    kafka_config = config_kwargs.get('kafka_consumer', {})
    webhook_config = config_kwargs.get('webhook_handler', {})
    
    return {
        'api_poller': create_api_poller(**api_config),
        'kafka_consumer': create_kafka_consumer(**kafka_config),
        'webhook_handler': create_webhook_handler(**webhook_config)
    }

# Export all components
__all__ = [
    # API Poller
    'APIPoller',
    'APIPollerConfig',
    'PolledData',
    'APIEndpoint',
    'PollingMode',
    'APIAuthType',
    'create_api_poller',
    
    # Kafka Consumer
    'KafkaConsumer',
    'KafkaConsumerConfig',
    'ConsumedMessage',
    'ConsumerMode',
    'MessageFormat',
    'create_kafka_consumer',
    
    # Webhook Handler
    'WebhookHandler',
    'WebhookHandlerConfig',
    'WebhookEvent',
    'WebhookSecurity',
    'WebhookMethod',
    'create_webhook_handler',
    
    # Factory functions
    'create_streaming_ingestion_pipeline'
]