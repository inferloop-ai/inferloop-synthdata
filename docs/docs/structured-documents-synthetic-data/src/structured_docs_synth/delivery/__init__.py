"""
Delivery module for structured document synthesis.

This module handles document delivery, export, and storage functionality
including API endpoints, format conversion, and integration with external systems.
"""

from .api.rest_api import RestAPI
from .api.graphql_api import GraphQLAPI
from .api.websocket_api import WebSocketAPI
from .export.format_exporters import (
    PDFExporter,
    DocxExporter, 
    JSONExporter,
    HTMLExporter,
    XMLExporter
)
from .export.batch_exporter import BatchExporter
from .export.streaming_exporter import StreamingExporter
from .export.rag_integrator import RAGIntegrator
from .storage.database_storage import DatabaseStorage
from .storage.cloud_storage import CloudStorage
from .storage.cache_manager import CacheManager
from .storage.vector_store import VectorStore

__all__ = [
    # API Components
    'RestAPI',
    'GraphQLAPI', 
    'WebSocketAPI',
    
    # Export Components
    'PDFExporter',
    'DocxExporter',
    'JSONExporter', 
    'HTMLExporter',
    'XMLExporter',
    'BatchExporter',
    'StreamingExporter',
    'RAGIntegrator',
    
    # Storage Components
    'DatabaseStorage',
    'CloudStorage',
    'CacheManager',
    'VectorStore'
]


def create_delivery_pipeline(config: dict = None):
    """
    Create a complete delivery pipeline with all components.
    
    Args:
        config: Configuration dictionary for delivery components
        
    Returns:
        Configured delivery pipeline instance
    """
    from .pipeline import DeliveryPipeline
    return DeliveryPipeline(config or {})


def create_api_server(api_type: str = 'rest', config: dict = None):
    """
    Create an API server instance.
    
    Args:
        api_type: Type of API server ('rest', 'graphql', 'websocket')
        config: Configuration dictionary
        
    Returns:
        Configured API server instance
    """
    config = config or {}
    
    if api_type == 'rest':
        return RestAPI(config)
    elif api_type == 'graphql':
        return GraphQLAPI(config)
    elif api_type == 'websocket':
        return WebSocketAPI(config)
    else:
        raise ValueError(f"Unsupported API type: {api_type}")


def create_exporter(format_type: str, config: dict = None):
    """
    Create a format exporter instance.
    
    Args:
        format_type: Export format type ('pdf', 'docx', 'json', 'html', 'xml')
        config: Configuration dictionary
        
    Returns:
        Configured exporter instance
    """
    config = config or {}
    
    exporters = {
        'pdf': PDFExporter,
        'docx': DocxExporter,
        'json': JSONExporter,
        'html': HTMLExporter,
        'xml': XMLExporter
    }
    
    if format_type not in exporters:
        raise ValueError(f"Unsupported export format: {format_type}")
        
    return exporters[format_type](config)


def create_storage_backend(storage_type: str, config: dict = None):
    """
    Create a storage backend instance.
    
    Args:
        storage_type: Type of storage ('database', 'cloud', 'cache', 'vector')
        config: Configuration dictionary
        
    Returns:
        Configured storage backend instance
    """
    config = config or {}
    
    storage_backends = {
        'database': DatabaseStorage,
        'cloud': CloudStorage,
        'cache': CacheManager,
        'vector': VectorStore
    }
    
    if storage_type not in storage_backends:
        raise ValueError(f"Unsupported storage type: {storage_type}")
        
    return storage_backends[storage_type](config)