#!/usr/bin/env python3
"""
Data Ingestion Module

This module provides comprehensive data ingestion capabilities for structured document
synthetic data generation, supporting both batch and streaming ingestion patterns.

Components:
- Batch processing: Dataset loaders, file processors (IMPLEMENTED)
- Streaming processing: API pollers, Kafka consumers, webhook handlers (STUB IMPLEMENTATIONS)
- External dataset adapters: Domain-specific data integration (STUB IMPLEMENTATIONS)

The ingestion module serves as the entry point for all data sources, ensuring
consistent data formatting and quality before passing to processing pipelines.
"""

# Batch processing imports (IMPLEMENTED)
from .batch import (
    DatasetLoader, DatasetLoaderConfig, LoadedDataset,
    DatasetMetadata, DocumentRecord, DatasetFormat, DatasetSource,
    FileProcessor, FileProcessorConfig, ProcessedFile,
    FileMetadata, FileType, ProcessingStatus,
    create_dataset_loader, create_file_processor,
    create_batch_processing_pipeline
)

# Factory functions for creating ingestion pipelines
def create_batch_ingestion_pipeline(**config_kwargs):
    """Create a complete batch ingestion pipeline"""
    return create_batch_processing_pipeline(**config_kwargs)

# Main ingestion orchestrator (simplified for implemented components)
class DataIngestionOrchestrator:
    """
    Orchestrates data ingestion from batch sources (implemented)
    """
    
    def __init__(self, config=None):
        from ..core.config import get_config
        from ..core.logging import get_logger
        
        self.config = config or get_config()
        self.logger = get_logger(__name__)
        
        # Initialize batch pipeline (only implemented component)
        self.batch_pipeline = create_batch_ingestion_pipeline()
        
        self.logger.info("Data Ingestion Orchestrator initialized (batch mode only)")
    
    def ingest_batch_data(self, source_type, source_path, **kwargs):
        """Ingest data from batch sources"""
        if source_type == 'dataset':
            return self.batch_pipeline['dataset_loader'].load_dataset(source_path, **kwargs)
        elif source_type == 'files':
            return self.batch_pipeline['file_processor'].process_files(source_path, **kwargs)
        else:
            raise ValueError(f"Unknown batch source type: {source_type}")
    
    def get_available_datasets(self):
        """Get list of available datasets"""
        return self.batch_pipeline['dataset_loader'].get_available_datasets()
    
    def discover_files(self, directory_path, **kwargs):
        """Discover files in a directory"""
        return self.batch_pipeline['file_processor'].discover_files(directory_path, **kwargs)

# Export implemented components only
__all__ = [
    # Batch processing (IMPLEMENTED)
    'DatasetLoader',
    'DatasetLoaderConfig', 
    'LoadedDataset',
    'DatasetMetadata',
    'DocumentRecord',
    'DatasetFormat',
    'DatasetSource',
    'FileProcessor',
    'FileProcessorConfig',
    'ProcessedFile',
    'FileMetadata',
    'FileType',
    'ProcessingStatus',
    'create_dataset_loader',
    'create_file_processor',
    'create_batch_processing_pipeline',
    
    # Factory functions
    'create_batch_ingestion_pipeline',
    
    # Main orchestrator
    'DataIngestionOrchestrator'
]