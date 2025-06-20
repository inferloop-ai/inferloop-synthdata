#!/usr/bin/env python3
"""
Batch Processing Module

Provides batch data ingestion capabilities including dataset loading
and file processing for structured document synthetic data generation.
"""

# Import main components
from .dataset_loader import (
    DatasetLoader, DatasetLoaderConfig, LoadedDataset,
    DatasetMetadata, DocumentRecord, DatasetFormat, DatasetSource,
    create_dataset_loader
)

from .file_processor import (
    FileProcessor, FileProcessorConfig, ProcessedFile,
    FileMetadata, FileType, ProcessingStatus,
    create_file_processor
)

# Factory functions for batch processing
def create_batch_processing_pipeline(**config_kwargs):
    """Create a complete batch processing pipeline"""
    dataset_config = config_kwargs.get('dataset_loader', {})
    file_config = config_kwargs.get('file_processor', {})
    
    return {
        'dataset_loader': create_dataset_loader(**dataset_config),
        'file_processor': create_file_processor(**file_config)
    }

# Export all components
__all__ = [
    # Dataset Loader
    'DatasetLoader',
    'DatasetLoaderConfig',
    'LoadedDataset',
    'DatasetMetadata',
    'DocumentRecord',
    'DatasetFormat',
    'DatasetSource',
    'create_dataset_loader',
    
    # File Processor
    'FileProcessor',
    'FileProcessorConfig',
    'ProcessedFile',
    'FileMetadata',
    'FileType',
    'ProcessingStatus',
    'create_file_processor',
    
    # Factory functions
    'create_batch_processing_pipeline'
]