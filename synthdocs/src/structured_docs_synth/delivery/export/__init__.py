"""
Export module for structured document synthesis delivery.

This module provides comprehensive export functionality for converting documents
to various formats with privacy protection and quality assurance.

Features:
- Multi-format export (COCO, YOLO, Pascal VOC, JSON, CSV, PDF, DOCX, HTML, XML)
- Batch processing capabilities
- Streaming export for large datasets
- Privacy-preserving export options
- RAG integration for enhanced document synthesis
- Quality validation during export
- Progress tracking and error handling
"""

from .format_exporters import (
    # Base classes
    BaseExporter,
    FormatExporterFactory,
    
    # Specific format exporters
    COCOExporter,
    YOLOExporter,
    PascalVOCExporter,
    JSONExporter,
    CSVExporter,
    PDFExporter,
    DocxExporter,
    HTMLExporter,
    XMLExporter,
    
    # Factory functions
    create_format_exporter,
    get_supported_export_formats,
)

from .batch_exporter import (
    ExportJob,
    BatchExporter,
    create_batch_exporter,
)

from .streaming_exporter import (
    StreamingExporter,
    create_streaming_exporter,
)

from .rag_integrator import (
    RAGIntegrator,
)

# Export utilities
__all__ = [
    # Base classes
    "BaseExporter",
    "FormatExporterFactory",
    
    # Format exporters
    "COCOExporter",
    "YOLOExporter", 
    "PascalVOCExporter",
    "JSONExporter",
    "CSVExporter",
    "PDFExporter",
    "DocxExporter",
    "HTMLExporter",
    "XMLExporter",
    
    # Batch processing
    "ExportJob",
    "BatchExporter",
    
    # Streaming
    "StreamingExporter",
    
    # RAG integration
    "RAGIntegrator",
    
    # Factory functions
    "create_format_exporter",
    "create_batch_exporter",
    "create_streaming_exporter",
    "get_supported_export_formats",
    
    # Utility functions
    "export_documents",
    "export_documents_batch",
    "export_documents_streaming",
]


# High-level export functions
async def export_documents(documents, format_type, output_path, **kwargs):
    """
    High-level function to export documents to specified format.
    
    Args:
        documents: List of document dictionaries to export
        format_type: Target export format (coco, yolo, json, etc.)
        output_path: Output directory path
        **kwargs: Additional export options
        
    Returns:
        Export result dictionary
        
    Example:
        >>> documents = [{'id': '1', 'content': 'test', 'annotations': []}]
        >>> result = await export_documents(documents, 'json', './output')
        >>> print(result['exported_count'])
    """
    exporter = create_format_exporter(
        format_type, 
        privacy_protection=kwargs.get('privacy_protection', True)
    )
    
    return await exporter.export(documents, output_path, kwargs)


async def export_documents_batch(document_ids, format_type, output_path, **kwargs):
    """
    Export documents in batch with job management and progress tracking.
    
    Args:
        document_ids: List of document IDs to export
        format_type: Target export format
        output_path: Output directory path
        **kwargs: Additional batch options
        
    Returns:
        Export job ID for tracking progress
        
    Example:
        >>> job_id = await export_documents_batch(['doc1', 'doc2'], 'coco', './output')
        >>> status = await batch_exporter.get_job_status(job_id)
    """
    batch_exporter = create_batch_exporter(
        batch_size=kwargs.get('batch_size', 100),
        max_workers=kwargs.get('max_workers', 4)
    )
    
    job_id = await batch_exporter.create_export_job(
        document_ids=document_ids,
        format_type=format_type,
        output_path=output_path,
        options=kwargs,
        user_id=kwargs.get('user_id')
    )
    
    await batch_exporter.start_export_job(
        job_id, 
        progress_callback=kwargs.get('progress_callback')
    )
    
    return job_id


async def export_documents_streaming(document_iterator, format_type, output_path, **kwargs):
    """
    Export documents using streaming for memory-efficient processing.
    
    Args:
        document_iterator: Iterator/generator yielding documents
        format_type: Target export format
        output_path: Output directory path
        **kwargs: Additional streaming options
        
    Returns:
        Streaming export result
        
    Example:
        >>> async def doc_generator():
        ...     for i in range(1000):
        ...         yield {'id': str(i), 'content': f'doc {i}'}
        >>> result = await export_documents_streaming(doc_generator(), 'json', './output')
    """
    streaming_exporter = create_streaming_exporter(
        chunk_size=kwargs.get('chunk_size', 100),
        memory_limit=kwargs.get('memory_limit', '1GB')
    )
    
    return await streaming_exporter.export_stream(
        document_iterator, 
        format_type, 
        output_path, 
        kwargs
    )


# Format detection utility
def detect_export_format(file_path_or_extension):
    """
    Detect appropriate export format based on file extension or path.
    
    Args:
        file_path_or_extension: File path or extension string
        
    Returns:
        Detected format string or None if not supported
        
    Example:
        >>> format_type = detect_export_format('./dataset.json')
        >>> print(format_type)  # 'json'
    """
    from pathlib import Path
    
    if isinstance(file_path_or_extension, (str, Path)):
        path = Path(file_path_or_extension)
        extension = path.suffix.lower().lstrip('.')
    else:
        extension = str(file_path_or_extension).lower().lstrip('.')
    
    extension_map = {
        'json': 'json',
        'jsonl': 'jsonl',
        'csv': 'csv',
        'xml': 'pascal_voc',
        'txt': 'yolo',
        'yaml': 'yolo',
        'yml': 'yolo',
        'pdf': 'pdf',
        'docx': 'docx',
        'html': 'html',
        'htm': 'html',
    }
    
    return extension_map.get(extension)


# Configuration validation
def validate_export_config(format_type, config):
    """
    Validate export configuration for specified format.
    
    Args:
        format_type: Export format type
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, validation_errors)
        
    Example:
        >>> is_valid, errors = validate_export_config('coco', {'copy_images': True})
        >>> if not is_valid:
        ...     print(f"Validation errors: {errors}")
    """
    supported_formats = get_supported_export_formats()
    
    if format_type not in supported_formats:
        return False, [f"Unsupported format: {format_type}"]
    
    errors = []
    
    # Format-specific validation
    if format_type in ['coco', 'yolo', 'pascal_voc']:
        if 'copy_images' in config and not isinstance(config['copy_images'], bool):
            errors.append("copy_images must be a boolean")
            
    if format_type == 'csv':
        if 'delimiter' in config and len(config['delimiter']) != 1:
            errors.append("CSV delimiter must be a single character")
    
    return len(errors) == 0, errors


# Module version and metadata
__version__ = "1.0.0"
__author__ = "InferLoop Team"