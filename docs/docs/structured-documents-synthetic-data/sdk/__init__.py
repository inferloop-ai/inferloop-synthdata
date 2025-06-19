"""
Structured Documents Synthetic Data SDK.
Python SDK for generating, validating, and managing synthetic structured documents.
"""

from .client import StructuredDocsClient
from .async_client import AsyncStructuredDocsClient

# Import all model types
from .models import (
    # Document types
    DocumentType, DocumentFormat, LayoutComplexity, ContentQuality, Language,
    
    # Core models
    BoundingBox, DocumentElement, DocumentMetadata, PageSize, DocumentContent,
    
    # Request models
    GenerationRequest, BatchGenerationRequest, ValidationRequest, ExportRequest,
    
    # Response models
    GenerationResponse, BatchGenerationResponse, ValidationResponse, ExportResponse,
    DocumentListResponse, StatusResponse, QuotaResponse, JobResponse,
    
    # Configuration models
    GenerationConfig, LayoutConfig, ContentConfig, PrivacyConfig,
    NoiseConfig, RenderingConfig, PerformanceConfig,
    
    # Factory functions
    create_generation_request, create_batch_request, get_config_preset,
    create_custom_config, get_default_config_for_document_type
)

# Version information
__version__ = "1.0.0"
__author__ = "Structured Documents Team"
__email__ = "support@structuredocs.ai"
__description__ = "Python SDK for structured document synthetic data generation"

# Public API
__all__ = [
    # Main clients
    "StructuredDocsClient",
    "AsyncStructuredDocsClient",
    
    # Document types and enums
    "DocumentType",
    "DocumentFormat", 
    "LayoutComplexity",
    "ContentQuality",
    "Language",
    
    # Core data models
    "BoundingBox",
    "DocumentElement",
    "DocumentMetadata",
    "PageSize",
    "DocumentContent",
    
    # Request models
    "GenerationRequest",
    "BatchGenerationRequest", 
    "ValidationRequest",
    "ExportRequest",
    
    # Response models
    "GenerationResponse",
    "BatchGenerationResponse",
    "ValidationResponse", 
    "ExportResponse",
    "DocumentListResponse",
    "StatusResponse",
    "QuotaResponse",
    "JobResponse",
    
    # Configuration
    "GenerationConfig",
    "LayoutConfig",
    "ContentConfig", 
    "PrivacyConfig",
    "NoiseConfig",
    "RenderingConfig",
    "PerformanceConfig",
    
    # Factory and utility functions
    "create_generation_request",
    "create_batch_request",
    "get_config_preset",
    "create_custom_config",
    "get_default_config_for_document_type",
    
    # Convenience functions
    "generate_document",
    "generate_batch",
    "validate_document",
    "export_documents"
]


# Convenience functions for quick access
def generate_document(
    document_type,
    api_key=None,
    **kwargs
):
    """
    Quick document generation function.
    
    Args:
        document_type: Type of document to generate
        api_key: API key for authentication
        **kwargs: Additional generation parameters
    
    Returns:
        GenerationResponse
    """
    client = StructuredDocsClient(api_key=api_key)
    return client.generate_document(document_type, **kwargs)


def generate_batch(
    requests,
    api_key=None,
    **kwargs
):
    """
    Quick batch generation function.
    
    Args:
        requests: List of generation requests
        api_key: API key for authentication
        **kwargs: Additional batch parameters
    
    Returns:
        BatchGenerationResponse
    """
    client = StructuredDocsClient(api_key=api_key)
    return client.generate_batch(requests, **kwargs)


def validate_document(
    document_id=None,
    document_content=None,
    api_key=None,
    **kwargs
):
    """
    Quick document validation function.
    
    Args:
        document_id: ID of document to validate
        document_content: Document content to validate
        api_key: API key for authentication
        **kwargs: Additional validation parameters
    
    Returns:
        ValidationResponse
    """
    client = StructuredDocsClient(api_key=api_key)
    return client.validate_document(
        document_id=document_id,
        document_content=document_content,
        **kwargs
    )


def export_documents(
    document_ids,
    export_format,
    output_path,
    api_key=None,
    **kwargs
):
    """
    Quick document export function.
    
    Args:
        document_ids: List of document IDs to export
        export_format: Export format
        output_path: Output path
        api_key: API key for authentication
        **kwargs: Additional export parameters
    
    Returns:
        ExportResponse
    """
    client = StructuredDocsClient(api_key=api_key)
    return client.export_documents(
        document_ids=document_ids,
        export_format=export_format,
        output_path=output_path,
        **kwargs
    )


# SDK configuration
class SDKConfig:
    """Global SDK configuration"""
    
    def __init__(self):
        self.default_api_key = None
        self.default_base_url = "https://api.structuredocs.ai"
        self.default_timeout = 300
        self.enable_logging = True
        self.log_level = "INFO"
    
    def set_api_key(self, api_key: str):
        """Set default API key"""
        self.default_api_key = api_key
    
    def set_base_url(self, base_url: str):
        """Set default base URL"""
        self.default_base_url = base_url
    
    def set_timeout(self, timeout: int):
        """Set default timeout"""
        self.default_timeout = timeout


# Global configuration instance
config = SDKConfig()


def configure(
    api_key=None,
    base_url=None,
    timeout=None,
    enable_logging=None,
    log_level=None
):
    """
    Configure global SDK settings.
    
    Args:
        api_key: Default API key
        base_url: Default base URL
        timeout: Default timeout
        enable_logging: Enable/disable logging
        log_level: Logging level
    """
    if api_key:
        config.set_api_key(api_key)
    if base_url:
        config.set_base_url(base_url)
    if timeout:
        config.set_timeout(timeout)
    if enable_logging is not None:
        config.enable_logging = enable_logging
    if log_level:
        config.log_level = log_level


# Common document type shortcuts
class DocumentTypes:
    """Shortcuts for common document types"""
    ACADEMIC = DocumentType.ACADEMIC_PAPER
    BUSINESS = DocumentType.BUSINESS_FORM
    TECHNICAL = DocumentType.TECHNICAL_MANUAL
    FINANCIAL = DocumentType.FINANCIAL_REPORT
    LEGAL = DocumentType.LEGAL_DOCUMENT
    MEDICAL = DocumentType.MEDICAL_RECORD
    INVOICE = DocumentType.INVOICE
    RESUME = DocumentType.RESUME


# Common format shortcuts
class Formats:
    """Shortcuts for common document formats"""
    PDF = DocumentFormat.PDF
    DOCX = DocumentFormat.DOCX
    HTML = DocumentFormat.HTML
    MARKDOWN = DocumentFormat.MARKDOWN


# Quick configuration presets
class ConfigPresets:
    """Quick access to configuration presets"""
    
    @staticmethod
    def fast():
        """Fast generation preset"""
        return get_config_preset("development")
    
    @staticmethod
    def quality():
        """Quality generation preset"""
        return get_config_preset("production")
    
    @staticmethod
    def privacy():
        """Privacy-focused preset"""
        return get_config_preset("privacy")
    
    @staticmethod
    def default():
        """Default configuration"""
        return get_config_preset("default")


# Module initialization
def _initialize_sdk():
    """Initialize SDK with default settings"""
    import os
    
    # Try to load API key from environment
    api_key = os.getenv("STRUCTURED_DOCS_API_KEY")
    if api_key:
        config.set_api_key(api_key)
    
    # Try to load base URL from environment  
    base_url = os.getenv("STRUCTURED_DOCS_BASE_URL")
    if base_url:
        config.set_base_url(base_url)


# Initialize on import
_initialize_sdk()


# Helpful aliases
Client = StructuredDocsClient
AsyncClient = AsyncStructuredDocsClient
Types = DocumentTypes
Config = ConfigPresets