"""
SDK models module for structured document synthetic data generation.
Provides comprehensive data models and type definitions for the SDK.
"""

from typing import Dict, List, Optional, Union, Any

# Document types and core models
from .document_types import (
    DocumentFormat,
    DocumentType,
    LayoutComplexity,
    ContentQuality,
    Language,
    BoundingBox,
    DocumentElement,
    DocumentMetadata,
    PageSize,
    DocumentContent,
    GenerationRequest,
    DocumentResponse,
    BatchGenerationRequest,
    BatchGenerationResponse,
    ValidationRequest,
    ValidationResponse,
    ExportRequest,
    ExportResponse,
    DOCUMENT_TEMPLATES,
    get_document_template,
    create_generation_request,
    create_batch_request
)

# Generation configuration models
from .generation_config import (
    GenerationMode,
    PrivacyLevel,
    NoiseType,
    RenderingQuality,
    LayoutConfig,
    ContentConfig,
    PrivacyConfig,
    NoiseConfig,
    RenderingConfig,
    PerformanceConfig,
    OutputConfig,
    ValidationConfig,
    GenerationConfig,
    CONFIG_PRESETS,
    get_config_preset,
    create_custom_config
)

# Response models
from .response_models import (
    ResponseStatus,
    ErrorType,
    ErrorDetail,
    BaseResponse,
    GenerationResponse,
    BatchGenerationResponse,
    ValidationResponse,
    ExportResponse,
    UploadResponse,
    DownloadResponse,
    StatusResponse,
    MetricsResponse,
    ConfigResponse,
    ListResponse,
    DocumentListResponse,
    JobResponse,
    AuthResponse,
    QuotaResponse,
    WebhookResponse,
    create_success_response,
    create_error_response,
    create_validation_error_response
)

# Module version
__version__ = "1.0.0"

# Public API
__all__ = [
    # Document Types
    "DocumentFormat",
    "DocumentType",
    "LayoutComplexity", 
    "ContentQuality",
    "Language",
    "BoundingBox",
    "DocumentElement",
    "DocumentMetadata",
    "PageSize",
    "DocumentContent",
    "GenerationRequest",
    "DocumentResponse",
    "BatchGenerationRequest",
    "BatchGenerationResponse",
    "ValidationRequest",
    "ValidationResponse",
    "ExportRequest",
    "ExportResponse",
    "DOCUMENT_TEMPLATES",
    "get_document_template",
    "create_generation_request",
    "create_batch_request",
    
    # Generation Configuration
    "GenerationMode",
    "PrivacyLevel",
    "NoiseType",
    "RenderingQuality",
    "LayoutConfig",
    "ContentConfig",
    "PrivacyConfig",
    "NoiseConfig",
    "RenderingConfig",
    "PerformanceConfig",
    "OutputConfig",
    "ValidationConfig",
    "GenerationConfig",
    "CONFIG_PRESETS",
    "get_config_preset",
    "create_custom_config",
    
    # Response Models
    "ResponseStatus",
    "ErrorType",
    "ErrorDetail",
    "BaseResponse",
    "GenerationResponse",
    "BatchGenerationResponse",
    "ValidationResponse",
    "ExportResponse",
    "UploadResponse",
    "DownloadResponse",
    "StatusResponse",
    "MetricsResponse",
    "ConfigResponse",
    "ListResponse",
    "DocumentListResponse",
    "JobResponse",
    "AuthResponse",
    "QuotaResponse",
    "WebhookResponse",
    "create_success_response",
    "create_error_response",
    "create_validation_error_response",
    
    # Factory functions
    "create_document_request",
    "create_batch_request",
    "validate_request_data"
]


def create_document_request(
    document_type: Union[str, DocumentType],
    document_format: Union[str, DocumentFormat] = DocumentFormat.PDF,
    count: int = 1,
    config: Optional[Union[Dict[str, Any], GenerationConfig]] = None,
    **kwargs
) -> GenerationRequest:
    """
    Create a document generation request with validation.
    
    Args:
        document_type: Type of document to generate
        document_format: Output format for the document
        count: Number of documents to generate
        config: Generation configuration
        **kwargs: Additional request parameters
    
    Returns:
        Validated GenerationRequest object
    """
    # Convert string enums to enum objects
    if isinstance(document_type, str):
        document_type = DocumentType(document_type)
    if isinstance(document_format, str):
        document_format = DocumentFormat(document_format)
    
    # Create or convert config
    if config is None:
        config = GenerationConfig()
    elif isinstance(config, dict):
        config = GenerationConfig(**config)
    
    # Apply document type defaults to config
    config.apply_document_type_defaults(document_type)
    
    # Create request
    request = GenerationRequest(
        document_type=document_type,
        document_format=document_format,
        count=count,
        **kwargs
    )
    
    return request


def validate_request_data(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Validate request data and return validation errors.
    
    Args:
        data: Request data to validate
    
    Returns:
        Dictionary of validation errors by field
    """
    errors = {}
    
    # Validate required fields
    required_fields = ["document_type"]
    for field in required_fields:
        if field not in data or data[field] is None:
            if field not in errors:
                errors[field] = []
            errors[field].append(f"Field '{field}' is required")
    
    # Validate document_type
    if "document_type" in data:
        try:
            DocumentType(data["document_type"])
        except ValueError:
            if "document_type" not in errors:
                errors["document_type"] = []
            errors["document_type"].append(f"Invalid document type: {data['document_type']}")
    
    # Validate document_format
    if "document_format" in data:
        try:
            DocumentFormat(data["document_format"])
        except ValueError:
            if "document_format" not in errors:
                errors["document_format"] = []
            errors["document_format"].append(f"Invalid document format: {data['document_format']}")
    
    # Validate count
    if "count" in data:
        count = data["count"]
        if not isinstance(count, int) or count < 1 or count > 1000:
            if "count" not in errors:
                errors["count"] = []
            errors["count"].append("Count must be an integer between 1 and 1000")
    
    # Validate page_size if provided
    if "page_size" in data and data["page_size"]:
        page_size = data["page_size"]
        if isinstance(page_size, dict):
            if "width" not in page_size or "height" not in page_size:
                if "page_size" not in errors:
                    errors["page_size"] = []
                errors["page_size"].append("Page size must include width and height")
            else:
                try:
                    width = float(page_size["width"])
                    height = float(page_size["height"])
                    if width <= 0 or height <= 0:
                        if "page_size" not in errors:
                            errors["page_size"] = []
                        errors["page_size"].append("Page dimensions must be positive")
                except (ValueError, TypeError):
                    if "page_size" not in errors:
                        errors["page_size"] = []
                    errors["page_size"].append("Page dimensions must be numeric")
    
    return errors


# Model registry for dynamic access
MODEL_REGISTRY = {
    "GenerationRequest": GenerationRequest,
    "BatchGenerationRequest": BatchGenerationRequest,
    "ValidationRequest": ValidationRequest,
    "ExportRequest": ExportRequest,
    "GenerationResponse": GenerationResponse,
    "BatchGenerationResponse": BatchGenerationResponse,
    "ValidationResponse": ValidationResponse,
    "ExportResponse": ExportResponse,
    "GenerationConfig": GenerationConfig,
    "DocumentMetadata": DocumentMetadata,
    "DocumentContent": DocumentContent
}


def get_model_by_name(model_name: str) -> type:
    """
    Get model class by name.
    
    Args:
        model_name: Name of the model class
    
    Returns:
        Model class
    
    Raises:
        ValueError: If model name is not found
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name]


def create_model_instance(model_name: str, data: Dict[str, Any]) -> Any:
    """
    Create model instance from data.
    
    Args:
        model_name: Name of the model class
        data: Data to create instance from
    
    Returns:
        Model instance
    """
    model_class = get_model_by_name(model_name)
    return model_class(**data)


# Type aliases for convenience
GenerationRequestType = GenerationRequest
BatchRequestType = BatchGenerationRequest
ValidationRequestType = ValidationRequest
ExportRequestType = ExportRequest
ConfigType = GenerationConfig

# Default configurations for common use cases
DEFAULT_CONFIGS = {
    "academic": GenerationConfig.quality_mode(),
    "business": GenerationConfig(),
    "medical": GenerationConfig.privacy_focused(),
    "legal": GenerationConfig.privacy_focused(),
    "financial": GenerationConfig.quality_mode(),
    "forms": GenerationConfig.fast_mode()
}


def get_default_config_for_document_type(document_type: Union[str, DocumentType]) -> GenerationConfig:
    """
    Get default configuration for a document type.
    
    Args:
        document_type: Document type
    
    Returns:
        Default configuration for the document type
    """
    if isinstance(document_type, str):
        document_type = DocumentType(document_type)
    
    # Map document types to default configs
    type_config_map = {
        DocumentType.ACADEMIC_PAPER: "academic",
        DocumentType.BUSINESS_FORM: "business",
        DocumentType.MEDICAL_RECORD: "medical",
        DocumentType.LEGAL_DOCUMENT: "legal",
        DocumentType.FINANCIAL_REPORT: "financial",
        DocumentType.INVOICE: "forms",
        DocumentType.RESUME: "business"
    }
    
    config_name = type_config_map.get(document_type, "business")
    config = DEFAULT_CONFIGS[config_name].copy()
    config.apply_document_type_defaults(document_type)
    
    return config