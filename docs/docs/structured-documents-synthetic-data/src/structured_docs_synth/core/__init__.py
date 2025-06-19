"""
Core framework for Structured Documents Synthetic Data Generator
"""

from .config import (
    Config,
    get_config,
    reload_config,
    get_document_type_config,
    list_document_types,
    DOCUMENT_TYPES
)

from .logging import (
    get_logger,
    configure_logging,
    log_document_generation,
    log_api_request,
    log_privacy_event,
    log_validation_result
)

from .exceptions import (
    StructuredDocsSynthError,
    ConfigurationError,
    DocumentGenerationError,
    TemplateNotFoundError,
    TemplateRenderingError,
    ValidationError,
    PrivacyError,
    ComplianceError,
    APIError,
    StorageError,
    ProcessingError,
    handle_exception,
    create_validation_error,
    create_compliance_error
)

__all__ = [
    # Config
    'Config',
    'get_config',
    'reload_config',
    'get_document_type_config',
    'list_document_types',
    'DOCUMENT_TYPES',
    
    # Logging
    'get_logger',
    'configure_logging',
    'log_document_generation',
    'log_api_request',
    'log_privacy_event',
    'log_validation_result',
    
    # Exceptions
    'StructuredDocsSynthError',
    'ConfigurationError',
    'DocumentGenerationError',
    'TemplateNotFoundError',
    'TemplateRenderingError',
    'ValidationError',
    'PrivacyError',
    'ComplianceError',
    'APIError',
    'StorageError',
    'ProcessingError',
    'handle_exception',
    'create_validation_error',
    'create_compliance_error',
]