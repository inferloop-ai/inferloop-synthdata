"""
Custom exceptions for Structured Documents Synthetic Data Generator
"""

from typing import Optional, Dict, Any, List


class StructuredDocsSynthError(Exception):
    """Base exception for all structured docs synth errors"""
    
    def __init__(self, 
                 message: str, 
                 error_code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary"""
        return {
            'error': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'details': self.details
        }


# Configuration Exceptions
class ConfigurationError(StructuredDocsSynthError):
    """Configuration related errors"""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Invalid configuration parameters"""
    pass


class MissingConfigurationError(ConfigurationError):
    """Missing required configuration"""
    pass


# Document Generation Exceptions
class DocumentGenerationError(StructuredDocsSynthError):
    """Document generation related errors"""
    pass


class TemplateNotFoundError(DocumentGenerationError):
    """Template file not found"""
    pass


class TemplateRenderingError(DocumentGenerationError):
    """Error during template rendering"""
    pass


class UnsupportedDocumentTypeError(DocumentGenerationError):
    """Unsupported document type"""
    pass


class UnsupportedFormatError(DocumentGenerationError):
    """Unsupported output format"""
    pass


class DocumentGenerationTimeoutError(DocumentGenerationError):
    """Document generation timeout"""
    pass


# Validation Exceptions
class ValidationError(StructuredDocsSynthError):
    """Validation related errors"""
    
    def __init__(self, 
                 message: str, 
                 validation_errors: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(message, **kwargs)
        self.validation_errors = validation_errors or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary with validation errors"""
        result = super().to_dict()
        result['validation_errors'] = self.validation_errors
        return result


class SchemaValidationError(ValidationError):
    """Schema validation errors"""
    pass


class ContentValidationError(ValidationError):
    """Content validation errors"""
    pass


class QualityValidationError(ValidationError):
    """Quality validation errors"""
    pass


# Privacy and Compliance Exceptions
class PrivacyError(StructuredDocsSynthError):
    """Privacy related errors"""
    pass


class PIIDetectionError(PrivacyError):
    """PII detection errors"""
    pass


class PIIMaskingError(PrivacyError):
    """PII masking errors"""
    pass


class ComplianceError(StructuredDocsSynthError):
    """Compliance related errors"""
    pass


class GDPRComplianceError(ComplianceError):
    """GDPR compliance violations"""
    pass


class HIPAAComplianceError(ComplianceError):
    """HIPAA compliance violations"""
    pass


class PCIDSSComplianceError(ComplianceError):
    """PCI-DSS compliance violations"""
    pass


class SOXComplianceError(ComplianceError):
    """SOX compliance violations"""
    pass


class DifferentialPrivacyError(PrivacyError):
    """Differential privacy errors"""
    pass


# Storage and I/O Exceptions
class StorageError(StructuredDocsSynthError):
    """Storage related errors"""
    pass


class FileNotFoundError(StorageError):
    """File not found"""
    pass


class FileWriteError(StorageError):
    """File write errors"""
    pass


class FileReadError(StorageError):
    """File read errors"""
    pass


class CloudStorageError(StorageError):
    """Cloud storage errors"""
    pass


class S3Error(CloudStorageError):
    """AWS S3 errors"""
    pass


class GCSError(CloudStorageError):
    """Google Cloud Storage errors"""
    pass


class AzureStorageError(CloudStorageError):
    """Azure storage errors"""
    pass


# API Exceptions
class APIError(StructuredDocsSynthError):
    """API related errors"""
    
    def __init__(self, 
                 message: str, 
                 status_code: int = 500,
                 **kwargs):
        super().__init__(message, **kwargs)
        self.status_code = status_code


class AuthenticationError(APIError):
    """Authentication errors"""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, status_code=401, **kwargs)


class AuthorizationError(APIError):
    """Authorization errors"""
    
    def __init__(self, message: str = "Access denied", **kwargs):
        super().__init__(message, status_code=403, **kwargs)


class RateLimitError(APIError):
    """Rate limiting errors"""
    
    def __init__(self, message: str = "Rate limit exceeded", **kwargs):
        super().__init__(message, status_code=429, **kwargs)


class InvalidRequestError(APIError):
    """Invalid request errors"""
    
    def __init__(self, message: str = "Invalid request", **kwargs):
        super().__init__(message, status_code=400, **kwargs)


class ResourceNotFoundError(APIError):
    """Resource not found errors"""
    
    def __init__(self, message: str = "Resource not found", **kwargs):
        super().__init__(message, status_code=404, **kwargs)


# Processing Exceptions
class ProcessingError(StructuredDocsSynthError):
    """Processing related errors"""
    pass


class OCRError(ProcessingError):
    """OCR processing errors"""
    pass


class NLPError(ProcessingError):
    """NLP processing errors"""
    pass


class AnnotationError(ProcessingError):
    """Annotation errors"""
    pass


class BatchProcessingError(ProcessingError):
    """Batch processing errors"""
    pass


# External Service Exceptions
class ExternalServiceError(StructuredDocsSynthError):
    """External service errors"""
    pass


class ExternalAPIError(ExternalServiceError):
    """External API errors"""
    pass


class ServiceUnavailableError(ExternalServiceError):
    """Service unavailable errors"""
    pass


class ServiceTimeoutError(ExternalServiceError):
    """Service timeout errors"""
    pass


# Database Exceptions
class DatabaseError(StructuredDocsSynthError):
    """Database related errors"""
    pass


class ConnectionError(DatabaseError):
    """Database connection errors"""
    pass


class QueryError(DatabaseError):
    """Database query errors"""
    pass


class MigrationError(DatabaseError):
    """Database migration errors"""
    pass


# Utility functions for exception handling
def handle_exception(func):
    """Decorator for consistent exception handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except StructuredDocsSynthError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            # Wrap other exceptions
            raise StructuredDocsSynthError(
                message=f"Unexpected error in {func.__name__}: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                details={'original_exception': str(e), 'function': func.__name__}
            ) from e
    return wrapper


def create_validation_error(field: str, 
                          value: Any, 
                          expected: str, 
                          error_code: Optional[str] = None) -> ValidationError:
    """Create a standardized validation error"""
    message = f"Invalid value for field '{field}': {value}. Expected: {expected}"
    return ValidationError(
        message=message,
        error_code=error_code or "VALIDATION_FAILED",
        details={
            'field': field,
            'value': str(value),
            'expected': expected
        }
    )


def create_compliance_error(regulation: str, 
                          violation: str, 
                          document_id: Optional[str] = None) -> ComplianceError:
    """Create a standardized compliance error"""
    message = f"{regulation} compliance violation: {violation}"
    details = {'regulation': regulation, 'violation': violation}
    
    if document_id:
        details['document_id'] = document_id
    
    # Return specific compliance error based on regulation
    if regulation.upper() == 'GDPR':
        return GDPRComplianceError(message, details=details)
    elif regulation.upper() == 'HIPAA':
        return HIPAAComplianceError(message, details=details)
    elif regulation.upper() == 'PCI-DSS':
        return PCIDSSComplianceError(message, details=details)
    elif regulation.upper() == 'SOX':
        return SOXComplianceError(message, details=details)
    else:
        return ComplianceError(message, details=details)