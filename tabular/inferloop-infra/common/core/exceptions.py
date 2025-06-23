"""Infrastructure-related exceptions."""

from typing import Optional, Dict, Any


class InfrastructureError(Exception):
    """Base exception for infrastructure operations."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize infrastructure error."""
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ResourceNotFoundError(InfrastructureError):
    """Resource not found exception."""
    
    def __init__(self, resource_id: str, resource_type: Optional[str] = None):
        """Initialize resource not found error."""
        message = f"Resource not found: {resource_id}"
        if resource_type:
            message = f"{resource_type} not found: {resource_id}"
        super().__init__(
            message,
            error_code="RESOURCE_NOT_FOUND",
            details={"resource_id": resource_id, "resource_type": resource_type},
        )


class ResourceCreationError(InfrastructureError):
    """Resource creation failed exception."""
    
    def __init__(
        self, resource_type: str, reason: str, details: Optional[Dict[str, Any]] = None
    ):
        """Initialize resource creation error."""
        super().__init__(
            f"Failed to create {resource_type}: {reason}",
            error_code="RESOURCE_CREATION_FAILED",
            details={"resource_type": resource_type, "reason": reason, **(details or {})},
        )


class AuthenticationError(InfrastructureError):
    """Authentication failed exception."""
    
    def __init__(self, provider: str, reason: str):
        """Initialize authentication error."""
        super().__init__(
            f"Authentication failed for {provider}: {reason}",
            error_code="AUTHENTICATION_FAILED",
            details={"provider": provider, "reason": reason},
        )


class ConfigurationError(InfrastructureError):
    """Configuration error exception."""
    
    def __init__(self, field: str, reason: str):
        """Initialize configuration error."""
        super().__init__(
            f"Configuration error for {field}: {reason}",
            error_code="CONFIGURATION_ERROR",
            details={"field": field, "reason": reason},
        )


class QuotaExceededError(InfrastructureError):
    """Quota exceeded exception."""
    
    def __init__(self, resource_type: str, limit: int, requested: int):
        """Initialize quota exceeded error."""
        super().__init__(
            f"Quota exceeded for {resource_type}: requested {requested}, limit {limit}",
            error_code="QUOTA_EXCEEDED",
            details={
                "resource_type": resource_type,
                "limit": limit,
                "requested": requested,
            },
        )


class NetworkError(InfrastructureError):
    """Network-related exception."""
    
    def __init__(self, operation: str, reason: str):
        """Initialize network error."""
        super().__init__(
            f"Network operation '{operation}' failed: {reason}",
            error_code="NETWORK_ERROR",
            details={"operation": operation, "reason": reason},
        )


class StorageError(InfrastructureError):
    """Storage-related exception."""
    
    def __init__(self, operation: str, bucket: str, reason: str):
        """Initialize storage error."""
        super().__init__(
            f"Storage operation '{operation}' failed for bucket '{bucket}': {reason}",
            error_code="STORAGE_ERROR",
            details={"operation": operation, "bucket": bucket, "reason": reason},
        )


class DeploymentError(InfrastructureError):
    """Deployment-related exception."""
    
    def __init__(self, application: str, reason: str, logs: Optional[str] = None):
        """Initialize deployment error."""
        super().__init__(
            f"Deployment failed for application '{application}': {reason}",
            error_code="DEPLOYMENT_FAILED",
            details={"application": application, "reason": reason, "logs": logs},
        )


class MonitoringError(InfrastructureError):
    """Monitoring-related exception."""
    
    def __init__(self, operation: str, reason: str):
        """Initialize monitoring error."""
        super().__init__(
            f"Monitoring operation '{operation}' failed: {reason}",
            error_code="MONITORING_ERROR",
            details={"operation": operation, "reason": reason},
        )


class CostEstimationError(InfrastructureError):
    """Cost estimation exception."""
    
    def __init__(self, reason: str):
        """Initialize cost estimation error."""
        super().__init__(
            f"Cost estimation failed: {reason}",
            error_code="COST_ESTIMATION_ERROR",
            details={"reason": reason},
        )