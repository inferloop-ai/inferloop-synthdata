"""
Response models for SDK API interactions.
Defines response structures for various SDK operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any

from pydantic import BaseModel, Field

from .document_types import DocumentType, DocumentFormat, DocumentMetadata, DocumentContent


class ResponseStatus(Enum):
    """Response status codes"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    IN_PROGRESS = "in_progress"
    QUEUED = "queued"
    CANCELLED = "cancelled"


class ErrorType(Enum):
    """Error type categories"""
    VALIDATION_ERROR = "validation_error"
    GENERATION_ERROR = "generation_error"
    STORAGE_ERROR = "storage_error"
    NETWORK_ERROR = "network_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    INTERNAL_ERROR = "internal_error"


class ErrorDetail(BaseModel):
    """Detailed error information"""
    error_type: ErrorType
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    trace_id: Optional[str] = None
    
    class Config:
        use_enum_values = True


class BaseResponse(BaseModel):
    """Base response model"""
    status: ResponseStatus
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None
    errors: List[ErrorDetail] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True
    
    @property
    def is_success(self) -> bool:
        """Check if response is successful"""
        return self.status == ResponseStatus.SUCCESS
    
    @property
    def is_failure(self) -> bool:
        """Check if response is a failure"""
        return self.status == ResponseStatus.FAILURE
    
    @property
    def has_errors(self) -> bool:
        """Check if response has errors"""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if response has warnings"""
        return len(self.warnings) > 0


class GenerationResponse(BaseResponse):
    """Document generation response"""
    document_id: Optional[str] = None
    document_type: Optional[DocumentType] = None
    document_format: Optional[DocumentFormat] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    metadata: Optional[DocumentMetadata] = None
    content: Optional[DocumentContent] = None
    
    # Generation metrics
    generation_time_seconds: Optional[float] = None
    quality_score: Optional[float] = None
    validation_results: Optional[Dict[str, Any]] = None
    
    class Config:
        use_enum_values = True


class BatchGenerationResponse(BaseResponse):
    """Batch document generation response"""
    batch_id: str
    total_requested: int
    successful_generations: int
    failed_generations: int
    documents: List[GenerationResponse] = Field(default_factory=list)
    
    # Batch metrics
    total_time_seconds: Optional[float] = None
    average_generation_time: Optional[float] = None
    throughput_docs_per_second: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requested == 0:
            return 0.0
        return self.successful_generations / self.total_requested
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage"""
        completed = self.successful_generations + self.failed_generations
        if self.total_requested == 0:
            return 0.0
        return (completed / self.total_requested) * 100


class ValidationResponse(BaseResponse):
    """Document validation response"""
    document_id: str
    validation_score: float = Field(ge=0.0, le=1.0)
    is_valid: bool
    validation_types: List[str] = Field(default_factory=list)
    
    # Validation results
    structural_validation: Optional[Dict[str, Any]] = None
    completeness_validation: Optional[Dict[str, Any]] = None
    semantic_validation: Optional[Dict[str, Any]] = None
    quality_validation: Optional[Dict[str, Any]] = None
    
    # Issues and recommendations
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # Metrics
    validation_time_seconds: Optional[float] = None


class ExportResponse(BaseResponse):
    """Document export response"""
    export_id: str
    export_format: str
    document_ids: List[str] = Field(default_factory=list)
    output_path: str
    
    # Export metrics
    exported_documents: int = 0
    failed_exports: int = 0
    total_file_size: Optional[int] = None
    export_time_seconds: Optional[float] = None
    
    @property
    def export_success_rate(self) -> float:
        """Calculate export success rate"""
        total = self.exported_documents + self.failed_exports
        if total == 0:
            return 0.0
        return self.exported_documents / total


class UploadResponse(BaseResponse):
    """File upload response"""
    file_id: str
    original_filename: str
    file_size: int
    file_type: str
    upload_path: str
    checksum: Optional[str] = None
    
    # Upload metrics
    upload_time_seconds: Optional[float] = None
    upload_speed_mbps: Optional[float] = None


class DownloadResponse(BaseResponse):
    """File download response"""
    file_id: str
    filename: str
    file_size: int
    file_type: str
    download_url: Optional[str] = None
    expires_at: Optional[datetime] = None
    
    # Download metrics
    download_time_seconds: Optional[float] = None


class StatusResponse(BaseResponse):
    """System status response"""
    service_name: str
    version: str
    uptime_seconds: float
    health_status: str
    
    # System metrics
    cpu_usage_percent: Optional[float] = None
    memory_usage_percent: Optional[float] = None
    disk_usage_percent: Optional[float] = None
    active_jobs: Optional[int] = None
    queue_size: Optional[int] = None
    
    # Feature availability
    features_enabled: List[str] = Field(default_factory=list)
    features_disabled: List[str] = Field(default_factory=list)


class MetricsResponse(BaseResponse):
    """System metrics response"""
    time_period: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    
    # Performance metrics
    total_documents_generated: Optional[int] = None
    average_generation_time: Optional[float] = None
    success_rate: Optional[float] = None
    error_rate: Optional[float] = None
    
    # Resource usage
    peak_memory_usage: Optional[float] = None
    average_cpu_usage: Optional[float] = None
    storage_usage: Optional[float] = None


class ConfigResponse(BaseResponse):
    """Configuration response"""
    config_version: str
    config_data: Dict[str, Any] = Field(default_factory=dict)
    last_updated: Optional[datetime] = None
    is_default: bool = True


class ListResponse(BaseResponse):
    """Generic list response"""
    items: List[Dict[str, Any]] = Field(default_factory=list)
    total_count: int = 0
    page: int = 1
    page_size: int = 50
    total_pages: int = 1
    
    # Filtering and sorting
    filters: Optional[Dict[str, Any]] = None
    sort_by: Optional[str] = None
    sort_order: str = "asc"
    
    @property
    def has_next_page(self) -> bool:
        """Check if there's a next page"""
        return self.page < self.total_pages
    
    @property
    def has_previous_page(self) -> bool:
        """Check if there's a previous page"""
        return self.page > 1


class DocumentListResponse(ListResponse):
    """Document list response"""
    items: List[GenerationResponse] = Field(default_factory=list)


class JobResponse(BaseResponse):
    """Asynchronous job response"""
    job_id: str
    job_type: str
    job_status: str
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Job progress
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    estimated_completion_time: Optional[datetime] = None
    
    # Job results
    result: Optional[Dict[str, Any]] = None
    result_url: Optional[str] = None
    
    @property
    def is_running(self) -> bool:
        """Check if job is currently running"""
        return self.job_status in ["running", "in_progress"]
    
    @property
    def is_completed(self) -> bool:
        """Check if job is completed"""
        return self.job_status in ["completed", "succeeded", "failed"]
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate job duration"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.now() - self.started_at).total_seconds()
        return None


class AuthResponse(BaseResponse):
    """Authentication response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    refresh_token: Optional[str] = None
    scope: Optional[str] = None
    
    # User information
    user_id: Optional[str] = None
    username: Optional[str] = None
    permissions: List[str] = Field(default_factory=list)


class QuotaResponse(BaseResponse):
    """Usage quota response"""
    user_id: str
    plan_type: str
    
    # Current usage
    documents_generated_today: int = 0
    documents_generated_month: int = 0
    storage_used_bytes: int = 0
    api_calls_today: int = 0
    
    # Limits
    daily_document_limit: int = 100
    monthly_document_limit: int = 1000
    storage_limit_bytes: int = 1073741824  # 1GB
    daily_api_limit: int = 1000
    
    # Calculated properties
    @property
    def daily_documents_remaining(self) -> int:
        """Calculate remaining daily documents"""
        return max(0, self.daily_document_limit - self.documents_generated_today)
    
    @property
    def monthly_documents_remaining(self) -> int:
        """Calculate remaining monthly documents"""
        return max(0, self.monthly_document_limit - self.documents_generated_month)
    
    @property
    def storage_usage_percentage(self) -> float:
        """Calculate storage usage percentage"""
        if self.storage_limit_bytes == 0:
            return 0.0
        return (self.storage_used_bytes / self.storage_limit_bytes) * 100
    
    @property
    def is_quota_exceeded(self) -> bool:
        """Check if any quota is exceeded"""
        return (
            self.documents_generated_today >= self.daily_document_limit or
            self.documents_generated_month >= self.monthly_document_limit or
            self.storage_used_bytes >= self.storage_limit_bytes or
            self.api_calls_today >= self.daily_api_limit
        )


class WebhookResponse(BaseResponse):
    """Webhook response"""
    webhook_id: str
    event_type: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    delivery_id: str
    attempt_number: int = 1
    max_attempts: int = 3
    
    # Delivery status
    delivered_at: Optional[datetime] = None
    response_status_code: Optional[int] = None
    response_body: Optional[str] = None
    
    @property
    def is_delivered(self) -> bool:
        """Check if webhook was delivered successfully"""
        return (
            self.delivered_at is not None and
            self.response_status_code is not None and
            200 <= self.response_status_code < 300
        )
    
    @property
    def should_retry(self) -> bool:
        """Check if webhook delivery should be retried"""
        return (
            not self.is_delivered and
            self.attempt_number < self.max_attempts
        )


# Response factory functions
def create_success_response(
    response_type: type = BaseResponse,
    message: str = "Operation completed successfully",
    **kwargs
) -> BaseResponse:
    """Create a success response"""
    return response_type(
        status=ResponseStatus.SUCCESS,
        message=message,
        **kwargs
    )


def create_error_response(
    error_type: ErrorType,
    message: str,
    error_code: str = "UNKNOWN_ERROR",
    response_type: type = BaseResponse,
    **kwargs
) -> BaseResponse:
    """Create an error response"""
    error = ErrorDetail(
        error_type=error_type,
        error_code=error_code,
        message=message
    )
    
    return response_type(
        status=ResponseStatus.FAILURE,
        message=message,
        errors=[error],
        **kwargs
    )


def create_validation_error_response(
    message: str,
    validation_errors: List[str] = None,
    **kwargs
) -> BaseResponse:
    """Create a validation error response"""
    error = ErrorDetail(
        error_type=ErrorType.VALIDATION_ERROR,
        error_code="VALIDATION_FAILED",
        message=message,
        details={"validation_errors": validation_errors or []}
    )
    
    return BaseResponse(
        status=ResponseStatus.FAILURE,
        message=message,
        errors=[error],
        **kwargs
    )