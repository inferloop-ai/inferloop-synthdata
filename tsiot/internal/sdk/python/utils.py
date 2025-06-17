"""
Utility functions and classes for the TSIOT Python SDK.

This module contains common utility functions, exception classes, logging configuration,
and helper functions used throughout the SDK.
"""

import logging
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import json


class TSIOTError(Exception):
    """Base exception class for TSIOT SDK errors."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
        self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "traceback": traceback.format_exc()
        }
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.error_code}: {self.message} (Details: {self.details})"
        return f"{self.error_code}: {self.message}"


class ValidationError(TSIOTError):
    """Exception raised when data validation fails."""
    
    def __init__(self, message: str, field: str = None, value: Any = None, **kwargs):
        details = kwargs.get("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        super().__init__(message, "VALIDATION_ERROR", details)


class GenerationError(TSIOTError):
    """Exception raised when data generation fails."""
    
    def __init__(self, message: str, generator_type: str = None, **kwargs):
        details = kwargs.get("details", {})
        if generator_type:
            details["generator_type"] = generator_type
        super().__init__(message, "GENERATION_ERROR", details)


class AnalyticsError(TSIOTError):
    """Exception raised when analytics operations fail."""
    
    def __init__(self, message: str, analysis_type: str = None, **kwargs):
        details = kwargs.get("details", {})
        if analysis_type:
            details["analysis_type"] = analysis_type
        super().__init__(message, "ANALYTICS_ERROR", details)


class NetworkError(TSIOTError):
    """Exception raised when network operations fail."""
    
    def __init__(self, message: str, status_code: int = None, response_body: str = None, **kwargs):
        details = kwargs.get("details", {})
        if status_code:
            details["status_code"] = status_code
        if response_body:
            details["response_body"] = response_body
        super().__init__(message, "NETWORK_ERROR", details)


class AuthenticationError(TSIOTError):
    """Exception raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, "AUTHENTICATION_ERROR", kwargs.get("details", {}))


class RateLimitError(TSIOTError):
    """Exception raised when rate limits are exceeded."""
    
    def __init__(self, message: str, retry_after: int = None, **kwargs):
        details = kwargs.get("details", {})
        if retry_after:
            details["retry_after"] = retry_after
        super().__init__(message, "RATE_LIMIT_ERROR", details)


class TimeoutError(TSIOTError):
    """Exception raised when operations timeout."""
    
    def __init__(self, message: str, timeout_seconds: float = None, **kwargs):
        details = kwargs.get("details", {})
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        super().__init__(message, "TIMEOUT_ERROR", details)


# Logging configuration
def configure_logging(
    level: str = "INFO",
    format_string: str = None,
    include_timestamp: bool = True,
    include_level: bool = True,
    include_name: bool = True
) -> logging.Logger:
    """
    Configure logging for the TSIOT SDK.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        include_timestamp: Whether to include timestamp in logs
        include_level: Whether to include log level in logs
        include_name: Whether to include logger name in logs
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("tsiot")
    
    # Set log level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    
    # Create formatter
    if format_string is None:
        format_parts = []
        if include_timestamp:
            format_parts.append("%(asctime)s")
        if include_level:
            format_parts.append("%(levelname)s")
        if include_name:
            format_parts.append("%(name)s")
        format_parts.append("%(message)s")
        format_string = " - ".join(format_parts)
    
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger


# Timestamp utilities
def format_timestamp(dt: datetime, include_timezone: bool = True) -> str:
    """
    Format a datetime object as an ISO 8601 string.
    
    Args:
        dt: Datetime object to format
        include_timezone: Whether to include timezone information
    
    Returns:
        ISO 8601 formatted timestamp string
    """
    if dt.tzinfo is None and include_timezone:
        dt = dt.replace(tzinfo=timezone.utc)
    
    if include_timezone:
        return dt.isoformat()
    else:
        return dt.replace(tzinfo=None).isoformat()


def parse_timestamp(timestamp_str: str) -> datetime:
    """
    Parse an ISO 8601 timestamp string to a datetime object.
    
    Args:
        timestamp_str: ISO 8601 timestamp string
    
    Returns:
        Parsed datetime object
    
    Raises:
        ValidationError: If timestamp format is invalid
    """
    try:
        # Try parsing with timezone info
        if '+' in timestamp_str or timestamp_str.endswith('Z'):
            if timestamp_str.endswith('Z'):
                timestamp_str = timestamp_str[:-1] + '+00:00'
            return datetime.fromisoformat(timestamp_str)
        else:
            # Parse without timezone and assume UTC
            dt = datetime.fromisoformat(timestamp_str)
            return dt.replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise ValidationError(f"Invalid timestamp format: {timestamp_str}", details={"original_error": str(e)})


def current_timestamp() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


# Data validation utilities
def validate_positive_number(value: Union[int, float], name: str) -> Union[int, float]:
    """Validate that a value is a positive number."""
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number", field=name, value=value)
    if value <= 0:
        raise ValidationError(f"{name} must be positive", field=name, value=value)
    return value


def validate_non_negative_number(value: Union[int, float], name: str) -> Union[int, float]:
    """Validate that a value is a non-negative number."""
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number", field=name, value=value)
    if value < 0:
        raise ValidationError(f"{name} must be non-negative", field=name, value=value)
    return value


def validate_list_not_empty(value: List[Any], name: str) -> List[Any]:
    """Validate that a list is not empty."""
    if not isinstance(value, list):
        raise ValidationError(f"{name} must be a list", field=name, value=type(value).__name__)
    if len(value) == 0:
        raise ValidationError(f"{name} cannot be empty", field=name)
    return value


def validate_string_not_empty(value: str, name: str) -> str:
    """Validate that a string is not empty."""
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a string", field=name, value=type(value).__name__)
    if len(value.strip()) == 0:
        raise ValidationError(f"{name} cannot be empty", field=name)
    return value.strip()


def validate_enum_value(value: Any, enum_class: type, name: str) -> Enum:
    """Validate that a value is a valid enum member."""
    if isinstance(value, enum_class):
        return value
    
    if isinstance(value, str):
        try:
            return enum_class(value)
        except ValueError:
            valid_values = [e.value for e in enum_class]
            raise ValidationError(
                f"{name} must be one of {valid_values}",
                field=name,
                value=value,
                details={"valid_values": valid_values}
            )
    
    raise ValidationError(f"{name} must be a valid {enum_class.__name__}", field=name, value=value)


# JSON utilities
def safe_json_dumps(obj: Any, indent: int = None) -> str:
    """Safely serialize object to JSON string."""
    try:
        return json.dumps(obj, indent=indent, default=str)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"Failed to serialize object to JSON: {str(e)}")


def safe_json_loads(json_str: str) -> Any:
    """Safely deserialize JSON string to object."""
    try:
        return json.loads(json_str)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"Failed to parse JSON string: {str(e)}")


# HTTP utilities
def build_query_params(params: Dict[str, Any]) -> str:
    """Build query parameter string from dictionary."""
    from urllib.parse import urlencode
    
    # Filter out None values and convert everything to strings
    filtered_params = {}
    for key, value in params.items():
        if value is not None:
            if isinstance(value, (list, tuple)):
                # Handle array parameters
                for i, item in enumerate(value):
                    filtered_params[f"{key}[{i}]"] = str(item)
            elif isinstance(value, bool):
                # Convert boolean to lowercase string
                filtered_params[key] = str(value).lower()
            else:
                filtered_params[key] = str(value)
    
    return urlencode(filtered_params)


def parse_content_type(content_type: str) -> tuple[str, Dict[str, str]]:
    """Parse Content-Type header into media type and parameters."""
    parts = content_type.split(';')
    media_type = parts[0].strip().lower()
    
    params = {}
    for part in parts[1:]:
        if '=' in part:
            key, value = part.split('=', 1)
            params[key.strip().lower()] = value.strip().strip('"')
    
    return media_type, params


# Retry utilities
import time
import random
from typing import Callable, TypeVar

T = TypeVar('T')

def exponential_backoff_retry(
    func: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
) -> T:
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delay
    
    Returns:
        Result of the function call
    
    Raises:
        The last exception raised by the function
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            
            if attempt == max_retries:
                break
            
            # Calculate delay
            delay = min(base_delay * (exponential_base ** attempt), max_delay)
            
            if jitter:
                delay *= (0.5 + random.random() * 0.5)  # Add ±25% jitter
            
            time.sleep(delay)
    
    raise last_exception


# Performance utilities
class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "Operation", logger: logging.Logger = None):
        self.name = name
        self.logger = logger or logging.getLogger("tsiot")
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        self.logger.debug(f"{self.name} completed in {duration:.3f} seconds")
    
    @property
    def duration(self) -> Optional[float]:
        """Get the duration of the timed operation."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


# Batch processing utilities
def batch_items(items: List[T], batch_size: int) -> List[List[T]]:
    """Split a list into batches of specified size."""
    if batch_size <= 0:
        raise ValidationError("Batch size must be positive", field="batch_size", value=batch_size)
    
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    
    return batches


# Memory utilities
def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent()
        }
    except ImportError:
        return {"error": "psutil not available"}


# Initialize default logger
_default_logger = configure_logging()

def get_logger(name: str = "tsiot") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)