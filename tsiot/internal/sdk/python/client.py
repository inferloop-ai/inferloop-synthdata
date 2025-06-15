"""
TSIOT Python SDK Client

This module provides the main client classes for interacting with the TSIOT service.
It includes both synchronous and asynchronous clients with comprehensive error handling,
authentication, and retry logic.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from urllib.parse import urljoin, urlparse
import json
import logging

try:
    import httpx
except ImportError:
    raise ImportError("httpx is required. Install it with: pip install httpx")

from .timeseries import TimeSeries, DataPoint, TimeSeriesMetadata, DataFormat
from .utils import (
    TSIOTError,
    NetworkError,
    AuthenticationError,
    RateLimitError,
    TimeoutError,
    ValidationError,
    validate_string_not_empty,
    validate_positive_number,
    exponential_backoff_retry,
    build_query_params,
    Timer,
    get_logger
)


class TSIOTClient:
    """
    Synchronous client for the TSIOT service.
    
    This client provides methods for generating, validating, and analyzing time series data.
    It handles authentication, retries, and error handling automatically.
    
    Example:
        ```python
        client = TSIOTClient(
            base_url="http://localhost:8080",
            api_key="your-api-key"
        )
        
        # Generate time series data
        request = {
            "type": "arima",
            "length": 1000,
            "parameters": {
                "ar_params": [0.5, -0.3],
                "ma_params": [0.2]
            }
        }
        
        time_series = client.generate(request)
        print(f"Generated {len(time_series)} data points")
        ```
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: str = None,
        jwt_token: str = None,
        timeout: float = 30.0,
        retries: int = 3,
        verify_ssl: bool = True,
        user_agent: str = None,
        logger: logging.Logger = None
    ):
        """
        Initialize the TSIOT client.
        
        Args:
            base_url: Base URL of the TSIOT service
            api_key: API key for authentication (optional)
            jwt_token: JWT token for authentication (optional)
            timeout: Request timeout in seconds
            retries: Number of retry attempts for failed requests
            verify_ssl: Whether to verify SSL certificates
            user_agent: Custom user agent string
            logger: Logger instance
        """
        self.base_url = validate_string_not_empty(base_url.rstrip('/'), "base_url")
        self.api_key = api_key
        self.jwt_token = jwt_token
        self.timeout = validate_positive_number(timeout, "timeout")
        self.retries = max(0, int(retries))
        self.verify_ssl = verify_ssl
        self.logger = logger or get_logger(__name__)
        
        # Validate that we have some form of authentication
        if not self.api_key and not self.jwt_token:
            self.logger.warning("No authentication provided. Some endpoints may not be accessible.")
        
        # Set up HTTP client
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": user_agent or f"tsiot-python-sdk/1.0.0"
        }
        
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        elif self.jwt_token:
            headers["Authorization"] = f"Bearer {self.jwt_token}"
        
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
            verify=self.verify_ssl
        )
        
        self.logger.info(f"Initialized TSIOT client for {self.base_url}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def close(self):
        """Close the HTTP client."""
        if hasattr(self, '_client'):
            self._client.close()
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Dict[str, Any] = None,
        params: Dict[str, Any] = None,
        timeout: float = None
    ) -> Dict[str, Any]:
        """
        Make an HTTP request with error handling and retries.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters
            timeout: Request timeout override
        
        Returns:
            Response data as dictionary
        
        Raises:
            TSIOTError: Various error types based on response
        """
        url = endpoint
        request_timeout = timeout or self.timeout
        
        def make_single_request():
            with Timer(f"{method} {url}", self.logger):
                try:
                    response = self._client.request(
                        method=method,
                        url=url,
                        json=data,
                        params=params,
                        timeout=request_timeout
                    )
                    
                    # Handle different response status codes
                    if response.status_code == 200:
                        return response.json()
                    elif response.status_code == 201:
                        return response.json()
                    elif response.status_code == 204:
                        return {}
                    elif response.status_code == 400:
                        error_data = response.json() if response.content else {}
                        raise ValidationError(
                            error_data.get("message", "Bad request"),
                            details=error_data
                        )
                    elif response.status_code == 401:
                        raise AuthenticationError("Authentication failed")
                    elif response.status_code == 403:
                        raise AuthenticationError("Access forbidden")
                    elif response.status_code == 404:
                        raise NetworkError(f"Endpoint not found: {url}", status_code=404)
                    elif response.status_code == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        raise RateLimitError(
                            "Rate limit exceeded",
                            retry_after=retry_after
                        )
                    elif response.status_code >= 500:
                        error_data = response.json() if response.content else {}
                        raise NetworkError(
                            f"Server error: {response.status_code}",
                            status_code=response.status_code,
                            response_body=str(error_data)
                        )
                    else:
                        raise NetworkError(
                            f"Unexpected status code: {response.status_code}",
                            status_code=response.status_code,
                            response_body=response.text
                        )
                
                except httpx.TimeoutException:
                    raise TimeoutError(f"Request timed out after {request_timeout} seconds")
                except httpx.NetworkError as e:
                    raise NetworkError(f"Network error: {str(e)}")
                except httpx.HTTPError as e:
                    raise NetworkError(f"HTTP error: {str(e)}")
        
        # Retry logic for certain errors
        if self.retries > 0:
            return exponential_backoff_retry(
                make_single_request,
                max_retries=self.retries,
                base_delay=1.0,
                max_delay=30.0
            )
        else:
            return make_single_request()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health status of the TSIOT service.
        
        Returns:
            Health status information
        """
        return self._make_request("GET", "/health")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the TSIOT service.
        
        Returns:
            Service information including version, capabilities, etc.
        """
        return self._make_request("GET", "/api/v1/info")
    
    def generate(
        self,
        request: Dict[str, Any],
        timeout: float = None
    ) -> TimeSeries:
        """
        Generate synthetic time series data.
        
        Args:
            request: Generation request parameters
            timeout: Request timeout override
        
        Returns:
            Generated time series data
        
        Example:
            ```python
            request = {
                "type": "arima",
                "length": 1000,
                "parameters": {
                    "ar_params": [0.5, -0.3],
                    "ma_params": [0.2],
                    "trend": "linear"
                },
                "metadata": {
                    "series_id": "test-series",
                    "name": "Test ARIMA Series"
                }
            }
            
            time_series = client.generate(request)
            ```
        """
        response = self._make_request(
            "POST",
            "/api/v1/generate",
            data=request,
            timeout=timeout
        )
        
        return TimeSeries.from_dict(response)
    
    def validate(
        self,
        time_series: Union[TimeSeries, Dict[str, Any]],
        validation_types: List[str] = None,
        timeout: float = None
    ) -> Dict[str, Any]:
        """
        Validate time series data.
        
        Args:
            time_series: Time series data to validate
            validation_types: Types of validation to perform
            timeout: Request timeout override
        
        Returns:
            Validation results
        
        Example:
            ```python
            validation_result = client.validate(
                time_series,
                validation_types=["quality", "statistical", "privacy"]
            )
            
            print(f"Quality score: {validation_result['quality_score']}")
            ```
        """
        if isinstance(time_series, TimeSeries):
            data = time_series.to_dict()
        else:
            data = time_series
        
        request = {
            "time_series": data,
            "validation_types": validation_types or ["quality", "statistical"]
        }
        
        return self._make_request(
            "POST",
            "/api/v1/validate",
            data=request,
            timeout=timeout
        )
    
    def analyze(
        self,
        time_series: Union[TimeSeries, Dict[str, Any]],
        analysis_types: List[str] = None,
        parameters: Dict[str, Any] = None,
        timeout: float = None
    ) -> Dict[str, Any]:
        """
        Perform analytics on time series data.
        
        Args:
            time_series: Time series data to analyze
            analysis_types: Types of analysis to perform
            parameters: Analysis parameters
            timeout: Request timeout override
        
        Returns:
            Analysis results
        
        Example:
            ```python
            analysis = client.analyze(
                time_series,
                analysis_types=["basic", "trend", "seasonality", "anomaly"],
                parameters={"forecast_horizon": 24}
            )
            
            print(f"Trend: {analysis['trend']['direction']}")
            print(f"Seasonality: {analysis['seasonality']['has_seasonality']}")
            ```
        """
        if isinstance(time_series, TimeSeries):
            data = time_series.to_dict()
        else:
            data = time_series
        
        request = {
            "time_series": data,
            "analysis_types": analysis_types or ["basic", "trend"],
            "parameters": parameters or {}
        }
        
        return self._make_request(
            "POST",
            "/api/v1/analyze",
            data=request,
            timeout=timeout
        )
    
    def list_generators(self) -> List[Dict[str, Any]]:
        """
        List available data generators.
        
        Returns:
            List of available generators with their capabilities
        """
        return self._make_request("GET", "/api/v1/generators")
    
    def list_validators(self) -> List[Dict[str, Any]]:
        """
        List available validators.
        
        Returns:
            List of available validators with their capabilities
        """
        return self._make_request("GET", "/api/v1/validators")
    
    def export_data(
        self,
        time_series: Union[TimeSeries, Dict[str, Any]],
        format: DataFormat = DataFormat.JSON,
        compression: bool = False,
        timeout: float = None
    ) -> Union[str, bytes]:
        """
        Export time series data in specified format.
        
        Args:
            time_series: Time series data to export
            format: Export format
            compression: Whether to compress the output
            timeout: Request timeout override
        
        Returns:
            Exported data as string or bytes
        """
        if isinstance(time_series, TimeSeries):
            data = time_series.to_dict()
        else:
            data = time_series
        
        request = {
            "time_series": data,
            "format": format.value,
            "compression": compression
        }
        
        response = self._make_request(
            "POST",
            "/api/v1/export",
            data=request,
            timeout=timeout
        )
        
        if compression:
            import base64
            return base64.b64decode(response["data"])
        else:
            return response["data"]
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get service metrics and statistics.
        
        Returns:
            Service metrics
        """
        return self._make_request("GET", "/metrics")
    
    def batch_generate(
        self,
        requests: List[Dict[str, Any]],
        timeout: float = None
    ) -> List[TimeSeries]:
        """
        Generate multiple time series in a single batch request.
        
        Args:
            requests: List of generation requests
            timeout: Request timeout override
        
        Returns:
            List of generated time series
        """
        batch_request = {"requests": requests}
        
        response = self._make_request(
            "POST",
            "/api/v1/batch/generate",
            data=batch_request,
            timeout=timeout
        )
        
        return [TimeSeries.from_dict(ts_data) for ts_data in response["results"]]
    
    def stream_generate(
        self,
        request: Dict[str, Any],
        chunk_size: int = 1000
    ) -> AsyncIterator[List[DataPoint]]:
        """
        Generate time series data as a stream of chunks.
        
        Args:
            request: Generation request parameters
            chunk_size: Size of each chunk
        
        Yields:
            Chunks of data points
        
        Note:
            This is a placeholder for streaming functionality.
            Actual implementation would require WebSocket or Server-Sent Events.
        """
        # For now, simulate streaming by generating full data and chunking it
        full_ts = self.generate(request)
        
        for i in range(0, len(full_ts), chunk_size):
            chunk = full_ts.data_points[i:i + chunk_size]
            yield chunk


class AsyncTSIOTClient:
    """
    Asynchronous client for the TSIOT service.
    
    This client provides async/await methods for high-performance applications
    that need to handle many concurrent requests.
    
    Example:
        ```python
        async with AsyncTSIOTClient(
            base_url="http://localhost:8080",
            api_key="your-api-key"
        ) as client:
            
            # Generate multiple time series concurrently
            requests = [
                {"type": "arima", "length": 1000},
                {"type": "lstm", "length": 1000},
                {"type": "statistical", "length": 1000}
            ]
            
            tasks = [client.generate(req) for req in requests]
            results = await asyncio.gather(*tasks)
            
            print(f"Generated {len(results)} time series")
        ```
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: str = None,
        jwt_token: str = None,
        timeout: float = 30.0,
        retries: int = 3,
        verify_ssl: bool = True,
        user_agent: str = None,
        logger: logging.Logger = None,
        max_connections: int = 100
    ):
        """
        Initialize the async TSIOT client.
        
        Args:
            base_url: Base URL of the TSIOT service
            api_key: API key for authentication (optional)
            jwt_token: JWT token for authentication (optional)
            timeout: Request timeout in seconds
            retries: Number of retry attempts for failed requests
            verify_ssl: Whether to verify SSL certificates
            user_agent: Custom user agent string
            logger: Logger instance
            max_connections: Maximum number of concurrent connections
        """
        self.base_url = validate_string_not_empty(base_url.rstrip('/'), "base_url")
        self.api_key = api_key
        self.jwt_token = jwt_token
        self.timeout = validate_positive_number(timeout, "timeout")
        self.retries = max(0, int(retries))
        self.verify_ssl = verify_ssl
        self.max_connections = validate_positive_number(max_connections, "max_connections")
        self.logger = logger or get_logger(__name__)
        
        # Validate that we have some form of authentication
        if not self.api_key and not self.jwt_token:
            self.logger.warning("No authentication provided. Some endpoints may not be accessible.")
        
        # Set up HTTP headers
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": user_agent or f"tsiot-python-sdk/1.0.0"
        }
        
        if self.api_key:
            self._headers["X-API-Key"] = self.api_key
        elif self.jwt_token:
            self._headers["Authorization"] = f"Bearer {self.jwt_token}"
        
        self._client = None
        self.logger.info(f"Initialized async TSIOT client for {self.base_url}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_client(self):
        """Ensure the HTTP client is initialized."""
        if self._client is None:
            limits = httpx.Limits(max_connections=self.max_connections)
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._headers,
                timeout=self.timeout,
                verify=self.verify_ssl,
                limits=limits
            )
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Dict[str, Any] = None,
        params: Dict[str, Any] = None,
        timeout: float = None
    ) -> Dict[str, Any]:
        """
        Make an async HTTP request with error handling and retries.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters
            timeout: Request timeout override
        
        Returns:
            Response data as dictionary
        
        Raises:
            TSIOTError: Various error types based on response
        """
        await self._ensure_client()
        
        url = endpoint
        request_timeout = timeout or self.timeout
        
        async def make_single_request():
            try:
                response = await self._client.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    timeout=request_timeout
                )
                
                # Handle different response status codes
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 201:
                    return response.json()
                elif response.status_code == 204:
                    return {}
                elif response.status_code == 400:
                    error_data = response.json() if response.content else {}
                    raise ValidationError(
                        error_data.get("message", "Bad request"),
                        details=error_data
                    )
                elif response.status_code == 401:
                    raise AuthenticationError("Authentication failed")
                elif response.status_code == 403:
                    raise AuthenticationError("Access forbidden")
                elif response.status_code == 404:
                    raise NetworkError(f"Endpoint not found: {url}", status_code=404)
                elif response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    raise RateLimitError(
                        "Rate limit exceeded",
                        retry_after=retry_after
                    )
                elif response.status_code >= 500:
                    error_data = response.json() if response.content else {}
                    raise NetworkError(
                        f"Server error: {response.status_code}",
                        status_code=response.status_code,
                        response_body=str(error_data)
                    )
                else:
                    raise NetworkError(
                        f"Unexpected status code: {response.status_code}",
                        status_code=response.status_code,
                        response_body=response.text
                    )
            
            except httpx.TimeoutException:
                raise TimeoutError(f"Request timed out after {request_timeout} seconds")
            except httpx.NetworkError as e:
                raise NetworkError(f"Network error: {str(e)}")
            except httpx.HTTPError as e:
                raise NetworkError(f"HTTP error: {str(e)}")
        
        # Retry logic for certain errors
        if self.retries > 0:
            for attempt in range(self.retries + 1):
                try:
                    return await make_single_request()
                except (NetworkError, TimeoutError) as e:
                    if attempt == self.retries:
                        raise e
                    
                    # Exponential backoff
                    delay = min(1.0 * (2 ** attempt), 30.0)
                    await asyncio.sleep(delay)
        else:
            return await make_single_request()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health status of the TSIOT service."""
        return await self._make_request("GET", "/health")
    
    async def get_info(self) -> Dict[str, Any]:
        """Get information about the TSIOT service."""
        return await self._make_request("GET", "/api/v1/info")
    
    async def generate(
        self,
        request: Dict[str, Any],
        timeout: float = None
    ) -> TimeSeries:
        """Generate synthetic time series data."""
        response = await self._make_request(
            "POST",
            "/api/v1/generate",
            data=request,
            timeout=timeout
        )
        
        return TimeSeries.from_dict(response)
    
    async def validate(
        self,
        time_series: Union[TimeSeries, Dict[str, Any]],
        validation_types: List[str] = None,
        timeout: float = None
    ) -> Dict[str, Any]:
        """Validate time series data."""
        if isinstance(time_series, TimeSeries):
            data = time_series.to_dict()
        else:
            data = time_series
        
        request = {
            "time_series": data,
            "validation_types": validation_types or ["quality", "statistical"]
        }
        
        return await self._make_request(
            "POST",
            "/api/v1/validate",
            data=request,
            timeout=timeout
        )
    
    async def analyze(
        self,
        time_series: Union[TimeSeries, Dict[str, Any]],
        analysis_types: List[str] = None,
        parameters: Dict[str, Any] = None,
        timeout: float = None
    ) -> Dict[str, Any]:
        """Perform analytics on time series data."""
        if isinstance(time_series, TimeSeries):
            data = time_series.to_dict()
        else:
            data = time_series
        
        request = {
            "time_series": data,
            "analysis_types": analysis_types or ["basic", "trend"],
            "parameters": parameters or {}
        }
        
        return await self._make_request(
            "POST",
            "/api/v1/analyze",
            data=request,
            timeout=timeout
        )
    
    async def batch_generate(
        self,
        requests: List[Dict[str, Any]],
        timeout: float = None
    ) -> List[TimeSeries]:
        """Generate multiple time series in a single batch request."""
        batch_request = {"requests": requests}
        
        response = await self._make_request(
            "POST",
            "/api/v1/batch/generate",
            data=batch_request,
            timeout=timeout
        )
        
        return [TimeSeries.from_dict(ts_data) for ts_data in response["results"]]
    
    async def concurrent_generate(
        self,
        requests: List[Dict[str, Any]],
        max_concurrent: int = 10
    ) -> List[TimeSeries]:
        """
        Generate multiple time series concurrently with controlled concurrency.
        
        Args:
            requests: List of generation requests
            max_concurrent: Maximum number of concurrent requests
        
        Returns:
            List of generated time series
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_semaphore(request):
            async with semaphore:
                return await self.generate(request)
        
        tasks = [generate_with_semaphore(req) for req in requests]
        return await asyncio.gather(*tasks)


# Convenience functions
def create_client(
    base_url: str,
    api_key: str = None,
    jwt_token: str = None,
    **kwargs
) -> TSIOTClient:
    """Create a synchronous TSIOT client."""
    return TSIOTClient(
        base_url=base_url,
        api_key=api_key,
        jwt_token=jwt_token,
        **kwargs
    )


def create_async_client(
    base_url: str,
    api_key: str = None,
    jwt_token: str = None,
    **kwargs
) -> AsyncTSIOTClient:
    """Create an asynchronous TSIOT client."""
    return AsyncTSIOTClient(
        base_url=base_url,
        api_key=api_key,
        jwt_token=jwt_token,
        **kwargs
    )