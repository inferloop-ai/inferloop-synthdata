#!/usr/bin/env python3
"""
API Poller for streaming data ingestion from external APIs
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import requests
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from ...core.config import get_config
from ...core.logging import get_logger
from ...core.exceptions import ProcessingError, ValidationError


class PollingMode(Enum):
    """API polling modes"""
    INTERVAL = "interval"
    EVENT_DRIVEN = "event_driven"
    WEBHOOK_TRIGGERED = "webhook_triggered"
    RATE_LIMITED = "rate_limited"


class APIAuthType(Enum):
    """API authentication types"""
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    OAUTH2 = "oauth2"
    BASIC_AUTH = "basic_auth"
    CUSTOM_HEADER = "custom_header"


@dataclass
class APIEndpoint:
    """API endpoint configuration"""
    url: str
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 30
    retry_count: int = 3
    retry_delay: float = 1.0


@dataclass
class PolledData:
    """Data polled from API"""
    endpoint_id: str
    timestamp: datetime
    status_code: int
    data: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    data_hash: str = ""
    is_new: bool = True


class APIPollerConfig(BaseModel):
    """API poller configuration"""
    
    # Polling settings
    polling_interval: float = Field(60.0, description="Polling interval in seconds")
    max_concurrent_requests: int = Field(10, description="Maximum concurrent requests")
    request_timeout: int = Field(30, description="Request timeout in seconds")
    
    # Authentication
    auth_type: APIAuthType = Field(APIAuthType.NONE, description="Authentication type")
    api_key: Optional[str] = Field(None, description="API key for authentication")
    auth_header: str = Field("Authorization", description="Authentication header name")
    auth_value_prefix: str = Field("Bearer ", description="Authentication value prefix")
    
    # Rate limiting
    rate_limit_requests: int = Field(100, description="Requests per rate limit window")
    rate_limit_window: int = Field(3600, description="Rate limit window in seconds")
    respect_rate_limits: bool = Field(True, description="Respect API rate limits")
    
    # Data processing
    deduplicate_data: bool = Field(True, description="Deduplicate polled data")
    store_history: bool = Field(True, description="Store polling history")
    max_history_size: int = Field(1000, description="Maximum history entries")
    
    # Error handling
    max_retries: int = Field(3, description="Maximum retry attempts")
    retry_backoff: float = Field(2.0, description="Retry backoff multiplier")
    skip_errors: bool = Field(False, description="Skip endpoints with errors")
    
    # Filtering
    response_filters: List[str] = Field(default=[], description="JSONPath filters for responses")
    required_fields: List[str] = Field(default=[], description="Required fields in response")


class APIPoller:
    """
    API Poller for streaming data ingestion from external APIs
    
    Supports multiple endpoints, authentication methods, rate limiting,
    deduplication, and error handling for robust data collection.
    """
    
    def __init__(self, config: Optional[APIPollerConfig] = None):
        self.logger = get_logger(__name__)
        self.config = config or APIPollerConfig()
        
        # State management
        self.endpoints: Dict[str, APIEndpoint] = {}
        self.polling_history: List[PolledData] = []
        self.data_hashes: set = set()
        self.rate_limit_tracker: Dict[str, List[float]] = {}
        
        # Async session for efficient requests
        self.session = requests.Session()
        self._setup_session_auth()
        
        # Callbacks
        self.data_callbacks: List[Callable[[PolledData], None]] = []
        self.error_callbacks: List[Callable[[str, Exception], None]] = []
        
        self.logger.info("API Poller initialized")
    
    def _setup_session_auth(self):
        """Setup session authentication"""
        if self.config.auth_type == APIAuthType.API_KEY and self.config.api_key:
            auth_value = f"{self.config.auth_value_prefix}{self.config.api_key}"
            self.session.headers[self.config.auth_header] = auth_value
        elif self.config.auth_type == APIAuthType.BEARER_TOKEN and self.config.api_key:
            self.session.headers["Authorization"] = f"Bearer {self.config.api_key}"
    
    def add_endpoint(self, endpoint_id: str, endpoint: APIEndpoint):
        """Add API endpoint for polling"""
        self.endpoints[endpoint_id] = endpoint
        self.rate_limit_tracker[endpoint_id] = []
        self.logger.info(f"Added endpoint '{endpoint_id}': {endpoint.url}")
    
    def add_data_callback(self, callback: Callable[[PolledData], None]):
        """Add callback for new data"""
        self.data_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[str, Exception], None]):
        """Add callback for errors"""
        self.error_callbacks.append(callback)
    
    def start_polling(self, duration: Optional[float] = None) -> List[PolledData]:
        """Start polling endpoints"""
        start_time = time.time()
        all_data = []
        
        self.logger.info(f"Starting API polling for {len(self.endpoints)} endpoints")
        
        try:
            while True:
                # Poll all endpoints
                batch_data = self._poll_batch()
                all_data.extend(batch_data)
                
                # Check duration limit
                if duration and (time.time() - start_time) >= duration:
                    break
                
                # Wait for next polling interval
                time.sleep(self.config.polling_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Polling interrupted by user")
        except Exception as e:
            self.logger.error(f"Polling failed: {e}")
            raise ProcessingError(f"API polling error: {e}")
        
        self.logger.info(f"Polling completed. Collected {len(all_data)} data points")
        return all_data
    
    def poll_once(self) -> List[PolledData]:
        """Poll all endpoints once"""
        return self._poll_batch()
    
    def _poll_batch(self) -> List[PolledData]:
        """Poll all endpoints in a batch"""
        batch_data = []
        
        for endpoint_id, endpoint in self.endpoints.items():
            try:
                # Check rate limits
                if not self._check_rate_limit(endpoint_id):
                    self.logger.debug(f"Rate limit exceeded for {endpoint_id}, skipping")
                    continue
                
                # Poll endpoint
                data = self._poll_endpoint(endpoint_id, endpoint)
                if data:
                    batch_data.append(data)
                    
                    # Update rate limit tracker
                    self._update_rate_limit(endpoint_id)
                    
            except Exception as e:
                self._handle_error(endpoint_id, e)
                if not self.config.skip_errors:
                    raise
        
        return batch_data
    
    def _poll_endpoint(self, endpoint_id: str, endpoint: APIEndpoint) -> Optional[PolledData]:
        """Poll a single endpoint"""
        # Prepare request
        headers = {**self.session.headers, **endpoint.headers}
        
        # Make request with retries
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.session.request(
                    method=endpoint.method,
                    url=endpoint.url,
                    headers=headers,
                    params=endpoint.params,
                    json=endpoint.data if endpoint.method != "GET" else None,
                    timeout=endpoint.timeout
                )
                
                # Check if request was successful
                if response.status_code >= 400:
                    if attempt < self.config.max_retries:
                        delay = endpoint.retry_delay * (self.config.retry_backoff ** attempt)
                        time.sleep(delay)
                        continue
                    else:
                        raise requests.HTTPError(f"HTTP {response.status_code}: {response.text}")
                
                # Parse response
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    data = {"text": response.text}
                
                # Apply filters
                filtered_data = self._apply_filters(data)
                
                # Validate required fields
                if not self._validate_required_fields(filtered_data):
                    self.logger.warning(f"Response from {endpoint_id} missing required fields")
                    return None
                
                # Create polled data object
                polled_data = PolledData(
                    endpoint_id=endpoint_id,
                    timestamp=datetime.now(),
                    status_code=response.status_code,
                    data=filtered_data,
                    headers=dict(response.headers),
                    metadata={
                        "url": endpoint.url,
                        "method": endpoint.method,
                        "attempt": attempt + 1
                    }
                )
                
                # Calculate data hash for deduplication
                if self.config.deduplicate_data:
                    data_str = json.dumps(filtered_data, sort_keys=True)
                    polled_data.data_hash = hashlib.md5(data_str.encode()).hexdigest()
                    
                    if polled_data.data_hash in self.data_hashes:
                        polled_data.is_new = False
                        self.logger.debug(f"Duplicate data from {endpoint_id}, skipping")
                        return None
                    else:
                        self.data_hashes.add(polled_data.data_hash)
                
                # Store in history
                if self.config.store_history:
                    self.polling_history.append(polled_data)
                    if len(self.polling_history) > self.config.max_history_size:
                        old_data = self.polling_history.pop(0)
                        if old_data.data_hash in self.data_hashes:
                            self.data_hashes.remove(old_data.data_hash)
                
                # Trigger callbacks
                for callback in self.data_callbacks:
                    try:
                        callback(polled_data)
                    except Exception as e:
                        self.logger.warning(f"Data callback failed: {e}")
                
                self.logger.debug(f"Successfully polled {endpoint_id}")
                return polled_data
                
            except Exception as e:
                if attempt < self.config.max_retries:
                    delay = endpoint.retry_delay * (self.config.retry_backoff ** attempt)
                    self.logger.warning(f"Polling {endpoint_id} failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    raise
        
        return None
    
    def _apply_filters(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply JSONPath filters to response data"""
        if not self.config.response_filters:
            return data
        
        # Simple filtering implementation (would use jsonpath-ng in practice)
        filtered_data = data
        
        # Placeholder for JSONPath filtering
        # In practice: from jsonpath_ng import parse
        # for filter_expr in self.config.response_filters:
        #     jsonpath_expr = parse(filter_expr)
        #     matches = [match.value for match in jsonpath_expr.find(data)]
        #     filtered_data = matches[0] if matches else {}
        
        return filtered_data
    
    def _validate_required_fields(self, data: Dict[str, Any]) -> bool:
        """Validate that response contains required fields"""
        if not self.config.required_fields:
            return True
        
        for field in self.config.required_fields:
            if field not in data:
                return False
        
        return True
    
    def _check_rate_limit(self, endpoint_id: str) -> bool:
        """Check if endpoint is within rate limits"""
        if not self.config.respect_rate_limits:
            return True
        
        now = time.time()
        window_start = now - self.config.rate_limit_window
        
        # Clean old requests
        self.rate_limit_tracker[endpoint_id] = [
            req_time for req_time in self.rate_limit_tracker[endpoint_id]
            if req_time > window_start
        ]
        
        # Check if under limit
        return len(self.rate_limit_tracker[endpoint_id]) < self.config.rate_limit_requests
    
    def _update_rate_limit(self, endpoint_id: str):
        """Update rate limit tracker"""
        self.rate_limit_tracker[endpoint_id].append(time.time())
    
    def _handle_error(self, endpoint_id: str, error: Exception):
        """Handle polling errors"""
        self.logger.error(f"Error polling {endpoint_id}: {error}")
        
        for callback in self.error_callbacks:
            try:
                callback(endpoint_id, error)
            except Exception as e:
                self.logger.warning(f"Error callback failed: {e}")
    
    def get_polling_statistics(self) -> Dict[str, Any]:
        """Get polling statistics"""
        total_polls = len(self.polling_history)
        successful_polls = sum(1 for data in self.polling_history if data.status_code < 400)
        unique_data = len(self.data_hashes)
        
        endpoint_stats = {}
        for endpoint_id in self.endpoints.keys():
            endpoint_data = [data for data in self.polling_history if data.endpoint_id == endpoint_id]
            endpoint_stats[endpoint_id] = {
                "total_polls": len(endpoint_data),
                "successful_polls": sum(1 for data in endpoint_data if data.status_code < 400),
                "unique_responses": len(set(data.data_hash for data in endpoint_data if data.data_hash))
            }
        
        return {
            "total_polls": total_polls,
            "successful_polls": successful_polls,
            "success_rate": successful_polls / total_polls if total_polls > 0 else 0,
            "unique_data_points": unique_data,
            "endpoint_statistics": endpoint_stats,
            "active_endpoints": len(self.endpoints)
        }
    
    def export_polled_data(self, format: str = "json") -> List[Dict[str, Any]]:
        """Export polled data"""
        exported_data = []
        
        for data in self.polling_history:
            exported_item = {
                "endpoint_id": data.endpoint_id,
                "timestamp": data.timestamp.isoformat(),
                "status_code": data.status_code,
                "data": data.data,
                "is_new": data.is_new
            }
            
            if data.data_hash:
                exported_item["data_hash"] = data.data_hash
            
            exported_data.append(exported_item)
        
        return exported_data
    
    def clear_history(self):
        """Clear polling history and data hashes"""
        self.polling_history.clear()
        self.data_hashes.clear()
        self.logger.info("Polling history cleared")
    
    def stop_polling(self):
        """Stop polling and cleanup"""
        self.session.close()
        self.logger.info("API polling stopped")


# Factory function
def create_api_poller(**config_kwargs) -> APIPoller:
    """Factory function to create API poller"""
    config = APIPollerConfig(**config_kwargs)
    return APIPoller(config)