"""
Asynchronous SDK client for structured document synthetic data generation.
Provides async/await interface for high-performance document generation.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, AsyncIterator

import aiohttp
import aiofiles
from aiohttp import ClientSession, ClientTimeout, TCPConnector

from .models import (
    GenerationRequest, BatchGenerationRequest, ValidationRequest, ExportRequest,
    GenerationResponse, BatchGenerationResponse, ValidationResponse, ExportResponse,
    DocumentListResponse, StatusResponse, QuotaResponse, JobResponse,
    GenerationConfig, DocumentType, DocumentFormat,
    create_success_response, create_error_response, ErrorType, ResponseStatus
)


class AsyncStructuredDocsClient:
    """
    Asynchronous client for the Structured Documents Synthetic Data API.
    
    Provides async methods for document generation, validation, export, and management.
    Optimized for high-throughput and concurrent operations.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.structuredocs.ai",
        timeout: int = 300,
        max_connections: int = 100,
        max_connections_per_host: int = 30,
        enable_retries: bool = True,
        max_retries: int = 3
    ):
        """
        Initialize the async client.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL for the API
            timeout: Request timeout in seconds
            max_connections: Maximum total connections
            max_connections_per_host: Maximum connections per host
            enable_retries: Enable automatic retries
            max_retries: Maximum number of retries
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries if enable_retries else 0
        
        # Configure connector
        self.connector = TCPConnector(
            limit=max_connections,
            limit_per_host=max_connections_per_host,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        # Configure timeout
        self.client_timeout = ClientTimeout(total=timeout)
        
        # Session will be created when needed
        self._session: Optional[ClientSession] = None
        self._session_created = False
    
    async def _get_session(self) -> ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            headers = {
                "User-Agent": "StructuredDocs-Python-SDK-Async/1.0.0",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            self._session = ClientSession(
                connector=self.connector,
                timeout=self.client_timeout,
                headers=headers
            )
            self._session_created = True
        
        return self._session
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an asynchronous HTTP request to the API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request data for POST/PUT
            params: Query parameters
            files: Files to upload
        
        Returns:
            Response data
        
        Raises:
            aiohttp.ClientError: For network/HTTP errors
        """
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"
        
        # Retry logic
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Prepare request kwargs
                kwargs = {"params": params}
                
                if files:
                    # For file uploads
                    form_data = aiohttp.FormData()
                    for key, value in files.items():
                        if isinstance(value, tuple):
                            filename, file_obj, content_type = value
                            form_data.add_field(key, file_obj, filename=filename, content_type=content_type)
                        else:
                            form_data.add_field(key, value)
                    
                    if data:
                        for key, value in data.items():
                            form_data.add_field(key, str(value))
                    
                    kwargs["data"] = form_data
                elif data:
                    kwargs["json"] = data
                
                # Make request
                async with session.request(method, url, **kwargs) as response:
                    response.raise_for_status()
                    
                    # Parse response
                    content_type = response.headers.get("content-type", "")
                    if content_type.startswith("application/json"):
                        return await response.json()
                    else:
                        content = await response.read()
                        return {"data": content, "content_type": content_type}
            
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise e
        
        # Should never reach here, but just in case
        raise last_exception or aiohttp.ClientError("Request failed")
    
    async def generate_document(
        self,
        document_type: Union[str, DocumentType],
        document_format: Union[str, DocumentFormat] = DocumentFormat.PDF,
        config: Optional[Union[Dict[str, Any], GenerationConfig]] = None,
        **kwargs
    ) -> GenerationResponse:
        """
        Generate a single document asynchronously.
        
        Args:
            document_type: Type of document to generate
            document_format: Output format
            config: Generation configuration
            **kwargs: Additional generation options
        
        Returns:
            Generation response with document details
        """
        # Create request
        request = GenerationRequest(
            document_type=DocumentType(document_type) if isinstance(document_type, str) else document_type,
            document_format=DocumentFormat(document_format) if isinstance(document_format, str) else document_format,
            count=1,
            custom_config=config if isinstance(config, dict) else config.dict() if config else {},
            **kwargs
        )
        
        try:
            response_data = await self._make_request("POST", "/api/v1/generate", data=request.dict())
            return GenerationResponse(**response_data)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            return GenerationResponse(
                status=ResponseStatus.FAILURE,
                message=f"Generation failed: {str(e)}",
                errors=[create_error_response(ErrorType.NETWORK_ERROR, str(e)).errors[0]]
            )
    
    async def generate_batch(
        self,
        requests: List[Union[GenerationRequest, Dict[str, Any]]],
        batch_id: Optional[str] = None,
        parallel_processing: bool = True,
        **kwargs
    ) -> BatchGenerationResponse:
        """
        Generate multiple documents in batch asynchronously.
        
        Args:
            requests: List of generation requests
            batch_id: Optional batch identifier
            parallel_processing: Enable parallel processing
            **kwargs: Additional batch options
        
        Returns:
            Batch generation response
        """
        # Convert dict requests to GenerationRequest objects
        processed_requests = []
        for req in requests:
            if isinstance(req, dict):
                processed_requests.append(GenerationRequest(**req))
            else:
                processed_requests.append(req)
        
        # Create batch request
        batch_request = BatchGenerationRequest(
            requests=processed_requests,
            batch_id=batch_id,
            parallel_processing=parallel_processing,
            **kwargs
        )
        
        try:
            response_data = await self._make_request("POST", "/api/v1/generate/batch", data=batch_request.dict())
            return BatchGenerationResponse(**response_data)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            return BatchGenerationResponse(
                status=ResponseStatus.FAILURE,
                message=f"Batch generation failed: {str(e)}",
                batch_id=batch_id or "unknown",
                total_requested=len(requests),
                successful_generations=0,
                failed_generations=len(requests),
                errors=[create_error_response(ErrorType.NETWORK_ERROR, str(e)).errors[0]]
            )
    
    async def generate_concurrent(
        self,
        requests: List[Union[GenerationRequest, Dict[str, Any]]],
        max_concurrent: int = 10
    ) -> List[GenerationResponse]:
        """
        Generate multiple documents concurrently with controlled concurrency.
        
        Args:
            requests: List of generation requests
            max_concurrent: Maximum concurrent requests
        
        Returns:
            List of generation responses
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_semaphore(req):
            async with semaphore:
                if isinstance(req, dict):
                    return await self.generate_document(**req)
                else:
                    return await self.generate_document(
                        document_type=req.document_type,
                        document_format=req.document_format,
                        config=req.custom_config,
                        **req.dict(exclude={"document_type", "document_format", "custom_config"})
                    )
        
        # Execute all requests concurrently
        tasks = [generate_with_semaphore(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def validate_document(
        self,
        document_id: Optional[str] = None,
        document_content: Optional[Dict[str, Any]] = None,
        document_type: Optional[Union[str, DocumentType]] = None,
        validation_types: List[str] = None,
        strict_mode: bool = False
    ) -> ValidationResponse:
        """
        Validate a document asynchronously.
        
        Args:
            document_id: ID of existing document to validate
            document_content: Document content to validate
            document_type: Type of document for validation rules
            validation_types: Types of validation to perform
            strict_mode: Enable strict validation mode
        
        Returns:
            Validation response
        """
        if not document_id and not document_content:
            return ValidationResponse(
                status=ResponseStatus.FAILURE,
                message="Either document_id or document_content must be provided",
                document_id="unknown",
                validation_score=0.0,
                is_valid=False
            )
        
        # Create validation request
        request_data = {
            "validation_types": validation_types or ["structural", "completeness", "semantic"],
            "strict_mode": strict_mode
        }
        
        if document_id:
            request_data["document_id"] = document_id
        if document_content:
            request_data["document_content"] = document_content
        if document_type:
            request_data["document_type"] = document_type
        
        try:
            response_data = await self._make_request("POST", "/api/v1/validate", data=request_data)
            return ValidationResponse(**response_data)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            return ValidationResponse(
                status=ResponseStatus.FAILURE,
                message=f"Validation failed: {str(e)}",
                document_id=document_id or "unknown",
                validation_score=0.0,
                is_valid=False,
                errors=[create_error_response(ErrorType.NETWORK_ERROR, str(e)).errors[0]]
            )
    
    async def export_documents(
        self,
        document_ids: List[str],
        export_format: str,
        output_path: str,
        include_images: bool = True,
        include_annotations: bool = True,
        compression: bool = False
    ) -> ExportResponse:
        """
        Export documents to various formats asynchronously.
        
        Args:
            document_ids: List of document IDs to export
            export_format: Export format (coco, yolo, pascal_voc, etc.)
            output_path: Output directory path
            include_images: Include image files
            include_annotations: Include annotation files
            compression: Compress exported files
        
        Returns:
            Export response
        """
        request = ExportRequest(
            document_ids=document_ids,
            export_format=export_format,
            output_path=output_path,
            include_images=include_images,
            include_annotations=include_annotations,
            compression=compression
        )
        
        try:
            response_data = await self._make_request("POST", "/api/v1/export", data=request.dict())
            return ExportResponse(**response_data)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            return ExportResponse(
                status=ResponseStatus.FAILURE,
                message=f"Export failed: {str(e)}",
                export_id="unknown",
                export_format=export_format,
                document_ids=document_ids,
                output_path=output_path,
                errors=[create_error_response(ErrorType.NETWORK_ERROR, str(e)).errors[0]]
            )
    
    async def list_documents(
        self,
        document_type: Optional[Union[str, DocumentType]] = None,
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        filters: Optional[Dict[str, Any]] = None
    ) -> DocumentListResponse:
        """
        List generated documents asynchronously.
        
        Args:
            document_type: Filter by document type
            page: Page number
            page_size: Number of documents per page
            sort_by: Field to sort by
            sort_order: Sort order (asc/desc)
            filters: Additional filters
        
        Returns:
            Document list response
        """
        params = {
            "page": page,
            "page_size": page_size,
            "sort_by": sort_by,
            "sort_order": sort_order
        }
        
        if document_type:
            params["document_type"] = document_type if isinstance(document_type, str) else document_type.value
        
        if filters:
            params.update(filters)
        
        try:
            response_data = await self._make_request("GET", "/api/v1/documents", params=params)
            return DocumentListResponse(**response_data)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            return DocumentListResponse(
                status=ResponseStatus.FAILURE,
                message=f"Failed to list documents: {str(e)}",
                items=[],
                total_count=0,
                page=page,
                page_size=page_size,
                total_pages=0
            )
    
    async def get_document(self, document_id: str) -> GenerationResponse:
        """
        Get details of a specific document asynchronously.
        
        Args:
            document_id: Document ID
        
        Returns:
            Generation response with document details
        """
        try:
            response_data = await self._make_request("GET", f"/api/v1/documents/{document_id}")
            return GenerationResponse(**response_data)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            return GenerationResponse(
                status=ResponseStatus.FAILURE,
                message=f"Failed to get document: {str(e)}",
                document_id=document_id,
                errors=[create_error_response(ErrorType.NETWORK_ERROR, str(e)).errors[0]]
            )
    
    async def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete a document asynchronously.
        
        Args:
            document_id: Document ID to delete
        
        Returns:
            Deletion response
        """
        try:
            return await self._make_request("DELETE", f"/api/v1/documents/{document_id}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            return {
                "status": "failure",
                "message": f"Failed to delete document: {str(e)}"
            }
    
    async def upload_file(self, file_path: Union[str, Path], file_type: str = "document") -> Dict[str, Any]:
        """
        Upload a file to the service asynchronously.
        
        Args:
            file_path: Path to file to upload
            file_type: Type of file being uploaded
        
        Returns:
            Upload response
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {
                "status": "failure",
                "message": f"File not found: {file_path}"
            }
        
        try:
            async with aiofiles.open(file_path, "rb") as f:
                file_content = await f.read()
                
                files = {"file": (file_path.name, file_content, "application/octet-stream")}
                data = {"file_type": file_type}
                
                return await self._make_request("POST", "/api/v1/upload", data=data, files=files)
        except (aiohttp.ClientError, asyncio.TimeoutError, IOError) as e:
            return {
                "status": "failure",
                "message": f"Upload failed: {str(e)}"
            }
    
    async def download_file(self, file_id: str, output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Download a file from the service asynchronously.
        
        Args:
            file_id: ID of file to download
            output_path: Local path to save file
        
        Returns:
            Download response
        """
        try:
            response_data = await self._make_request("GET", f"/api/v1/download/{file_id}")
            
            if output_path and "data" in response_data:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                async with aiofiles.open(output_path, "wb") as f:
                    await f.write(response_data["data"])
                
                response_data["local_path"] = str(output_path)
            
            return response_data
        except (aiohttp.ClientError, asyncio.TimeoutError, IOError) as e:
            return {
                "status": "failure",
                "message": f"Download failed: {str(e)}"
            }
    
    async def get_status(self) -> StatusResponse:
        """
        Get service status and health information asynchronously.
        
        Returns:
            Status response
        """
        try:
            response_data = await self._make_request("GET", "/api/v1/status")
            return StatusResponse(**response_data)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            return StatusResponse(
                status=ResponseStatus.FAILURE,
                message=f"Failed to get status: {str(e)}",
                service_name="structured-docs",
                version="unknown",
                uptime_seconds=0,
                health_status="unhealthy"
            )
    
    async def get_quota(self) -> QuotaResponse:
        """
        Get current usage quota and limits asynchronously.
        
        Returns:
            Quota response
        """
        try:
            response_data = await self._make_request("GET", "/api/v1/quota")
            return QuotaResponse(**response_data)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            return QuotaResponse(
                status=ResponseStatus.FAILURE,
                message=f"Failed to get quota: {str(e)}",
                user_id="unknown",
                plan_type="unknown"
            )
    
    async def get_job_status(self, job_id: str) -> JobResponse:
        """
        Get status of an asynchronous job.
        
        Args:
            job_id: Job ID
        
        Returns:
            Job status response
        """
        try:
            response_data = await self._make_request("GET", f"/api/v1/jobs/{job_id}")
            return JobResponse(**response_data)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            return JobResponse(
                status=ResponseStatus.FAILURE,
                message=f"Failed to get job status: {str(e)}",
                job_id=job_id,
                job_type="unknown",
                job_status="failed"
            )
    
    async def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel an asynchronous job.
        
        Args:
            job_id: Job ID to cancel
        
        Returns:
            Cancellation response
        """
        try:
            return await self._make_request("POST", f"/api/v1/jobs/{job_id}/cancel")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            return {
                "status": "failure",
                "message": f"Failed to cancel job: {str(e)}"
            }
    
    async def wait_for_job(
        self,
        job_id: str,
        timeout: int = 300,
        poll_interval: int = 5
    ) -> JobResponse:
        """
        Wait for an asynchronous job to complete.
        
        Args:
            job_id: Job ID to wait for
            timeout: Maximum time to wait in seconds
            poll_interval: Polling interval in seconds
        
        Returns:
            Final job status
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            job_status = await self.get_job_status(job_id)
            
            if job_status.is_completed:
                return job_status
            
            await asyncio.sleep(poll_interval)
        
        # Timeout reached
        return JobResponse(
            status=ResponseStatus.FAILURE,
            message="Job timeout reached",
            job_id=job_id,
            job_type="unknown",
            job_status="timeout"
        )
    
    async def stream_generation_progress(
        self,
        batch_id: str,
        poll_interval: int = 2
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream progress updates for a batch generation job.
        
        Args:
            batch_id: Batch ID to monitor
            poll_interval: Polling interval in seconds
        
        Yields:
            Progress updates
        """
        while True:
            try:
                response_data = await self._make_request("GET", f"/api/v1/batches/{batch_id}/progress")
                yield response_data
                
                if response_data.get("status") in ["completed", "failed", "cancelled"]:
                    break
                
                await asyncio.sleep(poll_interval)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                yield {
                    "status": "error",
                    "message": f"Failed to get progress: {str(e)}"
                }
                break
    
    # Context manager support
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def close(self):
        """Close the client session"""
        if self._session and not self._session.closed:
            await self._session.close()


# Async convenience functions
async def async_generate_academic_paper(
    title: str = None,
    author: str = None,
    page_count: int = 10,
    **kwargs
) -> GenerationResponse:
    """Generate an academic paper asynchronously with default settings"""
    async with AsyncStructuredDocsClient() as client:
        metadata = {}
        if title:
            metadata["title"] = title
        if author:
            metadata["author"] = author
        
        return await client.generate_document(
            document_type=DocumentType.ACADEMIC_PAPER,
            metadata=metadata,
            **kwargs
        )


async def async_generate_business_form(
    form_type: str = "application",
    required_fields: List[str] = None,
    **kwargs
) -> GenerationResponse:
    """Generate a business form asynchronously with default settings"""
    async with AsyncStructuredDocsClient() as client:
        custom_config = {
            "form_type": form_type,
            "required_fields": required_fields or ["name", "email", "phone"]
        }
        
        return await client.generate_document(
            document_type=DocumentType.BUSINESS_FORM,
            custom_config=custom_config,
            **kwargs
        )


async def async_generate_batch(
    requests: List[Dict[str, Any]],
    max_concurrent: int = 10
) -> List[GenerationResponse]:
    """Generate multiple documents asynchronously with controlled concurrency"""
    async with AsyncStructuredDocsClient() as client:
        return await client.generate_concurrent(requests, max_concurrent)


# Utility function for running async operations in sync context
def run_async_operation(coro):
    """Run an async operation in sync context"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)