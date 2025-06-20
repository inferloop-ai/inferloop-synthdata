"""
Main SDK client for structured document synthetic data generation.
Provides a comprehensive interface for document generation, validation, and management.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Iterator
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .models import (
    GenerationRequest, BatchGenerationRequest, ValidationRequest, ExportRequest,
    GenerationResponse, BatchGenerationResponse, ValidationResponse, ExportResponse,
    DocumentListResponse, StatusResponse, QuotaResponse, JobResponse,
    GenerationConfig, DocumentType, DocumentFormat, 
    create_success_response, create_error_response, ErrorType, ResponseStatus
)


class StructuredDocsClient:
    """
    Main client for the Structured Documents Synthetic Data API.
    
    Provides methods for document generation, validation, export, and management.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.structuredocs.ai",
        timeout: int = 300,
        max_retries: int = 3,
        enable_caching: bool = True
    ):
        """
        Initialize the client.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL for the API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            enable_caching: Enable response caching
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.enable_caching = enable_caching
        
        # Configure session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            "User-Agent": "StructuredDocs-Python-SDK/1.0.0",
            "Accept": "application/json",
            "Content-Type": "application/json"
        })
        
        if self.api_key:
            self.session.headers["Authorization"] = f"Bearer {self.api_key}"
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request data for POST/PUT
            params: Query parameters
            files: Files to upload
        
        Returns:
            Response data
        
        Raises:
            requests.RequestException: For network/HTTP errors
        """
        url = urljoin(self.base_url, endpoint)
        
        # Prepare request kwargs
        kwargs = {
            "timeout": self.timeout,
            "params": params
        }
        
        if files:
            # For file uploads, don't set JSON content type
            kwargs["files"] = files
            if data:
                kwargs["data"] = data
        elif data:
            kwargs["json"] = data
        
        # Make request
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        
        # Parse response
        if response.headers.get("content-type", "").startswith("application/json"):
            return response.json()
        else:
            return {"data": response.content, "content_type": response.headers.get("content-type")}
    
    def generate_document(
        self,
        document_type: Union[str, DocumentType],
        document_format: Union[str, DocumentFormat] = DocumentFormat.PDF,
        config: Optional[Union[Dict[str, Any], GenerationConfig]] = None,
        **kwargs
    ) -> GenerationResponse:
        """
        Generate a single document.
        
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
            response_data = self._make_request("POST", "/api/v1/generate", data=request.dict())
            return GenerationResponse(**response_data)
        except requests.RequestException as e:
            return GenerationResponse(
                status=ResponseStatus.FAILURE,
                message=f"Generation failed: {str(e)}",
                errors=[create_error_response(ErrorType.NETWORK_ERROR, str(e)).errors[0]]
            )
    
    def generate_batch(
        self,
        requests: List[Union[GenerationRequest, Dict[str, Any]]],
        batch_id: Optional[str] = None,
        parallel_processing: bool = True,
        **kwargs
    ) -> BatchGenerationResponse:
        """
        Generate multiple documents in batch.
        
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
            response_data = self._make_request("POST", "/api/v1/generate/batch", data=batch_request.dict())
            return BatchGenerationResponse(**response_data)
        except requests.RequestException as e:
            return BatchGenerationResponse(
                status=ResponseStatus.FAILURE,
                message=f"Batch generation failed: {str(e)}",
                batch_id=batch_id or "unknown",
                total_requested=len(requests),
                successful_generations=0,
                failed_generations=len(requests),
                errors=[create_error_response(ErrorType.NETWORK_ERROR, str(e)).errors[0]]
            )
    
    def validate_document(
        self,
        document_id: Optional[str] = None,
        document_content: Optional[Dict[str, Any]] = None,
        document_type: Optional[Union[str, DocumentType]] = None,
        validation_types: List[str] = None,
        strict_mode: bool = False
    ) -> ValidationResponse:
        """
        Validate a document.
        
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
            response_data = self._make_request("POST", "/api/v1/validate", data=request_data)
            return ValidationResponse(**response_data)
        except requests.RequestException as e:
            return ValidationResponse(
                status=ResponseStatus.FAILURE,
                message=f"Validation failed: {str(e)}",
                document_id=document_id or "unknown",
                validation_score=0.0,
                is_valid=False,
                errors=[create_error_response(ErrorType.NETWORK_ERROR, str(e)).errors[0]]
            )
    
    def export_documents(
        self,
        document_ids: List[str],
        export_format: str,
        output_path: str,
        include_images: bool = True,
        include_annotations: bool = True,
        compression: bool = False
    ) -> ExportResponse:
        """
        Export documents to various formats.
        
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
            response_data = self._make_request("POST", "/api/v1/export", data=request.dict())
            return ExportResponse(**response_data)
        except requests.RequestException as e:
            return ExportResponse(
                status=ResponseStatus.FAILURE,
                message=f"Export failed: {str(e)}",
                export_id="unknown",
                export_format=export_format,
                document_ids=document_ids,
                output_path=output_path,
                errors=[create_error_response(ErrorType.NETWORK_ERROR, str(e)).errors[0]]
            )
    
    def list_documents(
        self,
        document_type: Optional[Union[str, DocumentType]] = None,
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        filters: Optional[Dict[str, Any]] = None
    ) -> DocumentListResponse:
        """
        List generated documents.
        
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
            response_data = self._make_request("GET", "/api/v1/documents", params=params)
            return DocumentListResponse(**response_data)
        except requests.RequestException as e:
            return DocumentListResponse(
                status=ResponseStatus.FAILURE,
                message=f"Failed to list documents: {str(e)}",
                items=[],
                total_count=0,
                page=page,
                page_size=page_size,
                total_pages=0
            )
    
    def get_document(self, document_id: str) -> GenerationResponse:
        """
        Get details of a specific document.
        
        Args:
            document_id: Document ID
        
        Returns:
            Generation response with document details
        """
        try:
            response_data = self._make_request("GET", f"/api/v1/documents/{document_id}")
            return GenerationResponse(**response_data)
        except requests.RequestException as e:
            return GenerationResponse(
                status=ResponseStatus.FAILURE,
                message=f"Failed to get document: {str(e)}",
                document_id=document_id,
                errors=[create_error_response(ErrorType.NETWORK_ERROR, str(e)).errors[0]]
            )
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete a document.
        
        Args:
            document_id: Document ID to delete
        
        Returns:
            Deletion response
        """
        try:
            return self._make_request("DELETE", f"/api/v1/documents/{document_id}")
        except requests.RequestException as e:
            return {
                "status": "failure",
                "message": f"Failed to delete document: {str(e)}"
            }
    
    def upload_file(self, file_path: Union[str, Path], file_type: str = "document") -> Dict[str, Any]:
        """
        Upload a file to the service.
        
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
            with open(file_path, "rb") as f:
                files = {"file": (file_path.name, f, "application/octet-stream")}
                data = {"file_type": file_type}
                
                return self._make_request("POST", "/api/v1/upload", data=data, files=files)
        except requests.RequestException as e:
            return {
                "status": "failure",
                "message": f"Upload failed: {str(e)}"
            }
    
    def download_file(self, file_id: str, output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Download a file from the service.
        
        Args:
            file_id: ID of file to download
            output_path: Local path to save file
        
        Returns:
            Download response
        """
        try:
            response_data = self._make_request("GET", f"/api/v1/download/{file_id}")
            
            if output_path and "data" in response_data:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, "wb") as f:
                    f.write(response_data["data"])
                
                response_data["local_path"] = str(output_path)
            
            return response_data
        except requests.RequestException as e:
            return {
                "status": "failure",
                "message": f"Download failed: {str(e)}"
            }
    
    def get_status(self) -> StatusResponse:
        """
        Get service status and health information.
        
        Returns:
            Status response
        """
        try:
            response_data = self._make_request("GET", "/api/v1/status")
            return StatusResponse(**response_data)
        except requests.RequestException as e:
            return StatusResponse(
                status=ResponseStatus.FAILURE,
                message=f"Failed to get status: {str(e)}",
                service_name="structured-docs",
                version="unknown",
                uptime_seconds=0,
                health_status="unhealthy"
            )
    
    def get_quota(self) -> QuotaResponse:
        """
        Get current usage quota and limits.
        
        Returns:
            Quota response
        """
        try:
            response_data = self._make_request("GET", "/api/v1/quota")
            return QuotaResponse(**response_data)
        except requests.RequestException as e:
            return QuotaResponse(
                status=ResponseStatus.FAILURE,
                message=f"Failed to get quota: {str(e)}",
                user_id="unknown",
                plan_type="unknown"
            )
    
    def get_job_status(self, job_id: str) -> JobResponse:
        """
        Get status of an asynchronous job.
        
        Args:
            job_id: Job ID
        
        Returns:
            Job status response
        """
        try:
            response_data = self._make_request("GET", f"/api/v1/jobs/{job_id}")
            return JobResponse(**response_data)
        except requests.RequestException as e:
            return JobResponse(
                status=ResponseStatus.FAILURE,
                message=f"Failed to get job status: {str(e)}",
                job_id=job_id,
                job_type="unknown",
                job_status="failed"
            )
    
    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel an asynchronous job.
        
        Args:
            job_id: Job ID to cancel
        
        Returns:
            Cancellation response
        """
        try:
            return self._make_request("POST", f"/api/v1/jobs/{job_id}/cancel")
        except requests.RequestException as e:
            return {
                "status": "failure",
                "message": f"Failed to cancel job: {str(e)}"
            }
    
    def wait_for_job(
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
            job_status = self.get_job_status(job_id)
            
            if job_status.is_completed:
                return job_status
            
            time.sleep(poll_interval)
        
        # Timeout reached
        return JobResponse(
            status=ResponseStatus.FAILURE,
            message="Job timeout reached",
            job_id=job_id,
            job_type="unknown",
            job_status="timeout"
        )
    
    def stream_generation_progress(
        self,
        batch_id: str,
        poll_interval: int = 2
    ) -> Iterator[Dict[str, Any]]:
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
                response_data = self._make_request("GET", f"/api/v1/batches/{batch_id}/progress")
                yield response_data
                
                if response_data.get("status") in ["completed", "failed", "cancelled"]:
                    break
                
                time.sleep(poll_interval)
            except requests.RequestException as e:
                yield {
                    "status": "error",
                    "message": f"Failed to get progress: {str(e)}"
                }
                break
    
    # Context manager support
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()


# Convenience functions for common operations
def generate_academic_paper(
    title: str = None,
    author: str = None,
    page_count: int = 10,
    **kwargs
) -> GenerationResponse:
    """Generate an academic paper with default settings"""
    client = StructuredDocsClient()
    
    metadata = {}
    if title:
        metadata["title"] = title
    if author:
        metadata["author"] = author
    
    return client.generate_document(
        document_type=DocumentType.ACADEMIC_PAPER,
        metadata=metadata,
        **kwargs
    )


def generate_business_form(
    form_type: str = "application",
    required_fields: List[str] = None,
    **kwargs
) -> GenerationResponse:
    """Generate a business form with default settings"""
    client = StructuredDocsClient()
    
    custom_config = {
        "form_type": form_type,
        "required_fields": required_fields or ["name", "email", "phone"]
    }
    
    return client.generate_document(
        document_type=DocumentType.BUSINESS_FORM,
        custom_config=custom_config,
        **kwargs
    )


def generate_financial_report(
    company_name: str = None,
    report_period: str = None,
    **kwargs
) -> GenerationResponse:
    """Generate a financial report with default settings"""
    client = StructuredDocsClient()
    
    metadata = {}
    if company_name:
        metadata["title"] = f"{company_name} Financial Report"
        metadata["author"] = company_name
    if report_period:
        metadata["subject"] = f"Financial Report - {report_period}"
    
    return client.generate_document(
        document_type=DocumentType.FINANCIAL_REPORT,
        metadata=metadata,
        **kwargs
    )