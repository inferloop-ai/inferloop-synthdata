"""
REST API for Structured Documents Synthetic Data Generator
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from ...core import (
    get_logger,
    get_config,
    list_document_types,
    get_document_type_config,
    DocumentGenerationError,
    ValidationError,
    APIError
)
from ...generation.engines import get_template_engine, PDFGenerator, DOCXGenerator
from ...privacy import PIIDetector


# Request/Response Models
class DocumentGenerationRequest(BaseModel):
    """Request model for document generation"""
    document_type: str = Field(..., description="Type of document to generate")
    data: Dict[str, Any] = Field(..., description="Data to populate the document")
    output_format: str = Field(default="pdf", description="Output format (pdf, docx)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @validator('document_type')
    def validate_document_type(cls, v):
        if v not in list_document_types():
            raise ValueError(f"Invalid document type. Available types: {list_document_types()}")
        return v
    
    @validator('output_format')
    def validate_output_format(cls, v):
        if v.lower() not in ['pdf', 'docx']:
            raise ValueError("Output format must be 'pdf' or 'docx'")
        return v.lower()


class BatchGenerationRequest(BaseModel):
    """Request model for batch document generation"""
    document_type: str = Field(..., description="Type of document to generate")
    data_list: List[Dict[str, Any]] = Field(..., description="List of data objects")
    output_format: str = Field(default="pdf", description="Output format (pdf, docx)")
    output_prefix: str = Field(default="", description="Prefix for output filenames")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class SampleDataRequest(BaseModel):
    """Request model for sample data generation"""
    document_type: str = Field(..., description="Type of document to generate sample data for")
    count: int = Field(default=1, description="Number of sample data objects to generate")


class DocumentFieldsResponse(BaseModel):
    """Response model for document field information"""
    document_type: str
    name: str
    description: str
    required_fields: List[str]
    optional_fields: List[str]
    supported_formats: List[str]


class GenerationResponse(BaseModel):
    """Response model for document generation"""
    success: bool
    message: str
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    generation_time: Optional[float] = None


class BatchGenerationResponse(BaseModel):
    """Response model for batch generation"""
    success: bool
    message: str
    files_generated: int
    total_files: int
    file_paths: List[str] = []
    generation_time: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str = "1.0.0"
    available_document_types: List[str]


class PIIDetectionRequest(BaseModel):
    """Request model for PII detection"""
    data: Dict[str, Any] = Field(..., description="Data to scan for PII")


class PIIDetectionResponse(BaseModel):
    """Response model for PII detection"""
    has_pii: bool
    total_matches: int
    risk_level: str
    summary: Dict[str, Any]
    field_details: Dict[str, Any]
    recommendations: List[str]


# Initialize FastAPI app
def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="Structured Documents Synthetic Data API",
        description="API for generating synthetic structured documents",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure this properly for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize components
    logger = get_logger(__name__)
    config = get_config()
    template_engine = get_template_engine()
    
    # Initialize generators lazily
    _pdf_generator = None
    _docx_generator = None
    _pii_detector = None
    
    def get_pdf_generator() -> PDFGenerator:
        nonlocal _pdf_generator
        if _pdf_generator is None:
            _pdf_generator = PDFGenerator()
        return _pdf_generator
    
    def get_docx_generator() -> DOCXGenerator:
        nonlocal _docx_generator
        if _docx_generator is None:
            _docx_generator = DOCXGenerator()
        return _docx_generator
    
    def get_pii_detector() -> PIIDetector:
        nonlocal _pii_detector
        if _pii_detector is None:
            _pii_detector = PIIDetector()
        return _pii_detector
    
    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Root endpoint"""
        return {
            "message": "Structured Documents Synthetic Data API",
            "version": "1.0.0",
            "docs": "/docs"
        }
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint"""
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            available_document_types=list_document_types()
        )
    
    @app.get("/api/v1/document-types", response_model=List[str])
    async def get_document_types():
        """Get list of available document types"""
        try:
            return list_document_types()
        except Exception as e:
            logger.error(f"Error getting document types: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to retrieve document types")
    
    @app.get("/api/v1/document-types/{document_type}/fields", response_model=DocumentFieldsResponse)
    async def get_document_fields(document_type: str):
        """Get field information for a specific document type"""
        try:
            if document_type not in list_document_types():
                raise HTTPException(
                    status_code=404, 
                    detail=f"Document type '{document_type}' not found"
                )
            
            config = get_document_type_config(document_type)
            return DocumentFieldsResponse(
                document_type=document_type,
                name=config['name'],
                description=config['description'],
                required_fields=config['required_fields'],
                optional_fields=config['optional_fields'],
                supported_formats=config['supported_formats']
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting fields for {document_type}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to retrieve document fields")
    
    @app.post("/api/v1/sample-data/{document_type}")
    async def generate_sample_data(document_type: str, request: SampleDataRequest):
        """Generate sample data for a document type"""
        try:
            if document_type not in list_document_types():
                raise HTTPException(
                    status_code=404,
                    detail=f"Document type '{document_type}' not found"
                )
            
            sample_data_list = []
            for _ in range(request.count):
                sample_data = template_engine.generate_sample_data(document_type)
                sample_data_list.append(sample_data)
            
            return {
                "document_type": document_type,
                "count": request.count,
                "sample_data": sample_data_list[0] if request.count == 1 else sample_data_list
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error generating sample data for {document_type}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to generate sample data")
    
    @app.post("/api/v1/documents/validate")
    async def validate_document_data(request: DocumentGenerationRequest):
        """Validate document data without generating the document"""
        try:
            validation_errors = template_engine.validate_template_data(
                request.document_type, 
                request.data
            )
            
            return {
                "valid": len(validation_errors) == 0,
                "errors": validation_errors,
                "document_type": request.document_type
            }
            
        except Exception as e:
            logger.error(f"Error validating document data: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to validate document data")
    
    @app.post("/api/v1/documents/detect-pii", response_model=PIIDetectionResponse)
    async def detect_pii(request: PIIDetectionRequest):
        """Detect PII in document data"""
        try:
            pii_detector = get_pii_detector()
            
            # Detect PII in the document data
            detection_results = pii_detector.detect_pii_in_document(request.data)
            
            # Generate comprehensive report
            if detection_results:
                report = pii_detector.generate_pii_report(detection_results)
                
                return PIIDetectionResponse(
                    has_pii=True,
                    total_matches=report["summary"]["total_pii_matches"],
                    risk_level=report["summary"]["overall_risk_level"],
                    summary=report["summary"],
                    field_details=report["field_details"],
                    recommendations=report["recommendations"]
                )
            else:
                return PIIDetectionResponse(
                    has_pii=False,
                    total_matches=0,
                    risk_level="LOW",
                    summary={
                        "total_fields_with_pii": 0,
                        "total_pii_matches": 0,
                        "unique_pii_types": 0,
                        "pii_types_found": [],
                        "overall_risk_level": "LOW"
                    },
                    field_details={},
                    recommendations=["No PII detected - data appears safe for processing"]
                )
            
        except Exception as e:
            logger.error(f"Error detecting PII: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to detect PII")
    
    @app.post("/api/v1/documents/generate", response_model=GenerationResponse)
    async def generate_document(request: DocumentGenerationRequest):
        """Generate a single document"""
        start_time = datetime.now()
        
        try:
            # Generate document based on format
            if request.output_format == "pdf":
                generator = get_pdf_generator()
                output_path = generator.generate_pdf(
                    document_type=request.document_type,
                    data=request.data,
                    metadata=request.metadata
                )
            elif request.output_format == "docx":
                generator = get_docx_generator()
                output_path = generator.generate_docx(
                    document_type=request.document_type,
                    data=request.data,
                    metadata=request.metadata
                )
            else:
                raise HTTPException(status_code=400, detail="Invalid output format")
            
            # Calculate generation time
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Get file size
            file_size = output_path.stat().st_size if output_path.exists() else None
            
            return GenerationResponse(
                success=True,
                message="Document generated successfully",
                file_path=str(output_path),
                file_size=file_size,
                generation_time=generation_time
            )
            
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=f"Validation error: {e.message}")
        except DocumentGenerationError as e:
            raise HTTPException(status_code=500, detail=f"Generation error: {e.message}")
        except Exception as e:
            logger.error(f"Unexpected error generating document: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to generate document")
    
    @app.post("/api/v1/documents/generate-batch", response_model=BatchGenerationResponse)
    async def generate_batch_documents(request: BatchGenerationRequest):
        """Generate multiple documents in batch"""
        start_time = datetime.now()
        
        try:
            # Generate documents based on format
            if request.output_format == "pdf":
                generator = get_pdf_generator()
                output_paths = generator.generate_batch(
                    document_type=request.document_type,
                    data_list=request.data_list,
                    output_prefix=request.output_prefix
                )
            elif request.output_format == "docx":
                generator = get_docx_generator()
                output_paths = generator.generate_batch(
                    document_type=request.document_type,
                    data_list=request.data_list,
                    output_prefix=request.output_prefix
                )
            else:
                raise HTTPException(status_code=400, detail="Invalid output format")
            
            # Calculate generation time
            generation_time = (datetime.now() - start_time).total_seconds()
            
            return BatchGenerationResponse(
                success=True,
                message=f"Generated {len(output_paths)} documents successfully",
                files_generated=len(output_paths),
                total_files=len(request.data_list),
                file_paths=[str(path) for path in output_paths],
                generation_time=generation_time
            )
            
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=f"Validation error: {e.message}")
        except DocumentGenerationError as e:
            raise HTTPException(status_code=500, detail=f"Generation error: {e.message}")
        except Exception as e:
            logger.error(f"Unexpected error in batch generation: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to generate documents")
    
    @app.get("/api/v1/documents/download/{file_path:path}")
    async def download_file(file_path: str):
        """Download a generated document file"""
        try:
            full_path = Path(file_path)
            
            if not full_path.exists():
                raise HTTPException(status_code=404, detail="File not found")
            
            # Determine media type based on extension
            media_type = "application/pdf" if full_path.suffix == ".pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            
            return FileResponse(
                path=str(full_path),
                media_type=media_type,
                filename=full_path.name
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error downloading file {file_path}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to download file")
    
    # Exception handlers
    @app.exception_handler(APIError)
    async def api_error_handler(request, exc: APIError):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict()
        )
    
    @app.exception_handler(ValidationError)
    async def validation_error_handler(request, exc: ValidationError):
        return JSONResponse(
            status_code=400,
            content=exc.to_dict()
        )
    
    @app.exception_handler(DocumentGenerationError)
    async def generation_error_handler(request, exc: DocumentGenerationError):
        return JSONResponse(
            status_code=500,
            content=exc.to_dict()
        )
    
    logger.info("FastAPI application initialized successfully")
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    config = get_config()
    uvicorn.run(
        "rest_api:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug,
        log_level="info"
    )