"""
Document type definitions and models for the SDK.
Provides standardized document type definitions and data structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any

from pydantic import BaseModel, Field, validator


class DocumentFormat(Enum):
    """Supported document formats"""
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    MARKDOWN = "markdown"
    LATEX = "latex"
    TXT = "txt"
    JSON = "json"


class DocumentType(Enum):
    """Document types for generation"""
    ACADEMIC_PAPER = "academic_paper"
    BUSINESS_FORM = "business_form"
    TECHNICAL_MANUAL = "technical_manual"
    FINANCIAL_REPORT = "financial_report"
    LEGAL_DOCUMENT = "legal_document"
    MEDICAL_RECORD = "medical_record"
    INVOICE = "invoice"
    RESUME = "resume"
    PRESENTATION = "presentation"
    NEWSPAPER = "newspaper"
    RESEARCH_PROPOSAL = "research_proposal"
    PATENT_APPLICATION = "patent_application"
    INSURANCE_CLAIM = "insurance_claim"
    TAX_DOCUMENT = "tax_document"
    PRODUCT_CATALOG = "product_catalog"


class LayoutComplexity(Enum):
    """Layout complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"


class ContentQuality(Enum):
    """Content quality levels"""
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    PREMIUM = "premium"


class Language(Enum):
    """Supported languages"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    RUSSIAN = "ru"


@dataclass
class BoundingBox:
    """Bounding box for document elements"""
    x: float
    y: float
    width: float
    height: float
    
    def area(self) -> float:
        """Calculate area of bounding box"""
        return self.width * self.height
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height
        }


@dataclass
class DocumentElement:
    """Document element representation"""
    element_id: str
    element_type: str
    bbox: BoundingBox
    content: Optional[str] = None
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "element_id": self.element_id,
            "element_type": self.element_type,
            "bbox": self.bbox.to_dict(),
            "content": self.content,
            "confidence": self.confidence,
            "attributes": self.attributes
        }


class DocumentMetadata(BaseModel):
    """Document metadata"""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    language: Language = Language.ENGLISH
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    character_count: Optional[int] = None
    custom_fields: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class PageSize(BaseModel):
    """Page size specification"""
    width: float = Field(description="Page width in points")
    height: float = Field(description="Page height in points")
    name: Optional[str] = Field(default=None, description="Standard page size name")
    
    @classmethod
    def letter(cls) -> 'PageSize':
        """Standard US Letter size"""
        return cls(width=612, height=792, name="Letter")
    
    @classmethod
    def a4(cls) -> 'PageSize':
        """Standard A4 size"""
        return cls(width=595, height=842, name="A4")
    
    @classmethod
    def legal(cls) -> 'PageSize':
        """Standard US Legal size"""
        return cls(width=612, height=1008, name="Legal")


class DocumentContent(BaseModel):
    """Document content structure"""
    text: str = Field(description="Main text content")
    elements: List[DocumentElement] = Field(default_factory=list, description="Document elements")
    images: List[Dict[str, Any]] = Field(default_factory=list, description="Image elements")
    tables: List[Dict[str, Any]] = Field(default_factory=list, description="Table elements")
    annotations: List[Dict[str, Any]] = Field(default_factory=list, description="Annotations")
    
    def get_word_count(self) -> int:
        """Get total word count"""
        return len(self.text.split()) if self.text else 0
    
    def get_character_count(self) -> int:
        """Get total character count"""
        return len(self.text) if self.text else 0


class GenerationRequest(BaseModel):
    """Document generation request"""
    document_type: DocumentType
    document_format: DocumentFormat = DocumentFormat.PDF
    count: int = Field(default=1, ge=1, le=1000, description="Number of documents to generate")
    layout_complexity: LayoutComplexity = LayoutComplexity.MODERATE
    content_quality: ContentQuality = ContentQuality.STANDARD
    language: Language = Language.ENGLISH
    page_size: PageSize = Field(default_factory=PageSize.letter)
    metadata: Optional[DocumentMetadata] = None
    custom_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Generation options
    include_ocr_noise: bool = Field(default=False, description="Add OCR simulation noise")
    include_annotations: bool = Field(default=True, description="Include ground truth annotations")
    include_metadata: bool = Field(default=True, description="Include document metadata")
    
    # Privacy options
    anonymize_pii: bool = Field(default=True, description="Anonymize personally identifiable information")
    comply_with_gdpr: bool = Field(default=True, description="Ensure GDPR compliance")
    comply_with_hipaa: bool = Field(default=False, description="Ensure HIPAA compliance")
    
    class Config:
        use_enum_values = True
    
    @validator('count')
    def validate_count(cls, v):
        """Validate document count"""
        if v < 1 or v > 1000:
            raise ValueError("Count must be between 1 and 1000")
        return v


class DocumentResponse(BaseModel):
    """Document generation response"""
    document_id: str = Field(description="Unique document identifier")
    document_type: DocumentType
    document_format: DocumentFormat
    metadata: DocumentMetadata
    content: Optional[DocumentContent] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    generation_time: Optional[float] = None
    quality_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class BatchGenerationRequest(BaseModel):
    """Batch document generation request"""
    requests: List[GenerationRequest] = Field(description="List of generation requests")
    batch_id: Optional[str] = None
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    output_directory: Optional[str] = None
    compression: bool = Field(default=False, description="Compress output files")
    
    @validator('requests')
    def validate_requests(cls, v):
        """Validate generation requests"""
        if not v:
            raise ValueError("At least one generation request is required")
        if len(v) > 100:
            raise ValueError("Maximum 100 requests per batch")
        return v


class BatchGenerationResponse(BaseModel):
    """Batch document generation response"""
    batch_id: str = Field(description="Unique batch identifier")
    total_documents: int = Field(description="Total number of documents requested")
    successful_documents: int = Field(description="Number of successfully generated documents")
    failed_documents: int = Field(description="Number of failed document generations")
    documents: List[DocumentResponse] = Field(description="Generated documents")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Generation errors")
    total_generation_time: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_documents == 0:
            return 0.0
        return self.successful_documents / self.total_documents


class ValidationRequest(BaseModel):
    """Document validation request"""
    document_content: DocumentContent
    document_type: DocumentType
    validation_types: List[str] = Field(
        default=["structural", "completeness", "semantic"],
        description="Types of validation to perform"
    )
    strict_mode: bool = Field(default=False, description="Enable strict validation mode")
    
    class Config:
        use_enum_values = True


class ValidationResponse(BaseModel):
    """Document validation response"""
    document_id: str
    is_valid: bool
    validation_score: float = Field(ge=0.0, le=1.0)
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    validation_time: Optional[float] = None


class ExportRequest(BaseModel):
    """Document export request"""
    document_ids: List[str] = Field(description="List of document IDs to export")
    export_format: str = Field(description="Export format (coco, yolo, pascal_voc, etc.)")
    output_path: str = Field(description="Output directory path")
    include_images: bool = Field(default=True, description="Include image files")
    include_annotations: bool = Field(default=True, description="Include annotation files")
    compression: bool = Field(default=False, description="Compress exported files")
    
    @validator('document_ids')
    def validate_document_ids(cls, v):
        """Validate document IDs"""
        if not v:
            raise ValueError("At least one document ID is required")
        return v


class ExportResponse(BaseModel):
    """Document export response"""
    export_id: str
    exported_documents: int
    failed_exports: int
    output_path: str
    file_size: Optional[int] = None
    export_time: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate export success rate"""
        total = self.exported_documents + self.failed_exports
        if total == 0:
            return 0.0
        return self.exported_documents / total


# Standard document templates
DOCUMENT_TEMPLATES = {
    DocumentType.ACADEMIC_PAPER: {
        "required_sections": ["title", "abstract", "introduction", "methodology", "results", "conclusion", "references"],
        "optional_sections": ["acknowledgments", "appendix"],
        "typical_page_count": (8, 20),
        "typical_word_count": (3000, 8000)
    },
    DocumentType.BUSINESS_FORM: {
        "required_sections": ["header", "form_fields", "signature"],
        "optional_sections": ["instructions", "legal_text"],
        "typical_page_count": (1, 4),
        "typical_word_count": (200, 1000)
    },
    DocumentType.TECHNICAL_MANUAL: {
        "required_sections": ["title", "table_of_contents", "introduction", "procedures", "troubleshooting"],
        "optional_sections": ["glossary", "index", "appendix"],
        "typical_page_count": (10, 50),
        "typical_word_count": (5000, 25000)
    },
    DocumentType.FINANCIAL_REPORT: {
        "required_sections": ["executive_summary", "financial_statements", "notes", "auditor_report"],
        "optional_sections": ["management_discussion", "risk_factors"],
        "typical_page_count": (20, 100),
        "typical_word_count": (10000, 50000)
    },
    DocumentType.LEGAL_DOCUMENT: {
        "required_sections": ["header", "parties", "terms", "signatures"],
        "optional_sections": ["exhibits", "schedules"],
        "typical_page_count": (2, 20),
        "typical_word_count": (1000, 10000)
    }
}


def get_document_template(document_type: DocumentType) -> Dict[str, Any]:
    """Get template information for a document type"""
    return DOCUMENT_TEMPLATES.get(document_type, {})


def create_generation_request(
    document_type: Union[str, DocumentType],
    count: int = 1,
    **kwargs
) -> GenerationRequest:
    """
    Create a generation request with defaults.
    
    Args:
        document_type: Type of document to generate
        count: Number of documents to generate
        **kwargs: Additional configuration options
    
    Returns:
        GenerationRequest object
    """
    if isinstance(document_type, str):
        document_type = DocumentType(document_type)
    
    return GenerationRequest(
        document_type=document_type,
        count=count,
        **kwargs
    )


def create_batch_request(
    requests: List[Union[GenerationRequest, Dict[str, Any]]],
    **kwargs
) -> BatchGenerationRequest:
    """
    Create a batch generation request.
    
    Args:
        requests: List of generation requests
        **kwargs: Additional batch configuration
    
    Returns:
        BatchGenerationRequest object
    """
    processed_requests = []
    for req in requests:
        if isinstance(req, dict):
            processed_requests.append(GenerationRequest(**req))
        else:
            processed_requests.append(req)
    
    return BatchGenerationRequest(
        requests=processed_requests,
        **kwargs
    )