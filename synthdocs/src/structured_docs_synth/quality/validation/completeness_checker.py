"""
Completeness checker for document content and metadata validation.
Validates document completeness against expected content patterns and requirements.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Union, Any, Pattern

from pydantic import BaseModel, Field, validator

from ...core.config import BaseConfig
from ...core.exceptions import ValidationError
from ...core.logging import get_logger

logger = get_logger(__name__)


class CompletenessLevel(Enum):
    """Document completeness levels"""
    MINIMAL = "minimal"
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    COMPLETE = "complete"


class ContentType(Enum):
    """Content types for completeness checking"""
    TEXT = "text"
    METADATA = "metadata"
    STRUCTURAL = "structural"
    VISUAL = "visual"
    SEMANTIC = "semantic"


class MissingContentSeverity(Enum):
    """Severity levels for missing content"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ContentRequirement:
    """Content requirement specification"""
    requirement_id: str
    content_type: ContentType
    pattern: Optional[Pattern] = None
    expected_count: Optional[int] = None
    min_count: Optional[int] = None
    max_count: Optional[int] = None
    required_fields: Set[str] = field(default_factory=set)
    optional_fields: Set[str] = field(default_factory=set)
    validation_rules: List[str] = field(default_factory=list)
    severity: MissingContentSeverity = MissingContentSeverity.MEDIUM
    description: str = ""


@dataclass
class MissingContent:
    """Missing content finding"""
    content_id: str
    requirement: ContentRequirement
    actual_count: int
    expected_count: Optional[int] = None
    missing_fields: Set[str] = field(default_factory=set)
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "content_id": self.content_id,
            "requirement_id": self.requirement.requirement_id,
            "severity": self.requirement.severity.value,
            "actual_count": self.actual_count,
            "expected_count": self.expected_count,
            "missing_fields": list(self.missing_fields),
            "suggestions": self.suggestions,
            "description": self.requirement.description
        }


@dataclass
class CompletenessReport:
    """Document completeness assessment report"""
    document_id: str
    timestamp: datetime
    completeness_level: CompletenessLevel
    completeness_score: float
    missing_content: List[MissingContent]
    content_summary: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "document_id": self.document_id,
            "timestamp": self.timestamp.isoformat(),
            "completeness_level": self.completeness_level.value,
            "completeness_score": self.completeness_score,
            "missing_content": [mc.to_dict() for mc in self.missing_content],
            "content_summary": self.content_summary,
            "recommendations": self.recommendations
        }


@dataclass
class DocumentContent:
    """Document content container"""
    document_id: str
    text_content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    structural_elements: List[Dict[str, Any]] = field(default_factory=list)
    visual_elements: List[Dict[str, Any]] = field(default_factory=list)
    semantic_annotations: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_word_count(self) -> int:
        """Get total word count"""
        return len(self.text_content.split()) if self.text_content else 0
    
    def get_character_count(self) -> int:
        """Get total character count"""
        return len(self.text_content) if self.text_content else 0
    
    def has_metadata_field(self, field_name: str) -> bool:
        """Check if metadata field exists"""
        return field_name in self.metadata and self.metadata[field_name] is not None


class CompletenessCheckerConfig(BaseConfig):
    """Completeness checker configuration"""
    min_word_count: int = Field(default=100, description="Minimum word count for documents")
    min_paragraph_count: int = Field(default=3, description="Minimum paragraph count")
    required_metadata_fields: List[str] = Field(
        default=["title", "author", "date"],
        description="Required metadata fields"
    )
    text_quality_threshold: float = Field(default=0.7, description="Text quality threshold")
    
    # Completeness level thresholds
    minimal_threshold: float = Field(default=0.3, description="Minimal completeness threshold")
    basic_threshold: float = Field(default=0.5, description="Basic completeness threshold")
    standard_threshold: float = Field(default=0.7, description="Standard completeness threshold")
    comprehensive_threshold: float = Field(default=0.85, description="Comprehensive completeness threshold")
    
    # Content validation patterns
    email_pattern: str = Field(
        default=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        description="Email validation pattern"
    )
    phone_pattern: str = Field(
        default=r'^\+?[\d\s\-\(\)]{10,}$',
        description="Phone number validation pattern"
    )
    date_pattern: str = Field(
        default=r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}',
        description="Date validation pattern"
    )


class CompletenessChecker:
    """
    Document completeness checker.
    Validates document completeness against expected content patterns and requirements.
    """
    
    def __init__(self, config: Optional[CompletenessCheckerConfig] = None):
        """Initialize completeness checker"""
        self.config = config or CompletenessCheckerConfig()
        self.requirements = self._load_content_requirements()
        logger.info("Initialized CompletenessChecker")
    
    def check_document_completeness(
        self,
        document_content: DocumentContent,
        custom_requirements: Optional[List[ContentRequirement]] = None
    ) -> CompletenessReport:
        """
        Check document completeness against requirements.
        
        Args:
            document_content: Document content to check
            custom_requirements: Optional custom requirements
        
        Returns:
            Completeness assessment report
        """
        requirements = custom_requirements or self.requirements
        missing_content = []
        
        # Check each requirement
        for requirement in requirements:
            missing = self._check_requirement(document_content, requirement)
            if missing:
                missing_content.append(missing)
        
        # Calculate completeness score
        completeness_score = self._calculate_completeness_score(
            document_content, missing_content, requirements
        )
        
        # Determine completeness level
        completeness_level = self._determine_completeness_level(completeness_score)
        
        # Generate content summary
        content_summary = self._generate_content_summary(document_content)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(missing_content, document_content)
        
        return CompletenessReport(
            document_id=document_content.document_id,
            timestamp=datetime.now(),
            completeness_level=completeness_level,
            completeness_score=completeness_score,
            missing_content=missing_content,
            content_summary=content_summary,
            recommendations=recommendations
        )
    
    def _load_content_requirements(self) -> List[ContentRequirement]:
        """Load predefined content requirements"""
        requirements = []
        
        # Text content requirements
        requirements.append(ContentRequirement(
            requirement_id="min_text_length",
            content_type=ContentType.TEXT,
            min_count=self.config.min_word_count,
            severity=MissingContentSeverity.HIGH,
            description=f"Document must have at least {self.config.min_word_count} words"
        ))
        
        requirements.append(ContentRequirement(
            requirement_id="paragraph_structure",
            content_type=ContentType.STRUCTURAL,
            min_count=self.config.min_paragraph_count,
            severity=MissingContentSeverity.MEDIUM,
            description=f"Document must have at least {self.config.min_paragraph_count} paragraphs"
        ))
        
        # Metadata requirements
        requirements.append(ContentRequirement(
            requirement_id="required_metadata",
            content_type=ContentType.METADATA,
            required_fields=set(self.config.required_metadata_fields),
            severity=MissingContentSeverity.HIGH,
            description="Document must have required metadata fields"
        ))
        
        # Title requirement
        requirements.append(ContentRequirement(
            requirement_id="title_present",
            content_type=ContentType.METADATA,
            required_fields={"title"},
            severity=MissingContentSeverity.CRITICAL,
            description="Document must have a title"
        ))
        
        # Date format requirement
        requirements.append(ContentRequirement(
            requirement_id="valid_date_format",
            content_type=ContentType.METADATA,
            pattern=re.compile(self.config.date_pattern),
            severity=MissingContentSeverity.LOW,
            description="Dates should follow standard format"
        ))
        
        # Email format requirement
        requirements.append(ContentRequirement(
            requirement_id="valid_email_format",
            content_type=ContentType.METADATA,
            pattern=re.compile(self.config.email_pattern),
            severity=MissingContentSeverity.LOW,
            description="Email addresses should be properly formatted"
        ))
        
        # Phone format requirement
        requirements.append(ContentRequirement(
            requirement_id="valid_phone_format",
            content_type=ContentType.METADATA,
            pattern=re.compile(self.config.phone_pattern),
            severity=MissingContentSeverity.LOW,
            description="Phone numbers should be properly formatted"
        ))
        
        # Content quality requirements
        requirements.append(ContentRequirement(
            requirement_id="text_readability",
            content_type=ContentType.TEXT,
            severity=MissingContentSeverity.MEDIUM,
            description="Text should meet readability standards"
        ))
        
        # Structural requirements
        requirements.append(ContentRequirement(
            requirement_id="heading_structure",
            content_type=ContentType.STRUCTURAL,
            min_count=1,
            severity=MissingContentSeverity.MEDIUM,
            description="Document should have proper heading structure"
        ))
        
        return requirements
    
    def _check_requirement(
        self,
        document_content: DocumentContent,
        requirement: ContentRequirement
    ) -> Optional[MissingContent]:
        """Check a specific content requirement"""
        
        if requirement.content_type == ContentType.TEXT:
            return self._check_text_requirement(document_content, requirement)
        elif requirement.content_type == ContentType.METADATA:
            return self._check_metadata_requirement(document_content, requirement)
        elif requirement.content_type == ContentType.STRUCTURAL:
            return self._check_structural_requirement(document_content, requirement)
        elif requirement.content_type == ContentType.VISUAL:
            return self._check_visual_requirement(document_content, requirement)
        elif requirement.content_type == ContentType.SEMANTIC:
            return self._check_semantic_requirement(document_content, requirement)
        
        return None
    
    def _check_text_requirement(
        self,
        document_content: DocumentContent,
        requirement: ContentRequirement
    ) -> Optional[MissingContent]:
        """Check text content requirements"""
        
        if requirement.requirement_id == "min_text_length":
            word_count = document_content.get_word_count()
            min_count = requirement.min_count or 0
            
            if word_count < min_count:
                return MissingContent(
                    content_id=f"text_length_{document_content.document_id}",
                    requirement=requirement,
                    actual_count=word_count,
                    expected_count=min_count,
                    suggestions=[
                        f"Add more content to reach minimum {min_count} words",
                        "Expand existing paragraphs with more detail"
                    ]
                )
        
        elif requirement.requirement_id == "text_readability":
            readability_score = self._calculate_readability_score(document_content.text_content)
            
            if readability_score < self.config.text_quality_threshold:
                return MissingContent(
                    content_id=f"text_readability_{document_content.document_id}",
                    requirement=requirement,
                    actual_count=int(readability_score * 100),
                    expected_count=int(self.config.text_quality_threshold * 100),
                    suggestions=[
                        "Improve sentence structure and clarity",
                        "Use simpler vocabulary where appropriate",
                        "Break long sentences into shorter ones"
                    ]
                )
        
        return None
    
    def _check_metadata_requirement(
        self,
        document_content: DocumentContent,
        requirement: ContentRequirement
    ) -> Optional[MissingContent]:
        """Check metadata requirements"""
        
        missing_fields = set()
        
        # Check required fields
        for field in requirement.required_fields:
            if not document_content.has_metadata_field(field):
                missing_fields.add(field)
        
        # Check pattern validation for existing fields
        if requirement.pattern:
            for field_name, field_value in document_content.metadata.items():
                if isinstance(field_value, str) and field_value:
                    if requirement.requirement_id == "valid_email_format" and "@" in field_value:
                        if not requirement.pattern.match(field_value):
                            missing_fields.add(f"valid_{field_name}")
                    elif requirement.requirement_id == "valid_phone_format" and any(char.isdigit() for char in field_value):
                        if not requirement.pattern.match(field_value):
                            missing_fields.add(f"valid_{field_name}")
                    elif requirement.requirement_id == "valid_date_format" and "date" in field_name.lower():
                        if not requirement.pattern.match(field_value):
                            missing_fields.add(f"valid_{field_name}")
        
        if missing_fields:
            suggestions = []
            for field in missing_fields:
                if field.startswith("valid_"):
                    suggestions.append(f"Fix format for {field.replace('valid_', '')}")
                else:
                    suggestions.append(f"Add {field} to document metadata")
            
            return MissingContent(
                content_id=f"metadata_{requirement.requirement_id}_{document_content.document_id}",
                requirement=requirement,
                actual_count=len(requirement.required_fields) - len(missing_fields),
                expected_count=len(requirement.required_fields),
                missing_fields=missing_fields,
                suggestions=suggestions
            )
        
        return None
    
    def _check_structural_requirement(
        self,
        document_content: DocumentContent,
        requirement: ContentRequirement
    ) -> Optional[MissingContent]:
        """Check structural requirements"""
        
        if requirement.requirement_id == "paragraph_structure":
            paragraph_count = self._count_paragraphs(document_content.text_content)
            min_count = requirement.min_count or 0
            
            if paragraph_count < min_count:
                return MissingContent(
                    content_id=f"paragraph_count_{document_content.document_id}",
                    requirement=requirement,
                    actual_count=paragraph_count,
                    expected_count=min_count,
                    suggestions=[
                        f"Add more paragraphs to reach minimum {min_count}",
                        "Break long text blocks into separate paragraphs"
                    ]
                )
        
        elif requirement.requirement_id == "heading_structure":
            heading_count = len([elem for elem in document_content.structural_elements 
                               if elem.get("type") == "heading"])
            min_count = requirement.min_count or 0
            
            if heading_count < min_count:
                return MissingContent(
                    content_id=f"heading_count_{document_content.document_id}",
                    requirement=requirement,
                    actual_count=heading_count,
                    expected_count=min_count,
                    suggestions=[
                        "Add section headings to organize content",
                        "Use hierarchical heading structure"
                    ]
                )
        
        return None
    
    def _check_visual_requirement(
        self,
        document_content: DocumentContent,
        requirement: ContentRequirement
    ) -> Optional[MissingContent]:
        """Check visual content requirements"""
        
        visual_count = len(document_content.visual_elements)
        
        if requirement.min_count and visual_count < requirement.min_count:
            return MissingContent(
                content_id=f"visual_content_{document_content.document_id}",
                requirement=requirement,
                actual_count=visual_count,
                expected_count=requirement.min_count,
                suggestions=[
                    "Add relevant images or figures",
                    "Include charts or diagrams where appropriate"
                ]
            )
        
        return None
    
    def _check_semantic_requirement(
        self,
        document_content: DocumentContent,
        requirement: ContentRequirement
    ) -> Optional[MissingContent]:
        """Check semantic content requirements"""
        
        semantic_count = len(document_content.semantic_annotations)
        
        if requirement.min_count and semantic_count < requirement.min_count:
            return MissingContent(
                content_id=f"semantic_content_{document_content.document_id}",
                requirement=requirement,
                actual_count=semantic_count,
                expected_count=requirement.min_count,
                suggestions=[
                    "Add semantic annotations",
                    "Include entity recognition markup"
                ]
            )
        
        return None
    
    def _calculate_readability_score(self, text: str) -> float:
        """Calculate text readability score (simplified)"""
        if not text or len(text.strip()) == 0:
            return 0.0
        
        # Simple readability metrics
        words = text.split()
        sentences = text.count('.') + text.count('!') + text.count('?')
        sentences = max(1, sentences)  # Avoid division by zero
        
        avg_sentence_length = len(words) / sentences
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Simple scoring based on averages
        sentence_score = max(0, 1.0 - (abs(avg_sentence_length - 15) / 30))  # Target ~15 words per sentence
        word_score = max(0, 1.0 - (abs(avg_word_length - 5) / 10))  # Target ~5 chars per word
        
        return (sentence_score + word_score) / 2
    
    def _count_paragraphs(self, text: str) -> int:
        """Count paragraphs in text"""
        if not text:
            return 0
        
        # Split by double newlines or single newlines followed by significant whitespace
        paragraphs = re.split(r'\n\s*\n|\n\s{2,}', text.strip())
        # Filter out very short paragraphs (likely not real paragraphs)
        meaningful_paragraphs = [p for p in paragraphs if len(p.strip()) > 20]
        
        return len(meaningful_paragraphs)
    
    def _calculate_completeness_score(
        self,
        document_content: DocumentContent,
        missing_content: List[MissingContent],
        requirements: List[ContentRequirement]
    ) -> float:
        """Calculate overall completeness score"""
        if not requirements:
            return 1.0
        
        # Calculate penalty based on missing content severity
        total_penalty = 0.0
        max_penalty = 0.0
        
        for requirement in requirements:
            # Calculate max possible penalty for this requirement
            if requirement.severity == MissingContentSeverity.CRITICAL:
                req_max_penalty = 0.4
            elif requirement.severity == MissingContentSeverity.HIGH:
                req_max_penalty = 0.3
            elif requirement.severity == MissingContentSeverity.MEDIUM:
                req_max_penalty = 0.2
            else:  # LOW
                req_max_penalty = 0.1
            
            max_penalty += req_max_penalty
            
            # Check if this requirement has missing content
            for missing in missing_content:
                if missing.requirement.requirement_id == requirement.requirement_id:
                    # Calculate actual penalty based on severity
                    total_penalty += req_max_penalty
                    break
        
        # Calculate score
        if max_penalty == 0:
            return 1.0
        
        completeness_score = max(0.0, 1.0 - (total_penalty / max_penalty))
        
        # Apply content quality bonuses
        word_count = document_content.get_word_count()
        if word_count > self.config.min_word_count * 2:
            completeness_score = min(1.0, completeness_score + 0.1)
        
        if len(document_content.structural_elements) > 5:
            completeness_score = min(1.0, completeness_score + 0.05)
        
        if len(document_content.visual_elements) > 0:
            completeness_score = min(1.0, completeness_score + 0.05)
        
        return completeness_score
    
    def _determine_completeness_level(self, score: float) -> CompletenessLevel:
        """Determine completeness level from score"""
        if score >= self.config.comprehensive_threshold:
            return CompletenessLevel.COMPLETE
        elif score >= self.config.standard_threshold:
            return CompletenessLevel.COMPREHENSIVE
        elif score >= self.config.basic_threshold:
            return CompletenessLevel.STANDARD
        elif score >= self.config.minimal_threshold:
            return CompletenessLevel.BASIC
        else:
            return CompletenessLevel.MINIMAL
    
    def _generate_content_summary(self, document_content: DocumentContent) -> Dict[str, Any]:
        """Generate content summary statistics"""
        return {
            "word_count": document_content.get_word_count(),
            "character_count": document_content.get_character_count(),
            "paragraph_count": self._count_paragraphs(document_content.text_content),
            "metadata_fields": len(document_content.metadata),
            "structural_elements": len(document_content.structural_elements),
            "visual_elements": len(document_content.visual_elements),
            "semantic_annotations": len(document_content.semantic_annotations),
            "has_title": document_content.has_metadata_field("title"),
            "has_author": document_content.has_metadata_field("author"),
            "has_date": document_content.has_metadata_field("date"),
            "readability_score": self._calculate_readability_score(document_content.text_content)
        }
    
    def _generate_recommendations(
        self,
        missing_content: List[MissingContent],
        document_content: DocumentContent
    ) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Priority recommendations based on critical/high severity issues
        critical_issues = [mc for mc in missing_content if mc.requirement.severity == MissingContentSeverity.CRITICAL]
        high_issues = [mc for mc in missing_content if mc.requirement.severity == MissingContentSeverity.HIGH]
        
        if critical_issues:
            recommendations.append("Address critical missing content first:")
            for issue in critical_issues[:3]:  # Top 3 critical issues
                recommendations.extend(issue.suggestions)
        
        if high_issues:
            recommendations.append("Important improvements needed:")
            for issue in high_issues[:3]:  # Top 3 high-priority issues
                recommendations.extend(issue.suggestions)
        
        # General recommendations based on content analysis
        word_count = document_content.get_word_count()
        if word_count < self.config.min_word_count:
            recommendations.append(f"Expand content to at least {self.config.min_word_count} words")
        
        if not document_content.structural_elements:
            recommendations.append("Add structural elements (headings, lists, tables)")
        
        if not document_content.visual_elements:
            recommendations.append("Consider adding visual elements (images, charts, diagrams)")
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        seen = set()
        for rec in recommendations:
            if rec not in seen:
                unique_recommendations.append(rec)
                seen.add(rec)
        
        return unique_recommendations[:10]  # Limit to top 10 recommendations
    
    def batch_check_completeness(
        self,
        documents: List[DocumentContent],
        custom_requirements: Optional[List[ContentRequirement]] = None
    ) -> Dict[str, Any]:
        """Check completeness for multiple documents"""
        results = []
        
        for document in documents:
            try:
                result = self.check_document_completeness(document, custom_requirements)
                results.append({
                    "document_id": document.document_id,
                    "result": result
                })
            except Exception as e:
                logger.error(f"Failed to check completeness for document {document.document_id}: {str(e)}")
                results.append({
                    "document_id": document.document_id,
                    "error": str(e)
                })
        
        # Calculate aggregate statistics
        successful_results = [r["result"] for r in results if "result" in r]
        
        if successful_results:
            scores = [r.completeness_score for r in successful_results]
            levels = [r.completeness_level for r in successful_results]
            
            level_counts = {}
            for level in levels:
                level_counts[level.value] = level_counts.get(level.value, 0) + 1
            
            aggregate_stats = {
                "total_documents": len(documents),
                "successful_checks": len(successful_results),
                "failed_checks": len(documents) - len(successful_results),
                "mean_completeness_score": statistics.mean(scores),
                "median_completeness_score": statistics.median(scores),
                "min_completeness_score": min(scores),
                "max_completeness_score": max(scores),
                "completeness_levels": level_counts,
                "documents_needing_improvement": len([r for r in successful_results if r.completeness_score < 0.7])
            }
        else:
            aggregate_stats = {
                "total_documents": len(documents),
                "successful_checks": 0,
                "failed_checks": len(documents),
                "error": "No successful completeness checks"
            }
        
        return {
            "results": results,
            "aggregate_statistics": aggregate_stats
        }


def create_completeness_checker(
    config: Optional[Union[Dict[str, Any], CompletenessCheckerConfig]] = None
) -> CompletenessChecker:
    """Factory function to create completeness checker"""
    if isinstance(config, dict):
        config = CompletenessCheckerConfig(**config)
    return CompletenessChecker(config)


def check_completeness_sample() -> CompletenessReport:
    """Generate sample completeness check for testing"""
    checker = create_completeness_checker()
    
    # Create sample document content
    sample_content = DocumentContent(
        document_id="sample_doc_001",
        text_content="This is a sample document with some content. It has multiple paragraphs to demonstrate the completeness checking functionality. The text should be long enough to meet minimum requirements.",
        metadata={
            "title": "Sample Document",
            "author": "John Doe",
            "date": "2024-01-15"
        },
        structural_elements=[
            {"type": "heading", "level": 1, "content": "Introduction"},
            {"type": "paragraph", "content": "First paragraph..."},
            {"type": "paragraph", "content": "Second paragraph..."}
        ]
    )
    
    return checker.check_document_completeness(sample_content)