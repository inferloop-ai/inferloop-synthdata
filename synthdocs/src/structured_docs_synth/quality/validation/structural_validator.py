"""
Structural validator for document layout and hierarchy validation.
Validates document structure against predefined schemas and rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Union, Any, Tuple

from pydantic import BaseModel, Field, validator

from ...core.config import BaseConfig
from ...core.exceptions import ValidationError
from ...core.logging import get_logger

logger = get_logger(__name__)


class DocumentType(Enum):
    """Document types for structural validation"""
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


class ElementType(Enum):
    """Document element types"""
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    SIDEBAR = "sidebar"
    FORM_FIELD = "form_field"
    SIGNATURE = "signature"
    LOGO = "logo"
    WATERMARK = "watermark"
    PAGE_NUMBER = "page_number"


class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class BoundingBox:
    """Element bounding box"""
    x: float
    y: float
    width: float
    height: float
    
    def area(self) -> float:
        """Calculate area"""
        return self.width * self.height
    
    def overlaps(self, other: 'BoundingBox', threshold: float = 0.1) -> bool:
        """Check if bounding boxes overlap"""
        x_overlap = max(0, min(self.x + self.width, other.x + other.width) - max(self.x, other.x))
        y_overlap = max(0, min(self.y + self.height, other.y + other.height) - max(self.y, other.y))
        overlap_area = x_overlap * y_overlap
        
        min_area = min(self.area(), other.area())
        return (overlap_area / min_area) > threshold if min_area > 0 else False


@dataclass
class DocumentElement:
    """Document structure element"""
    element_id: str
    element_type: ElementType
    bbox: BoundingBox
    content: Optional[str] = None
    level: Optional[int] = None  # For hierarchical elements like headings
    page_number: int = 1
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "element_id": self.element_id,
            "element_type": self.element_type.value,
            "bbox": {
                "x": self.bbox.x,
                "y": self.bbox.y,
                "width": self.bbox.width,
                "height": self.bbox.height
            },
            "content": self.content,
            "level": self.level,
            "page_number": self.page_number,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "attributes": self.attributes
        }


@dataclass
class ValidationIssue:
    """Structural validation issue"""
    issue_id: str
    severity: ValidationSeverity
    message: str
    element_id: Optional[str] = None
    rule_name: str = ""
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "issue_id": self.issue_id,
            "severity": self.severity.value,
            "message": self.message,
            "element_id": self.element_id,
            "rule_name": self.rule_name,
            "suggestions": self.suggestions
        }


@dataclass
class ValidationResult:
    """Structural validation result"""
    document_type: DocumentType
    is_valid: bool
    validation_score: float
    issues: List[ValidationIssue]
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "document_type": self.document_type.value,
            "is_valid": self.is_valid,
            "validation_score": self.validation_score,
            "issues": [issue.to_dict() for issue in self.issues],
            "summary": self.summary
        }


@dataclass
class DocumentSchema:
    """Document structure schema definition"""
    document_type: DocumentType
    required_elements: Set[ElementType]
    forbidden_elements: Set[ElementType]
    element_hierarchy: Dict[ElementType, List[ElementType]]
    element_constraints: Dict[ElementType, Dict[str, Any]]
    nesting_rules: Dict[ElementType, Set[ElementType]]
    ordering_rules: List[Tuple[ElementType, ElementType]]  # (must_come_before, element)


class StructuralValidatorConfig(BaseConfig):
    """Structural validator configuration"""
    strict_mode: bool = Field(default=False, description="Enable strict validation mode")
    allow_unknown_elements: bool = Field(default=True, description="Allow unknown element types")
    overlap_threshold: float = Field(default=0.1, description="Overlap threshold for element validation")
    hierarchy_depth_limit: int = Field(default=10, description="Maximum hierarchy depth")
    
    # Scoring weights
    required_element_weight: float = Field(default=0.3, description="Weight for required elements")
    forbidden_element_weight: float = Field(default=0.2, description="Weight for forbidden elements")
    hierarchy_weight: float = Field(default=0.25, description="Weight for hierarchy validation")
    nesting_weight: float = Field(default=0.15, description="Weight for nesting validation")
    ordering_weight: float = Field(default=0.1, description="Weight for ordering validation")


class StructuralValidator:
    """
    Document structural validator.
    Validates document structure against predefined schemas and rules.
    """
    
    def __init__(self, config: Optional[StructuralValidatorConfig] = None):
        """Initialize structural validator"""
        self.config = config or StructuralValidatorConfig()
        self.schemas = self._load_document_schemas()
        logger.info("Initialized StructuralValidator")
    
    def validate_document_structure(
        self,
        elements: List[DocumentElement],
        document_type: DocumentType,
        custom_schema: Optional[DocumentSchema] = None
    ) -> ValidationResult:
        """
        Validate document structure against schema.
        
        Args:
            elements: List of document elements
            document_type: Type of document to validate
            custom_schema: Optional custom schema to use
        
        Returns:
            Validation result with issues and score
        """
        schema = custom_schema or self.schemas.get(document_type)
        if not schema:
            raise ValueError(f"No schema available for document type: {document_type}")
        
        issues = []
        
        # Create element lookup
        element_map = {elem.element_id: elem for elem in elements}
        type_map = {}
        for elem in elements:
            if elem.element_type not in type_map:
                type_map[elem.element_type] = []
            type_map[elem.element_type].append(elem)
        
        # Validate required elements
        required_issues = self._validate_required_elements(type_map, schema)
        issues.extend(required_issues)
        
        # Validate forbidden elements
        forbidden_issues = self._validate_forbidden_elements(type_map, schema)
        issues.extend(forbidden_issues)
        
        # Validate element hierarchy
        hierarchy_issues = self._validate_element_hierarchy(elements, element_map, schema)
        issues.extend(hierarchy_issues)
        
        # Validate nesting rules
        nesting_issues = self._validate_nesting_rules(elements, element_map, schema)
        issues.extend(nesting_issues)
        
        # Validate ordering rules
        ordering_issues = self._validate_ordering_rules(elements, schema)
        issues.extend(ordering_issues)
        
        # Validate element constraints
        constraint_issues = self._validate_element_constraints(elements, schema)
        issues.extend(constraint_issues)
        
        # Validate spatial consistency
        spatial_issues = self._validate_spatial_consistency(elements)
        issues.extend(spatial_issues)
        
        # Calculate validation score
        validation_score = self._calculate_validation_score(issues, elements, schema)
        is_valid = validation_score >= 0.7 and not any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)
        
        # Generate summary
        summary = self._generate_validation_summary(issues, elements, schema)
        
        return ValidationResult(
            document_type=document_type,
            is_valid=is_valid,
            validation_score=validation_score,
            issues=issues,
            summary=summary
        )
    
    def _load_document_schemas(self) -> Dict[DocumentType, DocumentSchema]:
        """Load predefined document schemas"""
        schemas = {}
        
        # Academic Paper Schema
        schemas[DocumentType.ACADEMIC_PAPER] = DocumentSchema(
            document_type=DocumentType.ACADEMIC_PAPER,
            required_elements={
                ElementType.TITLE,
                ElementType.HEADING,
                ElementType.PARAGRAPH
            },
            forbidden_elements={
                ElementType.FORM_FIELD,
                ElementType.SIGNATURE
            },
            element_hierarchy={
                ElementType.TITLE: [ElementType.HEADING],
                ElementType.HEADING: [ElementType.PARAGRAPH, ElementType.LIST, ElementType.TABLE, ElementType.FIGURE]
            },
            element_constraints={
                ElementType.TITLE: {"max_count": 1, "min_font_size": 16},
                ElementType.HEADING: {"max_level": 6},
                ElementType.FIGURE: {"requires_caption": True}
            },
            nesting_rules={
                ElementType.TITLE: set(),
                ElementType.HEADING: {ElementType.PARAGRAPH, ElementType.LIST}
            },
            ordering_rules=[
                (ElementType.TITLE, ElementType.HEADING),
                (ElementType.HEADING, ElementType.PARAGRAPH)
            ]
        )
        
        # Business Form Schema
        schemas[DocumentType.BUSINESS_FORM] = DocumentSchema(
            document_type=DocumentType.BUSINESS_FORM,
            required_elements={
                ElementType.TITLE,
                ElementType.FORM_FIELD
            },
            forbidden_elements={
                ElementType.FIGURE,
                ElementType.TABLE
            },
            element_hierarchy={
                ElementType.TITLE: [ElementType.FORM_FIELD, ElementType.PARAGRAPH]
            },
            element_constraints={
                ElementType.TITLE: {"max_count": 1},
                ElementType.FORM_FIELD: {"min_count": 1},
                ElementType.SIGNATURE: {"max_count": 3}
            },
            nesting_rules={
                ElementType.TITLE: set(),
                ElementType.FORM_FIELD: set()
            },
            ordering_rules=[
                (ElementType.TITLE, ElementType.FORM_FIELD),
                (ElementType.FORM_FIELD, ElementType.SIGNATURE)
            ]
        )
        
        # Technical Manual Schema
        schemas[DocumentType.TECHNICAL_MANUAL] = DocumentSchema(
            document_type=DocumentType.TECHNICAL_MANUAL,
            required_elements={
                ElementType.TITLE,
                ElementType.HEADING,
                ElementType.PARAGRAPH,
                ElementType.LIST
            },
            forbidden_elements={
                ElementType.FORM_FIELD,
                ElementType.SIGNATURE
            },
            element_hierarchy={
                ElementType.TITLE: [ElementType.HEADING],
                ElementType.HEADING: [ElementType.PARAGRAPH, ElementType.LIST, ElementType.TABLE, ElementType.FIGURE]
            },
            element_constraints={
                ElementType.TITLE: {"max_count": 1},
                ElementType.HEADING: {"max_level": 4},
                ElementType.LIST: {"min_items": 2}
            },
            nesting_rules={
                ElementType.HEADING: {ElementType.PARAGRAPH, ElementType.LIST, ElementType.TABLE}
            },
            ordering_rules=[
                (ElementType.TITLE, ElementType.HEADING)
            ]
        )
        
        # Financial Report Schema
        schemas[DocumentType.FINANCIAL_REPORT] = DocumentSchema(
            document_type=DocumentType.FINANCIAL_REPORT,
            required_elements={
                ElementType.TITLE,
                ElementType.HEADING,
                ElementType.TABLE,
                ElementType.PARAGRAPH
            },
            forbidden_elements={
                ElementType.FORM_FIELD,
                ElementType.SIGNATURE
            },
            element_hierarchy={
                ElementType.TITLE: [ElementType.HEADING],
                ElementType.HEADING: [ElementType.PARAGRAPH, ElementType.TABLE, ElementType.FIGURE]
            },
            element_constraints={
                ElementType.TITLE: {"max_count": 1},
                ElementType.TABLE: {"min_count": 1, "requires_caption": True},
                ElementType.FIGURE: {"requires_caption": True}
            },
            nesting_rules={
                ElementType.HEADING: {ElementType.PARAGRAPH, ElementType.TABLE}
            },
            ordering_rules=[
                (ElementType.TITLE, ElementType.HEADING),
                (ElementType.HEADING, ElementType.TABLE)
            ]
        )
        
        # Invoice Schema
        schemas[DocumentType.INVOICE] = DocumentSchema(
            document_type=DocumentType.INVOICE,
            required_elements={
                ElementType.TITLE,
                ElementType.TABLE,
                ElementType.PARAGRAPH
            },
            forbidden_elements={
                ElementType.FIGURE,
                ElementType.LIST
            },
            element_hierarchy={
                ElementType.TITLE: [ElementType.PARAGRAPH, ElementType.TABLE]
            },
            element_constraints={
                ElementType.TITLE: {"max_count": 1},
                ElementType.TABLE: {"min_count": 1, "max_count": 2},
                ElementType.LOGO: {"max_count": 1}
            },
            nesting_rules={},
            ordering_rules=[
                (ElementType.LOGO, ElementType.TITLE),
                (ElementType.TITLE, ElementType.TABLE)
            ]
        )
        
        return schemas
    
    def _validate_required_elements(
        self,
        type_map: Dict[ElementType, List[DocumentElement]],
        schema: DocumentSchema
    ) -> List[ValidationIssue]:
        """Validate presence of required elements"""
        issues = []
        
        for required_type in schema.required_elements:
            if required_type not in type_map or not type_map[required_type]:
                issues.append(ValidationIssue(
                    issue_id=f"missing_required_{required_type.value}",
                    severity=ValidationSeverity.ERROR,
                    message=f"Required element type '{required_type.value}' is missing",
                    rule_name="required_elements",
                    suggestions=[f"Add at least one {required_type.value} element to the document"]
                ))
        
        return issues
    
    def _validate_forbidden_elements(
        self,
        type_map: Dict[ElementType, List[DocumentElement]],
        schema: DocumentSchema
    ) -> List[ValidationIssue]:
        """Validate absence of forbidden elements"""
        issues = []
        
        for forbidden_type in schema.forbidden_elements:
            if forbidden_type in type_map and type_map[forbidden_type]:
                for element in type_map[forbidden_type]:
                    issues.append(ValidationIssue(
                        issue_id=f"forbidden_{forbidden_type.value}_{element.element_id}",
                        severity=ValidationSeverity.WARNING,
                        message=f"Forbidden element type '{forbidden_type.value}' found",
                        element_id=element.element_id,
                        rule_name="forbidden_elements",
                        suggestions=[f"Remove or replace the {forbidden_type.value} element"]
                    ))
        
        return issues
    
    def _validate_element_hierarchy(
        self,
        elements: List[DocumentElement],
        element_map: Dict[str, DocumentElement],
        schema: DocumentSchema
    ) -> List[ValidationIssue]:
        """Validate element hierarchy structure"""
        issues = []
        
        for element in elements:
            if element.parent_id:
                parent = element_map.get(element.parent_id)
                if not parent:
                    issues.append(ValidationIssue(
                        issue_id=f"missing_parent_{element.element_id}",
                        severity=ValidationSeverity.ERROR,
                        message=f"Parent element '{element.parent_id}' not found",
                        element_id=element.element_id,
                        rule_name="hierarchy_structure"
                    ))
                    continue
                
                # Check if parent-child relationship is allowed
                allowed_children = schema.element_hierarchy.get(parent.element_type, [])
                if allowed_children and element.element_type not in allowed_children:
                    issues.append(ValidationIssue(
                        issue_id=f"invalid_hierarchy_{element.element_id}",
                        severity=ValidationSeverity.WARNING,
                        message=f"Element '{element.element_type.value}' cannot be child of '{parent.element_type.value}'",
                        element_id=element.element_id,
                        rule_name="hierarchy_rules",
                        suggestions=[f"Move {element.element_type.value} to appropriate parent or restructure hierarchy"]
                    ))
        
        # Check hierarchy depth
        for element in elements:
            depth = self._calculate_element_depth(element, element_map)
            if depth > self.config.hierarchy_depth_limit:
                issues.append(ValidationIssue(
                    issue_id=f"hierarchy_too_deep_{element.element_id}",
                    severity=ValidationSeverity.WARNING,
                    message=f"Element hierarchy depth ({depth}) exceeds limit ({self.config.hierarchy_depth_limit})",
                    element_id=element.element_id,
                    rule_name="hierarchy_depth"
                ))
        
        return issues
    
    def _validate_nesting_rules(
        self,
        elements: List[DocumentElement],
        element_map: Dict[str, DocumentElement],
        schema: DocumentSchema
    ) -> List[ValidationIssue]:
        """Validate element nesting rules"""
        issues = []
        
        for element in elements:
            allowed_nested = schema.nesting_rules.get(element.element_type, set())
            
            for child_id in element.children_ids:
                child = element_map.get(child_id)
                if child and child.element_type not in allowed_nested and allowed_nested:
                    issues.append(ValidationIssue(
                        issue_id=f"invalid_nesting_{element.element_id}_{child_id}",
                        severity=ValidationSeverity.WARNING,
                        message=f"Element '{child.element_type.value}' cannot be nested in '{element.element_type.value}'",
                        element_id=element.element_id,
                        rule_name="nesting_rules",
                        suggestions=[f"Restructure to avoid nesting {child.element_type.value} in {element.element_type.value}"]
                    ))
        
        return issues
    
    def _validate_ordering_rules(
        self,
        elements: List[DocumentElement],
        schema: DocumentSchema
    ) -> List[ValidationIssue]:
        """Validate element ordering rules"""
        issues = []
        
        # Group elements by page and sort by position
        page_elements = {}
        for element in elements:
            page = element.page_number
            if page not in page_elements:
                page_elements[page] = []
            page_elements[page].append(element)
        
        for page, page_elems in page_elements.items():
            # Sort by y-coordinate (top to bottom)
            sorted_elements = sorted(page_elems, key=lambda e: e.bbox.y)
            
            for before_type, after_type in schema.ordering_rules:
                before_elements = [e for e in sorted_elements if e.element_type == before_type]
                after_elements = [e for e in sorted_elements if e.element_type == after_type]
                
                if before_elements and after_elements:
                    last_before = max(before_elements, key=lambda e: e.bbox.y)
                    first_after = min(after_elements, key=lambda e: e.bbox.y)
                    
                    if last_before.bbox.y > first_after.bbox.y:
                        issues.append(ValidationIssue(
                            issue_id=f"ordering_violation_{before_type.value}_{after_type.value}_page_{page}",
                            severity=ValidationSeverity.WARNING,
                            message=f"'{before_type.value}' should come before '{after_type.value}' on page {page}",
                            rule_name="ordering_rules",
                            suggestions=[f"Reorder elements so {before_type.value} appears before {after_type.value}"]
                        ))
        
        return issues
    
    def _validate_element_constraints(
        self,
        elements: List[DocumentElement],
        schema: DocumentSchema
    ) -> List[ValidationIssue]:
        """Validate element-specific constraints"""
        issues = []
        
        # Group elements by type
        type_counts = {}
        for element in elements:
            element_type = element.element_type
            if element_type not in type_counts:
                type_counts[element_type] = []
            type_counts[element_type].append(element)
        
        # Validate constraints
        for element_type, constraints in schema.element_constraints.items():
            elements_of_type = type_counts.get(element_type, [])
            
            # Check count constraints
            if "min_count" in constraints:
                min_count = constraints["min_count"]
                if len(elements_of_type) < min_count:
                    issues.append(ValidationIssue(
                        issue_id=f"min_count_violation_{element_type.value}",
                        severity=ValidationSeverity.ERROR,
                        message=f"Minimum count for '{element_type.value}' is {min_count}, found {len(elements_of_type)}",
                        rule_name="element_constraints"
                    ))
            
            if "max_count" in constraints:
                max_count = constraints["max_count"]
                if len(elements_of_type) > max_count:
                    issues.append(ValidationIssue(
                        issue_id=f"max_count_violation_{element_type.value}",
                        severity=ValidationSeverity.WARNING,
                        message=f"Maximum count for '{element_type.value}' is {max_count}, found {len(elements_of_type)}",
                        rule_name="element_constraints"
                    ))
            
            # Check individual element constraints
            for element in elements_of_type:
                # Check level constraints
                if "max_level" in constraints and element.level is not None:
                    max_level = constraints["max_level"]
                    if element.level > max_level:
                        issues.append(ValidationIssue(
                            issue_id=f"max_level_violation_{element.element_id}",
                            severity=ValidationSeverity.WARNING,
                            message=f"Maximum level for '{element_type.value}' is {max_level}, found {element.level}",
                            element_id=element.element_id,
                            rule_name="element_constraints"
                        ))
                
                # Check caption requirements
                if constraints.get("requires_caption", False):
                    # Look for caption element near this element
                    has_caption = self._has_nearby_caption(element, elements)
                    if not has_caption:
                        issues.append(ValidationIssue(
                            issue_id=f"missing_caption_{element.element_id}",
                            severity=ValidationSeverity.WARNING,
                            message=f"Element '{element_type.value}' requires a caption",
                            element_id=element.element_id,
                            rule_name="element_constraints",
                            suggestions=["Add a caption element near this element"]
                        ))
        
        return issues
    
    def _validate_spatial_consistency(self, elements: List[DocumentElement]) -> List[ValidationIssue]:
        """Validate spatial layout consistency"""
        issues = []
        
        # Check for overlapping elements
        for i, element1 in enumerate(elements):
            for j, element2 in enumerate(elements[i+1:], i+1):
                if element1.page_number == element2.page_number:
                    if element1.bbox.overlaps(element2.bbox, self.config.overlap_threshold):
                        issues.append(ValidationIssue(
                            issue_id=f"overlap_{element1.element_id}_{element2.element_id}",
                            severity=ValidationSeverity.WARNING,
                            message=f"Elements overlap: '{element1.element_type.value}' and '{element2.element_type.value}'",
                            rule_name="spatial_consistency",
                            suggestions=["Adjust element positions to avoid overlap"]
                        ))
        
        # Check for elements outside page boundaries (assuming standard page)
        page_width, page_height = 612, 792  # Standard letter size in points
        
        for element in elements:
            if (element.bbox.x < 0 or element.bbox.y < 0 or 
                element.bbox.x + element.bbox.width > page_width or 
                element.bbox.y + element.bbox.height > page_height):
                issues.append(ValidationIssue(
                    issue_id=f"out_of_bounds_{element.element_id}",
                    severity=ValidationSeverity.WARNING,
                    message=f"Element extends beyond page boundaries",
                    element_id=element.element_id,
                    rule_name="spatial_consistency",
                    suggestions=["Adjust element size or position to fit within page"]
                ))
        
        return issues
    
    def _calculate_element_depth(
        self,
        element: DocumentElement,
        element_map: Dict[str, DocumentElement]
    ) -> int:
        """Calculate element depth in hierarchy"""
        depth = 0
        current_element = element
        
        while current_element.parent_id:
            depth += 1
            current_element = element_map.get(current_element.parent_id)
            if not current_element:
                break
            
            # Prevent infinite loops
            if depth > 50:
                break
        
        return depth
    
    def _has_nearby_caption(
        self,
        element: DocumentElement,
        all_elements: List[DocumentElement]
    ) -> bool:
        """Check if element has a nearby caption"""
        caption_elements = [e for e in all_elements 
                          if e.element_type == ElementType.CAPTION and e.page_number == element.page_number]
        
        # Look for captions within reasonable distance
        for caption in caption_elements:
            distance = abs(caption.bbox.y - (element.bbox.y + element.bbox.height))
            if distance < 50:  # Within 50 points
                return True
        
        return False
    
    def _calculate_validation_score(
        self,
        issues: List[ValidationIssue],
        elements: List[DocumentElement],
        schema: DocumentSchema
    ) -> float:
        """Calculate overall validation score"""
        if not elements:
            return 0.0
        
        # Count issues by severity
        critical_count = len([i for i in issues if i.severity == ValidationSeverity.CRITICAL])
        error_count = len([i for i in issues if i.severity == ValidationSeverity.ERROR])
        warning_count = len([i for i in issues if i.severity == ValidationSeverity.WARNING])
        
        # Calculate penalty based on issue severity
        total_penalty = (
            critical_count * 0.5 +
            error_count * 0.2 +
            warning_count * 0.1
        )
        
        # Normalize by number of elements
        penalty_per_element = total_penalty / len(elements)
        
        # Calculate base score
        base_score = max(0.0, 1.0 - penalty_per_element)
        
        # Apply component weights
        component_scores = {
            "required_elements": self._score_required_elements(elements, schema),
            "forbidden_elements": self._score_forbidden_elements(elements, schema),
            "hierarchy": self._score_hierarchy(issues),
            "nesting": self._score_nesting(issues),
            "ordering": self._score_ordering(issues)
        }
        
        weighted_score = (
            component_scores["required_elements"] * self.config.required_element_weight +
            component_scores["forbidden_elements"] * self.config.forbidden_element_weight +
            component_scores["hierarchy"] * self.config.hierarchy_weight +
            component_scores["nesting"] * self.config.nesting_weight +
            component_scores["ordering"] * self.config.ordering_weight
        )
        
        return min(1.0, max(0.0, (base_score + weighted_score) / 2))
    
    def _score_required_elements(
        self,
        elements: List[DocumentElement],
        schema: DocumentSchema
    ) -> float:
        """Score required elements presence"""
        if not schema.required_elements:
            return 1.0
        
        element_types = {e.element_type for e in elements}
        present_required = len(schema.required_elements.intersection(element_types))
        
        return present_required / len(schema.required_elements)
    
    def _score_forbidden_elements(
        self,
        elements: List[DocumentElement],
        schema: DocumentSchema
    ) -> float:
        """Score forbidden elements absence"""
        if not schema.forbidden_elements:
            return 1.0
        
        element_types = {e.element_type for e in elements}
        forbidden_present = len(schema.forbidden_elements.intersection(element_types))
        
        return max(0.0, 1.0 - (forbidden_present / len(schema.forbidden_elements)))
    
    def _score_hierarchy(self, issues: List[ValidationIssue]) -> float:
        """Score hierarchy validation"""
        hierarchy_issues = [i for i in issues if i.rule_name in ["hierarchy_structure", "hierarchy_rules", "hierarchy_depth"]]
        return max(0.0, 1.0 - (len(hierarchy_issues) * 0.1))
    
    def _score_nesting(self, issues: List[ValidationIssue]) -> float:
        """Score nesting validation"""
        nesting_issues = [i for i in issues if i.rule_name == "nesting_rules"]
        return max(0.0, 1.0 - (len(nesting_issues) * 0.1))
    
    def _score_ordering(self, issues: List[ValidationIssue]) -> float:
        """Score ordering validation"""
        ordering_issues = [i for i in issues if i.rule_name == "ordering_rules"]
        return max(0.0, 1.0 - (len(ordering_issues) * 0.1))
    
    def _generate_validation_summary(
        self,
        issues: List[ValidationIssue],
        elements: List[DocumentElement],
        schema: DocumentSchema
    ) -> Dict[str, Any]:
        """Generate validation summary"""
        summary = {
            "total_elements": len(elements),
            "total_issues": len(issues),
            "issues_by_severity": {
                "critical": len([i for i in issues if i.severity == ValidationSeverity.CRITICAL]),
                "error": len([i for i in issues if i.severity == ValidationSeverity.ERROR]),
                "warning": len([i for i in issues if i.severity == ValidationSeverity.WARNING]),
                "info": len([i for i in issues if i.severity == ValidationSeverity.INFO])
            },
            "issues_by_rule": {},
            "element_type_distribution": {},
            "pages_validated": len(set(e.page_number for e in elements)),
            "schema_compliance": {
                "required_elements_present": self._check_required_elements_present(elements, schema),
                "forbidden_elements_absent": self._check_forbidden_elements_absent(elements, schema),
                "hierarchy_valid": len([i for i in issues if "hierarchy" in i.rule_name]) == 0
            }
        }
        
        # Count issues by rule
        for issue in issues:
            rule = issue.rule_name or "unknown"
            summary["issues_by_rule"][rule] = summary["issues_by_rule"].get(rule, 0) + 1
        
        # Count element types
        for element in elements:
            element_type = element.element_type.value
            summary["element_type_distribution"][element_type] = summary["element_type_distribution"].get(element_type, 0) + 1
        
        return summary
    
    def _check_required_elements_present(
        self,
        elements: List[DocumentElement],
        schema: DocumentSchema
    ) -> bool:
        """Check if all required elements are present"""
        element_types = {e.element_type for e in elements}
        return schema.required_elements.issubset(element_types)
    
    def _check_forbidden_elements_absent(
        self,
        elements: List[DocumentElement],
        schema: DocumentSchema
    ) -> bool:
        """Check if forbidden elements are absent"""
        element_types = {e.element_type for e in elements}
        return not schema.forbidden_elements.intersection(element_types)
    
    def validate_multiple_documents(
        self,
        documents: List[Tuple[List[DocumentElement], DocumentType]]
    ) -> Dict[str, Any]:
        """Validate multiple documents and provide aggregate statistics"""
        results = []
        
        for i, (elements, doc_type) in enumerate(documents):
            try:
                result = self.validate_document_structure(elements, doc_type)
                results.append({
                    "document_index": i,
                    "result": result
                })
            except Exception as e:
                logger.error(f"Failed to validate document {i}: {str(e)}")
                results.append({
                    "document_index": i,
                    "error": str(e)
                })
        
        # Calculate aggregate statistics
        successful_results = [r["result"] for r in results if "result" in r]
        
        if successful_results:
            scores = [r.validation_score for r in successful_results]
            valid_count = len([r for r in successful_results if r.is_valid])
            
            aggregate_stats = {
                "total_documents": len(documents),
                "successful_validations": len(successful_results),
                "failed_validations": len(documents) - len(successful_results),
                "valid_documents": valid_count,
                "invalid_documents": len(successful_results) - valid_count,
                "mean_score": sum(scores) / len(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "validation_rate": valid_count / len(successful_results)
            }
        else:
            aggregate_stats = {
                "total_documents": len(documents),
                "successful_validations": 0,
                "failed_validations": len(documents),
                "error": "No successful validations"
            }
        
        return {
            "results": results,
            "aggregate_statistics": aggregate_stats
        }


def create_structural_validator(
    config: Optional[Union[Dict[str, Any], StructuralValidatorConfig]] = None
) -> StructuralValidator:
    """Factory function to create structural validator"""
    if isinstance(config, dict):
        config = StructuralValidatorConfig(**config)
    return StructuralValidator(config)


def validate_document_sample() -> ValidationResult:
    """Generate sample document validation for testing"""
    validator = create_structural_validator()
    
    # Create sample document elements
    elements = [
        DocumentElement(
            element_id="title_1",
            element_type=ElementType.TITLE,
            bbox=BoundingBox(50, 50, 500, 40),
            content="Sample Academic Paper",
            page_number=1
        ),
        DocumentElement(
            element_id="heading_1",
            element_type=ElementType.HEADING,
            bbox=BoundingBox(50, 120, 400, 25),
            content="Introduction",
            level=1,
            page_number=1,
            parent_id="title_1"
        ),
        DocumentElement(
            element_id="paragraph_1",
            element_type=ElementType.PARAGRAPH,
            bbox=BoundingBox(50, 160, 500, 60),
            content="This is the introduction paragraph...",
            page_number=1,
            parent_id="heading_1"
        )
    ]
    
    return validator.validate_document_structure(elements, DocumentType.ACADEMIC_PAPER)