"""
Quality validation module for document analysis and verification.
Provides comprehensive validation for document structure, completeness, and semantic consistency.
"""

from typing import Dict, List, Optional, Union, Any

from .drift_detector import (
    DriftDetector,
    DriftDetectorConfig,
    DriftReport,
    DriftMetric,
    DriftType,
    DriftSeverity,
    DriftMethod,
    DataWindow,
    create_drift_detector,
    detect_drift_sample
)

from .structural_validator import (
    StructuralValidator,
    StructuralValidatorConfig,
    ValidationResult,
    ValidationIssue,
    DocumentElement,
    DocumentSchema,
    DocumentType,
    ElementType,
    ValidationSeverity,
    BoundingBox,
    create_structural_validator,
    validate_document_sample
)

from .completeness_checker import (
    CompletenessChecker,
    CompletenessCheckerConfig,
    CompletenessReport,
    DocumentContent,
    ContentRequirement,
    MissingContent,
    CompletenessLevel,
    ContentType,
    MissingContentSeverity,
    create_completeness_checker,
    check_completeness_sample
)

from .semantic_validator import (
    SemanticValidator,
    SemanticValidatorConfig,
    SemanticValidationResult,
    DocumentSemantics,
    Entity,
    SemanticRelation,
    SemanticIssue,
    EntityType,
    SemanticIssueType,
    SemanticSeverity,
    create_semantic_validator,
    validate_semantics_sample
)

from ...core.logging import get_logger

logger = get_logger(__name__)

# Module version
__version__ = "1.0.0"

# Public API
__all__ = [
    # Drift Detector
    "DriftDetector",
    "DriftDetectorConfig", 
    "DriftReport",
    "DriftMetric",
    "DriftType",
    "DriftSeverity",
    "DriftMethod",
    "DataWindow",
    "create_drift_detector",
    "detect_drift_sample",
    
    # Structural Validator
    "StructuralValidator",
    "StructuralValidatorConfig",
    "ValidationResult",
    "ValidationIssue",
    "DocumentElement",
    "DocumentSchema",
    "DocumentType",
    "ElementType",
    "ValidationSeverity",
    "BoundingBox",
    "create_structural_validator",
    "validate_document_sample",
    
    # Completeness Checker
    "CompletenessChecker",
    "CompletenessCheckerConfig",
    "CompletenessReport",
    "DocumentContent",
    "ContentRequirement",
    "MissingContent",
    "CompletenessLevel",
    "ContentType",
    "MissingContentSeverity",
    "create_completeness_checker",
    "check_completeness_sample",
    
    # Semantic Validator
    "SemanticValidator",
    "SemanticValidatorConfig",
    "SemanticValidationResult",
    "DocumentSemantics",
    "Entity",
    "SemanticRelation",
    "SemanticIssue",
    "EntityType",
    "SemanticIssueType",
    "SemanticSeverity",
    "create_semantic_validator",
    "validate_semantics_sample",
    
    # Factory functions
    "create_validation_suite",
    "run_comprehensive_validation",
    "generate_validation_report"
]


def create_validation_suite(
    drift_config: Optional[Union[Dict[str, Any], DriftDetectorConfig]] = None,
    structural_config: Optional[Union[Dict[str, Any], StructuralValidatorConfig]] = None,
    completeness_config: Optional[Union[Dict[str, Any], CompletenessCheckerConfig]] = None,
    semantic_config: Optional[Union[Dict[str, Any], SemanticValidatorConfig]] = None
) -> Dict[str, Any]:
    """
    Create a complete validation suite with all validators.
    
    Args:
        drift_config: Drift detector configuration
        structural_config: Structural validator configuration
        completeness_config: Completeness checker configuration
        semantic_config: Semantic validator configuration
    
    Returns:
        Dictionary with all validators
    """
    return {
        "drift_detector": create_drift_detector(drift_config),
        "structural_validator": create_structural_validator(structural_config),
        "completeness_checker": create_completeness_checker(completeness_config),
        "semantic_validator": create_semantic_validator(semantic_config)
    }


def run_comprehensive_validation(
    document_data: Dict[str, Any],
    validation_suite: Optional[Dict[str, Any]] = None,
    validation_options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run comprehensive validation on document data.
    
    Args:
        document_data: Document data to validate
        validation_suite: Pre-configured validation suite
        validation_options: Additional validation options
    
    Returns:
        Comprehensive validation results
    """
    if validation_suite is None:
        validation_suite = create_validation_suite()
    
    options = validation_options or {}
    results = {}
    
    # Drift Detection (if reference data available)
    if "reference_data" in document_data and "current_data" in document_data:
        try:
            drift_detector = validation_suite["drift_detector"]
            drift_detector.set_reference_data(document_data["reference_data"])
            drift_report = drift_detector.detect_drift(document_data["current_data"])
            results["drift_detection"] = drift_report.to_dict()
        except Exception as e:
            logger.error(f"Drift detection failed: {str(e)}")
            results["drift_detection"] = {"error": str(e)}
    
    # Structural Validation
    if "document_elements" in document_data:
        try:
            structural_validator = validation_suite["structural_validator"]
            document_type = document_data.get("document_type", DocumentType.ACADEMIC_PAPER)
            if isinstance(document_type, str):
                document_type = DocumentType(document_type)
            
            # Convert element data to DocumentElement objects
            elements = []
            for elem_data in document_data["document_elements"]:
                if isinstance(elem_data, dict):
                    bbox = BoundingBox(**elem_data["bbox"])
                    element = DocumentElement(
                        element_id=elem_data["element_id"],
                        element_type=ElementType(elem_data["element_type"]),
                        bbox=bbox,
                        content=elem_data.get("content"),
                        level=elem_data.get("level"),
                        page_number=elem_data.get("page_number", 1),
                        parent_id=elem_data.get("parent_id"),
                        children_ids=elem_data.get("children_ids", []),
                        attributes=elem_data.get("attributes", {})
                    )
                    elements.append(element)
                else:
                    elements.append(elem_data)
            
            validation_result = structural_validator.validate_document_structure(elements, document_type)
            results["structural_validation"] = validation_result.to_dict()
        except Exception as e:
            logger.error(f"Structural validation failed: {str(e)}")
            results["structural_validation"] = {"error": str(e)}
    
    # Completeness Checking
    if "document_content" in document_data:
        try:
            completeness_checker = validation_suite["completeness_checker"]
            content_data = document_data["document_content"]
            
            if isinstance(content_data, dict):
                document_content = DocumentContent(
                    document_id=content_data.get("document_id", "unknown"),
                    text_content=content_data.get("text_content", ""),
                    metadata=content_data.get("metadata", {}),
                    structural_elements=content_data.get("structural_elements", []),
                    visual_elements=content_data.get("visual_elements", []),
                    semantic_annotations=content_data.get("semantic_annotations", [])
                )
            else:
                document_content = content_data
            
            completeness_report = completeness_checker.check_document_completeness(document_content)
            results["completeness_checking"] = completeness_report.to_dict()
        except Exception as e:
            logger.error(f"Completeness checking failed: {str(e)}")
            results["completeness_checking"] = {"error": str(e)}
    
    # Semantic Validation
    if "document_semantics" in document_data:
        try:
            semantic_validator = validation_suite["semantic_validator"]
            semantics_data = document_data["document_semantics"]
            
            if isinstance(semantics_data, dict):
                document_semantics = DocumentSemantics(
                    document_id=semantics_data.get("document_id", "unknown"),
                    text=semantics_data.get("text", ""),
                    entities=[Entity(**e) if isinstance(e, dict) else e 
                             for e in semantics_data.get("entities", [])],
                    relations=[SemanticRelation(**r) if isinstance(r, dict) else r 
                              for r in semantics_data.get("relations", [])],
                    metadata=semantics_data.get("metadata", {}),
                    sections=semantics_data.get("sections", [])
                )
            else:
                document_semantics = semantics_data
            
            semantic_result = semantic_validator.validate_document_semantics(document_semantics)
            results["semantic_validation"] = semantic_result.to_dict()
        except Exception as e:
            logger.error(f"Semantic validation failed: {str(e)}")
            results["semantic_validation"] = {"error": str(e)}
    
    # Calculate overall validation score
    validation_scores = []
    for validation_type, validation_result in results.items():
        if isinstance(validation_result, dict) and "error" not in validation_result:
            if "validation_score" in validation_result:
                validation_scores.append(validation_result["validation_score"])
            elif "completeness_score" in validation_result:
                validation_scores.append(validation_result["completeness_score"])
            elif "semantic_score" in validation_result:
                validation_scores.append(validation_result["semantic_score"])
            elif "overall_drift_score" in validation_result:
                # For drift, lower score is better, so invert it
                validation_scores.append(1.0 - validation_result["overall_drift_score"])
    
    overall_score = sum(validation_scores) / len(validation_scores) if validation_scores else 0.0
    
    results["overall_validation"] = {
        "overall_score": overall_score,
        "validation_count": len([r for r in results.values() if not isinstance(r, dict) or "error" not in r]),
        "failed_validations": len([r for r in results.values() if isinstance(r, dict) and "error" in r]),
        "validation_grade": _get_validation_grade(overall_score),
        "needs_attention": overall_score < 0.7
    }
    
    return results


def generate_validation_report(
    validation_results: Dict[str, Any],
    output_format: str = "dict"
) -> Union[Dict[str, Any], str]:
    """
    Generate a formatted validation report from results.
    
    Args:
        validation_results: Results from comprehensive validation
        output_format: Output format ("dict", "json", "markdown")
    
    Returns:
        Formatted validation report
    """
    if output_format == "dict":
        return validation_results
    
    elif output_format == "json":
        import json
        return json.dumps(validation_results, indent=2, default=str)
    
    elif output_format == "markdown":
        return _generate_markdown_validation_report(validation_results)
    
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def _get_validation_grade(score: float) -> str:
    """Get validation grade from score"""
    if score >= 0.9:
        return "A"
    elif score >= 0.8:
        return "B"
    elif score >= 0.7:
        return "C"
    elif score >= 0.6:
        return "D"
    else:
        return "F"


def _generate_markdown_validation_report(results: Dict[str, Any]) -> str:
    """Generate markdown validation report"""
    lines = ["# Document Validation Report", ""]
    
    # Overall validation summary
    if "overall_validation" in results:
        overall = results["overall_validation"]
        lines.extend([
            "## Overall Validation",
            f"**Score:** {overall['overall_score']:.3f}",
            f"**Grade:** {overall['validation_grade']}",
            f"**Validations Completed:** {overall['validation_count']}",
            f"**Failed Validations:** {overall['failed_validations']}",
            f"**Needs Attention:** {'Yes' if overall['needs_attention'] else 'No'}",
            ""
        ])
    
    # Individual validation results
    validation_order = ["drift_detection", "structural_validation", "completeness_checking", "semantic_validation"]
    
    for validation_type in validation_order:
        if validation_type in results:
            validation_result = results[validation_type]
            section_title = validation_type.replace('_', ' ').title()
            lines.append(f"## {section_title}")
            
            if isinstance(validation_result, dict):
                if "error" in validation_result:
                    lines.append(f"**Error:** {validation_result['error']}")
                else:
                    # Add relevant metrics for each validation type
                    if validation_type == "drift_detection":
                        lines.append(f"**Drift Detected:** {'Yes' if validation_result.get('drift_detected', False) else 'No'}")
                        lines.append(f"**Drift Score:** {validation_result.get('overall_drift_score', 0):.3f}")
                    elif validation_type == "structural_validation":
                        lines.append(f"**Valid Structure:** {'Yes' if validation_result.get('is_valid', False) else 'No'}")
                        lines.append(f"**Validation Score:** {validation_result.get('validation_score', 0):.3f}")
                    elif validation_type == "completeness_checking":
                        lines.append(f"**Completeness Level:** {validation_result.get('completeness_level', 'Unknown')}")
                        lines.append(f"**Completeness Score:** {validation_result.get('completeness_score', 0):.3f}")
                    elif validation_type == "semantic_validation":
                        lines.append(f"**Semantic Score:** {validation_result.get('semantic_score', 0):.3f}")
                        lines.append(f"**Issues Found:** {len(validation_result.get('issues', []))}")
            
            lines.append("")
    
    return "\n".join(lines)


# Initialize module
logger.info(f"Initialized quality validation module v{__version__}")
logger.info("Available validators: Drift Detection, Structural Validation, Completeness Checking, Semantic Validation")