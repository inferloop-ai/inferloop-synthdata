"""
Quality metrics module for document analysis and evaluation.
Provides comprehensive metrics for OCR accuracy, layout quality, and table structure.
"""

from typing import Dict, List, Optional, Union, Any

from .ocr_metrics import (
    OCRMetricsCalculator,
    OCRMetricsConfig,
    OCRResult,
    OCRQualityReport,
    MetricResult,
    MetricType,
    create_ocr_metrics_calculator,
    evaluate_ocr_sample
)

from .layout_metrics import (
    LayoutMetricsCalculator,
    LayoutMetricsConfig,
    LayoutQualityReport,
    LayoutMetricResult,
    LayoutMetricType,
    LayoutElement,
    LayoutIssue,
    BoundingBox,
    create_layout_metrics_calculator,
    evaluate_layout_sample
)

from .teds_calculator import (
    TEDSCalculator,
    TEDSConfig,
    TEDSResult,
    TableNode,
    TableNodeType,
    EditOperation,
    create_teds_calculator,
    calculate_teds_sample
)

from .benchmark_runner import (
    BenchmarkRunner,
    BenchmarkConfig,
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkSuite,
    TestCase,
    BenchmarkType,
    TestDataType,
    create_benchmark_runner,
    run_sample_benchmark
)

from .content_metrics import (
    ContentMetricsCalculator,
    ContentMetricsConfig,
    ContentQualityReport,
    create_content_metrics_calculator
)

from ...core.logging import get_logger

logger = get_logger(__name__)

# Module version
__version__ = "1.0.0"

# Public API
__all__ = [
    # OCR Metrics
    "OCRMetricsCalculator",
    "OCRMetricsConfig", 
    "OCRResult",
    "OCRQualityReport",
    "MetricResult",
    "MetricType",
    "create_ocr_metrics_calculator",
    "evaluate_ocr_sample",
    
    # Layout Metrics
    "LayoutMetricsCalculator",
    "LayoutMetricsConfig",
    "LayoutQualityReport",
    "LayoutMetricResult",
    "LayoutMetricType",
    "LayoutElement",
    "LayoutIssue",
    "BoundingBox",
    "create_layout_metrics_calculator",
    "evaluate_layout_sample",
    
    # TEDS Calculator
    "TEDSCalculator",
    "TEDSConfig",
    "TEDSResult",
    "TableNode",
    "TableNodeType",
    "EditOperation",
    "create_teds_calculator",
    "calculate_teds_sample",
    
    # Benchmark Runner
    "BenchmarkRunner",
    "BenchmarkConfig",
    "BenchmarkReport",
    "BenchmarkResult",
    "BenchmarkSuite",
    "TestCase",
    "BenchmarkType",
    "TestDataType",
    "create_benchmark_runner",
    "run_sample_benchmark",
    
    # Content Metrics
    "ContentMetricsCalculator",
    "ContentMetricsConfig",
    "ContentQualityReport",
    "create_content_metrics_calculator",
    
    # Factory functions
    "create_quality_metrics_suite",
    "run_comprehensive_quality_assessment",
    "generate_quality_report"
]


def create_quality_metrics_suite(
    ocr_config: Optional[Union[Dict[str, Any], OCRMetricsConfig]] = None,
    layout_config: Optional[Union[Dict[str, Any], LayoutMetricsConfig]] = None,
    teds_config: Optional[Union[Dict[str, Any], TEDSConfig]] = None,
    content_config: Optional[Union[Dict[str, Any], ContentMetricsConfig]] = None,
    benchmark_config: Optional[Union[Dict[str, Any], BenchmarkConfig]] = None
) -> Dict[str, Any]:
    """
    Create a complete quality metrics suite with all calculators.
    
    Args:
        ocr_config: OCR metrics configuration
        layout_config: Layout metrics configuration
        teds_config: TEDS calculator configuration
        content_config: Content metrics configuration
        benchmark_config: Benchmark runner configuration
    
    Returns:
        Dictionary with all metric calculators and runner
    """
    return {
        "ocr_calculator": create_ocr_metrics_calculator(ocr_config),
        "layout_calculator": create_layout_metrics_calculator(layout_config),
        "teds_calculator": create_teds_calculator(teds_config),
        "content_calculator": create_content_metrics_calculator(content_config),
        "benchmark_runner": create_benchmark_runner(benchmark_config)
    }


def run_comprehensive_quality_assessment(
    document_data: Dict[str, Any],
    reference_data: Optional[Dict[str, Any]] = None,
    metrics_suite: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run comprehensive quality assessment on document data.
    
    Args:
        document_data: Document data to assess
        reference_data: Reference/ground truth data
        metrics_suite: Pre-configured metrics suite
    
    Returns:
        Comprehensive quality assessment results
    """
    if metrics_suite is None:
        metrics_suite = create_quality_metrics_suite()
    
    results = {}
    
    # OCR Quality Assessment
    if "ocr_data" in document_data:
        try:
            ocr_calculator = metrics_suite["ocr_calculator"]
            if reference_data and "reference_text" in reference_data:
                ocr_report = ocr_calculator.evaluate_ocr_quality(
                    reference_data["reference_text"],
                    document_data["ocr_data"]
                )
                results["ocr_quality"] = ocr_report.to_dict()
            else:
                logger.warning("No reference text provided for OCR assessment")
        except Exception as e:
            logger.error(f"OCR quality assessment failed: {str(e)}")
            results["ocr_quality"] = {"error": str(e)}
    
    # Layout Quality Assessment
    if "layout_elements" in document_data:
        try:
            layout_calculator = metrics_suite["layout_calculator"]
            layout_report = layout_calculator.evaluate_layout_quality(
                document_data["layout_elements"],
                document_data.get("page_info")
            )
            results["layout_quality"] = layout_report.to_dict()
        except Exception as e:
            logger.error(f"Layout quality assessment failed: {str(e)}")
            results["layout_quality"] = {"error": str(e)}
    
    # Table Structure Assessment
    if "table_structure" in document_data:
        try:
            teds_calculator = metrics_suite["teds_calculator"]
            if reference_data and "reference_table" in reference_data:
                teds_result = teds_calculator.calculate_teds(
                    document_data["table_structure"],
                    reference_data["reference_table"]
                )
                results["table_structure"] = teds_result.to_dict()
            else:
                # Analyze structure quality without reference
                structure_analysis = teds_calculator.evaluate_table_structure_quality(
                    document_data["table_structure"]
                )
                results["table_structure"] = structure_analysis
        except Exception as e:
            logger.error(f"Table structure assessment failed: {str(e)}")
            results["table_structure"] = {"error": str(e)}
    
    # Content Quality Assessment
    if "content" in document_data:
        try:
            content_calculator = metrics_suite["content_calculator"]
            content_report = content_calculator.evaluate_content_quality(
                document_data["content"],
                reference_data.get("reference_content") if reference_data else None
            )
            results["content_quality"] = content_report.to_dict()
        except Exception as e:
            logger.error(f"Content quality assessment failed: {str(e)}")
            results["content_quality"] = {"error": str(e)}
    
    # Calculate overall quality score
    quality_scores = []
    for assessment_type, assessment_result in results.items():
        if isinstance(assessment_result, dict) and "overall_score" in assessment_result:
            quality_scores.append(assessment_result["overall_score"])
        elif isinstance(assessment_result, dict) and "teds_score" in assessment_result:
            quality_scores.append(assessment_result["teds_score"])
    
    overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    results["overall_assessment"] = {
        "overall_quality_score": overall_quality,
        "assessment_count": len([r for r in results.values() if not isinstance(r, dict) or "error" not in r]),
        "failed_assessments": len([r for r in results.values() if isinstance(r, dict) and "error" in r]),
        "quality_grade": _get_quality_grade(overall_quality)
    }
    
    return results


def generate_quality_report(
    assessment_results: Dict[str, Any],
    output_format: str = "dict"
) -> Union[Dict[str, Any], str]:
    """
    Generate a formatted quality report from assessment results.
    
    Args:
        assessment_results: Results from quality assessment
        output_format: Output format ("dict", "json", "markdown")
    
    Returns:
        Formatted quality report
    """
    if output_format == "dict":
        return assessment_results
    
    elif output_format == "json":
        import json
        return json.dumps(assessment_results, indent=2, default=str)
    
    elif output_format == "markdown":
        return _generate_markdown_report(assessment_results)
    
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def _get_quality_grade(score: float) -> str:
    """Get quality grade from score"""
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


def _generate_markdown_report(results: Dict[str, Any]) -> str:
    """Generate markdown quality report"""
    lines = ["# Document Quality Assessment Report", ""]
    
    # Overall assessment
    if "overall_assessment" in results:
        overall = results["overall_assessment"]
        lines.extend([
            "## Overall Quality",
            f"**Score:** {overall['overall_quality_score']:.3f}",
            f"**Grade:** {overall['quality_grade']}",
            f"**Assessments Completed:** {overall['assessment_count']}",
            f"**Failed Assessments:** {overall['failed_assessments']}",
            ""
        ])
    
    # Individual assessments
    for assessment_type, assessment_result in results.items():
        if assessment_type == "overall_assessment":
            continue
        
        lines.append(f"## {assessment_type.replace('_', ' ').title()}")
        
        if isinstance(assessment_result, dict):
            if "error" in assessment_result:
                lines.append(f"**Error:** {assessment_result['error']}")
            elif "overall_score" in assessment_result:
                lines.append(f"**Score:** {assessment_result['overall_score']:.3f}")
            elif "teds_score" in assessment_result:
                lines.append(f"**TEDS Score:** {assessment_result['teds_score']:.3f}")
        
        lines.append("")
    
    return "\n".join(lines)


# Initialize module
logger.info(f"Initialized quality metrics module v{__version__}")
logger.info("Available calculators: OCR, Layout, TEDS, Content, Benchmark")