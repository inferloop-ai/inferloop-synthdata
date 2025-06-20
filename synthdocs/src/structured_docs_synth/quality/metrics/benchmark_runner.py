"""
Benchmark runner for performance testing and quality metrics evaluation.
Runs comprehensive benchmarks across different document types and metrics.
"""

from __future__ import annotations

import time
import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from pydantic import BaseModel, Field

from ...core.config import BaseConfig
from ...core.exceptions import ValidationError
from ...core.logging import get_logger
from .ocr_metrics import OCRMetricsCalculator, OCRResult, OCRQualityReport
from .layout_metrics import LayoutMetricsCalculator, LayoutElement, LayoutQualityReport
from .teds_calculator import TEDSCalculator, TEDSResult

logger = get_logger(__name__)


class BenchmarkType(Enum):
    """Benchmark test types"""
    OCR_ACCURACY = "ocr_accuracy"
    LAYOUT_QUALITY = "layout_quality"
    TABLE_STRUCTURE = "table_structure"
    GENERATION_SPEED = "generation_speed"
    MEMORY_USAGE = "memory_usage"
    SCALABILITY = "scalability"
    END_TO_END = "end_to_end"


class TestDataType(Enum):
    """Test data types"""
    SYNTHETIC = "synthetic"
    REAL_WORLD = "real_world"
    ADVERSARIAL = "adversarial"
    EDGE_CASES = "edge_cases"


@dataclass
class TestCase:
    """Individual test case"""
    test_id: str
    test_type: BenchmarkType
    data_type: TestDataType
    input_data: Any
    expected_output: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "test_id": self.test_id,
            "test_type": self.test_type.value,
            "data_type": self.data_type.value,
            "metadata": self.metadata
        }


@dataclass
class BenchmarkResult:
    """Individual benchmark result"""
    test_case: TestCase
    success: bool
    score: Optional[float] = None
    execution_time: float = 0.0
    memory_usage: Optional[float] = None
    error_message: Optional[str] = None
    detailed_results: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "test_case": self.test_case.to_dict(),
            "success": self.success,
            "score": self.score,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "error_message": self.error_message,
            "detailed_results": self.detailed_results
        }


@dataclass
class BenchmarkSuite:
    """Collection of related test cases"""
    suite_name: str
    benchmark_type: BenchmarkType
    test_cases: List[TestCase]
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "suite_name": self.suite_name,
            "benchmark_type": self.benchmark_type.value,
            "description": self.description,
            "test_count": len(self.test_cases)
        }


@dataclass
class BenchmarkReport:
    """Comprehensive benchmark report"""
    report_id: str
    timestamp: datetime
    suite_results: Dict[str, List[BenchmarkResult]]
    aggregate_statistics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp.isoformat(),
            "suite_results": {
                suite: [result.to_dict() for result in results]
                for suite, results in self.suite_results.items()
            },
            "aggregate_statistics": self.aggregate_statistics,
            "performance_metrics": self.performance_metrics,
            "recommendations": self.recommendations
        }


class BenchmarkConfig(BaseConfig):
    """Benchmark runner configuration"""
    timeout_seconds: float = Field(default=300.0, description="Timeout for individual tests")
    memory_monitoring: bool = Field(default=True, description="Monitor memory usage")
    parallel_execution: bool = Field(default=False, description="Run tests in parallel")
    save_detailed_results: bool = Field(default=True, description="Save detailed test results")
    output_directory: str = Field(default="./benchmark_results", description="Output directory")
    
    # Thresholds for pass/fail
    ocr_accuracy_threshold: float = Field(default=0.8, description="OCR accuracy threshold")
    layout_quality_threshold: float = Field(default=0.7, description="Layout quality threshold")
    teds_score_threshold: float = Field(default=0.85, description="TEDS score threshold")
    performance_threshold_ms: float = Field(default=1000.0, description="Performance threshold in ms")


class BenchmarkRunner:
    """
    Performance and quality benchmark runner.
    Executes comprehensive testing across different metrics and document types.
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """Initialize benchmark runner"""
        self.config = config or BenchmarkConfig()
        
        # Initialize metric calculators
        self.ocr_calculator = OCRMetricsCalculator()
        self.layout_calculator = LayoutMetricsCalculator()
        self.teds_calculator = TEDSCalculator()
        
        # Create output directory
        self.output_dir = Path(self.config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized BenchmarkRunner with output dir: {self.output_dir}")
    
    def run_benchmark_suite(self, suite: BenchmarkSuite) -> List[BenchmarkResult]:
        """Run a complete benchmark suite"""
        logger.info(f"Running benchmark suite: {suite.suite_name}")
        
        results = []
        start_time = time.time()
        
        for test_case in suite.test_cases:
            try:
                result = self._run_single_test(test_case)
                results.append(result)
                
                # Log progress
                if result.success:
                    logger.debug(f"Test {test_case.test_id} passed: score={result.score:.3f}")
                else:
                    logger.warning(f"Test {test_case.test_id} failed: {result.error_message}")
                    
            except Exception as e:
                error_result = BenchmarkResult(
                    test_case=test_case,
                    success=False,
                    error_message=str(e)
                )
                results.append(error_result)
                logger.error(f"Test {test_case.test_id} crashed: {str(e)}")
        
        total_time = time.time() - start_time
        logger.info(f"Suite {suite.suite_name} completed in {total_time:.2f}s")
        
        return results
    
    def _run_single_test(self, test_case: TestCase) -> BenchmarkResult:
        """Run a single test case"""
        start_time = time.time()
        memory_start = self._get_memory_usage() if self.config.memory_monitoring else None
        
        try:
            # Route to appropriate test handler
            if test_case.test_type == BenchmarkType.OCR_ACCURACY:
                result = self._run_ocr_test(test_case)
            elif test_case.test_type == BenchmarkType.LAYOUT_QUALITY:
                result = self._run_layout_test(test_case)
            elif test_case.test_type == BenchmarkType.TABLE_STRUCTURE:
                result = self._run_teds_test(test_case)
            elif test_case.test_type == BenchmarkType.GENERATION_SPEED:
                result = self._run_speed_test(test_case)
            elif test_case.test_type == BenchmarkType.MEMORY_USAGE:
                result = self._run_memory_test(test_case)
            elif test_case.test_type == BenchmarkType.SCALABILITY:
                result = self._run_scalability_test(test_case)
            elif test_case.test_type == BenchmarkType.END_TO_END:
                result = self._run_end_to_end_test(test_case)
            else:
                raise ValueError(f"Unknown test type: {test_case.test_type}")
            
            execution_time = time.time() - start_time
            memory_end = self._get_memory_usage() if self.config.memory_monitoring else None
            memory_usage = (memory_end - memory_start) if memory_start and memory_end else None
            
            result.execution_time = execution_time
            result.memory_usage = memory_usage
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            return BenchmarkResult(
                test_case=test_case,
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _run_ocr_test(self, test_case: TestCase) -> BenchmarkResult:
        """Run OCR accuracy test"""
        input_data = test_case.input_data
        
        if not isinstance(input_data, dict) or "reference" not in input_data or "ocr_output" not in input_data:
            raise ValueError("OCR test requires 'reference' and 'ocr_output' in input_data")
        
        reference = input_data["reference"]
        ocr_output = input_data["ocr_output"]
        
        # Convert to OCRResult if needed
        if isinstance(ocr_output, str):
            ocr_result = OCRResult(text=ocr_output)
        elif isinstance(ocr_output, dict):
            ocr_result = OCRResult(**ocr_output)
        else:
            ocr_result = ocr_output
        
        # Calculate OCR metrics
        report = self.ocr_calculator.evaluate_ocr_quality(reference, ocr_result)
        
        # Determine pass/fail
        success = report.overall_score >= self.config.ocr_accuracy_threshold
        
        return BenchmarkResult(
            test_case=test_case,
            success=success,
            score=report.overall_score,
            detailed_results=report.to_dict()
        )
    
    def _run_layout_test(self, test_case: TestCase) -> BenchmarkResult:
        """Run layout quality test"""
        input_data = test_case.input_data
        
        if not isinstance(input_data, dict) or "elements" not in input_data:
            raise ValueError("Layout test requires 'elements' in input_data")
        
        elements = input_data["elements"]
        page_info = input_data.get("page_info")
        
        # Convert elements to LayoutElement objects if needed
        layout_elements = []
        for elem_data in elements:
            if isinstance(elem_data, dict):
                from .layout_metrics import BoundingBox
                bbox = BoundingBox(**elem_data["bbox"])
                element = LayoutElement(
                    bbox=bbox,
                    element_type=elem_data["element_type"],
                    content=elem_data.get("content"),
                    style=elem_data.get("style", {}),
                    metadata=elem_data.get("metadata", {})
                )
                layout_elements.append(element)
            else:
                layout_elements.append(elem_data)
        
        # Calculate layout metrics
        report = self.layout_calculator.evaluate_layout_quality(layout_elements, page_info)
        
        # Determine pass/fail
        success = report.overall_score >= self.config.layout_quality_threshold
        
        return BenchmarkResult(
            test_case=test_case,
            success=success,
            score=report.overall_score,
            detailed_results=report.to_dict()
        )
    
    def _run_teds_test(self, test_case: TestCase) -> BenchmarkResult:
        """Run TEDS table structure test"""
        input_data = test_case.input_data
        
        if not isinstance(input_data, dict) or "predicted" not in input_data or "ground_truth" not in input_data:
            raise ValueError("TEDS test requires 'predicted' and 'ground_truth' in input_data")
        
        predicted = input_data["predicted"]
        ground_truth = input_data["ground_truth"]
        
        # Calculate TEDS score
        result = self.teds_calculator.calculate_teds(predicted, ground_truth)
        
        # Determine pass/fail
        success = result.teds_score >= self.config.teds_score_threshold
        
        return BenchmarkResult(
            test_case=test_case,
            success=success,
            score=result.teds_score,
            detailed_results=result.to_dict()
        )
    
    def _run_speed_test(self, test_case: TestCase) -> BenchmarkResult:
        """Run generation speed test"""
        input_data = test_case.input_data
        
        if not isinstance(input_data, dict) or "function" not in input_data:
            raise ValueError("Speed test requires 'function' in input_data")
        
        test_function = input_data["function"]
        args = input_data.get("args", [])
        kwargs = input_data.get("kwargs", {})
        iterations = input_data.get("iterations", 10)
        
        # Run multiple iterations
        execution_times = []
        
        for _ in range(iterations):
            start_time = time.time()
            
            # Execute function
            if callable(test_function):
                test_function(*args, **kwargs)
            else:
                raise ValueError("Function must be callable")
            
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            execution_times.append(execution_time)
        
        # Calculate statistics
        mean_time = statistics.mean(execution_times)
        median_time = statistics.median(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        
        # Determine pass/fail
        success = mean_time <= self.config.performance_threshold_ms
        
        return BenchmarkResult(
            test_case=test_case,
            success=success,
            score=1000.0 / mean_time if mean_time > 0 else 0.0,  # Inverse time as score
            detailed_results={
                "mean_time_ms": mean_time,
                "median_time_ms": median_time,
                "min_time_ms": min_time,
                "max_time_ms": max_time,
                "iterations": iterations,
                "all_times": execution_times
            }
        )
    
    def _run_memory_test(self, test_case: TestCase) -> BenchmarkResult:
        """Run memory usage test"""
        input_data = test_case.input_data
        
        if not isinstance(input_data, dict) or "function" not in input_data:
            raise ValueError("Memory test requires 'function' in input_data")
        
        test_function = input_data["function"]
        args = input_data.get("args", [])
        kwargs = input_data.get("kwargs", {})
        memory_limit_mb = input_data.get("memory_limit_mb", 1000)
        
        # Measure memory usage
        memory_start = self._get_memory_usage()
        
        # Execute function
        if callable(test_function):
            result = test_function(*args, **kwargs)
        else:
            raise ValueError("Function must be callable")
        
        memory_end = self._get_memory_usage()
        memory_used = memory_end - memory_start
        
        # Determine pass/fail
        success = memory_used <= memory_limit_mb
        
        return BenchmarkResult(
            test_case=test_case,
            success=success,
            score=max(0, 1.0 - (memory_used / memory_limit_mb)),
            detailed_results={
                "memory_used_mb": memory_used,
                "memory_limit_mb": memory_limit_mb,
                "memory_start_mb": memory_start,
                "memory_end_mb": memory_end
            }
        )
    
    def _run_scalability_test(self, test_case: TestCase) -> BenchmarkResult:
        """Run scalability test"""
        input_data = test_case.input_data
        
        if not isinstance(input_data, dict) or "function" not in input_data or "scale_factors" not in input_data:
            raise ValueError("Scalability test requires 'function' and 'scale_factors' in input_data")
        
        test_function = input_data["function"]
        scale_factors = input_data["scale_factors"]
        base_args = input_data.get("args", [])
        base_kwargs = input_data.get("kwargs", {})
        
        results = []
        
        for scale_factor in scale_factors:
            # Scale the arguments
            scaled_args = [arg * scale_factor if isinstance(arg, (int, float)) else arg for arg in base_args]
            scaled_kwargs = {
                k: v * scale_factor if isinstance(v, (int, float)) else v
                for k, v in base_kwargs.items()
            }
            
            # Measure execution time
            start_time = time.time()
            
            try:
                test_function(*scaled_args, **scaled_kwargs)
                execution_time = time.time() - start_time
                
                results.append({
                    "scale_factor": scale_factor,
                    "execution_time": execution_time,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "scale_factor": scale_factor,
                    "execution_time": time.time() - start_time,
                    "success": False,
                    "error": str(e)
                })
        
        # Analyze scalability
        successful_results = [r for r in results if r["success"]]
        
        if len(successful_results) >= 2:
            # Calculate time complexity approximation
            times = [r["execution_time"] for r in successful_results]
            scales = [r["scale_factor"] for r in successful_results]
            
            # Simple linear regression to estimate complexity
            complexity_score = self._estimate_time_complexity(scales, times)
            success = complexity_score <= 2.0  # Less than O(n^2)
        else:
            complexity_score = float('inf')
            success = False
        
        return BenchmarkResult(
            test_case=test_case,
            success=success,
            score=max(0, 1.0 / complexity_score) if complexity_score > 0 else 0,
            detailed_results={
                "scale_results": results,
                "complexity_estimate": complexity_score,
                "successful_scales": len(successful_results)
            }
        )
    
    def _run_end_to_end_test(self, test_case: TestCase) -> BenchmarkResult:
        """Run end-to-end integration test"""
        input_data = test_case.input_data
        
        if not isinstance(input_data, dict) or "pipeline" not in input_data:
            raise ValueError("End-to-end test requires 'pipeline' in input_data")
        
        pipeline_steps = input_data["pipeline"]
        pipeline_input = input_data.get("input")
        expected_output = test_case.expected_output
        
        # Execute pipeline steps
        current_output = pipeline_input
        step_results = []
        
        for i, step in enumerate(pipeline_steps):
            step_start = time.time()
            
            try:
                if callable(step):
                    current_output = step(current_output)
                elif isinstance(step, dict) and "function" in step:
                    func = step["function"]
                    args = step.get("args", [])
                    kwargs = step.get("kwargs", {})
                    current_output = func(current_output, *args, **kwargs)
                else:
                    raise ValueError(f"Invalid pipeline step: {step}")
                
                step_time = time.time() - step_start
                step_results.append({
                    "step": i,
                    "success": True,
                    "execution_time": step_time
                })
                
            except Exception as e:
                step_time = time.time() - step_start
                step_results.append({
                    "step": i,
                    "success": False,
                    "execution_time": step_time,
                    "error": str(e)
                })
                
                return BenchmarkResult(
                    test_case=test_case,
                    success=False,
                    error_message=f"Pipeline failed at step {i}: {str(e)}",
                    detailed_results={"step_results": step_results}
                )
        
        # Compare with expected output if provided
        if expected_output is not None:
            success = self._compare_outputs(current_output, expected_output)
            score = 1.0 if success else 0.0
        else:
            success = True  # No expected output to compare
            score = 1.0
        
        total_time = sum(r["execution_time"] for r in step_results)
        
        return BenchmarkResult(
            test_case=test_case,
            success=success,
            score=score,
            detailed_results={
                "step_results": step_results,
                "total_pipeline_time": total_time,
                "final_output": str(current_output)[:1000]  # Truncate for readability
            }
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            logger.warning("psutil not available, memory monitoring disabled")
            return 0.0
    
    def _estimate_time_complexity(self, scales: List[float], times: List[float]) -> float:
        """Estimate time complexity from scale factors and execution times"""
        if len(scales) < 2 or len(times) < 2:
            return 1.0
        
        # Calculate log-log slope (approximates complexity exponent)
        import math
        
        log_scales = [math.log(s) for s in scales if s > 0]
        log_times = [math.log(t) for t in times if t > 0]
        
        if len(log_scales) < 2:
            return 1.0
        
        # Simple linear regression on log-log plot
        n = len(log_scales)
        sum_x = sum(log_scales)
        sum_y = sum(log_times)
        sum_xy = sum(x * y for x, y in zip(log_scales, log_times))
        sum_x2 = sum(x * x for x in log_scales)
        
        # Calculate slope (complexity exponent)
        if n * sum_x2 - sum_x * sum_x != 0:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            return abs(slope)
        else:
            return 1.0
    
    def _compare_outputs(self, actual: Any, expected: Any) -> bool:
        """Compare actual and expected outputs"""
        if type(actual) != type(expected):
            return False
        
        if isinstance(actual, (str, int, float, bool)):
            return actual == expected
        
        elif isinstance(actual, (list, tuple)):
            if len(actual) != len(expected):
                return False
            return all(self._compare_outputs(a, e) for a, e in zip(actual, expected))
        
        elif isinstance(actual, dict):
            if set(actual.keys()) != set(expected.keys()):
                return False
            return all(self._compare_outputs(actual[k], expected[k]) for k in actual.keys())
        
        else:
            # For complex objects, convert to string and compare
            return str(actual) == str(expected)
    
    def run_comprehensive_benchmark(
        self,
        suites: List[BenchmarkSuite]
    ) -> BenchmarkReport:
        """Run comprehensive benchmark across multiple suites"""
        report_id = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting comprehensive benchmark: {report_id}")
        
        suite_results = {}
        all_results = []
        
        # Run all suites
        for suite in suites:
            results = self.run_benchmark_suite(suite)
            suite_results[suite.suite_name] = results
            all_results.extend(results)
        
        # Calculate aggregate statistics
        aggregate_stats = self._calculate_aggregate_statistics(all_results)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(all_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(aggregate_stats, performance_metrics)
        
        # Create report
        report = BenchmarkReport(
            report_id=report_id,
            timestamp=datetime.now(),
            suite_results=suite_results,
            aggregate_statistics=aggregate_stats,
            performance_metrics=performance_metrics,
            recommendations=recommendations
        )
        
        # Save report
        self._save_report(report)
        
        logger.info(f"Benchmark complete: {report_id}")
        return report
    
    def _calculate_aggregate_statistics(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Calculate aggregate statistics across all results"""
        if not results:
            return {}
        
        successful_results = [r for r in results if r.success]
        scores = [r.score for r in successful_results if r.score is not None]
        times = [r.execution_time for r in results if r.execution_time > 0]
        memory_usages = [r.memory_usage for r in results if r.memory_usage is not None]
        
        stats = {
            "total_tests": len(results),
            "successful_tests": len(successful_results),
            "failed_tests": len(results) - len(successful_results),
            "success_rate": len(successful_results) / len(results) if results else 0,
        }
        
        if scores:
            stats.update({
                "mean_score": statistics.mean(scores),
                "median_score": statistics.median(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "std_score": statistics.stdev(scores) if len(scores) > 1 else 0
            })
        
        if times:
            stats.update({
                "mean_execution_time": statistics.mean(times),
                "median_execution_time": statistics.median(times),
                "total_execution_time": sum(times)
            })
        
        if memory_usages:
            stats.update({
                "mean_memory_usage": statistics.mean(memory_usages),
                "peak_memory_usage": max(memory_usages)
            })
        
        return stats
    
    def _calculate_performance_metrics(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Calculate performance-specific metrics"""
        metrics = {}
        
        # Group by test type
        by_type = {}
        for result in results:
            test_type = result.test_case.test_type.value
            if test_type not in by_type:
                by_type[test_type] = []
            by_type[test_type].append(result)
        
        # Calculate metrics per type
        for test_type, type_results in by_type.items():
            successful = [r for r in type_results if r.success]
            scores = [r.score for r in successful if r.score is not None]
            
            metrics[test_type] = {
                "total_tests": len(type_results),
                "success_rate": len(successful) / len(type_results) if type_results else 0,
                "mean_score": statistics.mean(scores) if scores else 0,
                "passed_threshold": len([r for r in successful if self._passes_threshold(r)])
            }
        
        return metrics
    
    def _passes_threshold(self, result: BenchmarkResult) -> bool:
        """Check if result passes configured thresholds"""
        test_type = result.test_case.test_type
        score = result.score
        
        if score is None:
            return False
        
        if test_type == BenchmarkType.OCR_ACCURACY:
            return score >= self.config.ocr_accuracy_threshold
        elif test_type == BenchmarkType.LAYOUT_QUALITY:
            return score >= self.config.layout_quality_threshold
        elif test_type == BenchmarkType.TABLE_STRUCTURE:
            return score >= self.config.teds_score_threshold
        elif test_type == BenchmarkType.GENERATION_SPEED:
            return result.execution_time <= (self.config.performance_threshold_ms / 1000)
        else:
            return True  # Default pass for other types
    
    def _generate_recommendations(
        self,
        aggregate_stats: Dict[str, Any],
        performance_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Overall success rate
        success_rate = aggregate_stats.get("success_rate", 0)
        if success_rate < 0.8:
            recommendations.append("Overall success rate is below 80%. Review failed tests and improve reliability.")
        
        # Performance issues
        if "generation_speed" in performance_metrics:
            speed_metrics = performance_metrics["generation_speed"]
            if speed_metrics["success_rate"] < 0.7:
                recommendations.append("Generation speed is below threshold. Consider optimization strategies.")
        
        # Quality issues
        if "ocr_accuracy" in performance_metrics:
            ocr_metrics = performance_metrics["ocr_accuracy"]
            if ocr_metrics["mean_score"] < 0.85:
                recommendations.append("OCR accuracy could be improved. Review preprocessing and model selection.")
        
        if "layout_quality" in performance_metrics:
            layout_metrics = performance_metrics["layout_quality"]
            if layout_metrics["mean_score"] < 0.75:
                recommendations.append("Layout quality needs improvement. Focus on alignment and spacing consistency.")
        
        # Memory usage
        peak_memory = aggregate_stats.get("peak_memory_usage")
        if peak_memory and peak_memory > 1000:  # > 1GB
            recommendations.append("High memory usage detected. Consider memory optimization strategies.")
        
        if not recommendations:
            recommendations.append("All benchmarks performing well. Continue monitoring for consistency.")
        
        return recommendations
    
    def _save_report(self, report: BenchmarkReport) -> None:
        """Save benchmark report to file"""
        if self.config.save_detailed_results:
            report_file = self.output_dir / f"{report.report_id}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
            
            logger.info(f"Benchmark report saved: {report_file}")


def create_benchmark_runner(
    config: Optional[Union[Dict[str, Any], BenchmarkConfig]] = None
) -> BenchmarkRunner:
    """Factory function to create benchmark runner"""
    if isinstance(config, dict):
        config = BenchmarkConfig(**config)
    return BenchmarkRunner(config)


def run_sample_benchmark() -> BenchmarkReport:
    """Run sample benchmark for testing"""
    runner = create_benchmark_runner()
    
    # Create sample test suites
    sample_suites = [
        BenchmarkSuite(
            suite_name="OCR Quality Tests",
            benchmark_type=BenchmarkType.OCR_ACCURACY,
            test_cases=[
                TestCase(
                    test_id="ocr_perfect_match",
                    test_type=BenchmarkType.OCR_ACCURACY,
                    data_type=TestDataType.SYNTHETIC,
                    input_data={
                        "reference": "Hello World",
                        "ocr_output": OCRResult(text="Hello World", confidence=0.95)
                    }
                )
            ]
        )
    ]
    
    return runner.run_comprehensive_benchmark(sample_suites)