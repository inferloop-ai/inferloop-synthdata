"""
OCR quality assessment metrics for evaluating text recognition accuracy.
Implements BLEU, edit distance, character/word accuracy, and confidence scores.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any

import difflib
from pydantic import BaseModel, Field, validator

from ...core.config import BaseConfig
from ...core.exceptions import ValidationError
from ...core.logging import get_logger

logger = get_logger(__name__)


class MetricType(Enum):
    """OCR quality metric types"""
    CHARACTER_ACCURACY = "character_accuracy"
    WORD_ACCURACY = "word_accuracy"
    EDIT_DISTANCE = "edit_distance"
    BLEU_SCORE = "bleu_score"
    CONFIDENCE_SCORE = "confidence_score"
    FUZZY_MATCH = "fuzzy_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"


@dataclass
class OCRResult:
    """OCR result with confidence scores"""
    text: str
    confidence: float = 0.0
    word_confidences: List[float] = field(default_factory=list)
    bounding_boxes: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    """Individual metric calculation result"""
    metric_type: MetricType
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "metric": self.metric_type.value,
            "score": self.score,
            "details": self.details
        }


@dataclass
class OCRQualityReport:
    """Comprehensive OCR quality assessment report"""
    overall_score: float
    metrics: List[MetricResult]
    text_comparison: Dict[str, str]
    statistics: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "overall_score": self.overall_score,
            "metrics": [m.to_dict() for m in self.metrics],
            "text_comparison": self.text_comparison,
            "statistics": self.statistics,
            "recommendations": self.recommendations
        }


class OCRMetricsConfig(BaseConfig):
    """OCR metrics calculation configuration"""
    bleu_smoothing: bool = Field(default=True, description="Apply smoothing to BLEU score")
    case_sensitive: bool = Field(default=False, description="Case-sensitive comparison")
    normalize_whitespace: bool = Field(default=True, description="Normalize whitespace in comparisons")
    ignore_punctuation: bool = Field(default=False, description="Ignore punctuation in comparisons")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence threshold")
    fuzzy_threshold: float = Field(default=0.8, description="Fuzzy match threshold")
    
    @validator("confidence_threshold", "fuzzy_threshold")
    def validate_thresholds(cls, v):
        """Validate threshold values"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        return v


class OCRMetricsCalculator:
    """
    OCR quality metrics calculator.
    Evaluates text recognition accuracy using multiple metrics.
    """
    
    def __init__(self, config: Optional[OCRMetricsConfig] = None):
        """Initialize OCR metrics calculator"""
        self.config = config or OCRMetricsConfig()
        logger.info("Initialized OCRMetricsCalculator")
    
    def evaluate_ocr_quality(
        self,
        reference_text: str,
        ocr_result: Union[str, OCRResult],
        metrics: Optional[List[MetricType]] = None
    ) -> OCRQualityReport:
        """
        Evaluate OCR quality against reference text.
        
        Args:
            reference_text: Ground truth text
            ocr_result: OCR output text or result object
            metrics: List of metrics to calculate (None for all)
        
        Returns:
            Comprehensive quality report
        """
        # Parse OCR result
        if isinstance(ocr_result, str):
            ocr_text = ocr_result
            ocr_obj = OCRResult(text=ocr_result)
        else:
            ocr_text = ocr_result.text
            ocr_obj = ocr_result
        
        # Normalize texts
        ref_normalized = self._normalize_text(reference_text)
        ocr_normalized = self._normalize_text(ocr_text)
        
        # Calculate metrics
        if metrics is None:
            metrics = list(MetricType)
        
        metric_results = []
        for metric_type in metrics:
            try:
                result = self._calculate_metric(
                    metric_type, ref_normalized, ocr_normalized, ocr_obj
                )
                metric_results.append(result)
            except Exception as e:
                logger.warning(f"Failed to calculate {metric_type.value}: {str(e)}")
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(metric_results)
        
        # Generate statistics
        statistics = self._generate_statistics(ref_normalized, ocr_normalized, ocr_obj)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metric_results, statistics)
        
        return OCRQualityReport(
            overall_score=overall_score,
            metrics=metric_results,
            text_comparison={
                "reference": reference_text,
                "ocr_output": ocr_text,
                "reference_normalized": ref_normalized,
                "ocr_normalized": ocr_normalized
            },
            statistics=statistics,
            recommendations=recommendations
        )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        normalized = text
        
        if self.config.normalize_whitespace:
            # Normalize whitespace
            normalized = re.sub(r'\s+', ' ', normalized.strip())
        
        if not self.config.case_sensitive:
            normalized = normalized.lower()
        
        if self.config.ignore_punctuation:
            # Remove punctuation
            normalized = re.sub(r'[^\w\s]', '', normalized)
        
        return normalized
    
    def _calculate_metric(
        self,
        metric_type: MetricType,
        reference: str,
        ocr_text: str,
        ocr_result: OCRResult
    ) -> MetricResult:
        """Calculate individual metric"""
        if metric_type == MetricType.CHARACTER_ACCURACY:
            return self._calculate_character_accuracy(reference, ocr_text)
        
        elif metric_type == MetricType.WORD_ACCURACY:
            return self._calculate_word_accuracy(reference, ocr_text)
        
        elif metric_type == MetricType.EDIT_DISTANCE:
            return self._calculate_edit_distance(reference, ocr_text)
        
        elif metric_type == MetricType.BLEU_SCORE:
            return self._calculate_bleu_score(reference, ocr_text)
        
        elif metric_type == MetricType.CONFIDENCE_SCORE:
            return self._calculate_confidence_score(ocr_result)
        
        elif metric_type == MetricType.FUZZY_MATCH:
            return self._calculate_fuzzy_match(reference, ocr_text)
        
        elif metric_type == MetricType.SEMANTIC_SIMILARITY:
            return self._calculate_semantic_similarity(reference, ocr_text)
        
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
    
    def _calculate_character_accuracy(self, reference: str, ocr_text: str) -> MetricResult:
        """Calculate character-level accuracy"""
        if not reference:
            return MetricResult(MetricType.CHARACTER_ACCURACY, 0.0)
        
        # Use sequence matcher for character-level comparison
        matcher = difflib.SequenceMatcher(None, reference, ocr_text)
        matches = sum(match.size for match in matcher.get_matching_blocks())
        accuracy = matches / len(reference) if len(reference) > 0 else 0.0
        
        details = {
            "total_characters": len(reference),
            "matched_characters": matches,
            "ocr_length": len(ocr_text),
            "length_ratio": len(ocr_text) / len(reference) if len(reference) > 0 else 0
        }
        
        return MetricResult(MetricType.CHARACTER_ACCURACY, accuracy, details)
    
    def _calculate_word_accuracy(self, reference: str, ocr_text: str) -> MetricResult:
        """Calculate word-level accuracy"""
        ref_words = reference.split()
        ocr_words = ocr_text.split()
        
        if not ref_words:
            return MetricResult(MetricType.WORD_ACCURACY, 0.0)
        
        # Calculate word-level accuracy using sequence matching
        matcher = difflib.SequenceMatcher(None, ref_words, ocr_words)
        matches = sum(match.size for match in matcher.get_matching_blocks())
        accuracy = matches / len(ref_words) if len(ref_words) > 0 else 0.0
        
        details = {
            "total_words": len(ref_words),
            "matched_words": matches,
            "ocr_words": len(ocr_words),
            "word_ratio": len(ocr_words) / len(ref_words) if len(ref_words) > 0 else 0
        }
        
        return MetricResult(MetricType.WORD_ACCURACY, accuracy, details)
    
    def _calculate_edit_distance(self, reference: str, ocr_text: str) -> MetricResult:
        """Calculate normalized edit distance (Levenshtein)"""
        def levenshtein_distance(s1: str, s2: str) -> int:
            """Calculate Levenshtein distance between two strings"""
            if len(s1) < len(s2):
                s1, s2 = s2, s1
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein_distance(reference, ocr_text)
        max_length = max(len(reference), len(ocr_text))
        normalized_distance = distance / max_length if max_length > 0 else 0.0
        
        # Convert to similarity score (1 - normalized distance)
        similarity = 1.0 - normalized_distance
        
        details = {
            "edit_distance": distance,
            "max_length": max_length,
            "normalized_distance": normalized_distance,
            "similarity": similarity
        }
        
        return MetricResult(MetricType.EDIT_DISTANCE, similarity, details)
    
    def _calculate_bleu_score(self, reference: str, ocr_text: str) -> MetricResult:
        """Calculate BLEU score for OCR text"""
        def get_ngrams(text: str, n: int) -> List[Tuple[str, ...]]:
            """Get n-grams from text"""
            words = text.split()
            return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        
        def bleu_score(reference: str, candidate: str, max_n: int = 4) -> float:
            """Calculate BLEU score"""
            ref_words = reference.split()
            cand_words = candidate.split()
            
            if not ref_words or not cand_words:
                return 0.0
            
            # Calculate precision for each n-gram order
            precisions = []
            for n in range(1, min(max_n + 1, len(cand_words) + 1)):
                ref_ngrams = Counter(get_ngrams(reference, n))
                cand_ngrams = Counter(get_ngrams(candidate, n))
                
                if not cand_ngrams:
                    precisions.append(0.0)
                    continue
                
                matches = sum((ref_ngrams & cand_ngrams).values())
                total = sum(cand_ngrams.values())
                precision = matches / total if total > 0 else 0.0
                
                # Apply smoothing for zero counts
                if precision == 0.0 and self.config.bleu_smoothing:
                    precision = 1.0 / (2 * total) if total > 0 else 0.0
                
                precisions.append(precision)
            
            if not precisions or all(p == 0 for p in precisions):
                return 0.0
            
            # Geometric mean of precisions
            import math
            log_precisions = [math.log(p) for p in precisions if p > 0]
            if not log_precisions:
                return 0.0
            
            geometric_mean = math.exp(sum(log_precisions) / len(log_precisions))
            
            # Brevity penalty
            brevity_penalty = 1.0
            if len(cand_words) < len(ref_words):
                brevity_penalty = math.exp(1 - len(ref_words) / len(cand_words))
            
            return brevity_penalty * geometric_mean
        
        score = bleu_score(reference, ocr_text)
        
        details = {
            "reference_length": len(reference.split()),
            "candidate_length": len(ocr_text.split()),
            "ngram_precisions": []  # Could add detailed n-gram analysis
        }
        
        return MetricResult(MetricType.BLEU_SCORE, score, details)
    
    def _calculate_confidence_score(self, ocr_result: OCRResult) -> MetricResult:
        """Calculate confidence-based metrics"""
        overall_confidence = ocr_result.confidence
        word_confidences = ocr_result.word_confidences
        
        details = {
            "overall_confidence": overall_confidence,
            "word_count": len(word_confidences),
            "low_confidence_words": 0,
            "confidence_distribution": {}
        }
        
        if word_confidences:
            # Calculate statistics
            details["mean_confidence"] = statistics.mean(word_confidences)
            details["median_confidence"] = statistics.median(word_confidences)
            details["min_confidence"] = min(word_confidences)
            details["max_confidence"] = max(word_confidences)
            
            # Count low confidence words
            details["low_confidence_words"] = sum(
                1 for conf in word_confidences 
                if conf < self.config.confidence_threshold
            )
            
            # Confidence distribution
            bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            distribution = {f"{bins[i]}-{bins[i+1]}": 0 for i in range(len(bins)-1)}
            
            for conf in word_confidences:
                for i in range(len(bins)-1):
                    if bins[i] <= conf < bins[i+1] or (i == len(bins)-2 and conf == 1.0):
                        distribution[f"{bins[i]}-{bins[i+1]}"] += 1
                        break
            
            details["confidence_distribution"] = distribution
            
            # Use mean confidence as the metric score
            score = details["mean_confidence"]
        else:
            score = overall_confidence
        
        return MetricResult(MetricType.CONFIDENCE_SCORE, score, details)
    
    def _calculate_fuzzy_match(self, reference: str, ocr_text: str) -> MetricResult:
        """Calculate fuzzy string matching score"""
        # Simple fuzzy matching using difflib
        matcher = difflib.SequenceMatcher(None, reference, ocr_text)
        ratio = matcher.ratio()
        
        # Get matching blocks for detailed analysis
        matching_blocks = matcher.get_matching_blocks()
        total_matches = sum(block.size for block in matching_blocks)
        
        details = {
            "ratio": ratio,
            "matching_blocks": len(matching_blocks),
            "total_matching_chars": total_matches,
            "is_fuzzy_match": ratio >= self.config.fuzzy_threshold
        }
        
        return MetricResult(MetricType.FUZZY_MATCH, ratio, details)
    
    def _calculate_semantic_similarity(self, reference: str, ocr_text: str) -> MetricResult:
        """Calculate semantic similarity (simple implementation)"""
        # Simple word overlap-based semantic similarity
        ref_words = set(reference.lower().split())
        ocr_words = set(ocr_text.lower().split())
        
        if not ref_words:
            return MetricResult(MetricType.SEMANTIC_SIMILARITY, 0.0)
        
        # Jaccard similarity
        intersection = ref_words & ocr_words
        union = ref_words | ocr_words
        
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Word overlap ratio
        overlap_ratio = len(intersection) / len(ref_words) if ref_words else 0.0
        
        details = {
            "jaccard_similarity": jaccard,
            "word_overlap_ratio": overlap_ratio,
            "common_words": len(intersection),
            "reference_unique_words": len(ref_words),
            "ocr_unique_words": len(ocr_words)
        }
        
        # Use average of Jaccard and overlap ratio
        score = (jaccard + overlap_ratio) / 2.0
        
        return MetricResult(MetricType.SEMANTIC_SIMILARITY, score, details)
    
    def _calculate_overall_score(self, metric_results: List[MetricResult]) -> float:
        """Calculate weighted overall score"""
        if not metric_results:
            return 0.0
        
        # Define weights for different metrics
        weights = {
            MetricType.CHARACTER_ACCURACY: 0.25,
            MetricType.WORD_ACCURACY: 0.25,
            MetricType.EDIT_DISTANCE: 0.20,
            MetricType.BLEU_SCORE: 0.15,
            MetricType.CONFIDENCE_SCORE: 0.10,
            MetricType.FUZZY_MATCH: 0.03,
            MetricType.SEMANTIC_SIMILARITY: 0.02
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for result in metric_results:
            weight = weights.get(result.metric_type, 0.1)  # Default weight
            weighted_sum += result.score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _generate_statistics(
        self,
        reference: str,
        ocr_text: str,
        ocr_result: OCRResult
    ) -> Dict[str, Any]:
        """Generate detailed statistics"""
        return {
            "text_lengths": {
                "reference": len(reference),
                "ocr_output": len(ocr_text),
                "length_ratio": len(ocr_text) / len(reference) if reference else 0
            },
            "word_counts": {
                "reference": len(reference.split()),
                "ocr_output": len(ocr_text.split()),
                "word_ratio": len(ocr_text.split()) / len(reference.split()) if reference.split() else 0
            },
            "character_distribution": {
                "reference": dict(Counter(reference)),
                "ocr_output": dict(Counter(ocr_text))
            },
            "ocr_metadata": ocr_result.metadata
        }
    
    def _generate_recommendations(
        self,
        metric_results: List[MetricResult],
        statistics: Dict[str, Any]
    ) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Check individual metrics for issues
        for result in metric_results:
            if result.metric_type == MetricType.CHARACTER_ACCURACY and result.score < 0.8:
                recommendations.append("Character accuracy is low. Consider improving OCR preprocessing or using a different OCR engine.")
            
            elif result.metric_type == MetricType.WORD_ACCURACY and result.score < 0.7:
                recommendations.append("Word accuracy is low. Check for systematic word recognition errors.")
            
            elif result.metric_type == MetricType.CONFIDENCE_SCORE and result.score < 0.6:
                recommendations.append("Low OCR confidence scores detected. Consider image quality improvements.")
            
            elif result.metric_type == MetricType.BLEU_SCORE and result.score < 0.5:
                recommendations.append("BLEU score indicates poor text quality. Review document preprocessing.")
        
        # Check statistics for patterns
        length_ratio = statistics["text_lengths"]["length_ratio"]
        if length_ratio < 0.8:
            recommendations.append("OCR text is significantly shorter than reference. Check for missing content.")
        elif length_ratio > 1.2:
            recommendations.append("OCR text is significantly longer than reference. Check for false positive detections.")
        
        word_ratio = statistics["word_counts"]["word_ratio"]
        if word_ratio < 0.7:
            recommendations.append("Many words may be missing from OCR output.")
        elif word_ratio > 1.3:
            recommendations.append("OCR may be detecting spurious words.")
        
        if not recommendations:
            recommendations.append("OCR quality is acceptable. Continue monitoring for consistency.")
        
        return recommendations
    
    def batch_evaluate(
        self,
        test_cases: List[Tuple[str, Union[str, OCRResult]]],
        metrics: Optional[List[MetricType]] = None
    ) -> Dict[str, Any]:
        """Evaluate multiple OCR results in batch"""
        results = []
        
        for i, (reference, ocr_result) in enumerate(test_cases):
            try:
                report = self.evaluate_ocr_quality(reference, ocr_result, metrics)
                results.append({
                    "test_case": i,
                    "report": report
                })
            except Exception as e:
                logger.error(f"Failed to evaluate test case {i}: {str(e)}")
                results.append({
                    "test_case": i,
                    "error": str(e)
                })
        
        # Calculate aggregate statistics
        successful_results = [r["report"] for r in results if "report" in r]
        
        if successful_results:
            overall_scores = [r.overall_score for r in successful_results]
            aggregate_stats = {
                "total_cases": len(test_cases),
                "successful_evaluations": len(successful_results),
                "failed_evaluations": len(test_cases) - len(successful_results),
                "mean_overall_score": statistics.mean(overall_scores),
                "median_overall_score": statistics.median(overall_scores),
                "min_score": min(overall_scores),
                "max_score": max(overall_scores),
                "score_distribution": self._calculate_score_distribution(overall_scores)
            }
        else:
            aggregate_stats = {
                "total_cases": len(test_cases),
                "successful_evaluations": 0,
                "failed_evaluations": len(test_cases),
                "error": "No successful evaluations"
            }
        
        return {
            "results": results,
            "aggregate_statistics": aggregate_stats
        }
    
    def _calculate_score_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Calculate score distribution in bins"""
        bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        distribution = {f"{low}-{high}": 0 for low, high in bins}
        
        for score in scores:
            for low, high in bins:
                if low <= score < high or (high == 1.0 and score == 1.0):
                    distribution[f"{low}-{high}"] += 1
                    break
        
        return distribution


def create_ocr_metrics_calculator(
    config: Optional[Union[Dict[str, Any], OCRMetricsConfig]] = None
) -> OCRMetricsCalculator:
    """Factory function to create OCR metrics calculator"""
    if isinstance(config, dict):
        config = OCRMetricsConfig(**config)
    return OCRMetricsCalculator(config)


def evaluate_ocr_sample() -> OCRQualityReport:
    """Generate sample OCR evaluation for testing"""
    calculator = create_ocr_metrics_calculator()
    
    # Sample data
    reference = "The quick brown fox jumps over the lazy dog."
    ocr_output = OCRResult(
        text="The quiek brown fox junps over the lazy dog.",
        confidence=0.85,
        word_confidences=[0.9, 0.7, 0.95, 0.8, 0.6, 0.9, 0.88, 0.95, 0.92]
    )
    
    return calculator.evaluate_ocr_quality(reference, ocr_output)