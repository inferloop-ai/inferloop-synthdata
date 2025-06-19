#!/usr/bin/env python3
"""
Content Metrics for evaluating synthetic document quality
"""

import time
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import Counter

from pydantic import BaseModel, Field

from ...core.config import get_config
from ...core.logging import get_logger
from ...core.exceptions import ProcessingError


class MetricType(Enum):
    """Content metric types"""
    DIVERSITY = "diversity"
    REALISM = "realism"
    CONSISTENCY = "consistency"
    COMPLETENESS = "completeness"
    QUALITY = "quality"


@dataclass
class ContentMetricsResult:
    """Content metrics calculation result"""
    overall_score: float
    metric_scores: Dict[str, float]
    detailed_metrics: Dict[str, Any]
    processing_time: float


class ContentMetricsConfig(BaseModel):
    """Content metrics configuration"""
    enabled_metrics: List[MetricType] = Field(
        default=[MetricType.DIVERSITY, MetricType.REALISM, MetricType.CONSISTENCY],
        description="Enabled metric types"
    )
    min_sample_size: int = Field(10, description="Minimum sample size for metrics")
    weights: Dict[str, float] = Field(
        default={"diversity": 0.3, "realism": 0.4, "consistency": 0.3},
        description="Metric weights for overall score"
    )


class ContentMetrics:
    """
    Content Metrics calculator for synthetic document quality assessment
    
    Evaluates various aspects of generated content including diversity,
    realism, consistency, and overall quality.
    """
    
    def __init__(self, config: Optional[ContentMetricsConfig] = None):
        self.logger = get_logger(__name__)
        self.config = config or ContentMetricsConfig()
        
        self.logger.info("Content Metrics initialized")
    
    def calculate_metrics(self, documents: List[Dict[str, Any]]) -> ContentMetricsResult:
        """Calculate content metrics for document collection"""
        start_time = time.time()
        
        if len(documents) < self.config.min_sample_size:
            self.logger.warning(f"Sample size {len(documents)} below minimum {self.config.min_sample_size}")
        
        metric_scores = {}
        detailed_metrics = {}
        
        # Calculate individual metrics
        if MetricType.DIVERSITY in self.config.enabled_metrics:
            diversity_score, diversity_details = self._calculate_diversity(documents)
            metric_scores["diversity"] = diversity_score
            detailed_metrics["diversity"] = diversity_details
        
        if MetricType.REALISM in self.config.enabled_metrics:
            realism_score, realism_details = self._calculate_realism(documents)
            metric_scores["realism"] = realism_score
            detailed_metrics["realism"] = realism_details
        
        if MetricType.CONSISTENCY in self.config.enabled_metrics:
            consistency_score, consistency_details = self._calculate_consistency(documents)
            metric_scores["consistency"] = consistency_score
            detailed_metrics["consistency"] = consistency_details
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(metric_scores)
        
        processing_time = time.time() - start_time
        
        return ContentMetricsResult(
            overall_score=overall_score,
            metric_scores=metric_scores,
            detailed_metrics=detailed_metrics,
            processing_time=processing_time
        )
    
    def _calculate_diversity(self, documents: List[Dict[str, Any]]) -> tuple:
        """Calculate content diversity metrics"""
        try:
            # Text diversity
            texts = [doc.get('text_content', '') for doc in documents]
            text_diversity = self._calculate_text_diversity(texts)
            
            # Field diversity
            field_diversity = self._calculate_field_diversity(documents)
            
            # Value diversity
            value_diversity = self._calculate_value_diversity(documents)
            
            diversity_score = np.mean([text_diversity, field_diversity, value_diversity])
            
            details = {
                "text_diversity": text_diversity,
                "field_diversity": field_diversity,
                "value_diversity": value_diversity,
                "unique_documents": len(set(str(doc) for doc in documents)),
                "total_documents": len(documents)
            }
            
            return diversity_score, details
            
        except Exception as e:
            self.logger.error(f"Error calculating diversity: {e}")
            return 0.0, {"error": str(e)}
    
    def _calculate_realism(self, documents: List[Dict[str, Any]]) -> tuple:
        """Calculate content realism metrics"""
        try:
            # Check for realistic patterns
            realism_checks = {
                "valid_dates": self._check_date_validity(documents),
                "valid_formats": self._check_format_validity(documents),
                "realistic_values": self._check_value_realism(documents),
                "domain_consistency": self._check_domain_consistency(documents)
            }
            
            realism_score = np.mean(list(realism_checks.values()))
            
            details = {
                **realism_checks,
                "total_checks": len(realism_checks),
                "passed_checks": sum(1 for v in realism_checks.values() if v > 0.8)
            }
            
            return realism_score, details
            
        except Exception as e:
            self.logger.error(f"Error calculating realism: {e}")
            return 0.0, {"error": str(e)}
    
    def _calculate_consistency(self, documents: List[Dict[str, Any]]) -> tuple:
        """Calculate content consistency metrics"""
        try:
            # Schema consistency
            schema_consistency = self._check_schema_consistency(documents)
            
            # Format consistency  
            format_consistency = self._check_format_consistency(documents)
            
            # Value consistency
            value_consistency = self._check_value_consistency(documents)
            
            consistency_score = np.mean([schema_consistency, format_consistency, value_consistency])
            
            details = {
                "schema_consistency": schema_consistency,
                "format_consistency": format_consistency,
                "value_consistency": value_consistency,
                "consistent_fields": self._get_consistent_fields(documents)
            }
            
            return consistency_score, details
            
        except Exception as e:
            self.logger.error(f"Error calculating consistency: {e}")
            return 0.0, {"error": str(e)}
    
    def _calculate_text_diversity(self, texts: List[str]) -> float:
        """Calculate text content diversity"""
        if not texts:
            return 0.0
        
        # Simple diversity based on unique n-grams
        all_words = []
        for text in texts:
            words = text.lower().split()
            all_words.extend(words)
        
        if not all_words:
            return 0.0
        
        unique_words = len(set(all_words))
        total_words = len(all_words)
        
        return unique_words / total_words if total_words > 0 else 0.0
    
    def _calculate_field_diversity(self, documents: List[Dict[str, Any]]) -> float:
        """Calculate field-level diversity"""
        if not documents:
            return 0.0
        
        all_fields = set()
        for doc in documents:
            all_fields.update(doc.keys())
        
        # Check how many documents have each field
        field_coverage = {}
        for field in all_fields:
            count = sum(1 for doc in documents if field in doc)
            field_coverage[field] = count / len(documents)
        
        # Diversity is higher when fields appear in varied frequencies
        if not field_coverage:
            return 0.0
        
        coverage_values = list(field_coverage.values())
        return 1.0 - np.std(coverage_values)  # Lower std = more consistent = less diverse
    
    def _calculate_value_diversity(self, documents: List[Dict[str, Any]]) -> float:
        """Calculate value-level diversity"""
        if not documents:
            return 0.0
        
        # Flatten all values
        all_values = []
        for doc in documents:
            for value in doc.values():
                if isinstance(value, (str, int, float)):
                    all_values.append(str(value))
        
        if not all_values:
            return 0.0
        
        unique_values = len(set(all_values))
        total_values = len(all_values)
        
        return unique_values / total_values if total_values > 0 else 0.0
    
    def _check_date_validity(self, documents: List[Dict[str, Any]]) -> float:
        """Check for valid date formats"""
        # Simplified date validation
        date_fields = []
        valid_dates = 0
        
        for doc in documents:
            for key, value in doc.items():
                if 'date' in key.lower() and isinstance(value, str):
                    date_fields.append(value)
                    # Simple date pattern check
                    if any(char.isdigit() for char in value) and ('-' in value or '/' in value):
                        valid_dates += 1
        
        return valid_dates / len(date_fields) if date_fields else 1.0
    
    def _check_format_validity(self, documents: List[Dict[str, Any]]) -> float:
        """Check for valid field formats"""
        # Placeholder implementation
        return 0.85  # Assume 85% format validity
    
    def _check_value_realism(self, documents: List[Dict[str, Any]]) -> float:
        """Check for realistic values"""
        # Placeholder implementation
        return 0.80  # Assume 80% value realism
    
    def _check_domain_consistency(self, documents: List[Dict[str, Any]]) -> float:
        """Check for domain-specific consistency"""
        # Placeholder implementation
        return 0.75  # Assume 75% domain consistency
    
    def _check_schema_consistency(self, documents: List[Dict[str, Any]]) -> float:
        """Check schema consistency across documents"""
        if not documents:
            return 1.0
        
        # Get field sets for each document
        field_sets = [set(doc.keys()) for doc in documents]
        
        if not field_sets:
            return 1.0
        
        # Calculate Jaccard similarity across all documents
        base_fields = field_sets[0]
        similarities = []
        
        for field_set in field_sets[1:]:
            intersection = len(base_fields & field_set)
            union = len(base_fields | field_set)
            similarity = intersection / union if union > 0 else 0.0
            similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 1.0
    
    def _check_format_consistency(self, documents: List[Dict[str, Any]]) -> float:
        """Check format consistency for similar fields"""
        # Placeholder implementation
        return 0.90  # Assume 90% format consistency
    
    def _check_value_consistency(self, documents: List[Dict[str, Any]]) -> float:
        """Check value consistency patterns"""
        # Placeholder implementation
        return 0.85  # Assume 85% value consistency
    
    def _get_consistent_fields(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Get list of fields that appear consistently"""
        if not documents:
            return []
        
        field_counts = Counter()
        for doc in documents:
            field_counts.update(doc.keys())
        
        total_docs = len(documents)
        consistent_fields = [
            field for field, count in field_counts.items()
            if count / total_docs >= 0.8  # Appears in 80%+ of documents
        ]
        
        return consistent_fields
    
    def _calculate_overall_score(self, metric_scores: Dict[str, float]) -> float:
        """Calculate weighted overall score"""
        if not metric_scores:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, score in metric_scores.items():
            weight = self.config.weights.get(metric, 1.0)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0


# Factory function
def create_content_metrics(**config_kwargs) -> ContentMetrics:
    """Factory function to create content metrics"""
    config = ContentMetricsConfig(**config_kwargs)
    return ContentMetrics(config)