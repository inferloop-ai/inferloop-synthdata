# audio_synth/core/validators/base.py
"""
Base class for audio validators
"""

import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class BaseValidator(ABC):
    """Base class for all audio validators"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize validator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def validate(self, audio: torch.Tensor, metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Validate a single audio sample
        
        Args:
            audio: Audio tensor to validate
            metadata: Metadata for the audio sample
            
        Returns:
            Dictionary with validation metrics
        """
        pass
    
    def validate_batch(self, 
                      audios: List[torch.Tensor], 
                      metadata: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """
        Validate a batch of audio samples
        
        Args:
            audios: List of audio tensors
            metadata: List of metadata dictionaries
            
        Returns:
            List of validation results
        """
        if len(audios) != len(metadata):
            raise ValueError("Number of audios must match number of metadata entries")
        
        results = []
        for audio, meta in zip(audios, metadata):
            try:
                result = self.validate(audio, meta)
                results.append(result)
            except Exception as e:
                logger.error(f"Validation failed for sample: {e}")
                results.append({})
        
        return results
    
    def get_threshold(self, metric_name: str) -> float:
        """
        Get threshold for a specific metric
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Threshold value
        """
        # Default thresholds
        default_thresholds = {
            "quality": 0.7,
            "privacy": 0.8,
            "fairness": 0.75
        }
        
        return self.config.get(f"{metric_name}_threshold", 
                              default_thresholds.get(metric_name, 0.5))
    
    def is_passing(self, metrics: Dict[str, float], threshold: Optional[float] = None) -> bool:
        """
        Check if metrics pass validation
        
        Args:
            metrics: Validation metrics
            threshold: Optional threshold override
            
        Returns:
            True if passing, False otherwise
        """
        if not metrics:
            return False
        
        if threshold is None:
            threshold = self.get_threshold(self.__class__.__name__.lower().replace("validator", ""))
        
        # Use overall score if available, otherwise average all metrics
        if "overall_score" in metrics:
            return metrics["overall_score"] >= threshold
        elif "overall_quality" in metrics:
            return metrics["overall_quality"] >= threshold
        else:
            # Average all numeric metrics
            numeric_metrics = [v for v in metrics.values() if isinstance(v, (int, float))]
            if numeric_metrics:
                avg_score = sum(numeric_metrics) / len(numeric_metrics)
                return avg_score >= threshold
            else:
                return False
    
    def summarize_results(self, results: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Summarize batch validation results
        
        Args:
            results: List of validation results
            
        Returns:
            Summary statistics
        """
        if not results:
            return {}
        
        # Collect all metric names
        all_metrics = set()
        for result in results:
            all_metrics.update(result.keys())
        
        summary = {}
        
        for metric in all_metrics:
            values = [result.get(metric, 0) for result in results if metric in result]
            
            if values:
                import numpy as np
                summary[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values)
                }
        
        # Calculate pass rate
        threshold = self.get_threshold(self.__class__.__name__.lower().replace("validator", ""))
        passing_results = [result for result in results if self.is_passing(result, threshold)]
        
        summary["pass_rate"] = len(passing_results) / len(results) if results else 0.0
        summary["total_samples"] = len(results)
        summary["passing_samples"] = len(passing_results)
        summary["threshold_used"] = threshold
        
        return summary