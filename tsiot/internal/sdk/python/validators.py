"""
Time series validators for the TSIOT Python SDK.

This module provides validation classes for assessing the quality,
statistical properties, and privacy characteristics of time series data.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

from .client import TSIOTClient, AsyncTSIOTClient
from .timeseries import TimeSeries
from .utils import (
    ValidationError,
    TSIOTError,
    validate_positive_number,
    validate_non_negative_number,
    validate_string_not_empty,
    get_logger
)


class ValidationLevel(Enum):
    """Validation level options."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class QualityScore(Enum):
    """Quality score categories."""
    EXCELLENT = "excellent"  # 0.9 - 1.0
    GOOD = "good"           # 0.7 - 0.9
    FAIR = "fair"           # 0.5 - 0.7
    POOR = "poor"           # 0.0 - 0.5


@dataclass
class QualityMetrics:
    """
    Data quality metrics for time series.
    
    Attributes:
        completeness: Percentage of non-missing values (0.0 - 1.0)
        consistency: Consistency score (0.0 - 1.0)
        accuracy: Accuracy score (0.0 - 1.0)
        validity: Validity score (0.0 - 1.0)
        overall_score: Overall quality score (0.0 - 1.0)
        missing_count: Number of missing values
        outlier_count: Number of detected outliers
        duplicate_count: Number of duplicate timestamps
        quality_issues: List of identified quality issues
    """
    completeness: float
    consistency: float
    accuracy: float
    validity: float
    overall_score: float
    missing_count: int
    outlier_count: int
    duplicate_count: int
    quality_issues: List[str]
    
    @property
    def quality_category(self) -> QualityScore:
        """Get the quality category based on overall score."""
        if self.overall_score >= 0.9:
            return QualityScore.EXCELLENT
        elif self.overall_score >= 0.7:
            return QualityScore.GOOD
        elif self.overall_score >= 0.5:
            return QualityScore.FAIR
        else:
            return QualityScore.POOR
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "completeness": self.completeness,
            "consistency": self.consistency,
            "accuracy": self.accuracy,
            "validity": self.validity,
            "overall_score": self.overall_score,
            "missing_count": self.missing_count,
            "outlier_count": self.outlier_count,
            "duplicate_count": self.duplicate_count,
            "quality_issues": self.quality_issues,
            "quality_category": self.quality_category.value
        }


@dataclass
class StatisticalMetrics:
    """
    Statistical validation metrics.
    
    Attributes:
        stationarity_test: Results of stationarity tests
        normality_test: Results of normality tests
        autocorrelation_test: Results of autocorrelation tests
        trend_test: Results of trend tests
        seasonality_test: Results of seasonality tests
        outlier_test: Results of outlier detection
        distribution_fit: Best fitting distribution
        statistical_summary: Basic statistical summary
    """
    stationarity_test: Dict[str, Any]
    normality_test: Dict[str, Any]
    autocorrelation_test: Dict[str, Any]
    trend_test: Dict[str, Any]
    seasonality_test: Dict[str, Any]
    outlier_test: Dict[str, Any]
    distribution_fit: Dict[str, Any]
    statistical_summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "stationarity_test": self.stationarity_test,
            "normality_test": self.normality_test,
            "autocorrelation_test": self.autocorrelation_test,
            "trend_test": self.trend_test,
            "seasonality_test": self.seasonality_test,
            "outlier_test": self.outlier_test,
            "distribution_fit": self.distribution_fit,
            "statistical_summary": self.statistical_summary
        }


@dataclass
class PrivacyMetrics:
    """
    Privacy validation metrics.
    
    Attributes:
        k_anonymity: K-anonymity score
        l_diversity: L-diversity score
        t_closeness: T-closeness score
        differential_privacy: Differential privacy analysis
        privacy_budget: Remaining privacy budget
        privacy_risk: Overall privacy risk assessment
        recommendations: Privacy improvement recommendations
    """
    k_anonymity: Optional[Dict[str, Any]]
    l_diversity: Optional[Dict[str, Any]]
    t_closeness: Optional[Dict[str, Any]]
    differential_privacy: Optional[Dict[str, Any]]
    privacy_budget: Optional[float]
    privacy_risk: str
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "k_anonymity": self.k_anonymity,
            "l_diversity": self.l_diversity,
            "t_closeness": self.t_closeness,
            "differential_privacy": self.differential_privacy,
            "privacy_budget": self.privacy_budget,
            "privacy_risk": self.privacy_risk,
            "recommendations": self.recommendations
        }


@dataclass
class ValidationResult:
    """
    Complete validation result.
    
    Attributes:
        series_id: Time series identifier
        validation_timestamp: When validation was performed
        quality_metrics: Data quality metrics
        statistical_metrics: Statistical validation metrics
        privacy_metrics: Privacy validation metrics
        overall_pass: Whether validation passed overall
        recommendations: List of improvement recommendations
        validation_level: Level of validation performed
    """
    series_id: str
    validation_timestamp: datetime
    quality_metrics: Optional[QualityMetrics]
    statistical_metrics: Optional[StatisticalMetrics]
    privacy_metrics: Optional[PrivacyMetrics]
    overall_pass: bool
    recommendations: List[str]
    validation_level: ValidationLevel
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "series_id": self.series_id,
            "validation_timestamp": self.validation_timestamp.isoformat(),
            "overall_pass": self.overall_pass,
            "recommendations": self.recommendations,
            "validation_level": self.validation_level.value
        }
        
        if self.quality_metrics:
            result["quality_metrics"] = self.quality_metrics.to_dict()
        
        if self.statistical_metrics:
            result["statistical_metrics"] = self.statistical_metrics.to_dict()
        
        if self.privacy_metrics:
            result["privacy_metrics"] = self.privacy_metrics.to_dict()
        
        return result


class BaseValidator(ABC):
    """
    Abstract base class for time series validators.
    
    All validator classes inherit from this base and implement
    the validate method with specific validation logic.
    """
    
    def __init__(self, client: Union[TSIOTClient, AsyncTSIOTClient], logger: logging.Logger = None):
        """
        Initialize the validator.
        
        Args:
            client: TSIOT client instance
            logger: Logger instance
        """
        self.client = client
        self.logger = logger or get_logger(__name__)
        self._validator_type = self.__class__.__name__.replace("Validator", "").lower()
    
    @abstractmethod
    def validate(self, time_series: TimeSeries, **kwargs) -> ValidationResult:
        """
        Validate time series data.
        
        This method must be implemented by each validator subclass
        with specific validation logic.
        
        Args:
            time_series: Time series to validate
            **kwargs: Validator-specific parameters
        
        Returns:
            Validation results
        """
        pass
    
    def _build_request(
        self,
        time_series: TimeSeries,
        validation_types: List[str],
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Build a validation request dictionary.
        
        Args:
            time_series: Time series to validate
            validation_types: Types of validation to perform
            parameters: Validation parameters
        
        Returns:
            Request dictionary
        """
        return {
            "time_series": time_series.to_dict(),
            "validation_types": validation_types,
            "parameters": parameters or {}
        }


class TimeSeriesValidator(BaseValidator):
    """
    Main time series validator that supports multiple validation types.
    
    This is a high-level validator that provides convenient methods
    for validating time series using different validation approaches.
    
    Example:
        ```python
        validator = TimeSeriesValidator(client)
        
        # Comprehensive validation
        result = validator.validate(
            time_series,
            validation_types=["quality", "statistical", "privacy"],
            level=ValidationLevel.COMPREHENSIVE
        )
        
        print(f"Overall quality: {result.quality_metrics.quality_category.value}")
        print(f"Validation passed: {result.overall_pass}")
        ```
    """
    
    def validate(
        self,
        time_series: TimeSeries,
        validation_types: List[str] = None,
        level: ValidationLevel = ValidationLevel.STANDARD,
        quality_threshold: float = 0.7,
        **kwargs
    ) -> ValidationResult:
        """
        Validate time series data with multiple validation types.
        
        Args:
            time_series: Time series to validate
            validation_types: Types of validation to perform
            level: Validation level (basic, standard, comprehensive)
            quality_threshold: Minimum quality threshold for pass/fail
            **kwargs: Additional validation parameters
        
        Returns:
            Complete validation results
        """
        if not isinstance(time_series, TimeSeries):
            raise ValidationError("time_series must be a TimeSeries instance")
        
        if time_series.is_empty:
            raise ValidationError("Cannot validate empty time series")
        
        # Default validation types based on level
        if validation_types is None:
            if level == ValidationLevel.BASIC:
                validation_types = ["quality"]
            elif level == ValidationLevel.STANDARD:
                validation_types = ["quality", "statistical"]
            else:  # COMPREHENSIVE
                validation_types = ["quality", "statistical", "privacy"]
        
        # Build request
        parameters = {
            "level": level.value,
            "quality_threshold": validate_non_negative_number(quality_threshold, "quality_threshold"),
            **kwargs
        }
        
        request = self._build_request(time_series, validation_types, parameters)
        
        try:
            # Make validation request
            response = self.client.validate(
                time_series,
                validation_types=validation_types,
                timeout=kwargs.get("timeout")
            )
            
            # Parse response into ValidationResult
            return self._parse_validation_response(response, level)
            
        except Exception as e:
            raise ValidationError(f"Validation failed: {str(e)}")
    
    def _parse_validation_response(
        self,
        response: Dict[str, Any],
        level: ValidationLevel
    ) -> ValidationResult:
        """Parse validation response into ValidationResult object."""
        # Parse quality metrics
        quality_metrics = None
        if "quality_metrics" in response:
            qm_data = response["quality_metrics"]
            quality_metrics = QualityMetrics(
                completeness=qm_data.get("completeness", 1.0),
                consistency=qm_data.get("consistency", 1.0),
                accuracy=qm_data.get("accuracy", 1.0),
                validity=qm_data.get("validity", 1.0),
                overall_score=qm_data.get("overall_score", 1.0),
                missing_count=qm_data.get("missing_count", 0),
                outlier_count=qm_data.get("outlier_count", 0),
                duplicate_count=qm_data.get("duplicate_count", 0),
                quality_issues=qm_data.get("quality_issues", [])
            )
        
        # Parse statistical metrics
        statistical_metrics = None
        if "statistical_metrics" in response:
            sm_data = response["statistical_metrics"]
            statistical_metrics = StatisticalMetrics(
                stationarity_test=sm_data.get("stationarity_test", {}),
                normality_test=sm_data.get("normality_test", {}),
                autocorrelation_test=sm_data.get("autocorrelation_test", {}),
                trend_test=sm_data.get("trend_test", {}),
                seasonality_test=sm_data.get("seasonality_test", {}),
                outlier_test=sm_data.get("outlier_test", {}),
                distribution_fit=sm_data.get("distribution_fit", {}),
                statistical_summary=sm_data.get("statistical_summary", {})
            )
        
        # Parse privacy metrics
        privacy_metrics = None
        if "privacy_metrics" in response:
            pm_data = response["privacy_metrics"]
            privacy_metrics = PrivacyMetrics(
                k_anonymity=pm_data.get("k_anonymity"),
                l_diversity=pm_data.get("l_diversity"),
                t_closeness=pm_data.get("t_closeness"),
                differential_privacy=pm_data.get("differential_privacy"),
                privacy_budget=pm_data.get("privacy_budget"),
                privacy_risk=pm_data.get("privacy_risk", "unknown"),
                recommendations=pm_data.get("recommendations", [])
            )
        
        return ValidationResult(
            series_id=response.get("series_id", "unknown"),
            validation_timestamp=datetime.now(),
            quality_metrics=quality_metrics,
            statistical_metrics=statistical_metrics,
            privacy_metrics=privacy_metrics,
            overall_pass=response.get("overall_pass", False),
            recommendations=response.get("recommendations", []),
            validation_level=level
        )


class QualityValidator(BaseValidator):
    """Specialized validator for data quality assessment."""
    
    def validate(
        self,
        time_series: TimeSeries,
        completeness_threshold: float = 0.95,
        consistency_threshold: float = 0.9,
        outlier_threshold: float = 3.0,
        **kwargs
    ) -> QualityMetrics:
        """
        Validate data quality of time series.
        
        Args:
            time_series: Time series to validate
            completeness_threshold: Minimum completeness threshold
            consistency_threshold: Minimum consistency threshold
            outlier_threshold: Standard deviations for outlier detection
            **kwargs: Additional parameters
        
        Returns:
            Quality metrics
        """
        parameters = {
            "completeness_threshold": completeness_threshold,
            "consistency_threshold": consistency_threshold,
            "outlier_threshold": outlier_threshold,
            **kwargs
        }
        
        request = self._build_request(time_series, ["quality"], parameters)
        
        try:
            response = self.client.validate(
                time_series,
                validation_types=["quality"],
                timeout=kwargs.get("timeout")
            )
            
            qm_data = response.get("quality_metrics", {})
            return QualityMetrics(
                completeness=qm_data.get("completeness", 1.0),
                consistency=qm_data.get("consistency", 1.0),
                accuracy=qm_data.get("accuracy", 1.0),
                validity=qm_data.get("validity", 1.0),
                overall_score=qm_data.get("overall_score", 1.0),
                missing_count=qm_data.get("missing_count", 0),
                outlier_count=qm_data.get("outlier_count", 0),
                duplicate_count=qm_data.get("duplicate_count", 0),
                quality_issues=qm_data.get("quality_issues", [])
            )
            
        except Exception as e:
            raise ValidationError(f"Quality validation failed: {str(e)}")


class StatisticalValidator(BaseValidator):
    """Specialized validator for statistical properties."""
    
    def validate(
        self,
        time_series: TimeSeries,
        test_stationarity: bool = True,
        test_normality: bool = True,
        test_autocorrelation: bool = True,
        test_trend: bool = True,
        test_seasonality: bool = True,
        significance_level: float = 0.05,
        **kwargs
    ) -> StatisticalMetrics:
        """
        Validate statistical properties of time series.
        
        Args:
            time_series: Time series to validate
            test_stationarity: Whether to test for stationarity
            test_normality: Whether to test for normality
            test_autocorrelation: Whether to test for autocorrelation
            test_trend: Whether to test for trend
            test_seasonality: Whether to test for seasonality
            significance_level: Statistical significance level
            **kwargs: Additional parameters
        
        Returns:
            Statistical metrics
        """
        parameters = {
            "test_stationarity": test_stationarity,
            "test_normality": test_normality,
            "test_autocorrelation": test_autocorrelation,
            "test_trend": test_trend,
            "test_seasonality": test_seasonality,
            "significance_level": significance_level,
            **kwargs
        }
        
        request = self._build_request(time_series, ["statistical"], parameters)
        
        try:
            response = self.client.validate(
                time_series,
                validation_types=["statistical"],
                timeout=kwargs.get("timeout")
            )
            
            sm_data = response.get("statistical_metrics", {})
            return StatisticalMetrics(
                stationarity_test=sm_data.get("stationarity_test", {}),
                normality_test=sm_data.get("normality_test", {}),
                autocorrelation_test=sm_data.get("autocorrelation_test", {}),
                trend_test=sm_data.get("trend_test", {}),
                seasonality_test=sm_data.get("seasonality_test", {}),
                outlier_test=sm_data.get("outlier_test", {}),
                distribution_fit=sm_data.get("distribution_fit", {}),
                statistical_summary=sm_data.get("statistical_summary", {})
            )
            
        except Exception as e:
            raise ValidationError(f"Statistical validation failed: {str(e)}")


class PrivacyValidator(BaseValidator):
    """Specialized validator for privacy assessment."""
    
    def validate(
        self,
        time_series: TimeSeries,
        k_anonymity_k: int = 5,
        l_diversity_l: int = 2,
        t_closeness_t: float = 0.2,
        epsilon: float = 1.0,
        **kwargs
    ) -> PrivacyMetrics:
        """
        Validate privacy characteristics of time series.
        
        Args:
            time_series: Time series to validate
            k_anonymity_k: K parameter for k-anonymity
            l_diversity_l: L parameter for l-diversity
            t_closeness_t: T parameter for t-closeness
            epsilon: Epsilon parameter for differential privacy
            **kwargs: Additional parameters
        
        Returns:
            Privacy metrics
        """
        parameters = {
            "k_anonymity_k": k_anonymity_k,
            "l_diversity_l": l_diversity_l,
            "t_closeness_t": t_closeness_t,
            "epsilon": epsilon,
            **kwargs
        }
        
        request = self._build_request(time_series, ["privacy"], parameters)
        
        try:
            response = self.client.validate(
                time_series,
                validation_types=["privacy"],
                timeout=kwargs.get("timeout")
            )
            
            pm_data = response.get("privacy_metrics", {})
            return PrivacyMetrics(
                k_anonymity=pm_data.get("k_anonymity"),
                l_diversity=pm_data.get("l_diversity"),
                t_closeness=pm_data.get("t_closeness"),
                differential_privacy=pm_data.get("differential_privacy"),
                privacy_budget=pm_data.get("privacy_budget"),
                privacy_risk=pm_data.get("privacy_risk", "unknown"),
                recommendations=pm_data.get("recommendations", [])
            )
            
        except Exception as e:
            raise ValidationError(f"Privacy validation failed: {str(e)}")


# Convenience functions
def create_validator(
    client: Union[TSIOTClient, AsyncTSIOTClient],
    validator_type: str = "comprehensive"
) -> BaseValidator:
    """
    Create a validator instance of the specified type.
    
    Args:
        client: TSIOT client instance
        validator_type: Type of validator to create
    
    Returns:
        Validator instance
    """
    validator_classes = {
        "comprehensive": TimeSeriesValidator,
        "quality": QualityValidator,
        "statistical": StatisticalValidator,
        "privacy": PrivacyValidator
    }
    
    if validator_type not in validator_classes:
        raise ValidationError(
            f"Unknown validator type: {validator_type}",
            details={"available_types": list(validator_classes.keys())}
        )
    
    return validator_classes[validator_type](client)


def list_available_validators() -> List[str]:
    """Get list of available validator types."""
    return ["comprehensive", "quality", "statistical", "privacy"]


def quick_quality_check(time_series: TimeSeries) -> Dict[str, Any]:
    """
    Perform a quick quality check without server communication.
    
    Args:
        time_series: Time series to check
    
    Returns:
        Basic quality metrics
    """
    if time_series.is_empty:
        return {"error": "Empty time series"}
    
    values = time_series.values
    
    # Basic statistics
    import numpy as np
    
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    # Detect outliers (simple z-score method)
    z_scores = np.abs((values - mean_val) / std_val) if std_val > 0 else np.zeros_like(values)
    outliers = np.sum(z_scores > 3)
    
    # Check for missing values (NaN or infinite)
    missing = np.sum(np.isnan(values) | np.isinf(values))
    
    # Completeness score
    completeness = 1.0 - (missing / len(values))
    
    # Basic quality score
    outlier_ratio = outliers / len(values)
    quality_score = completeness * (1.0 - min(outlier_ratio, 0.5))
    
    return {
        "length": len(values),
        "completeness": completeness,
        "missing_count": int(missing),
        "outlier_count": int(outliers),
        "quality_score": quality_score,
        "mean": mean_val,
        "std": std_val,
        "min": np.min(values),
        "max": np.max(values)
    }