"""
Data quality drift detection for monitoring changes in document characteristics over time.
Detects distribution shifts, concept drift, and quality degradation patterns.
"""

from __future__ import annotations

import statistics
import warnings
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any

import math
from pydantic import BaseModel, Field, validator

from ...core.config import BaseConfig
from ...core.exceptions import ValidationError
from ...core.logging import get_logger

logger = get_logger(__name__)


class DriftType(Enum):
    """Types of drift detection"""
    STATISTICAL = "statistical"
    DISTRIBUTION = "distribution"
    PERFORMANCE = "performance"
    CONCEPT = "concept"
    COVARIATE = "covariate"
    LABEL = "label"


class DriftSeverity(Enum):
    """Drift severity levels"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftMethod(Enum):
    """Drift detection methods"""
    KS_TEST = "kolmogorov_smirnov"
    CHI_SQUARE = "chi_square"
    POPULATION_STABILITY = "population_stability_index"
    JENSEN_SHANNON = "jensen_shannon_divergence"
    WASSERSTEIN = "wasserstein_distance"
    STATISTICAL_MOMENTS = "statistical_moments"


@dataclass
class DriftMetric:
    """Individual drift metric result"""
    metric_name: str
    drift_score: float
    p_value: Optional[float] = None
    threshold: float = 0.05
    is_drift: bool = False
    severity: DriftSeverity = DriftSeverity.NONE
    method: DriftMethod = DriftMethod.KS_TEST
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "metric_name": self.metric_name,
            "drift_score": self.drift_score,
            "p_value": self.p_value,
            "threshold": self.threshold,
            "is_drift": self.is_drift,
            "severity": self.severity.value,
            "method": self.method.value,
            "details": self.details
        }


@dataclass
class DataWindow:
    """Time window of data for drift analysis"""
    timestamp: datetime
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def get_feature_values(self, feature_name: str) -> List[Any]:
        """Extract values for a specific feature"""
        return [item.get(feature_name) for item in self.data if feature_name in item]
    
    def get_numeric_features(self) -> List[str]:
        """Get list of numeric feature names"""
        if not self.data:
            return []
        
        sample = self.data[0]
        numeric_features = []
        
        for key, value in sample.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_features.append(key)
        
        return numeric_features
    
    def get_categorical_features(self) -> List[str]:
        """Get list of categorical feature names"""
        if not self.data:
            return []
        
        sample = self.data[0]
        categorical_features = []
        
        for key, value in sample.items():
            if isinstance(value, (str, bool)) or value is None:
                categorical_features.append(key)
        
        return categorical_features


@dataclass
class DriftReport:
    """Comprehensive drift detection report"""
    timestamp: datetime
    reference_window: DataWindow
    current_window: DataWindow
    drift_metrics: List[DriftMetric]
    overall_drift_score: float
    drift_detected: bool
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "reference_window_size": len(self.reference_window),
            "current_window_size": len(self.current_window),
            "drift_metrics": [m.to_dict() for m in self.drift_metrics],
            "overall_drift_score": self.overall_drift_score,
            "drift_detected": self.drift_detected,
            "recommendations": self.recommendations
        }


class DriftDetectorConfig(BaseConfig):
    """Drift detector configuration"""
    detection_methods: List[DriftMethod] = Field(
        default=[DriftMethod.KS_TEST, DriftMethod.POPULATION_STABILITY],
        description="Drift detection methods to use"
    )
    significance_level: float = Field(default=0.05, description="Statistical significance level")
    min_samples: int = Field(default=30, description="Minimum samples for drift detection")
    window_size: int = Field(default=1000, description="Size of data window")
    drift_threshold: float = Field(default=0.1, description="Drift score threshold")
    
    # Severity thresholds
    low_drift_threshold: float = Field(default=0.05, description="Low drift threshold")
    medium_drift_threshold: float = Field(default=0.1, description="Medium drift threshold")
    high_drift_threshold: float = Field(default=0.2, description="High drift threshold")
    critical_drift_threshold: float = Field(default=0.5, description="Critical drift threshold")
    
    @validator("significance_level", "drift_threshold")
    def validate_thresholds(cls, v):
        """Validate threshold values"""
        if not 0 < v < 1:
            raise ValueError("Thresholds must be between 0 and 1")
        return v


class DriftDetector:
    """
    Data quality drift detector.
    Monitors changes in data distributions and characteristics over time.
    """
    
    def __init__(self, config: Optional[DriftDetectorConfig] = None):
        """Initialize drift detector"""
        self.config = config or DriftDetectorConfig()
        self.reference_data: Optional[DataWindow] = None
        self.drift_history: List[DriftReport] = []
        
        logger.info("Initialized DriftDetector")
    
    def set_reference_data(self, data: List[Dict[str, Any]], timestamp: Optional[datetime] = None) -> None:
        """Set reference data for drift detection"""
        if len(data) < self.config.min_samples:
            raise ValueError(f"Reference data must have at least {self.config.min_samples} samples")
        
        self.reference_data = DataWindow(
            timestamp=timestamp or datetime.now(),
            data=data[:self.config.window_size]  # Limit to window size
        )
        
        logger.info(f"Set reference data with {len(self.reference_data)} samples")
    
    def detect_drift(
        self,
        current_data: List[Dict[str, Any]],
        timestamp: Optional[datetime] = None
    ) -> DriftReport:
        """Detect drift between reference and current data"""
        if self.reference_data is None:
            raise ValueError("Reference data must be set before drift detection")
        
        if len(current_data) < self.config.min_samples:
            raise ValueError(f"Current data must have at least {self.config.min_samples} samples")
        
        current_window = DataWindow(
            timestamp=timestamp or datetime.now(),
            data=current_data[:self.config.window_size]
        )
        
        # Calculate drift metrics
        drift_metrics = []
        
        # Detect drift for numeric features
        numeric_features = self.reference_data.get_numeric_features()
        for feature in numeric_features:
            if feature in current_window.get_numeric_features():
                metrics = self._detect_numeric_drift(feature, current_window)
                drift_metrics.extend(metrics)
        
        # Detect drift for categorical features
        categorical_features = self.reference_data.get_categorical_features()
        for feature in categorical_features:
            if feature in current_window.get_categorical_features():
                metrics = self._detect_categorical_drift(feature, current_window)
                drift_metrics.extend(metrics)
        
        # Calculate overall drift score
        overall_score = self._calculate_overall_drift_score(drift_metrics)
        drift_detected = overall_score > self.config.drift_threshold
        
        # Generate recommendations
        recommendations = self._generate_recommendations(drift_metrics, overall_score)
        
        # Create report
        report = DriftReport(
            timestamp=current_window.timestamp,
            reference_window=self.reference_data,
            current_window=current_window,
            drift_metrics=drift_metrics,
            overall_drift_score=overall_score,
            drift_detected=drift_detected,
            recommendations=recommendations
        )
        
        # Store in history
        self.drift_history.append(report)
        
        if drift_detected:
            logger.warning(f"Drift detected with score {overall_score:.3f}")
        else:
            logger.info(f"No significant drift detected (score: {overall_score:.3f})")
        
        return report
    
    def _detect_numeric_drift(self, feature: str, current_window: DataWindow) -> List[DriftMetric]:
        """Detect drift for numeric features"""
        ref_values = [v for v in self.reference_data.get_feature_values(feature) if v is not None]
        cur_values = [v for v in current_window.get_feature_values(feature) if v is not None]
        
        if not ref_values or not cur_values:
            return []
        
        metrics = []
        
        # Kolmogorov-Smirnov test
        if DriftMethod.KS_TEST in self.config.detection_methods:
            ks_metric = self._kolmogorov_smirnov_test(feature, ref_values, cur_values)
            metrics.append(ks_metric)
        
        # Population Stability Index
        if DriftMethod.POPULATION_STABILITY in self.config.detection_methods:
            psi_metric = self._population_stability_index(feature, ref_values, cur_values)
            metrics.append(psi_metric)
        
        # Statistical moments comparison
        if DriftMethod.STATISTICAL_MOMENTS in self.config.detection_methods:
            moments_metric = self._statistical_moments_drift(feature, ref_values, cur_values)
            metrics.append(moments_metric)
        
        # Wasserstein distance (simplified implementation)
        if DriftMethod.WASSERSTEIN in self.config.detection_methods:
            wasserstein_metric = self._wasserstein_distance(feature, ref_values, cur_values)
            metrics.append(wasserstein_metric)
        
        return metrics
    
    def _detect_categorical_drift(self, feature: str, current_window: DataWindow) -> List[DriftMetric]:
        """Detect drift for categorical features"""
        ref_values = [v for v in self.reference_data.get_feature_values(feature) if v is not None]
        cur_values = [v for v in current_window.get_feature_values(feature) if v is not None]
        
        if not ref_values or not cur_values:
            return []
        
        metrics = []
        
        # Chi-square test
        if DriftMethod.CHI_SQUARE in self.config.detection_methods:
            chi2_metric = self._chi_square_test(feature, ref_values, cur_values)
            metrics.append(chi2_metric)
        
        # Population Stability Index for categorical
        if DriftMethod.POPULATION_STABILITY in self.config.detection_methods:
            psi_metric = self._categorical_psi(feature, ref_values, cur_values)
            metrics.append(psi_metric)
        
        # Jensen-Shannon divergence
        if DriftMethod.JENSEN_SHANNON in self.config.detection_methods:
            js_metric = self._jensen_shannon_divergence(feature, ref_values, cur_values)
            metrics.append(js_metric)
        
        return metrics
    
    def _kolmogorov_smirnov_test(self, feature: str, ref_values: List[float], cur_values: List[float]) -> DriftMetric:
        """Kolmogorov-Smirnov test for numeric drift"""
        # Simple implementation of KS test
        def empirical_cdf(values: List[float], x: float) -> float:
            return sum(1 for v in values if v <= x) / len(values)
        
        # Get all unique values
        all_values = sorted(set(ref_values + cur_values))
        
        # Calculate maximum difference between CDFs
        max_diff = 0.0
        for x in all_values:
            ref_cdf = empirical_cdf(ref_values, x)
            cur_cdf = empirical_cdf(cur_values, x)
            diff = abs(ref_cdf - cur_cdf)
            max_diff = max(max_diff, diff)
        
        # Calculate critical value (simplified)
        n1, n2 = len(ref_values), len(cur_values)
        critical_value = 1.36 * math.sqrt((n1 + n2) / (n1 * n2))
        
        is_drift = max_diff > critical_value
        p_value = math.exp(-2 * n1 * n2 * max_diff**2 / (n1 + n2)) if max_diff > 0 else 1.0
        
        return DriftMetric(
            metric_name=f"ks_test_{feature}",
            drift_score=max_diff,
            p_value=p_value,
            threshold=critical_value,
            is_drift=is_drift,
            severity=self._get_drift_severity(max_diff),
            method=DriftMethod.KS_TEST,
            details={
                "critical_value": critical_value,
                "max_difference": max_diff,
                "ref_samples": n1,
                "cur_samples": n2
            }
        )
    
    def _population_stability_index(self, feature: str, ref_values: List[float], cur_values: List[float]) -> DriftMetric:
        """Population Stability Index for numeric features"""
        # Create bins based on reference data quantiles
        ref_sorted = sorted(ref_values)
        n_bins = min(10, len(set(ref_values)))  # Max 10 bins
        
        if n_bins < 2:
            return DriftMetric(
                metric_name=f"psi_{feature}",
                drift_score=0.0,
                is_drift=False,
                severity=DriftSeverity.NONE,
                method=DriftMethod.POPULATION_STABILITY
            )
        
        # Calculate bin edges
        bin_edges = []
        for i in range(n_bins + 1):
            if i == 0:
                bin_edges.append(float('-inf'))
            elif i == n_bins:
                bin_edges.append(float('inf'))
            else:
                idx = int(i * len(ref_sorted) / n_bins)
                bin_edges.append(ref_sorted[idx])
        
        # Count values in each bin
        def count_in_bins(values: List[float], edges: List[float]) -> List[int]:
            counts = [0] * (len(edges) - 1)
            for value in values:
                for i in range(len(edges) - 1):
                    if edges[i] <= value < edges[i + 1]:
                        counts[i] += 1
                        break
            return counts
        
        ref_counts = count_in_bins(ref_values, bin_edges)
        cur_counts = count_in_bins(cur_values, bin_edges)
        
        # Calculate PSI
        psi = 0.0
        for ref_count, cur_count in zip(ref_counts, cur_counts):
            ref_pct = (ref_count + 1e-6) / len(ref_values)  # Add small epsilon
            cur_pct = (cur_count + 1e-6) / len(cur_values)
            psi += (cur_pct - ref_pct) * math.log(cur_pct / ref_pct)
        
        is_drift = psi > 0.1  # Common PSI threshold
        
        return DriftMetric(
            metric_name=f"psi_{feature}",
            drift_score=psi,
            threshold=0.1,
            is_drift=is_drift,
            severity=self._get_drift_severity(psi),
            method=DriftMethod.POPULATION_STABILITY,
            details={
                "n_bins": n_bins,
                "ref_distribution": ref_counts,
                "cur_distribution": cur_counts,
                "bin_edges": bin_edges[1:-1]  # Exclude inf values
            }
        )
    
    def _statistical_moments_drift(self, feature: str, ref_values: List[float], cur_values: List[float]) -> DriftMetric:
        """Compare statistical moments (mean, std, skewness, kurtosis)"""
        def safe_moment(values: List[float], moment: int) -> float:
            if len(values) < 2:
                return 0.0
            
            if moment == 1:  # Mean
                return statistics.mean(values)
            elif moment == 2:  # Variance
                return statistics.variance(values)
            elif moment == 3:  # Skewness (simplified)
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 1.0
                if std_val == 0:
                    return 0.0
                skew = sum(((x - mean_val) / std_val) ** 3 for x in values) / len(values)
                return skew
            elif moment == 4:  # Kurtosis (simplified)
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 1.0
                if std_val == 0:
                    return 0.0
                kurt = sum(((x - mean_val) / std_val) ** 4 for x in values) / len(values) - 3
                return kurt
            return 0.0
        
        # Calculate moments
        ref_moments = [safe_moment(ref_values, i) for i in range(1, 5)]
        cur_moments = [safe_moment(cur_values, i) for i in range(1, 5)]
        
        # Calculate normalized differences
        moment_diffs = []
        for ref_m, cur_m in zip(ref_moments, cur_moments):
            if ref_m != 0:
                diff = abs(cur_m - ref_m) / abs(ref_m)
            else:
                diff = abs(cur_m)
            moment_diffs.append(diff)
        
        # Overall drift score is mean of moment differences
        drift_score = statistics.mean(moment_diffs)
        is_drift = drift_score > 0.1
        
        return DriftMetric(
            metric_name=f"moments_{feature}",
            drift_score=drift_score,
            threshold=0.1,
            is_drift=is_drift,
            severity=self._get_drift_severity(drift_score),
            method=DriftMethod.STATISTICAL_MOMENTS,
            details={
                "ref_moments": {"mean": ref_moments[0], "var": ref_moments[1], 
                               "skew": ref_moments[2], "kurt": ref_moments[3]},
                "cur_moments": {"mean": cur_moments[0], "var": cur_moments[1], 
                               "skew": cur_moments[2], "kurt": cur_moments[3]},
                "moment_diffs": moment_diffs
            }
        )
    
    def _wasserstein_distance(self, feature: str, ref_values: List[float], cur_values: List[float]) -> DriftMetric:
        """Simplified Wasserstein distance (Earth Mover's Distance)"""
        # Sort both distributions
        ref_sorted = sorted(ref_values)
        cur_sorted = sorted(cur_values)
        
        # Normalize to same length (simplified approach)
        n = min(len(ref_sorted), len(cur_sorted))
        
        if n < 2:
            return DriftMetric(
                metric_name=f"wasserstein_{feature}",
                drift_score=0.0,
                is_drift=False,
                severity=DriftSeverity.NONE,
                method=DriftMethod.WASSERSTEIN
            )
        
        ref_sample = [ref_sorted[int(i * len(ref_sorted) / n)] for i in range(n)]
        cur_sample = [cur_sorted[int(i * len(cur_sorted) / n)] for i in range(n)]
        
        # Calculate average absolute difference
        distance = sum(abs(r - c) for r, c in zip(ref_sample, cur_sample)) / n
        
        # Normalize by reference range
        ref_range = max(ref_values) - min(ref_values)
        if ref_range > 0:
            normalized_distance = distance / ref_range
        else:
            normalized_distance = 0.0
        
        is_drift = normalized_distance > 0.1
        
        return DriftMetric(
            metric_name=f"wasserstein_{feature}",
            drift_score=normalized_distance,
            threshold=0.1,
            is_drift=is_drift,
            severity=self._get_drift_severity(normalized_distance),
            method=DriftMethod.WASSERSTEIN,
            details={
                "raw_distance": distance,
                "ref_range": ref_range,
                "sample_size": n
            }
        )
    
    def _chi_square_test(self, feature: str, ref_values: List[str], cur_values: List[str]) -> DriftMetric:
        """Chi-square test for categorical drift"""
        # Get all unique categories
        all_categories = list(set(ref_values + cur_values))
        
        if len(all_categories) < 2:
            return DriftMetric(
                metric_name=f"chi2_{feature}",
                drift_score=0.0,
                is_drift=False,
                severity=DriftSeverity.NONE,
                method=DriftMethod.CHI_SQUARE
            )
        
        # Count occurrences
        ref_counts = Counter(ref_values)
        cur_counts = Counter(cur_values)
        
        # Calculate chi-square statistic
        chi2_stat = 0.0
        total_ref = len(ref_values)
        total_cur = len(cur_values)
        
        for category in all_categories:
            ref_count = ref_counts.get(category, 0)
            cur_count = cur_counts.get(category, 0)
            
            # Expected counts (proportional)
            total_count = ref_count + cur_count
            if total_count > 0:
                expected_ref = total_count * total_ref / (total_ref + total_cur)
                expected_cur = total_count * total_cur / (total_ref + total_cur)
                
                if expected_ref > 0:
                    chi2_stat += (ref_count - expected_ref) ** 2 / expected_ref
                if expected_cur > 0:
                    chi2_stat += (cur_count - expected_cur) ** 2 / expected_cur
        
        # Degrees of freedom
        df = len(all_categories) - 1
        
        # Critical value (simplified, for alpha=0.05)
        critical_values = {1: 3.84, 2: 5.99, 3: 7.81, 4: 9.49, 5: 11.07}
        critical_value = critical_values.get(df, 3.84 * df)  # Approximation for higher df
        
        is_drift = chi2_stat > critical_value
        
        return DriftMetric(
            metric_name=f"chi2_{feature}",
            drift_score=chi2_stat / critical_value,  # Normalized score
            threshold=1.0,
            is_drift=is_drift,
            severity=self._get_drift_severity(chi2_stat / critical_value),
            method=DriftMethod.CHI_SQUARE,
            details={
                "chi2_statistic": chi2_stat,
                "degrees_of_freedom": df,
                "critical_value": critical_value,
                "categories": all_categories,
                "ref_distribution": dict(ref_counts),
                "cur_distribution": dict(cur_counts)
            }
        )
    
    def _categorical_psi(self, feature: str, ref_values: List[str], cur_values: List[str]) -> DriftMetric:
        """Population Stability Index for categorical features"""
        # Count occurrences
        ref_counts = Counter(ref_values)
        cur_counts = Counter(cur_values)
        
        # Get all categories
        all_categories = set(ref_values + cur_values)
        
        # Calculate PSI
        psi = 0.0
        total_ref = len(ref_values)
        total_cur = len(cur_values)
        
        for category in all_categories:
            ref_pct = (ref_counts.get(category, 0) + 1e-6) / total_ref
            cur_pct = (cur_counts.get(category, 0) + 1e-6) / total_cur
            psi += (cur_pct - ref_pct) * math.log(cur_pct / ref_pct)
        
        is_drift = psi > 0.1
        
        return DriftMetric(
            metric_name=f"categorical_psi_{feature}",
            drift_score=psi,
            threshold=0.1,
            is_drift=is_drift,
            severity=self._get_drift_severity(psi),
            method=DriftMethod.POPULATION_STABILITY,
            details={
                "categories": list(all_categories),
                "ref_distribution": dict(ref_counts),
                "cur_distribution": dict(cur_counts)
            }
        )
    
    def _jensen_shannon_divergence(self, feature: str, ref_values: List[str], cur_values: List[str]) -> DriftMetric:
        """Jensen-Shannon divergence for categorical distributions"""
        # Count occurrences and get probabilities
        ref_counts = Counter(ref_values)
        cur_counts = Counter(cur_values)
        all_categories = set(ref_values + cur_values)
        
        total_ref = len(ref_values)
        total_cur = len(cur_values)
        
        # Calculate probability distributions
        ref_probs = {cat: (ref_counts.get(cat, 0) + 1e-6) / total_ref for cat in all_categories}
        cur_probs = {cat: (cur_counts.get(cat, 0) + 1e-6) / total_cur for cat in all_categories}
        
        # Calculate JS divergence
        def kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
            return sum(p[cat] * math.log(p[cat] / q[cat]) for cat in p.keys())
        
        # Average distribution
        avg_probs = {cat: (ref_probs[cat] + cur_probs[cat]) / 2 for cat in all_categories}
        
        js_div = 0.5 * kl_divergence(ref_probs, avg_probs) + 0.5 * kl_divergence(cur_probs, avg_probs)
        js_distance = math.sqrt(js_div)  # JS distance is square root of divergence
        
        is_drift = js_distance > 0.1
        
        return DriftMetric(
            metric_name=f"js_div_{feature}",
            drift_score=js_distance,
            threshold=0.1,
            is_drift=is_drift,
            severity=self._get_drift_severity(js_distance),
            method=DriftMethod.JENSEN_SHANNON,
            details={
                "js_divergence": js_div,
                "ref_probs": ref_probs,
                "cur_probs": cur_probs,
                "avg_probs": avg_probs
            }
        )
    
    def _get_drift_severity(self, score: float) -> DriftSeverity:
        """Determine drift severity based on score"""
        if score < self.config.low_drift_threshold:
            return DriftSeverity.NONE
        elif score < self.config.medium_drift_threshold:
            return DriftSeverity.LOW
        elif score < self.config.high_drift_threshold:
            return DriftSeverity.MEDIUM
        elif score < self.config.critical_drift_threshold:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL
    
    def _calculate_overall_drift_score(self, drift_metrics: List[DriftMetric]) -> float:
        """Calculate overall drift score from individual metrics"""
        if not drift_metrics:
            return 0.0
        
        # Weight different types of metrics
        weights = {
            DriftMethod.KS_TEST: 0.3,
            DriftMethod.CHI_SQUARE: 0.3,
            DriftMethod.POPULATION_STABILITY: 0.25,
            DriftMethod.JENSEN_SHANNON: 0.2,
            DriftMethod.WASSERSTEIN: 0.15,
            DriftMethod.STATISTICAL_MOMENTS: 0.1
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric in drift_metrics:
            weight = weights.get(metric.method, 0.1)
            weighted_sum += metric.drift_score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _generate_recommendations(self, drift_metrics: List[DriftMetric], overall_score: float) -> List[str]:
        """Generate recommendations based on drift detection results"""
        recommendations = []
        
        # Overall drift assessment
        if overall_score > self.config.critical_drift_threshold:
            recommendations.append("CRITICAL: Significant data drift detected. Immediate model retraining recommended.")
        elif overall_score > self.config.high_drift_threshold:
            recommendations.append("HIGH: Notable data drift detected. Consider model retraining within 1-2 weeks.")
        elif overall_score > self.config.medium_drift_threshold:
            recommendations.append("MEDIUM: Moderate drift detected. Monitor closely and consider retraining within a month.")
        elif overall_score > self.config.low_drift_threshold:
            recommendations.append("LOW: Minor drift detected. Continue monitoring.")
        
        # Feature-specific recommendations
        high_drift_features = [m for m in drift_metrics if m.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]]
        if high_drift_features:
            feature_names = [m.metric_name.split('_', 1)[-1] for m in high_drift_features]
            recommendations.append(f"Features with high drift: {', '.join(set(feature_names))}")
        
        # Method-specific insights
        ks_failures = [m for m in drift_metrics if m.method == DriftMethod.KS_TEST and m.is_drift]
        if ks_failures:
            recommendations.append("Distribution shape changes detected. Review data preprocessing steps.")
        
        psi_failures = [m for m in drift_metrics if m.method == DriftMethod.POPULATION_STABILITY and m.is_drift]
        if psi_failures:
            recommendations.append("Population stability issues detected. Check data source consistency.")
        
        if not recommendations:
            recommendations.append("No significant drift detected. Continue normal monitoring.")
        
        return recommendations
    
    def get_drift_trend(self, window_size: int = 10) -> Dict[str, Any]:
        """Analyze drift trend over recent history"""
        if len(self.drift_history) < 2:
            return {"trend": "insufficient_data", "message": "Need at least 2 drift reports for trend analysis"}
        
        recent_reports = self.drift_history[-window_size:]
        scores = [report.overall_drift_score for report in recent_reports]
        
        # Calculate trend
        if len(scores) >= 3:
            # Simple linear trend
            x = list(range(len(scores)))
            n = len(scores)
            sum_x = sum(x)
            sum_y = sum(scores)
            sum_xy = sum(xi * yi for xi, yi in zip(x, scores))
            sum_x2 = sum(xi * xi for xi in x)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if n * sum_x2 - sum_x * sum_x != 0 else 0
            
            if slope > 0.01:
                trend = "increasing"
            elif slope < -0.01:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "unknown"
        
        return {
            "trend": trend,
            "recent_scores": scores,
            "mean_score": statistics.mean(scores),
            "score_variance": statistics.variance(scores) if len(scores) > 1 else 0,
            "slope": slope if 'slope' in locals() else 0,
            "reports_analyzed": len(recent_reports)
        }


def create_drift_detector(
    config: Optional[Union[Dict[str, Any], DriftDetectorConfig]] = None
) -> DriftDetector:
    """Factory function to create drift detector"""
    if isinstance(config, dict):
        config = DriftDetectorConfig(**config)
    return DriftDetector(config)


def detect_drift_sample() -> DriftReport:
    """Generate sample drift detection for testing"""
    detector = create_drift_detector()
    
    # Create sample reference data
    reference_data = [
        {"feature1": 1.0 + i * 0.1, "feature2": "category_a" if i % 2 == 0 else "category_b"}
        for i in range(100)
    ]
    
    # Create sample current data with some drift
    current_data = [
        {"feature1": 1.5 + i * 0.1, "feature2": "category_a" if i % 3 == 0 else "category_b"}
        for i in range(100)
    ]
    
    detector.set_reference_data(reference_data)
    return detector.detect_drift(current_data)