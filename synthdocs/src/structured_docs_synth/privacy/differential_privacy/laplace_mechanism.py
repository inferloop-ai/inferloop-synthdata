"""
Laplace Mechanism for Differential Privacy
Implements the Laplace mechanism for adding calibrated noise to numerical queries
"""

import numpy as np
from typing import Dict, List, Union, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import math
import random

from ...core import get_logger, PrivacyError


class SensitivityType(Enum):
    """Types of sensitivity measures"""
    L1_SENSITIVITY = "l1_sensitivity"  # Sum of absolute differences
    L2_SENSITIVITY = "l2_sensitivity"  # Euclidean distance
    GLOBAL_SENSITIVITY = "global_sensitivity"  # Maximum over all databases
    LOCAL_SENSITIVITY = "local_sensitivity"  # Maximum for specific database


class QueryType(Enum):
    """Types of queries supported"""
    COUNT = "count"
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    VARIANCE = "variance"
    HISTOGRAM = "histogram"
    RANGE_QUERY = "range_query"
    CUSTOM = "custom"


@dataclass
class NoiseParameters:
    """Parameters for noise generation"""
    epsilon: float  # Privacy parameter
    delta: float = 0.0  # For (epsilon, delta)-differential privacy
    sensitivity: float = 1.0  # Query sensitivity
    sensitivity_type: SensitivityType = SensitivityType.GLOBAL_SENSITIVITY
    clipping_bound: Optional[float] = None  # For bounded sensitivity


@dataclass
class LaplaceNoise:
    """Laplace noise sample"""
    noise_value: float
    scale_parameter: float
    epsilon_used: float
    sensitivity_used: float
    query_type: QueryType


@dataclass
class PrivacyGuarantee:
    """Privacy guarantee provided by the mechanism"""
    epsilon: float
    delta: float
    mechanism: str
    sensitivity: float
    noise_scale: float
    confidence_level: float = 0.95


class LaplaceMechanism:
    """
    Laplace Mechanism for Differential Privacy
    
    Adds Laplace noise with scale Delta_f/epsilon where:
    - Delta_f is the sensitivity of query f
    - epsilon is the privacy parameter
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        self.logger = get_logger(__name__)
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # Query sensitivity cache
        self.sensitivity_cache: Dict[str, float] = {}
        
        # Supported query functions
        self.query_functions = {
            QueryType.COUNT: self._count_query,
            QueryType.SUM: self._sum_query,
            QueryType.MEAN: self._mean_query,
            QueryType.MEDIAN: self._median_query,
            QueryType.VARIANCE: self._variance_query,
            QueryType.HISTOGRAM: self._histogram_query
        }
        
        self.logger.info("Laplace Mechanism initialized")
    
    def add_noise(
        self,
        true_value: Union[float, List[float]],
        noise_params: NoiseParameters
    ) -> Union[float, List[float]]:
        """Add Laplace noise to a true value or list of values"""
        
        try:
            # Validate parameters
            self._validate_noise_parameters(noise_params)
            
            # Calculate noise scale
            scale = noise_params.sensitivity / noise_params.epsilon
            
            if isinstance(true_value, (int, float)):
                # Single value
                noise = np.random.laplace(0, scale)
                noisy_value = true_value + noise
                
                self.logger.debug(
                    f"Added Laplace noise: true={true_value}, "
                    f"noise={noise:.4f}, noisy={noisy_value:.4f}, scale={scale:.4f}"
                )
                
                return noisy_value
            
            elif isinstance(true_value, (list, np.ndarray)):
                # Multiple values
                true_array = np.array(true_value)
                noise_array = np.random.laplace(0, scale, size=true_array.shape)
                noisy_array = true_array + noise_array
                
                self.logger.debug(
                    f"Added Laplace noise to {len(true_value)} values, scale={scale:.4f}"
                )
                
                return noisy_array.tolist()
            
            else:
                raise PrivacyError(f"Unsupported value type: {type(true_value)}")
                
        except Exception as e:
            self.logger.error(f"Error adding Laplace noise: {str(e)}")
            raise PrivacyError(f"Laplace noise addition failed: {str(e)}")
    
    def execute_query(
        self,
        data: Union[List[float], np.ndarray],
        query_type: QueryType,
        noise_params: NoiseParameters,
        query_args: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a differentially private query"""
        
        try:
            # Get query function
            if query_type not in self.query_functions:
                raise PrivacyError(f"Unsupported query type: {query_type}")
            
            query_func = self.query_functions[query_type]
            
            # Calculate true result
            true_result = query_func(data, query_args or {})
            
            # Calculate sensitivity if not provided
            if noise_params.sensitivity <= 0:
                noise_params.sensitivity = self._calculate_sensitivity(
                    data, query_type, query_args or {}
                )
            
            # Add noise
            noisy_result = self.add_noise(true_result, noise_params)
            
            # Generate noise sample info
            scale = noise_params.sensitivity / noise_params.epsilon
            noise_sample = LaplaceNoise(
                noise_value=noisy_result - true_result if isinstance(noisy_result, (int, float)) else 0,
                scale_parameter=scale,
                epsilon_used=noise_params.epsilon,
                sensitivity_used=noise_params.sensitivity,
                query_type=query_type
            )
            
            # Calculate privacy guarantee
            privacy_guarantee = PrivacyGuarantee(
                epsilon=noise_params.epsilon,
                delta=noise_params.delta,
                mechanism="Laplace",
                sensitivity=noise_params.sensitivity,
                noise_scale=scale
            )
            
            return {
                "true_result": true_result,
                "noisy_result": noisy_result,
                "noise_added": noise_sample,
                "privacy_guarantee": privacy_guarantee,
                "query_type": query_type.value,
                "data_size": len(data)
            }
            
        except Exception as e:
            self.logger.error(f"Error executing private query: {str(e)}")
            raise PrivacyError(f"Private query execution failed: {str(e)}")
    
    def calibrate_noise(
        self,
        target_accuracy: float,
        confidence_level: float,
        sensitivity: float,
        data_size: int
    ) -> NoiseParameters:
        """Calibrate noise parameters to achieve target accuracy"""
        
        try:
            # For Laplace mechanism, noise follows Laplace(0, Delta_f/epsilon)
            # P(|noise| <= t) = 1 - exp(-epsilon*t/Delta_f)
            # For confidence level alpha, we want P(|noise| <= target_accuracy) >= alpha
            
            # Solve: 1 - exp(-epsilon*target_accuracy/sensitivity) = confidence_level
            # epsilon = -sensitivity * ln(1 - confidence_level) / target_accuracy
            
            if confidence_level >= 1.0 or confidence_level <= 0.0:
                raise PrivacyError("Confidence level must be between 0 and 1")
            
            epsilon = -sensitivity * math.log(1 - confidence_level) / target_accuracy
            
            # Ensure epsilon is reasonable
            if epsilon <= 0:
                raise PrivacyError("Calculated epsilon is non-positive")
            
            if epsilon > 10:  # Very high privacy cost
                self.logger.warning(
                    f"High epsilon value ({epsilon:.4f}) required for target accuracy"
                )
            
            noise_params = NoiseParameters(
                epsilon=epsilon,
                sensitivity=sensitivity
            )
            
            self.logger.info(
                f"Calibrated noise: epsilon={epsilon:.4f}, accuracy={target_accuracy}, "
                f"confidence={confidence_level}"
            )
            
            return noise_params
            
        except Exception as e:
            self.logger.error(f"Error calibrating noise: {str(e)}")
            raise PrivacyError(f"Noise calibration failed: {str(e)}")
    
    def estimate_accuracy(
        self,
        noise_params: NoiseParameters,
        confidence_level: float = 0.95
    ) -> float:
        """Estimate accuracy for given noise parameters"""
        
        try:
            # For Laplace(0, b) where b = sensitivity/epsilon
            # P(|X| d t) = 1 - exp(-t/b)
            # Solve for t: t = -b * ln(1 - confidence_level)
            
            scale = noise_params.sensitivity / noise_params.epsilon
            accuracy = -scale * math.log(1 - confidence_level)
            
            return accuracy
            
        except Exception as e:
            self.logger.error(f"Error estimating accuracy: {str(e)}")
            raise PrivacyError(f"Accuracy estimation failed: {str(e)}")
    
    def _validate_noise_parameters(self, params: NoiseParameters) -> None:
        """Validate noise parameters"""
        
        if params.epsilon <= 0:
            raise PrivacyError("Epsilon must be positive")
        
        if params.delta < 0:
            raise PrivacyError("Delta must be non-negative")
        
        if params.sensitivity <= 0:
            raise PrivacyError("Sensitivity must be positive")
        
        if params.epsilon > 10:
            self.logger.warning(f"High epsilon value: {params.epsilon}")
    
    def _calculate_sensitivity(
        self,
        data: Union[List[float], np.ndarray],
        query_type: QueryType,
        query_args: Dict[str, Any]
    ) -> float:
        """Calculate query sensitivity"""
        
        # Cache key
        cache_key = f"{query_type.value}_{len(data)}_{hash(str(sorted(query_args.items())))}"
        
        if cache_key in self.sensitivity_cache:
            return self.sensitivity_cache[cache_key]
        
        try:
            if query_type == QueryType.COUNT:
                # Count queries have sensitivity 1
                sensitivity = 1.0
            
            elif query_type == QueryType.SUM:
                # Sum sensitivity depends on value bounds
                clipping_bound = query_args.get("clipping_bound", max(abs(min(data)), abs(max(data))))
                sensitivity = clipping_bound
            
            elif query_type == QueryType.MEAN:
                # Mean sensitivity = max_value / n
                clipping_bound = query_args.get("clipping_bound", max(abs(min(data)), abs(max(data))))
                sensitivity = clipping_bound / len(data)
            
            elif query_type == QueryType.MEDIAN:
                # Median sensitivity for sorted data
                data_sorted = sorted(data)
                n = len(data_sorted)
                if n % 2 == 1:
                    # Odd number of elements
                    sensitivity = abs(data_sorted[n//2] - data_sorted[n//2 - 1]) / 2
                else:
                    # Even number of elements
                    sensitivity = abs(data_sorted[n//2] - data_sorted[n//2 - 1]) / 2
                sensitivity = max(sensitivity, 1e-6)  # Minimum sensitivity
            
            elif query_type == QueryType.VARIANCE:
                # Variance sensitivity (simplified)
                clipping_bound = query_args.get("clipping_bound", max(abs(min(data)), abs(max(data))))
                sensitivity = 2 * clipping_bound**2 / len(data)
            
            elif query_type == QueryType.HISTOGRAM:
                # Histogram sensitivity = 1 (count-based)
                sensitivity = 1.0
            
            else:
                # Default sensitivity
                sensitivity = 1.0
                self.logger.warning(f"Using default sensitivity for {query_type}")
            
            # Cache the result
            self.sensitivity_cache[cache_key] = sensitivity
            
            self.logger.debug(f"Calculated sensitivity: {sensitivity} for {query_type}")
            return sensitivity
            
        except Exception as e:
            self.logger.error(f"Error calculating sensitivity: {str(e)}")
            return 1.0  # Default fallback
    
    def _count_query(self, data: Union[List[float], np.ndarray], args: Dict[str, Any]) -> float:
        """Count query implementation"""
        predicate = args.get("predicate", lambda x: True)
        return sum(1 for x in data if predicate(x))
    
    def _sum_query(self, data: Union[List[float], np.ndarray], args: Dict[str, Any]) -> float:
        """Sum query implementation"""
        clipping_bound = args.get("clipping_bound")
        data_array = np.array(data)
        
        if clipping_bound:
            data_array = np.clip(data_array, -clipping_bound, clipping_bound)
        
        return float(np.sum(data_array))
    
    def _mean_query(self, data: Union[List[float], np.ndarray], args: Dict[str, Any]) -> float:
        """Mean query implementation"""
        clipping_bound = args.get("clipping_bound")
        data_array = np.array(data)
        
        if clipping_bound:
            data_array = np.clip(data_array, -clipping_bound, clipping_bound)
        
        return float(np.mean(data_array))
    
    def _median_query(self, data: Union[List[float], np.ndarray], args: Dict[str, Any]) -> float:
        """Median query implementation"""
        return float(np.median(data))
    
    def _variance_query(self, data: Union[List[float], np.ndarray], args: Dict[str, Any]) -> float:
        """Variance query implementation"""
        clipping_bound = args.get("clipping_bound")
        data_array = np.array(data)
        
        if clipping_bound:
            data_array = np.clip(data_array, -clipping_bound, clipping_bound)
        
        return float(np.var(data_array))
    
    def _histogram_query(self, data: Union[List[float], np.ndarray], args: Dict[str, Any]) -> List[float]:
        """Histogram query implementation"""
        bins = args.get("bins", 10)
        range_bounds = args.get("range", (min(data), max(data)))
        
        hist, _ = np.histogram(data, bins=bins, range=range_bounds)
        return hist.tolist()
    
    def batch_query(
        self,
        data: Union[List[float], np.ndarray],
        queries: List[Dict[str, Any]],
        total_epsilon: float
    ) -> List[Dict[str, Any]]:
        """Execute multiple queries with epsilon budget allocation"""
        
        try:
            num_queries = len(queries)
            if num_queries == 0:
                return []
            
            # Simple uniform budget allocation
            epsilon_per_query = total_epsilon / num_queries
            
            results = []
            
            for i, query_spec in enumerate(queries):
                query_type = QueryType(query_spec["type"])
                query_args = query_spec.get("args", {})
                sensitivity = query_spec.get("sensitivity", 1.0)
                
                # Create noise parameters
                noise_params = NoiseParameters(
                    epsilon=epsilon_per_query,
                    sensitivity=sensitivity
                )
                
                # Execute query
                result = self.execute_query(data, query_type, noise_params, query_args)
                result["query_index"] = i
                result["epsilon_allocated"] = epsilon_per_query
                
                results.append(result)
            
            self.logger.info(f"Executed {num_queries} queries with epsilon={total_epsilon}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch query: {str(e)}")
            raise PrivacyError(f"Batch query failed: {str(e)}")
    
    def get_privacy_cost(self, noise_params: NoiseParameters) -> Dict[str, float]:
        """Calculate privacy cost metrics"""
        
        return {
            "epsilon": noise_params.epsilon,
            "delta": noise_params.delta,
            "sensitivity": noise_params.sensitivity,
            "noise_scale": noise_params.sensitivity / noise_params.epsilon,
            "signal_to_noise_ratio": 1.0 / (noise_params.sensitivity / noise_params.epsilon)
        }
    
    def optimize_accuracy(
        self,
        data: Union[List[float], np.ndarray],
        query_type: QueryType,
        epsilon_budget: float,
        target_accuracy: Optional[float] = None,
        query_args: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Optimize query accuracy within epsilon budget"""
        
        try:
            query_args = query_args or {}
            
            # Calculate optimal sensitivity
            sensitivity = self._calculate_sensitivity(data, query_type, query_args)
            
            # Use clipping if beneficial
            if query_type in [QueryType.SUM, QueryType.MEAN]:
                data_array = np.array(data)
                natural_bound = max(abs(data_array.min()), abs(data_array.max()))
                
                # Try different clipping bounds
                best_accuracy = float('inf')
                best_params = None
                best_result = None
                
                clipping_bounds = [natural_bound * factor for factor in [1.0, 0.8, 0.6, 0.4, 0.2]]
                
                for bound in clipping_bounds:
                    test_args = query_args.copy()
                    test_args["clipping_bound"] = bound
                    
                    test_sensitivity = self._calculate_sensitivity(data, query_type, test_args)
                    test_params = NoiseParameters(epsilon=epsilon_budget, sensitivity=test_sensitivity)
                    
                    expected_accuracy = self.estimate_accuracy(test_params)
                    
                    if expected_accuracy < best_accuracy:
                        best_accuracy = expected_accuracy
                        best_params = test_params
                        best_result = self.execute_query(data, query_type, test_params, test_args)
                
                if best_result:
                    best_result["optimization"] = {
                        "method": "clipping_optimization",
                        "expected_accuracy": best_accuracy,
                        "clipping_bound": best_result.get("clipping_bound")
                    }
                    return best_result
            
            # Standard execution without optimization
            noise_params = NoiseParameters(epsilon=epsilon_budget, sensitivity=sensitivity)
            result = self.execute_query(data, query_type, noise_params, query_args)
            
            result["optimization"] = {
                "method": "standard",
                "expected_accuracy": self.estimate_accuracy(noise_params)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error optimizing accuracy: {str(e)}")
            raise PrivacyError(f"Accuracy optimization failed: {str(e)}")
    
    def get_mechanism_info(self) -> Dict[str, Any]:
        """Get information about the Laplace mechanism"""
        
        return {
            "mechanism_name": "Laplace Mechanism",
            "privacy_model": "Differential Privacy",
            "supported_queries": [qt.value for qt in QueryType],
            "sensitivity_types": [st.value for st in SensitivityType],
            "noise_distribution": "Laplace",
            "epsilon_range": "epsilon > 0",
            "delta_support": "delta >= 0 (pure DP when delta = 0)",
            "composition": "Linear in epsilon",
            "query_cache_size": len(self.sensitivity_cache)
        }