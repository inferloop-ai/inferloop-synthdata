"""
Exponential Mechanism for Differential Privacy
Implements the exponential mechanism for selecting outputs from a discrete set
"""

import numpy as np
from typing import Dict, List, Union, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import math
import random
from abc import ABC, abstractmethod

from ...core import get_logger, PrivacyError


class SelectionStrategy(Enum):
    """Strategies for selecting from exponential distribution"""
    SAMPLE = "sample"  # Sample according to probability distribution
    MAXIMUM = "maximum"  # Select maximum utility (deterministic)
    TOP_K = "top_k"  # Select from top-k highest utilities


class UtilityType(Enum):
    """Types of utility functions"""
    ACCURACY = "accuracy"  # How close to true answer
    DISTANCE = "distance"  # Distance-based utility
    FREQUENCY = "frequency"  # Frequency-based utility
    RANK = "rank"  # Rank-based utility
    CUSTOM = "custom"  # Custom utility function


@dataclass
class ExponentialResult:
    """Result from exponential mechanism"""
    selected_output: Any
    utility_score: float
    selection_probability: float
    total_outputs: int
    epsilon_used: float
    sensitivity_used: float
    strategy: SelectionStrategy


class UtilityFunction(ABC):
    """Abstract base class for utility functions"""
    
    @abstractmethod
    def compute_utility(
        self, 
        data: Union[List, np.ndarray], 
        output: Any, 
        **kwargs
    ) -> float:
        """Compute utility of an output given the data"""
        pass
    
    @abstractmethod
    def get_sensitivity(self) -> float:
        """Return the sensitivity of this utility function"""
        pass


class AccuracyUtility(UtilityFunction):
    """Utility function based on accuracy to true answer"""
    
    def __init__(self, true_answer: float, sensitivity: float = 1.0):
        self.true_answer = true_answer
        self.sensitivity_value = sensitivity
    
    def compute_utility(self, data: Union[List, np.ndarray], output: Any, **kwargs) -> float:
        """Higher utility for outputs closer to true answer"""
        if isinstance(output, (int, float)):
            return -abs(output - self.true_answer)  # Negative distance
        return 0.0
    
    def get_sensitivity(self) -> float:
        return self.sensitivity_value


class FrequencyUtility(UtilityFunction):
    """Utility function based on frequency in dataset"""
    
    def __init__(self, sensitivity: float = 1.0):
        self.sensitivity_value = sensitivity
    
    def compute_utility(self, data: Union[List, np.ndarray], output: Any, **kwargs) -> float:
        """Higher utility for more frequent values"""
        if hasattr(data, '__iter__'):
            return sum(1 for x in data if x == output)
        return 0.0
    
    def get_sensitivity(self) -> float:
        return self.sensitivity_value


class RankUtility(UtilityFunction):
    """Utility function based on rank ordering"""
    
    def __init__(self, ranking: List[Any], sensitivity: float = 1.0):
        self.ranking = ranking
        self.rank_map = {item: len(ranking) - i for i, item in enumerate(ranking)}
        self.sensitivity_value = sensitivity
    
    def compute_utility(self, data: Union[List, np.ndarray], output: Any, **kwargs) -> float:
        """Higher utility for higher-ranked items"""
        return self.rank_map.get(output, 0)
    
    def get_sensitivity(self) -> float:
        return self.sensitivity_value


class CustomUtility(UtilityFunction):
    """Custom utility function"""
    
    def __init__(self, utility_func: Callable, sensitivity: float):
        self.utility_func = utility_func
        self.sensitivity_value = sensitivity
    
    def compute_utility(self, data: Union[List, np.ndarray], output: Any, **kwargs) -> float:
        """Use custom utility function"""
        return self.utility_func(data, output, **kwargs)
    
    def get_sensitivity(self) -> float:
        return self.sensitivity_value


class ExponentialMechanism:
    """
    Exponential Mechanism for Differential Privacy
    
    Selects outputs from a discrete set with probability proportional to
    exp(epsilon * utility(output) / (2 * sensitivity))
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        self.logger = get_logger(__name__)
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # Cache for computed utilities
        self.utility_cache: Dict[str, Dict[Any, float]] = {}
        
        self.logger.info("Exponential Mechanism initialized")
    
    def select_output(
        self,
        data: Union[List, np.ndarray],
        output_domain: List[Any],
        utility_function: UtilityFunction,
        epsilon: float,
        strategy: SelectionStrategy = SelectionStrategy.SAMPLE,
        **kwargs
    ) -> ExponentialResult:
        """Select output using exponential mechanism"""
        
        try:
            # Validate parameters
            self._validate_parameters(epsilon, output_domain)
            
            # Compute utilities for all outputs
            utilities = self._compute_utilities(data, output_domain, utility_function, **kwargs)
            
            # Get sensitivity
            sensitivity = utility_function.get_sensitivity()
            
            # Compute probabilities
            probabilities = self._compute_probabilities(utilities, epsilon, sensitivity)
            
            # Select output based on strategy
            selected_idx, selection_prob = self._select_by_strategy(
                probabilities, strategy, **kwargs
            )
            
            selected_output = output_domain[selected_idx]
            utility_score = utilities[selected_idx]
            
            result = ExponentialResult(
                selected_output=selected_output,
                utility_score=utility_score,
                selection_probability=selection_prob,
                total_outputs=len(output_domain),
                epsilon_used=epsilon,
                sensitivity_used=sensitivity,
                strategy=strategy
            )
            
            self.logger.debug(
                f"Selected output: {selected_output}, utility: {utility_score:.4f}, "
                f"probability: {selection_prob:.4f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in exponential mechanism: {str(e)}")
            raise PrivacyError(f"Exponential mechanism failed: {str(e)}")
    
    def select_histogram_bin(
        self,
        data: Union[List[float], np.ndarray],
        bins: Union[int, List[float]],
        epsilon: float,
        range_bounds: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """Select histogram bin using exponential mechanism"""
        
        try:
            # Create histogram
            if isinstance(bins, int):
                if range_bounds is None:
                    range_bounds = (min(data), max(data))
                hist, bin_edges = np.histogram(data, bins=bins, range=range_bounds)
            else:
                hist, bin_edges = np.histogram(data, bins=bins)
            
            # Create output domain (bin centers)
            bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
            
            # Create frequency-based utility function
            utility_func = FrequencyUtility(sensitivity=1.0)
            
            # Select bin
            result = self.select_output(
                data=data,
                output_domain=bin_centers,
                utility_function=utility_func,
                epsilon=epsilon
            )
            
            # Find selected bin index
            selected_bin_idx = bin_centers.index(result.selected_output)
            
            return {
                "selected_bin": result.selected_output,
                "selected_bin_index": selected_bin_idx,
                "bin_count": hist[selected_bin_idx],
                "bin_edges": (bin_edges[selected_bin_idx], bin_edges[selected_bin_idx + 1]),
                "selection_probability": result.selection_probability,
                "privacy_cost": result.epsilon_used,
                "total_bins": len(bin_centers)
            }
            
        except Exception as e:
            self.logger.error(f"Error selecting histogram bin: {str(e)}")
            raise PrivacyError(f"Histogram bin selection failed: {str(e)}")
    
    def private_top_k(
        self,
        data: Union[List, np.ndarray],
        candidates: List[Any],
        k: int,
        utility_function: UtilityFunction,
        epsilon: float
    ) -> List[ExponentialResult]:
        """Select top-k elements using exponential mechanism"""
        
        try:
            if k <= 0 or k > len(candidates):
                raise PrivacyError(f"Invalid k value: {k}")
            
            # Budget allocation for k selections
            epsilon_per_selection = epsilon / k
            
            results = []
            remaining_candidates = candidates.copy()
            
            for i in range(k):
                # Select one output
                result = self.select_output(
                    data=data,
                    output_domain=remaining_candidates,
                    utility_function=utility_function,
                    epsilon=epsilon_per_selection,
                    strategy=SelectionStrategy.SAMPLE
                )
                
                results.append(result)
                
                # Remove selected output from remaining candidates
                remaining_candidates.remove(result.selected_output)
            
            self.logger.info(f"Selected top-{k} outputs with epsilon={epsilon}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in private top-k selection: {str(e)}")
            raise PrivacyError(f"Private top-k selection failed: {str(e)}")
    
    def private_median(
        self,
        data: Union[List[float], np.ndarray],
        epsilon: float,
        candidate_range: Optional[Tuple[float, float]] = None,
        num_candidates: int = 100
    ) -> Dict[str, Any]:
        """Compute private median using exponential mechanism"""
        
        try:
            data_array = np.array(data)
            
            # Define candidate range
            if candidate_range is None:
                candidate_range = (data_array.min(), data_array.max())
            
            # Create candidate medians
            candidates = np.linspace(
                candidate_range[0], 
                candidate_range[1], 
                num_candidates
            ).tolist()
            
            # Define utility function for median
            def median_utility(data_vals, candidate_median, **kwargs):
                # Utility = negative number of elements that would change median
                data_sorted = sorted(data_vals)
                n = len(data_sorted)
                true_median = data_sorted[n//2] if n % 2 == 1 else (data_sorted[n//2-1] + data_sorted[n//2]) / 2
                return -abs(candidate_median - true_median)
            
            utility_func = CustomUtility(median_utility, sensitivity=1.0)
            
            # Select private median
            result = self.select_output(
                data=data,
                output_domain=candidates,
                utility_function=utility_func,
                epsilon=epsilon
            )
            
            # Calculate true median for comparison
            true_median = float(np.median(data_array))
            
            return {
                "private_median": result.selected_output,
                "true_median": true_median,
                "error": abs(result.selected_output - true_median),
                "selection_probability": result.selection_probability,
                "utility_score": result.utility_score,
                "privacy_cost": result.epsilon_used,
                "candidates_considered": num_candidates
            }
            
        except Exception as e:
            self.logger.error(f"Error computing private median: {str(e)}")
            raise PrivacyError(f"Private median computation failed: {str(e)}")
    
    def private_mode(
        self,
        data: Union[List, np.ndarray],
        epsilon: float,
        candidates: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """Compute private mode using exponential mechanism"""
        
        try:
            # Use unique values as candidates if not provided
            if candidates is None:
                candidates = list(set(data))
            
            # Use frequency-based utility
            utility_func = FrequencyUtility(sensitivity=1.0)
            
            # Select private mode
            result = self.select_output(
                data=data,
                output_domain=candidates,
                utility_function=utility_func,
                epsilon=epsilon
            )
            
            # Calculate true mode for comparison
            from collections import Counter
            counter = Counter(data)
            true_mode = counter.most_common(1)[0][0]
            true_frequency = counter.most_common(1)[0][1]
            
            private_frequency = sum(1 for x in data if x == result.selected_output)
            
            return {
                "private_mode": result.selected_output,
                "private_frequency": private_frequency,
                "true_mode": true_mode,
                "true_frequency": true_frequency,
                "selection_probability": result.selection_probability,
                "utility_score": result.utility_score,
                "privacy_cost": result.epsilon_used,
                "candidates_considered": len(candidates)
            }
            
        except Exception as e:
            self.logger.error(f"Error computing private mode: {str(e)}")
            raise PrivacyError(f"Private mode computation failed: {str(e)}")
    
    def _validate_parameters(self, epsilon: float, output_domain: List[Any]) -> None:
        """Validate mechanism parameters"""
        
        if epsilon <= 0:
            raise PrivacyError("Epsilon must be positive")
        
        if not output_domain:
            raise PrivacyError("Output domain cannot be empty")
        
        if len(set(output_domain)) != len(output_domain):
            self.logger.warning("Output domain contains duplicates")
    
    def _compute_utilities(
        self,
        data: Union[List, np.ndarray],
        output_domain: List[Any],
        utility_function: UtilityFunction,
        **kwargs
    ) -> List[float]:
        """Compute utilities for all outputs in domain"""
        
        # Create cache key
        data_hash = hash(str(data)) if len(data) < 1000 else hash(str(data[:100]))
        cache_key = f"{data_hash}_{hash(str(output_domain))}"
        
        if cache_key in self.utility_cache:
            utilities_dict = self.utility_cache[cache_key]
            return [utilities_dict[output] for output in output_domain]
        
        # Compute utilities
        utilities = []
        utilities_dict = {}
        
        for output in output_domain:
            utility = utility_function.compute_utility(data, output, **kwargs)
            utilities.append(utility)
            utilities_dict[output] = utility
        
        # Cache results
        self.utility_cache[cache_key] = utilities_dict
        
        return utilities
    
    def _compute_probabilities(
        self,
        utilities: List[float],
        epsilon: float,
        sensitivity: float
    ) -> np.ndarray:
        """Compute selection probabilities using exponential mechanism"""
        
        # Exponential weights: exp(epsilon * utility / (2 * sensitivity))
        scaled_utilities = np.array(utilities) * epsilon / (2 * sensitivity)
        
        # Prevent overflow by subtracting max
        max_utility = np.max(scaled_utilities)
        exp_utilities = np.exp(scaled_utilities - max_utility)
        
        # Normalize to get probabilities
        probabilities = exp_utilities / np.sum(exp_utilities)
        
        return probabilities
    
    def _select_by_strategy(
        self,
        probabilities: np.ndarray,
        strategy: SelectionStrategy,
        **kwargs
    ) -> Tuple[int, float]:
        """Select output index based on strategy"""
        
        if strategy == SelectionStrategy.SAMPLE:
            # Sample according to probability distribution
            selected_idx = np.random.choice(len(probabilities), p=probabilities)
            return selected_idx, probabilities[selected_idx]
        
        elif strategy == SelectionStrategy.MAXIMUM:
            # Select maximum probability (deterministic)
            selected_idx = np.argmax(probabilities)
            return selected_idx, probabilities[selected_idx]
        
        elif strategy == SelectionStrategy.TOP_K:
            # Select from top-k with uniform probability
            k = kwargs.get('k', 3)
            k = min(k, len(probabilities))
            
            top_k_indices = np.argsort(probabilities)[-k:]
            selected_idx = np.random.choice(top_k_indices)
            return selected_idx, probabilities[selected_idx]
        
        else:
            raise PrivacyError(f"Unknown selection strategy: {strategy}")
    
    def analyze_privacy_utility_tradeoff(
        self,
        data: Union[List, np.ndarray],
        output_domain: List[Any],
        utility_function: UtilityFunction,
        epsilon_values: List[float]
    ) -> Dict[str, List[float]]:
        """Analyze privacy-utility tradeoff for different epsilon values"""
        
        try:
            results = {
                "epsilon_values": epsilon_values,
                "max_utilities": [],
                "entropy_values": [],
                "expected_utilities": []
            }
            
            # Compute utilities once
            utilities = self._compute_utilities(data, output_domain, utility_function)
            sensitivity = utility_function.get_sensitivity()
            
            for epsilon in epsilon_values:
                # Compute probabilities
                probabilities = self._compute_probabilities(utilities, epsilon, sensitivity)
                
                # Max utility (best possible outcome)
                max_utility = np.max(utilities)
                results["max_utilities"].append(max_utility)
                
                # Entropy (measure of uncertainty)
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
                results["entropy_values"].append(entropy)
                
                # Expected utility
                expected_utility = np.sum(probabilities * utilities)
                results["expected_utilities"].append(expected_utility)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing tradeoff: {str(e)}")
            raise PrivacyError(f"Tradeoff analysis failed: {str(e)}")
    
    def estimate_selection_probability(
        self,
        data: Union[List, np.ndarray],
        output_domain: List[Any],
        target_output: Any,
        utility_function: UtilityFunction,
        epsilon: float
    ) -> float:
        """Estimate probability of selecting a specific output"""
        
        try:
            if target_output not in output_domain:
                return 0.0
            
            # Compute utilities and probabilities
            utilities = self._compute_utilities(data, output_domain, utility_function)
            sensitivity = utility_function.get_sensitivity()
            probabilities = self._compute_probabilities(utilities, epsilon, sensitivity)
            
            # Find target output index
            target_idx = output_domain.index(target_output)
            
            return float(probabilities[target_idx])
            
        except Exception as e:
            self.logger.error(f"Error estimating selection probability: {str(e)}")
            return 0.0
    
    def get_mechanism_info(self) -> Dict[str, Any]:
        """Get information about the exponential mechanism"""
        
        return {
            "mechanism_name": "Exponential Mechanism",
            "privacy_model": "Differential Privacy",
            "output_type": "Discrete",
            "selection_strategies": [s.value for s in SelectionStrategy],
            "utility_types": [ut.value for ut in UtilityType],
            "epsilon_range": "epsilon > 0",
            "composition": "Linear in epsilon",
            "cache_entries": len(self.utility_cache)
        }