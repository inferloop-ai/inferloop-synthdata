"""
Composition Analyzer for Differential Privacy
Advanced privacy accounting and composition analysis for differential privacy mechanisms
"""

import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math
from abc import ABC, abstractmethod
from datetime import datetime

from ...core import get_logger, PrivacyError


class AccountingMethod(Enum):
    """Privacy accounting methods"""
    BASIC_COMPOSITION = "basic_composition"
    ADVANCED_COMPOSITION = "advanced_composition"
    MOMENTS_ACCOUNTANT = "moments_accountant"
    RDP_ACCOUNTANT = "rdp_accountant"
    GDP_ACCOUNTANT = "gdp_accountant"  # Gaussian Differential Privacy
    PRIVACY_AMPLIFICATION = "privacy_amplification"


class PrivacyNotion(Enum):
    """Different notions of differential privacy"""
    PURE_DP = "pure_dp"  # (epsilon, 0)-DP
    APPROXIMATE_DP = "approximate_dp"  # (epsilon, delta)-DP
    CONCENTRATED_DP = "concentrated_dp"  # (rho)-CDP
    RENYI_DP = "renyi_dp"  # (alpha, epsilon)-RDP
    GAUSSIAN_DP = "gaussian_dp"  # mu-GDP


@dataclass
class PrivacyParameters:
    """Privacy parameters for different notions"""
    epsilon: Optional[float] = None
    delta: Optional[float] = None
    rho: Optional[float] = None  # For concentrated DP
    alpha: Optional[float] = None  # For Renyi DP
    mu: Optional[float] = None  # For Gaussian DP
    notion: PrivacyNotion = PrivacyNotion.APPROXIMATE_DP


@dataclass
class MechanismExecution:
    """Record of a mechanism execution"""
    mechanism_name: str
    privacy_params: PrivacyParameters
    query_sensitivity: float
    noise_scale: Optional[float] = None
    sampling_rate: Optional[float] = None  # For subsampling
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = None


@dataclass
class CompositionResult:
    """Result of privacy composition analysis"""
    composed_epsilon: float
    composed_delta: float
    privacy_loss: float
    accounting_method: AccountingMethod
    mechanism_count: int
    total_queries: int
    confidence_level: float
    is_valid: bool
    warnings: List[str]
    recommendations: List[str]


class PrivacyAccountant(ABC):
    """Abstract base class for privacy accountants"""
    
    @abstractmethod
    def compose(self, executions: List[MechanismExecution]) -> CompositionResult:
        """Compute privacy composition"""
        pass
    
    @abstractmethod
    def convert_to_dp(self, params: PrivacyParameters) -> Tuple[float, float]:
        """Convert to (epsilon, delta)-DP parameters"""
        pass


class BasicCompositionAccountant(PrivacyAccountant):
    """Basic composition: linear in epsilon"""
    
    def compose(self, executions: List[MechanismExecution]) -> CompositionResult:
        total_epsilon = sum(
            exec.privacy_params.epsilon or 0 for exec in executions
        )
        total_delta = sum(
            exec.privacy_params.delta or 0 for exec in executions
        )
        
        return CompositionResult(
            composed_epsilon=total_epsilon,
            composed_delta=total_delta,
            privacy_loss=total_epsilon,
            accounting_method=AccountingMethod.BASIC_COMPOSITION,
            mechanism_count=len(executions),
            total_queries=len(executions),
            confidence_level=1.0 - total_delta,
            is_valid=True,
            warnings=[],
            recommendations=[]
        )
    
    def convert_to_dp(self, params: PrivacyParameters) -> Tuple[float, float]:
        return (params.epsilon or 0, params.delta or 0)


class AdvancedCompositionAccountant(PrivacyAccountant):
    """Advanced composition with tighter bounds"""
    
    def __init__(self, delta_prime: float = 1e-6):
        self.delta_prime = delta_prime
    
    def compose(self, executions: List[MechanismExecution]) -> CompositionResult:
        # Group by (epsilon, delta) pairs
        mechanism_groups = {}
        for exec in executions:
            eps = exec.privacy_params.epsilon or 0
            delta = exec.privacy_params.delta or 0
            key = (eps, delta)
            mechanism_groups[key] = mechanism_groups.get(key, 0) + 1
        
        # Calculate advanced composition
        total_epsilon = 0
        total_delta = 0
        
        for (eps, delta), count in mechanism_groups.items():
            if eps == 0:
                continue
                
            # Advanced composition theorem
            # epsilon' = sqrt(2k log(1/delta')) epsilon + k epsilon (exp(epsilon) - 1)
            k = count
            sqrt_term = math.sqrt(2 * k * math.log(1 / self.delta_prime)) * eps
            linear_term = k * eps * (math.exp(eps) - 1) if eps < 1 else k * eps * eps
            
            composed_eps = sqrt_term + linear_term
            composed_delta = k * delta + self.delta_prime
            
            total_epsilon += composed_eps
            total_delta += composed_delta
        
        warnings = []
        if total_epsilon > 10:
            warnings.append("Very high epsilon value - consider reducing queries")
        
        return CompositionResult(
            composed_epsilon=total_epsilon,
            composed_delta=total_delta,
            privacy_loss=total_epsilon,
            accounting_method=AccountingMethod.ADVANCED_COMPOSITION,
            mechanism_count=len(mechanism_groups),
            total_queries=len(executions),
            confidence_level=1.0 - total_delta,
            is_valid=total_delta < 1.0,
            warnings=warnings,
            recommendations=[]
        )
    
    def convert_to_dp(self, params: PrivacyParameters) -> Tuple[float, float]:
        return (params.epsilon or 0, params.delta or 0)


class MomentsAccountant(PrivacyAccountant):
    """Moments accountant for tighter composition bounds"""
    
    def __init__(self, target_delta: float = 1e-5):
        self.target_delta = target_delta
    
    def compose(self, executions: List[MechanismExecution]) -> CompositionResult:
        # Simplified moments accountant implementation
        # In practice, this would use more sophisticated moment calculations
        
        total_moments = 0
        valid_executions = []
        
        for exec in executions:
            if exec.noise_scale and exec.query_sensitivity:
                # Calculate moment contribution
                sigma = exec.noise_scale
                sensitivity = exec.query_sensitivity
                
                # Simplified moment calculation for Gaussian noise
                if sigma > 0:
                    moment_contrib = sensitivity**2 / (2 * sigma**2)
                    total_moments += moment_contrib
                    valid_executions.append(exec)
        
        # Convert moments to (epsilon, delta)
        if total_moments > 0:
            # Simplified conversion (actual implementation would be more complex)
            epsilon = math.sqrt(2 * total_moments * math.log(1 / self.target_delta))
            delta = self.target_delta
        else:
            epsilon = sum(exec.privacy_params.epsilon or 0 for exec in executions)
            delta = sum(exec.privacy_params.delta or 0 for exec in executions)
        
        return CompositionResult(
            composed_epsilon=epsilon,
            composed_delta=delta,
            privacy_loss=epsilon,
            accounting_method=AccountingMethod.MOMENTS_ACCOUNTANT,
            mechanism_count=len(valid_executions),
            total_queries=len(executions),
            confidence_level=1.0 - delta,
            is_valid=True,
            warnings=[],
            recommendations=[]
        )
    
    def convert_to_dp(self, params: PrivacyParameters) -> Tuple[float, float]:
        return (params.epsilon or 0, params.delta or 0)


class RDPAccountant(PrivacyAccountant):
    """Renyi Differential Privacy accountant"""
    
    def __init__(self, alpha_max: float = 32):
        self.alpha_max = alpha_max
        self.alphas = [1 + x / 10.0 for x in range(1, int((alpha_max - 1) * 10) + 1)]
    
    def compose(self, executions: List[MechanismExecution]) -> CompositionResult:
        # RDP composition is additive
        rdp_budget = {}
        
        for exec in executions:
            alpha = exec.privacy_params.alpha
            epsilon_alpha = exec.privacy_params.epsilon
            
            if alpha and epsilon_alpha:
                rdp_budget[alpha] = rdp_budget.get(alpha, 0) + epsilon_alpha
        
        # Convert to (epsilon, delta)-DP using optimal alpha
        best_epsilon = float('inf')
        best_delta = 0
        
        for alpha, eps_alpha in rdp_budget.items():
            if alpha > 1:
                # Convert RDP to DP: epsilon = epsilon_alpha + log(1/delta)/(alpha-1)
                delta = 1e-5  # target delta
                epsilon = eps_alpha + math.log(1/delta) / (alpha - 1)
                
                if epsilon < best_epsilon:
                    best_epsilon = epsilon
                    best_delta = delta
        
        if best_epsilon == float('inf'):
            # Fallback to basic composition
            best_epsilon = sum(exec.privacy_params.epsilon or 0 for exec in executions)
            best_delta = sum(exec.privacy_params.delta or 0 for exec in executions)
        
        return CompositionResult(
            composed_epsilon=best_epsilon,
            composed_delta=best_delta,
            privacy_loss=best_epsilon,
            accounting_method=AccountingMethod.RDP_ACCOUNTANT,
            mechanism_count=len(rdp_budget),
            total_queries=len(executions),
            confidence_level=1.0 - best_delta,
            is_valid=True,
            warnings=[],
            recommendations=[]
        )
    
    def convert_to_dp(self, params: PrivacyParameters) -> Tuple[float, float]:
        if params.alpha and params.epsilon:
            # Convert (alpha, epsilon)-RDP to (epsilon', delta)-DP
            delta = 1e-5
            epsilon = params.epsilon + math.log(1/delta) / (params.alpha - 1)
            return (epsilon, delta)
        return (params.epsilon or 0, params.delta or 0)


class CompositionAnalyzer:
    """
    Advanced composition analyzer for differential privacy
    
    Provides sophisticated privacy accounting using multiple methods
    and privacy amplification techniques.
    """
    
    def __init__(self, default_method: AccountingMethod = AccountingMethod.ADVANCED_COMPOSITION):
        self.logger = get_logger(__name__)
        self.default_method = default_method
        
        # Initialize accountants
        self.accountants = {
            AccountingMethod.BASIC_COMPOSITION: BasicCompositionAccountant(),
            AccountingMethod.ADVANCED_COMPOSITION: AdvancedCompositionAccountant(),
            AccountingMethod.MOMENTS_ACCOUNTANT: MomentsAccountant(),
            AccountingMethod.RDP_ACCOUNTANT: RDPAccountant()
        }
        
        # Execution history
        self.execution_history: List[MechanismExecution] = []
        
        self.logger.info(f"Composition Analyzer initialized with {default_method.value}")
    
    def add_mechanism_execution(
        self,
        mechanism_name: str,
        privacy_params: PrivacyParameters,
        query_sensitivity: float,
        noise_scale: Optional[float] = None,
        sampling_rate: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a mechanism execution to the history"""
        
        execution = MechanismExecution(
            mechanism_name=mechanism_name,
            privacy_params=privacy_params,
            query_sensitivity=query_sensitivity,
            noise_scale=noise_scale,
            sampling_rate=sampling_rate,
            metadata=metadata or {}
        )
        
        self.execution_history.append(execution)
        
        self.logger.debug(
            f"Added execution: {mechanism_name}, epsilon={privacy_params.epsilon}, "
            f"delta={privacy_params.delta}"
        )
    
    def analyze_composition(
        self,
        method: Optional[AccountingMethod] = None,
        executions: Optional[List[MechanismExecution]] = None
    ) -> CompositionResult:
        """Analyze privacy composition using specified method"""
        
        try:
            method = method or self.default_method
            executions = executions or self.execution_history
            
            if not executions:
                return CompositionResult(
                    composed_epsilon=0,
                    composed_delta=0,
                    privacy_loss=0,
                    accounting_method=method,
                    mechanism_count=0,
                    total_queries=0,
                    confidence_level=1.0,
                    is_valid=True,
                    warnings=[],
                    recommendations=[]
                )
            
            accountant = self.accountants.get(method)
            if not accountant:
                raise PrivacyError(f"Unknown accounting method: {method}")
            
            result = accountant.compose(executions)
            
            # Add recommendations
            result.recommendations = self._generate_recommendations(result, executions)
            
            self.logger.info(
                f"Composition analysis complete: epsilon={result.composed_epsilon:.6f}, "
                f"delta={result.composed_delta:.9f}, method={method.value}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in composition analysis: {str(e)}")
            raise PrivacyError(f"Composition analysis failed: {str(e)}")
    
    def compare_accounting_methods(
        self,
        executions: Optional[List[MechanismExecution]] = None
    ) -> Dict[AccountingMethod, CompositionResult]:
        """Compare results from different accounting methods"""
        
        executions = executions or self.execution_history
        results = {}
        
        for method in AccountingMethod:
            if method in self.accountants:
                try:
                    result = self.analyze_composition(method, executions)
                    results[method] = result
                except Exception as e:
                    self.logger.warning(f"Failed to analyze with {method.value}: {e}")
        
        return results
    
    def estimate_privacy_amplification(
        self,
        base_epsilon: float,
        base_delta: float,
        sampling_rate: float,
        num_steps: int
    ) -> Tuple[float, float]:
        """Estimate privacy amplification from subsampling"""
        
        try:
            if sampling_rate <= 0 or sampling_rate > 1:
                raise PrivacyError("Sampling rate must be in (0, 1]")
            
            # Privacy amplification for subsampling (simplified)
            # Actual implementation would use more sophisticated bounds
            
            if sampling_rate == 1.0:
                # No amplification
                return (base_epsilon, base_delta)
            
            # Amplified epsilon (simplified bound)
            # epsilon' ~ q * epsilon for small epsilon and q
            if base_epsilon <= 1.0:
                amplified_epsilon = sampling_rate * base_epsilon
            else:
                # For larger epsilon, use different bound
                amplified_epsilon = sampling_rate * base_epsilon * 0.5
            
            # Amplified delta
            amplified_delta = sampling_rate * base_delta
            
            # Composition over multiple steps
            total_epsilon = num_steps * amplified_epsilon
            total_delta = num_steps * amplified_delta
            
            self.logger.debug(
                f"Privacy amplification: q={sampling_rate}, steps={num_steps}, "
                f"epsilon={base_epsilon:.4f}->{total_epsilon:.4f}, "
                f"delta={base_delta:.6f}->{total_delta:.6f}"
            )
            
            return (total_epsilon, total_delta)
            
        except Exception as e:
            self.logger.error(f"Error estimating privacy amplification: {str(e)}")
            return (base_epsilon * num_steps, base_delta * num_steps)
    
    def analyze_privacy_loss_over_time(
        self,
        time_windows: List[int],
        executions: Optional[List[MechanismExecution]] = None
    ) -> Dict[int, CompositionResult]:
        """Analyze how privacy loss accumulates over time"""
        
        executions = executions or self.execution_history
        results = {}
        
        for window in sorted(time_windows):
            if window <= len(executions):
                window_executions = executions[:window]
                result = self.analyze_composition(executions=window_executions)
                results[window] = result
        
        return results
    
    def optimize_epsilon_allocation(
        self,
        total_epsilon: float,
        query_sensitivities: List[float],
        target_accuracy: Optional[List[float]] = None
    ) -> List[float]:
        """Optimize epsilon allocation across multiple queries"""
        
        try:
            num_queries = len(query_sensitivities)
            
            if target_accuracy and len(target_accuracy) != num_queries:
                raise PrivacyError("Target accuracy length must match query count")
            
            # Simple allocation strategies
            if not target_accuracy:
                # Uniform allocation
                return [total_epsilon / num_queries] * num_queries
            
            # Allocation proportional to inverse sensitivity
            # Higher sensitivity queries need more epsilon for same accuracy
            weights = [1.0 / sens for sens in query_sensitivities]
            total_weight = sum(weights)
            
            allocations = [
                total_epsilon * weight / total_weight 
                for weight in weights
            ]
            
            self.logger.info(
                f"Optimized epsilon allocation: total={total_epsilon}, "
                f"queries={num_queries}, allocations={allocations}"
            )
            
            return allocations
            
        except Exception as e:
            self.logger.error(f"Error optimizing allocation: {str(e)}")
            # Fallback to uniform allocation
            return [total_epsilon / len(query_sensitivities)] * len(query_sensitivities)
    
    def validate_privacy_parameters(
        self,
        epsilon: float,
        delta: float,
        data_size: int,
        domain_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Validate privacy parameters for reasonableness"""
        
        validation = {
            "is_valid": True,
            "warnings": [],
            "recommendations": [],
            "risk_level": "LOW"
        }
        
        # Check epsilon range
        if epsilon <= 0:
            validation["is_valid"] = False
            validation["warnings"].append("Epsilon must be positive")
        elif epsilon > 10:
            validation["warnings"].append("Very high epsilon - weak privacy")
            validation["risk_level"] = "HIGH"
        elif epsilon > 1:
            validation["warnings"].append("High epsilon - consider reducing")
            validation["risk_level"] = "MEDIUM"
        
        # Check delta range
        if delta < 0:
            validation["is_valid"] = False
            validation["warnings"].append("Delta must be non-negative")
        elif delta > 1/data_size:
            validation["warnings"].append("Delta too large for dataset size")
            validation["risk_level"] = "HIGH"
        
        # Check data-dependent bounds
        if domain_size and epsilon * domain_size > data_size:
            validation["warnings"].append(
                "Privacy parameters may not provide meaningful protection"
            )
        
        # Generate recommendations
        if epsilon > 1:
            validation["recommendations"].append(
                "Consider using composition methods to reduce epsilon"
            )
        
        if delta > 1e-5:
            validation["recommendations"].append(
                "Consider reducing delta for stronger privacy"
            )
        
        return validation
    
    def _generate_recommendations(
        self,
        result: CompositionResult,
        executions: List[MechanismExecution]
    ) -> List[str]:
        """Generate recommendations based on composition analysis"""
        
        recommendations = []
        
        if result.composed_epsilon > 10:
            recommendations.append(
                "Very high privacy cost - consider reducing number of queries"
            )
        elif result.composed_epsilon > 1:
            recommendations.append(
                "High privacy cost - consider using advanced composition"
            )
        
        if result.composed_delta > 1e-3:
            recommendations.append(
                "High delta value - consider reducing for stronger privacy"
            )
        
        # Check for optimization opportunities
        mechanism_counts = {}
        for exec in executions:
            mechanism_counts[exec.mechanism_name] = mechanism_counts.get(exec.mechanism_name, 0) + 1
        
        if len(mechanism_counts) > 1:
            recommendations.append(
                "Multiple mechanisms used - consider batch processing for better composition"
            )
        
        # Check for subsampling opportunities
        has_sampling = any(exec.sampling_rate for exec in executions)
        if not has_sampling and len(executions) > 5:
            recommendations.append(
                "Consider subsampling for privacy amplification with many queries"
            )
        
        return recommendations
    
    def clear_history(self) -> None:
        """Clear execution history"""
        self.execution_history.clear()
        self.logger.info("Execution history cleared")
    
    def export_composition_analysis(self, output_path: str) -> str:
        """Export detailed composition analysis"""
        
        import json
        
        try:
            # Analyze with all methods
            analysis_results = self.compare_accounting_methods()
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "total_executions": len(self.execution_history),
                "composition_results": {},
                "executions": []
            }
            
            # Add composition results
            for method, result in analysis_results.items():
                report["composition_results"][method.value] = {
                    "epsilon": result.composed_epsilon,
                    "delta": result.composed_delta,
                    "privacy_loss": result.privacy_loss,
                    "is_valid": result.is_valid,
                    "warnings": result.warnings,
                    "recommendations": result.recommendations
                }
            
            # Add execution details
            for exec in self.execution_history:
                exec_dict = {
                    "mechanism": exec.mechanism_name,
                    "epsilon": exec.privacy_params.epsilon,
                    "delta": exec.privacy_params.delta,
                    "sensitivity": exec.query_sensitivity,
                    "noise_scale": exec.noise_scale,
                    "sampling_rate": exec.sampling_rate
                }
                report["executions"].append(exec_dict)
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Composition analysis exported to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error exporting analysis: {str(e)}")
            raise PrivacyError(f"Export failed: {str(e)}")
    
    def get_analyzer_info(self) -> Dict[str, Any]:
        """Get information about the composition analyzer"""
        
        return {
            "analyzer_name": "Composition Analyzer",
            "privacy_model": "Differential Privacy",
            "accounting_methods": [method.value for method in AccountingMethod],
            "privacy_notions": [notion.value for notion in PrivacyNotion],
            "default_method": self.default_method.value,
            "execution_history_size": len(self.execution_history),
            "available_accountants": list(self.accountants.keys())
        }