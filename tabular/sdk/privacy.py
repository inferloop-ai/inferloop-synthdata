"""
Privacy metrics and differential privacy validation
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import warnings


@dataclass
class PrivacyMetrics:
    """Container for privacy metrics results"""
    epsilon: float  # Differential privacy parameter
    delta: float  # Differential privacy parameter
    k_anonymity: int  # Minimum group size
    l_diversity: float  # Diversity measure
    t_closeness: float  # Distribution closeness
    membership_disclosure_risk: float  # Risk of identifying individuals
    attribute_disclosure_risk: float  # Risk of learning attributes
    privacy_score: float  # Overall privacy score (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'epsilon': self.epsilon,
            'delta': self.delta,
            'k_anonymity': self.k_anonymity,
            'l_diversity': self.l_diversity,
            't_closeness': self.t_closeness,
            'membership_disclosure_risk': self.membership_disclosure_risk,
            'attribute_disclosure_risk': self.attribute_disclosure_risk,
            'privacy_score': self.privacy_score
        }


class DifferentialPrivacyValidator:
    """Validate differential privacy guarantees"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Initialize differential privacy validator
        
        Args:
            epsilon: Privacy budget (lower = more private)
            delta: Probability of privacy breach
        """
        self.epsilon = epsilon
        self.delta = delta
    
    def estimate_epsilon(self,
                        real_data: pd.DataFrame,
                        synthetic_data: pd.DataFrame,
                        num_trials: int = 100) -> float:
        """
        Estimate epsilon from empirical analysis
        
        Args:
            real_data: Original dataset
            synthetic_data: Synthetic dataset
            num_trials: Number of trials for estimation
            
        Returns:
            Estimated epsilon value
        """
        epsilons = []
        
        for col in real_data.select_dtypes(include=[np.number]).columns:
            if col not in synthetic_data.columns:
                continue
            
            # Add noise to create neighboring datasets
            for _ in range(num_trials):
                # Create neighboring dataset by changing one value
                neighbor_data = real_data.copy()
                idx = np.random.randint(0, len(neighbor_data))
                
                # Small perturbation
                std = real_data[col].std()
                if std > 0:
                    neighbor_data.loc[idx, col] += np.random.normal(0, std * 0.1)
                
                # Compute statistics on both datasets
                stat_real = real_data[col].mean()
                stat_neighbor = neighbor_data[col].mean()
                stat_synthetic = synthetic_data[col].mean()
                
                # Estimate epsilon using differential privacy definition
                if stat_neighbor != stat_real:
                    ratio = abs(stat_synthetic - stat_real) / abs(stat_neighbor - stat_real)
                    if ratio > 0:
                        epsilon_est = np.log(ratio)
                        epsilons.append(abs(epsilon_est))
        
        return np.median(epsilons) if epsilons else float('inf')
    
    def check_differential_privacy(self,
                                 real_data: pd.DataFrame,
                                 synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check if synthetic data satisfies differential privacy
        
        Returns:
            Dictionary with privacy analysis results
        """
        estimated_epsilon = self.estimate_epsilon(real_data, synthetic_data)
        
        # Check if estimated epsilon is within bounds
        satisfies_dp = estimated_epsilon <= self.epsilon
        
        # Calculate privacy loss
        privacy_loss = min(estimated_epsilon / self.epsilon, 1.0) if self.epsilon > 0 else 1.0
        
        return {
            'satisfies_dp': satisfies_dp,
            'estimated_epsilon': estimated_epsilon,
            'target_epsilon': self.epsilon,
            'privacy_loss': privacy_loss,
            'delta': self.delta
        }
    
    def laplace_mechanism_check(self,
                              real_data: pd.DataFrame,
                              synthetic_data: pd.DataFrame,
                              sensitivity: float = 1.0) -> Dict[str, Any]:
        """
        Check if noise follows Laplace mechanism
        
        Args:
            sensitivity: Global sensitivity of the query
            
        Returns:
            Analysis of noise distribution
        """
        results = {}
        
        for col in real_data.select_dtypes(include=[np.number]).columns:
            if col not in synthetic_data.columns:
                continue
            
            # Compute differences
            real_mean = real_data[col].mean()
            synthetic_mean = synthetic_data[col].mean()
            difference = synthetic_mean - real_mean
            
            # Expected scale for Laplace noise
            expected_scale = sensitivity / self.epsilon
            
            # Check if difference follows Laplace distribution
            # This is a simplified check
            if abs(difference) <= 3 * expected_scale:  # 99.7% confidence
                follows_laplace = True
            else:
                follows_laplace = False
            
            results[col] = {
                'difference': difference,
                'expected_scale': expected_scale,
                'follows_laplace': follows_laplace
            }
        
        return results


class KAnonymityValidator:
    """Validate k-anonymity property"""
    
    def __init__(self, quasi_identifiers: Optional[List[str]] = None):
        """
        Initialize k-anonymity validator
        
        Args:
            quasi_identifiers: Columns that could identify individuals
        """
        self.quasi_identifiers = quasi_identifiers
    
    def compute_k_anonymity(self, data: pd.DataFrame) -> int:
        """
        Compute k-anonymity value
        
        Returns:
            Minimum group size (k value)
        """
        if self.quasi_identifiers is None:
            # Use all columns as quasi-identifiers
            qi_cols = data.columns.tolist()
        else:
            qi_cols = [col for col in self.quasi_identifiers if col in data.columns]
        
        if not qi_cols:
            return len(data)  # No quasi-identifiers
        
        # Group by quasi-identifiers
        grouped = data.groupby(qi_cols).size()
        
        # k-anonymity is the minimum group size
        k_anonymity = grouped.min() if len(grouped) > 0 else 0
        
        return int(k_anonymity)
    
    def check_k_anonymity(self,
                         data: pd.DataFrame,
                         k_threshold: int = 5) -> Dict[str, Any]:
        """
        Check if dataset satisfies k-anonymity
        
        Args:
            k_threshold: Minimum acceptable k value
            
        Returns:
            k-anonymity analysis results
        """
        k_value = self.compute_k_anonymity(data)
        
        return {
            'k_value': k_value,
            'k_threshold': k_threshold,
            'satisfies_k_anonymity': k_value >= k_threshold,
            'quasi_identifiers': self.quasi_identifiers or data.columns.tolist()
        }
    
    def get_risky_groups(self,
                        data: pd.DataFrame,
                        k_threshold: int = 5) -> pd.DataFrame:
        """
        Identify groups that violate k-anonymity
        
        Returns:
            DataFrame with risky group information
        """
        if self.quasi_identifiers is None:
            qi_cols = data.columns.tolist()
        else:
            qi_cols = [col for col in self.quasi_identifiers if col in data.columns]
        
        if not qi_cols:
            return pd.DataFrame()
        
        # Group by quasi-identifiers
        grouped = data.groupby(qi_cols).size().reset_index(name='group_size')
        
        # Filter risky groups
        risky_groups = grouped[grouped['group_size'] < k_threshold]
        
        return risky_groups


class LDiversityValidator:
    """Validate l-diversity property"""
    
    def __init__(self,
                quasi_identifiers: Optional[List[str]] = None,
                sensitive_attributes: Optional[List[str]] = None):
        """
        Initialize l-diversity validator
        
        Args:
            quasi_identifiers: Columns that could identify individuals
            sensitive_attributes: Sensitive columns to protect
        """
        self.quasi_identifiers = quasi_identifiers
        self.sensitive_attributes = sensitive_attributes
    
    def compute_l_diversity(self, data: pd.DataFrame) -> float:
        """
        Compute l-diversity value
        
        Returns:
            Minimum diversity value across all groups
        """
        if self.quasi_identifiers is None:
            qi_cols = [col for col in data.columns 
                      if col not in (self.sensitive_attributes or [])]
        else:
            qi_cols = [col for col in self.quasi_identifiers if col in data.columns]
        
        if self.sensitive_attributes is None:
            # Assume last column is sensitive
            sensitive_cols = [data.columns[-1]]
        else:
            sensitive_cols = [col for col in self.sensitive_attributes 
                            if col in data.columns]
        
        if not qi_cols or not sensitive_cols:
            return float('inf')
        
        min_diversity = float('inf')
        
        for sensitive_col in sensitive_cols:
            # Group by quasi-identifiers
            grouped = data.groupby(qi_cols)[sensitive_col].apply(
                lambda x: x.nunique()
            )
            
            # l-diversity is the minimum diversity
            if len(grouped) > 0:
                min_diversity = min(min_diversity, grouped.min())
        
        return min_diversity
    
    def check_l_diversity(self,
                         data: pd.DataFrame,
                         l_threshold: int = 2) -> Dict[str, Any]:
        """
        Check if dataset satisfies l-diversity
        
        Args:
            l_threshold: Minimum acceptable l value
            
        Returns:
            l-diversity analysis results
        """
        l_value = self.compute_l_diversity(data)
        
        return {
            'l_value': l_value,
            'l_threshold': l_threshold,
            'satisfies_l_diversity': l_value >= l_threshold,
            'quasi_identifiers': self.quasi_identifiers,
            'sensitive_attributes': self.sensitive_attributes
        }
    
    def compute_entropy_l_diversity(self, data: pd.DataFrame) -> float:
        """
        Compute entropy l-diversity
        
        Returns:
            Minimum entropy across all groups
        """
        if self.quasi_identifiers is None:
            qi_cols = [col for col in data.columns 
                      if col not in (self.sensitive_attributes or [])]
        else:
            qi_cols = [col for col in self.quasi_identifiers if col in data.columns]
        
        if self.sensitive_attributes is None:
            sensitive_cols = [data.columns[-1]]
        else:
            sensitive_cols = [col for col in self.sensitive_attributes 
                            if col in data.columns]
        
        if not qi_cols or not sensitive_cols:
            return float('inf')
        
        min_entropy = float('inf')
        
        for sensitive_col in sensitive_cols:
            # Group by quasi-identifiers
            for _, group in data.groupby(qi_cols):
                # Calculate entropy for this group
                value_counts = group[sensitive_col].value_counts()
                probabilities = value_counts / len(group)
                
                # Shannon entropy
                entropy = -sum(p * np.log(p) for p in probabilities if p > 0)
                
                min_entropy = min(min_entropy, entropy)
        
        # Convert to l-value (2^entropy)
        return 2 ** min_entropy if min_entropy != float('inf') else float('inf')


class TClosenessValidator:
    """Validate t-closeness property"""
    
    def __init__(self,
                quasi_identifiers: Optional[List[str]] = None,
                sensitive_attributes: Optional[List[str]] = None):
        """
        Initialize t-closeness validator
        """
        self.quasi_identifiers = quasi_identifiers
        self.sensitive_attributes = sensitive_attributes
    
    def compute_t_closeness(self, data: pd.DataFrame) -> float:
        """
        Compute t-closeness value
        
        Returns:
            Maximum distance between group and overall distributions
        """
        if self.quasi_identifiers is None:
            qi_cols = [col for col in data.columns 
                      if col not in (self.sensitive_attributes or [])]
        else:
            qi_cols = [col for col in self.quasi_identifiers if col in data.columns]
        
        if self.sensitive_attributes is None:
            sensitive_cols = [data.columns[-1]]
        else:
            sensitive_cols = [col for col in self.sensitive_attributes 
                            if col in data.columns]
        
        if not qi_cols or not sensitive_cols:
            return 0.0
        
        max_distance = 0.0
        
        for sensitive_col in sensitive_cols:
            # Overall distribution
            overall_dist = data[sensitive_col].value_counts(normalize=True)
            
            # Group by quasi-identifiers
            for _, group in data.groupby(qi_cols):
                if len(group) == 0:
                    continue
                
                # Group distribution
                group_dist = group[sensitive_col].value_counts(normalize=True)
                
                # Calculate Earth Mover's Distance (simplified)
                all_values = set(overall_dist.index) | set(group_dist.index)
                
                distance = 0.0
                for value in all_values:
                    p1 = overall_dist.get(value, 0)
                    p2 = group_dist.get(value, 0)
                    distance += abs(p1 - p2)
                
                distance /= 2  # Normalize
                max_distance = max(max_distance, distance)
        
        return max_distance
    
    def check_t_closeness(self,
                         data: pd.DataFrame,
                         t_threshold: float = 0.2) -> Dict[str, Any]:
        """
        Check if dataset satisfies t-closeness
        
        Args:
            t_threshold: Maximum acceptable t value
            
        Returns:
            t-closeness analysis results
        """
        t_value = self.compute_t_closeness(data)
        
        return {
            't_value': t_value,
            't_threshold': t_threshold,
            'satisfies_t_closeness': t_value <= t_threshold,
            'quasi_identifiers': self.quasi_identifiers,
            'sensitive_attributes': self.sensitive_attributes
        }


class MembershipInferenceAttack:
    """Test vulnerability to membership inference attacks"""
    
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
    
    def compute_membership_risk(self,
                              real_data: pd.DataFrame,
                              synthetic_data: pd.DataFrame,
                              test_size: float = 0.2) -> float:
        """
        Estimate risk of membership inference attack
        
        Returns:
            Risk score (0-1, lower is better)
        """
        # Convert to numeric only
        real_numeric = real_data.select_dtypes(include=[np.number])
        synthetic_numeric = synthetic_data.select_dtypes(include=[np.number])
        
        if real_numeric.empty or synthetic_numeric.empty:
            return 0.0
        
        # Sample test points from real data
        n_test = int(len(real_data) * test_size)
        test_indices = np.random.choice(len(real_data), n_test, replace=False)
        test_points = real_numeric.iloc[test_indices].values
        
        # Fit nearest neighbors on synthetic data
        nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        nn.fit(synthetic_numeric.values)
        
        # Find distances to nearest synthetic neighbors
        distances, _ = nn.kneighbors(test_points)
        avg_distances = distances.mean(axis=1)
        
        # Compute risk score
        # If real points are very close to synthetic points, higher risk
        threshold = np.percentile(avg_distances, 10)
        risk_score = (avg_distances < threshold).mean()
        
        return float(risk_score)


class AttributeDisclosureRisk:
    """Measure attribute disclosure risk"""
    
    def compute_attribute_risk(self,
                             real_data: pd.DataFrame,
                             synthetic_data: pd.DataFrame,
                             sensitive_columns: Optional[List[str]] = None) -> float:
        """
        Compute attribute disclosure risk
        
        Returns:
            Risk score (0-1, lower is better)
        """
        if sensitive_columns is None:
            # Consider all non-numeric columns as potentially sensitive
            sensitive_columns = real_data.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if not sensitive_columns:
            return 0.0
        
        risks = []
        
        for col in sensitive_columns:
            if col not in synthetic_data.columns:
                continue
            
            # Check if synthetic data preserves rare values
            real_value_counts = real_data[col].value_counts()
            synthetic_value_counts = synthetic_data[col].value_counts()
            
            # Find rare values in real data
            rare_threshold = 0.01 * len(real_data)
            rare_values = real_value_counts[real_value_counts < rare_threshold].index
            
            if len(rare_values) == 0:
                continue
            
            # Check if rare values appear in synthetic data
            disclosure_count = 0
            for value in rare_values:
                if value in synthetic_value_counts.index:
                    # Rare value disclosed
                    disclosure_count += 1
            
            risk = disclosure_count / len(rare_values) if len(rare_values) > 0 else 0
            risks.append(risk)
        
        return np.mean(risks) if risks else 0.0


class PrivacyEvaluator:
    """Comprehensive privacy evaluation"""
    
    def __init__(self,
                epsilon: float = 1.0,
                delta: float = 1e-5,
                k_threshold: int = 5,
                l_threshold: int = 2,
                t_threshold: float = 0.2):
        """
        Initialize privacy evaluator
        
        Args:
            epsilon: Differential privacy parameter
            delta: Differential privacy parameter
            k_threshold: k-anonymity threshold
            l_threshold: l-diversity threshold
            t_threshold: t-closeness threshold
        """
        self.dp_validator = DifferentialPrivacyValidator(epsilon, delta)
        self.k_anonymity_validator = KAnonymityValidator()
        self.l_diversity_validator = LDiversityValidator()
        self.t_closeness_validator = TClosenessValidator()
        self.membership_attack = MembershipInferenceAttack()
        self.attribute_risk = AttributeDisclosureRisk()
        
        self.k_threshold = k_threshold
        self.l_threshold = l_threshold
        self.t_threshold = t_threshold
    
    def evaluate_privacy(self,
                        real_data: pd.DataFrame,
                        synthetic_data: pd.DataFrame) -> PrivacyMetrics:
        """
        Perform comprehensive privacy evaluation
        
        Returns:
            PrivacyMetrics object with all results
        """
        # Differential privacy
        dp_results = self.dp_validator.check_differential_privacy(
            real_data, synthetic_data
        )
        
        # k-anonymity
        k_results = self.k_anonymity_validator.check_k_anonymity(
            synthetic_data, self.k_threshold
        )
        
        # l-diversity
        l_results = self.l_diversity_validator.check_l_diversity(
            synthetic_data, self.l_threshold
        )
        
        # t-closeness
        t_results = self.t_closeness_validator.check_t_closeness(
            synthetic_data, self.t_threshold
        )
        
        # Attack risks
        membership_risk = self.membership_attack.compute_membership_risk(
            real_data, synthetic_data
        )
        
        attribute_risk = self.attribute_risk.compute_attribute_risk(
            real_data, synthetic_data
        )
        
        # Compute overall privacy score
        privacy_components = [
            1.0 - dp_results['privacy_loss'],
            1.0 if k_results['satisfies_k_anonymity'] else 0.5,
            1.0 if l_results['satisfies_l_diversity'] else 0.5,
            1.0 if t_results['satisfies_t_closeness'] else 0.5,
            1.0 - membership_risk,
            1.0 - attribute_risk
        ]
        
        privacy_score = np.mean(privacy_components)
        
        return PrivacyMetrics(
            epsilon=dp_results['estimated_epsilon'],
            delta=dp_results['delta'],
            k_anonymity=k_results['k_value'],
            l_diversity=l_results['l_value'],
            t_closeness=t_results['t_value'],
            membership_disclosure_risk=membership_risk,
            attribute_disclosure_risk=attribute_risk,
            privacy_score=privacy_score
        )
    
    def generate_privacy_report(self,
                              real_data: pd.DataFrame,
                              synthetic_data: pd.DataFrame) -> str:
        """
        Generate detailed privacy report
        
        Returns:
            Formatted privacy report
        """
        metrics = self.evaluate_privacy(real_data, synthetic_data)
        
        report = []
        report.append("=" * 60)
        report.append("PRIVACY EVALUATION REPORT")
        report.append("=" * 60)
        
        report.append("\nDifferential Privacy:")
        report.append(f"  Estimated ε: {metrics.epsilon:.3f}")
        report.append(f"  δ parameter: {metrics.delta}")
        report.append(f"  Status: {'✓ Satisfied' if metrics.epsilon <= self.dp_validator.epsilon else '✗ Violated'}")
        
        report.append("\nk-Anonymity:")
        report.append(f"  k value: {metrics.k_anonymity}")
        report.append(f"  Threshold: {self.k_threshold}")
        report.append(f"  Status: {'✓ Satisfied' if metrics.k_anonymity >= self.k_threshold else '✗ Violated'}")
        
        report.append("\nl-Diversity:")
        report.append(f"  l value: {metrics.l_diversity:.2f}")
        report.append(f"  Threshold: {self.l_threshold}")
        report.append(f"  Status: {'✓ Satisfied' if metrics.l_diversity >= self.l_threshold else '✗ Violated'}")
        
        report.append("\nt-Closeness:")
        report.append(f"  t value: {metrics.t_closeness:.3f}")
        report.append(f"  Threshold: {self.t_threshold}")
        report.append(f"  Status: {'✓ Satisfied' if metrics.t_closeness <= self.t_threshold else '✗ Violated'}")
        
        report.append("\nAttack Risks:")
        report.append(f"  Membership Inference Risk: {metrics.membership_disclosure_risk:.1%}")
        report.append(f"  Attribute Disclosure Risk: {metrics.attribute_disclosure_risk:.1%}")
        
        report.append(f"\nOverall Privacy Score: {metrics.privacy_score:.2f}/1.00")
        
        if metrics.privacy_score >= 0.8:
            report.append("Privacy Level: HIGH ✓")
        elif metrics.privacy_score >= 0.6:
            report.append("Privacy Level: MEDIUM ⚠")
        else:
            report.append("Privacy Level: LOW ✗")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)