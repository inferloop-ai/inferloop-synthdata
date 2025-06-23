# inferloop-synthetic/sdk/validator.py
"""
Validation and quality assessment for synthetic data
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

try:
    from sdmetrics.reports.single_table import QualityReport
    from sdmetrics.single_table import evaluate_quality
    SDMETRICS_AVAILABLE = True
except ImportError:
    SDMETRICS_AVAILABLE = False
    logger.warning("SDMetrics not available. Install with: pip install sdmetrics")


class SyntheticDataValidator:
    """Comprehensive validator for synthetic data quality"""
    
    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.validation_results = {}
        
    def validate_all(self) -> Dict[str, Any]:
        """Run comprehensive validation suite"""
        logger.info("Running comprehensive synthetic data validation...")
        
        results = {
            'basic_stats': self.validate_basic_statistics(),
            'distribution_similarity': self.validate_distributions(),
            'correlation_preservation': self.validate_correlations(),
            'privacy_metrics': self.validate_privacy(),
            'utility_metrics': self.validate_utility(),
            'overall_quality': 0.0
        }
        
        # Calculate overall quality score
        scores = []
        for metric_group in results.values():
            if isinstance(metric_group, dict) and 'score' in metric_group:
                scores.append(metric_group['score'])
        
        if scores:
            results['overall_quality'] = np.mean(scores)
        
        # Add SDMetrics evaluation if available
        if SDMETRICS_AVAILABLE:
            results['sdmetrics'] = self.validate_with_sdmetrics()
        
        self.validation_results = results
        return results
    
    def validate_basic_statistics(self) -> Dict[str, Any]:
        """Validate basic statistical properties"""
        logger.info("Validating basic statistics...")
        
        real_stats = self.real_data.describe()
        synthetic_stats = self.synthetic_data.describe()
        
        # Calculate differences
        differences = {}
        scores = []
        
        for col in real_stats.columns:
            if col in synthetic_stats.columns:
                real_col = real_stats[col]
                synth_col = synthetic_stats[col]
                
                # Calculate relative differences for key statistics
                col_diffs = {}
                for stat in ['mean', 'std', 'min', 'max']:
                    if stat in real_col.index and stat in synth_col.index:
                        real_val = real_col[stat]
                        synth_val = synth_col[stat]
                        if real_val != 0:
                            rel_diff = abs((synth_val - real_val) / real_val)
                            col_diffs[stat] = rel_diff
                            scores.append(max(0, 1 - rel_diff))  # Score decreases with difference
                
                differences[col] = col_diffs
        
        avg_score = np.mean(scores) if scores else 0.0
        
        return {
            'score': avg_score,
            'differences': differences,
            'real_stats': real_stats.to_dict(),
            'synthetic_stats': synthetic_stats.to_dict()
        }
    
    def validate_distributions(self) -> Dict[str, Any]:
        """Validate distribution similarity using statistical tests"""
        logger.info("Validating distribution similarity...")
        
        ks_tests = {}
        chi2_tests = {}
        overall_scores = []
        
        for col in self.real_data.columns:
            if col in self.synthetic_data.columns:
                real_col = self.real_data[col].dropna()
                synth_col = self.synthetic_data[col].dropna()
                
                if real_col.dtype in ['int64', 'float64'] and synth_col.dtype in ['int64', 'float64']:
                    # Kolmogorov-Smirnov test for continuous variables
                    ks_stat, ks_p = stats.ks_2samp(real_col, synth_col)
                    ks_tests[col] = {'statistic': ks_stat, 'p_value': ks_p}
                    
                    # Score based on KS statistic (lower is better)
                    ks_score = max(0, 1 - ks_stat)
                    overall_scores.append(ks_score)
                
                else:
                    # Chi-square test for categorical variables
                    try:
                        real_counts = real_col.value_counts()
                        synth_counts = synth_col.value_counts()
                        
                        # Align categories
                        all_categories = set(real_counts.index) | set(synth_counts.index)
                        real_aligned = [real_counts.get(cat, 0) for cat in all_categories]
                        synth_aligned = [synth_counts.get(cat, 0) for cat in all_categories]
                        
                        if sum(synth_aligned) > 0:  # Avoid division by zero
                            chi2_stat, chi2_p = stats.chisquare(synth_aligned, real_aligned)
                            chi2_tests[col] = {'statistic': chi2_stat, 'p_value': chi2_p}
                            
                            # Score based on p-value (higher is better)
                            chi2_score = min(1.0, chi2_p)
                            overall_scores.append(chi2_score)
                    
                    except Exception as e:
                        logger.warning(f"Chi-square test failed for column {col}: {e}")
        
        avg_score = np.mean(overall_scores) if overall_scores else 0.0
        
        return {
            'score': avg_score,
            'ks_tests': ks_tests,
            'chi2_tests': chi2_tests
        }
    
    def validate_correlations(self) -> Dict[str, Any]:
        """Validate correlation structure preservation"""
        logger.info("Validating correlation preservation...")
        
        # Get numerical columns
        numerical_cols = self.real_data.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            return {'score': 1.0, 'message': 'Insufficient numerical columns for correlation analysis'}
        
        real_corr = self.real_data[numerical_cols].corr()
        synthetic_corr = self.synthetic_data[numerical_cols].corr()
        
        # Calculate correlation difference
        corr_diff = abs(real_corr - synthetic_corr)
        avg_corr_diff = corr_diff.mean().mean()
        
        # Score based on correlation preservation (lower difference is better)
        correlation_score = max(0, 1 - avg_corr_diff)
        
        return {
            'score': correlation_score,
            'average_correlation_difference': avg_corr_diff,
            'real_correlation_matrix': real_corr.to_dict(),
            'synthetic_correlation_matrix': synthetic_corr.to_dict()
        }
    
    def validate_privacy(self) -> Dict[str, Any]:
        """Validate privacy preservation with comprehensive metrics"""
        logger.info("Validating privacy preservation...")
        
        # Import privacy evaluator
        try:
            from .privacy import PrivacyEvaluator
            
            # Create privacy evaluator
            privacy_evaluator = PrivacyEvaluator()
            
            # Evaluate privacy
            privacy_metrics = privacy_evaluator.evaluate_privacy(
                self.real_data,
                self.synthetic_data
            )
            
            # Generate privacy report
            privacy_report = privacy_evaluator.generate_privacy_report(
                self.real_data,
                self.synthetic_data
            )
            
            return {
                'score': privacy_metrics.privacy_score,
                'epsilon': privacy_metrics.epsilon,
                'delta': privacy_metrics.delta,
                'k_anonymity': privacy_metrics.k_anonymity,
                'l_diversity': privacy_metrics.l_diversity,
                't_closeness': privacy_metrics.t_closeness,
                'membership_disclosure_risk': privacy_metrics.membership_disclosure_risk,
                'attribute_disclosure_risk': privacy_metrics.attribute_disclosure_risk,
                'detailed_report': privacy_report,
                'metrics': privacy_metrics.to_dict()
            }
            
        except ImportError:
            logger.warning("Privacy module not available, using basic metrics")
            
            # Fallback to basic distance-based metrics
            numerical_cols = self.real_data.select_dtypes(include=[np.number]).columns
            
            if len(numerical_cols) == 0:
                return {'score': 1.0, 'message': 'No numerical columns for privacy analysis'}
            
            real_numeric = self.real_data[numerical_cols].values
            synthetic_numeric = self.synthetic_data[numerical_cols].values
            
            # Normalize data
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            real_scaled = scaler.fit_transform(real_numeric)
            synthetic_scaled = scaler.transform(synthetic_numeric)
            
            # Calculate minimum distances
            min_distances = []
            for synth_point in synthetic_scaled:
                distances = np.linalg.norm(real_scaled - synth_point, axis=1)
                min_distances.append(np.min(distances))
            
            avg_min_distance = np.mean(min_distances)
            
            # Score based on average minimum distance (higher is better for privacy)
            privacy_score = min(1.0, avg_min_distance / 2.0)  # Normalize to [0,1]
            
            return {
                'score': privacy_score,
                'average_minimum_distance': avg_min_distance,
                'minimum_distances': min_distances,
                'message': 'Using basic privacy metrics'
            }
    
    def validate_utility(self) -> Dict[str, Any]:
        """Validate utility through machine learning task performance"""
        logger.info("Validating utility through ML task performance...")
        
        # Find a suitable target column (categorical with reasonable number of classes)
        target_col = None
        categorical_cols = self.real_data.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            unique_vals = self.real_data[col].nunique()
            if 2 <= unique_vals <= 10:  # Good for classification
                target_col = col
                break
        
        if target_col is None:
            return {'score': 1.0, 'message': 'No suitable target column found for utility analysis'}
        
        # Prepare features and targets
        feature_cols = [col for col in self.real_data.columns if col != target_col]
        feature_cols = self.real_data[feature_cols].select_dtypes(include=[np.number]).columns
        
        if len(feature_cols) == 0:
            return {'score': 1.0, 'message': 'No numerical features for utility analysis'}
        
        try:
            # Train on real data, test on synthetic
            X_real = self.real_data[feature_cols].fillna(0)
            y_real = self.real_data[target_col]
            
            X_synthetic = self.synthetic_data[feature_cols].fillna(0)
            y_synthetic = self.synthetic_data[target_col]
            
            # Train model on real data
            rf_real = RandomForestClassifier(n_estimators=50, random_state=42)
            rf_real.fit(X_real, y_real)
            
            # Train model on synthetic data
            rf_synthetic = RandomForestClassifier(n_estimators=50, random_state=42)
            rf_synthetic.fit(X_synthetic, y_synthetic)
            
            # Test both models on a held-out portion of real data
            X_test, _, y_test, _ = train_test_split(X_real, y_real, test_size=0.3, random_state=42)
            
            real_accuracy = accuracy_score(y_test, rf_real.predict(X_test))
            synthetic_accuracy = accuracy_score(y_test, rf_synthetic.predict(X_test))
            
            # Score based on how close synthetic model performance is to real
            utility_score = min(1.0, synthetic_accuracy / real_accuracy) if real_accuracy > 0 else 0.0
            
            return {
                'score': utility_score,
                'real_model_accuracy': real_accuracy,
                'synthetic_model_accuracy': synthetic_accuracy,
                'target_column': target_col,
                'feature_columns': list(feature_cols)
            }
        
        except Exception as e:
            logger.warning(f"Utility validation failed: {e}")
            return {'score': 0.5, 'error': str(e)}
    
    def validate_with_sdmetrics(self) -> Dict[str, Any]:
        """Validate using SDMetrics library"""
        if not SDMETRICS_AVAILABLE:
            return {'message': 'SDMetrics not available'}
        
        try:
            logger.info("Running SDMetrics evaluation...")
            
            # Generate quality report
            quality_report = QualityReport()
            quality_report.generate(self.real_data, self.synthetic_data)
            
            # Get overall score
            overall_score = evaluate_quality(self.real_data, self.synthetic_data)
            
            return {
                'score': overall_score,
                'detailed_report': quality_report.get_details(),
                'properties': quality_report.get_properties()
            }
        
        except Exception as e:
            logger.warning(f"SDMetrics evaluation failed: {e}")
            return {'error': str(e)}
    
    def generate_report(self) -> str:
        """Generate a human-readable validation report"""
        if not self.validation_results:
            self.validate_all()
        
        report_lines = [
            "Synthetic Data Validation Report",
            "=" * 40,
            "",
            f"Overall Quality Score: {self.validation_results['overall_quality']:.3f}",
            ""
        ]
        
        # Basic statistics
        basic_stats = self.validation_results.get('basic_stats', {})
        report_lines.extend([
            f"Basic Statistics Score: {basic_stats.get('score', 0):.3f}",
            ""
        ])
        
        # Distribution similarity
        dist_sim = self.validation_results.get('distribution_similarity', {})
        report_lines.extend([
            f"Distribution Similarity Score: {dist_sim.get('score', 0):.3f}",
            ""
        ])
        
        # Correlation preservation
        corr_pres = self.validation_results.get('correlation_preservation', {})
        report_lines.extend([
            f"Correlation Preservation Score: {corr_pres.get('score', 0):.3f}",
            ""
        ])
        
        # Privacy metrics
        privacy = self.validation_results.get('privacy_metrics', {})
        report_lines.extend([
            f"Privacy Score: {privacy.get('score', 0):.3f}",
            ""
        ])
        
        # Utility metrics
        utility = self.validation_results.get('utility_metrics', {})
        report_lines.extend([
            f"Utility Score: {utility.get('score', 0):.3f}",
            ""
        ])
        
        return "\n".join(report_lines)
