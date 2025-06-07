import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)

class DistributionModeler:
    """Model statistical distributions of image features for synthetic generation conditioning."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.feature_history = []
        self.current_distributions = {}
        
    def add_features(self, features: Dict) -> None:
        """Add new feature measurements to the rolling window."""
        self.feature_history.append(features)
        
        # Maintain rolling window
        if len(self.feature_history) > self.window_size:
            self.feature_history = self.feature_history[-self.window_size:]
    
    def fit_distributions(self) -> Dict:
        """Fit statistical distributions to accumulated features."""
        if len(self.feature_history) < 10:
            logger.warning(f"Insufficient data for distribution fitting: {len(self.feature_history)} samples")
            return {}
        
        distributions = {}
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(self.feature_history)
        
        # Fit distributions for numeric features
        numeric_features = df.select_dtypes(include=[np.number]).columns
        
        for feature in numeric_features:
            if feature in df.columns and df[feature].notna().sum() > 5:
                values = df[feature].dropna().values
                distributions[feature] = self._fit_single_distribution(values, feature)
        
        # Handle categorical features (like detected classes)
        distributions.update(self._model_categorical_features(df))
        
        # Model feature correlations
        distributions['correlations'] = self._calculate_correlations(df[numeric_features])
        
        # Calculate diversity metrics
        distributions['diversity'] = self._calculate_diversity_metrics(df)
        
        self.current_distributions = distributions
        return distributions
    
    def _fit_single_distribution(self, values: np.ndarray, feature_name: str) -> Dict:
        """Fit multiple distribution types and select the best one."""
        if len(values) < 5:
            return {'type': 'insufficient_data', 'count': len(values)}
        
        distributions_tried = {}
        
        # Normal distribution
        try:
            mu, sigma = stats.norm.fit(values)
            ksstat, pvalue = stats.kstest(values, lambda x: stats.norm.cdf(x, mu, sigma))
            distributions_tried['normal'] = {
                'params': {'mu': float(mu), 'sigma': float(sigma)},
                'ks_statistic': float(ksstat),
                'p_value': float(pvalue),
                'aic': self._calculate_aic(values, stats.norm.pdf, mu, sigma)
            }
        except Exception as e:
            logger.debug(f"Normal distribution fitting failed for {feature_name}: {e}")
        
        # Log-normal distribution (for positive values)
        if np.all(values > 0):
            try:
                shape, loc, scale = stats.lognorm.fit(values)
                ksstat, pvalue = stats.kstest(values, lambda x: stats.lognorm.cdf(x, shape, loc, scale))
                distributions_tried['lognormal'] = {
                    'params': {'shape': float(shape), 'loc': float(loc), 'scale': float(scale)},
                    'ks_statistic': float(ksstat),
                    'p_value': float(pvalue),
                    'aic': self._calculate_aic(values, stats.lognorm.pdf, shape, loc, scale)
                }
            except Exception as e:
                logger.debug(f"Log-normal distribution fitting failed for {feature_name}: {e}")
        
        # Gamma distribution (for positive values)
        if np.all(values > 0):
            try:
                a, loc, scale = stats.gamma.fit(values)
                ksstat, pvalue = stats.kstest(values, lambda x: stats.gamma.cdf(x, a, loc, scale))
                distributions_tried['gamma'] = {
                    'params': {'a': float(a), 'loc': float(loc), 'scale': float(scale)},
                    'ks_statistic': float(ksstat),
                    'p_value': float(pvalue),
                    'aic': self._calculate_aic(values, stats.gamma.pdf, a, loc, scale)
                }
            except Exception as e:
                logger.debug(f"Gamma distribution fitting failed for {feature_name}: {e}")
        
        # Gaussian Mixture Model
        try:
            # Try 1-3 components
            best_gmm = None
            best_aic = float('inf')
            
            for n_components in range(1, min(4, len(values)//3 + 1)):
                gmm = GaussianMixture(n_components=n_components, random_state=42)
                gmm.fit(values.reshape(-1, 1))
                aic = gmm.aic(values.reshape(-1, 1))
                
                if aic < best_aic:
                    best_aic = aic
                    best_gmm = gmm
            
            if best_gmm is not None:
                distributions_tried['gaussian_mixture'] = {
                    'n_components': int(best_gmm.n_components),
                    'weights': best_gmm.weights_.tolist(),
                    'means': best_gmm.means_.flatten().tolist(),
                    'covariances': best_gmm.covariances_.flatten().tolist(),
                    'aic': float(best_aic)
                }
        except Exception as e:
            logger.debug(f"GMM fitting failed for {feature_name}: {e}")
        
        # Select best distribution based on AIC (lower is better)
        if distributions_tried:
            best_dist = min(distributions_tried.items(), 
                          key=lambda x: x[1].get('aic', float('inf')))
            
            result = {
                'best_distribution': best_dist[0],
                'best_params': best_dist[1],
                'all_distributions': distributions_tried,
                'basic_stats': {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'skewness': float(stats.skew(values)),
                    'kurtosis': float(stats.kurtosis(values))
                }
            }
        else:
            # Fallback to basic statistics
            result = {
                'best_distribution': 'empirical',
                'basic_stats': {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'skewness': float(stats.skew(values)),
                    'kurtosis': float(stats.kurtosis(values))
                },
                'empirical_values': values.tolist()[:50]  # Store sample of values
            }
        
        return result
    
    def _calculate_aic(self, data: np.ndarray, pdf_func, *params) -> float:
        """Calculate Akaike Information Criterion."""
        try:
            log_likelihood = np.sum(np.log(pdf_func(data, *params) + 1e-10))
            k = len(params)
            n = len(data)
            return 2 * k - 2 * log_likelihood
        except:
            return float('inf')
    
    def _model_categorical_features(self, df: pd.DataFrame) -> Dict:
        """Model categorical features like detected object classes."""
        categorical_models = {}
        
        # Handle detected classes
        if 'detected_classes' in df.columns:
            all_classes = []
            for classes_list in df['detected_classes'].dropna():
                if isinstance(classes_list, list):
                    all_classes.extend(classes_list)
            
            if all_classes:
                class_counts = pd.Series(all_classes).value_counts()
                total_detections = len(all_classes)
                
                categorical_models['detected_classes'] = {
                    'total_detections': total_detections,
                    'unique_classes': len(class_counts),
                    'class_probabilities': (class_counts / total_detections).to_dict(),
                    'top_classes': class_counts.head(10).to_dict(),
                    'class_entropy': float(stats.entropy(class_counts.values))
                }
        
        # Handle scene classifications
        scene_cols = [col for col in df.columns if col.startswith('scene_')]
        if scene_cols:
            scene_probs = df[scene_cols].mean().to_dict()
            categorical_models['scene_distribution'] = {
                'scene_probabilities': scene_probs,
                'most_likely_scene': max(scene_probs, key=scene_probs.get),
                'scene_diversity': float(stats.entropy(list(scene_probs.values())))
            }
        
        return categorical_models
    
    def _calculate_correlations(self, df: pd.DataFrame) -> Dict:
        """Calculate feature correlations."""
        if df.shape[1] < 2:
            return {}
        
        try:
            corr_matrix = df.corr()
            
            # Find strongest correlations
            correlations = {}
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    feat1, feat2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    corr_val = corr_matrix.iloc[i, j]
                    
                    if not np.isnan(corr_val) and abs(corr_val) > 0.3:
                        correlations[f"{feat1}_vs_{feat2}"] = float(corr_val)
            
            return {
                'strong_correlations': correlations,
                'correlation_matrix': corr_matrix.to_dict() if corr_matrix.shape[0] <= 20 else {}
            }
            
        except Exception as e:
            logger.error(f"Correlation calculation failed: {e}")
            return {}
    
    def _calculate_diversity_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate diversity and clustering metrics."""
        try:
            numeric_df = df.select_dtypes(include=[np.number]).dropna()
            
            if numeric_df.shape[0] < 5 or numeric_df.shape[1] < 2:
                return {'insufficient_data': True}
            
            # Normalize features for clustering
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)
            
            # K-means clustering to find data groups
            n_clusters = min(5, len(scaled_data) // 3)
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(scaled_data)
                
                cluster_counts = pd.Series(clusters).value_counts()
                cluster_entropy = float(stats.entropy(cluster_counts.values))
                
                # Silhouette-like metric using pairwise distances
                distances = pdist(scaled_data)
                avg_distance = float(np.mean(distances))
                
                return {
                    'n_clusters': n_clusters,
                    'cluster_entropy': cluster_entropy,
                    'cluster_distribution': cluster_counts.to_dict(),
                    'avg_pairwise_distance': avg_distance,
                    'diversity_score': cluster_entropy * (1 + avg_distance)  # Combined metric
                }
            else:
                return {'insufficient_data_for_clustering': True}
                
        except Exception as e:
            logger.error(f"Diversity calculation failed: {e}")
            return {}
    
    def generate_conditioning_profile(self) -> Dict:
        """Generate a conditioning profile for synthetic data generation."""
        distributions = self.fit_distributions()
        
        if not distributions:
            return {}
        
        # Create a simplified profile for generation conditioning
        conditioning = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'sample_count': len(self.feature_history),
            'generation_hints': {}
        }
        
        # Extract key parameters for generation
        for feature, dist_info in distributions.items():
            if isinstance(dist_info, dict) and 'basic_stats' in dist_info:
                stats = dist_info['basic_stats']
                conditioning['generation_hints'][feature] = {
                    'target_mean': stats['mean'],
                    'target_std': stats['std'],
                    'target_range': [stats['min'], stats['max']],
                    'distribution_type': dist_info.get('best_distribution', 'empirical')
                }
        
        # Add categorical guidance
        if 'detected_classes' in distributions:
            class_info = distributions['detected_classes']
            conditioning['generation_hints']['object_guidance'] = {
                'preferred_classes': list(class_info.get('top_classes', {}).keys())[:5],
                'class_probabilities': class_info.get('class_probabilities', {}),
                'target_object_count': class_info.get('total_detections', 0) / len(self.feature_history)
            }
        
        if 'scene_distribution' in distributions:
            scene_info = distributions['scene_distribution']
            conditioning['generation_hints']['scene_guidance'] = {
                'preferred_scene': scene_info.get('most_likely_scene', 'outdoor'),
                'scene_weights': scene_info.get('scene_probabilities', {})
            }
        
        return conditioning
    
    def save_profile(self, filepath: str) -> None:
        """Save the current distribution profile to a JSON file."""
        profile = {
            'distributions': self.current_distributions,
            'conditioning_profile': self.generate_conditioning_profile(),
            'metadata': {
                'sample_count': len(self.feature_history),
                'window_size': self.window_size,
                'timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(profile, f, indent=2, default=str)
        
        logger.info(f"Distribution profile saved to {filepath}")
    
    def load_profile(self, filepath: str) -> Dict:
        """Load a distribution profile from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                profile = json.load(f)
            
            self.current_distributions = profile.get('distributions', {})
            return profile
            
        except Exception as e:
            logger.error(f"Failed to load profile from {filepath}: {e}")
            return {}

if __name__ == "__main__":
    # Test the distribution modeler
    modeler = DistributionModeler(window_size=50)
    
    # Generate some test data
    for i in range(60):
        test_features = {
            'brightness': np.random.normal(120, 30),
            'contrast': np.random.exponential(25),
            'object_count': np.random.poisson(3),
            'detected_classes': np.random.choice(['car', 'person', 'dog'], 
                                               size=np.random.randint(0, 5)).tolist()
        }
        modeler.add_features(test_features)
    
    # Fit distributions
    distributions = modeler.fit_distributions()
    print("Fitted distributions:")
    for key, value in distributions.items():
        print(f"{key}: {type(value)}")
    
    # Generate conditioning profile
    conditioning = modeler.generate_conditioning_profile()
    print("\\nConditioning profile:")
    print(json.dumps(conditioning, indent=2, default=str))
