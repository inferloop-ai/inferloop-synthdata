# ==================== realtime/profiler/image_profiler.py ====================
import cv2
import numpy as np
from skimage import measure, feature
from skimage.filters import gaussian
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

class ImageProfiler:
    """Extract visual characteristics from images for distribution modeling."""
    
    def __init__(self):
        self.features = {}
        
    def profile_single_image(self, image: np.ndarray) -> Dict:
        """Extract comprehensive features from a single image."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        features = {}
        
        # Basic visual features
        features.update(self._extract_basic_features(gray))
        
        # Texture features
        features.update(self._extract_texture_features(gray))
        
        # Color features (if color image)
        if len(image.shape) == 3:
            features.update(self._extract_color_features(image))
            
        # Quality metrics
        features.update(self._extract_quality_metrics(gray))
        
        return features
    
    def _extract_basic_features(self, gray: np.ndarray) -> Dict:
        """Extract basic visual statistics."""
        return {
            'brightness': float(np.mean(gray)),
            'contrast': float(np.std(gray)),
            'entropy': self._calculate_entropy(gray),
            'dynamic_range': float(np.max(gray) - np.min(gray)),
            'median_brightness': float(np.median(gray))
        }
    
    def _extract_texture_features(self, gray: np.ndarray) -> Dict:
        """Extract texture-related features."""
        # Edge density
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Local Binary Pattern variance (texture measure)
        lbp = feature.local_binary_pattern(gray, 8, 1.5, method='uniform')
        lbp_var = np.var(lbp)
        
        return {
            'edge_density': float(edge_density),
            'texture_variance': float(lbp_var),
            'smoothness': float(1 - edge_density)
        }
    
    def _extract_color_features(self, image: np.ndarray) -> Dict:
        """Extract color-related features."""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate color moments
        color_features = {}
        for i, channel in enumerate(['hue', 'saturation', 'value']):
            channel_data = hsv[:, :, i]
            color_features[f'{channel}_mean'] = float(np.mean(channel_data))
            color_features[f'{channel}_std'] = float(np.std(channel_data))
            
        # Color diversity (number of unique colors)
        unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))
        color_features['color_diversity'] = unique_colors
        
        return color_features
    
    def _extract_quality_metrics(self, gray: np.ndarray) -> Dict:
        """Extract image quality indicators."""
        # Blur detection using Laplacian variance
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Noise estimation
        noise_level = self._estimate_noise(gray)
        
        return {
            'blur_score': float(blur_score),
            'noise_level': float(noise_level),
            'sharpness': float(1 / (1 + np.exp(-blur_score/100)))  # Sigmoid normalization
        }
    
    def _calculate_entropy(self, gray: np.ndarray) -> float:
        """Calculate Shannon entropy of the image."""
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist / hist.sum()  # Normalize
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return float(entropy)
    
    def _estimate_noise(self, gray: np.ndarray) -> float:
        """Estimate noise level using wavelet decomposition approximation."""
        # Simple noise estimation using high-pass filter
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        noise = np.std(filtered)
        return noise
    
    def profile_batch(self, images: List[np.ndarray]) -> Dict:
        """Profile a batch of images and return aggregated statistics."""
        all_features = []
        
        for img in images:
            features = self.profile_single_image(img)
            all_features.append(features)
        
        # Aggregate features
        aggregated = {}
        if all_features:
            feature_names = all_features[0].keys()
            
            for feature_name in feature_names:
                values = [f[feature_name] for f in all_features]
                aggregated[f'{feature_name}_mean'] = float(np.mean(values))
                aggregated[f'{feature_name}_std'] = float(np.std(values))
                aggregated[f'{feature_name}_min'] = float(np.min(values))
                aggregated[f'{feature_name}_max'] = float(np.max(values))
                aggregated[f'{feature_name}_median'] = float(np.median(values))
        
        return aggregated

if __name__ == "__main__":
    # Test the profiler
    profiler = ImageProfiler()
    
    # Create a test image
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    features = profiler.profile_single_image(test_img)
    
    print("Image Profile Features:")
    for key, value in features.items():
        print(f"{key}: {value:.4f}")
