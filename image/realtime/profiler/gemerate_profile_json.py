import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

from .image_profiler import ImageProfiler
from .semantic_profiler import SemanticProfiler
from .distribution_modeler import DistributionModeler

logger = logging.getLogger(__name__)

class ProfileGenerator:
    """Orchestrates the complete profiling pipeline and generates JSON profiles."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Initialize profilers
        self.image_profiler = ImageProfiler()
        self.semantic_profiler = SemanticProfiler(
            yolo_model=self.config.get('yolo_model', 'yolov8n.pt'),
            clip_model=self.config.get('clip_model', 'ViT-B/32')
        )
        self.distribution_modeler = DistributionModeler(
            window_size=self.config.get('window_size', 100)
        )
        
        self.profile_dir = Path(self.config.get('profile_dir', './profiles'))
        self.profile_dir.mkdir(exist_ok=True)
        
    def process_image_batch(self, images: List, source_name: str = "unknown") -> Dict:
        """Process a batch of images and update the distribution model."""
        logger.info(f"Processing batch of {len(images)} images from {source_name}")
        
        batch_features = []
        
        for i, image in enumerate(images):
            try:
                # Get visual features
                visual_features = self.image_profiler.profile_single_image(image)
                
                # Get semantic features
                semantic_features = self.semantic_profiler.profile_single_image(image)
                
                # Combine features
                combined_features = {
                    **visual_features,
                    **semantic_features,
                    'source': source_name,
                    'batch_index': i,
                    'timestamp': datetime.now().isoformat()
                }
                
                batch_features.append(combined_features)
                
                # Add to distribution modeler
                self.distribution_modeler.add_features(combined_features)
                
            except Exception as e:
                logger.error(f"Failed to process image {i} from {source_name}: {e}")
                continue
        
        return {
            'processed_count': len(batch_features),
            'source': source_name,
            'batch_features': batch_features
        }
    
    def generate_profile(self, source_name: str = "latest") -> Dict:
        """Generate a complete profile JSON for the current data."""
        logger.info(f"Generating profile for source: {source_name}")
        
        # Fit distributions on accumulated data
        distributions = self.distribution_modeler.fit_distributions()
        
        # Generate conditioning profile
        conditioning = self.distribution_modeler.generate_conditioning_profile()
        
        # Create comprehensive profile
        profile = {
            'metadata': {
                'source_name': source_name,
                'generation_timestamp': datetime.now().isoformat(),
                'sample_count': len(self.distribution_modeler.feature_history),
                'window_size': self.distribution_modeler.window_size,
                'profiler_version': '1.0.0'
            },
            'distributions': distributions,
            'conditioning_profile': conditioning,
            'generation_guidance': self._create_generation_guidance(distributions, conditioning)
        }
        
        return profile
    
    def _create_generation_guidance(self, distributions: Dict, conditioning: Dict) -> Dict:
        """Create specific guidance for different generation methods."""
        guidance = {
            'diffusion': {},
            'gan': {},
            'simulation': {}
        }
        
        # Extract key metrics for generation guidance
        if conditioning.get('generation_hints'):
            hints = conditioning['generation_hints']
            
            # Diffusion model guidance
            diffusion_prompts = []
            scene_guidance = hints.get('scene_guidance', {})
            object_guidance = hints.get('object_guidance', {})
            
            if scene_guidance.get('preferred_scene'):
                diffusion_prompts.append(scene_guidance['preferred_scene'])
            
            if object_guidance.get('preferred_classes'):
                top_objects = object_guidance['preferred_classes'][:3]
                diffusion_prompts.extend(top_objects)
            
            guidance['diffusion'] = {
                'suggested_prompts': diffusion_prompts,
                'recommended_steps': 50,
                'guidance_scale': 7.5,
                'conditioning_strength': 0.8
            }
            
            # GAN guidance
            if 'brightness' in hints:
                brightness_info = hints['brightness']
                guidance['gan'] = {
                    'truncation_psi': 0.7 if brightness_info['target_std'] < 50 else 0.9,
                    'noise_mode': 'random',
                    'style_mixing_prob': 0.5,
                    'target_brightness': brightness_info['target_mean']
                }
            
            # Simulation guidance
            guidance['simulation'] = {
                'environment_type': scene_guidance.get('preferred_scene', 'urban'),
                'weather_conditions': 'clear',  # Could be derived from scene analysis
                'lighting_conditions': 'day',   # Could be derived from brightness
                'object_density': object_guidance.get('target_object_count', 3)
            }
        
        return guidance
    
    def save_profile(self, profile: Dict, source_name: str = "latest") -> str:
        """Save profile to JSON file."""
        filename = f"stream_{source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.profile_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(profile, f, indent=2, default=str)
            
            # Also save as latest for easy access
            latest_path = self.profile_dir / f"stream_{source_name}.json"
            with open(latest_path, 'w') as f:
                json.dump(profile, f, indent=2, default=str)
            
            logger.info(f"Profile saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save profile: {e}")
            raise
    
    def load_profile(self, source_name: str) -> Optional[Dict]:
        """Load the latest profile for a source."""
        filepath = self.profile_dir / f"stream_{source_name}.json"
        
        if not filepath.exists():
            logger.warning(f"Profile not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r') as f:
                profile = json.load(f)
            return profile
        except Exception as e:
            logger.error(f"Failed to load profile from {filepath}: {e}")
            return None
    
    def process_and_save(self, images: List, source_name: str) -> str:
        """Complete pipeline: process images, generate profile, and save."""
        # Process the batch
        batch_result = self.process_image_batch(images, source_name)
        
        # Generate profile
        profile = self.generate_profile(source_name)
        
        # Save profile
        filepath = self.save_profile(profile, source_name)
        
        logger.info(f"Completed profiling pipeline for {source_name}: {len(images)} images processed")
        return filepath

if __name__ == "__main__":
    import numpy as np
    
    # Test the profile generator
    generator = ProfileGenerator()
    
    # Create test images
    test_images = [
        np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        for _ in range(10)
    ]
    
    # Process and save
    filepath = generator.process_and_save(test_images, "test_source")
    print(f"Profile saved to: {filepath}")
    
    # Load and display
    profile = generator.load_profile("test_source")
    if profile:
        print("Profile loaded successfully")
        print(f"Sample count: {profile['metadata']['sample_count']}")
