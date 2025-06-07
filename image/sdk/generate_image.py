import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import uuid
import time

# Import generation modules
from generation.generate_diffusion import DiffusionImageGenerator
from generation.generate_gan import StyleGAN2Generator
from generation.generate_simulation import SimulationGenerator

logger = logging.getLogger(__name__)

class ImageGenerator:
    """SDK interface for synthetic image generation.
    
    This class provides a simplified interface for generating synthetic images
    using various generation methods (diffusion, GAN, simulation).
    """
    
    def __init__(self, output_dir: Optional[str] = None, config_path: Optional[str] = None):
        """Initialize the image generator.
        
        Args:
            output_dir: Directory to save generated images. Defaults to './data/generated'.
            config_path: Path to configuration file. If None, uses default settings.
        """
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'data', 'generated')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load configuration if provided
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        # Initialize generators
        self.diffusion_generator = None
        self.gan_generator = None
        self.simulation_generator = None
        
        # Track generation metadata
        self.last_generation_id = None
        self.last_generation_params = None
        self.last_generation_time = None
    
    def _init_diffusion_generator(self, model_path: Optional[str] = None) -> DiffusionImageGenerator:
        """Initialize the diffusion generator."""
        if self.diffusion_generator is None:
            model_path = model_path or self.config.get('diffusion_model_path')
            self.diffusion_generator = DiffusionImageGenerator(model_path=model_path)
        return self.diffusion_generator
    
    def _init_gan_generator(self, model_path: Optional[str] = None) -> StyleGAN2Generator:
        """Initialize the GAN generator."""
        if self.gan_generator is None:
            model_path = model_path or self.config.get('gan_model_path')
            self.gan_generator = StyleGAN2Generator(model_path=model_path)
        return self.gan_generator
    
    def _init_simulation_generator(self) -> SimulationGenerator:
        """Initialize the simulation generator."""
        if self.simulation_generator is None:
            self.simulation_generator = SimulationGenerator()
        return self.simulation_generator
    
    def generate_from_text(self, 
                          prompt: str, 
                          negative_prompt: Optional[str] = None,
                          count: int = 1, 
                          width: int = 512, 
                          height: int = 512,
                          guidance_scale: float = 7.5,
                          num_inference_steps: int = 50,
                          seed: Optional[int] = None,
                          save: bool = True) -> List[np.ndarray]:
        """Generate images from text prompts using diffusion models.
        
        Args:
            prompt: Text prompt describing the desired image.
            negative_prompt: Text prompt describing what to avoid in the image.
            count: Number of images to generate.
            width: Width of generated images.
            height: Height of generated images.
            guidance_scale: How closely to follow the prompt (higher = more faithful).
            num_inference_steps: Number of denoising steps (higher = better quality but slower).
            seed: Random seed for reproducibility.
            save: Whether to save the generated images.
            
        Returns:
            List of generated images as numpy arrays.
        """
        generator = self._init_diffusion_generator()
        
        # Generate unique ID for this generation batch
        generation_id = str(uuid.uuid4())
        self.last_generation_id = generation_id
        self.last_generation_params = {
            'method': 'diffusion',
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'count': count,
            'width': width,
            'height': height,
            'guidance_scale': guidance_scale,
            'num_inference_steps': num_inference_steps,
            'seed': seed
        }
        
        start_time = time.time()
        images = []
        
        for i in range(count):
            # Generate image
            current_seed = seed + i if seed is not None else None
            image = generator.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=current_seed
            )
            
            images.append(image)
            
            # Save image if requested
            if save:
                img_filename = f"{generation_id}_{i:03d}.png"
                img_path = os.path.join(self.output_dir, img_filename)
                generator.save_image(image, img_path)
                logger.info(f"Saved image to {img_path}")
        
        self.last_generation_time = time.time() - start_time
        logger.info(f"Generated {count} images in {self.last_generation_time:.2f} seconds")
        
        return images
    
    def generate_from_latents(self,
                             latent_vectors: Optional[List[np.ndarray]] = None,
                             count: int = 1,
                             truncation_psi: float = 0.7,
                             seed: Optional[int] = None,
                             save: bool = True) -> List[np.ndarray]:
        """Generate images from latent vectors using GAN models.
        
        Args:
            latent_vectors: List of latent vectors to use. If None, random vectors are generated.
            count: Number of images to generate if latent_vectors is None.
            truncation_psi: Controls variation vs. quality tradeoff (lower = better quality but less diverse).
            seed: Random seed for reproducibility.
            save: Whether to save the generated images.
            
        Returns:
            List of generated images as numpy arrays.
        """
        generator = self._init_gan_generator()
        
        # Generate unique ID for this generation batch
        generation_id = str(uuid.uuid4())
        self.last_generation_id = generation_id
        self.last_generation_params = {
            'method': 'gan',
            'count': count if latent_vectors is None else len(latent_vectors),
            'truncation_psi': truncation_psi,
            'seed': seed
        }
        
        start_time = time.time()
        
        # Generate images
        if latent_vectors is None:
            images = generator.generate_batch(
                batch_size=count,
                truncation_psi=truncation_psi,
                seed=seed
            )
        else:
            images = []
            for z in latent_vectors:
                img = generator.generate_from_latent(z, truncation_psi=truncation_psi)
                images.append(img)
        
        # Save images if requested
        if save:
            for i, image in enumerate(images):
                img_filename = f"{generation_id}_{i:03d}.png"
                img_path = os.path.join(self.output_dir, img_filename)
                generator.save_image(image, img_path)
                logger.info(f"Saved image to {img_path}")
        
        self.last_generation_time = time.time() - start_time
        logger.info(f"Generated {len(images)} images in {self.last_generation_time:.2f} seconds")
        
        return images
    
    def generate_simulation(self,
                          scene_type: str = 'urban',
                          objects: Optional[List[Dict]] = None,
                          lighting: Optional[Dict] = None,
                          weather: Optional[Dict] = None,
                          count: int = 1,
                          width: int = 1024,
                          height: int = 768,
                          save: bool = True) -> List[np.ndarray]:
        """Generate synthetic images using procedural simulation.
        
        Args:
            scene_type: Type of scene to generate ('urban', 'indoor', 'nature', 'industrial').
            objects: List of objects to place in the scene, each with position, scale, rotation.
            lighting: Lighting parameters (intensity, color, direction).
            weather: Weather parameters (type, intensity).
            count: Number of variations to generate.
            width: Width of generated images.
            height: Height of generated images.
            save: Whether to save the generated images.
            
        Returns:
            List of generated images as numpy arrays.
        """
        generator = self._init_simulation_generator()
        
        # Generate unique ID for this generation batch
        generation_id = str(uuid.uuid4())
        self.last_generation_id = generation_id
        self.last_generation_params = {
            'method': 'simulation',
            'scene_type': scene_type,
            'objects': objects,
            'lighting': lighting,
            'weather': weather,
            'count': count,
            'width': width,
            'height': height
        }
        
        start_time = time.time()
        images = []
        
        # Set up scene
        generator.setup_scene(
            scene_type=scene_type,
            width=width,
            height=height
        )
        
        # Add objects if specified
        if objects:
            for obj in objects:
                generator.add_object(**obj)
        
        # Set lighting if specified
        if lighting:
            generator.set_lighting(**lighting)
        
        # Set weather if specified
        if weather:
            generator.set_weather(**weather)
        
        # Generate variations
        for i in range(count):
            # Slightly vary parameters for each image
            if i > 0:
                generator.randomize_parameters(strength=0.2)
            
            # Render image
            image = generator.render()
            images.append(image)
            
            # Save image if requested
            if save:
                img_filename = f"{generation_id}_{i:03d}.png"
                img_path = os.path.join(self.output_dir, img_filename)
                generator.save_image(image, img_path)
                logger.info(f"Saved image to {img_path}")
        
        self.last_generation_time = time.time() - start_time
        logger.info(f"Generated {count} simulation images in {self.last_generation_time:.2f} seconds")
        
        return images
    
    def generate_from_profile(self,
                            profile_path: str,
                            count: int = 10,
                            method: str = 'auto',
                            save: bool = True) -> List[np.ndarray]:
        """Generate images based on a profile JSON file.
        
        Args:
            profile_path: Path to the profile JSON file.
            count: Number of images to generate.
            method: Generation method ('diffusion', 'gan', 'simulation', or 'auto').
            save: Whether to save the generated images.
            
        Returns:
            List of generated images as numpy arrays.
        """
        # Load profile
        with open(profile_path, 'r') as f:
            profile = json.load(f)
        
        # Determine best generation method if auto
        if method == 'auto':
            if 'text_prompts' in profile and profile['text_prompts']:
                method = 'diffusion'
            elif 'scene_types' in profile and profile['scene_types']:
                method = 'simulation'
            else:
                method = 'gan'
        
        # Generate images based on method
        if method == 'diffusion':
            # Extract prompts from profile
            prompts = profile.get('text_prompts', [])
            if not prompts:
                prompts = ["A synthetic image"]
            
            # Get other parameters
            width = profile.get('width', 512)
            height = profile.get('height', 512)
            
            # Generate images with different prompts
            images = []
            for i in range(count):
                prompt_idx = i % len(prompts)
                prompt = prompts[prompt_idx]
                
                img = self.generate_from_text(
                    prompt=prompt,
                    count=1,
                    width=width,
                    height=height,
                    save=save
                )[0]
                
                images.append(img)
            
            return images
            
        elif method == 'gan':
            # Extract style parameters from profile
            truncation_psi = profile.get('style_variation', 0.7)
            
            # Generate images
            return self.generate_from_latents(
                count=count,
                truncation_psi=truncation_psi,
                save=save
            )
            
        elif method == 'simulation':
            # Extract scene parameters from profile
            scene_types = profile.get('scene_types', ['urban'])
            width = profile.get('width', 1024)
            height = profile.get('height', 768)
            
            # Generate images with different scene types
            images = []
            for i in range(count):
                scene_idx = i % len(scene_types)
                scene_type = scene_types[scene_idx]
                
                img = self.generate_simulation(
                    scene_type=scene_type,
                    count=1,
                    width=width,
                    height=height,
                    save=save
                )[0]
                
                images.append(img)
            
            return images
        
        else:
            raise ValueError(f"Unknown generation method: {method}")
    
    def get_generation_info(self) -> Dict:
        """Get information about the last generation."""
        if self.last_generation_id is None:
            return {}
        
        return {
            'generation_id': self.last_generation_id,
            'parameters': self.last_generation_params,
            'generation_time': self.last_generation_time,
            'output_directory': self.output_dir
        }


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic images')
    parser.add_argument('--method', choices=['text', 'gan', 'simulation', 'profile'], 
                      required=True, help='Generation method')
    parser.add_argument('--prompt', help='Text prompt for diffusion generation')
    parser.add_argument('--profile', help='Path to profile JSON for profile-based generation')
    parser.add_argument('--scene', choices=['urban', 'indoor', 'nature', 'industrial'],
                      help='Scene type for simulation generation')
    parser.add_argument('--count', type=int, default=1, help='Number of images to generate')
    parser.add_argument('--output', help='Output directory')
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize generator
    generator = ImageGenerator(output_dir=args.output)
    
    # Generate images based on method
    if args.method == 'text':
        if not args.prompt:
            parser.error("--prompt is required for text generation")
        
        images = generator.generate_from_text(
            prompt=args.prompt,
            count=args.count
        )
        
    elif args.method == 'gan':
        images = generator.generate_from_latents(
            count=args.count
        )
        
    elif args.method == 'simulation':
        if not args.scene:
            parser.error("--scene is required for simulation generation")
        
        images = generator.generate_simulation(
            scene_type=args.scene,
            count=args.count
        )
        
    elif args.method == 'profile':
        if not args.profile:
            parser.error("--profile is required for profile-based generation")
        
        images = generator.generate_from_profile(
            profile_path=args.profile,
            count=args.count
        )
    
    # Print generation info
    info = generator.get_generation_info()
    print(f"Generated {len(images)} images")
    print(f"Generation ID: {info['generation_id']}")
    print(f"Output directory: {info['output_directory']}")
    print(f"Generation time: {info['generation_time']:.2f} seconds")
