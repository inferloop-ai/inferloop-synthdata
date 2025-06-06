import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import DPMSolverMultistepScheduler
import numpy as np
from PIL import Image
import json
from typing import List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

class DiffusionGenerator:
    """Generate synthetic images using Stable Diffusion with conditioning from real-time profiles."""
    
    def __init__(self, 
                 model_name: str = "runwayml/stable-diffusion-v1-5",
                 device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize pipelines
        self.txt2img_pipeline = None
        self.img2img_pipeline = None
        
        self._load_models()
    
    def _load_models(self):
        """Load diffusion models."""
        try:
            logger.info(f"Loading Stable Diffusion model: {self.model_name}")
            
            # Text-to-image pipeline
            self.txt2img_pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True
            )
            
            # Use DPM solver for faster generation
            self.txt2img_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.txt2img_pipeline.scheduler.config
            )
            
            self.txt2img_pipeline = self.txt2img_pipeline.to(self.device)
            
            # Image-to-image pipeline (for conditioning on existing images)
            self.img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True
            )
            self.img2img_pipeline = self.img2img_pipeline.to(self.device)
            
            # Enable memory efficient attention
            if hasattr(self.txt2img_pipeline, "enable_attention_slicing"):
                self.txt2img_pipeline.enable_attention_slicing()
                self.img2img_pipeline.enable_attention_slicing()
            
            logger.info("Diffusion models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load diffusion models: {e}")
            raise
    
    def generate_from_profile(self, 
                            profile_path: str,
                            num_images: int = 10,
                            batch_size: int = 4,
                            num_inference_steps: int = 50,
                            guidance_scale: float = 7.5) -> List[Image.Image]:
        """Generate images based on a distribution profile."""
        
        # Load profile
        with open(profile_path, 'r') as f:
            profile = json.load(f)
        
        # Extract generation guidance
        guidance = profile.get('generation_guidance', {}).get('diffusion', {})
        prompts = self._create_prompts_from_profile(profile)
        
        # Use profile guidance if available
        num_inference_steps = guidance.get('recommended_steps', num_inference_steps)
        guidance_scale = guidance.get('guidance_scale', guidance_scale)
        
        return self.generate_from_prompts(
            prompts=prompts,
            num_images=num_images,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
    
    def _create_prompts_from_profile(self, profile: Dict) -> List[str]:
        """Create text prompts based on profile characteristics."""
        base_prompts = []
        
        # Extract guidance from profile
        guidance = profile.get('generation_guidance', {}).get('diffusion', {})
        suggested_prompts = guidance.get('suggested_prompts', [])
        
        # Use conditioning profile for more detailed prompts
        conditioning = profile.get('conditioning_profile', {})
        hints = conditioning.get('generation_hints', {})
        
        # Scene guidance
        scene_guidance = hints.get('scene_guidance', {})
        preferred_scene = scene_guidance.get('preferred_scene', 'outdoor scene')
        
        # Object guidance
        object_guidance = hints.get('object_guidance', {})
        preferred_objects = object_guidance.get('preferred_classes', [])
        
        # Create varied prompts
        if suggested_prompts:
            for prompt in suggested_prompts:
                base_prompts.append(prompt)
        else:
            # Fallback prompts based on scene and objects
            base_prompts = [
                f"high quality {preferred_scene}",
                f"detailed {preferred_scene} with realistic lighting",
                f"photorealistic {preferred_scene}, professional photography"
            ]
        
        # Add object-specific prompts
        if preferred_objects:
            for obj in preferred_objects[:3]:  # Top 3 objects
                base_prompts.append(f"high quality photo of {obj} in {preferred_scene}")
        
        # Add quality and style modifiers
        quality_modifiers = [
            ", high quality, detailed, photorealistic",
            ", professional photography, sharp focus, good lighting",
            ", realistic, high resolution, clear details",
            ", natural lighting, high dynamic range"
        ]
        
        enhanced_prompts = []
        for i, prompt in enumerate(base_prompts):
            modifier = quality_modifiers[i % len(quality_modifiers)]
            enhanced_prompts.append(prompt + modifier)
        
        return enhanced_prompts
    
    def generate_from_prompts(self,
                            prompts: List[str],
                            num_images: int = 10,
                            batch_size: int = 4,
                            num_inference_steps: int = 50,
                            guidance_scale: float = 7.5,
                            height: int = 512,
                            width: int = 512) -> List[Image.Image]:
        """Generate images from text prompts."""
        
        if self.txt2img_pipeline is None:
            raise RuntimeError("Text-to-image pipeline not initialized")
        
        all_images = []
        images_generated = 0
        
        while images_generated < num_images:
            # Select prompt for this batch
            prompt = prompts[images_generated % len(prompts)]
            
            # Calculate batch size for this iteration
            current_batch_size = min(batch_size, num_images - images_generated)
            
            try:
                logger.info(f"Generating {current_batch_size} images with prompt: '{prompt[:50]}...'")
                
                # Generate images
                with torch.no_grad():
                    result = self.txt2img_pipeline(
                        prompt=[prompt] * current_batch_size,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        height=height,
                        width=width,
                        generator=torch.Generator(device=self.device).manual_seed(
                            np.random.randint(0, 2**32)
                        )
                    )
                
                batch_images = result.images
                all_images.extend(batch_images)
                images_generated += len(batch_images)
                
                logger.info(f"Generated {images_generated}/{num_images} images")
                
            except Exception as e:
                logger.error(f"Failed to generate batch: {e}")
                # Continue with next batch
                images_generated += current_batch_size
        
        return all_images[:num_images]
    
    def generate_img2img_from_reference(self,
                                      reference_images: List[Image.Image],
                                      prompts: List[str],
                                      num_images: int = 10,
                                      strength: float = 0.7,
                                      num_inference_steps: int = 50,
                                      guidance_scale: float = 7.5) -> List[Image.Image]:
        """Generate images using image-to-image conditioning."""
        
        if self.img2img_pipeline is None:
            raise RuntimeError("Image-to-image pipeline not initialized")
        
        all_images = []
        
        for i in range(num_images):
            # Select reference image and prompt
            ref_image = reference_images[i % len(reference_images)]
            prompt = prompts[i % len(prompts)]
            
            try:
                logger.debug(f"Generating img2img {i+1}/{num_images}")
                
                with torch.no_grad():
                    result = self.img2img_pipeline(
                        prompt=prompt,
                        image=ref_image,
                        strength=strength,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=torch.Generator(device=self.device).manual_seed(
                            np.random.randint(0, 2**32)
                        )
                    )
                
                all_images.extend(result.images)
                
            except Exception as e:
                logger.error(f"Failed to generate img2img {i+1}: {e}")
                continue
        
        return all_images
    
    def save_images(self, images: List[Image.Image], output_dir: str, prefix: str = "diffusion") -> List[str]:
        """Save generated images to directory."""
        import os
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for i, image in enumerate(images):
            filename = f"{prefix}_{i:04d}.png"
            filepath = output_path / filename
            
            try:
                image.save(filepath)
                saved_paths.append(str(filepath))
            except Exception as e:
                logger.error(f"Failed to save image {filename}: {e}")
        
        logger.info(f"Saved {len(saved_paths)} images to {output_dir}")
        return saved_paths
    
    def cleanup(self):
        """Clean up GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Test the diffusion generator
    try:
        generator = DiffusionGenerator()
        
        # Test prompt-based generation
        test_prompts = [
            "urban street scene with cars and pedestrians",
            "modern office building exterior",
            "busy intersection with traffic lights"
        ]
        
        images = generator.generate_from_prompts(
            prompts=test_prompts,
            num_images=3,
            batch_size=1,
            num_inference_steps=20  # Faster for testing
        )
        
        print(f"Generated {len(images)} images")
        
        # Save test images
        saved_paths = generator.save_images(images, "./data/generated", "test_diffusion")
        print(f"Saved to: {saved_paths}")
        
        generator.cleanup()
        
    except Exception as e:
        print(f"Test failed: {e}")
