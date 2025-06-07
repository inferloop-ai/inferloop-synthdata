import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import pickle
import json

logger = logging.getLogger(__name__)

class StyleGAN2Generator:
    """Generate synthetic images using StyleGAN2 architecture."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.generator = None
        self.latent_dim = 512
        self.num_layers = 18
        
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load a pre-trained StyleGAN2 model."""
        try:
            logger.info(f"Loading StyleGAN2 model from {model_path}")
            
            # Load pickle file (typical StyleGAN2 format)
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Extract generator
            if 'G_ema' in model_data:
                self.generator = model_data['G_ema'].to(self.device)
            elif 'generator' in model_data:
                self.generator = model_data['generator'].to(self.device)
            else:
                raise ValueError("Could not find generator in model file")
            
            self.generator.eval()
            logger.info("StyleGAN2 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load StyleGAN2 model: {e}")
            # Fall back to creating a dummy generator for demonstration
            self._create_dummy_generator()
    
    def _create_dummy_generator(self):
        """Create a simple dummy generator for demonstration purposes."""
        logger.warning("Creating dummy generator - for demonstration only")
        
        class DummyGenerator(nn.Module):
            def __init__(self, latent_dim=512, output_size=512):
                super().__init__()
                self.latent_dim = latent_dim
                
                # Simple upsampling network
                self.network = nn.Sequential(
                    # Start from 4x4
                    nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0),
                    nn.BatchNorm2d(512),
                    nn.ReLU(True),
                    
                    # 8x8
                    nn.ConvTranspose2d(512, 256, 4, 2, 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(True),
                    
                    # 16x16
                    nn.ConvTranspose2d(256, 128, 4, 2, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    
                    # 32x32
                    nn.ConvTranspose2d(128, 64, 4, 2, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    
                    # 64x64
                    nn.ConvTranspose2d(64, 32, 4, 2, 1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(True),
                    
                    # 128x128
                    nn.ConvTranspose2d(32, 16, 4, 2, 1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(True),
                    
                    # 256x256
                    nn.ConvTranspose2d(16, 8, 4, 2, 1),
                    nn.BatchNorm2d(8),
                    nn.ReLU(True),
                    
                    # 512x512
                    nn.ConvTranspose2d(8, 3, 4, 2, 1),
                    nn.Tanh()
                )
            
            def forward(self, z):
                z = z.view(z.size(0), -1, 1, 1)
                return self.network(z)
        
        self.generator = DummyGenerator().to(self.device)
    
    def generate_from_noise(self,
                           num_images: int = 10,
                           truncation_psi: float = 0.7,
                           noise_mode: str = 'random',
                           seed: Optional[int] = None) -> List[Image.Image]:
        """Generate images from random noise."""
        
        if self.generator is None:
            raise RuntimeError("No generator loaded")
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        logger.info(f"Generating {num_images} images with StyleGAN2")
        
        images = []
        
        with torch.no_grad():
            for i in range(num_images):
                # Generate latent code
                z = self._generate_latent_code(truncation_psi, noise_mode)
                
                # Generate image
                try:
                    if hasattr(self.generator, 'synthesis'):
                        # Official StyleGAN2 interface
                        img_tensor = self.generator.synthesis(z)
                    else:
                        # Simple generator interface
                        img_tensor = self.generator(z)
                    
                    # Convert to PIL Image
                    img = self._tensor_to_pil(img_tensor[0])
                    images.append(img)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Generated {i + 1}/{num_images} images")
                        
                except Exception as e:
                    logger.error(f"Failed to generate image {i + 1}: {e}")
                    continue
        
        return images
    
    def generate_with_style_mixing(self,
                                  num_images: int = 10,
                                  mixing_probability: float = 0.5,
                                  truncation_psi: float = 0.7) -> List[Image.Image]:
        """Generate images with style mixing for increased diversity."""
        
        if self.generator is None:
            raise RuntimeError("No generator loaded")
        
        logger.info(f"Generating {num_images} images with style mixing")
        
        images = []
        
        with torch.no_grad():
            for i in range(num_images):
                # Generate two latent codes
                z1 = self._generate_latent_code(truncation_psi, 'random')
                z2 = self._generate_latent_code(truncation_psi, 'random')
                
                # Decide whether to mix styles
                if np.random.random() < mixing_probability:
                    # Mix styles at a random layer
                    mixing_layer = np.random.randint(1, self.num_layers)
                    z_mixed = self._mix_styles(z1, z2, mixing_layer)
                else:
                    z_mixed = z1
                
                # Generate image
                try:
                    if hasattr(self.generator, 'synthesis'):
                        img_tensor = self.generator.synthesis(z_mixed)
                    else:
                        img_tensor = self.generator(z_mixed)
                    
                    img = self._tensor_to_pil(img_tensor[0])
                    images.append(img)
                    
                except Exception as e:
                    logger.error(f"Failed to generate mixed image {i + 1}: {e}")
                    continue
        
        return images
    
    def generate_from_profile(self,
                             profile_path: str,
                             num_images: int = 10,
                             adaptation_strength: float = 0.8) -> List[Image.Image]:
        """Generate images conditioned on a distribution profile."""
        
        # Load profile
        with open(profile_path, 'r') as f:
            profile = json.load(f)
        
        # Extract generation guidance for GANs
        guidance = profile.get('generation_guidance', {}).get('gan', {})
        
        # Use profile-specific parameters
        truncation_psi = guidance.get('truncation_psi', 0.7)
        style_mixing_prob = guidance.get('style_mixing_prob', 0.5)
        
        # Adjust truncation based on profile diversity
        if 'diversity' in profile.get('distributions', {}):
            diversity_info = profile['distributions']['diversity']
            if 'diversity_score' in diversity_info:
                diversity_score = diversity_info['diversity_score']
                # Higher diversity -> less truncation
                truncation_psi = max(0.3, truncation_psi * (1 - diversity_score * adaptation_strength))
        
        logger.info(f"Generating from profile with truncation_psi={truncation_psi:.3f}")
        
        return self.generate_with_style_mixing(
            num_images=num_images,
            mixing_probability=style_mixing_prob,
            truncation_psi=truncation_psi
        )
    
    def _generate_latent_code(self, truncation_psi: float, noise_mode: str) -> torch.Tensor:
        """Generate a latent code."""
        
        if noise_mode == 'random':
            z = torch.randn(1, self.latent_dim, device=self.device)
        elif noise_mode == 'const':
            z = torch.ones(1, self.latent_dim, device=self.device)
        else:
            z = torch.randn(1, self.latent_dim, device=self.device)
        
        # Apply truncation
        if truncation_psi < 1.0:
            z = z * truncation_psi
        
        return z
    
    def _mix_styles(self, z1: torch.Tensor, z2: torch.Tensor, mixing_layer: int) -> torch.Tensor:
        """Mix two style codes at a specific layer."""
        
        # For simplified implementation, just interpolate
        alpha = np.random.random()
        z_mixed = alpha * z1 + (1 - alpha) * z2
        
        return z_mixed
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert a tensor to PIL Image."""
        
        # Denormalize from [-1, 1] to [0, 1]
        tensor = (tensor + 1) / 2
        
        # Clamp to valid range
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to numpy
        if tensor.dim() == 3:  # CHW
            tensor = tensor.permute(1, 2, 0)  # HWC
        
        numpy_img = tensor.cpu().numpy()
        
        # Convert to uint8
        numpy_img = (numpy_img * 255).astype(np.uint8)
        
        # Create PIL Image
        return Image.fromarray(numpy_img)
    
    def save_images(self, images: List[Image.Image], output_dir: str, prefix: str = "gan") -> List[str]:
        """Save generated images to directory."""
        
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
        
        logger.info(f"Saved {len(saved_paths)} GAN images to {output_dir}")
        return saved_paths
    
    def interpolate_between_images(self,
                                  num_steps: int = 10,
                                  truncation_psi: float = 0.7) -> List[Image.Image]:
        """Generate an interpolation sequence between two random points."""
        
        if self.generator is None:
            raise RuntimeError("No generator loaded")
        
        # Generate two random latent codes
        z1 = self._generate_latent_code(truncation_psi, 'random')
        z2 = self._generate_latent_code(truncation_psi, 'random')
        
        images = []
        
        with torch.no_grad():
            for i in range(num_steps):
                # Interpolate between z1 and z2
                alpha = i / (num_steps - 1)
                z_interp = (1 - alpha) * z1 + alpha * z2
                
                # Generate image
                try:
                    if hasattr(self.generator, 'synthesis'):
                        img_tensor = self.generator.synthesis(z_interp)
                    else:
                        img_tensor = self.generator(z_interp)
                    
                    img = self._tensor_to_pil(img_tensor[0])
                    images.append(img)
                    
                except Exception as e:
                    logger.error(f"Failed to generate interpolated image {i + 1}: {e}")
                    continue
        
        return images
    
    def cleanup(self):
        """Clean up GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Test the GAN generator
    try:
        # Initialize with dummy generator for testing
        generator = StyleGAN2Generator()
        
        # Generate some test images
        images = generator.generate_from_noise(
            num_images=5,
            truncation_psi=0.8,
            seed=42
        )
        
        print(f"Generated {len(images)} images")
        
        # Save test images
        saved_paths = generator.save_images(images, "./data/generated", "test_gan")
        print(f"Saved to: {saved_paths}")
        
        # Test interpolation
        interp_images = generator.interpolate_between_images(num_steps=5)
        interp_paths = generator.save_images(interp_images, "./data/generated", "test_interp")
        print(f"Interpolation saved to: {interp_paths}")
        
        generator.cleanup()
        
    except Exception as e:
        print(f"Test failed: {e}")
