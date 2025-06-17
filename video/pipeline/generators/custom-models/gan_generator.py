"""
GAN-based Video Generator
Part of the Inferloop SynthData Video Pipeline

This module provides GAN-based video synthesis capabilities,
supporting various GAN architectures for synthetic video generation.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json
import yaml
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TemporalBlock(nn.Module):
    """Temporal block for video GAN generation"""
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)):
        super(TemporalBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class VideoGANGenerator(nn.Module):
    """
    Generator network for video GAN.
    Transforms latent vectors into video sequences.
    """
    def __init__(self, latent_dim=100, num_frames=16, channels=3, width=64, height=64):
        super(VideoGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.num_frames = num_frames
        self.channels = channels
        self.width = width
        self.height = height
        
        # Initial projection and reshaping
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4 * 2)  # 2 frames initially
        
        # Temporal-spatial generation layers
        self.temporal_layers = nn.ModuleList([
            TemporalBlock(512, 256, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            TemporalBlock(256, 128, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            TemporalBlock(128, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
        ])
        
        # Frame expansion layer
        self.frame_expansion = nn.ConvTranspose3d(64, 64, kernel_size=(4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        
        # Final output layer
        self.output_layer = nn.Conv3d(64, channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.output_activation = nn.Tanh()
        
    def forward(self, z):
        batch_size = z.size(0)
        
        # Project and reshape
        x = self.fc(z)
        x = x.view(batch_size, 512, 2, 4, 4)
        
        # Apply temporal-spatial layers
        for layer in self.temporal_layers:
            x = layer(x)
        
        # Expand frames
        x = self.frame_expansion(x)
        
        # Ensure we have the right number of frames
        if x.size(2) < self.num_frames:
            # Use interpolation to reach desired frame count
            x = F.interpolate(
                x, 
                size=(self.num_frames, x.size(3), x.size(4)), 
                mode='trilinear', 
                align_corners=False
            )
        
        # Final output
        x = self.output_layer(x)
        x = self.output_activation(x)
        
        return x

class VideoGANDiscriminator(nn.Module):
    """
    Discriminator network for video GAN.
    Evaluates whether video sequences are real or generated.
    """
    def __init__(self, num_frames=16, channels=3, width=64, height=64):
        super(VideoGANDiscriminator, self).__init__()
        
        # 3D convolutional layers for spatio-temporal analysis
        self.layers = nn.Sequential(
            nn.Conv3d(channels, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(64, 128, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(128, 256, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(256, 512, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2),
        )
        
        # Temporal pooling
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Final classification
        self.fc = nn.Linear(512, 1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Apply convolutional layers
        x = self.layers(x)
        
        # Temporal pooling
        x = self.temporal_pool(x)
        x = x.view(batch_size, -1)
        
        # Classification
        x = self.fc(x)
        
        return x

class VideoGANTrainer:
    """
    Trainer for VideoGAN models.
    Handles training, evaluation, and video generation.
    """
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the VideoGAN trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model parameters
        self.latent_dim = self.config.get('latent_dim', 100)
        self.num_frames = self.config.get('num_frames', 16)
        self.channels = self.config.get('channels', 3)
        self.width = self.config.get('width', 64)
        self.height = self.config.get('height', 64)
        
        # Initialize models
        self.generator = VideoGANGenerator(
            self.latent_dim, 
            self.num_frames, 
            self.channels, 
            self.width, 
            self.height
        ).to(self.device)
        
        self.discriminator = VideoGANDiscriminator(
            self.num_frames, 
            self.channels, 
            self.width, 
            self.height
        ).to(self.device)
        
        # Optimizers
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), 
            lr=self.config.get('g_lr', 0.0002),
            betas=(self.config.get('beta1', 0.5), 0.999)
        )
        
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=self.config.get('d_lr', 0.0002),
            betas=(self.config.get('beta1', 0.5), 0.999)
        )
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        logger.info(f"VideoGANTrainer initialized on {self.device}")
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return {}
    
    def train_step(self, real_videos):
        """
        Perform a single training step.
        
        Args:
            real_videos: Batch of real videos (B, C, T, H, W)
            
        Returns:
            Dictionary of losses
        """
        batch_size = real_videos.size(0)
        real_videos = real_videos.to(self.device)
        
        # Labels
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        
        # ---------------------
        # Train Discriminator
        # ---------------------
        self.d_optimizer.zero_grad()
        
        # Real videos
        real_output = self.discriminator(real_videos)
        d_real_loss = self.criterion(real_output, real_labels)
        
        # Fake videos
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_videos = self.generator(z)
        fake_output = self.discriminator(fake_videos.detach())
        d_fake_loss = self.criterion(fake_output, fake_labels)
        
        # Combined loss
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        # ---------------------
        # Train Generator
        # ---------------------
        self.g_optimizer.zero_grad()
        
        # Generate videos
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_videos = self.generator(z)
        fake_output = self.discriminator(fake_videos)
        
        # Generator loss (fool discriminator)
        g_loss = self.criterion(fake_output, real_labels)
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'd_real_loss': d_real_loss.item(),
            'd_fake_loss': d_fake_loss.item()
        }
    
    def generate_video(self, num_videos: int = 1, latent_vectors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate videos using the trained generator.
        
        Args:
            num_videos: Number of videos to generate
            latent_vectors: Optional latent vectors to use
            
        Returns:
            Generated videos as tensor (B, C, T, H, W)
        """
        self.generator.eval()
        
        with torch.no_grad():
            if latent_vectors is None:
                latent_vectors = torch.randn(num_videos, self.latent_dim, device=self.device)
                
            videos = self.generator(latent_vectors)
            
        return videos
    
    def interpolate_videos(self, num_steps: int = 10) -> torch.Tensor:
        """
        Generate videos by interpolating between two latent vectors.
        
        Args:
            num_steps: Number of interpolation steps
            
        Returns:
            Interpolated videos as tensor (num_steps, C, T, H, W)
        """
        self.generator.eval()
        
        # Generate two random latent vectors
        z_start = torch.randn(1, self.latent_dim, device=self.device)
        z_end = torch.randn(1, self.latent_dim, device=self.device)
        
        # Interpolate between them
        videos = []
        with torch.no_grad():
            for step in range(num_steps):
                alpha = step / (num_steps - 1)
                z = (1 - alpha) * z_start + alpha * z_end
                video = self.generator(z)
                videos.append(video)
        
        return torch.cat(videos, dim=0)
    
    def save_models(self, save_dir: str):
        """Save generator and discriminator models."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save generator
        torch.save(self.generator.state_dict(), os.path.join(save_dir, 'generator.pth'))
        
        # Save discriminator
        torch.save(self.discriminator.state_dict(), os.path.join(save_dir, 'discriminator.pth'))
        
        # Save config
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump({
                'latent_dim': self.latent_dim,
                'num_frames': self.num_frames,
                'channels': self.channels,
                'width': self.width,
                'height': self.height,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"Models saved to {save_dir}")
    
    def load_models(self, load_dir: str):
        """Load generator and discriminator models."""
        # Load generator
        generator_path = os.path.join(load_dir, 'generator.pth')
        if os.path.exists(generator_path):
            self.generator.load_state_dict(torch.load(generator_path, map_location=self.device))
            logger.info(f"Generator loaded from {generator_path}")
        
        # Load discriminator
        discriminator_path = os.path.join(load_dir, 'discriminator.pth')
        if os.path.exists(discriminator_path):
            self.discriminator.load_state_dict(torch.load(discriminator_path, map_location=self.device))
            logger.info(f"Discriminator loaded from {discriminator_path}")
        
        # Load config if available
        config_path = os.path.join(load_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                logger.info(f"Loaded model config from {config_path}")
                
                # Check if dimensions match
                if (loaded_config.get('latent_dim') != self.latent_dim or
                    loaded_config.get('num_frames') != self.num_frames or
                    loaded_config.get('width') != self.width or
                    loaded_config.get('height') != self.height):
                    logger.warning("Loaded model has different dimensions than current configuration")
    
    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert a video tensor to numpy array.
        
        Args:
            tensor: Video tensor (C, T, H, W) or (B, C, T, H, W)
            
        Returns:
            Video as numpy array (T, H, W, C) or (B, T, H, W, C)
        """
        # Move to CPU and detach
        video = tensor.detach().cpu()
        
        # Handle batch dimension
        if video.dim() == 5:
            # (B, C, T, H, W) -> (B, T, H, W, C)
            video = video.permute(0, 2, 3, 4, 1)
        else:
            # (C, T, H, W) -> (T, H, W, C)
            video = video.permute(1, 2, 3, 0)
        
        # Convert to numpy and scale to [0, 255]
        video = ((video + 1) / 2 * 255).clamp(0, 255).to(torch.uint8).numpy()
        
        return video

# Example usage
if __name__ == "__main__":
    # Create trainer with default settings
    trainer = VideoGANTrainer()
    
    # Generate a sample video
    videos = trainer.generate_video(num_videos=1)
    
    # Convert to numpy for visualization
    videos_np = trainer.tensor_to_numpy(videos)
    
    print(f"Generated video shape: {videos_np.shape}")
    
    # In a real application, you would save this video or visualize it
    output_dir = "outputs/generated_videos"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    trainer.save_models(os.path.join(output_dir, "video_gan_model"))
