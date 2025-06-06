"""
VAE-based audio generation models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import logging

from .base import AudioGenerator

logger = logging.getLogger(__name__)

class AudioVAE(nn.Module):
    """Variational Autoencoder for audio generation"""
    
    def __init__(self,
                 input_length: int = 22050,
                 latent_dim: int = 128,
                 num_channels: int = 1):
        """
        Initialize VAE model
        
        Args:
            input_length: Length of audio samples
            latent_dim: Dimension of latent space
            num_channels: Number of audio channels (1 for mono, 2 for stereo)
        """
        super().__init__()
        
        self.input_length = input_length
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        
        # Calculate dimensions
        self.down_layers = 5
        self.initial_hidden = 32
        self.final_hidden = self.initial_hidden * (2 ** (self.down_layers - 1))
        
        # Encoder
        layers = []
        current_channels = num_channels
        hidden_channels = self.initial_hidden
        
        for i in range(self.down_layers):
            layers.append(nn.Conv1d(current_channels, hidden_channels, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm1d(hidden_channels))
            layers.append(nn.LeakyReLU())
            
            current_channels = hidden_channels
            if i < self.down_layers - 1:
                hidden_channels *= 2
                
        self.encoder = nn.Sequential(*layers)
        
        # Calculate flattened size
        self.flattened_size = current_channels * (input_length // (2 ** self.down_layers))
        
        # Latent projections
        self.mu_projection = nn.Linear(self.flattened_size, latent_dim)
        self.logvar_projection = nn.Linear(self.flattened_size, latent_dim)
        
        # Decoder initial projection
        self.latent_to_features = nn.Linear(latent_dim, self.flattened_size)
        
        # Decoder
        layers = []
        current_channels = self.final_hidden
        
        for i in range(self.down_layers):
            hidden_channels = current_channels // 2 if i < self.down_layers - 1 else self.num_channels
            
            layers.append(nn.ConvTranspose1d(
                current_channels,
                hidden_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ))
            
            if i < self.down_layers - 1:
                layers.append(nn.BatchNorm1d(hidden_channels))
                layers.append(nn.LeakyReLU())
            else:
                layers.append(nn.Tanh())  # Output activation to constrain to [-1, 1]
                
            current_channels = hidden_channels
            
        self.decoder = nn.Sequential(*layers)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode audio to latent representation
        
        Args:
            x: Audio tensor [B, C, T]
            
        Returns:
            Tuple of (mu, logvar) tensors
        """
        # Encode
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Get latent parameters
        mu = self.mu_projection(x)
        logvar = self.logvar_projection(x)
        
        return mu, logvar
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick
        
        Args:
            mu: Mean tensor [B, D]
            logvar: Log variance tensor [B, D]
            
        Returns:
            Sampled latent vector [B, D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to audio
        
        Args:
            z: Latent vector [B, D]
            
        Returns:
            Reconstructed audio [B, C, T]
        """
        # Project to feature space
        x = self.latent_to_features(z)
        x = x.view(x.size(0), self.final_hidden, -1)  # Reshape
        
        # Decode
        x = self.decoder(x)
        return x
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input audio tensor [B, C, T]
            
        Returns:
            Tuple of (reconstructed audio, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAEAudioGenerator(AudioGenerator):
    """Audio generator using VAE model"""
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 sample_rate: int = 22050,
                 device: Optional[torch.device] = None,
                 latent_dim: int = 128,
                 **kwargs):
        """
        Initialize VAE audio generator
        
        Args:
            model_path: Path to pretrained model weights
            sample_rate: Target sample rate for generated audio
            device: Torch device to use
            latent_dim: Dimension of latent space
        """
        super().__init__(sample_rate=sample_rate, device=device)
        
        # Model parameters
        self.latent_dim = latent_dim
        
        # Initialize VAE model
        self.model = AudioVAE(
            input_length=sample_rate,  # 1 second of audio
            latent_dim=latent_dim,
            num_channels=1
        ).to(self.device)
        
        # Load weights if provided
        if model_path:
            self.load_checkpoints(model_path)
    
    def load_checkpoints(self, checkpoint_path: str) -> None:
        """Load model weights from checkpoint"""
        try:
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded VAE model weights from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load VAE weights: {e}")
    
    def generate(self,
                prompt: Optional[str] = None,
                num_samples: int = 1,
                seed: Optional[int] = None,
                temperature: float = 1.0,
                **kwargs) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Generate audio samples using VAE
        
        Args:
            prompt: Text prompt (unused in basic VAE)
            num_samples: Number of samples to generate
            seed: Random seed for reproducibility
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            Tuple containing:
                - List of audio tensors
                - Dict with metadata about the generation
        """
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            
        # Generate from random latent vectors
        self.model.eval()
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(num_samples, self.latent_dim).to(self.device) * temperature
            
            # Decode to audio
            audio_batch = self.model.decode(z)
        
        # Process outputs
        audios = []
        for i in range(num_samples):
            # Get single audio
            audio = audio_batch[i].squeeze(0).cpu()
            
            # Ensure in range [-1, 1]
            if torch.max(torch.abs(audio)) > 0:
                audio = audio / torch.max(torch.abs(audio))
                
            audios.append(audio)
        
        # Return audios and metadata
        metadata = {
            "model_type": "vae",
            "sample_rate": self.sample_rate,
            "latent_dim": self.latent_dim,
            "seed": seed,
            "temperature": temperature,
        }
        
        return audios, metadata