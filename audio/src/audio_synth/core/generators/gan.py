# audio_synth/core/generators/gan.py
"""
GAN-based Audio Generator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .base import BaseAudioGenerator
from ..utils.config import GenerationConfig, AudioConfig

class GANAudioGenerator(BaseAudioGenerator):
    """GAN-based audio generator"""
    
    def __init__(self, config: GenerationConfig, audio_config: AudioConfig):
        super().__init__(config, audio_config)
        self.latent_dim = 128
        self.generator = None
        self.discriminator = None
        self.condition_embedding = None
        
    def load_model(self, model_path: str) -> None:
        """Load GAN models"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Initialize models
            self.generator = AudioGenerator(
                latent_dim=self.latent_dim,
                output_length=int(self.audio_config.duration * self.audio_config.sample_rate),
                num_channels=self.audio_config.channels
            )
            
            self.discriminator = AudioDiscriminator(
                input_length=int(self.audio_config.duration * self.audio_config.sample_rate),
                num_channels=self.audio_config.channels
            )
            
            # Load weights
            self.generator.load_state_dict(checkpoint['generator'])
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            
            # Set to evaluation mode
            self.generator.eval()
            self.discriminator.eval()
            
            print(f"GAN models loaded from {model_path}")
            
        except Exception as e:
            print(f"Could not load GAN models: {e}")
            print("Using randomly initialized GAN models")
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models with random weights"""
        output_length = int(self.audio_config.duration * self.audio_config.sample_rate)
        
        self.generator = AudioGenerator(
            latent_dim=self.latent_dim,
            output_length=output_length,
            num_channels=self.audio_config.channels
        )
        
        self.discriminator = AudioDiscriminator(
            input_length=output_length,
            num_channels=self.audio_config.channels
        )
        
        # Initialize condition embedding
        self.condition_embedding = ConditionEmbedding(
            embedding_dim=64,
            latent_dim=self.latent_dim
        )
    
    def generate(self, 
                 prompt: Optional[str] = None,
                 conditions: Optional[Dict[str, Any]] = None,
                 **kwargs) -> torch.Tensor:
        """Generate audio using GAN"""
        
        if self.generator is None:
            self._initialize_models()
        
        # Sample from latent space
        latent_vector = torch.randn(1, self.latent_dim)
        
        # Apply conditioning if provided
        if conditions:
            condition_vector = self._encode_conditions(conditions)
            latent_vector = self._apply_conditioning(latent_vector, condition_vector)
        
        # Generate audio
        with torch.no_grad():
            audio = self.generator(latent_vector)
        
        # Post-process
        audio = audio.squeeze()
        audio = self._post_process(audio, conditions)
        
        return audio
    
    def generate_batch(self, 
                      prompts: List[str],
                      conditions: Optional[List[Dict[str, Any]]] = None,
                      **kwargs) -> List[torch.Tensor]:
        """Generate batch of audio samples"""
        
        batch_size = len(prompts)
        
        if self.generator is None:
            self._initialize_models()
        
        # Sample batch of latent vectors
        latent_batch = torch.randn(batch_size, self.latent_dim)
        
        # Apply conditioning if provided
        if conditions:
            condition_batch = torch.stack([
                self._encode_conditions(cond) for cond in conditions
            ])
            latent_batch = self._apply_conditioning(latent_batch, condition_batch)
        
        # Generate batch
        with torch.no_grad():
            audio_batch = self.generator(latent_batch)
        
        # Post-process and convert to list
        audios = []
        for i in range(batch_size):
            audio = audio_batch[i].squeeze()
            cond = conditions[i] if conditions else {}
            audio = self._post_process(audio, cond)
            audios.append(audio)
        
        return audios
    
    def generate_from_latent(self, latent_vector: torch.Tensor) -> torch.Tensor:
        """Generate audio from specific latent vector"""
        
        if self.generator is None:
            self._initialize_models()
        
        if len(latent_vector.shape) == 1:
            latent_vector = latent_vector.unsqueeze(0)
        
        with torch.no_grad():
            audio = self.generator(latent_vector)
        
        return audio.squeeze()
    
    def interpolate(self, 
                   latent1: torch.Tensor, 
                   latent2: torch.Tensor, 
                   steps: int = 10) -> List[torch.Tensor]:
        """Interpolate between two latent vectors"""
        
        interpolated_audios = []
        
        for i in range(steps):
            alpha = i / (steps - 1)
            interpolated_latent = (1 - alpha) * latent1 + alpha * latent2
            audio = self.generate_from_latent(interpolated_latent)
            interpolated_audios.append(audio)
        
        return interpolated_audios
    
    def _encode_conditions(self, conditions: Dict[str, Any]) -> torch.Tensor:
        """Encode conditions into vector representation"""
        
        if self.condition_embedding is None:
            self.condition_embedding = ConditionEmbedding(
                embedding_dim=64,
                latent_dim=self.latent_dim
            )
        
        # Extract condition features
        features = {}
        
        # Demographics
        demographics = conditions.get("demographics", {})
        features.update({
            "gender_male": 1.0 if demographics.get("gender") == "male" else 0.0,
            "gender_female": 1.0 if demographics.get("gender") == "female" else 0.0,
            "age_child": 1.0 if demographics.get("age_group") == "child" else 0.0,
            "age_adult": 1.0 if demographics.get("age_group") == "adult" else 0.0,
            "age_elderly": 1.0 if demographics.get("age_group") == "elderly" else 0.0
        })
        
        # Audio characteristics
        features.update({
            "pitch_low": 0.0,
            "pitch_medium": 1.0,
            "pitch_high": 0.0,
            "tempo_slow": 0.0,
            "tempo_medium": 1.0,
            "tempo_fast": 0.0
        })
        
        # Convert to tensor
        feature_vector = torch.tensor(list(features.values()), dtype=torch.float32)
        
        # Embed conditions
        condition_vector = self.condition_embedding(feature_vector.unsqueeze(0))
        
        return condition_vector.squeeze()
    
    def _apply_conditioning(self, 
                           latent_vector: torch.Tensor, 
                           condition_vector: torch.Tensor) -> torch.Tensor:
        """Apply conditioning to latent vector"""
        
        # Simple concatenation and projection
        if len(condition_vector.shape) == 1:
            condition_vector = condition_vector.unsqueeze(0)
        
        # Ensure batch dimensions match
        if latent_vector.shape[0] != condition_vector.shape[0]:
            condition_vector = condition_vector.repeat(latent_vector.shape[0], 1)
        
        # Combine latent and condition vectors
        combined = torch.cat([latent_vector, condition_vector], dim=1)
        
        # Project back to latent dimension
        projection = nn.Linear(combined.shape[1], self.latent_dim)
        conditioned_latent = projection(combined)
        
        return conditioned_latent
    
    def _post_process(self, audio: torch.Tensor, conditions: Optional[Dict[str, Any]]) -> torch.Tensor:
        """Post-process generated audio"""
        
        # Normalize
        audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
        
        # Apply privacy transformations
        if conditions:
            privacy_level = conditions.get("privacy_level", "medium")
            if privacy_level == "high":
                audio = self._apply_privacy_noise(audio)
        
        # Ensure correct length
        target_length = int(self.audio_config.duration * self.audio_config.sample_rate)
        if len(audio) != target_length:
            audio = F.interpolate(
                audio.unsqueeze(0).unsqueeze(0),
                size=target_length,
                mode='linear'
            ).squeeze()
        
        return audio
    
    def _apply_privacy_noise(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply privacy-preserving noise"""
        noise_level = 0.05
        noise = torch.randn_like(audio) * noise_level
        return audio + noise

class AudioGenerator(nn.Module):
    """Generator network for audio synthesis"""
    
    def __init__(self, latent_dim: int, output_length: int, num_channels: int = 1):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_length = output_length
        self.num_channels = num_channels
        
        # Calculate intermediate dimensions
        self.init_size = output_length // (2 ** 6)  # 6 upsampling layers
        
        # Initial projection
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size),
            nn.BatchNorm1d(128 * self.init_size),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling layers
        self.conv_blocks = nn.Sequential(
            # 128 -> 64
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            # 64 -> 32
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
            # 32 -> 16
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            
            # 16 -> 8
            nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            
            # 8 -> 4
            nn.ConvTranspose1d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True),
            
            # 4 -> 1 (output channels)
            nn.ConvTranspose1d(4, num_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Project and reshape
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size)
        
        # Generate audio
        audio = self.conv_blocks(out)
        
        # Ensure correct output length
        if audio.shape[-1] != self.output_length:
            audio = F.interpolate(audio, size=self.output_length, mode='linear')
        
        return audio

class AudioDiscriminator(nn.Module):
    """Discriminator network for audio"""
    
    def __init__(self, input_length: int, num_channels: int = 1):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, stride=2):
            return nn.Sequential(
                nn.Conv1d(in_filters, out_filters, kernel_size=4, stride=stride, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25)
            )
        
        self.model = nn.Sequential(
            discriminator_block(num_channels, 16),
            discriminator_block(16, 32),
            discriminator_block(32, 64),
            discriminator_block(64, 128),
            discriminator_block(128, 256),
            discriminator_block(256, 512)
        )
        
        # Calculate the size after convolutions
        self.adv_layer = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        out = self.model(audio)
        validity = self.adv_layer(out)
        return validity

class ConditionEmbedding(nn.Module):
    """Condition embedding network"""
    
    def __init__(self, embedding_dim: int, latent_dim: int):
        super().__init__()
        
        self.embedding = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
            nn.Tanh()
        )
    
    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        return self.embedding(conditions)
