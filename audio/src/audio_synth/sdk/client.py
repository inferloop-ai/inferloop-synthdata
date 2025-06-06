# audio_synth/sdk/client.py
"""
Core SDK client for Audio Synthetic Data Framework
"""

import torch
import torchaudio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import yaml
import json
from datetime import datetime

from ..core.generators.diffusion import DiffusionAudioGenerator
from ..core.generators.tts import TTSAudioGenerator
from ..core.generators.gan import GANAudioGenerator
from ..core.generators.vae import VAEAudioGenerator
from ..core.validators.quality import QualityValidator
from ..core.validators.privacy import PrivacyValidator
from ..core.validators.fairness import FairnessValidator
from ..core.utils.config import load_config, AudioConfig, GenerationConfig

logger = logging.getLogger(__name__)

class AudioSynthSDK:
    """Main SDK client for audio synthesis and validation"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the SDK
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        if config_path:
            self.config = load_config(config_path)
        else:
            self.config = self._get_default_config()
        
        # Initialize generators
        self.generators = {}
        self._init_generators()
        
        # Initialize validators
        self.validators = {}
        self._init_validators()
        
        logger.info("AudioSynthSDK initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "audio": {
                "sample_rate": 22050,
                "duration": 5.0,
                "channels": 1,
                "format": "wav"
            },
            "generation": {
                "default_method": "diffusion",
                "privacy_level": "medium",
                "num_samples": 1
            },
            "validation": {
                "quality_threshold": 0.7,
                "privacy_threshold": 0.8,
                "fairness_threshold": 0.75
            },
            "models": {
                "base_model_dir": "./models"
            }
        }
    
    def _init_generators(self):
        """Initialize audio generators"""
        audio_config = AudioConfig(**self.config["audio"])
        
        # Initialize available generators
        try:
            self.generators["diffusion"] = DiffusionAudioGenerator(
                sample_rate=audio_config.sample_rate
            )
        except Exception as e:
            logger.warning(f"Failed to initialize diffusion generator: {e}")
        
        try:
            self.generators["tts"] = TTSAudioGenerator(
                sample_rate=audio_config.sample_rate
            )
        except Exception as e:
            logger.warning(f"Failed to initialize TTS generator: {e}")
        
        try:
            self.generators["gan"] = GANAudioGenerator(
                sample_rate=audio_config.sample_rate
            )
        except Exception as e:
            logger.warning(f"Failed to initialize GAN generator: {e}")
        
        try:
            self.generators["vae"] = VAEAudioGenerator(
                sample_rate=audio_config.sample_rate
            )
        except Exception as e:
            logger.warning(f"Failed to initialize VAE generator: {e}")
    
    def _init_validators(self):
        """Initialize validators"""
        validation_config = self.config.get("validation", {})
        
        self.validators["quality"] = QualityValidator(validation_config)
        self.validators["privacy"] = PrivacyValidator(validation_config)
        self.validators["fairness"] = FairnessValidator(validation_config)
    
    def generate(self, 
                 method: str = "diffusion",
                 prompt: Optional[str] = None,
                 num_samples: int = 1,
                 conditions: Optional[Dict[str, Any]] = None,
                 seed: Optional[int] = None,
                 **kwargs) -> List[torch.Tensor]:
        """
        Generate synthetic audio
        
        Args:
            method: Generation method (diffusion, tts, gan, vae)
            prompt: Text prompt for generation
            num_samples: Number of samples to generate
            conditions: Generation conditions
            seed: Random seed
            **kwargs: Additional parameters
            
        Returns:
            List of generated audio tensors
        """
        if method not in self.generators:
            raise ValueError(f"Unknown generation method: {method}")
        
        generator = self.generators[method]
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        try:
            if hasattr(generator, 'generate'):
                audios, _ = generator.generate(
                    prompt=prompt,
                    num_samples=num_samples,
                    seed=seed,
                    **kwargs
                )
            else:
                # Fallback for generators without standard interface
                audios = []
                for _ in range(num_samples):
                    audio = generator(prompt=prompt, conditions=conditions, **kwargs)
                    audios.append(audio)
            
            logger.info(f"Generated {len(audios)} audio samples using {method}")
            return audios
            
        except Exception as e:
            logger.error(f"Generation failed with {method}: {e}")
            raise
    
    def validate(self,
                 audios: List[torch.Tensor],
                 metadata: Optional[List[Dict[str, Any]]] = None,
                 validators: List[str] = ["quality", "privacy", "fairness"]) -> Dict[str, List[Dict[str, float]]]:
        """
        Validate audio samples
        
        Args:
            audios: List of audio tensors to validate
            metadata: Optional metadata for each audio sample
            validators: List of validators to run
            
        Returns:
            Dictionary with validation results
        """
        if metadata is None:
            metadata = [{}] * len(audios)
        
        if len(metadata) != len(audios):
            raise ValueError("Metadata list must match audio list length")
        
        results = {}
        
        for validator_name in validators:
            if validator_name not in self.validators:
                logger.warning(f"Unknown validator: {validator_name}")
                continue
            
            validator = self.validators[validator_name]
            validator_results = []
            
            for audio, meta in zip(audios, metadata):
                try:
                    result = validator.validate(audio, meta)
                    validator_results.append(result)
                except Exception as e:
                    logger.error(f"Validation failed for {validator_name}: {e}")
                    validator_results.append({})
            
            results[validator_name] = validator_results
        
        logger.info(f"Validated {len(audios)} samples with {len(validators)} validators")
        return results
    
    def generate_and_validate(self,
                            method: str = "diffusion",
                            prompt: Optional[str] = None,
                            num_samples: int = 1,
                            validators: List[str] = ["quality", "privacy", "fairness"],
                            conditions: Optional[Dict[str, Any]] = None,
                            **kwargs) -> Dict[str, Any]:
        """
        Generate and validate audio in one call
        
        Args:
            method: Generation method
            prompt: Text prompt
            num_samples: Number of samples
            validators: Validators to run
            conditions: Generation conditions
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with audios and validation results
        """
        # Generate audio
        audios = self.generate(
            method=method,
            prompt=prompt,
            num_samples=num_samples,
            conditions=conditions,
            **kwargs
        )
        
        # Prepare metadata
        metadata = []
        for i in range(len(audios)):
            meta = {
                "method": method,
                "prompt": prompt,
                "sample_index": i,
                "generation_params": conditions or {}
            }
            metadata.append(meta)
        
        # Validate
        validation_results = self.validate(
            audios=audios,
            metadata=metadata,
            validators=validators
        )
        
        return {
            "audios": audios,
            "validation": validation_results,
            "metadata": metadata
        }
    
    def enhance_privacy(self,
                       audio: torch.Tensor,
                       privacy_level: str = "medium",
                       target_speaker: Optional[str] = None) -> torch.Tensor:
        """
        Enhance privacy of existing audio
        
        Args:
            audio: Input audio tensor
            privacy_level: Privacy enhancement level
            target_speaker: Target speaker for voice conversion
            
        Returns:
            Privacy-enhanced audio
        """
        # Apply privacy transformations
        enhanced_audio = audio.clone()
        
        if privacy_level == "high":
            # Strong privacy transformations
            enhanced_audio = self._apply_strong_privacy(enhanced_audio)
        elif privacy_level == "medium":
            # Moderate privacy transformations
            enhanced_audio = self._apply_medium_privacy(enhanced_audio)
        elif privacy_level == "low":
            # Light privacy transformations
            enhanced_audio = self._apply_light_privacy(enhanced_audio)
        
        return enhanced_audio
    
    def _apply_strong_privacy(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply strong privacy transformations"""
        # Pitch shifting
        pitch_factor = 0.7 + torch.rand(1).item() * 0.6  # 0.7 to 1.3
        
        # Add voice conversion noise
        noise = torch.randn_like(audio) * 0.05
        audio = audio + noise
        
        # Apply spectral masking
        stft = torch.stft(audio, n_fft=1024, hop_length=256, return_complex=True)
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Mask some frequency bins for privacy
        freq_mask = torch.rand_like(magnitude) > 0.1
        magnitude = magnitude * freq_mask
        
        # Reconstruct
        masked_stft = magnitude * torch.exp(1j * phase)
        audio = torch.istft(masked_stft, n_fft=1024, hop_length=256)
        
        return audio
    
    def _apply_medium_privacy(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply medium privacy transformations"""
        # Light pitch shifting
        pitch_factor = 0.9 + torch.rand(1).item() * 0.2  # 0.9 to 1.1
        
        # Add noise
        noise = torch.randn_like(audio) * 0.02
        audio = audio + noise
        
        return audio
    
    def _apply_light_privacy(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply light privacy transformations"""
        # Very light noise
        noise = torch.randn_like(audio) * 0.01
        audio = audio + noise
        
        return audio
    
    def save_audio(self,
                   audio: torch.Tensor,
                   filepath: Union[str, Path],
                   sample_rate: Optional[int] = None) -> None:
        """
        Save audio to file
        
        Args:
            audio: Audio tensor to save
            filepath: Output file path
            sample_rate: Sample rate (uses config default if None)
        """
        if sample_rate is None:
            sample_rate = self.config["audio"]["sample_rate"]
        
        # Ensure audio is 2D for torchaudio
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        
        torchaudio.save(str(filepath), audio, sample_rate)
        logger.info(f"Audio saved to {filepath}")
    
    def load_audio(self, filepath: Union[str, Path]) -> torch.Tensor:
        """
        Load audio from file
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Loaded audio tensor
        """
        audio, sample_rate = torchaudio.load(str(filepath))
        
        # Resample if necessary
        target_sr = self.config["audio"]["sample_rate"]
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            audio = resampler(audio)
        
        # Convert to mono if necessary
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        return audio.squeeze()

class AudioConfig:
    """Audio configuration class"""
    
    def __init__(self, 
                 sample_rate: int = 22050,
                 duration: float = 5.0,
                 channels: int = 1,
                 format: str = "wav",
                 bit_depth: int = 16):
        self.sample_rate = sample_rate
        self.duration = duration
        self.channels = channels
        self.format = format
        self.bit_depth = bit_depth

class GenerationConfig:
    """Generation configuration class"""
    
    def __init__(self,
                 method: str = "diffusion",
                 privacy_level: str = "medium",
                 num_samples: int = 1,
                 seed: Optional[int] = None):
        self.method = method
        self.privacy_level = privacy_level
        self.num_samples = num_samples
        self.seed = seed