# audio_synth/core/utils/config.py
"""
Configuration utilities for Audio Synthetic Data Framework
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class AudioConfig:
    """Audio configuration parameters"""
    sample_rate: int = 22050
    duration: float = 5.0
    channels: int = 1
    format: str = "wav"
    bit_depth: int = 16
    
    def __post_init__(self):
        """Validate audio configuration"""
        if self.sample_rate < 8000 or self.sample_rate > 48000:
            raise ValueError(f"Invalid sample rate: {self.sample_rate}")
        if self.duration <= 0:
            raise ValueError(f"Invalid duration: {self.duration}")
        if self.channels not in [1, 2]:
            raise ValueError(f"Invalid channels: {self.channels}")

@dataclass
class GenerationConfig:
    """Generation configuration parameters"""
    default_method: str = "diffusion"
    privacy_level: str = "medium"
    num_samples: int = 1
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate generation configuration"""
        valid_methods = ["diffusion", "gan", "vae", "tts", "vocoder"]
        if self.default_method not in valid_methods:
            raise ValueError(f"Invalid method: {self.default_method}")
        
        valid_privacy = ["low", "medium", "high"]
        if self.privacy_level not in valid_privacy:
            raise ValueError(f"Invalid privacy level: {self.privacy_level}")

@dataclass
class ValidationConfig:
    """Validation configuration parameters"""
    quality_threshold: float = 0.7
    privacy_threshold: float = 0.8
    fairness_threshold: float = 0.75
    protected_attributes: list = None
    
    def __post_init__(self):
        """Initialize default protected attributes"""
        if self.protected_attributes is None:
            self.protected_attributes = ["gender", "age", "accent", "language"]

@dataclass
class ModelConfig:
    """Model configuration parameters"""
    base_model_dir: str = "./models"
    diffusion_model_path: Optional[str] = None
    tts_model_path: Optional[str] = None
    gan_model_path: Optional[str] = None
    vae_model_path: Optional[str] = None
    
    def __post_init__(self):
        """Set default model paths"""
        base_path = Path(self.base_model_dir)
        
        if self.diffusion_model_path is None:
            self.diffusion_model_path = str(base_path / "diffusion_model.pt")
        if self.tts_model_path is None:
            self.tts_model_path = str(base_path / "tts_model.pt")
        if self.gan_model_path is None:
            self.gan_model_path = str(base_path / "gan_model.pt")
        if self.vae_model_path is None:
            self.vae_model_path = str(base_path / "vae_model.pt")

class Config:
    """Main configuration class"""
    
    def __init__(self, 
                 audio: Optional[AudioConfig] = None,
                 generation: Optional[GenerationConfig] = None,
                 validation: Optional[ValidationConfig] = None,
                 models: Optional[ModelConfig] = None):
        self.audio = audio or AudioConfig()
        self.generation = generation or GenerationConfig()
        self.validation = validation or ValidationConfig()
        self.models = models or ModelConfig()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary"""
        audio_config = AudioConfig(**config_dict.get("audio", {}))
        generation_config = GenerationConfig(**config_dict.get("generation", {}))
        validation_config = ValidationConfig(**config_dict.get("validation", {}))
        model_config = ModelConfig(**config_dict.get("models", {}))
        
        return cls(
            audio=audio_config,
            generation=generation_config,
            validation=validation_config,
            models=model_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "audio": self.audio.__dict__,
            "generation": self.generation.__dict__,
            "validation": self.validation.__dict__,
            "models": self.models.__dict__
        }

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return get_default_config()
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return get_default_config()

def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved configuration to {config_path}")
        
    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {e}")

def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        "audio": {
            "sample_rate": 22050,
            "duration": 5.0,
            "channels": 1,
            "format": "wav",
            "bit_depth": 16
        },
        "generation": {
            "default_method": "diffusion",
            "privacy_level": "medium",
            "num_samples": 1,
            "seed": None
        },
        "validation": {
            "quality_threshold": 0.7,
            "privacy_threshold": 0.8,
            "fairness_threshold": 0.75,
            "protected_attributes": ["gender", "age", "accent", "language"]
        },
        "models": {
            "base_model_dir": "./models",
            "diffusion_model_path": None,
            "tts_model_path": None,
            "gan_model_path": None,
            "vae_model_path": None
        },
        "output": {
            "default_format": "wav",
            "normalize": True,
            "add_metadata": True
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "max_file_size": 100,
            "max_samples_per_request": 100
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }

def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration
    """
    merged = {}
    
    for config in configs:
        merged = _deep_merge(merged, config)
    
    return merged

def _deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Create Config object to validate
        Config.from_dict(config)
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

def get_config_from_env() -> Dict[str, Any]:
    """
    Get configuration from environment variables
    
    Returns:
        Configuration dictionary
    """
    config = get_default_config()
    
    # Override with environment variables
    env_mappings = {
        "AUDIO_SYNTH_SAMPLE_RATE": ("audio", "sample_rate", int),
        "AUDIO_SYNTH_DURATION": ("audio", "duration", float),
        "AUDIO_SYNTH_METHOD": ("generation", "default_method", str),
        "AUDIO_SYNTH_PRIVACY_LEVEL": ("generation", "privacy_level", str),
        "AUDIO_SYNTH_MODEL_DIR": ("models", "base_model_dir", str),
        "AUDIO_SYNTH_API_HOST": ("api", "host", str),
        "AUDIO_SYNTH_API_PORT": ("api", "port", int),
        "AUDIO_SYNTH_LOG_LEVEL": ("logging", "level", str)
    }
    
    for env_var, (section, key, type_func) in env_mappings.items():
        if env_var in os.environ:
            try:
                config[section][key] = type_func(os.environ[env_var])
            except (ValueError, KeyError) as e:
                logger.warning(f"Invalid environment variable {env_var}: {e}")
    
    return config