# audio_synth/core/generators/__init__.py
"""
Audio generation modules
"""

from .diffusion import DiffusionAudioGenerator
from .tts import TTSAudioGenerator
from .gan import GANAudioGenerator  
from .vae import VAEAudioGenerator
from .base import AudioGenerator

__all__ = [
    "AudioGenerator",
    "DiffusionAudioGenerator", 
    "TTSAudioGenerator",
    "GANAudioGenerator",
    "VAEAudioGenerator"
]
