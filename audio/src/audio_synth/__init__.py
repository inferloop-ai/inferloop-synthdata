# audio_synth/__init__.py
"""
Audio Synthetic Data Framework

A comprehensive framework for generating and validating synthetic audio data
with privacy and fairness guarantees.
"""

__version__ = "1.0.0"
__author__ = "Audio Synth Team"
__email__ = "team@audiosynth.ai"

from .sdk.client import AudioSynthSDK
from .core.utils.config import load_config, get_default_config

# Main exports
__all__ = [
    "AudioSynthSDK",
    "load_config", 
    "get_default_config",
    "__version__"
]

# Convenience imports
def create_sdk(config_path=None):
    """Create an AudioSynthSDK instance"""
    return AudioSynthSDK(config_path)

