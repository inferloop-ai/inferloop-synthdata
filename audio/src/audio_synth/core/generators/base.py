"""
Base class for audio generation models
"""

import torch
from typing import Dict, List, Optional, Any, Tuple, Union
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class AudioGenerator(ABC):
    """Base class for all audio generation models"""
    
    def __init__(self, 
                 sample_rate: int = 22050,
                 device: Optional[torch.device] = None):
        """
        Initialize the generator
        
        Args:
            sample_rate: Target sample rate for generated audio
            device: Torch device to use (defaults to CUDA if available)
        """
        self.sample_rate = sample_rate
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing {self.__class__.__name__} on {self.device}")
    
    @abstractmethod
    def generate(self,
                prompt: Optional[str] = None,
                num_samples: int = 1,
                seed: Optional[int] = None,
                **kwargs) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Generate audio samples
        
        Args:
            prompt: Text prompt to condition generation
            num_samples: Number of samples to generate
            seed: Random seed for reproducibility
            **kwargs: Additional generator-specific parameters
            
        Returns:
            Tuple containing:
                - List of audio tensors
                - Dict with metadata about the generation
        """
        pass
    
    def __call__(self, *args, **kwargs) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """Convenience wrapper around generate method"""
        return self.generate(*args, **kwargs)
        
    @property
    def model_info(self) -> Dict[str, Any]:
        """Return information about the model"""
        return {
            "type": self.__class__.__name__,
            "sample_rate": self.sample_rate,
            "device": str(self.device)
        }
    
    def load_checkpoints(self, checkpoint_path: str) -> None:
        """
        Load model weights from checkpoint
        
        Args:
            checkpoint_path: Path to model checkpoint file
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement checkpoint loading")

    def train(self, *args, **kwargs) -> None:
        """
        Train the model (if supported)
        
        Args:
            *args, **kwargs: Training parameters
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support training")

    def preprocess_input(self, input_data: Any) -> Any:
        """
        Preprocess input data for the model
        
        Args:
            input_data: Raw input data
            
        Returns:
            Processed input data ready for the model
        """
        return input_data