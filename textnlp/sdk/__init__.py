# =====================================
# SDK CORE COMPONENTS
# =====================================

# sdk/__init__.py
"""
Inferloop NLP Synthetic Data Generation SDK
"""
__version__ = "0.1.0"

from .base_generator import BaseGenerator
from .llm_gpt2 import GPT2Generator
from .langchain_template import LangChainTemplate
from .formatter import DataFormatter

__all__ = ["BaseGenerator", "GPT2Generator", "LangChainTemplate", "DataFormatter"]


# sdk/base_generator.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class BaseGenerator(ABC):
    """Abstract base class for all text generators"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
        self._setup_model()
    
    @abstractmethod
    def _setup_model(self):
        """Initialize the specific model"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, max_length: int = 100, **kwargs) -> str:
        """Generate text from a prompt"""
        pass
    
    @abstractmethod
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts"""
        pass
    
    def validate_input(self, prompt: str) -> bool:
        """Validate input prompt"""
        if not prompt or not isinstance(prompt, str):
            logger.warning("Invalid prompt provided")
            return False
        return True





