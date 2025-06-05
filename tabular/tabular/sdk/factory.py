# inferloop-synthetic/sdk/factory.py
"""
Factory for creating synthetic data generators
"""

from typing import Dict, Any, Type, Optional
import logging

from .base import BaseSyntheticGenerator, SyntheticDataConfig
from .sdv_generator import SDVGenerator
from .ctgan_generator import CTGANGenerator
from .ydata_generator import YDataGenerator

logger = logging.getLogger(__name__)


class GeneratorFactory:
    """Factory for creating synthetic data generators"""
    
    _generators: Dict[str, Type[BaseSyntheticGenerator]] = {
        'sdv': SDVGenerator,
        'ctgan': CTGANGenerator,
        'ydata': YDataGenerator
    }
    
    @classmethod
    def create_generator(cls, config: SyntheticDataConfig) -> BaseSyntheticGenerator:
        """Create a generator based on configuration"""
        generator_type = config.generator_type.lower()
        
        if generator_type not in cls._generators:
            available = list(cls._generators.keys())
            raise ValueError(f"Unknown generator type: {generator_type}. Available: {available}")
        
        generator_class = cls._generators[generator_type]
        return generator_class(config)
    
    @classmethod
    def register_generator(cls, name: str, generator_class: Type[BaseSyntheticGenerator]):
        """Register a custom generator"""
        cls._generators[name] = generator_class
        logger.info(f"Registered custom generator: {name}")
    
    @classmethod
    def list_generators(cls) -> list:
        """List available generator types"""
        return list(cls._generators.keys())
    
    @classmethod
    def create_from_dict(cls, config_dict: Dict[str, Any]) -> BaseSyntheticGenerator:
        """Create generator from configuration dictionary"""
        config = SyntheticDataConfig(**config_dict)
        return cls.create_generator(config)