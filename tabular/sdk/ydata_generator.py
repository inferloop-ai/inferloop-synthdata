# inferloop-synthetic/sdk/ydata_generator.py
"""
Wrapper for YData Synthetic library
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import time
import logging

from .base import BaseSyntheticGenerator, SyntheticDataConfig, GenerationResult

logger = logging.getLogger(__name__)

try:
    from ydata_synthetic.synthesizers.regular import RegularSynthesizer
    from ydata_synthetic.synthesizers import ModelParameters
    YDATA_AVAILABLE = True
except ImportError:
    YDATA_AVAILABLE = False
    logger.warning("YData Synthetic not available. Install with: pip install ydata-synthetic")
    # Define dummy classes to avoid NameError
    RegularSynthesizer = None
    ModelParameters = None


class YDataGenerator(BaseSyntheticGenerator):
    """YData Synthetic-based generator"""
    
    MODEL_TYPES = {
        'wgan_gp': 'wgan_gp',
        'cramer_gan': 'cramer_gan',
        'dragan': 'dragan'
    }
    
    def __init__(self, config: SyntheticDataConfig):
        if not YDATA_AVAILABLE:
            raise ImportError("YData Synthetic library not available. Install with: pip install ydata-synthetic")
        
        super().__init__(config)
        
    def _create_model_parameters(self) -> ModelParameters:
        """Create YData model parameters"""
        params = ModelParameters(
            batch_size=self.config.batch_size,
            lr=self.config.learning_rate,
            noise_dim=self.config.hyperparameters.get('noise_dim', 32),
            layers_dim=self.config.hyperparameters.get('layers_dim', 128)
        )
        return params
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit YData model to training data"""
        logger.info(f"Fitting YData {self.config.model_type} model...")
        
        self.validate_data(data)
        prepared_data = self.prepare_data(data)
        
        # Create model parameters
        model_params = self._create_model_parameters()
        
        # Create synthesizer
        self.model = RegularSynthesizer(
            modelname=self.config.model_type,
            model_parameters=model_params
        )
        
        start_time = time.time()
        self.model.fit(data=prepared_data, train_arguments={'epochs': self.config.epochs})
        fit_time = time.time() - start_time
        
        self.is_fitted = True
        logger.info(f"Model fitted in {fit_time:.2f} seconds")
    
    def generate(self, num_samples: Optional[int] = None) -> GenerationResult:
        """Generate synthetic data using fitted YData model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating data")
        
        num_samples = num_samples or self.config.num_samples
        
        logger.info(f"Generating {num_samples} synthetic samples...")
        
        start_time = time.time()
        synthetic_data = self.model.sample(n_samples=num_samples)
        generation_time = time.time() - start_time
        
        logger.info(f"Generated {len(synthetic_data)} samples in {generation_time:.2f} seconds")
        
        # Create result
        result = GenerationResult(
            synthetic_data=synthetic_data,
            config=self.config,
            generation_time=generation_time,
            model_info=self.get_model_info()
        )
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the fitted YData model"""
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        info = {
            'library': 'ydata_synthetic',
            'model_type': self.config.model_type,
            'hyperparameters': self.config.hyperparameters,
            'epochs': self.config.epochs,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate
        }
        
        return info
