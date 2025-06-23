# inferloop-synthetic/sdk/ctgan_generator.py
"""
Wrapper for CTGAN library
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import time
import logging

from .base import BaseSyntheticGenerator, SyntheticDataConfig, GenerationResult

logger = logging.getLogger(__name__)

try:
    from ctgan import CTGAN
    from ctgan.data_transformer import DataTransformer
    CTGAN_AVAILABLE = True
except ImportError:
    CTGAN_AVAILABLE = False
    logger.warning("CTGAN not available. Install with: pip install ctgan")
    # Define dummy classes to avoid NameError
    CTGAN = None
    DataTransformer = None


class CTGANGenerator(BaseSyntheticGenerator):
    """CTGAN-based synthetic data generator"""
    
    def __init__(self, config: SyntheticDataConfig):
        if not CTGAN_AVAILABLE:
            raise ImportError("CTGAN library not available. Install with: pip install ctgan")
        
        super().__init__(config)
        self.data_transformer = None
        self.transformed_data = None
        self.discrete_columns = []
        
    def _prepare_discrete_columns(self, data: pd.DataFrame) -> list:
        """Prepare list of discrete columns for CTGAN"""
        discrete_columns = []
        
        # Add categorical columns
        if self.config.categorical_columns:
            discrete_columns.extend([col for col in self.config.categorical_columns if col in data.columns])
        
        # Auto-detect categorical columns if not specified
        if not discrete_columns:
            for col in data.columns:
                if data[col].dtype == 'object' or data[col].dtype.name == 'category':
                    discrete_columns.append(col)
                elif data[col].dtype in ['int64', 'int32', 'int16', 'int8']:
                    # Check if integer column has few unique values (likely categorical)
                    unique_ratio = len(data[col].unique()) / len(data[col])
                    if unique_ratio < 0.05:  # Less than 5% unique values
                        discrete_columns.append(col)
        
        return discrete_columns
    
    def _create_model(self) -> CTGAN:
        """Create CTGAN model based on configuration"""
        hyperparams = self.config.hyperparameters.copy()
        
        # Extract CTGAN-specific parameters
        generator_dim = hyperparams.pop('generator_dim', (256, 256))
        discriminator_dim = hyperparams.pop('discriminator_dim', (256, 256))
        generator_lr = hyperparams.pop('generator_lr', 2e-4)
        discriminator_lr = hyperparams.pop('discriminator_lr', 2e-4)
        discriminator_steps = hyperparams.pop('discriminator_steps', 1)
        log_frequency = hyperparams.pop('log_frequency', True)
        verbose = hyperparams.pop('verbose', self.config.verbose)
        epochs = self.config.epochs or hyperparams.pop('epochs', 300)
        batch_size = self.config.batch_size or hyperparams.pop('batch_size', 500)
        pac = hyperparams.pop('pac', 10)
        cuda = hyperparams.pop('cuda', False)
        
        return CTGAN(
            epochs=epochs,
            batch_size=batch_size,
            generator_dim=generator_dim,
            discriminator_dim=discriminator_dim,
            generator_lr=generator_lr,
            discriminator_lr=discriminator_lr,
            discriminator_steps=discriminator_steps,
            log_frequency=log_frequency,
            verbose=verbose,
            pac=pac,
            cuda=cuda
        )
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit CTGAN model to training data"""
        logger.info("Fitting CTGAN model...")
        
        self.validate_data(data)
        prepared_data = self.prepare_data(data)
        
        # Identify discrete columns
        self.discrete_columns = self._prepare_discrete_columns(prepared_data)
        logger.info(f"Identified discrete columns: {self.discrete_columns}")
        
        # Create and fit model
        self.model = self._create_model()
        
        start_time = time.time()
        self.model.fit(prepared_data, discrete_columns=self.discrete_columns)
        fit_time = time.time() - start_time
        
        self.is_fitted = True
        logger.info(f"Model fitted in {fit_time:.2f} seconds")
    
    def generate(self, num_samples: Optional[int] = None) -> GenerationResult:
        """Generate synthetic data using fitted CTGAN model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating data")
        
        num_samples = num_samples or self.config.num_samples
        
        logger.info(f"Generating {num_samples} synthetic samples...")
        
        start_time = time.time()
        synthetic_data = self.model.sample(num_samples)
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
        """Get information about the fitted CTGAN model"""
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        info = {
            'library': 'ctgan',
            'model_type': 'ctgan',
            'discrete_columns': self.discrete_columns,
            'hyperparameters': {
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                **self.config.hyperparameters
            }
        }
        
        # Add model architecture info if available
        if hasattr(self.model, '_generator'):
            info['generator_layers'] = str(self.model._generator)
        if hasattr(self.model, '_discriminator'):
            info['discriminator_layers'] = str(self.model._discriminator)
        
        return info
    
    def save_model(self, path: str) -> None:
        """Save CTGAN model to disk"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load CTGAN model from disk"""
        self.model = CTGAN.load(path)
        self.is_fitted = True
        logger.info(f"Model loaded from {path}")