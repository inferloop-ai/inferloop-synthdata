# inferloop-synthetic/sdk/sdv_generator.py
"""
Wrapper for Synthetic Data Vault (SDV) library
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import time
import logging

from .base import BaseSyntheticGenerator, SyntheticDataConfig, GenerationResult

logger = logging.getLogger(__name__)

try:
    from sdv.single_table import GaussianCopulaSynthesizer, CopulaGANSynthesizer, CTGANSynthesizer, TVAESynthesizer
    from sdv.metadata import SingleTableMetadata
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False
    logger.warning("SDV not available. Install with: pip install sdv")
    # Define dummy classes to avoid NameError
    SingleTableMetadata = None
    GaussianCopulaSynthesizer = None
    CopulaGANSynthesizer = None
    CTGANSynthesizer = None
    TVAESynthesizer = None


class SDVGenerator(BaseSyntheticGenerator):
    """SDV-based synthetic data generator"""
    
    MODEL_TYPES = {
        'gaussian_copula': 'GaussianCopulaSynthesizer',
        'copula_gan': 'CopulaGANSynthesizer', 
        'ctgan': 'CTGANSynthesizer',
        'tvae': 'TVAESynthesizer'
    }
    
    def __init__(self, config: SyntheticDataConfig):
        if not SDV_AVAILABLE:
            raise ImportError("SDV library not available. Install with: pip install sdv")
        
        super().__init__(config)
        self.metadata = None
        
    def _create_metadata(self, data: pd.DataFrame) -> 'SingleTableMetadata':
        """Create SDV metadata from dataframe"""
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)
        
        # Override with user-specified column types
        if self.config.categorical_columns:
            for col in self.config.categorical_columns:
                if col in data.columns:
                    metadata.update_column(col, sdtype='categorical')
        
        if self.config.continuous_columns:
            for col in self.config.continuous_columns:
                if col in data.columns:
                    metadata.update_column(col, sdtype='numerical')
        
        if self.config.datetime_columns:
            for col in self.config.datetime_columns:
                if col in data.columns:
                    metadata.update_column(col, sdtype='datetime')
        
        # Set primary key if specified
        if self.config.primary_key and self.config.primary_key in data.columns:
            metadata.set_primary_key(self.config.primary_key)
        
        return metadata
    
    def _create_model(self) -> Any:
        """Create SDV model based on configuration"""
        model_type = self.config.model_type
        hyperparams = self.config.hyperparameters.copy()
        
        if model_type == 'gaussian_copula':
            return GaussianCopulaSynthesizer(
                metadata=self.metadata,
                **hyperparams
            )
        elif model_type == 'copula_gan':
            return CopulaGANSynthesizer(
                metadata=self.metadata,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                **hyperparams
            )
        elif model_type == 'ctgan':
            return CTGANSynthesizer(
                metadata=self.metadata,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                **hyperparams
            )
        elif model_type == 'tvae':
            return TVAESynthesizer(
                metadata=self.metadata,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                **hyperparams
            )
        else:
            raise ValueError(f"Unsupported SDV model type: {model_type}")
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit SDV model to training data"""
        logger.info(f"Fitting SDV {self.config.model_type} model...")
        
        self.validate_data(data)
        prepared_data = self.prepare_data(data)
        
        # Create metadata
        self.metadata = self._create_metadata(prepared_data)
        
        # Create and fit model
        self.model = self._create_model()
        
        start_time = time.time()
        self.model.fit(prepared_data)
        fit_time = time.time() - start_time
        
        self.is_fitted = True
        logger.info(f"Model fitted in {fit_time:.2f} seconds")
    
    def generate(self, num_samples: Optional[int] = None) -> GenerationResult:
        """Generate synthetic data using fitted SDV model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating data")
        
        num_samples = num_samples or self.config.num_samples
        
        logger.info(f"Generating {num_samples} synthetic samples...")
        
        start_time = time.time()
        synthetic_data = self.model.sample(num_rows=num_samples)
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
        """Get information about the fitted SDV model"""
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        info = {
            'library': 'sdv',
            'model_type': self.config.model_type,
            'model_class': self.MODEL_TYPES.get(self.config.model_type, 'Unknown'),
            'metadata': self.metadata.to_dict() if self.metadata else None,
            'hyperparameters': self.config.hyperparameters
        }
        
        # Add model-specific info
        if hasattr(self.model, 'get_parameters'):
            info['model_parameters'] = self.model.get_parameters()
        
        return info