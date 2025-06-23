import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List
import logging

try:
    from sdv.tabular import GaussianCopula, CTGAN, CopulaGAN, TVAE
    from sdv.metadata import SingleTableMetadata
except ImportError:
    logging.warning("SDV package not installed. Install with: pip install sdv")

from .base import BaseSyntheticGenerator

class SDVGenerator(BaseSyntheticGenerator):
    """Implementation of the SyntheticDataGenerator using SDV library.
    
    This generator uses the SDV (Synthetic Data Vault) library to create
    synthetic tabular data. It supports multiple models including:
    - GaussianCopula (default)
    - CTGAN
    - CopulaGAN
    - TVAE
    """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.model_type = "gaussian_copula"  # default
        self.metadata = None
        self.config = {
            "model_parameters": {},
        }
        
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the generator with the provided settings.
        
        Args:
            config: Dictionary containing configuration parameters
                - model_type: Type of SDV model to use (gaussian_copula, ctgan, copulagan, tvae)
                - model_parameters: Parameters to pass to the model constructor
        """
        self.config.update(config)
        
        # Set model type
        if "model_type" in config:
            self.model_type = config["model_type"]
            
    def _create_model(self):
        """Create the appropriate SDV model based on configuration"""
        model_params = self.config.get("model_parameters", {})
        
        if self.model_type == "gaussian_copula":
            return GaussianCopula(**model_params)
        elif self.model_type == "ctgan":
            return CTGAN(**model_params)
        elif self.model_type == "copulagan":
            return CopulaGAN(**model_params)
        elif self.model_type == "tvae":
            return TVAE(**model_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _create_metadata(self, data: pd.DataFrame) -> SingleTableMetadata:
        """Create and populate metadata for the dataset"""
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)
        
        # Apply any custom metadata configurations
        if "metadata" in self.config:
            for column, properties in self.config["metadata"].items():
                if column in metadata.columns:
                    for prop, value in properties.items():
                        metadata.update_column(column, **{prop: value})
        
        return metadata
        
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the model to the provided data.
        
        Args:
            data: DataFrame containing the training data
        """
        self.data_columns = list(data.columns)
        self.metadata = self._create_metadata(data)
        
        # Create and fit the model
        self.model = self._create_model()
        self.model.fit(data, metadata=self.metadata)
        
    def generate(self, num_rows: int = 100) -> pd.DataFrame:
        """Generate synthetic data.
        
        Args:
            num_rows: Number of synthetic rows to generate
            
        Returns:
            DataFrame containing synthetic data
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() before generate()")
            
        return self.model.sample(num_rows)
    
    def save(self, path: str) -> None:
        """Save the model to disk.
        
        Args:
            path: Path to save the model to
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() before save()")
            
        self.model.save(path)
    
    def load(self, path: str) -> None:
        """Load a saved model from disk.
        
        Args:
            path: Path to load the model from
        """
        model_map = {
            "gaussian_copula": GaussianCopula,
            "ctgan": CTGAN,
            "copulagan": CopulaGAN,
            "tvae": TVAE
        }
        
        model_class = model_map.get(self.model_type)
        if model_class is None:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        self.model = model_class.load(path)
