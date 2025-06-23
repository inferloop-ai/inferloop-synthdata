# inferloop-synthetic/sdk/base.py
"""
Base classes and interfaces for synthetic data generation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Callable
import pandas as pd
import numpy as np
from pathlib import Path
import time
import logging

from .progress import ProgressTracker, ProgressStage, ProgressInfo, ProgressMixin, with_progress

logger = logging.getLogger(__name__)


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation"""
    
    # Core settings
    generator_type: str  # 'sdv', 'ctgan', 'ydata', etc.
    model_type: str     # 'gaussian_copula', 'ctgan', 'wgan_gp', etc.
    
    # Data settings
    num_samples: int = 1000
    target_columns: Optional[List[str]] = None
    categorical_columns: Optional[List[str]] = None
    continuous_columns: Optional[List[str]] = None
    datetime_columns: Optional[List[str]] = None
    
    # Model hyperparameters
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Training settings
    epochs: int = 300
    batch_size: int = 500
    learning_rate: float = 2e-4
    
    # Quality settings
    validate_output: bool = True
    quality_threshold: float = 0.8
    
    # Metadata
    primary_key: Optional[str] = None
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    
    # Progress tracking
    progress_callback: Optional[Callable[[ProgressInfo], None]] = None
    enable_progress: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'generator_type': self.generator_type,
            'model_type': self.model_type,
            'num_samples': self.num_samples,
            'target_columns': self.target_columns,
            'categorical_columns': self.categorical_columns,
            'continuous_columns': self.continuous_columns,
            'datetime_columns': self.datetime_columns,
            'hyperparameters': self.hyperparameters,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'validate_output': self.validate_output,
            'quality_threshold': self.quality_threshold,
            'primary_key': self.primary_key,
            'constraints': self.constraints
        }


@dataclass
class GenerationResult:
    """Result of synthetic data generation"""
    
    # Generated data
    synthetic_data: pd.DataFrame
    
    # Metadata
    config: SyntheticDataConfig
    generation_time: float
    model_info: Dict[str, Any]
    
    # Quality metrics
    quality_scores: Dict[str, float] = field(default_factory=dict)
    validation_passed: bool = True
    validation_errors: List[str] = field(default_factory=list)
    
    # Training metrics
    training_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def save(self, filepath: Union[str, Path], include_metadata: bool = True):
        """Save synthetic data and metadata"""
        filepath = Path(filepath)
        
        # Save main data
        if filepath.suffix.lower() == '.csv':
            self.synthetic_data.to_csv(filepath, index=False)
        elif filepath.suffix.lower() == '.parquet':
            self.synthetic_data.to_parquet(filepath, index=False)
        elif filepath.suffix.lower() == '.json':
            self.synthetic_data.to_json(filepath, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        # Save metadata if requested
        if include_metadata:
            metadata_path = filepath.with_suffix('.metadata.json')
            import json
            metadata = {
                'config': self.config.to_dict(),
                'generation_time': self.generation_time,
                'model_info': self.model_info,
                'quality_scores': self.quality_scores,
                'validation_passed': self.validation_passed,
                'validation_errors': self.validation_errors,
                'training_metrics': self.training_metrics
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)


class BaseSyntheticGenerator(ABC, ProgressMixin):
    """Abstract base class for synthetic data generators"""
    
    def __init__(self, config: SyntheticDataConfig):
        super().__init__()
        self.config = config
        self.model = None
        self.is_fitted = False
        self.metadata = {}
        
        # Set up progress tracking
        if config.enable_progress and config.progress_callback:
            self.set_progress_callback(config.progress_callback)
        
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the synthetic data model to training data"""
        pass
    
    @abstractmethod
    def generate(self, num_samples: Optional[int] = None) -> GenerationResult:
        """Generate synthetic data"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the fitted model"""
        pass
    
    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data format and quality"""
        if data.empty:
            raise ValueError("Input data cannot be empty")
        
        # Check for minimum samples
        if len(data) < 10:
            logger.warning("Very small dataset (< 10 samples) may not generate quality synthetic data")
        
        # Check for missing values
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_pct > 0.5:
            logger.warning(f"High percentage of missing values: {missing_pct:.1%}")
        
        # Validate column types match configuration
        if self.config.categorical_columns:
            for col in self.config.categorical_columns:
                if col not in data.columns:
                    raise ValueError(f"Categorical column '{col}' not found in data")
        
        if self.config.continuous_columns:
            for col in self.config.continuous_columns:
                if col not in data.columns:
                    raise ValueError(f"Continuous column '{col}' not found in data")
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for training (preprocessing)"""
        prepared_data = data.copy()
        
        # Handle missing values
        for col in prepared_data.columns:
            if prepared_data[col].dtype in ['object', 'category']:
                prepared_data[col] = prepared_data[col].fillna('Unknown')
            else:
                prepared_data[col] = prepared_data[col].fillna(prepared_data[col].mean())
        
        return prepared_data
    
    def fit_generate(self, data: pd.DataFrame, num_samples: Optional[int] = None) -> GenerationResult:
        """Convenience method to fit and generate in one step"""
        self.fit(data)
        return self.generate(num_samples)
