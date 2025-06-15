"""
Time series data generators for the TSIOT Python SDK.

This module provides high-level generator classes that simplify the process
of generating synthetic time series data using various algorithms.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import logging

from .client import TSIOTClient, AsyncTSIOTClient
from .timeseries import TimeSeries, TimeSeriesMetadata, Frequency, create_time_series
from .utils import (
    GenerationError,
    ValidationError,
    validate_positive_number,
    validate_non_negative_number,
    validate_list_not_empty,
    validate_string_not_empty,
    get_logger
)


class BaseGenerator(ABC):
    """
    Abstract base class for time series generators.
    
    All generator classes inherit from this base and implement
    the generate method with algorithm-specific parameters.
    """
    
    def __init__(self, client: Union[TSIOTClient, AsyncTSIOTClient], logger: logging.Logger = None):
        """
        Initialize the generator.
        
        Args:
            client: TSIOT client instance
            logger: Logger instance
        """
        self.client = client
        self.logger = logger or get_logger(__name__)
        self._generator_type = self.__class__.__name__.replace("Generator", "").lower()
    
    @abstractmethod
    def generate(self, **kwargs) -> TimeSeries:
        """
        Generate time series data.
        
        This method must be implemented by each generator subclass
        with algorithm-specific parameters.
        
        Returns:
            Generated time series data
        """
        pass
    
    def _build_request(
        self,
        length: int,
        parameters: Dict[str, Any],
        metadata: Dict[str, Any] = None,
        seed: int = None
    ) -> Dict[str, Any]:
        """
        Build a generation request dictionary.
        
        Args:
            length: Number of data points to generate
            parameters: Algorithm-specific parameters
            metadata: Time series metadata
            seed: Random seed for reproducibility
        
        Returns:
            Request dictionary
        """
        request = {
            "type": self._generator_type,
            "length": validate_positive_number(length, "length"),
            "parameters": parameters or {}
        }
        
        if metadata:
            request["metadata"] = metadata
        
        if seed is not None:
            request["seed"] = int(seed)
        
        return request
    
    def _validate_common_params(
        self,
        length: int,
        series_id: str = None,
        name: str = None
    ) -> Dict[str, Any]:
        """
        Validate common generation parameters.
        
        Args:
            length: Number of data points
            series_id: Series identifier
            name: Series name
        
        Returns:
            Validated metadata dictionary
        """
        validate_positive_number(length, "length")
        
        metadata = {}
        if series_id:
            metadata["series_id"] = validate_string_not_empty(series_id, "series_id")
        if name:
            metadata["name"] = validate_string_not_empty(name, "name")
        
        return metadata


class TimeSeriesGenerator(BaseGenerator):
    """
    Main time series generator that supports multiple algorithms.
    
    This is a high-level generator that provides convenient methods
    for generating time series using different algorithms.
    
    Example:
        ```python
        generator = TimeSeriesGenerator(client)
        
        # Generate ARIMA time series
        ts = generator.generate_arima(
            length=1000,
            ar_params=[0.5, -0.3],
            ma_params=[0.2],
            trend="linear"
        )
        
        # Generate LSTM time series
        ts = generator.generate_lstm(
            length=1000,
            sequence_length=50,
            hidden_size=64
        )
        ```
    """
    
    def __init__(self, client: Union[TSIOTClient, AsyncTSIOTClient], logger: logging.Logger = None):
        super().__init__(client, logger)
        self._generator_type = "auto"  # Let the server choose the best algorithm
    
    def generate(
        self,
        algorithm: str,
        length: int,
        parameters: Dict[str, Any] = None,
        series_id: str = None,
        name: str = None,
        seed: int = None,
        **kwargs
    ) -> TimeSeries:
        """
        Generate time series using specified algorithm.
        
        Args:
            algorithm: Generation algorithm to use
            length: Number of data points to generate
            parameters: Algorithm-specific parameters
            series_id: Series identifier
            name: Series name
            seed: Random seed for reproducibility
            **kwargs: Additional parameters
        
        Returns:
            Generated time series data
        """
        metadata = self._validate_common_params(length, series_id, name)
        
        # Merge additional kwargs into parameters
        all_parameters = parameters or {}
        all_parameters.update(kwargs)
        
        request = {
            "type": algorithm,
            "length": length,
            "parameters": all_parameters,
            "metadata": metadata
        }
        
        if seed is not None:
            request["seed"] = seed
        
        try:
            return self.client.generate(request)
        except Exception as e:
            raise GenerationError(
                f"Failed to generate {algorithm} time series: {str(e)}",
                generator_type=algorithm
            )
    
    def generate_arima(
        self,
        length: int,
        ar_params: List[float] = None,
        ma_params: List[float] = None,
        d: int = 0,
        trend: str = "none",
        seasonal_periods: int = None,
        noise_std: float = 1.0,
        series_id: str = None,
        name: str = None,
        seed: int = None
    ) -> TimeSeries:
        """
        Generate ARIMA time series.
        
        Args:
            length: Number of data points to generate
            ar_params: Autoregressive parameters
            ma_params: Moving average parameters
            d: Degree of differencing
            trend: Trend type ("none", "linear", "quadratic")
            seasonal_periods: Number of periods for seasonality
            noise_std: Standard deviation of noise
            series_id: Series identifier
            name: Series name
            seed: Random seed for reproducibility
        
        Returns:
            Generated ARIMA time series
        """
        parameters = {
            "ar_params": ar_params or [0.5],
            "ma_params": ma_params or [],
            "d": validate_non_negative_number(d, "d"),
            "trend": trend,
            "noise_std": validate_positive_number(noise_std, "noise_std")
        }
        
        if seasonal_periods is not None:
            parameters["seasonal_periods"] = validate_positive_number(seasonal_periods, "seasonal_periods")
        
        return self.generate(
            algorithm="arima",
            length=length,
            parameters=parameters,
            series_id=series_id,
            name=name,
            seed=seed
        )
    
    def generate_lstm(
        self,
        length: int,
        sequence_length: int = 50,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        epochs: int = 100,
        pattern: str = "random",
        series_id: str = None,
        name: str = None,
        seed: int = None
    ) -> TimeSeries:
        """
        Generate LSTM time series.
        
        Args:
            length: Number of data points to generate
            sequence_length: Length of input sequences
            hidden_size: Hidden layer size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            pattern: Pattern type to learn
            series_id: Series identifier
            name: Series name
            seed: Random seed for reproducibility
        
        Returns:
            Generated LSTM time series
        """
        parameters = {
            "sequence_length": validate_positive_number(sequence_length, "sequence_length"),
            "hidden_size": validate_positive_number(hidden_size, "hidden_size"),
            "num_layers": validate_positive_number(num_layers, "num_layers"),
            "dropout": validate_non_negative_number(dropout, "dropout"),
            "learning_rate": validate_positive_number(learning_rate, "learning_rate"),
            "epochs": validate_positive_number(epochs, "epochs"),
            "pattern": pattern
        }
        
        return self.generate(
            algorithm="lstm",
            length=length,
            parameters=parameters,
            series_id=series_id,
            name=name,
            seed=seed
        )
    
    def generate_gru(
        self,
        length: int,
        sequence_length: int = 50,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        epochs: int = 100,
        pattern: str = "random",
        series_id: str = None,
        name: str = None,
        seed: int = None
    ) -> TimeSeries:
        """
        Generate GRU time series.
        
        Args:
            length: Number of data points to generate
            sequence_length: Length of input sequences
            hidden_size: Hidden layer size
            num_layers: Number of GRU layers
            dropout: Dropout rate
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            pattern: Pattern type to learn
            series_id: Series identifier
            name: Series name
            seed: Random seed for reproducibility
        
        Returns:
            Generated GRU time series
        """
        parameters = {
            "sequence_length": validate_positive_number(sequence_length, "sequence_length"),
            "hidden_size": validate_positive_number(hidden_size, "hidden_size"),
            "num_layers": validate_positive_number(num_layers, "num_layers"),
            "dropout": validate_non_negative_number(dropout, "dropout"),
            "learning_rate": validate_positive_number(learning_rate, "learning_rate"),
            "epochs": validate_positive_number(epochs, "epochs"),
            "pattern": pattern
        }
        
        return self.generate(
            algorithm="gru",
            length=length,
            parameters=parameters,
            series_id=series_id,
            name=name,
            seed=seed
        )
    
    def generate_timegan(
        self,
        length: int,
        hidden_dim: int = 24,
        num_layers: int = 3,
        epochs: int = 10000,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        gamma: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        series_id: str = None,
        name: str = None,
        seed: int = None
    ) -> TimeSeries:
        """
        Generate TimeGAN time series.
        
        Args:
            length: Number of data points to generate
            hidden_dim: Hidden dimension size
            num_layers: Number of layers
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            gamma: Loss balancing parameter
            beta1: Adam optimizer beta1
            beta2: Adam optimizer beta2
            series_id: Series identifier
            name: Series name
            seed: Random seed for reproducibility
        
        Returns:
            Generated TimeGAN time series
        """
        parameters = {
            "hidden_dim": validate_positive_number(hidden_dim, "hidden_dim"),
            "num_layers": validate_positive_number(num_layers, "num_layers"),
            "epochs": validate_positive_number(epochs, "epochs"),
            "batch_size": validate_positive_number(batch_size, "batch_size"),
            "learning_rate": validate_positive_number(learning_rate, "learning_rate"),
            "gamma": validate_positive_number(gamma, "gamma"),
            "beta1": validate_non_negative_number(beta1, "beta1"),
            "beta2": validate_non_negative_number(beta2, "beta2")
        }
        
        return self.generate(
            algorithm="timegan",
            length=length,
            parameters=parameters,
            series_id=series_id,
            name=name,
            seed=seed
        )
    
    def generate_statistical(
        self,
        length: int,
        distribution: str = "normal",
        mean: float = 0.0,
        std: float = 1.0,
        trend_slope: float = 0.0,
        seasonal_amplitude: float = 0.0,
        seasonal_period: int = None,
        noise_level: float = 0.1,
        series_id: str = None,
        name: str = None,
        seed: int = None
    ) -> TimeSeries:
        """
        Generate statistical time series.
        
        Args:
            length: Number of data points to generate
            distribution: Distribution type ("normal", "uniform", "exponential")
            mean: Mean value
            std: Standard deviation
            trend_slope: Linear trend slope
            seasonal_amplitude: Amplitude of seasonal component
            seasonal_period: Period of seasonal component
            noise_level: Level of random noise
            series_id: Series identifier
            name: Series name
            seed: Random seed for reproducibility
        
        Returns:
            Generated statistical time series
        """
        parameters = {
            "distribution": distribution,
            "mean": mean,
            "std": validate_positive_number(std, "std"),
            "trend_slope": trend_slope,
            "seasonal_amplitude": validate_non_negative_number(seasonal_amplitude, "seasonal_amplitude"),
            "noise_level": validate_non_negative_number(noise_level, "noise_level")
        }
        
        if seasonal_period is not None:
            parameters["seasonal_period"] = validate_positive_number(seasonal_period, "seasonal_period")
        
        return self.generate(
            algorithm="statistical",
            length=length,
            parameters=parameters,
            series_id=series_id,
            name=name,
            seed=seed
        )


class ARIMAGenerator(BaseGenerator):
    """Specialized generator for ARIMA time series."""
    
    def __init__(self, client: Union[TSIOTClient, AsyncTSIOTClient], logger: logging.Logger = None):
        super().__init__(client, logger)
        self._generator_type = "arima"
    
    def generate(
        self,
        length: int,
        ar_params: List[float] = None,
        ma_params: List[float] = None,
        d: int = 0,
        trend: str = "none",
        seasonal_periods: int = None,
        noise_std: float = 1.0,
        series_id: str = None,
        name: str = None,
        seed: int = None
    ) -> TimeSeries:
        """Generate ARIMA time series with specified parameters."""
        metadata = self._validate_common_params(length, series_id, name)
        
        parameters = {
            "ar_params": ar_params or [0.5],
            "ma_params": ma_params or [],
            "d": validate_non_negative_number(d, "d"),
            "trend": trend,
            "noise_std": validate_positive_number(noise_std, "noise_std")
        }
        
        if seasonal_periods is not None:
            parameters["seasonal_periods"] = validate_positive_number(seasonal_periods, "seasonal_periods")
        
        request = self._build_request(length, parameters, metadata, seed)
        
        try:
            return self.client.generate(request)
        except Exception as e:
            raise GenerationError(f"Failed to generate ARIMA time series: {str(e)}", generator_type="arima")


class TimeGANGenerator(BaseGenerator):
    """Specialized generator for TimeGAN time series."""
    
    def __init__(self, client: Union[TSIOTClient, AsyncTSIOTClient], logger: logging.Logger = None):
        super().__init__(client, logger)
        self._generator_type = "timegan"
    
    def generate(
        self,
        length: int,
        hidden_dim: int = 24,
        num_layers: int = 3,
        epochs: int = 10000,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        gamma: float = 1.0,
        series_id: str = None,
        name: str = None,
        seed: int = None
    ) -> TimeSeries:
        """Generate TimeGAN time series with specified parameters."""
        metadata = self._validate_common_params(length, series_id, name)
        
        parameters = {
            "hidden_dim": validate_positive_number(hidden_dim, "hidden_dim"),
            "num_layers": validate_positive_number(num_layers, "num_layers"),
            "epochs": validate_positive_number(epochs, "epochs"),
            "batch_size": validate_positive_number(batch_size, "batch_size"),
            "learning_rate": validate_positive_number(learning_rate, "learning_rate"),
            "gamma": validate_positive_number(gamma, "gamma")
        }
        
        request = self._build_request(length, parameters, metadata, seed)
        
        try:
            return self.client.generate(request)
        except Exception as e:
            raise GenerationError(f"Failed to generate TimeGAN time series: {str(e)}", generator_type="timegan")


class LSTMGenerator(BaseGenerator):
    """Specialized generator for LSTM time series."""
    
    def __init__(self, client: Union[TSIOTClient, AsyncTSIOTClient], logger: logging.Logger = None):
        super().__init__(client, logger)
        self._generator_type = "lstm"
    
    def generate(
        self,
        length: int,
        sequence_length: int = 50,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        epochs: int = 100,
        pattern: str = "random",
        series_id: str = None,
        name: str = None,
        seed: int = None
    ) -> TimeSeries:
        """Generate LSTM time series with specified parameters."""
        metadata = self._validate_common_params(length, series_id, name)
        
        parameters = {
            "sequence_length": validate_positive_number(sequence_length, "sequence_length"),
            "hidden_size": validate_positive_number(hidden_size, "hidden_size"),
            "num_layers": validate_positive_number(num_layers, "num_layers"),
            "dropout": validate_non_negative_number(dropout, "dropout"),
            "learning_rate": validate_positive_number(learning_rate, "learning_rate"),
            "epochs": validate_positive_number(epochs, "epochs"),
            "pattern": pattern
        }
        
        request = self._build_request(length, parameters, metadata, seed)
        
        try:
            return self.client.generate(request)
        except Exception as e:
            raise GenerationError(f"Failed to generate LSTM time series: {str(e)}", generator_type="lstm")


class GRUGenerator(BaseGenerator):
    """Specialized generator for GRU time series."""
    
    def __init__(self, client: Union[TSIOTClient, AsyncTSIOTClient], logger: logging.Logger = None):
        super().__init__(client, logger)
        self._generator_type = "gru"
    
    def generate(
        self,
        length: int,
        sequence_length: int = 50,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        epochs: int = 100,
        pattern: str = "random",
        series_id: str = None,
        name: str = None,
        seed: int = None
    ) -> TimeSeries:
        """Generate GRU time series with specified parameters."""
        metadata = self._validate_common_params(length, series_id, name)
        
        parameters = {
            "sequence_length": validate_positive_number(sequence_length, "sequence_length"),
            "hidden_size": validate_positive_number(hidden_size, "hidden_size"),
            "num_layers": validate_positive_number(num_layers, "num_layers"),
            "dropout": validate_non_negative_number(dropout, "dropout"),
            "learning_rate": validate_positive_number(learning_rate, "learning_rate"),
            "epochs": validate_positive_number(epochs, "epochs"),
            "pattern": pattern
        }
        
        request = self._build_request(length, parameters, metadata, seed)
        
        try:
            return self.client.generate(request)
        except Exception as e:
            raise GenerationError(f"Failed to generate GRU time series: {str(e)}", generator_type="gru")


class StatisticalGenerator(BaseGenerator):
    """Specialized generator for statistical time series."""
    
    def __init__(self, client: Union[TSIOTClient, AsyncTSIOTClient], logger: logging.Logger = None):
        super().__init__(client, logger)
        self._generator_type = "statistical"
    
    def generate(
        self,
        length: int,
        distribution: str = "normal",
        mean: float = 0.0,
        std: float = 1.0,
        trend_slope: float = 0.0,
        seasonal_amplitude: float = 0.0,
        seasonal_period: int = None,
        noise_level: float = 0.1,
        series_id: str = None,
        name: str = None,
        seed: int = None
    ) -> TimeSeries:
        """Generate statistical time series with specified parameters."""
        metadata = self._validate_common_params(length, series_id, name)
        
        parameters = {
            "distribution": distribution,
            "mean": mean,
            "std": validate_positive_number(std, "std"),
            "trend_slope": trend_slope,
            "seasonal_amplitude": validate_non_negative_number(seasonal_amplitude, "seasonal_amplitude"),
            "noise_level": validate_non_negative_number(noise_level, "noise_level")
        }
        
        if seasonal_period is not None:
            parameters["seasonal_period"] = validate_positive_number(seasonal_period, "seasonal_period")
        
        request = self._build_request(length, parameters, metadata, seed)
        
        try:
            return self.client.generate(request)
        except Exception as e:
            raise GenerationError(f"Failed to generate statistical time series: {str(e)}", generator_type="statistical")


# Convenience functions
def create_generator(
    client: Union[TSIOTClient, AsyncTSIOTClient],
    generator_type: str = "auto"
) -> BaseGenerator:
    """
    Create a generator instance of the specified type.
    
    Args:
        client: TSIOT client instance
        generator_type: Type of generator to create
    
    Returns:
        Generator instance
    """
    generator_classes = {
        "auto": TimeSeriesGenerator,
        "arima": ARIMAGenerator,
        "timegan": TimeGANGenerator,
        "lstm": LSTMGenerator,
        "gru": GRUGenerator,
        "statistical": StatisticalGenerator
    }
    
    if generator_type not in generator_classes:
        raise ValidationError(
            f"Unknown generator type: {generator_type}",
            details={"available_types": list(generator_classes.keys())}
        )
    
    return generator_classes[generator_type](client)


def list_available_generators() -> List[str]:
    """Get list of available generator types."""
    return ["auto", "arima", "timegan", "lstm", "gru", "statistical"]