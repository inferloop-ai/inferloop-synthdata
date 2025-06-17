"""
TSIOT Python SDK

A Python client library for interacting with the Time Series IoT Synthetic Data (TSIOT) service.
This SDK provides comprehensive functionality for generating, validating, and analyzing time series data.

Key Features:
- Time series data generation with multiple algorithms
- Data validation and quality assessment
- Analytics and statistical analysis
- Support for various data formats and protocols
- Async/await support for high-performance operations

Example Usage:
    ```python
    from tsiot import TSIOTClient, TimeSeriesGenerator
    
    # Initialize client
    client = TSIOTClient(base_url="http://localhost:8080", api_key="your-api-key")
    
    # Generate time series data
    generator = TimeSeriesGenerator(client)
    data = await generator.generate_arima(
        length=1000,
        ar_params=[0.5, -0.3],
        ma_params=[0.2],
        trend="linear"
    )
    
    # Validate data
    validation_result = await client.validate(data)
    print(f"Data quality score: {validation_result.quality_score}")
    
    # Perform analytics
    analysis = await client.analyze(data, ["basic", "trend", "seasonality"])
    print(f"Trend direction: {analysis.trend.direction}")
    ```
"""

__version__ = "1.0.0"
__author__ = "TSIOT Development Team"
__email__ = "dev@tsiot.com"
__license__ = "MIT"

# Core imports
from .client import TSIOTClient, AsyncTSIOTClient
from .timeseries import (
    TimeSeries,
    DataPoint,
    TimeSeriesMetadata,
    DataFormat,
    Frequency
)
from .generators import (
    TimeSeriesGenerator,
    ARIMAGenerator,
    TimeGANGenerator,
    LSTMGenerator,
    GRUGenerator,
    StatisticalGenerator
)
from .validators import (
    TimeSeriesValidator,
    QualityValidator,
    StatisticalValidator,
    PrivacyValidator,
    ValidationResult,
    QualityMetrics
)
from .utils import (
    configure_logging,
    TSIOTError,
    ValidationError,
    GenerationError,
    AnalyticsError,
    format_timestamp,
    parse_timestamp
)

# Convenience imports for common use cases
from .client import TSIOTClient as Client
from .timeseries import TimeSeries as TS
from .generators import TimeSeriesGenerator as Generator
from .validators import TimeSeriesValidator as Validator

__all__ = [
    # Core classes
    "TSIOTClient",
    "AsyncTSIOTClient", 
    "TimeSeries",
    "DataPoint",
    "TimeSeriesMetadata",
    "TimeSeriesGenerator",
    "TimeSeriesValidator",
    
    # Generator classes
    "ARIMAGenerator",
    "TimeGANGenerator", 
    "LSTMGenerator",
    "GRUGenerator",
    "StatisticalGenerator",
    
    # Validator classes
    "QualityValidator",
    "StatisticalValidator",
    "PrivacyValidator",
    "ValidationResult",
    "QualityMetrics",
    
    # Enums and types
    "DataFormat",
    "Frequency",
    
    # Utility functions and exceptions
    "configure_logging",
    "TSIOTError",
    "ValidationError", 
    "GenerationError",
    "AnalyticsError",
    "format_timestamp",
    "parse_timestamp",
    
    # Convenience aliases
    "Client",
    "TS", 
    "Generator",
    "Validator",
]