# Python SDK Guide

The TSIoT Python SDK provides a pythonic interface for generating and validating synthetic time series data.

## Installation

### Using pip

```bash
pip install tsiot-sdk
```

### From source

```bash
git clone https://github.com/inferloop/tsiot-python-sdk.git
cd tsiot-python-sdk
pip install -e .
```

### Requirements

- Python 3.8+
- requests >= 2.25.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- websockets >= 10.0 (for streaming)

## Quick Start

```python
from tsiot import Client
import pandas as pd

# Connect to TSIoT server
client = Client("http://localhost:8080")

# Generate synthetic data
data = client.generate(
    generator="statistical",
    sensor_type="temperature",
    points=1000,
    frequency="1m"
)

# Convert to pandas DataFrame
df = data.to_pandas()
print(df.head())

# Save to file
data.to_csv("temperature_data.csv")
```

## API Reference

### Client

The main client class for interacting with TSIoT server.

```python
class Client:
    def __init__(self, base_url: str, api_key: str = None, timeout: int = 30)
```

#### Parameters
- `base_url`: TSIoT server URL
- `api_key`: Optional API key for authentication
- `timeout`: Request timeout in seconds

#### Example
```python
# Basic client
client = Client("http://localhost:8080")

# With authentication
client = Client(
    "https://tsiot.example.com",
    api_key="your-api-key"
)

# Custom timeout
client = Client(
    "http://localhost:8080",
    timeout=60
)
```

### Generate Data

```python
def generate(
    self,
    generator: str = "statistical",
    sensor_type: str = "temperature",
    sensor_id: str = None,
    points: int = 100,
    start_time: Union[str, datetime] = None,
    end_time: Union[str, datetime] = None,
    duration: str = "24h",
    frequency: str = "1m",
    anomalies: int = 0,
    anomaly_magnitude: float = 3.0,
    noise_level: float = 0.05,
    trend: float = 0.0,
    seasonality: bool = False,
    seasonal_period: str = "24h",
    seed: int = None,
    privacy: bool = False,
    privacy_epsilon: float = 1.0,
    **kwargs
) -> TimeSeriesData
```

#### Parameters
- `generator`: Generation algorithm (`statistical`, `arima`, `timegan`, `rnn`, `ydata`)
- `sensor_type`: Type of sensor (`temperature`, `humidity`, `pressure`, etc.)
- `sensor_id`: Unique sensor identifier
- `points`: Number of data points to generate
- `start_time`: Start timestamp (ISO format or datetime object)
- `end_time`: End timestamp (ISO format or datetime object)
- `duration`: Duration string (e.g., "24h", "7d", "1y")
- `frequency`: Data frequency (e.g., "1s", "5m", "1h")
- `anomalies`: Number of anomalies to inject
- `anomaly_magnitude`: Anomaly size in standard deviations
- `noise_level`: Noise level (0.0 to 1.0)
- `trend`: Trend coefficient
- `seasonality`: Enable seasonal patterns
- `seasonal_period`: Seasonal period
- `seed`: Random seed for reproducibility
- `privacy`: Enable privacy-preserving generation
- `privacy_epsilon`: Privacy budget parameter

#### Examples

```python
# Basic generation
data = client.generate(
    generator="statistical",
    sensor_type="temperature",
    points=1000
)

# Time range generation
from datetime import datetime, timedelta

start = datetime.now() - timedelta(days=7)
end = datetime.now()

data = client.generate(
    generator="arima",
    sensor_type="humidity",
    start_time=start,
    end_time=end,
    frequency="5m"
)

# With anomalies and seasonality
data = client.generate(
    generator="timegan",
    sensor_type="energy",
    points=10000,
    anomalies=20,
    anomaly_magnitude=2.5,
    seasonality=True,
    seasonal_period="24h",
    seed=42
)

# Privacy-preserving
data = client.generate(
    generator="ydata",
    sensor_type="medical",
    points=5000,
    privacy=True,
    privacy_epsilon=0.5
)
```

### Validate Data

```python
def validate(
    self,
    data: Union[TimeSeriesData, str, pd.DataFrame],
    reference: Union[TimeSeriesData, str, pd.DataFrame] = None,
    metrics: List[str] = None,
    statistical_tests: List[str] = None,
    quality_threshold: float = 0.8
) -> ValidationResult
```

#### Parameters
- `data`: Data to validate (TimeSeriesData, file path, or DataFrame)
- `reference`: Reference data for comparison
- `metrics`: List of validation metrics
- `statistical_tests`: List of statistical tests
- `quality_threshold`: Minimum quality score

#### Examples

```python
# Basic validation
result = client.validate(data)
print(f"Quality score: {result.quality_score}")

# Compare with reference
reference_data = pd.read_csv("real_data.csv")
result = client.validate(
    data=synthetic_data,
    reference=reference_data,
    metrics=["distribution", "temporal", "similarity"]
)

# Validate from file
result = client.validate(
    data="synthetic_data.csv",
    reference="real_data.csv",
    statistical_tests=["ks", "anderson", "ljung_box"]
)
```

### Analyze Data

```python
def analyze(
    self,
    data: Union[TimeSeriesData, str, pd.DataFrame],
    detect_anomalies: bool = False,
    anomaly_method: str = "isolation_forest",
    decompose: bool = False,
    forecast: bool = False,
    forecast_periods: int = 10,
    patterns: bool = False
) -> AnalysisResult
```

#### Examples

```python
# Basic analysis
result = client.analyze(data)
print(result.summary_stats)

# Anomaly detection
result = client.analyze(
    data=sensor_data,
    detect_anomalies=True,
    anomaly_method="isolation_forest"
)
print(f"Found {len(result.anomalies)} anomalies")

# Time series decomposition and forecasting
result = client.analyze(
    data=historical_data,
    decompose=True,
    forecast=True,
    forecast_periods=24
)
```

### Stream Data

```python
def stream(
    self,
    generator: str = "statistical",
    sensor_type: str = "temperature",
    frequency: str = "1s",
    callback: Callable = None,
    **kwargs
) -> Iterator[DataPoint]
```

#### Examples

```python
# Basic streaming
for point in client.stream(frequency="5s"):
    print(f"{point.timestamp}: {point.value}")

# With callback
def process_point(point):
    print(f"Received: {point.sensor_id} = {point.value}")
    # Send to database, queue, etc.

client.stream(
    generator="arima",
    sensor_type="pressure",
    frequency="1s",
    callback=process_point
)

# Async streaming
import asyncio

async def stream_handler():
    async for point in client.stream_async(frequency="1s"):
        await process_async(point)

asyncio.run(stream_handler())
```

## Data Classes

### TimeSeriesData

Represents time series data with metadata.

```python
class TimeSeriesData:
    def __init__(self, points: List[DataPoint], metadata: dict = None)
    
    # Properties
    @property
    def timestamps(self) -> List[datetime]
    @property
    def values(self) -> List[float]
    @property
    def sensor_ids(self) -> List[str]
    
    # Methods
    def to_pandas(self) -> pd.DataFrame
    def to_csv(self, filename: str) -> None
    def to_json(self, filename: str = None) -> Union[str, None]
    def to_parquet(self, filename: str) -> None
    def filter(self, start_time: datetime = None, end_time: datetime = None) -> 'TimeSeriesData'
    def resample(self, frequency: str, method: str = "mean") -> 'TimeSeriesData'
    def add_noise(self, level: float = 0.05) -> 'TimeSeriesData'
    def normalize(self, method: str = "minmax") -> 'TimeSeriesData'
```

### DataPoint

```python
class DataPoint:
    timestamp: datetime
    sensor_id: str
    sensor_type: str
    value: float
    unit: str
    metadata: dict
```

### ValidationResult

```python
class ValidationResult:
    quality_score: float
    metrics: dict
    statistical_tests: dict
    anomalies: List[dict]
    recommendations: List[str]
    passed: bool
```

### AnalysisResult

```python
class AnalysisResult:
    summary_stats: dict
    anomalies: List[dict]
    decomposition: dict
    forecast: dict
    patterns: List[dict]
```

## Advanced Usage

### Batch Generation

```python
from concurrent.futures import ThreadPoolExecutor

def generate_sensor_data(sensor_id):
    return client.generate(
        generator="arima",
        sensor_type="temperature",
        sensor_id=sensor_id,
        points=1440,  # 24 hours of minute data
        frequency="1m"
    )

# Generate data for 100 sensors in parallel
sensor_ids = [f"sensor_{i:03d}" for i in range(100)]
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(generate_sensor_data, sensor_ids))

# Combine all data
all_data = TimeSeriesData.concat(results)
all_data.to_parquet("all_sensors.parquet")
```

### Custom Generators

```python
# Define custom generation parameters
custom_params = {
    "distribution": "exponential",
    "scale": 2.0,
    "offset": 10.0,
    "seasonality_components": [
        {"period": "24h", "amplitude": 5.0},
        {"period": "7d", "amplitude": 2.0}
    ]
}

data = client.generate(
    generator="statistical",
    sensor_type="custom",
    points=10000,
    custom_params=custom_params
)
```

### Integration with Pandas

```python
import pandas as pd
import matplotlib.pyplot as plt

# Generate data
data = client.generate(
    generator="timegan",
    sensor_type="temperature",
    points=2000,
    frequency="5m",
    seasonality=True
)

# Convert to DataFrame
df = data.to_pandas()

# Pandas operations
df['hour'] = df['timestamp'].dt.hour
hourly_avg = df.groupby('hour')['value'].mean()

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
df.plot(x='timestamp', y='value', title='Time Series')

plt.subplot(1, 2, 2)
hourly_avg.plot(kind='bar', title='Hourly Average')
plt.show()
```

### Real-time Processing

```python
import asyncio
from collections import deque

class RealTimeProcessor:
    def __init__(self, window_size=100):
        self.buffer = deque(maxlen=window_size)
        self.client = Client("http://localhost:8080")
    
    async def process_stream(self):
        async for point in self.client.stream_async(
            generator="arima",
            frequency="1s"
        ):
            self.buffer.append(point.value)
            
            if len(self.buffer) == self.buffer.maxlen:
                # Analyze windowed data
                analysis = await self.analyze_window()
                if analysis.has_anomaly:
                    await self.handle_anomaly(point, analysis)
    
    async def analyze_window(self):
        # Convert buffer to TimeSeriesData
        ts_data = TimeSeriesData.from_values(list(self.buffer))
        return await self.client.analyze_async(
            ts_data,
            detect_anomalies=True
        )
    
    async def handle_anomaly(self, point, analysis):
        print(f"Anomaly detected at {point.timestamp}: {point.value}")
        # Send alert, log, etc.

# Run processor
processor = RealTimeProcessor()
asyncio.run(processor.process_stream())
```

### Error Handling

```python
from tsiot.exceptions import (
    TSIoTError,
    ConnectionError,
    ValidationError,
    GenerationError
)

try:
    data = client.generate(
        generator="invalid_generator",
        points=1000
    )
except GenerationError as e:
    print(f"Generation failed: {e}")
except ConnectionError as e:
    print(f"Cannot connect to server: {e}")
except TSIoTError as e:
    print(f"TSIoT error: {e}")
```

## Configuration

### Environment Variables

```python
import os
from tsiot import Client

# Client will automatically use these environment variables
os.environ['TSIOT_SERVER_URL'] = 'http://localhost:8080'
os.environ['TSIOT_API_KEY'] = 'your-api-key'
os.environ['TSIOT_TIMEOUT'] = '60'

client = Client()  # Uses environment variables
```

### Configuration File

```python
# ~/.tsiot/config.yaml
server:
  url: http://localhost:8080
  api_key: your-api-key
  timeout: 30

defaults:
  generator: timegan
  sensor_type: temperature
  points: 1000
  frequency: 1m

logging:
  level: INFO
```

```python
from tsiot import Client, load_config

config = load_config()  # Loads from ~/.tsiot/config.yaml
client = Client.from_config(config)
```

## Testing

### Mock Client

```python
from tsiot.testing import MockClient

# For unit testing
mock_client = MockClient()
mock_client.set_generate_response([
    DataPoint(timestamp=datetime.now(), value=25.0, sensor_id="test")
])

# Test your code with mock client
data = mock_client.generate(points=1)
assert len(data.points) == 1
```

### Test Utilities

```python
from tsiot.testing import (
    generate_test_data,
    assert_time_series_equal,
    validate_data_quality
)

# Generate test data
test_data = generate_test_data(
    sensor_type="temperature",
    points=100,
    pattern="linear"
)

# Assertions
assert_time_series_equal(expected_data, actual_data, tolerance=0.1)
assert validate_data_quality(data, min_score=0.8)
```

## Examples

See the [examples directory](../../../examples/python-sdk/) for complete examples:

- [Basic Usage](../../../examples/python-sdk/basic_usage.py)
- [Advanced Features](../../../examples/python-sdk/advanced_features.py)
- [Batch Generation](../../../examples/python-sdk/batch_generation.py)
- [Streaming Data](../../../examples/python-sdk/streaming_data.py)

## Support

- **Documentation**: https://docs.inferloop.com/tsiot/python-sdk
- **GitHub Issues**: https://github.com/inferloop/tsiot-python-sdk/issues
- **PyPI**: https://pypi.org/project/tsiot-sdk/
- **Examples**: https://github.com/inferloop/tsiot/tree/main/examples/python-sdk