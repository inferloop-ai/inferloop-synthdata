# Test Data Generator

A tool for generating synthetic time series data for testing and development purposes.

## Features

- **Multiple Patterns**: Generate data with various patterns including sine waves, linear trends, seasonal patterns, and random walks
- **Configurable Noise**: Add Gaussian or uniform noise to make data more realistic
- **Multiple Output Formats**: Export data in JSON or CSV format
- **Flexible Configuration**: Use command-line flags or JSON configuration files
- **Quality Metrics**: Generate quality scores for each data point

## Usage

### Command Line

```bash
# Generate 10 series with 1000 points each
./test-data-generator -series 10 -points 1000 -output test_data.json

# Generate with custom format
./test-data-generator -series 5 -points 500 -format csv -output test_data.csv

# Use configuration file
./test-data-generator -config config.json

# Enable verbose logging
./test-data-generator -verbose -series 3 -points 100
```

### Configuration File

Create a JSON configuration file to customize generation parameters:

```json
{
  "num_series": 20,
  "points_per_series": 2000,
  "time_interval": "1m",
  "output_format": "json",
  "output_file": "synthetic_data.json",
  "patterns": [
    {
      "type": "sine",
      "amplitude": 100.0,
      "frequency": 0.05,
      "phase": 0.0
    },
    {
      "type": "seasonal",
      "amplitude": 30.0,
      "seasonality": 24
    },
    {
      "type": "linear",
      "trend": 0.2
    },
    {
      "type": "random",
      "amplitude": 15.0
    }
  ],
  "noise": {
    "enabled": true,
    "type": "gaussian",
    "level": 8.0,
    "mean": 0.0,
    "std_dev": 3.0
  }
}
```

## Pattern Types

### Sine Wave
```json
{
  "type": "sine",
  "amplitude": 50.0,
  "frequency": 0.1,
  "phase": 0.0
}
```

### Cosine Wave
```json
{
  "type": "cosine",
  "amplitude": 50.0,
  "frequency": 0.1,
  "phase": 0.0
}
```

### Linear Trend
```json
{
  "type": "linear",
  "trend": 0.1
}
```

### Seasonal Pattern
```json
{
  "type": "seasonal",
  "amplitude": 20.0,
  "seasonality": 24
}
```

### Exponential Growth/Decay
```json
{
  "type": "exponential",
  "amplitude": 10.0,
  "trend": 0.01
}
```

### Logarithmic
```json
{
  "type": "logarithmic",
  "amplitude": 15.0
}
```

### Random Walk
```json
{
  "type": "random",
  "amplitude": 10.0
}
```

### Step Function
```json
{
  "type": "step",
  "amplitude": 25.0
}
```

## Noise Configuration

### Gaussian Noise
```json
{
  "enabled": true,
  "type": "gaussian",
  "level": 5.0,
  "mean": 0.0,
  "std_dev": 2.0
}
```

### Uniform Noise
```json
{
  "enabled": true,
  "type": "uniform",
  "level": 10.0
}
```

## Output Formats

### JSON Format
The JSON format includes metadata and preserves all time series properties:

```json
{
  "metadata": {
    "generated_at": "2024-01-15T10:30:00Z",
    "generator": "test-data-generator",
    "version": "1.0.0",
    "series_count": 10
  },
  "data": [
    {
      "id": "test_series_0",
      "name": "Test Series 0",
      "points": [
        {
          "timestamp": 1705312200,
          "value": 45.234,
          "quality": 0.95
        }
      ],
      "metadata": {
        "generator": "test-data-generator",
        "series_type": "synthetic",
        "created_at": "2024-01-15T10:30:00Z",
        "series_index": 0
      },
      "properties": {
        "unit": "units",
        "description": "Generated test time series 0"
      },
      "tags": ["test", "synthetic", "series_0"]
    }
  ]
}
```

### CSV Format
The CSV format provides a flat structure suitable for analysis tools:

```csv
series_id,series_name,timestamp,value,quality
test_series_0,Test Series 0,1705312200,45.234000,0.950
test_series_0,Test Series 0,1705312260,47.123000,0.923
```

## Examples

### Generate IoT Sensor Data
```bash
# Temperature sensor with daily seasonality
./test-data-generator \
  -series 1 \
  -points 1440 \
  -output temperature_data.json \
  -config temperature_config.json
```

```json
{
  "patterns": [
    {
      "type": "sine",
      "amplitude": 10.0,
      "frequency": 1.0,
      "phase": 0.0
    },
    {
      "type": "seasonal",
      "amplitude": 5.0,
      "seasonality": 1440
    }
  ],
  "noise": {
    "enabled": true,
    "type": "gaussian",
    "std_dev": 1.5
  }
}
```

### Generate Network Traffic Data
```bash
# Network traffic with business hours pattern
./test-data-generator \
  -series 5 \
  -points 2880 \
  -format csv \
  -output network_traffic.csv
```

### Generate Financial Time Series
```bash
# Stock price simulation with trend and volatility
./test-data-generator \
  -series 10 \
  -points 252 \
  -output stock_prices.json
```

## Building

```bash
go build -o test-data-generator main.go
```

## Integration

The generated data is compatible with the tsiot validation and processing pipeline:

```go
import "github.com/inferloop/tsiot/pkg/models"

// Load generated data
data := loadGeneratedData("test_data.json")

// Use with validation engine
validator := validation.NewValidationEngine(nil, nil)
result, err := validator.ValidateDataset(ctx, data, nil)
```