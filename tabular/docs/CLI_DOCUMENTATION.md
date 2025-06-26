# Tabular CLI Documentation

## Overview

The Tabular Command Line Interface (CLI) provides a powerful and user-friendly way to generate synthetic tabular data directly from your terminal. This documentation covers all commands, options, and features available in the CLI.

## Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Commands](#commands)
4. [Global Options](#global-options)
5. [Examples](#examples)
6. [Configuration](#configuration)
7. [Advanced Usage](#advanced-usage)

## Installation

```bash
# Install the package
pip install inferloop-tabular

# Verify installation
inferloop-tabular --version

# Get help
inferloop-tabular --help
```

## Basic Usage

```bash
# Simple data generation
inferloop-tabular generate data.csv --rows 1000

# Generate with specific algorithm
inferloop-tabular generate data.csv --algorithm ctgan --output synthetic.csv

# Profile your data
inferloop-tabular profile data.csv --report profile.html

# Validate synthetic data
inferloop-tabular validate real.csv synthetic.csv --output validation.json
```

## Commands

### `generate` - Generate synthetic data

The main command for synthetic data generation.

```bash
inferloop-tabular generate [INPUT] [OPTIONS]
```

#### Arguments

- `INPUT` (required): Path to input data file (CSV, Parquet, JSON) or database connection string

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--algorithm` | `-a` | `sdv` | Algorithm to use (sdv, ctgan, ydata) |
| `--rows` | `-r` | Same as input | Number of rows to generate |
| `--output` | `-o` | `synthetic_{input}` | Output file path |
| `--format` | `-f` | Same as input | Output format (csv, parquet, json) |
| `--model-path` | `-m` | None | Path to pre-trained model |
| `--config` | `-c` | None | Path to configuration file |
| `--seed` | `-s` | None | Random seed for reproducibility |
| `--gpu` | | False | Use GPU acceleration |
| `--verbose` | `-v` | False | Verbose output |

#### Algorithm-specific Options

**SDV Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--sdv-model` | `gaussian_copula` | SDV model type |
| `--distribution` | `gaussian` | Default distribution |
| `--enforce-bounds` | True | Enforce min/max values |

**CTGAN Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 300 | Training epochs |
| `--batch-size` | 500 | Batch size |
| `--embedding-dim` | 128 | Embedding dimension |
| `--discriminator-steps` | 1 | Discriminator updates |

**YData Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--noise-dim` | 128 | Noise dimension |
| `--layers-dim` | 128 | Layer dimensions |
| `--learning-rate` | 2e-4 | Learning rate |
| `--privacy` | None | Privacy level (low, medium, high) |

#### Examples

```bash
# Basic generation
inferloop-tabular generate customers.csv --rows 5000

# CTGAN with custom parameters
inferloop-tabular generate sales.csv \
  --algorithm ctgan \
  --epochs 500 \
  --batch-size 1000 \
  --gpu \
  --output synthetic_sales.csv

# Generate from database
inferloop-tabular generate "postgresql://user:pass@host/db" \
  --query "SELECT * FROM customers" \
  --algorithm ydata \
  --privacy high

# Use configuration file
inferloop-tabular generate data.csv --config generation_config.yaml
```

### `profile` - Profile your data

Analyze and understand your data before generation.

```bash
inferloop-tabular profile [INPUT] [OPTIONS]
```

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `profile.html` | Output report path |
| `--format` | `-f` | `html` | Report format (html, json, pdf) |
| `--sample` | | None | Sample size for large datasets |
| `--include` | | All | Specific analyses to include |
| `--exclude` | | None | Analyses to exclude |

#### Examples

```bash
# Generate HTML profile report
inferloop-tabular profile customers.csv

# JSON output for programmatic use
inferloop-tabular profile sales.csv --format json --output profile.json

# Profile with sampling
inferloop-tabular profile large_dataset.csv --sample 10000
```

### `validate` - Validate synthetic data

Compare synthetic data quality against original data.

```bash
inferloop-tabular validate [REAL] [SYNTHETIC] [OPTIONS]
```

#### Arguments

- `REAL`: Path to real/original data
- `SYNTHETIC`: Path to synthetic data

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--metrics` | `-m` | All | Metrics to calculate |
| `--output` | `-o` | `validation.json` | Output file |
| `--format` | `-f` | `json` | Output format |
| `--visual` | | False | Generate visual report |
| `--privacy` | | False | Include privacy metrics |

#### Available Metrics

- `statistical`: KS test, Chi-squared, correlations
- `ml_efficacy`: Train/test on real vs synthetic
- `privacy`: Distance to closest record, membership inference
- `constraints`: Business rule validation
- `visual`: Distribution plots, PCA, t-SNE

#### Examples

```bash
# Basic validation
inferloop-tabular validate original.csv synthetic.csv

# Full validation with visuals
inferloop-tabular validate real.csv synthetic.csv \
  --visual \
  --output validation_report.html

# Privacy-focused validation
inferloop-tabular validate sensitive.csv synthetic.csv \
  --metrics privacy \
  --privacy
```

### `benchmark` - Benchmark algorithms

Compare different algorithms on your data.

```bash
inferloop-tabular benchmark [INPUT] [OPTIONS]
```

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--algorithms` | `-a` | All | Algorithms to benchmark |
| `--metrics` | `-m` | All | Metrics to evaluate |
| `--samples` | `-n` | 1000 | Samples to generate |
| `--timeout` | | 3600 | Max time per algorithm (seconds) |
| `--output` | `-o` | `benchmark.json` | Results file |

#### Examples

```bash
# Benchmark all algorithms
inferloop-tabular benchmark data.csv

# Compare specific algorithms
inferloop-tabular benchmark data.csv \
  --algorithms sdv,ctgan \
  --samples 5000

# Quick benchmark
inferloop-tabular benchmark data.csv \
  --algorithms sdv \
  --samples 100 \
  --timeout 300
```

### `train` - Train and save model

Train a model for later use.

```bash
inferloop-tabular train [INPUT] [OPTIONS]
```

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--algorithm` | `-a` | `sdv` | Algorithm to train |
| `--output` | `-o` | `model.pkl` | Model save path |
| `--config` | `-c` | None | Training configuration |
| `--validate` | | True | Validate after training |

#### Examples

```bash
# Train and save model
inferloop-tabular train customers.csv \
  --algorithm ctgan \
  --output customer_model.pkl

# Train with configuration
inferloop-tabular train data.csv \
  --config training_config.yaml \
  --output trained_model.pkl
```

### `serve` - Start API server

Launch REST API server for synthetic data generation.

```bash
inferloop-tabular serve [OPTIONS]
```

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--host` | `-h` | `0.0.0.0` | Host to bind |
| `--port` | `-p` | `8000` | Port to listen |
| `--workers` | `-w` | `1` | Number of workers |
| `--reload` | | False | Auto-reload on changes |
| `--models-dir` | | `./models` | Models directory |

#### Examples

```bash
# Start basic server
inferloop-tabular serve

# Production server
inferloop-tabular serve \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 4

# Development server
inferloop-tabular serve --reload --port 8000
```

### `batch` - Batch processing

Process multiple files or large datasets.

```bash
inferloop-tabular batch [INPUT_DIR] [OPTIONS]
```

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--pattern` | `-p` | `*.csv` | File pattern |
| `--output-dir` | `-o` | `./synthetic` | Output directory |
| `--algorithm` | `-a` | `sdv` | Algorithm to use |
| `--parallel` | | `1` | Parallel jobs |
| `--config` | `-c` | None | Batch configuration |

#### Examples

```bash
# Process all CSV files
inferloop-tabular batch ./data --pattern "*.csv"

# Parallel processing
inferloop-tabular batch ./datasets \
  --parallel 4 \
  --algorithm ctgan \
  --output-dir ./synthetic_data

# With configuration
inferloop-tabular batch ./input \
  --config batch_config.yaml
```

### `stream` - Stream synthetic data

Generate synthetic data in streaming mode.

```bash
inferloop-tabular stream [MODEL] [OPTIONS]
```

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--rate` | `-r` | `100` | Records per second |
| `--duration` | `-d` | None | Duration (seconds) |
| `--total` | `-t` | None | Total records |
| `--output` | `-o` | `stdout` | Output destination |
| `--format` | `-f` | `json` | Output format |

#### Examples

```bash
# Stream to stdout
inferloop-tabular stream model.pkl --rate 50

# Stream to file
inferloop-tabular stream model.pkl \
  --output stream.jsonl \
  --duration 3600

# Stream to Kafka
inferloop-tabular stream model.pkl \
  --output kafka://localhost:9092/topic \
  --rate 1000
```

### `privacy` - Privacy analysis

Analyze privacy risks and apply protection.

```bash
inferloop-tabular privacy [COMMAND] [OPTIONS]
```

#### Subcommands

- `analyze`: Analyze privacy risks
- `protect`: Apply privacy protection
- `evaluate`: Evaluate privacy guarantees

#### Examples

```bash
# Analyze privacy risks
inferloop-tabular privacy analyze data.csv \
  --quasi-identifiers age,zipcode,gender \
  --sensitive income,health

# Apply k-anonymity
inferloop-tabular privacy protect data.csv \
  --method k-anonymity \
  --k 5 \
  --output anonymous.csv

# Evaluate privacy
inferloop-tabular privacy evaluate \
  original.csv synthetic.csv \
  --metrics all
```

### `config` - Manage configuration

Configure default settings and preferences.

```bash
inferloop-tabular config [SUBCOMMAND] [OPTIONS]
```

#### Subcommands

- `show`: Display current configuration
- `set`: Set configuration values
- `reset`: Reset to defaults
- `export`: Export configuration
- `import`: Import configuration

#### Examples

```bash
# Show configuration
inferloop-tabular config show

# Set default algorithm
inferloop-tabular config set default.algorithm ctgan

# Set API key
inferloop-tabular config set api.key "your-api-key"

# Export configuration
inferloop-tabular config export --output config.yaml
```

## Global Options

These options work with all commands:

| Option | Short | Description |
|--------|-------|-------------|
| `--help` | `-h` | Show help message |
| `--version` | `-V` | Show version |
| `--verbose` | `-v` | Increase verbosity (-vv for debug) |
| `--quiet` | `-q` | Suppress output |
| `--config` | `-c` | Path to config file |
| `--no-color` | | Disable colored output |
| `--json` | | Output in JSON format |

## Configuration

### Configuration File

Create a configuration file at `~/.inferloop/tabular/config.yaml`:

```yaml
# Default settings
default:
  algorithm: ctgan
  output_format: csv
  gpu: auto
  seed: 42

# Algorithm-specific defaults
algorithms:
  sdv:
    model_type: gaussian_copula
    default_distribution: beta
  ctgan:
    epochs: 300
    batch_size: 500
    embedding_dim: 128
  ydata:
    noise_dim: 128
    privacy_level: medium

# API settings
api:
  key: your-api-key
  endpoint: https://api.inferloop.com

# Performance settings
performance:
  batch_size: 10000
  n_jobs: -1  # Use all cores
  cache_models: true

# Privacy defaults
privacy:
  check_privacy: true
  min_k_anonymity: 5
  epsilon: 1.0
```

### Environment Variables

Configure via environment variables:

```bash
# API configuration
export INFERLOOP_API_KEY="your-api-key"
export INFERLOOP_API_ENDPOINT="https://api.inferloop.com"

# Default settings
export INFERLOOP_DEFAULT_ALGORITHM="ctgan"
export INFERLOOP_GPU_ENABLED="true"
export INFERLOOP_CACHE_DIR="~/.inferloop/cache"

# Privacy settings
export INFERLOOP_PRIVACY_CHECK="true"
export INFERLOOP_PRIVACY_EPSILON="1.0"
```

## Advanced Usage

### Using Configuration Files

Create `generation_config.yaml`:

```yaml
algorithm: ctgan
algorithm_params:
  epochs: 500
  batch_size: 1000
  embedding_dim: 256
  discriminator_steps: 5

generation:
  num_rows: 10000
  conditions:
    status: "active"
    balance: ">1000"

constraints:
  - column: age
    type: range
    min: 0
    max: 120
  - column: email
    type: regex
    pattern: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"

privacy:
  method: differential_privacy
  epsilon: 1.0
  delta: 1e-5

output:
  format: parquet
  compression: snappy
```

Use the configuration:

```bash
inferloop-tabular generate data.csv --config generation_config.yaml
```

### Piping and Chaining

```bash
# Pipe data through multiple commands
cat data.csv | inferloop-tabular generate - --algorithm sdv | \
  inferloop-tabular validate data.csv - --metrics statistical

# Chain with other tools
inferloop-tabular generate large.csv --format json | \
  jq '.[] | select(.age > 18)' | \
  inferloop-tabular validate original.csv -

# Generate and analyze
inferloop-tabular generate data.csv --output - | \
  python analyze.py
```

### Database Integration

```bash
# PostgreSQL
inferloop-tabular generate \
  "postgresql://user:pass@localhost/db" \
  --table customers \
  --where "created_at > '2023-01-01'"

# MySQL
inferloop-tabular generate \
  "mysql://user:pass@localhost/db" \
  --query "SELECT * FROM orders JOIN customers ON ..."

# SQLite
inferloop-tabular generate \
  "sqlite:///local.db" \
  --table transactions
```

### Batch Processing Script

Create `batch_generate.sh`:

```bash
#!/bin/bash

# Configuration
ALGORITHM="ctgan"
INPUT_DIR="./raw_data"
OUTPUT_DIR="./synthetic_data"
LOG_FILE="generation.log"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process each file
for file in "$INPUT_DIR"/*.csv; do
  basename=$(basename "$file")
  echo "Processing $basename..." | tee -a "$LOG_FILE"
  
  inferloop-tabular generate "$file" \
    --algorithm "$ALGORITHM" \
    --output "$OUTPUT_DIR/synthetic_$basename" \
    --verbose \
    2>&1 | tee -a "$LOG_FILE"
    
  # Validate generated data
  inferloop-tabular validate "$file" "$OUTPUT_DIR/synthetic_$basename" \
    --output "$OUTPUT_DIR/validation_$basename.json" \
    2>&1 | tee -a "$LOG_FILE"
done

echo "Batch processing complete!" | tee -a "$LOG_FILE"
```

### Integration with Data Pipelines

```bash
# Apache Airflow DAG
inferloop-tabular generate {{ params.input }} \
  --algorithm {{ params.algorithm }} \
  --rows {{ params.rows }} \
  --output {{ params.output }}

# Luigi task
python -m luigi --module synthetic_pipeline GenerateTask \
  --input-path /data/raw \
  --synthetic-path /data/synthetic

# Prefect flow
inferloop-tabular generate $INPUT \
  --config $CONFIG \
  --output $OUTPUT
```

### Custom Plugins

Create custom algorithm plugin:

```python
# my_algorithm.py
from tabular.sdk.base import BaseGenerator

class MyCustomGenerator(BaseGenerator):
    def fit(self, data):
        # Custom implementation
        pass
        
    def generate(self, num_samples):
        # Custom implementation
        pass

# Register plugin
# inferloop-tabular generate data.csv --algorithm my_algorithm.MyCustomGenerator
```

## Performance Tips

1. **Use GPU**: Add `--gpu` flag for 10-100x speedup
2. **Batch large files**: Process in chunks with `--batch-size`
3. **Parallel processing**: Use `--parallel` for multiple files
4. **Cache models**: Save trained models with `train` command
5. **Optimize parameters**: Start with smaller epochs for testing

## Troubleshooting

### Common Issues

1. **Command not found**
   ```bash
   # Add to PATH
   export PATH="$HOME/.local/bin:$PATH"
   ```

2. **Memory errors**
   ```bash
   # Reduce batch size
   inferloop-tabular generate large.csv --batch-size 100
   
   # Use sampling
   inferloop-tabular generate large.csv --sample 10000
   ```

3. **GPU not detected**
   ```bash
   # Check CUDA
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Force CPU
   inferloop-tabular generate data.csv --no-gpu
   ```

4. **Slow generation**
   ```bash
   # Use simpler algorithm
   inferloop-tabular generate data.csv --algorithm sdv
   
   # Reduce epochs
   inferloop-tabular generate data.csv --epochs 100
   ```

## Best Practices

1. **Always profile first**: Use `profile` command to understand data
2. **Validate results**: Use `validate` command to check quality
3. **Start small**: Test with small samples before full generation
4. **Use appropriate algorithm**: SDV for speed, CTGAN for quality
5. **Save models**: Use `train` command for reusable models
6. **Monitor privacy**: Always check privacy metrics for sensitive data

## Examples Gallery

### Financial Data

```bash
# Generate synthetic financial transactions
inferloop-tabular generate transactions.csv \
  --algorithm ydata \
  --privacy high \
  --rows 100000 \
  --config financial_config.yaml

# Validate financial constraints
inferloop-tabular validate real_trans.csv synthetic_trans.csv \
  --metrics constraints \
  --constraints financial_rules.json
```

### Healthcare Data

```bash
# HIPAA-compliant generation
inferloop-tabular privacy protect patient_data.csv \
  --method k-anonymity \
  --k 5 \
  --quasi-identifiers age,zipcode,gender

# Generate with clinical constraints
inferloop-tabular generate medical_records.csv \
  --algorithm ctgan \
  --config healthcare_config.yaml \
  --constraints clinical_rules.json
```

### Time Series

```bash
# Generate time series data
inferloop-tabular generate timeseries.csv \
  --algorithm ydata \
  --model-type timegan \
  --sequence-length 24 \
  --temporal-columns timestamp
```

## Support

- **Documentation**: Run `inferloop-tabular --help` or `inferloop-tabular [command] --help`
- **Examples**: See `/usr/share/inferloop/examples/` after installation
- **Issues**: Report at [github.com/inferloop/tabular/issues](https://github.com/inferloop/tabular/issues)
- **Community**: Join our Discord at [discord.gg/inferloop](https://discord.gg/inferloop)