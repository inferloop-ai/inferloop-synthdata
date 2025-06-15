# CLI Reference

Comprehensive reference for the TSIoT command-line interface.

## Global Options

These options are available for all commands:

```bash
tsiot-cli [global options] command [command options] [arguments...]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Path to config file | `~/.tsiot/config.yaml` |
| `--server` | `-s` | Server URL | `http://localhost:8080` |
| `--output` | `-o` | Output destination | `stdout` |
| `--format` | `-f` | Output format (json, csv, table) | `json` |
| `--verbose` | `-v` | Enable verbose logging | `false` |
| `--quiet` | `-q` | Suppress non-error output | `false` |
| `--no-color` | | Disable colored output | `false` |
| `--help` | `-h` | Show help | |
| `--version` | | Show version | |

## Commands

### generate

Generate synthetic time series data.

```bash
tsiot-cli generate [options]
```

#### Options

| Option | Description | Default | Example |
|--------|-------------|---------|----------|
| `--generator` | Generation algorithm | `statistical` | `timegan`, `arima`, `rnn` |
| `--sensor-type` | Type of sensor | `temperature` | `humidity`, `pressure` |
| `--sensor-id` | Unique sensor identifier | `auto-generated` | `sensor_001` |
| `--points` | Number of data points | `100` | `1000` |
| `--start-time` | Start timestamp | `now - duration` | `2024-01-01T00:00:00Z` |
| `--end-time` | End timestamp | `now` | `2024-01-02T00:00:00Z` |
| `--duration` | Time duration | `24h` | `7d`, `1h`, `30m` |
| `--frequency` | Data frequency | `1m` | `5s`, `1h`, `1d` |
| `--anomalies` | Number of anomalies | `0` | `10` |
| `--anomaly-magnitude` | Anomaly size (std devs) | `3.0` | `2.5` |
| `--anomaly-type` | Type of anomaly | `spike` | `drift`, `noise`, `pattern` |
| `--noise-level` | Noise percentage | `0.05` | `0.1` |
| `--trend` | Trend coefficient | `0.0` | `0.01` |
| `--seasonality` | Enable seasonality | `false` | `true` |
| `--seasonal-period` | Seasonal period | `24h` | `7d`, `1y` |
| `--seed` | Random seed | `random` | `42` |
| `--batch-size` | Batch size for generation | `1000` | `5000` |
| `--privacy` | Enable privacy preservation | `false` | `true` |
| `--privacy-epsilon` | Privacy budget | `1.0` | `0.5` |
| `--metadata` | Include metadata | `true` | `false` |
| `--compress` | Compress output | `false` | `true` |

#### Examples

```bash
# Basic generation
tsiot-cli generate --sensor-type temperature --points 1000

# Generate with specific time range
tsiot-cli generate \
  --generator arima \
  --start-time "2024-01-01T00:00:00Z" \
  --end-time "2024-01-31T23:59:59Z" \
  --frequency 1h

# Generate with anomalies and seasonality
tsiot-cli generate \
  --generator timegan \
  --sensor-type energy \
  --points 10000 \
  --anomalies 20 \
  --anomaly-magnitude 2.5 \
  --seasonality \
  --seasonal-period 24h \
  --output energy_data.csv

# Privacy-preserving generation
tsiot-cli generate \
  --generator ydata \
  --privacy \
  --privacy-epsilon 0.5 \
  --points 5000
```

### validate

Validate synthetic data quality.

```bash
tsiot-cli validate [options]
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input data file | required |
| `--reference` | Reference data for comparison | none |
| `--metrics` | Validation metrics | `basic` |
| `--statistical-tests` | Statistical tests to run | `all` |
| `--quality-threshold` | Minimum quality score | `0.8` |
| `--report-format` | Report format | `json` |
| `--report-output` | Report file path | `stdout` |
| `--visualize` | Generate visualizations | `false` |
| `--parallel` | Parallel validation | `true` |

#### Available Metrics

- `basic` - Mean, variance, range
- `distribution` - Skewness, kurtosis, quantiles
- `temporal` - Autocorrelation, trends, seasonality
- `similarity` - DTW, correlation, KS test
- `privacy` - Re-identification risk, utility
- `all` - All available metrics

#### Examples

```bash
# Basic validation
tsiot-cli validate --input data.csv

# Compare with reference
tsiot-cli validate \
  --input synthetic.csv \
  --reference real.csv \
  --metrics all

# Generate HTML report with visualizations
tsiot-cli validate \
  --input data.csv \
  --report-format html \
  --visualize \
  --report-output validation_report.html

# Check specific metrics
tsiot-cli validate \
  --input data.csv \
  --metrics "temporal,distribution" \
  --statistical-tests "ks,anderson"
```

### analyze

Analyze time series data.

```bash
tsiot-cli analyze [options]
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input data file | required |
| `--detect-anomalies` | Detect anomalies | `false` |
| `--anomaly-method` | Detection method | `isolation_forest` |
| `--decompose` | Decompose series | `false` |
| `--forecast` | Generate forecast | `false` |
| `--forecast-periods` | Periods to forecast | `10` |
| `--forecast-method` | Forecasting method | `auto` |
| `--patterns` | Detect patterns | `false` |
| `--summary-stats` | Show summary statistics | `true` |

#### Examples

```bash
# Basic analysis
tsiot-cli analyze --input data.csv

# Anomaly detection
tsiot-cli analyze \
  --input sensor_data.csv \
  --detect-anomalies \
  --anomaly-method isolation_forest

# Time series decomposition and forecast
tsiot-cli analyze \
  --input temperature.csv \
  --decompose \
  --forecast \
  --forecast-periods 24 \
  --output analysis_report.json
```

### stream

Stream synthetic data in real-time.

```bash
tsiot-cli stream [options]
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--generator` | Generation algorithm | `statistical` |
| `--sensor-type` | Type of sensor | `temperature` |
| `--frequency` | Stream frequency | `1s` |
| `--buffer-size` | Output buffer size | `100` |
| `--websocket` | WebSocket endpoint | none |
| `--kafka-topic` | Kafka topic | none |
| `--mqtt-topic` | MQTT topic | none |

#### Examples

```bash
# Stream to stdout
tsiot-cli stream --frequency 5s

# Stream to Kafka
tsiot-cli stream \
  --generator arima \
  --sensor-type humidity \
  --frequency 1s \
  --kafka-topic iot-sensors

# Stream to WebSocket
tsiot-cli stream \
  --websocket ws://localhost:8080/stream \
  --sensor-type temperature
```

### convert

Convert between data formats.

```bash
tsiot-cli convert [options]
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input file | required |
| `--output` | Output file | required |
| `--input-format` | Input format | `auto` |
| `--output-format` | Output format | `auto` |
| `--timezone` | Timezone conversion | `UTC` |
| `--resample` | Resample frequency | none |
| `--aggregate` | Aggregation method | `mean` |

#### Supported Formats

- `csv` - Comma-separated values
- `json` - JSON array or JSONL
- `parquet` - Apache Parquet
- `influx` - InfluxDB line protocol
- `prometheus` - Prometheus exposition format

#### Examples

```bash
# Convert CSV to Parquet
tsiot-cli convert \
  --input data.csv \
  --output data.parquet

# Convert with resampling
tsiot-cli convert \
  --input high_freq.json \
  --output hourly.csv \
  --resample 1h \
  --aggregate mean

# Convert timezone
tsiot-cli convert \
  --input utc_data.csv \
  --output local_data.csv \
  --timezone America/New_York
```

### migrate

Migrate data between storage backends.

```bash
tsiot-cli migrate [options]
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--source` | Source URL | required |
| `--dest` | Destination URL | required |
| `--query` | Filter query | none |
| `--start-time` | Start time filter | none |
| `--end-time` | End time filter | none |
| `--batch-size` | Migration batch size | `1000` |
| `--parallel` | Parallel workers | `4` |
| `--dry-run` | Preview migration | `false` |
| `--verify` | Verify after migration | `true` |

#### URL Formats

- File: `file:///path/to/data.csv`
- InfluxDB: `influx://user:pass@host:8086/db`
- TimescaleDB: `timescale://user:pass@host:5432/db`
- S3: `s3://bucket/prefix`

#### Examples

```bash
# Migrate from file to InfluxDB
tsiot-cli migrate \
  --source file:///data/sensors.csv \
  --dest influx://localhost:8086/iot

# Migrate with time filter
tsiot-cli migrate \
  --source influx://old-server:8086/sensors \
  --dest timescale://new-server:5432/sensors \
  --start-time "2024-01-01" \
  --end-time "2024-12-31"

# Dry run
tsiot-cli migrate \
  --source s3://old-bucket/data \
  --dest s3://new-bucket/archived \
  --dry-run
```

### config

Manage configuration.

```bash
tsiot-cli config [subcommand] [options]
```

#### Subcommands

##### config init

Initialize configuration.

```bash
tsiot-cli config init [--force]
```

##### config get

Get configuration value.

```bash
tsiot-cli config get <key>
```

##### config set

Set configuration value.

```bash
tsiot-cli config set <key> <value>
```

##### config list

List all configuration.

```bash
tsiot-cli config list [--show-defaults]
```

#### Examples

```bash
# Initialize config
tsiot-cli config init

# Set server URL
tsiot-cli config set server.url http://tsiot.example.com:8080

# Get current storage backend
tsiot-cli config get storage.backend

# List all settings
tsiot-cli config list --show-defaults
```

### server

Manage TSIoT server.

```bash
tsiot-cli server [subcommand] [options]
```

#### Subcommands

##### server start

Start server.

```bash
tsiot-cli server start [--daemon] [--port 8080]
```

##### server stop

Stop server.

```bash
tsiot-cli server stop [--force]
```

##### server status

Check server status.

```bash
tsiot-cli server status [--detailed]
```

##### server logs

View server logs.

```bash
tsiot-cli server logs [--follow] [--tail 100]
```

#### Examples

```bash
# Start server in background
tsiot-cli server start --daemon

# Check status
tsiot-cli server status --detailed

# Follow logs
tsiot-cli server logs --follow --tail 50

# Stop server
tsiot-cli server stop
```

### plugin

Manage plugins.

```bash
tsiot-cli plugin [subcommand] [options]
```

#### Subcommands

- `install` - Install a plugin
- `uninstall` - Remove a plugin
- `list` - List installed plugins
- `search` - Search available plugins
- `info` - Show plugin information

#### Examples

```bash
# Install plugin
tsiot-cli plugin install custom-generator

# List plugins
tsiot-cli plugin list

# Search plugins
tsiot-cli plugin search anomaly
```

## Environment Variables

TSIoT CLI respects the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `TSIOT_CONFIG` | Config file path | `~/.tsiot/config.yaml` |
| `TSIOT_SERVER` | Server URL | `http://localhost:8080` |
| `TSIOT_LOG_LEVEL` | Log level | `info` |
| `TSIOT_NO_COLOR` | Disable colors | `false` |
| `TSIOT_CACHE_DIR` | Cache directory | `~/.tsiot/cache` |
| `TSIOT_PLUGIN_DIR` | Plugin directory | `~/.tsiot/plugins` |

## Output Formats

### JSON Format

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "sensor_id": "sensor_001",
  "sensor_type": "temperature",
  "value": 23.5,
  "unit": "celsius",
  "metadata": {
    "location": "room_1",
    "quality": 0.95
  }
}
```

### CSV Format

```csv
timestamp,sensor_id,sensor_type,value,unit
2024-01-15T10:30:00Z,sensor_001,temperature,23.5,celsius
2024-01-15T10:31:00Z,sensor_001,temperature,23.6,celsius
```

### Table Format

```
                     ,            ,             ,       ,         
 TIMESTAMP            SENSOR ID   SENSOR TYPE  VALUE  UNIT    
                     <            <             <       <         $
 2024-01-15 10:30:00  sensor_001  temperature  23.5   celsius 
 2024-01-15 10:31:00  sensor_001  temperature  23.6   celsius 
                     4            4             4       4         
```

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | General error |
| 2 | Misuse of shell command |
| 3 | Configuration error |
| 4 | Connection error |
| 5 | Authentication error |
| 6 | Input/output error |
| 7 | Insufficient permissions |
| 8 | Resource not found |
| 9 | Operation timeout |
| 10 | Validation failed |

## Advanced Usage

### Chaining Commands

```bash
# Generate, validate, and convert in pipeline
tsiot-cli generate --generator timegan --points 1000 --output - | \
tsiot-cli validate --input - --output - | \
tsiot-cli convert --input - --output final.parquet --output-format parquet
```

### Using with jq

```bash
# Extract specific fields
tsiot-cli generate --format json | jq '.[] | {time: .timestamp, val: .value}'

# Filter anomalies
tsiot-cli analyze --input data.csv --detect-anomalies --format json | \
jq '.anomalies[] | select(.score > 0.9)'
```

### Batch Processing

```bash
#!/bin/bash
# Process multiple files
for file in data/*.csv; do
  echo "Processing $file..."
  tsiot-cli validate --input "$file" \
    --report-output "reports/$(basename $file .csv)_report.json"
done
```

### Integration with Other Tools

```bash
# Send to Prometheus
tsiot-cli stream --format prometheus | \
curl -X POST http://prometheus:9091/metrics/job/tsiot/instance/sensor1 --data-binary @-

# Import to InfluxDB
tsiot-cli generate --format influx | \
influx write --bucket sensors --precision s
```

## Tips and Tricks

1. **Use aliases for common commands**:
   ```bash
   alias tsg='tsiot-cli generate'
   alias tsv='tsiot-cli validate'
   ```

2. **Save common configurations**:
   ```bash
   tsiot-cli config set defaults.generator timegan
   tsiot-cli config set defaults.sensor_type temperature
   ```

3. **Use `--dry-run` for testing**:
   ```bash
   tsiot-cli migrate --source old --dest new --dry-run
   ```

4. **Enable debug logging for troubleshooting**:
   ```bash
   TSIOT_LOG_LEVEL=debug tsiot-cli generate --verbose
   ```

5. **Use `--seed` for reproducible results**:
   ```bash
   tsiot-cli generate --seed 42 --points 1000
   ```