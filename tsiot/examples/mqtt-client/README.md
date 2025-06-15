# TSIOT MQTT Client Examples

## Overview

This directory contains examples demonstrating how to integrate TSIOT with MQTT for IoT scenarios. These examples show how to publish synthetic time series data to MQTT brokers and subscribe to real-time data streams.

## Prerequisites

### Required Software
- **MQTT Broker** (Mosquitto, HiveMQ, or AWS IoT Core)
- **Go** 1.21+ (for Go examples)
- **Python** 3.8+ (for Python examples)
- **Docker** (for local MQTT broker)

### Dependencies
```bash
# Go dependencies
go mod download

# Python dependencies
pip install paho-mqtt pandas streamlit plotly
```

## Quick Start

### 1. Start Local MQTT Broker
```bash
# Using Docker
docker run -it -p 1883:1883 -p 9001:9001 eclipse-mosquitto

# Or install locally
sudo apt-get install mosquitto mosquitto-clients
sudo systemctl start mosquitto
```

### 2. Run Examples
```bash
# Go publisher
go run publisher/main.go

# Python subscriber  
python subscriber/main.py

# Real-time dashboard
streamlit run dashboard.py
```

## Message Schema

### IoT Sensor Message
```json
{
  "device_id": "device-001",
  "sensor_type": "temperature", 
  "timestamp": "2023-01-01T12:00:00Z",
  "value": 22.5,
  "unit": "°C",
  "location": {
    "latitude": 37.7749,
    "longitude": -122.4194
  },
  "metadata": {
    "firmware_version": "1.2.3",
    "battery_level": 85,
    "signal_strength": -45
  }
}
```

## Topic Structure
```
iot/
  sensors/
    device-001/
      temperature
      humidity
      pressure
      vibration
  alerts/
    device-001
  status/
    device-001
```

## Examples Included

1. **Basic Publisher** (`publisher/main.go`) - Simple MQTT publisher using TSIOT generators
2. **Data Collector** (`subscriber/main.py`) - MQTT subscriber with database storage
3. **Real-time Dashboard** (`dashboard.py`) - Streamlit dashboard for live data visualization
4. **Batch Publisher** (`batch_publisher.go`) - High-throughput batch publishing
5. **Alert System** (`alert_handler.py`) - Anomaly detection and alerting

## Configuration

### Environment Variables
```bash
export MQTT_BROKER="localhost:1883"
export MQTT_USERNAME=""  # Optional
export MQTT_PASSWORD=""  # Optional
export MQTT_CLIENT_ID="tsiot-client"
```

### Security (Production)
```bash
# TLS Configuration
export MQTT_TLS_ENABLED=true
export MQTT_CA_CERT="ca.crt"
export MQTT_CLIENT_CERT="client.crt" 
export MQTT_CLIENT_KEY="client.key"
```

## Performance

- **Throughput**: Up to 10,000 messages/second per client
- **Latency**: <10ms for local brokers
- **Scalability**: Supports hundreds of concurrent devices
- **Reliability**: QoS levels 0, 1, and 2 supported

## Integration

### Cloud Platforms
- AWS IoT Core
- Azure IoT Hub  
- Google Cloud IoT Core
- HiveMQ Cloud

### Monitoring
- Prometheus metrics export
- Grafana dashboards
- Real-time alerting
- Performance monitoring

For detailed examples and documentation, see individual files in this directory.