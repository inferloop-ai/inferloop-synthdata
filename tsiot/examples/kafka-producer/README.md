# TSIOT Kafka Producer Examples

## Overview

This directory contains examples demonstrating how to integrate TSIOT with Apache Kafka for real-time streaming of synthetic time series data. These examples show various patterns for producing time series data to Kafka topics.

## Prerequisites

### Required Software
- **Apache Kafka** 2.8+
- **Go** 1.21+ (for Go examples)
- **Python** 3.8+ (for Python examples)
- **Docker** and **Docker Compose** (for local setup)

### Dependencies
```bash
# Go dependencies
go mod download

# Python dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Start Kafka with Docker Compose
```bash
# Start Kafka and Zookeeper
docker-compose up -d

# Verify services are running
docker-compose ps
```

### 2. Create Kafka Topics
```bash
# Create topics for time series data
docker-compose exec kafka kafka-topics.sh \
  --create \
  --topic tsiot-timeseries \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1

docker-compose exec kafka kafka-topics.sh \
  --create \
  --topic tsiot-alerts \
  --bootstrap-server localhost:9092 \
  --partitions 1 \
  --replication-factor 1

docker-compose exec kafka kafka-topics.sh \
  --create \
  --topic tsiot-metrics \
  --bootstrap-server localhost:9092 \
  --partitions 2 \
  --replication-factor 1
```

### 3. Run Examples
```bash
# Go producer example
go run producer/main.go

# Python producer example
python producer/main.py

# Consumer example
go run consumer/main.go
```

## Examples

### Basic Producer (Go)

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "time"

    "github.com/segmentio/kafka-go"
    "github.com/your-org/tsiot/pkg/generators"
)

type TimeSeriesMessage struct {
    ID          string                 `json:"id"`
    GeneratorID string                 `json:"generator_id"`
    Timestamp   time.Time              `json:"timestamp"`
    Values      []float64              `json:"values"`
    Metadata    map[string]interface{} `json:"metadata"`
}

func main() {
    // Configure Kafka writer
    writer := &kafka.Writer{
        Addr:         kafka.TCP("localhost:9092"),
        Topic:        "tsiot-timeseries",
        Balancer:     &kafka.LeastBytes{},
        BatchTimeout: 10 * time.Millisecond,
        BatchSize:    100,
    }
    defer writer.Close()

    // Initialize TSIOT generator
    generator := generators.NewLSTMGenerator(&generators.LSTMConfig{
        SequenceLength: 100,
        Features:       1,
        HiddenSize:     50,
        NumLayers:      2,
    })

    ctx := context.Background()

    // Generate and send time series data
    for i := 0; i < 1000; i++ {
        // Generate synthetic data
        data, err := generator.Generate(ctx, &generators.GenerationRequest{
            Length: 100,
            Parameters: map[string]interface{}{
                "trend":       0.1,
                "seasonality": 24,
                "noise":       0.05,
            },
        })
        if err != nil {
            log.Printf("Generation failed: %v", err)
            continue
        }

        // Create message
        message := TimeSeriesMessage{
            ID:          fmt.Sprintf("ts-%d", i),
            GeneratorID: "lstm-generator",
            Timestamp:   time.Now(),
            Values:      data.Values,
            Metadata: map[string]interface{}{
                "generator_type": "lstm",
                "sequence_id":    i,
                "quality_score":  data.QualityScore,
            },
        }

        // Serialize to JSON
        payload, err := json.Marshal(message)
        if err != nil {
            log.Printf("JSON marshal failed: %v", err)
            continue
        }

        // Send to Kafka
        err = writer.WriteMessages(ctx, kafka.Message{
            Key:   []byte(message.ID),
            Value: payload,
            Headers: []kafka.Header{
                {Key: "content-type", Value: []byte("application/json")},
                {Key: "generator-type", Value: []byte("lstm")},
            },
        })

        if err != nil {
            log.Printf("Failed to send message: %v", err)
            continue
        }

        fmt.Printf("Sent time series %s with %d data points\n", 
                   message.ID, len(message.Values))

        time.Sleep(100 * time.Millisecond)
    }
}
```

### Batch Producer (Go)

```go
package main

import (
    "context"
    "encoding/json"
    "log"
    "sync"
    "time"

    "github.com/segmentio/kafka-go"
    "github.com/your-org/tsiot/pkg/generators"
)

type BatchProducer struct {
    writer     *kafka.Writer
    batchSize  int
    messages   []kafka.Message
    mu         sync.Mutex
    flushTimer *time.Timer
}

func NewBatchProducer(brokers []string, topic string, batchSize int) *BatchProducer {
    return &BatchProducer{
        writer: &kafka.Writer{
            Addr:         kafka.TCP(brokers...),
            Topic:        topic,
            Balancer:     &kafka.Hash{},
            BatchTimeout: 1 * time.Second,
            BatchSize:    batchSize,
        },
        batchSize: batchSize,
        messages:  make([]kafka.Message, 0, batchSize),
    }
}

func (bp *BatchProducer) AddMessage(key, value []byte, headers []kafka.Header) {
    bp.mu.Lock()
    defer bp.mu.Unlock()

    bp.messages = append(bp.messages, kafka.Message{
        Key:     key,
        Value:   value,
        Headers: headers,
    })

    if len(bp.messages) >= bp.batchSize {
        bp.flush()
    } else if bp.flushTimer == nil {
        bp.flushTimer = time.AfterFunc(5*time.Second, func() {
            bp.mu.Lock()
            bp.flush()
            bp.mu.Unlock()
        })
    }
}

func (bp *BatchProducer) flush() {
    if len(bp.messages) == 0 {
        return
    }

    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    err := bp.writer.WriteMessages(ctx, bp.messages...)
    if err != nil {
        log.Printf("Failed to send batch: %v", err)
    } else {
        log.Printf("Sent batch of %d messages", len(bp.messages))
    }

    bp.messages = bp.messages[:0]
    if bp.flushTimer != nil {
        bp.flushTimer.Stop()
        bp.flushTimer = nil
    }
}

func (bp *BatchProducer) Close() {
    bp.mu.Lock()
    defer bp.mu.Unlock()
    bp.flush()
    bp.writer.Close()
}

func main() {
    producer := NewBatchProducer(
        []string{"localhost:9092"}, 
        "tsiot-timeseries", 
        50,
    )
    defer producer.Close()

    generator := generators.NewARIMAGenerator(&generators.ARIMAConfig{
        ARParams: []float64{0.5, -0.3},
        MAParams: []float64{0.2},
        Seasonal: true,
        Period:   24,
    })

    ctx := context.Background()

    // Generate batches of time series
    for batch := 0; batch < 20; batch++ {
        for i := 0; i < 50; i++ {
            data, err := generator.Generate(ctx, &generators.GenerationRequest{
                Length: 200,
                Parameters: map[string]interface{}{
                    "mean":     100.0,
                    "variance": 25.0,
                },
            })
            if err != nil {
                continue
            }

            message := TimeSeriesMessage{
                ID:          fmt.Sprintf("batch-%d-ts-%d", batch, i),
                GeneratorID: "arima-generator",
                Timestamp:   time.Now(),
                Values:      data.Values,
                Metadata: map[string]interface{}{
                    "batch_id":     batch,
                    "sequence_id":  i,
                    "generator":    "arima",
                },
            }

            payload, _ := json.Marshal(message)
            producer.AddMessage(
                []byte(message.ID),
                payload,
                []kafka.Header{
                    {Key: "batch-id", Value: []byte(fmt.Sprintf("%d", batch))},
                    {Key: "generator-type", Value: []byte("arima")},
                },
            )
        }

        log.Printf("Queued batch %d", batch)
        time.Sleep(2 * time.Second)
    }
}
```

### Python Producer Example

```python
import json
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, List
from kafka import KafkaProducer
from kafka.errors import KafkaError
import numpy as np

class TSIOTKafkaProducer:
    def __init__(self, bootstrap_servers: List[str], topic: str):
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            batch_size=16384,
            linger_ms=10,
            compression_type='gzip',
            retries=3,
            acks='all'
        )
    
    def generate_synthetic_series(self, length: int, 
                                series_type: str = 'random_walk') -> List[float]:
        """Generate synthetic time series data."""
        if series_type == 'random_walk':
            steps = np.random.normal(0, 1, length)
            return np.cumsum(steps).tolist()
        
        elif series_type == 'seasonal':
            t = np.arange(length)
            trend = 0.1 * t
            seasonal = 10 * np.sin(2 * np.pi * t / 24)
            noise = np.random.normal(0, 1, length)
            return (trend + seasonal + noise).tolist()
        
        elif series_type == 'exponential':
            base = np.random.uniform(0.98, 1.02, length)
            return np.cumprod(base).tolist()
        
        else:
            return np.random.randn(length).tolist()
    
    def send_time_series(self, series_id: str, values: List[float], 
                        metadata: Dict[str, Any] = None) -> bool:
        """Send time series data to Kafka."""
        message = {
            'id': series_id,
            'timestamp': datetime.utcnow().isoformat(),
            'values': values,
            'metadata': metadata or {}
        }
        
        try:
            future = self.producer.send(
                self.topic,
                key=series_id,
                value=message,
                headers=[
                    ('content-type', b'application/json'),
                    ('series-type', metadata.get('type', 'unknown').encode())
                ]
            )
            
            # Wait for send to complete
            future.get(timeout=10)
            return True
            
        except KafkaError as e:
            print(f"Failed to send message: {e}")
            return False
    
    def send_batch(self, series_data: List[Dict[str, Any]]) -> int:
        """Send multiple time series in batch."""
        sent_count = 0
        
        for data in series_data:
            if self.send_time_series(**data):
                sent_count += 1
        
        # Flush to ensure all messages are sent
        self.producer.flush()
        return sent_count
    
    def close(self):
        """Close the producer."""
        self.producer.close()

async def main():
    # Initialize producer
    producer = TSIOTKafkaProducer(
        bootstrap_servers=['localhost:9092'],
        topic='tsiot-timeseries'
    )
    
    try:
        # Generate different types of time series
        series_types = ['random_walk', 'seasonal', 'exponential']
        
        for i in range(100):
            series_type = series_types[i % len(series_types)]
            values = producer.generate_synthetic_series(
                length=200, 
                series_type=series_type
            )
            
            success = producer.send_time_series(
                series_id=f"python-{series_type}-{i}",
                values=values,
                metadata={
                    'type': series_type,
                    'length': len(values),
                    'generator': 'python-synthetic',
                    'created_at': datetime.utcnow().isoformat()
                }
            )
            
            if success:
                print(f"Sent {series_type} series {i} with {len(values)} points")
            
            await asyncio.sleep(0.1)
    
    finally:
        producer.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Consumer Example (Go)

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"

    "github.com/segmentio/kafka-go"
)

func main() {
    // Configure Kafka reader
    reader := kafka.NewReader(kafka.ReaderConfig{
        Brokers:   []string{"localhost:9092"},
        Topic:     "tsiot-timeseries",
        GroupID:   "tsiot-consumer-group",
        Partition: 0,
        MinBytes:  10e3, // 10KB
        MaxBytes:  10e6, // 10MB
    })
    defer reader.Close()

    fmt.Println("Starting TSIOT Kafka consumer...")

    for {
        // Read message
        message, err := reader.ReadMessage(context.Background())
        if err != nil {
            log.Printf("Error reading message: %v", err)
            continue
        }

        // Parse time series message
        var tsMessage TimeSeriesMessage
        if err := json.Unmarshal(message.Value, &tsMessage); err != nil {
            log.Printf("Error parsing message: %v", err)
            continue
        }

        // Process the time series data
        fmt.Printf("Received time series %s with %d data points\n",
                   tsMessage.ID, len(tsMessage.Values))

        // Example processing: calculate basic statistics
        if len(tsMessage.Values) > 0 {
            var sum, min, max float64
            min = tsMessage.Values[0]
            max = tsMessage.Values[0]

            for _, value := range tsMessage.Values {
                sum += value
                if value < min {
                    min = value
                }
                if value > max {
                    max = value
                }
            }

            mean := sum / float64(len(tsMessage.Values))
            
            fmt.Printf("  Statistics: mean=%.2f, min=%.2f, max=%.2f\n", 
                       mean, min, max)
        }

        // Print metadata
        if len(tsMessage.Metadata) > 0 {
            fmt.Printf("  Metadata: %+v\n", tsMessage.Metadata)
        }
    }
}
```

## Configuration

### Docker Compose Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "2181:2181"

  kafka:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
      KAFKA_NUM_PARTITIONS: 3

  kafka-ui:
    image: provectuslabs/kafka-ui:latest
    depends_on:
      - kafka
    ports:
      - "8080:8080"
    environment:
      KAFKA_CLUSTERS_0_NAME: local
      KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: kafka:9092
```

### Producer Configuration

```go
// producer_config.go
type ProducerConfig struct {
    Brokers       []string      `yaml:"brokers"`
    Topic         string        `yaml:"topic"`
    BatchSize     int           `yaml:"batch_size"`
    BatchTimeout  time.Duration `yaml:"batch_timeout"`
    Compression   string        `yaml:"compression"`
    Retries       int           `yaml:"retries"`
    Acks          string        `yaml:"acks"`
    EnableMetrics bool          `yaml:"enable_metrics"`
}

func DefaultProducerConfig() *ProducerConfig {
    return &ProducerConfig{
        Brokers:       []string{"localhost:9092"},
        Topic:         "tsiot-timeseries",
        BatchSize:     100,
        BatchTimeout:  10 * time.Millisecond,
        Compression:   "gzip",
        Retries:       3,
        Acks:          "all",
        EnableMetrics: true,
    }
}
```

## Message Schemas

### Time Series Message Schema

```json
{
  "id": "string",
  "generator_id": "string", 
  "timestamp": "2023-01-01T00:00:00Z",
  "values": [1.0, 2.0, 3.0],
  "metadata": {
    "generator_type": "lstm|arima|rnn",
    "sequence_id": 123,
    "quality_score": 0.95,
    "parameters": {
      "trend": 0.1,
      "seasonality": 24,
      "noise": 0.05
    }
  }
}
```

### Alert Message Schema

```json
{
  "id": "string",
  "series_id": "string",
  "alert_type": "anomaly|threshold|pattern",
  "severity": "low|medium|high|critical",
  "timestamp": "2023-01-01T00:00:00Z",
  "description": "string",
  "value": 123.45,
  "threshold": 100.0,
  "metadata": {
    "detector": "statistical|ml|rule_based",
    "confidence": 0.95
  }
}
```

## Performance Optimization

### Producer Optimization

```go
// High-throughput producer configuration
writer := &kafka.Writer{
    Addr:         kafka.TCP("localhost:9092"),
    Topic:        "tsiot-timeseries",
    Balancer:     &kafka.Hash{}, // Consistent partitioning
    BatchSize:    1000,          // Larger batches
    BatchTimeout: 100 * time.Millisecond,
    RequiredAcks: kafka.RequireOne, // Faster acknowledgment
    Compression:  kafka.Gzip,    // Compress large time series
    WriteTimeout: 10 * time.Second,
    ReadTimeout:  10 * time.Second,
}
```

### Consumer Optimization

```go
// High-throughput consumer configuration
reader := kafka.NewReader(kafka.ReaderConfig{
    Brokers:     []string{"localhost:9092"},
    Topic:       "tsiot-timeseries",
    GroupID:     "tsiot-consumer-group",
    MinBytes:    10e3,  // 10KB minimum
    MaxBytes:    10e6,  // 10MB maximum
    MaxWait:     1 * time.Second,
    CommitInterval: 1 * time.Second,
})
```

## Monitoring and Metrics

### Producer Metrics

```go
type ProducerMetrics struct {
    MessagesSent      int64
    MessagesSucceeded int64
    MessagesFailed    int64
    BytesSent         int64
    AvgLatency        time.Duration
    ErrorRate         float64
}

func (m *ProducerMetrics) RecordSuccess(latency time.Duration, bytes int) {
    atomic.AddInt64(&m.MessagesSent, 1)
    atomic.AddInt64(&m.MessagesSucceeded, 1)
    atomic.AddInt64(&m.BytesSent, int64(bytes))
    // Update average latency...
}
```

### Health Checks

```bash
#!/bin/bash
# health_check.sh

# Check Kafka connectivity
kafka-topics.sh --bootstrap-server localhost:9092 --list

# Check consumer lag
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --group tsiot-consumer-group --describe

# Check topic details
kafka-topics.sh --bootstrap-server localhost:9092 \
  --topic tsiot-timeseries --describe
```

## Troubleshooting

### Common Issues

**Producer Connection Errors**
```bash
# Check Kafka broker status
docker-compose logs kafka

# Test connectivity
telnet localhost 9092
```

**Consumer Lag Issues**
```bash
# Check consumer group status
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --group tsiot-consumer-group --describe

# Reset consumer offset if needed
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --group tsiot-consumer-group --reset-offsets --to-earliest \
  --topic tsiot-timeseries --execute
```

**Performance Issues**
```bash
# Monitor Kafka performance
kafka-run-class.sh kafka.tools.JmxTool \
  --object-name kafka.server:type=BrokerTopicMetrics,name=MessagesInPerSec

# Check disk usage
df -h /var/kafka-logs
```

## Integration with TSIOT API

### REST API Integration

```go
// api_integration.go
func (h *Handler) StreamToKafka(c *gin.Context) {
    var request GenerationRequest
    if err := c.ShouldBindJSON(&request); err != nil {
        c.JSON(400, gin.H{"error": err.Error()})
        return
    }

    // Generate time series
    data, err := h.generator.Generate(c.Request.Context(), &request)
    if err != nil {
        c.JSON(500, gin.H{"error": err.Error()})
        return
    }

    // Send to Kafka
    err = h.kafkaProducer.SendTimeSeries(data.ID, data.Values, data.Metadata)
    if err != nil {
        c.JSON(500, gin.H{"error": "Failed to send to Kafka"})
        return
    }

    c.JSON(200, gin.H{
        "id": data.ID,
        "status": "sent_to_kafka",
        "topic": "tsiot-timeseries",
    })
}
```

## Best Practices

1. **Partitioning Strategy**: Use consistent hashing based on series ID for ordered processing
2. **Error Handling**: Implement dead letter queues for failed messages
3. **Monitoring**: Set up comprehensive metrics and alerting
4. **Schema Evolution**: Use Avro or Protobuf for schema management
5. **Security**: Enable SASL/SSL for production deployments
6. **Backup**: Regular topic backups and disaster recovery planning

## Next Steps

- Explore [Schema Registry integration](../schema-registry/)
- Check out [Stream processing examples](../stream-processing/)
- Review [Kafka Connect examples](../kafka-connect/)
- Learn about [Multi-cluster deployment](../multi-cluster/)

For more information, see the [TSIOT documentation](../../docs/).