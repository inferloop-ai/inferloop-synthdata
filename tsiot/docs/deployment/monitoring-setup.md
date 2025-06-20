# TSIOT Monitoring Setup Guide

## Overview

This guide provides comprehensive monitoring setup for TSIOT using Prometheus, Grafana, Jaeger, and other observability tools. It covers metrics collection, alerting, distributed tracing, and log aggregation.

## Architecture

```
                                                
   TSIOT        �  Prometheus     �   Grafana   
  Service           (Metrics)       (Dashboard) 
                                                
                           
       �                    �
                               
   Jaeger         AlertManager 
  (Tracing)       (Alerting)   
                               
       
       �
                     
     ELK Stack       
 (Log Aggregation)   
                     
```

## Quick Setup

### Docker Compose Monitoring Stack

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: tsiot-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/prometheus/rules:/etc/prometheus/rules
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:10.0.0
    container_name: tsiot-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
    depends_on:
      - prometheus
    networks:
      - monitoring

  alertmanager:
    image: prom/alertmanager:v0.26.0
    container_name: tsiot-alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager_data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
      - '--web.external-url=http://localhost:9093'
    networks:
      - monitoring

  jaeger:
    image: jaegertracing/all-in-one:1.47
    container_name: tsiot-jaeger
    ports:
      - "16686:16686"
      - "14268:14268"
      - "14250:14250"
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    volumes:
      - jaeger_data:/badger
    networks:
      - monitoring

  node-exporter:
    image: prom/node-exporter:v1.6.0
    container_name: tsiot-node-exporter
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    networks:
      - monitoring

  postgres-exporter:
    image: prometheuscommunity/postgres-exporter:v0.13.2
    container_name: tsiot-postgres-exporter
    ports:
      - "9187:9187"
    environment:
      - DATA_SOURCE_NAME=postgresql://tsiot:password@postgres:5432/tsiot?sslmode=disable
    depends_on:
      - postgres
    networks:
      - monitoring

  redis-exporter:
    image: oliver006/redis_exporter:v1.53.0
    container_name: tsiot-redis-exporter
    ports:
      - "9121:9121"
    environment:
      - REDIS_ADDR=redis://redis:6379
    depends_on:
      - redis
    networks:
      - monitoring

volumes:
  prometheus_data:
  grafana_data:
  alertmanager_data:
  jaeger_data:

networks:
  monitoring:
    driver: bridge
```

## Prometheus Configuration

### Main Configuration

```yaml
# monitoring/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'tsiot-cluster'
    replica: 'prometheus-01'

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # TSIOT Application
  - job_name: 'tsiot-api'
    static_configs:
      - targets: ['tsiot-api:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s
    scrape_timeout: 10s
    params:
      format: ['prometheus']

  # TSIOT Workers
  - job_name: 'tsiot-workers'
    static_configs:
      - targets: ['tsiot-worker-1:8081', 'tsiot-worker-2:8081']
    metrics_path: '/metrics'

  # System Metrics
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  # Database Metrics
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  # Kubernetes (if applicable)
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - tsiot
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)

  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # Grafana
  - job_name: 'grafana'
    static_configs:
      - targets: ['grafana:3000']

  # Jaeger
  - job_name: 'jaeger'
    static_configs:
      - targets: ['jaeger:14269']
```

### Alert Rules

```yaml
# monitoring/prometheus/rules/tsiot_alerts.yml
groups:
  - name: tsiot.rules
    rules:
      # Service Health
      - alert: TSIOTServiceDown
        expr: up{job="tsiot-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "TSIOT service is down"
          description: "TSIOT service has been down for more than 1 minute."

      # High Error Rate
      - alert: TSIOTHighErrorRate
        expr: rate(tsiot_http_requests_total{status=~"5.."}[5m]) / rate(tsiot_http_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} for {{ $labels.instance }}"

      # High Latency
      - alert: TSIOTHighLatency
        expr: histogram_quantile(0.95, rate(tsiot_http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "95th percentile latency is {{ $value }}s for {{ $labels.instance }}"

      # High Memory Usage
      - alert: TSIOTHighMemoryUsage
        expr: (tsiot_process_resident_memory_bytes / tsiot_process_virtual_memory_max_bytes) > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value | humanizePercentage }} for {{ $labels.instance }}"

      # High CPU Usage
      - alert: TSIOTHighCPUUsage
        expr: rate(tsiot_process_cpu_seconds_total[5m]) > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage"
          description: "CPU usage is {{ $value | humanizePercentage }} for {{ $labels.instance }}"

      # Database Connection Issues
      - alert: TSIOTDatabaseConnectionFailure
        expr: tsiot_database_connections_failed_total > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database connection failures"
          description: "{{ $value }} database connection failures detected"

      # Queue Depth
      - alert: TSIOTHighQueueDepth
        expr: tsiot_queue_depth > 1000
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High queue depth"
          description: "Queue depth is {{ $value }} items"

      # Generation Failures
      - alert: TSIOTGenerationFailures
        expr: rate(tsiot_generation_failures_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High generation failure rate"
          description: "Generation failure rate is {{ $value }} per second"

  - name: infrastructure.rules
    rules:
      # Disk Space
      - alert: HighDiskUsage
        expr: (node_filesystem_size_bytes - node_filesystem_free_bytes) / node_filesystem_size_bytes > 0.85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High disk usage"
          description: "Disk usage is {{ $value | humanizePercentage }} on {{ $labels.device }}"

      # PostgreSQL
      - alert: PostgreSQLDown
        expr: pg_up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "PostgreSQL is down"
          description: "PostgreSQL instance is down"

      - alert: PostgreSQLHighConnections
        expr: pg_stat_database_numbackends / pg_settings_max_connections > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High PostgreSQL connections"
          description: "PostgreSQL connection usage is {{ $value | humanizePercentage }}"

      # Redis
      - alert: RedisDown
        expr: redis_up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis is down"
          description: "Redis instance is down"

      - alert: RedisHighMemory
        expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High Redis memory usage"
          description: "Redis memory usage is {{ $value | humanizePercentage }}"
```

## AlertManager Configuration

```yaml
# monitoring/alertmanager/alertmanager.yml
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@yourcompany.com'
  smtp_auth_username: 'alerts@yourcompany.com'
  smtp_auth_password: 'your-app-password'

templates:
  - '/etc/alertmanager/templates/*.tmpl'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
      group_wait: 0s
      repeat_interval: 5m
    - match:
        severity: warning
      receiver: 'warning-alerts'
      repeat_interval: 30m

receivers:
  - name: 'web.hook'
    webhook_configs:
      - url: 'http://localhost:5001/'

  - name: 'critical-alerts'
    email_configs:
      - to: 'oncall@yourcompany.com'
        subject: 'CRITICAL: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        channel: '#alerts-critical'
        title: 'CRITICAL Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

  - name: 'warning-alerts'
    email_configs:
      - to: 'team@yourcompany.com'
        subject: 'WARNING: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'instance']
```

## Grafana Configuration

### Data Sources

```yaml
# monitoring/grafana/provisioning/datasources/datasources.yml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    jsonData:
      timeInterval: 15s
      queryTimeout: 60s
    editable: true

  - name: Jaeger
    type: jaeger
    access: proxy
    url: http://jaeger:16686
    jsonData:
      tracesToLogs:
        datasourceUid: Loki
      nodeGraph:
        enabled: true

  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    jsonData:
      derivedFields:
        - datasourceUid: Jaeger
          matcherRegex: "trace_id=(\\w+)"
          name: TraceID
          url: "$${__value.raw}"
```

### Dashboard Provisioning

```yaml
# monitoring/grafana/provisioning/dashboards/dashboards.yml
apiVersion: 1

providers:
  - name: 'TSIOT Dashboards'
    type: file
    folder: 'TSIOT'
    options:
      path: /var/lib/grafana/dashboards/tsiot
    updateIntervalSeconds: 30
    allowUiUpdates: true
    foldersFromFilesStructure: true

  - name: 'Infrastructure Dashboards'
    type: file
    folder: 'Infrastructure'
    options:
      path: /var/lib/grafana/dashboards/infrastructure
    updateIntervalSeconds: 30
    allowUiUpdates: true
```

### Main TSIOT Dashboard

```json
{
  "dashboard": {
    "id": null,
    "title": "TSIOT Overview",
    "tags": ["tsiot"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Request Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(rate(tsiot_http_requests_total[5m]))",
            "legendFormat": "Requests/sec"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "reqps",
            "min": 0
          }
        }
      },
      {
        "title": "Error Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(rate(tsiot_http_requests_total{status=~\"5..\"}[5m])) / sum(rate(tsiot_http_requests_total[5m]))",
            "legendFormat": "Error Rate"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percentunit",
            "min": 0,
            "max": 1,
            "thresholds": {
              "steps": [
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 0.01},
                {"color": "red", "value": 0.05}
              ]
            }
          }
        }
      },
      {
        "title": "Response Time",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(tsiot_http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          },
          {
            "expr": "histogram_quantile(0.95, rate(tsiot_http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.99, rate(tsiot_http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "99th percentile"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "s",
            "min": 0
          }
        }
      },
      {
        "title": "Active Connections",
        "type": "timeseries",
        "targets": [
          {
            "expr": "tsiot_active_connections",
            "legendFormat": "Active Connections"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "timeseries",
        "targets": [
          {
            "expr": "tsiot_process_resident_memory_bytes",
            "legendFormat": "Resident Memory"
          },
          {
            "expr": "tsiot_process_virtual_memory_bytes",
            "legendFormat": "Virtual Memory"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "bytes"
          }
        }
      },
      {
        "title": "CPU Usage",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(tsiot_process_cpu_seconds_total[5m])",
            "legendFormat": "CPU Usage"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percentunit",
            "min": 0,
            "max": 1
          }
        }
      },
      {
        "title": "Generation Metrics",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(tsiot_generations_total[5m])",
            "legendFormat": "Generations/sec"
          },
          {
            "expr": "rate(tsiot_generation_failures_total[5m])",
            "legendFormat": "Failures/sec"
          }
        ]
      },
      {
        "title": "Queue Depth",
        "type": "timeseries",
        "targets": [
          {
            "expr": "tsiot_queue_depth",
            "legendFormat": "Queue Depth"
          }
        ]
      }
    ],
    "refresh": "5s",
    "time": {
      "from": "now-1h",
      "to": "now"
    }
  }
}
```

## Application Instrumentation

### Go Metrics Implementation

```go
// metrics/metrics.go
package metrics

import (
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
)

var (
    // HTTP Metrics
    HttpRequestsTotal = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "tsiot_http_requests_total",
            Help: "Total number of HTTP requests",
        },
        []string{"method", "endpoint", "status"},
    )

    HttpRequestDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "tsiot_http_request_duration_seconds",
            Help:    "HTTP request duration in seconds",
            Buckets: []float64{.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10},
        },
        []string{"method", "endpoint"},
    )

    // Generation Metrics
    GenerationsTotal = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "tsiot_generations_total",
            Help: "Total number of time series generations",
        },
        []string{"generator_type", "status"},
    )

    GenerationDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "tsiot_generation_duration_seconds",
            Help:    "Time series generation duration in seconds",
            Buckets: []float64{.1, .5, 1, 2, 5, 10, 30, 60, 120, 300},
        },
        []string{"generator_type"},
    )

    // Queue Metrics
    QueueDepth = promauto.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "tsiot_queue_depth",
            Help: "Current depth of processing queues",
        },
        []string{"queue_name"},
    )

    // Database Metrics
    DatabaseConnections = promauto.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "tsiot_database_connections",
            Help: "Current number of database connections",
        },
        []string{"state"}, // active, idle, etc.
    )

    DatabaseQueryDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "tsiot_database_query_duration_seconds",
            Help:    "Database query duration in seconds",
            Buckets: []float64{.001, .005, .01, .025, .05, .1, .25, .5, 1, 2, 5},
        },
        []string{"operation"},
    )

    // Cache Metrics
    CacheHits = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "tsiot_cache_hits_total",
            Help: "Total cache hits",
        },
        []string{"cache_type"},
    )

    CacheMisses = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "tsiot_cache_misses_total",
            Help: "Total cache misses",
        },
        []string{"cache_type"},
    )
)

// Middleware for HTTP metrics
func HTTPMetricsMiddleware() gin.HandlerFunc {
    return gin.HandlerFunc(func(c *gin.Context) {
        start := time.Now()
        
        c.Next()
        
        duration := time.Since(start).Seconds()
        status := strconv.Itoa(c.Writer.Status())
        
        HttpRequestsTotal.WithLabelValues(
            c.Request.Method,
            c.FullPath(),
            status,
        ).Inc()
        
        HttpRequestDuration.WithLabelValues(
            c.Request.Method,
            c.FullPath(),
        ).Observe(duration)
    })
}
```

## Distributed Tracing with Jaeger

### OpenTelemetry Configuration

```go
// tracing/tracer.go
package tracing

import (
    "context"
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/exporters/jaeger"
    "go.opentelemetry.io/otel/sdk/resource"
    "go.opentelemetry.io/otel/sdk/trace"
    semconv "go.opentelemetry.io/otel/semconv/v1.17.0"
)

func InitTracer(serviceName, jaegerEndpoint string) (*trace.TracerProvider, error) {
    // Create Jaeger exporter
    exp, err := jaeger.New(jaeger.WithCollectorEndpoint(
        jaeger.WithEndpoint(jaegerEndpoint),
    ))
    if err != nil {
        return nil, err
    }

    // Create resource
    res, err := resource.Merge(
        resource.Default(),
        resource.NewWithAttributes(
            semconv.SchemaURL,
            semconv.ServiceNameKey.String(serviceName),
            semconv.ServiceVersionKey.String("1.0.0"),
        ),
    )
    if err != nil {
        return nil, err
    }

    // Create tracer provider
    tp := trace.NewTracerProvider(
        trace.WithBatcher(exp),
        trace.WithResource(res),
        trace.WithSampler(trace.TraceIDRatioBased(0.1)), // Sample 10% of traces
    )

    otel.SetTracerProvider(tp)
    return tp, nil
}

// Middleware for HTTP tracing
func TracingMiddleware() gin.HandlerFunc {
    return otelgin.Middleware("tsiot-api")
}
```

## Log Aggregation

### ELK Stack Configuration

```yaml
# monitoring/elk/docker-compose.elk.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
    container_name: tsiot-elasticsearch
    environment:
      - node.name=elasticsearch
      - cluster.name=tsiot-logs
      - discovery.type=single-node
      - bootstrap.memory_lock=true
      - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
      - xpack.security.enabled=false
    ulimits:
      memlock:
        soft: -1
        hard: -1
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"
    networks:
      - elk

  logstash:
    image: docker.elastic.co/logstash/logstash:8.8.0
    container_name: tsiot-logstash
    volumes:
      - ./monitoring/logstash/pipeline/logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    ports:
      - "5044:5044"
      - "5000:5000/tcp"
      - "5000:5000/udp"
      - "9600:9600"
    environment:
      LS_JAVA_OPTS: "-Xmx512m -Xms512m"
    depends_on:
      - elasticsearch
    networks:
      - elk

  kibana:
    image: docker.elastic.co/kibana/kibana:8.8.0
    container_name: tsiot-kibana
    ports:
      - "5601:5601"
    environment:
      ELASTICSEARCH_HOSTS: http://elasticsearch:9200
    depends_on:
      - elasticsearch
    networks:
      - elk

  filebeat:
    image: docker.elastic.co/beats/filebeat:8.8.0
    container_name: tsiot-filebeat
    user: root
    volumes:
      - ./monitoring/filebeat/filebeat.yml:/usr/share/filebeat/filebeat.yml:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
    depends_on:
      - logstash
    networks:
      - elk

networks:
  elk:
    driver: bridge

volumes:
  elasticsearch_data:
```

### Logstash Pipeline

```ruby
# monitoring/logstash/pipeline/logstash.conf
input {
  beats {
    port => 5044
  }
  
  tcp {
    port => 5000
    codec => json_lines
  }
}

filter {
  if [fields][service] == "tsiot" {
    json {
      source => "message"
    }
    
    date {
      match => [ "timestamp", "ISO8601" ]
    }
    
    mutate {
      add_field => { "service" => "tsiot" }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "tsiot-logs-%{+YYYY.MM.dd}"
  }
  
  stdout {
    codec => rubydebug
  }
}
```

## Kubernetes Monitoring

### ServiceMonitor for Prometheus Operator

```yaml
# k8s/servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: tsiot-metrics
  namespace: tsiot
  labels:
    app: tsiot
spec:
  selector:
    matchLabels:
      app: tsiot
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
    honorLabels: true
```

### PrometheusRule

```yaml
# k8s/prometheusrule.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: tsiot-rules
  namespace: tsiot
spec:
  groups:
  - name: tsiot.rules
    rules:
    - alert: TSIOTPodCrashLooping
      expr: rate(kube_pod_container_status_restarts_total{namespace="tsiot"}[5m]) > 0
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "TSIOT pod is crash looping"
        description: "Pod {{ $labels.pod }} is crash looping"
        
    - alert: TSIOTPodNotReady
      expr: kube_pod_status_ready{condition="false", namespace="tsiot"} > 0
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "TSIOT pod not ready"
        description: "Pod {{ $labels.pod }} is not ready"
```

## Maintenance and Best Practices

### Backup Monitoring Data

```bash
#!/bin/bash
# backup-monitoring.sh

# Backup Prometheus data
docker run --rm -v prometheus_data:/data -v $(pwd):/backup alpine \
  tar czf /backup/prometheus-backup-$(date +%Y%m%d).tar.gz /data

# Backup Grafana data
docker run --rm -v grafana_data:/data -v $(pwd):/backup alpine \
  tar czf /backup/grafana-backup-$(date +%Y%m%d).tar.gz /data
```

### Health Check Script

```bash
#!/bin/bash
# health-check.sh

echo "Checking monitoring stack health..."

# Check Prometheus
if curl -sf http://localhost:9090/-/healthy > /dev/null; then
  echo " Prometheus is healthy"
else
  echo " Prometheus is unhealthy"
fi

# Check Grafana
if curl -sf http://localhost:3000/api/health > /dev/null; then
  echo " Grafana is healthy"
else
  echo " Grafana is unhealthy"
fi

# Check AlertManager
if curl -sf http://localhost:9093/-/healthy > /dev/null; then
  echo " AlertManager is healthy"
else
  echo " AlertManager is unhealthy"
fi

# Check Jaeger
if curl -sf http://localhost:16686/api/services > /dev/null; then
  echo " Jaeger is healthy"
else
  echo " Jaeger is unhealthy"
fi
```

For additional monitoring configurations and troubleshooting, see:
- [Kubernetes Guide](./kubernetes-guide.md)
- [Docker Guide](./docker-guide.md)
- [Troubleshooting Guide](./troubleshooting.md)