# Monitoring and Alerting Runbook

This runbook covers comprehensive monitoring and alerting procedures for the Inferloop Synthetic Data SDK platform.

## Table of Contents
1. [Monitoring Overview](#monitoring-overview)
2. [Metrics and KPIs](#metrics-and-kpis)
3. [Alert Configuration](#alert-configuration)
4. [Dashboard Management](#dashboard-management)
5. [Log Management](#log-management)
6. [Incident Response](#incident-response)
7. [Performance Monitoring](#performance-monitoring)
8. [Security Monitoring](#security-monitoring)

## Monitoring Overview

### Architecture Components
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **AlertManager**: Alert routing and notification
- **ELK Stack**: Log aggregation and analysis
- **Jaeger**: Distributed tracing
- **PagerDuty**: Incident management

### Monitoring Stack Deployment
```bash
# Deploy monitoring namespace
kubectl apply -f monitoring/namespace.yaml

# Deploy Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --values monitoring/prometheus-values.yaml

# Deploy Grafana
helm install grafana grafana/grafana \
  --namespace monitoring \
  --values monitoring/grafana-values.yaml

# Deploy AlertManager
kubectl apply -f monitoring/alertmanager-config.yaml

# Deploy log collection
helm install fluent-bit fluent/fluent-bit \
  --namespace monitoring \
  --values monitoring/fluent-bit-values.yaml
```

### Monitoring Endpoints
- **Prometheus**: https://prometheus.inferloop.com
- **Grafana**: https://grafana.inferloop.com
- **AlertManager**: https://alertmanager.inferloop.com
- **Kibana**: https://kibana.inferloop.com
- **Jaeger**: https://jaeger.inferloop.com

## Metrics and KPIs

### Application Metrics

#### API Performance Metrics
```yaml
# Request rate
rate(http_requests_total[5m])

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# Response time (95th percentile)
histogram_quantile(0.95, http_request_duration_seconds_bucket)

# Request volume
sum(rate(http_requests_total[5m])) by (method, endpoint)
```

#### Synthetic Data Generation Metrics
```yaml
# Generation success rate
rate(synthdata_generation_success_total[5m]) / rate(synthdata_generation_total[5m])

# Generation duration
histogram_quantile(0.95, synthdata_generation_duration_seconds_bucket)

# Queue length
synthdata_generation_queue_length

# Model training time
histogram_quantile(0.95, synthdata_model_training_duration_seconds_bucket)
```

#### Resource Utilization Metrics
```yaml
# CPU utilization
rate(container_cpu_usage_seconds_total[5m])

# Memory utilization
container_memory_usage_bytes / container_spec_memory_limit_bytes

# Disk utilization
(node_filesystem_size_bytes - node_filesystem_free_bytes) / node_filesystem_size_bytes

# Network utilization
rate(container_network_receive_bytes_total[5m])
```

### Infrastructure Metrics

#### Kubernetes Metrics
```yaml
# Pod restart count
increase(kube_pod_container_status_restarts_total[1h])

# Pod status
kube_pod_status_phase

# Node status
kube_node_status_condition

# Persistent volume usage
kubelet_volume_stats_used_bytes / kubelet_volume_stats_capacity_bytes
```

#### Database Metrics
```yaml
# Connection pool usage
pg_stat_database_numbackends / pg_settings_max_connections

# Query performance
pg_stat_statements_mean_time

# Database size
pg_database_size_bytes

# Transaction rate
rate(pg_stat_database_xact_commit[5m]) + rate(pg_stat_database_xact_rollback[5m])
```

### Business Metrics

#### User Activity Metrics
```yaml
# Active users
synthdata_active_users_total

# Data processed
synthdata_data_processed_bytes_total

# API usage by customer
synthdata_api_requests_total by customer_id

# Revenue metrics
synthdata_revenue_generated_total
```

## Alert Configuration

### Critical Alerts (Immediate Response)

#### Service Availability
```yaml
# API availability < 99%
- alert: APIAvailabilityLow
  expr: rate(http_requests_total{status!~"5.."}[5m]) / rate(http_requests_total[5m]) < 0.99
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "API availability is below 99%"
    description: "API availability is {{ $value | humanizePercentage }} which is below the 99% SLA"
```

#### High Error Rate
```yaml
# Error rate > 1%
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.01
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "High error rate detected"
    description: "Error rate is {{ $value | humanizePercentage }} for the last 5 minutes"
```

#### Database Connectivity
```yaml
# Database connection failures
- alert: DatabaseConnectionFailure
  expr: pg_up == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Database connection failed"
    description: "Unable to connect to PostgreSQL database"
```

### Warning Alerts (Proactive Monitoring)

#### High Response Time
```yaml
# Response time > 1 second
- alert: HighResponseTime
  expr: histogram_quantile(0.95, http_request_duration_seconds_bucket) > 1
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "High response time detected"
    description: "95th percentile response time is {{ $value }}s"
```

#### Resource Utilization
```yaml
# CPU utilization > 80%
- alert: HighCPUUtilization
  expr: rate(container_cpu_usage_seconds_total[5m]) > 0.8
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "High CPU utilization"
    description: "CPU utilization is {{ $value | humanizePercentage }}"

# Memory utilization > 80%
- alert: HighMemoryUtilization
  expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.8
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "High memory utilization"
    description: "Memory utilization is {{ $value | humanizePercentage }}"
```

#### Disk Space
```yaml
# Disk usage > 85%
- alert: HighDiskUsage
  expr: (node_filesystem_size_bytes - node_filesystem_free_bytes) / node_filesystem_size_bytes > 0.85
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High disk usage"
    description: "Disk usage is {{ $value | humanizePercentage }} on {{ $labels.device }}"
```

### Alert Routing Configuration

#### PagerDuty Integration
```yaml
# alertmanager.yml
route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
  - match:
      severity: critical
    receiver: pagerduty-critical
  - match:
      severity: warning
    receiver: slack-warnings

receivers:
- name: 'pagerduty-critical'
  pagerduty_configs:
  - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
    description: '{{ .GroupLabels.alertname }} - {{ .CommonAnnotations.summary }}'
    
- name: 'slack-warnings'
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK_URL'
    channel: '#alerts'
    title: 'Alert: {{ .GroupLabels.alertname }}'
    text: '{{ .CommonAnnotations.description }}'
```

## Dashboard Management

### Production Dashboards

#### System Overview Dashboard
```json
{
  "dashboard": {
    "title": "Inferloop Synthetic Data - System Overview",
    "panels": [
      {
        "title": "API Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m])) by (method)"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, http_request_duration_seconds_bucket)"
          }
        ]
      }
    ]
  }
}
```

#### Infrastructure Dashboard
```bash
# Import dashboard from Grafana marketplace
./scripts/import-dashboard.sh --id 7249 --title "Node Exporter Full"

# Import custom Kubernetes dashboard
./scripts/import-dashboard.sh --file monitoring/dashboards/kubernetes-overview.json

# Import application dashboard
./scripts/import-dashboard.sh --file monitoring/dashboards/synthdata-application.json
```

#### Business Metrics Dashboard
```json
{
  "dashboard": {
    "title": "Business Metrics",
    "panels": [
      {
        "title": "Daily Active Users",
        "type": "stat",
        "targets": [
          {
            "expr": "synthdata_active_users_total"
          }
        ]
      },
      {
        "title": "Data Generated (GB/day)",
        "type": "graph",
        "targets": [
          {
            "expr": "increase(synthdata_data_processed_bytes_total[24h]) / 1024^3"
          }
        ]
      }
    ]
  }
}
```

### Dashboard Automation
```bash
# Auto-generate dashboards from templates
./scripts/generate-dashboards.sh --template monitoring/templates/service-dashboard.json

# Deploy dashboards via Terraform
terraform apply -target=grafana_dashboard.synthdata_overview

# Backup dashboards
./scripts/backup-dashboards.sh --output backup/grafana-dashboards-$(date +%Y%m%d).json
```

## Log Management

### Log Collection Configuration

#### Fluent Bit Configuration
```yaml
# fluent-bit.conf
[SERVICE]
    Flush         1
    Log_Level     info
    Daemon        off
    Parsers_File  parsers.conf

[INPUT]
    Name              tail
    Path              /var/log/containers/*.log
    Parser            docker
    Tag               kube.*
    Refresh_Interval  5
    Mem_Buf_Limit     50MB
    Skip_Long_Lines   On

[FILTER]
    Name                kubernetes
    Match               kube.*
    Kube_URL            https://kubernetes.default.svc:443
    Kube_CA_File        /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    Kube_Token_File     /var/run/secrets/kubernetes.io/serviceaccount/token

[OUTPUT]
    Name  es
    Match kube.*
    Host  elasticsearch.monitoring.svc.cluster.local
    Port  9200
    Index synthdata-logs
```

#### Application Logging
```python
# Configure structured logging
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
```

### Log Analysis Queries

#### Error Analysis
```bash
# Search for errors in the last hour
curl -X GET "kibana.inferloop.com:9200/synthdata-logs/_search" -H 'Content-Type: application/json' -d'
{
  "query": {
    "bool": {
      "must": [
        {
          "range": {
            "@timestamp": {
              "gte": "now-1h"
            }
          }
        },
        {
          "term": {
            "level": "ERROR"
          }
        }
      ]
    }
  }
}
'
```

#### Performance Analysis
```bash
# Find slow requests
curl -X GET "kibana.inferloop.com:9200/synthdata-logs/_search" -H 'Content-Type: application/json' -d'
{
  "query": {
    "bool": {
      "must": [
        {
          "range": {
            "response_time": {
              "gte": 1000
            }
          }
        }
      ]
    }
  },
  "sort": [
    {
      "response_time": {
        "order": "desc"
      }
    }
  ]
}
'
```

## Incident Response

### Incident Classification

#### Severity Levels
- **SEV-1 (Critical)**: Complete service outage
- **SEV-2 (High)**: Major functionality impaired
- **SEV-3 (Medium)**: Minor functionality impaired
- **SEV-4 (Low)**: No user impact

#### Response Times
- **SEV-1**: 15 minutes
- **SEV-2**: 1 hour
- **SEV-3**: 4 hours
- **SEV-4**: Next business day

### Incident Response Procedures

#### SEV-1 Incident Response
```bash
# Immediate response checklist
1. Acknowledge alert within 15 minutes
2. Assess impact and scope
3. Start incident bridge call
4. Notify stakeholders
5. Begin mitigation efforts
6. Update status page
7. Document actions taken
```

#### Alert Investigation
```bash
# Check service health
./scripts/health-check.sh --comprehensive

# Check recent deployments
./scripts/check-recent-deployments.sh --last 4h

# Check resource utilization
./scripts/check-resources.sh --detailed

# Check error logs
./scripts/check-error-logs.sh --last 1h --level ERROR
```

#### Mitigation Strategies
```bash
# Scale up resources
kubectl scale deployment synthdata-api --replicas=10

# Restart services
kubectl rollout restart deployment synthdata-api

# Enable circuit breaker
./scripts/enable-circuit-breaker.sh --service synthdata-api

# Redirect traffic
./scripts/redirect-traffic.sh --from prod --to dr
```

## Performance Monitoring

### Performance Baselines

#### API Performance
- **Response Time**: < 500ms P95
- **Throughput**: > 1000 RPS
- **Error Rate**: < 0.1%
- **Availability**: > 99.9%

#### Generation Performance
- **Small Dataset (< 1MB)**: < 30 seconds
- **Medium Dataset (1-100MB)**: < 5 minutes
- **Large Dataset (> 100MB)**: < 30 minutes

### Performance Monitoring Scripts
```bash
# Run performance benchmarks
./scripts/performance-benchmark.sh --duration 10m --rps 100

# Generate performance report
./scripts/performance-report.sh --period 7d --format pdf

# Compare performance with baseline
./scripts/compare-performance.sh --baseline last-week --current today
```

### Performance Optimization
```bash
# Enable auto-scaling
kubectl apply -f k8s/hpa.yaml

# Optimize database queries
./scripts/optimize-queries.sh --analyze --explain

# Configure caching
./scripts/configure-cache.sh --redis --ttl 3600

# Update resource limits
./scripts/update-resources.sh --cpu 4 --memory 8Gi
```

## Security Monitoring

### Security Metrics
```yaml
# Failed authentication attempts
rate(auth_failures_total[5m]) > 10

# Unusual API access patterns
rate(http_requests_total[5m]) by (source_ip) > 1000

# Privileged operations
increase(privileged_operations_total[5m]) > 5

# Certificate expiration
cert_expiry_days < 30
```

### Security Alerts
```yaml
# Brute force attack detection
- alert: BruteForceAttack
  expr: rate(auth_failures_total[5m]) > 10
  for: 2m
  labels:
    severity: warning
  annotations:
    summary: "Potential brute force attack detected"
    description: "{{ $value }} failed authentication attempts per second"

# DDoS attack detection
- alert: DDoSAttack
  expr: rate(http_requests_total[1m]) by (source_ip) > 1000
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Potential DDoS attack detected"
    description: "High request rate from IP {{ $labels.source_ip }}"
```

### Security Monitoring Scripts
```bash
# Check for suspicious activity
./scripts/security-check.sh --suspicious-ips --failed-logins

# Analyze access patterns
./scripts/analyze-access-patterns.sh --period 24h --anomalies

# Generate security report
./scripts/security-report.sh --period 7d --threats --vulnerabilities
```

## Monitoring Maintenance

### Regular Maintenance Tasks
```bash
# Weekly monitoring maintenance
./scripts/monitoring-maintenance.sh --weekly

# Update monitoring components
helm upgrade prometheus prometheus-community/kube-prometheus-stack

# Clean up old metrics
./scripts/cleanup-metrics.sh --older-than 90d

# Backup monitoring configuration
./scripts/backup-monitoring-config.sh
```

### Monitoring Health Checks
```bash
# Check Prometheus health
curl -f http://prometheus.inferloop.com:9090/-/healthy

# Check Grafana health
curl -f http://grafana.inferloop.com:3000/api/health

# Check AlertManager health
curl -f http://alertmanager.inferloop.com:9093/-/healthy

# Check log pipeline health
./scripts/check-log-pipeline.sh
```

## Contact Information

### Monitoring Team
- **Monitoring Lead**: monitoring-lead@inferloop.com
- **SRE Team**: sre@inferloop.com
- **DevOps Team**: devops@inferloop.com

### Emergency Contacts
- **On-Call SRE**: +1-XXX-XXX-XXXX
- **Monitoring Manager**: +1-XXX-XXX-XXXX
- **Platform Lead**: +1-XXX-XXX-XXXX