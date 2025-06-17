# TSIOT Troubleshooting Guide

## Overview

This guide provides comprehensive troubleshooting steps for common issues encountered when deploying and running TSIOT. It covers Docker, Kubernetes, and application-specific problems.

## Quick Diagnosis

### Health Check Commands
```bash
# Check service health
curl http://localhost:8080/health

# Check readiness
curl http://localhost:8080/ready

# Check service info
curl http://localhost:8080/api/v1/info

# Check metrics
curl http://localhost:8080/metrics
```

### Log Analysis
```bash
# Docker logs
docker logs tsiot-api --tail=100 -f

# Kubernetes logs
kubectl logs deployment/tsiot -n tsiot --tail=100 -f

# Check for errors
kubectl logs deployment/tsiot -n tsiot | grep -i error
```

## Common Issues

### 1. Service Won't Start

#### Symptoms
- Container/pod keeps restarting
- Health checks failing
- Service unreachable

#### Diagnosis Steps
```bash
# Check container status
docker ps -a
kubectl get pods -n tsiot

# Check events
kubectl describe pod <pod-name> -n tsiot

# Check resource usage
kubectl top pods -n tsiot
docker stats
```

#### Common Causes & Solutions

**Port Already in Use**
```bash
# Find process using port
sudo lsof -i :8080
sudo netstat -tulpn | grep :8080

# Kill process or change port
kill -9 <PID>
# OR
export TSIOT_PORT=8081
```

**Insufficient Resources**
```bash
# Check available resources
kubectl describe nodes
docker system df

# Increase resource limits
kubectl patch deployment tsiot -n tsiot -p '{"spec":{"template":{"spec":{"containers":[{"name":"tsiot","resources":{"limits":{"memory":"2Gi","cpu":"1000m"}}}]}}}}'
```

**Configuration Issues**
```bash
# Validate configuration
kubectl get configmap tsiot-config -n tsiot -o yaml

# Check environment variables
kubectl exec deployment/tsiot -n tsiot -- env | grep TSIOT
```

### 2. Database Connection Issues

#### Symptoms
- "connection refused" errors
- Database timeout errors
- Authentication failures

#### Diagnosis Steps
```bash
# Test database connectivity
kubectl exec deployment/tsiot -n tsiot -- nc -zv postgres 5432

# Check database status
kubectl get pods -l app=postgres -n tsiot
kubectl logs deployment/postgres -n tsiot

# Test authentication
kubectl exec deployment/tsiot -n tsiot -- psql -h postgres -U tsiot -d tsiot -c "SELECT 1;"
```

#### Solutions

**Connection Refused**
```bash
# Check service endpoints
kubectl get endpoints postgres -n tsiot

# Verify network policies
kubectl get networkpolicy -n tsiot

# Test from debug pod
kubectl run debug --image=busybox -it --rm -- nc -zv postgres.tsiot.svc.cluster.local 5432
```

**Authentication Failures**
```bash
# Check credentials in secret
kubectl get secret tsiot-postgres -n tsiot -o yaml

# Verify password
kubectl get secret tsiot-postgres -n tsiot -o jsonpath="{.data.password}" | base64 -d

# Reset password if needed
kubectl patch secret tsiot-postgres -n tsiot -p '{"data":{"password":"<new-base64-password>"}}'
```

**Connection Pool Exhaustion**
```yaml
# Increase connection pool size
database:
  maxConnections: 50
  maxIdleConnections: 10
  connectionTimeout: "30s"
```

### 3. Performance Issues

#### Symptoms
- High response times
- High CPU/memory usage
- Request timeouts

#### Diagnosis Steps
```bash
# Check resource usage
kubectl top pods -n tsiot
kubectl top nodes

# Check metrics
curl http://localhost:8080/metrics | grep -E "(cpu|memory|requests)"

# Profile application
kubectl exec deployment/tsiot -n tsiot -- wget -qO- localhost:6060/debug/pprof/goroutine?debug=1
```

#### Solutions

**High CPU Usage**
```bash
# Check for CPU-intensive operations
kubectl exec deployment/tsiot -n tsiot -- top

# Enable CPU profiling
curl http://localhost:8080/debug/pprof/profile?seconds=30 > cpu.prof

# Increase CPU limits
kubectl patch deployment tsiot -n tsiot -p '{"spec":{"template":{"spec":{"containers":[{"name":"tsiot","resources":{"limits":{"cpu":"2000m"}}}]}}}}'
```

**Memory Issues**
```bash
# Check memory usage
kubectl exec deployment/tsiot -n tsiot -- free -h

# Get memory profile
curl http://localhost:8080/debug/pprof/heap > heap.prof

# Increase memory limits
kubectl patch deployment tsiot -n tsiot -p '{"spec":{"template":{"spec":{"containers":[{"name":"tsiot","resources":{"limits":{"memory":"4Gi"}}}]}}}}'
```

**Slow Database Queries**
```sql
-- Enable query logging in PostgreSQL
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_min_duration_statement = 1000;
SELECT pg_reload_conf();

-- Check slow queries
SELECT query, mean_exec_time, calls 
FROM pg_stat_statements 
ORDER BY mean_exec_time DESC 
LIMIT 10;
```

### 4. Storage Issues

#### Symptoms
- Disk space errors
- PVC mount failures
- Data corruption

#### Diagnosis Steps
```bash
# Check PVC status
kubectl get pvc -n tsiot

# Check storage usage
kubectl exec deployment/tsiot -n tsiot -- df -h

# Check storage class
kubectl get storageclass
```

#### Solutions

**Disk Space Full**
```bash
# Clean up old logs
kubectl exec deployment/tsiot -n tsiot -- find /var/log -name "*.log" -mtime +7 -delete

# Resize PVC (if supported)
kubectl patch pvc data-pvc -n tsiot -p '{"spec":{"resources":{"requests":{"storage":"100Gi"}}}}'

# Clean up old data
kubectl exec deployment/postgres -n tsiot -- psql -U postgres -c "DELETE FROM timeseries WHERE created_at < NOW() - INTERVAL '30 days';"
```

**PVC Mount Issues**
```bash
# Check node storage
kubectl describe node <node-name>

# Check PV status
kubectl get pv

# Delete and recreate PVC
kubectl delete pvc data-pvc -n tsiot
kubectl apply -f pvc.yaml
```

### 5. Network Issues

#### Symptoms
- Service discovery failures
- Ingress not working
- Intermittent connectivity

#### Diagnosis Steps
```bash
# Test service resolution
kubectl exec deployment/tsiot -n tsiot -- nslookup tsiot.tsiot.svc.cluster.local

# Check ingress
kubectl get ingress -n tsiot
kubectl describe ingress tsiot-ingress -n tsiot

# Check network policies
kubectl get networkpolicy -n tsiot
```

#### Solutions

**Service Discovery Issues**
```bash
# Check CoreDNS
kubectl get pods -n kube-system -l k8s-app=kube-dns

# Test DNS resolution
kubectl run debug --image=busybox -it --rm -- nslookup kubernetes.default.svc.cluster.local

# Restart CoreDNS if needed
kubectl rollout restart deployment/coredns -n kube-system
```

**Ingress Issues**
```bash
# Check ingress controller
kubectl get pods -n ingress-nginx

# Check ingress logs
kubectl logs deployment/ingress-nginx-controller -n ingress-nginx

# Verify TLS certificates
kubectl get certificate -n tsiot
kubectl describe certificate tsiot-tls -n tsiot
```

### 6. Security Issues

#### Symptoms
- Authentication failures
- Permission denied errors
- TLS certificate issues

#### Diagnosis Steps
```bash
# Check RBAC
kubectl auth can-i --list --as=system:serviceaccount:tsiot:tsiot

# Check certificates
openssl x509 -in /etc/certs/tls.crt -text -noout

# Check security contexts
kubectl get pod <pod-name> -n tsiot -o yaml | grep -A 10 securityContext
```

#### Solutions

**RBAC Issues**
```bash
# Check service account
kubectl get serviceaccount tsiot -n tsiot

# Verify role bindings
kubectl get rolebinding -n tsiot
kubectl describe rolebinding tsiot -n tsiot

# Fix permissions
kubectl apply -f rbac.yaml
```

**Certificate Issues**
```bash
# Renew certificates
kubectl delete certificate tsiot-tls -n tsiot
kubectl apply -f certificate.yaml

# Check cert-manager
kubectl get pods -n cert-manager
kubectl logs deployment/cert-manager -n cert-manager
```

## Application-Specific Issues

### 7. Generation Failures

#### Symptoms
- Generator errors
- Invalid parameters
- Out of memory during generation

#### Diagnosis
```bash
# Check generator logs
kubectl logs deployment/tsiot -n tsiot | grep -i generator

# Test generator directly
curl -X POST http://localhost:8080/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"type":"arima","length":100,"parameters":{"ar_params":[0.5]}}'
```

#### Solutions

**Parameter Validation**
```bash
# Check available generators
curl http://localhost:8080/api/v1/generators

# Validate parameters
curl -X POST http://localhost:8080/api/v1/validate \
  -H "Content-Type: application/json" \
  -d '{"generator":"arima","parameters":{"ar_params":[0.5]}}'
```

**Memory Issues**
```yaml
# Increase generator limits
generators:
  maxLength: 1000000
  timeout: "10m"
  memoryLimit: "2Gi"
```

### 8. Validation Failures

#### Symptoms
- Validation timeouts
- False positives/negatives
- Validator crashes

#### Diagnosis
```bash
# Check validation engines
curl http://localhost:8080/api/v1/validators

# Test validation
curl -X POST http://localhost:8080/api/v1/validate \
  -H "Content-Type: application/json" \
  -d '{"time_series":{"values":[1,2,3],"timestamps":["2023-01-01T00:00:00Z"]}}'
```

#### Solutions

**Timeout Issues**
```yaml
# Increase validation timeout
validation:
  timeout: "5m"
  maxConcurrentValidations: 10
```

**Performance Optimization**
```yaml
# Enable validation caching
validation:
  cacheEnabled: true
  cacheTTL: "1h"
  maxCacheSize: 1000
```

## Monitoring and Alerting

### Setting Up Alerts

**High Error Rate Alert**
```yaml
- alert: TSIOTHighErrorRate
  expr: rate(tsiot_http_requests_total{status=~"5.."}[5m]) > 0.1
  for: 2m
  labels:
    severity: warning
  annotations:
    summary: "TSIOT high error rate detected"
    description: "Error rate is {{ $value }} errors per second"
```

**Resource Usage Alert**
```yaml
- alert: TSIOTHighMemoryUsage
  expr: container_memory_usage_bytes{pod=~"tsiot.*"} / container_spec_memory_limit_bytes > 0.9
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "TSIOT high memory usage"
    description: "Memory usage is {{ $value | humanizePercentage }}"
```

### Debug Endpoints

**Health Debugging**
```bash
# Detailed health check
curl http://localhost:8080/health?verbose=true

# Component status
curl http://localhost:8080/debug/status

# Configuration dump
curl http://localhost:8080/debug/config
```

**Performance Profiling**
```bash
# CPU profile
curl http://localhost:8080/debug/pprof/profile?seconds=30 > cpu.prof

# Memory profile
curl http://localhost:8080/debug/pprof/heap > heap.prof

# Goroutine dump
curl http://localhost:8080/debug/pprof/goroutine?debug=1
```

## Emergency Procedures

### Service Recovery

**Quick Restart**
```bash
# Docker
docker restart tsiot-api

# Kubernetes
kubectl rollout restart deployment/tsiot -n tsiot
```

**Rollback Deployment**
```bash
# Check rollout history
kubectl rollout history deployment/tsiot -n tsiot

# Rollback to previous version
kubectl rollout undo deployment/tsiot -n tsiot

# Rollback to specific revision
kubectl rollout undo deployment/tsiot --to-revision=2 -n tsiot
```

**Scale Down/Up**
```bash
# Scale down to zero
kubectl scale deployment tsiot --replicas=0 -n tsiot

# Scale back up
kubectl scale deployment tsiot --replicas=3 -n tsiot
```

### Data Recovery

**Database Recovery**
```bash
# Restore from backup
kubectl exec deployment/postgres -n tsiot -- pg_restore -U postgres -d tsiot /backups/latest.sql

# Point-in-time recovery
kubectl exec deployment/postgres -n tsiot -- pg_basebackup -U postgres -D /var/lib/postgresql/recovery
```

**Configuration Recovery**
```bash
# Restore from backup
kubectl apply -f config-backup.yaml

# Reset to defaults
helm upgrade tsiot tsiot/tsiot --reset-values -n tsiot
```

## Performance Optimization

### Database Optimization

**Index Creation**
```sql
-- Add indexes for common queries
CREATE INDEX idx_timeseries_created_at ON timeseries(created_at);
CREATE INDEX idx_timeseries_generator ON timeseries(generator);
CREATE INDEX idx_timeseries_metadata ON timeseries USING gin(metadata);
```

**Query Optimization**
```sql
-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM timeseries WHERE created_at > NOW() - INTERVAL '1 day';

-- Update statistics
ANALYZE timeseries;
```

### Application Optimization

**Connection Pooling**
```yaml
database:
  maxOpenConnections: 25
  maxIdleConnections: 5
  connMaxLifetime: "1h"
```

**Caching Configuration**
```yaml
cache:
  enabled: true
  ttl: "1h"
  maxSize: 1000
  evictionPolicy: "lru"
```

## Tools and Utilities

### Log Analysis Tools
```bash
# Structured log parsing
kubectl logs deployment/tsiot -n tsiot | jq '.level="error"'

# Log aggregation with stern
stern tsiot -n tsiot --since 1h

# Error pattern analysis
kubectl logs deployment/tsiot -n tsiot | grep -E "(ERROR|FATAL)" | sort | uniq -c
```

### Network Debugging Tools
```bash
# Network connectivity test
kubectl run netshoot --image=nicolaka/netshoot -it --rm

# DNS debugging
kubectl run dnsutils --image=tutum/dnsutils -it --rm -- nslookup tsiot.tsiot.svc.cluster.local

# Traffic analysis
kubectl run tcpdump --image=corfr/tcpdump -it --rm -- tcpdump -i any -n port 8080
```

### Resource Monitoring
```bash
# Resource usage over time
kubectl top pods -n tsiot --containers --use-protocol-buffers

# Node resource availability
kubectl describe nodes | grep -A 5 "Allocated resources"

# Storage usage
kubectl exec deployment/tsiot -n tsiot -- du -sh /data/*
```

## Contact and Escalation

### Log Collection for Support
```bash
#!/bin/bash
# collect-logs.sh
NAMESPACE="tsiot"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_DIR="tsiot-logs-${TIMESTAMP}"

mkdir -p "${OUTPUT_DIR}"

# Collect pod logs
kubectl logs deployment/tsiot -n ${NAMESPACE} > "${OUTPUT_DIR}/tsiot.log"

# Collect events
kubectl get events -n ${NAMESPACE} > "${OUTPUT_DIR}/events.log"

# Collect resource descriptions
kubectl describe deployment tsiot -n ${NAMESPACE} > "${OUTPUT_DIR}/deployment.yaml"
kubectl describe pods -l app=tsiot -n ${NAMESPACE} > "${OUTPUT_DIR}/pods.yaml"

# Collect configuration
kubectl get configmap tsiot-config -n ${NAMESPACE} -o yaml > "${OUTPUT_DIR}/config.yaml"

# Create archive
tar -czf "tsiot-diagnostics-${TIMESTAMP}.tar.gz" "${OUTPUT_DIR}"
echo "Diagnostic bundle created: tsiot-diagnostics-${TIMESTAMP}.tar.gz"
```

### Support Channels
- **GitHub Issues**: [Report bugs](https://github.com/your-org/tsiot/issues)
- **Documentation**: [Full docs](https://docs.tsiot.io)
- **Community**: [Discord support](https://discord.gg/tsiot)
- **Enterprise**: support@tsiot.io

### Escalation Criteria
- **P1 (Critical)**: Service completely unavailable
- **P2 (High)**: Significant functionality impaired
- **P3 (Medium)**: Minor functionality issues
- **P4 (Low)**: Enhancement requests or questions