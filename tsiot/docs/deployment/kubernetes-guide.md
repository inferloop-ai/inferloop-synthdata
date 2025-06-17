# TSIOT Kubernetes Deployment Guide

## Overview

This guide covers deploying TSIOT on Kubernetes clusters, from local development to production environments. It includes configuration options, scaling strategies, and best practices.

## Prerequisites

### Required Tools
- **Kubernetes cluster** (1.24+)
- **kubectl** (compatible with cluster version)
- **Helm** (3.12+)
- **Docker** (for building custom images)

### Cluster Requirements
- **Minimum**: 4 CPU cores, 8GB RAM, 50GB storage
- **Recommended**: 8+ CPU cores, 16GB+ RAM, 100GB+ storage
- **Storage classes**: For persistent volumes
- **Load balancer**: For external access

## Quick Start

### 1. Add TSIOT Helm Repository
```bash
# Add the repository
helm repo add tsiot https://charts.tsiot.io
helm repo update

# Verify repository
helm search repo tsiot
```

### 2. Install with Default Configuration
```bash
# Create namespace
kubectl create namespace tsiot

# Install TSIOT
helm install tsiot tsiot/tsiot \
  --namespace tsiot \
  --wait
```

### 3. Verify Installation
```bash
# Check deployment status
kubectl get pods -n tsiot

# Check services
kubectl get svc -n tsiot

# View logs
kubectl logs -f deployment/tsiot -n tsiot
```

## Configuration Options

### Values File Structure
```yaml
# values.yaml
global:
  imageRegistry: "docker.io"
  imagePullSecrets: []
  storageClass: "default"

image:
  repository: "tsiot/tsiot"
  tag: "1.0.0"
  pullPolicy: "IfNotPresent"

replicaCount: 3

service:
  type: "ClusterIP"
  port: 80
  targetPort: 8080

ingress:
  enabled: true
  className: "nginx"
  hosts:
    - host: "tsiot.example.com"
      paths:
        - path: "/"
          pathType: "Prefix"
  tls:
    - secretName: "tsiot-tls"
      hosts:
        - "tsiot.example.com"

database:
  type: "postgres"
  host: "postgres.tsiot.svc.cluster.local"
  port: 5432
  name: "tsiot"
  user: "tsiot"
  # Password from secret

redis:
  enabled: true
  host: "redis.tsiot.svc.cluster.local"
  port: 6379

kafka:
  enabled: true
  brokers: "kafka.tsiot.svc.cluster.local:9092"

persistence:
  enabled: true
  storageClass: "fast-ssd"
  size: "50Gi"
  accessMode: "ReadWriteOnce"

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

resources:
  limits:
    cpu: "2000m"
    memory: "4Gi"
  requests:
    cpu: "500m"
    memory: "1Gi"

nodeSelector: {}
tolerations: []
affinity: {}

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
  prometheusRule:
    enabled: true

security:
  podSecurityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000
  securityContext:
    allowPrivilegeEscalation: false
    capabilities:
      drop:
        - ALL
    readOnlyRootFilesystem: true
```

## Deployment Scenarios

### Development Environment
```bash
# Minimal development setup
helm install tsiot-dev tsiot/tsiot \
  --namespace tsiot-dev \
  --create-namespace \
  --set replicaCount=1 \
  --set resources.requests.cpu=100m \
  --set resources.requests.memory=256Mi \
  --set database.type=sqlite \
  --set redis.enabled=false \
  --set kafka.enabled=false
```

### Staging Environment
```bash
# Staging with external dependencies
helm install tsiot-staging tsiot/tsiot \
  --namespace tsiot-staging \
  --create-namespace \
  --values values-staging.yaml \
  --set image.tag=v1.0.0-rc1
```

### Production Environment
```bash
# Production deployment
helm install tsiot-prod tsiot/tsiot \
  --namespace tsiot-prod \
  --create-namespace \
  --values values-production.yaml \
  --set image.tag=v1.0.0
```

## Multi-Environment Setup

### Production Values (values-production.yaml)
```yaml
replicaCount: 5

image:
  tag: "1.0.0"

service:
  type: "LoadBalancer"
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
  hosts:
    - host: "api.tsiot.com"
      paths:
        - path: "/"
          pathType: "Prefix"
  tls:
    - secretName: "tsiot-prod-tls"
      hosts:
        - "api.tsiot.com"

database:
  type: "postgres"
  host: "prod-postgres.cluster.local"
  ssl: true
  connectionPool:
    maxConnections: 25

redis:
  enabled: true
  cluster: true
  nodes: 3

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20

resources:
  limits:
    cpu: "4000m"
    memory: "8Gi"
  requests:
    cpu: "1000m"
    memory: "2Gi"

persistence:
  enabled: true
  storageClass: "fast-ssd"
  size: "200Gi"

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
  prometheusRule:
    enabled: true

security:
  networkPolicy:
    enabled: true
  podDisruptionBudget:
    enabled: true
    minAvailable: 2
```

## Scaling and Performance

### Horizontal Pod Autoscaler
```yaml
# HPA configuration
autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 15
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
    - type: External
      external:
        metric:
          name: kafka_consumer_lag
        target:
          type: AverageValue
          averageValue: "100"
```

### Vertical Pod Autoscaler
```yaml
# VPA configuration
vpa:
  enabled: true
  updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: tsiot
      maxAllowed:
        cpu: "8"
        memory: "16Gi"
      minAllowed:
        cpu: "100m"
        memory: "256Mi"
```

### Node Affinity and Anti-Affinity
```yaml
affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
          - key: app.kubernetes.io/name
            operator: In
            values:
            - tsiot
        topologyKey: kubernetes.io/hostname
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: node-type
          operator: In
          values:
          - compute-optimized
```

## Monitoring and Observability

### Prometheus Integration
```yaml
monitoring:
  serviceMonitor:
    enabled: true
    namespace: monitoring
    labels:
      app: tsiot
    scrapeInterval: 30s
    scrapeTimeout: 10s
    path: /metrics
    
  prometheusRule:
    enabled: true
    namespace: monitoring
    groups:
    - name: tsiot.rules
      rules:
      - alert: TSIOTHighErrorRate
        expr: rate(tsiot_http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
      - alert: TSIOTHighLatency
        expr: histogram_quantile(0.95, rate(tsiot_http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
```

### Grafana Dashboard
```bash
# Apply Grafana dashboard ConfigMap
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: tsiot-dashboard
  namespace: monitoring
  labels:
    grafana_dashboard: "1"
data:
  tsiot.json: |
    {
      "dashboard": {
        "title": "TSIOT Dashboard",
        "panels": [
          {
            "title": "Request Rate",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(tsiot_http_requests_total[5m])"
              }
            ]
          }
        ]
      }
    }
EOF
```

## Security

### Network Policies
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: tsiot-network-policy
  namespace: tsiot
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: tsiot
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: database
    ports:
    - protocol: TCP
      port: 5432
```

### Pod Security Standards
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: tsiot
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

## Storage Configuration

### PostgreSQL with High Availability
```yaml
postgresql:
  enabled: true
  architecture: replication
  auth:
    postgresPassword: "strong-password"
    username: tsiot
    password: "tsiot-password"
    database: tsiot
  primary:
    persistence:
      enabled: true
      storageClass: "fast-ssd"
      size: 100Gi
    resources:
      limits:
        cpu: 2000m
        memory: 4Gi
      requests:
        cpu: 500m
        memory: 1Gi
  readReplicas:
    replicaCount: 2
    persistence:
      enabled: true
      storageClass: "fast-ssd"
      size: 100Gi
```

### Redis Cluster
```yaml
redis:
  enabled: true
  architecture: replication
  auth:
    enabled: true
    password: "redis-password"
  master:
    persistence:
      enabled: true
      storageClass: "fast-ssd"
      size: 50Gi
    resources:
      limits:
        cpu: 1000m
        memory: 2Gi
      requests:
        cpu: 250m
        memory: 512Mi
  replica:
    replicaCount: 2
    persistence:
      enabled: true
      storageClass: "fast-ssd"
      size: 50Gi
```

## Backup and Recovery

### Database Backup CronJob
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: tsiot-db-backup
  namespace: tsiot
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:15
            env:
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: tsiot-postgres
                  key: password
            command:
            - /bin/bash
            - -c
            - |
              pg_dump -h postgres -U tsiot tsiot | \
              gzip > /backup/tsiot-$(date +%Y%m%d-%H%M%S).sql.gz
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
```

## Troubleshooting

### Common Issues

#### Pod Startup Issues
```bash
# Check pod status
kubectl get pods -n tsiot

# Check pod events
kubectl describe pod <pod-name> -n tsiot

# Check logs
kubectl logs <pod-name> -n tsiot --previous
```

#### Service Discovery Issues
```bash
# Check service endpoints
kubectl get endpoints -n tsiot

# Test service connectivity
kubectl run debug --image=busybox -it --rm -- nslookup tsiot.tsiot.svc.cluster.local
```

#### Storage Issues
```bash
# Check PVCs
kubectl get pvc -n tsiot

# Check storage class
kubectl get storageclass

# Check volume mounts
kubectl describe pod <pod-name> -n tsiot
```

### Debug Commands
```bash
# Get all resources
kubectl get all -n tsiot

# Check resource usage
kubectl top pods -n tsiot
kubectl top nodes

# Port forward for debugging
kubectl port-forward svc/tsiot 8080:80 -n tsiot

# Execute commands in pod
kubectl exec -it <pod-name> -n tsiot -- /bin/bash
```

## Performance Tuning

### JVM Tuning (for Java components)
```yaml
env:
- name: JAVA_OPTS
  value: "-Xmx2g -Xms1g -XX:+UseG1GC -XX:MaxGCPauseMillis=200"
```

### Resource Optimization
```yaml
resources:
  limits:
    cpu: "2000m"
    memory: "4Gi"
    ephemeral-storage: "10Gi"
  requests:
    cpu: "500m"
    memory: "1Gi"
    ephemeral-storage: "5Gi"
```

## Migration and Updates

### Rolling Updates
```bash
# Update image
helm upgrade tsiot tsiot/tsiot \
  --namespace tsiot \
  --set image.tag=v1.1.0 \
  --wait

# Check rollout status
kubectl rollout status deployment/tsiot -n tsiot

# Rollback if needed
helm rollback tsiot -n tsiot
```

### Blue-Green Deployment
```bash
# Deploy new version to separate namespace
kubectl create namespace tsiot-green
helm install tsiot-green tsiot/tsiot \
  --namespace tsiot-green \
  --set image.tag=v1.1.0

# Switch traffic
kubectl patch service tsiot -n tsiot \
  -p '{"spec":{"selector":{"version":"green"}}}'
```

## Best Practices

### 1. Resource Management
- Set appropriate resource requests and limits
- Use QoS classes effectively
- Monitor resource usage

### 2. High Availability
- Use multiple replicas
- Configure anti-affinity rules
- Implement health checks

### 3. Security
- Use least privilege principle
- Enable network policies
- Scan images for vulnerabilities

### 4. Monitoring
- Set up comprehensive monitoring
- Configure alerting rules
- Use distributed tracing

### 5. Backup and Recovery
- Regular database backups
- Test recovery procedures
- Document runbooks

## Cost Optimization

### Right-sizing Resources
```bash
# Use VPA recommendations
kubectl describe vpa tsiot-vpa -n tsiot

# Monitor actual usage
kubectl top pods -n tsiot --containers
```

### Spot Instances
```yaml
nodeSelector:
  kubernetes.io/arch: amd64
  node.kubernetes.io/instance-type: "spot"

tolerations:
- key: "spot"
  operator: "Equal"
  value: "true"
  effect: "NoSchedule"
```

### Resource Quotas
```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: tsiot-quota
  namespace: tsiot
spec:
  hard:
    requests.cpu: "10"
    requests.memory: 20Gi
    limits.cpu: "20"
    limits.memory: 40Gi
    persistentvolumeclaims: "5"
```

For additional help, refer to:
- [Troubleshooting Guide](./troubleshooting.md)
- [Monitoring Setup](./monitoring-setup.md)
- [Docker Guide](./docker-guide.md)