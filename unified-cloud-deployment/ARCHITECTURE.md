# Unified Cloud Deployment Architecture

## Overview

This document provides the detailed technical architecture for the unified cloud deployment platform that serves all Inferloop synthetic data services (tabular, textnlp, syndoc, etc.). The architecture is designed to be cloud-agnostic, highly scalable, and cost-effective while maintaining service independence and flexibility.

## Architecture Principles

### 1. Cloud-Native Design
- Container-first approach using Kubernetes
- Microservices architecture with clear boundaries
- Stateless services with external state management
- Event-driven communication where appropriate

### 2. Infrastructure as Code
- Everything defined in code (Terraform, Kubernetes manifests)
- GitOps for configuration management
- Immutable infrastructure patterns
- Automated provisioning and scaling

### 3. Service Mesh Architecture
- Zero-trust networking between services
- Automatic service discovery and load balancing
- Built-in resilience patterns (circuit breakers, retries)
- End-to-end encryption for service communication

### 4. Observability First
- Distributed tracing for all requests
- Centralized logging with structured logs
- Comprehensive metrics collection
- Real-time alerting and anomaly detection

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              External Users                              │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────────────────────┐
│                          Global Load Balancer                            │
│                      (CloudFront/Front Door/Cloud CDN)                   │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────────────────────┐
│                            WAF & DDoS Protection                         │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────────────────────┐
│                              API Gateway                                 │
│                    (Kong/Apigee/AWS API Gateway)                        │
│  ┌─────────────┬──────────────┬──────────────┬─────────────────────┐  │
│  │   Auth      │ Rate Limiting│   Caching    │   Transformation    │  │
│  └─────────────┴──────────────┴──────────────┴─────────────────────┘  │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────────────────────┐
│                           Kubernetes Cluster                             │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                        Istio Service Mesh                        │  │
│  ├─────────────────────────────────────────────────────────────────┤  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │  │
│  │  │ Tabular  │ │ TextNLP  │ │  SynDoc  │ │ Shared Services │  │  │
│  │  │ Service  │ │ Service  │ │ Service  │ │   - Auth        │  │  │
│  │  │          │ │          │ │          │ │   - Billing     │  │  │
│  │  │ ┌──────┐ │ │ ┌──────┐ │ │ ┌──────┐ │ │   - Notification│  │  │
│  │  │ │ API  │ │ │ │ API  │ │ │ │ API  │ │ │   - Analytics   │  │  │
│  │  │ └──────┘ │ │ └──────┘ │ │ └──────┘ │ └──────────────────┘  │  │
│  │  │ ┌──────┐ │ │ ┌──────┐ │ │ ┌──────┐ │                       │  │
│  │  │ │Worker│ │ │ │Worker│ │ │ │Worker│ │                       │  │
│  │  │ └──────┘ │ │ └──────┘ │ │ └──────┘ │                       │  │
│  │  └──────────┘ └──────────┘ └──────────┘                       │  │
│  └─────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
┌───────▼────────┐ ┌─────────────────▼──────────────┐ ┌───────────▼────────┐
│   Data Layer   │ │    Message Queue Layer         │ │  External Services │
├────────────────┤ ├────────────────────────────────┤ ├────────────────────┤
│ • PostgreSQL   │ │ • Kafka/Pub-Sub/Service Bus   │ │ • AI/ML APIs       │
│ • Redis        │ │ • Event Streaming              │ │ • Payment Gateway  │
│ • Object Store │ │ • Task Queues                  │ │ • Email Service    │
└────────────────┘ └────────────────────────────────┘ └────────────────────┘
```

## Component Architecture

### 1. API Gateway Layer

#### Purpose
- Single entry point for all external requests
- Cross-cutting concerns (auth, rate limiting, caching)
- Request routing and load balancing
- API versioning and transformation

#### Implementation
```yaml
# Kong API Gateway Configuration
apiVersion: configuration.konghq.com/v1
kind: KongIngress
metadata:
  name: inferloop-ingress
route:
  methods:
    - GET
    - POST
    - PUT
    - DELETE
  strip_path: false
upstream:
  load_balancing: consistent_hashing
  hash_on: header
  hash_on_header: X-User-ID
  healthchecks:
    active:
      healthy:
        interval: 10
        successes: 3
      unhealthy:
        interval: 5
        http_failures: 3
```

### 2. Service Mesh Layer

#### Istio Configuration
```yaml
# Service mesh configuration for inter-service communication
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: inferloop-services
spec:
  host: "*.inferloop.svc.cluster.local"
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        h2UpgradePolicy: UPGRADE
        useClientProtocol: true
    loadBalancer:
      simple: LEAST_REQUEST
    outlierDetection:
      consecutiveErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
```

### 3. Service Architecture

#### Base Service Template
```yaml
# Base configuration for all services
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .ServiceName }}
  namespace: inferloop
  labels:
    app: {{ .ServiceName }}
    version: {{ .Version }}
spec:
  replicas: {{ .Replicas }}
  selector:
    matchLabels:
      app: {{ .ServiceName }}
  template:
    metadata:
      labels:
        app: {{ .ServiceName }}
        version: {{ .Version }}
      annotations:
        sidecar.istio.io/inject: "true"
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
    spec:
      serviceAccountName: {{ .ServiceName }}-sa
      containers:
      - name: {{ .ServiceName }}
        image: {{ .Image }}
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: SERVICE_NAME
          value: {{ .ServiceName }}
        - name: ENVIRONMENT
          value: {{ .Environment }}
        resources:
          requests:
            memory: {{ .Resources.Memory.Request }}
            cpu: {{ .Resources.CPU.Request }}
          limits:
            memory: {{ .Resources.Memory.Limit }}
            cpu: {{ .Resources.CPU.Limit }}
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 4. Data Layer Architecture

#### Multi-Tenant Database Design
```sql
-- Shared database with service-level isolation
CREATE SCHEMA IF NOT EXISTS tabular;
CREATE SCHEMA IF NOT EXISTS textnlp;
CREATE SCHEMA IF NOT EXISTS syndoc;
CREATE SCHEMA IF NOT EXISTS shared;

-- Service-specific tables
CREATE TABLE tabular.synthetic_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    dataset_name VARCHAR(255) NOT NULL,
    configuration JSONB,
    generated_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE textnlp.generations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    prompt TEXT NOT NULL,
    model VARCHAR(100),
    generated_text TEXT,
    tokens_used INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Shared tables
CREATE TABLE shared.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    organization_id UUID,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE shared.api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES shared.users(id),
    key_hash VARCHAR(255) NOT NULL,
    permissions JSONB,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Caching Strategy
```yaml
# Redis cluster configuration
apiVersion: redis.redis.opstreelabs.in/v1beta1
kind: RedisCluster
metadata:
  name: inferloop-cache
spec:
  clusterSize: 6
  kubernetesConfig:
    image: redis:7-alpine
    resources:
      requests:
        cpu: 100m
        memory: 256Mi
      limits:
        cpu: 200m
        memory: 512Mi
  storage:
    volumeClaimTemplate:
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 10Gi
  # Cache partitioning by service
  redisConfig:
    save: "900 1 300 10"
    maxmemory: "2gb"
    maxmemory-policy: "allkeys-lru"
```

### 5. Message Queue Architecture

#### Event-Driven Communication
```yaml
# Kafka configuration for event streaming
apiVersion: kafka.strimzi.io/v1beta2
kind: Kafka
metadata:
  name: inferloop-events
spec:
  kafka:
    version: 3.5.0
    replicas: 3
    listeners:
      - name: plain
        port: 9092
        type: internal
        tls: false
      - name: tls
        port: 9093
        type: internal
        tls: true
    config:
      offsets.topic.replication.factor: 3
      transaction.state.log.replication.factor: 3
      transaction.state.log.min.isr: 2
      log.retention.hours: 168
      auto.create.topics.enable: false
    storage:
      type: persistent-claim
      size: 100Gi
  zookeeper:
    replicas: 3
    storage:
      type: persistent-claim
      size: 10Gi
```

#### Event Schema Registry
```json
{
  "namespace": "io.inferloop.events",
  "type": "record",
  "name": "ServiceEvent",
  "fields": [
    {"name": "eventId", "type": "string"},
    {"name": "eventType", "type": "string"},
    {"name": "serviceName", "type": "string"},
    {"name": "userId", "type": "string"},
    {"name": "timestamp", "type": "long"},
    {"name": "payload", "type": "string"},
    {"name": "metadata", "type": ["null", {"type": "map", "values": "string"}], "default": null}
  ]
}
```

## Security Architecture

### 1. Zero-Trust Security Model

```yaml
# Network policies for service isolation
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: service-isolation
  namespace: inferloop
spec:
  podSelector:
    matchLabels:
      app: tabular
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: inferloop
      podSelector:
        matchLabels:
          app: api-gateway
    - podSelector:
        matchLabels:
          app: tabular
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: inferloop
      podSelector:
        matchLabels:
          role: database
  - to:
    - namespaceSelector:
        matchLabels:
          name: inferloop
      podSelector:
        matchLabels:
          role: cache
```

### 2. Authentication & Authorization

```yaml
# OAuth2/OIDC configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: auth-config
  namespace: inferloop
data:
  auth.yaml: |
    authentication:
      providers:
        - name: jwt
          type: jwt
          config:
            issuer: https://auth.inferloop.io
            audience: inferloop-api
            jwks_uri: https://auth.inferloop.io/.well-known/jwks.json
        - name: api-key
          type: api-key
          config:
            header_name: X-API-Key
            query_param: api_key
    authorization:
      policies:
        - name: service-access
          rules:
            - resource: /api/tabular/*
              permissions: ["tabular:read", "tabular:write"]
            - resource: /api/textnlp/*
              permissions: ["textnlp:read", "textnlp:write"]
```

### 3. Secrets Management

```yaml
# External Secrets Operator configuration
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
  namespace: inferloop
spec:
  provider:
    vault:
      server: "https://vault.inferloop.io"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "inferloop-services"
---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: service-secrets
  namespace: inferloop
spec:
  refreshInterval: 15m
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: service-secrets
    creationPolicy: Owner
  data:
  - secretKey: database-url
    remoteRef:
      key: inferloop/database
      property: connection_string
  - secretKey: redis-url
    remoteRef:
      key: inferloop/redis
      property: connection_string
  - secretKey: jwt-secret
    remoteRef:
      key: inferloop/auth
      property: jwt_secret
```

## Monitoring and Observability

### 1. Metrics Collection

```yaml
# Prometheus configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    rule_files:
      - /etc/prometheus/rules/*.yml
    
    scrape_configs:
      - job_name: 'kubernetes-pods'
        kubernetes_sd_configs:
        - role: pod
        relabel_configs:
        - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
          action: keep
          regex: true
        - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
          action: replace
          target_label: __metrics_path__
          regex: (.+)
        - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
          action: replace
          regex: ([^:]+)(?::\d+)?;(\d+)
          replacement: $1:$2
          target_label: __address__
        - source_labels: [__meta_kubernetes_namespace]
          action: replace
          target_label: kubernetes_namespace
        - source_labels: [__meta_kubernetes_pod_name]
          action: replace
          target_label: kubernetes_pod_name
        - source_labels: [__meta_kubernetes_pod_label_app]
          action: replace
          target_label: service
```

### 2. Distributed Tracing

```yaml
# Jaeger configuration
apiVersion: jaegertracing.io/v1
kind: Jaeger
metadata:
  name: inferloop-tracing
spec:
  strategy: production
  storage:
    type: elasticsearch
    options:
      es:
        server-urls: https://elasticsearch:9200
        index-prefix: inferloop
  ingester:
    options:
      kafka:
        producer:
          topic: jaeger-spans
          brokers: kafka:9092
  query:
    options:
      query:
        base-path: /jaeger
```

### 3. Centralized Logging

```yaml
# Fluentd configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/*.log
      pos_file /var/log/fluentd-containers.log.pos
      tag kubernetes.*
      read_from_head true
      <parse>
        @type json
        time_format %Y-%m-%dT%H:%M:%S.%NZ
      </parse>
    </source>
    
    <filter kubernetes.**>
      @type kubernetes_metadata
      @id filter_kube_metadata
    </filter>
    
    <filter kubernetes.**>
      @type parser
      key_name log
      <parse>
        @type json
      </parse>
    </filter>
    
    <match kubernetes.**>
      @type elasticsearch
      @id out_es
      @log_level info
      include_tag_key true
      host elasticsearch
      port 9200
      logstash_format true
      logstash_prefix inferloop
      <buffer>
        @type file
        path /var/log/fluentd-buffers/kubernetes.system.buffer
        flush_mode interval
        retry_type exponential_backoff
        flush_thread_count 2
        flush_interval 5s
        retry_forever
        retry_max_interval 30
        chunk_limit_size 2M
        queue_limit_length 8
        overflow_action block
      </buffer>
    </match>
```

## Deployment Patterns

### 1. Blue-Green Deployment

```yaml
# Flagger configuration for automated canary deployments
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: tabular-service
  namespace: inferloop
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: tabular
  service:
    port: 8000
    targetPort: 8000
  analysis:
    interval: 30s
    threshold: 5
    maxWeight: 50
    stepWeight: 10
    metrics:
    - name: request-success-rate
      thresholdRange:
        min: 99
      interval: 1m
    - name: request-duration
      thresholdRange:
        max: 500
      interval: 30s
    webhooks:
    - name: load-test
      url: http://flagger-loadtester.test/
      timeout: 5s
      metadata:
        cmd: "hey -z 1m -q 10 -c 2 http://tabular-canary.inferloop:8000/"
```

### 2. Rolling Updates

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: textnlp
spec:
  replicas: 10
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2
      maxUnavailable: 1
  template:
    spec:
      containers:
      - name: textnlp
        image: inferloop/textnlp:v2.0.0
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 15"]
```

## Disaster Recovery

### 1. Backup Strategy

```yaml
# Velero backup configuration
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: daily-backup
  namespace: velero
spec:
  schedule: "0 2 * * *"
  template:
    hooks:
      resources:
      - name: database-backup
        includedNamespaces:
        - inferloop
        labelSelector:
          matchLabels:
            component: database
        post:
        - exec:
            container: postgres
            command:
            - /bin/bash
            - -c
            - pg_dump -U $POSTGRES_USER $POSTGRES_DB > /backup/dump.sql
            onError: Fail
            timeout: 10m
    includedNamespaces:
    - inferloop
    storageLocation: default
    ttl: 720h0m0s
```

### 2. Multi-Region Failover

```hcl
# Terraform configuration for multi-region setup
module "primary_region" {
  source = "./modules/inferloop-platform"
  
  region = "us-east-1"
  environment = "production"
  
  cluster_config = {
    name = "inferloop-primary"
    version = "1.28"
  }
}

module "dr_region" {
  source = "./modules/inferloop-platform"
  
  region = "us-west-2"
  environment = "production-dr"
  
  cluster_config = {
    name = "inferloop-dr"
    version = "1.28"
  }
  
  # Cross-region replication
  replication_config = {
    source_region = module.primary_region.region
    source_cluster = module.primary_region.cluster_id
  }
}

# Global load balancer for failover
resource "aws_route53_record" "api" {
  zone_id = var.route53_zone_id
  name    = "api.inferloop.io"
  type    = "A"
  
  set_identifier = "primary"
  
  alias {
    name                   = module.primary_region.load_balancer_dns
    zone_id                = module.primary_region.load_balancer_zone_id
    evaluate_target_health = true
  }
  
  failover_routing_policy {
    type = "PRIMARY"
  }
}

resource "aws_route53_record" "api_failover" {
  zone_id = var.route53_zone_id
  name    = "api.inferloop.io"
  type    = "A"
  
  set_identifier = "secondary"
  
  alias {
    name                   = module.dr_region.load_balancer_dns
    zone_id                = module.dr_region.load_balancer_zone_id
    evaluate_target_health = true
  }
  
  failover_routing_policy {
    type = "SECONDARY"
  }
}
```

## Performance Optimization

### 1. Caching Layers

```yaml
# Multi-level caching strategy
CachingStrategy:
  CDN:
    - Static assets
    - API responses (with proper cache headers)
    - Geo-distributed edge locations
  
  APIGateway:
    - Response caching (5-60 seconds)
    - Request coalescing
    - Rate limit counters
  
  Application:
    - Redis for session data
    - In-memory caches for hot data
    - Query result caching
  
  Database:
    - Connection pooling
    - Query plan caching
    - Materialized views
```

### 2. Resource Optimization

```yaml
# Vertical Pod Autoscaler configuration
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: vpa-textnlp
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: textnlp
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: textnlp
      minAllowed:
        cpu: 100m
        memory: 256Mi
      maxAllowed:
        cpu: 2
        memory: 4Gi
      controlledResources: ["cpu", "memory"]
```

## Conclusion

This unified architecture provides a robust, scalable foundation for all Inferloop services while maintaining flexibility and independence. The design emphasizes operational excellence, security, and cost optimization while enabling rapid development and deployment of new services.