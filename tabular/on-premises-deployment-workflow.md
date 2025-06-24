# On-Premises Deployment Workflow - Inferloop Synthetic Data

## Overview

This document explains how on-premises deployment works, from initial setup to production operation. The on-premises deployment allows organizations to run the Inferloop Synthetic Data platform entirely within their own data centers.

## Deployment Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Enterprise Data Center                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Kubernetes     │  │    Storage      │  │  Databases  │ │
│  │    Cluster       │  │   (MinIO/NFS)   │  │ (PostgreSQL)│ │
│  │  ┌───────────┐  │  │                 │  │             │ │
│  │  │ Synthdata │  │  │  ┌───────────┐  │  │ ┌─────────┐ │ │
│  │  │   Pods    │  │  │  │ S3-Like   │  │  │ │Primary  │ │ │
│  │  │           │  │  │  │  Storage  │  │  │ │Instance │ │ │
│  │  └───────────┘  │  │  └───────────┘  │  │ └─────────┘ │ │
│  │                  │  │                 │  │             │ │
│  │  ┌───────────┐  │  │  ┌───────────┐  │  │ ┌─────────┐ │ │
│  │  │Monitoring │  │  │  │   Logs    │  │  │ │Replica  │ │ │
│  │  │  Stack    │  │  │  │  Storage  │  │  │ │Instance │ │ │
│  │  └───────────┘  │  │  └───────────┘  │  │ └─────────┘ │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## How It Works - Step by Step

### Phase 1: Prerequisites Check

```bash
# 1. System requirements check
inferloop-synthetic deploy onprem check-requirements

Checking system requirements...
✓ Operating System: Ubuntu 20.04 LTS (supported)
✓ CPU Cores: 32 (minimum 16 required)
✓ Memory: 128GB (minimum 64GB required)
✓ Storage: 2TB available (minimum 500GB required)
✓ Network: 10Gbps connectivity detected
✓ Container Runtime: containerd 1.6.8 installed
✓ Kubernetes: Not installed (will be installed)

All requirements met. Ready for deployment.
```

### Phase 2: Infrastructure Setup

#### Step 1: Initialize Kubernetes Cluster

```bash
# 2. Initialize on-premises deployment
inferloop-synthetic deploy onprem init --name production-cluster

Initializing on-premises deployment...

? Select Kubernetes distribution:
  > Vanilla Kubernetes (recommended)
    OpenShift
    Rancher
    Docker Swarm

? Select networking plugin:
  > Calico (recommended for security)
    Flannel (simple overlay)
    Cilium (eBPF-based)

? Number of master nodes: 3
? Number of worker nodes: 5

Generating deployment configuration...
Configuration saved to: onprem-deployment.yaml
```

#### Step 2: Deploy Kubernetes

```bash
# 3. Deploy Kubernetes cluster
inferloop-synthetic deploy onprem create-cluster --config onprem-deployment.yaml

Creating Kubernetes cluster...

[Master-1] Installing Kubernetes components...
[Master-1] Initializing control plane...
[Master-1] Generating certificates...
[Master-2] Joining control plane...
[Master-3] Joining control plane...
[Worker-1] Joining cluster...
[Worker-2] Joining cluster...
[Worker-3] Joining cluster...
[Worker-4] Joining cluster...
[Worker-5] Joining cluster...

✓ Kubernetes cluster created successfully
✓ High availability enabled (3 masters)
✓ Network policy enabled
✓ RBAC configured

Cluster endpoint: https://k8s-master-lb.internal:6443
```

### Phase 3: Storage Setup

#### Step 1: Deploy Storage System

```bash
# 4. Setup storage
inferloop-synthetic deploy onprem setup-storage --type distributed

Setting up storage system...

? Select storage backend:
  > MinIO (S3-compatible)
    Ceph (distributed storage)
    NFS (network attached)
    Local persistent volumes

? Storage configuration:
  - Number of MinIO nodes: 4
  - Storage per node: 500GB
  - Replication factor: 2
  - Erasure coding: enabled

Deploying MinIO cluster...
✓ MinIO nodes deployed
✓ Distributed mode enabled
✓ S3 API endpoint: http://minio.synthdata.local:9000
✓ Console available at: http://minio.synthdata.local:9001

Storage capacity: 2TB usable (with redundancy)
```

#### Step 2: Create Storage Classes

```yaml
# Generated storage configuration
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/no-provisioner
volumeBindingMode: WaitForFirstConsumer
---
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: standard
provisioner: minio.io/tenant
parameters:
  storageType: "distributed"
```

### Phase 4: Database Deployment

```bash
# 5. Deploy databases
inferloop-synthetic deploy onprem setup-database --type postgresql --ha true

Deploying PostgreSQL database...

? Database configuration:
  - Version: PostgreSQL 14
  - High Availability: Yes
  - Number of replicas: 2
  - Storage size: 100GB
  - Backup schedule: Daily at 2 AM

Creating PostgreSQL cluster...
✓ Primary instance deployed
✓ Replica instances deployed
✓ Streaming replication configured
✓ Automatic failover enabled
✓ Connection pooling (PgBouncer) deployed

Database endpoint: postgresql://postgres.synthdata.local:5432/synthdata
Replica endpoints: 
  - postgresql://postgres-replica-1.synthdata.local:5432/synthdata
  - postgresql://postgres-replica-2.synthdata.local:5432/synthdata
```

### Phase 5: Application Deployment

#### Step 1: Deploy Core Application

```bash
# 6. Deploy Inferloop Synthetic Data application
inferloop-synthetic deploy onprem install-app --environment production

Deploying Inferloop Synthetic Data...

? Deployment configuration:
  - Replicas: 3
  - CPU per replica: 4 cores
  - Memory per replica: 16GB
  - Autoscaling: enabled (min: 3, max: 10)

Creating application resources...
✓ Namespace created: synthdata-prod
✓ ConfigMaps created
✓ Secrets configured
✓ Deployment created (3/3 replicas ready)
✓ Service created
✓ Ingress configured
✓ SSL certificates generated

Application URL: https://synthdata.company.internal
API Endpoint: https://api.synthdata.company.internal
```

#### Step 2: The Deployment Process

Here's what happens behind the scenes:

```yaml
# 1. Namespace and RBAC
apiVersion: v1
kind: Namespace
metadata:
  name: synthdata-prod
---
# 2. Application Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: synthdata-api
  namespace: synthdata-prod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: synthdata-api
  template:
    metadata:
      labels:
        app: synthdata-api
    spec:
      containers:
      - name: synthdata
        image: inferloop/synthdata:v1.0.0
        resources:
          requests:
            cpu: 4
            memory: 16Gi
          limits:
            cpu: 8
            memory: 32Gi
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: synthdata-secrets
              key: database-url
        - name: S3_ENDPOINT
          value: "http://minio.synthdata.local:9000"
        - name: STORAGE_BUCKET
          value: "synthdata-storage"
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

### Phase 6: Monitoring Setup

```bash
# 7. Setup monitoring
inferloop-synthetic deploy onprem setup-monitoring

Setting up monitoring stack...

Deploying Prometheus...
✓ Prometheus server deployed
✓ Service discovery configured
✓ Metrics retention: 30 days
✓ Alert rules configured

Deploying Grafana...
✓ Grafana deployed
✓ Data sources configured
✓ Dashboards imported:
  - Kubernetes Overview
  - Application Metrics
  - Synthetic Data Generation
  - Resource Utilization

Deploying Alertmanager...
✓ Alertmanager deployed
✓ Notification channels configured:
  - Email: ops-team@company.com
  - Slack: #synthdata-alerts
  - PagerDuty: enabled

Monitoring URLs:
- Prometheus: http://prometheus.synthdata.local:9090
- Grafana: http://grafana.synthdata.local:3000
- Alertmanager: http://alertmanager.synthdata.local:9093
```

### Phase 7: Security Configuration

#### Step 1: Authentication Setup

```bash
# 8. Configure authentication
inferloop-synthetic deploy onprem setup-auth --provider ldap

Configuring authentication...

? LDAP Configuration:
  - Server: ldap://ldap.company.internal:389
  - Base DN: dc=company,dc=internal
  - Bind DN: cn=synthdata,ou=services,dc=company,dc=internal
  - User search base: ou=users,dc=company,dc=internal
  - Group search base: ou=groups,dc=company,dc=internal

Testing LDAP connection...
✓ LDAP connection successful
✓ Found 1,247 users
✓ Found 85 groups

Configuring RBAC...
✓ Admin group mapped: cn=synthdata-admins,ou=groups,dc=company,dc=internal
✓ User group mapped: cn=synthdata-users,ou=groups,dc=company,dc=internal
✓ Viewer group mapped: cn=synthdata-viewers,ou=groups,dc=company,dc=internal

Authentication configured successfully.
```

#### Step 2: SSL/TLS Setup

```bash
# 9. Configure SSL/TLS
inferloop-synthetic deploy onprem setup-tls --provider internal-ca

Setting up TLS...

? Certificate configuration:
  - CA: Internal Company CA
  - Domains: 
    - synthdata.company.internal
    - *.synthdata.company.internal
  - Validity: 365 days
  - Auto-renewal: enabled

Generating certificates...
✓ Certificate request created
✓ Certificate signed by CA
✓ Certificate deployed to Ingress
✓ TLS enabled for all services

Certificate details:
- Subject: CN=synthdata.company.internal
- Issuer: CN=Company Internal CA
- Valid from: 2024-01-15
- Valid until: 2025-01-15
```

### Phase 8: Integration with Existing Systems

```bash
# 10. Configure integrations
inferloop-synthetic deploy onprem configure-integrations

Configuring integrations...

? Select integrations to configure:
  [x] Corporate proxy
  [x] Internal DNS
  [x] Backup system
  [x] Monitoring integration
  [x] Log forwarding

Configuring corporate proxy...
✓ HTTP_PROXY configured: http://proxy.company.internal:8080
✓ NO_PROXY configured: .company.internal,10.0.0.0/8

Configuring DNS...
✓ DNS entries created:
  - synthdata.company.internal → 10.100.50.100
  - api.synthdata.company.internal → 10.100.50.100

Configuring backup integration...
✓ Velero installed
✓ Backup location: nfs://backup.company.internal/synthdata
✓ Schedule: Daily at 2 AM, 30-day retention

Configuring log forwarding...
✓ Logs forwarding to: syslog://logging.company.internal:514
✓ Log format: JSON
✓ Log level: INFO
```

## Day 2 Operations

### Managing the Deployment

```bash
# Check status
inferloop-synthetic deploy onprem status

Cluster Status: Healthy
├── Masters: 3/3 ready
├── Workers: 5/5 ready
├── Storage: 1.2TB/2TB used
└── Database: Primary + 2 replicas healthy

Application Status: Running
├── API Pods: 3/3 ready
├── Worker Pods: 5/5 ready
├── Jobs Completed: 1,247
└── Jobs Failed: 3

Resource Usage:
├── CPU: 45% (36/80 cores)
├── Memory: 62% (248GB/400GB)
├── Storage: 60% (1.2TB/2TB)
└── Network: 120Mbps average

Recent Alerts: None
```

### Scaling Operations

```bash
# Scale workers
inferloop-synthetic deploy onprem scale --component workers --replicas 10

Scaling workers from 5 to 10...
✓ Horizontal pod autoscaler updated
✓ New pods scheduled
✓ 10/10 workers ready

# Add cluster nodes
inferloop-synthetic deploy onprem add-node --type worker --count 2

Adding 2 worker nodes...
✓ Node worker-6 joined cluster
✓ Node worker-7 joined cluster
✓ Pods rescheduled to new nodes
```

### Updating the Application

```bash
# Update to new version
inferloop-synthetic deploy onprem upgrade --version v1.1.0

Upgrading Inferloop Synthetic Data...

Current version: v1.0.0
Target version: v1.1.0

✓ Pre-upgrade checks passed
✓ Database migrations: None required
✓ Rolling update started
  - Pod 1/3 updated
  - Pod 2/3 updated
  - Pod 3/3 updated
✓ Health checks passed
✓ Upgrade completed successfully

New features available:
- Enhanced CTGAN support
- Faster data generation
- New validation metrics
```

## Disaster Recovery

### Backup Process

```bash
# Manual backup
inferloop-synthetic deploy onprem backup --name pre-upgrade-backup

Creating backup...
✓ Application state saved
✓ Database backed up (snapshot + WAL)
✓ Configuration exported
✓ Storage data indexed

Backup completed: pre-upgrade-backup-20240115-1430
Size: 45GB
Location: nfs://backup.company.internal/synthdata/backups/
```

### Restore Process

```bash
# Restore from backup
inferloop-synthetic deploy onprem restore --backup pre-upgrade-backup-20240115-1430

Restoring from backup...

⚠️  This will replace current data. Continue? [y/N]: y

✓ Application scaled down
✓ Database restored
✓ Configuration applied
✓ Storage data restored
✓ Application restarted
✓ Health checks passed

Restore completed successfully.
Time taken: 12 minutes
```

## Air-Gapped Deployment

For completely isolated environments:

```bash
# 1. Create offline bundle
inferloop-synthetic deploy onprem create-offline-bundle --output synthdata-offline.tar

Creating offline deployment bundle...
✓ Container images exported (2.3GB)
✓ Helm charts packaged
✓ Dependencies included
✓ Documentation added

Bundle created: synthdata-offline.tar (2.8GB)

# 2. Transfer to air-gapped environment and deploy
tar -xf synthdata-offline.tar
cd synthdata-offline
./deploy.sh --offline

Deploying in offline mode...
✓ Images loaded to local registry
✓ Helm charts installed
✓ No external connections required
```

## Architecture Decision Points

### Why This Approach?

1. **Kubernetes-First**: 
   - Industry standard for container orchestration
   - Handles scaling, failover, and updates automatically
   - Rich ecosystem of tools and operators

2. **MinIO for Storage**:
   - S3-compatible API (no code changes)
   - Distributed and highly available
   - Works well in air-gapped environments

3. **PostgreSQL for Metadata**:
   - Proven reliability
   - ACID compliance for job tracking
   - Built-in replication

4. **Prometheus/Grafana Stack**:
   - Native Kubernetes integration
   - No external dependencies
   - Powerful querying and visualization

5. **Helm for Packaging**:
   - Templated deployments
   - Easy upgrades and rollbacks
   - Configuration management

## Summary

The on-premises deployment:
1. **Runs entirely in your data center** - no cloud dependencies
2. **Uses standard Kubernetes** - works with existing k8s clusters
3. **Provides high availability** - no single points of failure
4. **Integrates with enterprise systems** - LDAP, DNS, proxies
5. **Supports air-gapped environments** - offline deployment possible
6. **Scales horizontally** - add nodes as needed
7. **Maintains data sovereignty** - all data stays on-premises

This approach gives organizations complete control while maintaining the same features and performance as cloud deployments.