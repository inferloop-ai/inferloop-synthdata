# On-Premises Deployment, Cross-Platform Features, and Testing/Documentation Design

## Table of Contents
1. [On-Premises Deployment Architecture](#on-premises-deployment-architecture)
2. [On-Premises Deployment Workflow](#on-premises-deployment-workflow)
3. [Cross-Platform Features Design](#cross-platform-features-design)
4. [Testing Strategy](#testing-strategy)
5. [Documentation Framework](#documentation-framework)

---

# On-Premises Deployment Architecture

## Overview

The on-premises deployment architecture enables organizations to run the Inferloop Synthetic Data SDK within their own data centers, providing complete control over data, infrastructure, and security. This deployment model is essential for organizations with strict data residency requirements, air-gapped environments, or existing on-premises infrastructure investments.

## Architecture Principles

### 1. **Infrastructure Agnostic**
- Support for bare metal, VMware, OpenStack
- Hardware flexibility (x86_64, ARM64)
- Storage agnostic (SAN, NAS, local)
- Network topology independent

### 2. **Production Grade**
- High availability by default
- Horizontal scaling capability
- Automated failover
- Zero-downtime updates

### 3. **Security First**
- Air-gap compatible
- Enterprise authentication integration
- End-to-end encryption
- Compliance ready (HIPAA, PCI-DSS)

### 4. **Operational Excellence**
- GitOps-based deployment
- Comprehensive monitoring
- Automated backup/restore
- Self-healing capabilities

## Core Components

### 1. Container Orchestration Platforms

#### **Kubernetes (Vanilla)**
```yaml
# Cluster Architecture
Master Nodes: 3 (HA configuration)
Worker Nodes: 3-100 (scalable)
etcd: External cluster (recommended)

# Core Components
- API Server: HA with load balancer
- Controller Manager: Leader election
- Scheduler: Custom policies supported
- kubelet: Container runtime (containerd/CRI-O)
- kube-proxy: IPVS mode for performance

# Networking
- CNI: Calico/Cilium for network policies
- Service Mesh: Optional Istio/Linkerd
- Ingress: NGINX/Traefik with SSL

# Storage
- CSI Drivers: NFS, iSCSI, Ceph, vSphere
- Storage Classes: Fast SSD, Standard HDD
- Persistent Volumes: Dynamic provisioning
```

**Installation Architecture:**
```bash
# Kubeadm-based deployment
kubeadm init --config=kubeadm-config.yaml \
  --control-plane-endpoint="k8s-lb.internal:6443" \
  --upload-certs

# Join additional masters
kubeadm join k8s-lb.internal:6443 \
  --token <token> \
  --discovery-token-ca-cert-hash sha256:<hash> \
  --control-plane --certificate-key <cert-key>

# Join workers
kubeadm join k8s-lb.internal:6443 \
  --token <token> \
  --discovery-token-ca-cert-hash sha256:<hash>
```

#### **OpenShift Container Platform**
```yaml
# OCP 4.x Architecture
Control Plane: 3 masters (minimum)
Compute Nodes: 2+ workers
Infrastructure Nodes: 3 (router, registry, monitoring)

# Additional Features
- Built-in CI/CD (Tekton)
- Service Mesh (Istio-based)
- Serverless (Knative)
- Developer Console
- Integrated Registry

# Security
- SELinux enforcing
- SCC (Security Context Constraints)
- Network Policies by default
- Image signing/scanning
```

#### **Docker Swarm**
```yaml
# Swarm Cluster
Manager Nodes: 3 or 5 (odd number)
Worker Nodes: Unlimited

# Features
- Built-in load balancing
- Service discovery
- Rolling updates
- Secrets management
- Simple networking model

# Deployment
docker swarm init --advertise-addr <MANAGER-IP>
docker swarm join-token manager
docker swarm join-token worker
```

### 2. Storage Solutions

#### **MinIO (S3-Compatible Object Storage)**
```yaml
# Distributed MinIO Architecture
apiVersion: v1
kind: Service
metadata:
  name: minio
spec:
  type: LoadBalancer
  ports:
    - port: 9000
      name: api
    - port: 9001
      name: console
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: minio
spec:
  serviceName: minio
  replicas: 4  # Minimum for erasure coding
  template:
    spec:
      containers:
      - name: minio
        image: minio/minio:latest
        command:
        - /bin/bash
        - -c
        args:
        - minio server http://minio-{0...3}.minio.default.svc.cluster.local/data --console-address :9001
        env:
        - name: MINIO_ROOT_USER
          valueFrom:
            secretKeyRef:
              name: minio-secret
              key: root-user
        - name: MINIO_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: minio-secret
              key: root-password
        volumeMounts:
        - name: data
          mountPath: /data
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 1Ti
```

#### **Persistent Volume Architecture**
```yaml
# Storage Classes
---
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/no-provisioner
volumeBindingMode: WaitForFirstConsumer
reclaimPolicy: Retain
---
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: standard-hdd
provisioner: nfs.csi.k8s.io
parameters:
  server: nfs-server.internal
  share: /kubernetes-volumes
reclaimPolicy: Delete
mountOptions:
  - hard
  - nfsvers=4.1
---
# Ceph/Rook for advanced storage
apiVersion: ceph.rook.io/v1
kind: CephCluster
metadata:
  name: rook-ceph
spec:
  dataDirHostPath: /var/lib/rook
  mon:
    count: 3
    allowMultiplePerNode: false
  dashboard:
    enabled: true
    ssl: true
  storage:
    useAllNodes: true
    useAllDevices: true
```

### 3. Database Solutions

#### **PostgreSQL High Availability**
```yaml
# PostgreSQL Operator (Zalando)
apiVersion: acid.zalan.do/v1
kind: postgresql
metadata:
  name: synthdata-postgres
spec:
  teamId: "synthdata"
  volume:
    size: 100Gi
    storageClass: fast-ssd
  numberOfInstances: 3
  users:
    synthdata:
    - superuser
    - createdb
  databases:
    synthdata: synthdata
  postgresql:
    version: "14"
    parameters:
      max_connections: "200"
      shared_buffers: "256MB"
      effective_cache_size: "1GB"
  patroni:
    initdb:
      encoding: "UTF8"
      locale: "en_US.UTF-8"
    ttl: 30
    loop_wait: 10
    retry_timeout: 10
    maximum_lag_on_failover: 33554432
```

#### **MongoDB Replica Set**
```yaml
# MongoDB Community Operator
apiVersion: mongodbcommunity.mongodb.com/v1
kind: MongoDBCommunity
metadata:
  name: synthdata-mongodb
spec:
  members: 3
  type: ReplicaSet
  version: "5.0.9"
  security:
    authentication:
      modes: ["SCRAM"]
  users:
    - name: synthdata
      db: admin
      passwordSecretRef:
        name: mongodb-password
      roles:
        - name: clusterAdmin
          db: admin
        - name: userAdminAnyDatabase
          db: admin
  statefulSet:
    spec:
      volumeClaimTemplates:
        - metadata:
            name: data-volume
          spec:
            accessModes: ["ReadWriteOnce"]
            resources:
              requests:
                storage: 100Gi
```

### 4. Monitoring & Observability Stack

#### **Prometheus Stack**
```yaml
# Kube-Prometheus-Stack Helm Values
prometheus:
  prometheusSpec:
    replicas: 2
    retention: 30d
    retentionSize: 100GB
    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: fast-ssd
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 100Gi
    serviceMonitorSelectorNilUsesHelmValues: false
    podMonitorSelectorNilUsesHelmValues: false
    ruleSelectorNilUsesHelmValues: false
    
alertmanager:
  alertmanagerSpec:
    replicas: 3
    storage:
      volumeClaimTemplate:
        spec:
          storageClassName: fast-ssd
          resources:
            requests:
              storage: 10Gi

grafana:
  persistence:
    enabled: true
    storageClassName: fast-ssd
    size: 10Gi
  ingress:
    enabled: true
    hosts:
      - grafana.synthdata.internal
  dashboardProviders:
    dashboardproviders.yaml:
      providers:
      - name: 'synthdata'
        folder: 'Synthetic Data'
        type: file
        disableDeletion: true
        editable: true
        options:
          path: /var/lib/grafana/dashboards/synthdata
```

#### **ELK/EFK Stack**
```yaml
# Elasticsearch
apiVersion: elasticsearch.k8s.elastic.co/v1
kind: Elasticsearch
metadata:
  name: synthdata-es
spec:
  version: 8.5.0
  nodeSets:
  - name: masters
    count: 3
    config:
      node.roles: ["master"]
    volumeClaimTemplates:
    - metadata:
        name: elasticsearch-data
      spec:
        accessModes:
        - ReadWriteOnce
        resources:
          requests:
            storage: 10Gi
  - name: data
    count: 3
    config:
      node.roles: ["data", "ingest"]
    volumeClaimTemplates:
    - metadata:
        name: elasticsearch-data
      spec:
        accessModes:
        - ReadWriteOnce
        resources:
          requests:
            storage: 100Gi
---
# Kibana
apiVersion: kibana.k8s.elastic.co/v1
kind: Kibana
metadata:
  name: synthdata-kibana
spec:
  version: 8.5.0
  count: 1
  elasticsearchRef:
    name: synthdata-es
  http:
    tls:
      selfSignedCertificate:
        disabled: true
```

### 5. Security & Authentication

#### **LDAP/Active Directory Integration**
```yaml
# Dex OIDC Provider Configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: dex-config
data:
  config.yaml: |
    issuer: https://dex.synthdata.internal
    storage:
      type: kubernetes
      config:
        inCluster: true
    web:
      http: 0.0.0.0:5556
    connectors:
    - type: ldap
      id: ldap
      name: "Enterprise LDAP"
      config:
        host: ldap.company.internal:636
        rootCA: /etc/dex/ldap-ca.crt
        bindDN: cn=serviceaccount,ou=services,dc=company,dc=com
        bindPW: "$LDAP_PASSWORD"
        userSearch:
          baseDN: ou=users,dc=company,dc=com
          filter: "(objectClass=person)"
          username: uid
          idAttr: uid
          emailAttr: mail
          nameAttr: cn
        groupSearch:
          baseDN: ou=groups,dc=company,dc=com
          filter: "(objectClass=groupOfNames)"
          userMatchers:
          - userAttr: DN
            groupAttr: member
          nameAttr: cn
    oauth2:
      skipApprovalScreen: true
    staticClients:
    - id: kubernetes
      redirectURIs:
      - 'http://localhost:8000'
      - 'https://kubernetes.synthdata.internal/callback'
      name: 'Kubernetes'
      secret: $K8S_OIDC_SECRET
```

#### **Certificate Management**
```yaml
# Cert-Manager Configuration
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: internal-ca-issuer
spec:
  ca:
    secretName: internal-ca-keypair
---
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: synthdata-tls
spec:
  secretName: synthdata-tls-secret
  duration: 8760h # 1 year
  renewBefore: 720h # 30 days
  subject:
    organizations:
      - synthdata
  commonName: synthdata.internal
  dnsNames:
  - synthdata.internal
  - "*.synthdata.internal"
  issuerRef:
    name: internal-ca-issuer
    kind: ClusterIssuer
```

### 6. Deployment Automation

#### **Helm Chart Structure**
```
synthdata-helm/
├── Chart.yaml
├── values.yaml
├── values-dev.yaml
├── values-prod.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── pvc.yaml
│   ├── servicemonitor.yaml
│   └── _helpers.tpl
├── charts/
│   ├── postgresql/
│   ├── mongodb/
│   └── minio/
└── README.md
```

**Main Chart Configuration:**
```yaml
# Chart.yaml
apiVersion: v2
name: inferloop-synthdata
description: Synthetic Data Generation Platform
type: application
version: 1.0.0
appVersion: "1.0.0"
dependencies:
  - name: postgresql
    version: "11.9.13"
    repository: "https://charts.bitnami.com/bitnami"
    condition: postgresql.enabled
  - name: mongodb
    version: "13.3.1"
    repository: "https://charts.bitnami.com/bitnami"
    condition: mongodb.enabled
  - name: minio
    version: "11.10.24"
    repository: "https://charts.bitnami.com/bitnami"
    condition: minio.enabled
```

#### **GitOps with ArgoCD**
```yaml
# ArgoCD Application
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: synthdata-production
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://git.company.internal/synthdata/deployment
    targetRevision: HEAD
    path: environments/production
    helm:
      valueFiles:
      - values-prod.yaml
  destination:
    server: https://kubernetes.default.svc
    namespace: synthdata-prod
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
      allowEmpty: false
    syncOptions:
    - Validate=true
    - CreateNamespace=true
    - PrunePropagationPolicy=foreground
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
```

### 7. Backup & Disaster Recovery

#### **Velero Backup Configuration**
```yaml
# Backup Schedule
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: synthdata-daily-backup
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  template:
    ttl: 720h  # 30 days retention
    includedNamespaces:
    - synthdata-prod
    includedResources:
    - persistentvolumeclaims
    - persistentvolumes
    - configmaps
    - secrets
    - deployments
    - services
    - ingresses
    storageLocation: default
    volumeSnapshotLocations:
    - default
---
# Backup Storage Location
apiVersion: velero.io/v1
kind: BackupStorageLocation
metadata:
  name: default
spec:
  provider: aws
  objectStorage:
    bucket: synthdata-velero-backup
    prefix: k8s-backups
  config:
    region: us-east-1
    s3ForcePathStyle: "true"
    s3Url: http://minio.synthdata.internal:9000
```

---

# On-Premises Deployment Workflow

## How On-Premises Deployment Works

This section explains the practical workflow of deploying Inferloop Synthetic Data on-premises, from initial setup to production operation.

### Deployment Architecture Overview

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

## Step-by-Step Deployment Process

### Phase 1: Prerequisites Validation

```bash
# System requirements check
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

### Phase 2: Kubernetes Cluster Setup

```bash
# Initialize deployment
inferloop-synthetic deploy onprem init --name production-cluster

? Select Kubernetes distribution:
  > Vanilla Kubernetes (recommended)
    OpenShift
    Rancher

? Number of master nodes: 3
? Number of worker nodes: 5

# Deploy cluster
inferloop-synthetic deploy onprem create-cluster --config onprem-deployment.yaml

Creating Kubernetes cluster...
[Master-1] Initializing control plane...
[Master-2,3] Joining control plane...
[Worker-1 to 5] Joining cluster...

✓ Kubernetes cluster created successfully
✓ High availability enabled (3 masters)
Cluster endpoint: https://k8s-master-lb.internal:6443
```

### Phase 3: Storage Configuration

```bash
# Setup distributed storage
inferloop-synthetic deploy onprem setup-storage --type distributed

? Select storage backend:
  > MinIO (S3-compatible)
    Ceph (distributed storage)
    NFS (network attached)

Deploying MinIO cluster...
✓ 4-node MinIO cluster deployed
✓ Distributed mode enabled
✓ S3 API endpoint: http://minio.synthdata.local:9000
✓ Storage capacity: 2TB usable (with redundancy)
```

### Phase 4: Database Deployment

```bash
# Deploy HA PostgreSQL
inferloop-synthetic deploy onprem setup-database --type postgresql --ha true

Creating PostgreSQL cluster...
✓ Primary instance deployed
✓ 2 replica instances deployed
✓ Streaming replication configured
✓ Automatic failover enabled

Database endpoint: postgresql://postgres.synthdata.local:5432/synthdata
```

### Phase 5: Application Installation

```bash
# Deploy core application
inferloop-synthetic deploy onprem install-app --environment production

? Deployment configuration:
  - Replicas: 3
  - CPU per replica: 4 cores
  - Memory per replica: 16GB
  - Autoscaling: enabled (min: 3, max: 10)

✓ Application deployed (3/3 replicas ready)
✓ Service created
✓ Ingress configured
✓ SSL certificates generated

Application URL: https://synthdata.company.internal
API Endpoint: https://api.synthdata.company.internal
```

### Phase 6: Monitoring & Security Setup

```bash
# Setup monitoring
inferloop-synthetic deploy onprem setup-monitoring

✓ Prometheus deployed (30-day retention)
✓ Grafana deployed with dashboards
✓ Alertmanager configured

# Configure authentication
inferloop-synthetic deploy onprem setup-auth --provider ldap

✓ LDAP connection successful
✓ Found 1,247 users
✓ RBAC policies applied

# Setup TLS
inferloop-synthetic deploy onprem setup-tls --provider internal-ca

✓ Certificates generated and deployed
✓ TLS enabled for all services
```

## Day 2 Operations

### Monitoring & Management

```bash
# Check deployment status
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
```

### Scaling Operations

```bash
# Scale application
inferloop-synthetic deploy onprem scale --component workers --replicas 10

# Add cluster nodes
inferloop-synthetic deploy onprem add-node --type worker --count 2
✓ Node worker-6 joined cluster
✓ Node worker-7 joined cluster
```

### Upgrades

```bash
# Upgrade to new version
inferloop-synthetic deploy onprem upgrade --version v1.1.0

✓ Rolling update started
✓ Zero downtime achieved
✓ All health checks passed
```

## Air-Gapped Deployment

For isolated environments without internet access:

```bash
# On internet-connected machine
inferloop-synthetic deploy onprem create-offline-bundle

Creating offline deployment bundle...
✓ Container images exported (2.3GB)
✓ Helm charts packaged
✓ Dependencies included
Bundle created: synthdata-offline.tar (2.8GB)

# In air-gapped environment
tar -xf synthdata-offline.tar
./deploy.sh --offline

Deploying in offline mode...
✓ Images loaded to local registry
✓ No external connections required
```

## Enterprise Integration

```bash
# Configure integrations
inferloop-synthetic deploy onprem configure-integrations

✓ Corporate proxy configured
✓ Internal DNS entries created
✓ Backup system integrated (Velero)
✓ Log forwarding to SIEM enabled
✓ Monitoring integrated with existing systems
```

## Key Benefits

1. **Complete Control**: All infrastructure and data remain on-premises
2. **No Internet Required**: Supports air-gapped deployments
3. **Enterprise Ready**: Integrates with LDAP, DNS, proxies
4. **High Availability**: No single points of failure
5. **Horizontal Scaling**: Add nodes as workload grows
6. **Standard Kubernetes**: Works with existing k8s knowledge

---

# Cross-Platform Features Design

## Overview

Cross-platform features enable seamless operation across multiple cloud providers and on-premises deployments, providing a unified experience regardless of the underlying infrastructure.

## 1. Unified CLI Architecture

### **Command Structure**
```bash
# Provider-agnostic commands
inferloop-synthetic deploy --provider [aws|gcp|azure|onprem] \
                          --config multi-cloud.yaml \
                          --environment production

# Multi-cloud deployment
inferloop-synthetic deploy-multi --primary aws \
                               --secondary gcp \
                               --failover azure \
                               --config ha-deployment.yaml

# Cross-cloud migration
inferloop-synthetic migrate --from aws \
                          --to gcp \
                          --resource-mapping mapping.yaml \
                          --data-sync enabled
```

### **Configuration Management**
```yaml
# multi-cloud.yaml
apiVersion: synthdata.inferloop.io/v1
kind: MultiCloudDeployment
metadata:
  name: synthdata-global
spec:
  providers:
    aws:
      enabled: true
      regions: [us-east-1, eu-west-1]
      priority: 1
      config:
        instance_type: t3.xlarge
        storage_class: gp3
    gcp:
      enabled: true
      regions: [us-central1, europe-west1]
      priority: 2
      config:
        machine_type: n2-standard-4
        storage_type: pd-ssd
    azure:
      enabled: true
      regions: [eastus, westeurope]
      priority: 3
      config:
        vm_size: Standard_D4s_v3
        storage_sku: Premium_LRS
    onprem:
      enabled: false
      datacenters: [dc1, dc2]
      config:
        kubernetes_version: "1.25"
        storage_provider: ceph
  
  routing:
    strategy: geo-proximity  # latency-based, round-robin, failover
    health_check:
      endpoint: /health
      interval: 30s
      timeout: 10s
      threshold: 3
  
  data:
    replication:
      enabled: true
      mode: async  # sync, async
      consistency: eventual  # strong, eventual
    backup:
      enabled: true
      schedule: "0 */6 * * *"
      retention: 30d
```

### **Provider Abstraction Layer**
```python
# base_provider_interface.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class UnifiedCloudProvider(ABC):
    """Unified interface for all cloud providers."""
    
    @abstractmethod
    def deploy_compute(self, config: ComputeConfig) -> DeploymentResult:
        """Deploy compute resources."""
        pass
    
    @abstractmethod
    def deploy_storage(self, config: StorageConfig) -> DeploymentResult:
        """Deploy storage resources."""
        pass
    
    @abstractmethod
    def deploy_network(self, config: NetworkConfig) -> DeploymentResult:
        """Deploy network resources."""
        pass
    
    @abstractmethod
    def get_cost_estimate(self, resources: List[ResourceConfig]) -> CostEstimate:
        """Estimate deployment costs."""
        pass
    
    @abstractmethod
    def migrate_data(self, source: str, target: str, options: MigrationOptions) -> MigrationResult:
        """Migrate data between providers."""
        pass

# provider_factory.py
class CloudProviderFactory:
    """Factory for creating cloud provider instances."""
    
    _providers = {
        'aws': AWSProvider,
        'gcp': GCPProvider,
        'azure': AzureProvider,
        'onprem': OnPremProvider
    }
    
    @classmethod
    def create_provider(cls, provider_type: str, config: Dict[str, Any]) -> UnifiedCloudProvider:
        """Create a cloud provider instance."""
        if provider_type not in cls._providers:
            raise ValueError(f"Unknown provider: {provider_type}")
        
        provider_class = cls._providers[provider_type]
        return provider_class(**config)
    
    @classmethod
    def create_multi_provider(cls, configs: Dict[str, Dict]) -> MultiCloudProvider:
        """Create a multi-cloud provider instance."""
        providers = {}
        for provider_type, config in configs.items():
            providers[provider_type] = cls.create_provider(provider_type, config)
        
        return MultiCloudProvider(providers)
```

## 2. Multi-Cloud Orchestration

### **Deployment Orchestrator**
```python
# orchestrator.py
class MultiCloudOrchestrator:
    """Orchestrates deployments across multiple clouds."""
    
    def __init__(self, providers: Dict[str, UnifiedCloudProvider]):
        self.providers = providers
        self.deployment_state = DeploymentState()
    
    async def deploy_multi_cloud(self, deployment_spec: MultiCloudSpec) -> DeploymentResult:
        """Deploy across multiple clouds."""
        results = {}
        
        # Phase 1: Validate all providers
        for provider_name, provider in self.providers.items():
            if not await provider.validate_credentials():
                raise AuthenticationError(f"Failed to authenticate with {provider_name}")
        
        # Phase 2: Deploy infrastructure in parallel
        deploy_tasks = []
        for provider_name, provider in self.providers.items():
            if deployment_spec.is_provider_enabled(provider_name):
                task = asyncio.create_task(
                    self._deploy_to_provider(provider_name, provider, deployment_spec)
                )
                deploy_tasks.append(task)
        
        # Wait for all deployments
        deploy_results = await asyncio.gather(*deploy_tasks, return_exceptions=True)
        
        # Phase 3: Configure cross-cloud networking
        if deployment_spec.enable_cross_cloud_networking:
            await self._setup_cross_cloud_networking(deploy_results)
        
        # Phase 4: Setup data replication
        if deployment_spec.enable_data_replication:
            await self._setup_data_replication(deploy_results)
        
        return DeploymentResult(
            success=all(r.success for r in deploy_results),
            results=results,
            deployment_id=self.deployment_state.deployment_id
        )
    
    async def _deploy_to_provider(
        self,
        provider_name: str,
        provider: UnifiedCloudProvider,
        spec: MultiCloudSpec
    ) -> ProviderDeploymentResult:
        """Deploy to a specific provider."""
        try:
            # Get provider-specific configuration
            provider_config = spec.get_provider_config(provider_name)
            
            # Deploy infrastructure components
            network = await provider.deploy_network(provider_config.network)
            storage = await provider.deploy_storage(provider_config.storage)
            compute = await provider.deploy_compute(provider_config.compute)
            
            # Register with service discovery
            await self._register_services(provider_name, compute.endpoints)
            
            return ProviderDeploymentResult(
                provider=provider_name,
                success=True,
                resources={
                    'network': network,
                    'storage': storage,
                    'compute': compute
                }
            )
        except Exception as e:
            return ProviderDeploymentResult(
                provider=provider_name,
                success=False,
                error=str(e)
            )
```

### **Cross-Cloud Service Discovery**
```yaml
# Service Registry Configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: cross-cloud-registry
data:
  services.yaml: |
    services:
      synthdata-api:
        endpoints:
          aws:
            - region: us-east-1
              url: https://api-aws-use1.synthdata.io
              priority: 1
              health: /health
          gcp:
            - region: us-central1
              url: https://api-gcp-usc1.synthdata.io
              priority: 2
              health: /health
          azure:
            - region: eastus
              url: https://api-azure-eus.synthdata.io
              priority: 3
              health: /health
        routing:
          strategy: latency-based
          failover:
            enabled: true
            timeout: 30s
      
      synthdata-storage:
        endpoints:
          aws:
            - bucket: synthdata-aws-primary
              region: us-east-1
              type: s3
          gcp:
            - bucket: synthdata-gcp-primary
              region: us-central1
              type: gcs
          azure:
            - container: synthdata-azure-primary
              account: synthdatastorage
              type: blob
        replication:
          mode: async
          consistency: eventual
```

## 3. Unified Monitoring Dashboard

### **Architecture**
```
┌─────────────────────────────────────────────────────────┐
│                 Unified Monitoring Portal                │
├─────────────────────────────────────────────────────────┤
│  Data Collection Layer                                   │
│  ├─ AWS CloudWatch Exporter                            │
│  ├─ GCP Stackdriver Exporter                           │
│  ├─ Azure Monitor Exporter                             │
│  └─ On-Prem Prometheus Federation                      │
├─────────────────────────────────────────────────────────┤
│  Processing Layer                                        │
│  ├─ Metrics Aggregation (Cortex/Thanos)               │
│  ├─ Log Aggregation (Loki)                            │
│  └─ Trace Collection (Tempo)                          │
├─────────────────────────────────────────────────────────┤
│  Visualization Layer                                     │
│  ├─ Grafana Dashboards                                 │
│  ├─ Custom React Dashboard                             │
│  └─ Mobile App                                         │
└─────────────────────────────────────────────────────────┘
```

### **Grafana Dashboard Configuration**
```json
{
  "dashboard": {
    "title": "Inferloop Synthetic Data - Multi-Cloud Overview",
    "panels": [
      {
        "title": "Global Request Rate",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m])) by (provider)",
            "legendFormat": "{{provider}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Cross-Cloud Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, http_request_duration_seconds_bucket)",
            "legendFormat": "p95 latency"
          }
        ],
        "type": "heatmap"
      },
      {
        "title": "Provider Costs",
        "targets": [
          {
            "expr": "sum(cloud_costs_usd) by (provider, service)",
            "legendFormat": "{{provider}} - {{service}}"
          }
        ],
        "type": "piechart"
      },
      {
        "title": "Data Replication Lag",
        "targets": [
          {
            "expr": "max(replication_lag_seconds) by (source, target)",
            "legendFormat": "{{source}} → {{target}}"
          }
        ],
        "type": "table"
      }
    ]
  }
}
```

## 4. Cost Management & Optimization

### **Multi-Cloud Cost Analyzer**
```python
# cost_analyzer.py
class MultiCloudCostAnalyzer:
    """Analyzes and optimizes costs across multiple clouds."""
    
    def __init__(self):
        self.cost_collectors = {
            'aws': AWSCostCollector(),
            'gcp': GCPCostCollector(),
            'azure': AzureCostCollector(),
            'onprem': OnPremCostCalculator()
        }
    
    async def analyze_costs(self, timeframe: TimeFrame) -> CostAnalysis:
        """Analyze costs across all providers."""
        costs = {}
        recommendations = []
        
        # Collect costs from all providers
        for provider, collector in self.cost_collectors.items():
            costs[provider] = await collector.get_costs(timeframe)
        
        # Analyze for optimization opportunities
        total_costs = sum(c.total for c in costs.values())
        
        # Check for underutilized resources
        for provider, cost_data in costs.items():
            underutilized = self._find_underutilized_resources(cost_data)
            if underutilized:
                recommendations.append({
                    'provider': provider,
                    'type': 'rightsizing',
                    'resources': underutilized,
                    'potential_savings': self._calculate_savings(underutilized)
                })
        
        # Check for multi-cloud arbitrage opportunities
        arbitrage = self._find_arbitrage_opportunities(costs)
        if arbitrage:
            recommendations.extend(arbitrage)
        
        # Check for commitment discounts
        commitment_opps = self._analyze_commitment_opportunities(costs)
        if commitment_opps:
            recommendations.extend(commitment_opps)
        
        return CostAnalysis(
            total_cost=total_costs,
            breakdown_by_provider=costs,
            recommendations=recommendations,
            projected_savings=sum(r['potential_savings'] for r in recommendations)
        )
    
    def _find_arbitrage_opportunities(self, costs: Dict[str, CostData]) -> List[Dict]:
        """Find opportunities to move workloads to cheaper providers."""
        opportunities = []
        
        # Compare similar resources across providers
        resource_costs = self._normalize_resource_costs(costs)
        
        for resource_type, provider_costs in resource_costs.items():
            sorted_costs = sorted(provider_costs.items(), key=lambda x: x[1])
            cheapest_provider, cheapest_cost = sorted_costs[0]
            
            for provider, cost in sorted_costs[1:]:
                if cost > cheapest_cost * 1.2:  # 20% threshold
                    opportunities.append({
                        'type': 'arbitrage',
                        'resource': resource_type,
                        'from_provider': provider,
                        'to_provider': cheapest_provider,
                        'current_cost': cost,
                        'new_cost': cheapest_cost,
                        'potential_savings': cost - cheapest_cost
                    })
        
        return opportunities
```

## 5. Migration Tools

### **Data Migration Framework**
```python
# migration_engine.py
class CrossCloudMigrationEngine:
    """Handles data migration between cloud providers."""
    
    def __init__(self):
        self.migration_strategies = {
            ('aws', 'gcp'): AWSS3ToGCSMigration(),
            ('aws', 'azure'): AWSS3ToAzureBlobMigration(),
            ('gcp', 'aws'): GCSToAWSS3Migration(),
            ('gcp', 'azure'): GCSToAzureBlobMigration(),
            ('azure', 'aws'): AzureBlobToAWSS3Migration(),
            ('azure', 'gcp'): AzureBlobToGCSMigration(),
            ('any', 'onprem'): CloudToOnPremMigration(),
            ('onprem', 'any'): OnPremToCloudMigration()
        }
    
    async def migrate_data(
        self,
        source_provider: str,
        target_provider: str,
        migration_spec: MigrationSpec
    ) -> MigrationResult:
        """Migrate data between providers."""
        
        # Get appropriate migration strategy
        strategy_key = (source_provider, target_provider)
        if strategy_key not in self.migration_strategies:
            strategy_key = ('any', 'any')  # Fallback to generic
        
        strategy = self.migration_strategies[strategy_key]
        
        # Pre-migration validation
        validation = await strategy.validate(migration_spec)
        if not validation.is_valid:
            return MigrationResult(
                success=False,
                errors=validation.errors
            )
        
        # Create migration job
        job = MigrationJob(
            id=generate_uuid(),
            source=source_provider,
            target=target_provider,
            spec=migration_spec,
            status='initializing'
        )
        
        try:
            # Phase 1: Inventory
            await job.update_status('inventory')
            inventory = await strategy.create_inventory(migration_spec)
            job.total_objects = inventory.object_count
            job.total_size = inventory.total_size
            
            # Phase 2: Initial sync
            await job.update_status('initial_sync')
            await strategy.initial_sync(inventory, job)
            
            # Phase 3: Incremental sync (if needed)
            if migration_spec.enable_incremental:
                await job.update_status('incremental_sync')
                await strategy.incremental_sync(inventory, job)
            
            # Phase 4: Validation
            await job.update_status('validation')
            validation_result = await strategy.validate_migration(inventory, job)
            
            # Phase 5: Cutover (if specified)
            if migration_spec.auto_cutover and validation_result.is_valid:
                await job.update_status('cutover')
                await strategy.perform_cutover(migration_spec)
            
            await job.update_status('completed')
            return MigrationResult(
                success=True,
                job_id=job.id,
                objects_migrated=job.objects_migrated,
                bytes_transferred=job.bytes_transferred,
                duration=job.duration,
                validation_report=validation_result
            )
            
        except Exception as e:
            await job.update_status('failed', error=str(e))
            return MigrationResult(
                success=False,
                job_id=job.id,
                errors=[str(e)]
            )
```

---

# Testing Strategy

## Overview

Comprehensive testing ensures reliability, performance, and compatibility across all deployment targets and features.

## 1. Unit Testing

### **Test Structure**
```
tests/
├── unit/
│   ├── test_providers/
│   │   ├── test_aws_provider.py
│   │   ├── test_gcp_provider.py
│   │   ├── test_azure_provider.py
│   │   └── test_onprem_provider.py
│   ├── test_core/
│   │   ├── test_base_provider.py
│   │   ├── test_config_parser.py
│   │   └── test_validators.py
│   ├── test_services/
│   │   ├── test_compute_service.py
│   │   ├── test_storage_service.py
│   │   └── test_network_service.py
│   └── test_utils/
│       ├── test_cost_calculator.py
│       └── test_migration_tools.py
├── integration/
├── e2e/
└── performance/
```

### **Provider Unit Tests Example**
```python
# test_aws_provider.py
import pytest
from unittest.mock import Mock, patch, MagicMock
from deploy.aws.provider import AWSProvider
from deploy.base import ResourceConfig, DeploymentResult

class TestAWSProvider:
    """Unit tests for AWS provider."""
    
    @pytest.fixture
    def aws_provider(self):
        """Create AWS provider instance with mocked boto3."""
        with patch('boto3.Session') as mock_session:
            provider = AWSProvider(
                project_id="test-project",
                region="us-east-1"
            )
            provider.session = mock_session
            return provider
    
    @pytest.fixture
    def mock_ec2_client(self):
        """Create mocked EC2 client."""
        client = MagicMock()
        client.run_instances.return_value = {
            'Instances': [{
                'InstanceId': 'i-1234567890abcdef0',
                'State': {'Name': 'pending'},
                'PublicIpAddress': None
            }]
        }
        return client
    
    def test_deploy_infrastructure_success(self, aws_provider, mock_ec2_client):
        """Test successful infrastructure deployment."""
        # Arrange
        aws_provider._get_client = Mock(return_value=mock_ec2_client)
        config = ResourceConfig(
            name="test-resource",
            instance_type="t3.medium",
            disk_size_gb=100
        )
        
        # Act
        result = aws_provider.deploy_infrastructure(config)
        
        # Assert
        assert result.success is True
        assert 'instance_id' in result.resources
        assert result.resources['instance_id'] == 'i-1234567890abcdef0'
        mock_ec2_client.run_instances.assert_called_once()
    
    def test_deploy_infrastructure_failure(self, aws_provider):
        """Test infrastructure deployment failure handling."""
        # Arrange
        aws_provider._get_client = Mock(side_effect=Exception("AWS API Error"))
        config = ResourceConfig(name="test-resource")
        
        # Act
        result = aws_provider.deploy_infrastructure(config)
        
        # Assert
        assert result.success is False
        assert "AWS API Error" in result.message
    
    @pytest.mark.parametrize("instance_type,expected_cost", [
        ("t3.micro", 7.488),
        ("t3.medium", 30.096),
        ("m5.large", 69.12),
        ("c5.xlarge", 122.4)
    ])
    def test_cost_estimation(self, aws_provider, instance_type, expected_cost):
        """Test cost estimation for different instance types."""
        # Arrange
        config = ResourceConfig(
            name="test",
            instance_type=instance_type,
            disk_size_gb=100
        )
        
        # Act
        costs = aws_provider.estimate_cost(config)
        
        # Assert
        assert costs['compute'] == pytest.approx(expected_cost, rel=0.01)
        assert costs['storage'] == 10.0  # 100GB * $0.10
        assert 'total' in costs
```

## 2. Integration Testing

### **Cloud Provider Integration Tests**
```python
# tests/integration/test_aws_integration.py
import pytest
import os
from deploy.aws.provider import AWSProvider

@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv('AWS_INTEGRATION_TESTS'),
    reason="AWS integration tests not enabled"
)
class TestAWSIntegration:
    """Integration tests with real AWS services."""
    
    @pytest.fixture(scope="class")
    def aws_provider(self):
        """Create real AWS provider instance."""
        return AWSProvider(
            project_id="test-integration",
            region=os.getenv('AWS_TEST_REGION', 'us-east-1')
        )
    
    @pytest.fixture(scope="class")
    def test_vpc(self, aws_provider):
        """Create test VPC for integration tests."""
        # Create VPC for testing
        vpc_config = NetworkConfig(
            name="test-vpc",
            cidr="10.0.0.0/16"
        )
        result = aws_provider.deploy_network(vpc_config)
        yield result.resources['vpc_id']
        
        # Cleanup
        aws_provider.destroy_network(result.resources['vpc_id'])
    
    def test_end_to_end_deployment(self, aws_provider, test_vpc):
        """Test complete deployment workflow."""
        # Deploy storage
        storage_config = StorageConfig(
            name="test-bucket",
            storage_class="STANDARD"
        )
        storage_result = aws_provider.deploy_storage(storage_config)
        assert storage_result.success
        
        # Deploy compute
        compute_config = ComputeConfig(
            name="test-instance",
            instance_type="t3.micro",
            vpc_id=test_vpc
        )
        compute_result = aws_provider.deploy_compute(compute_config)
        assert compute_result.success
        
        # Verify resources are accessible
        status = aws_provider.get_status()
        assert len(status['resources']['instances']) > 0
        assert len(status['resources']['buckets']) > 0
        
        # Cleanup
        aws_provider.destroy(compute_result.resources)
        aws_provider.destroy(storage_result.resources)
```

### **Multi-Cloud Integration Tests**
```python
# tests/integration/test_multi_cloud.py
@pytest.mark.integration
@pytest.mark.slow
class TestMultiCloudIntegration:
    """Test multi-cloud deployment scenarios."""
    
    def test_cross_cloud_data_replication(self):
        """Test data replication between AWS and GCP."""
        # Setup providers
        aws = AWSProvider(project_id="multi-cloud-test")
        gcp = GCPProvider(project_id="multi-cloud-test")
        
        # Create buckets in both clouds
        aws_bucket = aws.deploy_storage(StorageConfig(name="test-aws"))
        gcp_bucket = gcp.deploy_storage(StorageConfig(name="test-gcp"))
        
        # Upload test data to AWS
        test_data = b"test synthetic data content"
        aws.upload_data(aws_bucket.resources['bucket_name'], "test.csv", test_data)
        
        # Setup replication
        replication_config = ReplicationConfig(
            source_provider="aws",
            source_bucket=aws_bucket.resources['bucket_name'],
            target_provider="gcp",
            target_bucket=gcp_bucket.resources['bucket_name']
        )
        
        replicator = CrossCloudReplicator()
        result = replicator.setup_replication(replication_config)
        assert result.success
        
        # Verify data appears in GCP
        import time
        time.sleep(30)  # Wait for replication
        
        gcp_data = gcp.download_data(gcp_bucket.resources['bucket_name'], "test.csv")
        assert gcp_data == test_data
```

## 3. End-to-End Testing

### **Deployment Scenarios**
```python
# tests/e2e/test_deployment_scenarios.py
@pytest.mark.e2e
class TestDeploymentScenarios:
    """End-to-end deployment scenario tests."""
    
    def test_high_availability_deployment(self):
        """Test HA deployment across multiple regions."""
        config = """
        deployment:
          name: synthdata-ha
          type: high-availability
          providers:
            aws:
              regions: [us-east-1, us-west-2]
              instance_count: 2
            gcp:
              regions: [us-central1, us-east1]
              instance_count: 2
        """
        
        # Deploy
        result = run_cli_command(f"deploy --config-string '{config}'")
        assert result.exit_code == 0
        
        # Verify all instances are running
        status = run_cli_command("status --format json")
        status_data = json.loads(status.output)
        
        assert len(status_data['aws']['instances']) == 4
        assert len(status_data['gcp']['instances']) == 4
        
        # Test failover
        # Simulate failure in primary region
        simulate_region_failure('aws', 'us-east-1')
        
        # Verify traffic redirects to other regions
        response = requests.get("https://api.synthdata-test.io/health")
        assert response.status_code == 200
        assert response.headers['X-Served-By'] != 'aws-us-east-1'
    
    def test_data_pipeline_deployment(self):
        """Test complete data pipeline deployment."""
        # Deploy data pipeline
        result = run_cli_command(
            "deploy --template data-pipeline "
            "--provider aws "
            "--config pipeline.yaml"
        )
        assert result.exit_code == 0
        
        # Submit synthetic data generation job
        job_result = run_cli_command(
            "generate --input s3://test-data/source.csv "
            "--output s3://test-data/synthetic.csv "
            "--generator-type sdv "
            "--num-samples 10000"
        )
        assert job_result.exit_code == 0
        
        # Monitor job progress
        job_id = json.loads(job_result.output)['job_id']
        
        # Wait for completion
        wait_for_job_completion(job_id, timeout=600)
        
        # Verify output
        output_exists = check_s3_object_exists(
            "test-data",
            "synthetic.csv"
        )
        assert output_exists
```

## 4. Performance Testing

### **Load Testing**
```python
# tests/performance/test_load.py
import asyncio
import aiohttp
from locust import HttpUser, task, between

class SyntheticDataUser(HttpUser):
    """Load test user for synthetic data API."""
    wait_time = between(1, 5)
    
    @task(weight=3)
    def generate_small_dataset(self):
        """Generate small synthetic dataset."""
        self.client.post("/api/generate", json={
            "input_data": "sample_small.csv",
            "num_samples": 1000,
            "generator_type": "ctgan"
        })
    
    @task(weight=1)
    def generate_large_dataset(self):
        """Generate large synthetic dataset."""
        self.client.post("/api/generate", json={
            "input_data": "sample_large.csv",
            "num_samples": 100000,
            "generator_type": "sdv"
        })
    
    @task(weight=2)
    def check_job_status(self):
        """Check job status."""
        # Get a random job ID from previous requests
        if hasattr(self, 'job_ids') and self.job_ids:
            job_id = random.choice(self.job_ids)
            self.client.get(f"/api/jobs/{job_id}")

# Async performance test
async def test_concurrent_generation():
    """Test concurrent data generation requests."""
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        # Create 100 concurrent requests
        for i in range(100):
            task = asyncio.create_task(
                generate_synthetic_data(session, f"dataset_{i}")
            )
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks)
        
        # Analyze results
        successful = sum(1 for r in results if r['success'])
        avg_time = sum(r['duration'] for r in results) / len(results)
        
        assert successful >= 95  # 95% success rate
        assert avg_time < 10.0  # Average response time under 10s
```

### **Benchmark Testing**
```python
# tests/performance/test_benchmarks.py
@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_deployment_speed(self, benchmark):
        """Benchmark deployment speed."""
        provider = AWSProvider(project_id="benchmark")
        config = ResourceConfig(
            name="benchmark-test",
            instance_type="t3.micro"
        )
        
        # Benchmark deployment
        result = benchmark(provider.deploy_infrastructure, config)
        assert result.success
        
        # Cleanup
        provider.destroy(result.resources)
    
    def test_data_generation_performance(self, benchmark):
        """Benchmark synthetic data generation."""
        generator = SDVGenerator()
        data = pd.read_csv("tests/fixtures/sample_1000.csv")
        
        # Benchmark generation
        def generate():
            generator.fit(data)
            return generator.generate(num_samples=10000)
        
        synthetic_data = benchmark(generate)
        assert len(synthetic_data) == 10000
    
    def test_migration_throughput(self, benchmark):
        """Benchmark data migration throughput."""
        source = MockS3Bucket(size_gb=10)
        target = MockGCSBucket()
        migrator = DataMigrator()
        
        # Benchmark migration
        result = benchmark(
            migrator.migrate,
            source=source,
            target=target,
            parallel_streams=10
        )
        
        # Should achieve at least 100MB/s
        throughput_mbps = (10 * 1024) / result.duration
        assert throughput_mbps >= 100
```

## 5. Security Testing

### **Security Test Suite**
```python
# tests/security/test_security.py
import requests
from zapv2 import ZAPv2

@pytest.mark.security
class TestSecurity:
    """Security testing suite."""
    
    def test_api_authentication(self, api_endpoint):
        """Test API requires authentication."""
        # Try without auth
        response = requests.get(f"{api_endpoint}/api/data")
        assert response.status_code == 401
        
        # Try with invalid token
        headers = {"Authorization": "Bearer invalid-token"}
        response = requests.get(f"{api_endpoint}/api/data", headers=headers)
        assert response.status_code == 403
        
        # Try with valid token
        token = get_test_auth_token()
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{api_endpoint}/api/data", headers=headers)
        assert response.status_code == 200
    
    def test_sql_injection(self, api_endpoint):
        """Test for SQL injection vulnerabilities."""
        payloads = [
            "' OR '1'='1",
            "1; DROP TABLE users--",
            "admin'--",
            "' UNION SELECT * FROM users--"
        ]
        
        for payload in payloads:
            response = requests.get(
                f"{api_endpoint}/api/search",
                params={"q": payload}
            )
            # Should not return database errors
            assert "SQL" not in response.text
            assert "syntax error" not in response.text
            assert response.status_code in [200, 400]
    
    def test_zap_security_scan(self, api_endpoint):
        """Run OWASP ZAP security scan."""
        zap = ZAPv2(proxies={'http': 'http://127.0.0.1:8080'})
        
        # Spider the API
        scan_id = zap.spider.scan(api_endpoint)
        while int(zap.spider.status(scan_id)) < 100:
            time.sleep(2)
        
        # Run active scan
        scan_id = zap.ascan.scan(api_endpoint)
        while int(zap.ascan.status(scan_id)) < 100:
            time.sleep(5)
        
        # Check for high-risk alerts
        alerts = zap.core.alerts(baseurl=api_endpoint)
        high_risk_alerts = [a for a in alerts if a['risk'] == 'High']
        
        assert len(high_risk_alerts) == 0, f"Found high-risk vulnerabilities: {high_risk_alerts}"
```

---

# Documentation Framework

## Overview

Comprehensive documentation ensures users can effectively deploy, operate, and troubleshoot the Inferloop Synthetic Data platform.

## 1. Documentation Structure

```
docs/
├── getting-started/
│   ├── README.md
│   ├── quickstart.md
│   ├── installation/
│   │   ├── aws.md
│   │   ├── gcp.md
│   │   ├── azure.md
│   │   └── on-premises.md
│   └── first-deployment.md
├── architecture/
│   ├── overview.md
│   ├── aws-architecture.md
│   ├── gcp-architecture.md
│   ├── azure-architecture.md
│   ├── on-premises-architecture.md
│   └── security-architecture.md
├── deployment/
│   ├── planning.md
│   ├── prerequisites.md
│   ├── deployment-guide.md
│   ├── configuration.md
│   ├── multi-cloud.md
│   └── troubleshooting.md
├── operations/
│   ├── monitoring.md
│   ├── scaling.md
│   ├── backup-restore.md
│   ├── disaster-recovery.md
│   ├── maintenance.md
│   └── upgrades.md
├── api-reference/
│   ├── rest-api.md
│   ├── cli-reference.md
│   ├── sdk-reference.md
│   └── openapi.yaml
├── tutorials/
│   ├── basic-deployment.md
│   ├── ha-deployment.md
│   ├── multi-region.md
│   ├── data-migration.md
│   └── cost-optimization.md
└── reference/
    ├── configuration-reference.md
    ├── error-codes.md
    ├── glossary.md
    └── faq.md
```

## 2. Documentation Standards

### **Style Guide**
```markdown
# Documentation Style Guide

## General Principles
- Write in clear, concise language
- Use active voice
- Present tense for descriptions
- Imperative mood for instructions

## Structure
- Use hierarchical headings (H1 -> H2 -> H3)
- Include table of contents for long documents
- Add "Prerequisites" section when needed
- Include "Next Steps" at the end

## Code Examples
- Test all code examples
- Include language identifier for syntax highlighting
- Show both command and expected output
- Provide context and explanation

## Formatting
- Use **bold** for UI elements and emphasis
- Use `code` for commands, filenames, and values
- Use > blockquotes for important notes
- Use tables for structured data

## Images and Diagrams
- Use Mermaid for architecture diagrams
- Include alt text for accessibility
- Keep file sizes under 500KB
- Use PNG for screenshots, SVG for diagrams
```

### **Documentation Template**
```markdown
# [Feature/Component Name]

## Overview
Brief description of what this document covers and why it's important.

## Prerequisites
- Required knowledge
- Required tools/access
- Related documentation

## Architecture
[Include architecture diagram if applicable]

## Configuration
### Basic Configuration
```yaml
# Example configuration
```

### Advanced Configuration
Detailed configuration options with examples.

## Deployment
Step-by-step deployment instructions.

### Step 1: Prepare Environment
Detailed instructions...

### Step 2: Deploy Resources
Detailed instructions...

## Verification
How to verify the deployment was successful.

## Monitoring
What metrics to monitor and how.

## Troubleshooting
### Common Issues
#### Issue 1: [Description]
**Symptoms:**
- Symptom 1
- Symptom 2

**Resolution:**
1. Step 1
2. Step 2

## Security Considerations
Security best practices and recommendations.

## Next Steps
- Link to related documentation
- Suggested next actions
```

## 3. API Documentation

### **OpenAPI Specification**
```yaml
openapi: 3.0.0
info:
  title: Inferloop Synthetic Data API
  version: 1.0.0
  description: API for synthetic data generation and management
  contact:
    name: API Support
    email: api-support@inferloop.io

servers:
  - url: https://api.synthdata.io/v1
    description: Production server
  - url: https://staging-api.synthdata.io/v1
    description: Staging server

paths:
  /generate:
    post:
      summary: Generate synthetic data
      operationId: generateSyntheticData
      tags:
        - Generation
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/GenerationRequest'
      responses:
        '202':
          description: Generation job accepted
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/GenerationResponse'
        '400':
          description: Invalid request
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
                
  /jobs/{jobId}:
    get:
      summary: Get job status
      operationId: getJobStatus
      tags:
        - Jobs
      parameters:
        - name: jobId
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: Job status
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/JobStatus'

components:
  schemas:
    GenerationRequest:
      type: object
      required:
        - input_data
        - generator_type
      properties:
        input_data:
          type: string
          description: Path to input data
        generator_type:
          type: string
          enum: [sdv, ctgan, ydata]
        num_samples:
          type: integer
          minimum: 1
          default: 1000
        config:
          type: object
          additionalProperties: true
          
    GenerationResponse:
      type: object
      properties:
        job_id:
          type: string
          format: uuid
        status:
          type: string
          enum: [accepted, processing, completed, failed]
        created_at:
          type: string
          format: date-time
```

## 4. Automated Documentation

### **Documentation Generation**
```python
# docs/generate_docs.py
import os
import ast
import inspect
from typing import List, Dict
from pathlib import Path

class DocumentationGenerator:
    """Generates documentation from source code."""
    
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_api_docs(self):
        """Generate API documentation from source code."""
        modules = self._find_python_modules()
        
        for module_path in modules:
            module_docs = self._extract_module_docs(module_path)
            self._write_module_docs(module_docs)
    
    def generate_cli_docs(self):
        """Generate CLI documentation from click commands."""
        from click.testing import CliRunner
        from inferloop_synthetic.cli import cli
        
        runner = CliRunner()
        
        # Get all commands
        commands = self._get_all_commands(cli)
        
        # Generate docs for each command
        docs = []
        for cmd_name, cmd_obj in commands.items():
            result = runner.invoke(cmd_obj, ['--help'])
            docs.append({
                'name': cmd_name,
                'help': result.output
            })
        
        self._write_cli_docs(docs)
    
    def generate_config_schema_docs(self):
        """Generate configuration schema documentation."""
        from inferloop_synthetic.config import ConfigSchema
        
        schema = ConfigSchema()
        schema_docs = self._extract_schema_docs(schema)
        self._write_schema_docs(schema_docs)
```

### **Docstring Standards**
```python
def deploy_infrastructure(
    provider: str,
    config: Dict[str, Any],
    dry_run: bool = False
) -> DeploymentResult:
    """Deploy infrastructure to specified cloud provider.
    
    This function handles the complete infrastructure deployment process,
    including resource creation, configuration, and validation.
    
    Args:
        provider: Cloud provider name ('aws', 'gcp', 'azure', 'onprem').
        config: Deployment configuration dictionary containing:
            - name (str): Deployment name
            - region (str): Target region
            - resources (dict): Resource specifications
        dry_run: If True, validate configuration without deploying.
    
    Returns:
        DeploymentResult: Object containing:
            - success (bool): Whether deployment succeeded
            - resources (dict): Created resource identifiers
            - errors (list): Any errors encountered
    
    Raises:
        ValidationError: If configuration is invalid.
        AuthenticationError: If provider authentication fails.
        DeploymentError: If deployment fails.
    
    Example:
        >>> config = {
        ...     'name': 'prod-deployment',
        ...     'region': 'us-east-1',
        ...     'resources': {
        ...         'compute': {'type': 'ec2', 'count': 3}
        ...     }
        ... }
        >>> result = deploy_infrastructure('aws', config)
        >>> print(f"Deployed: {result.success}")
        Deployed: True
    
    Note:
        Ensure provider credentials are configured before calling.
        See :ref:`authentication` for credential setup.
    """
```

## 5. Documentation CI/CD

### **Documentation Pipeline**
```yaml
# .github/workflows/documentation.yml
name: Documentation

on:
  push:
    branches: [main]
    paths:
      - 'docs/**'
      - 'src/**'
      - 'mkdocs.yml'
  pull_request:
    paths:
      - 'docs/**'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r docs/requirements.txt
        pip install -e .
    
    - name: Generate API docs
      run: |
        python docs/generate_docs.py
    
    - name: Build documentation
      run: |
        mkdocs build --strict
    
    - name: Check links
      run: |
        pip install linkchecker
        linkchecker site/
    
    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./site
```

### **Documentation Testing**
```python
# tests/test_documentation.py
import pytest
import subprocess
from pathlib import Path

class TestDocumentation:
    """Test documentation completeness and accuracy."""
    
    def test_all_modules_documented(self):
        """Ensure all modules have documentation."""
        src_modules = set(Path('src').rglob('*.py'))
        doc_modules = set(Path('docs/api').rglob('*.md'))
        
        undocumented = src_modules - doc_modules
        assert len(undocumented) == 0, f"Undocumented modules: {undocumented}"
    
    def test_code_examples_run(self):
        """Test that code examples in docs actually run."""
        doc_files = Path('docs').rglob('*.md')
        
        for doc_file in doc_files:
            examples = extract_code_examples(doc_file)
            for example in examples:
                if example.language == 'python':
                    # Test Python examples
                    exec(example.code)
                elif example.language == 'bash':
                    # Test shell commands
                    result = subprocess.run(
                        example.code,
                        shell=True,
                        capture_output=True,
                        text=True
                    )
                    assert result.returncode == 0
    
    def test_links_valid(self):
        """Test all documentation links are valid."""
        from linkchecker import check_links
        
        results = check_links('docs/')
        broken_links = [r for r in results if r.status != 200]
        
        assert len(broken_links) == 0, f"Broken links: {broken_links}"
```

## Conclusion

This comprehensive design document outlines:

1. **On-Premises Deployment**: Complete architecture for Kubernetes, storage, databases, monitoring, and security in private data centers

2. **Cross-Platform Features**: Unified CLI, multi-cloud orchestration, migration tools, and centralized monitoring across all providers

3. **Testing Strategy**: Unit, integration, E2E, performance, and security testing frameworks ensuring quality and reliability

4. **Documentation Framework**: Structured documentation with automated generation, API references, and comprehensive guides

These components complete the Inferloop Synthetic Data platform, enabling deployment across any infrastructure while maintaining consistency, reliability, and ease of use.