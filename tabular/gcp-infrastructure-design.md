# GCP Infrastructure Design Description - Inferloop Synthetic Data

## Overview

This document describes the Google Cloud Platform (GCP) infrastructure design for the Inferloop Synthetic Data SDK. The architecture leverages GCP's fully managed services to provide a scalable, secure, and cost-effective platform for synthetic data generation workloads.

## Architecture Principles

### 1. **Serverless First**
- Prioritizes fully managed services (Cloud Run, Cloud Functions)
- Reduces operational overhead
- Automatic scaling from zero to thousands of instances
- Pay-per-use pricing model

### 2. **Global Scale**
- Multi-region deployment capabilities
- Global load balancing
- Edge caching with Cloud CDN
- Low-latency access worldwide

### 3. **Security by Default**
- Identity-based access control
- Automatic encryption
- Private Google network backbone
- Built-in DDoS protection

### 4. **Developer Friendly**
- Native container support
- GitOps-ready deployments
- Integrated CI/CD with Cloud Build
- Comprehensive monitoring

## Core Components

### 1. Networking Infrastructure

#### **VPC Architecture**
```
┌─────────────────────────────────────────────────────────────┐
│                    VPC Network                               │
│                  (10.0.0.0/16)                              │
├─────────────────────────┬───────────────────────────────────┤
│   Region: us-central1   │      Region: us-east1            │
├─────────────────────────┼───────────────────────────────────┤
│  Subnet: 10.0.1.0/24   │   Subnet: 10.0.2.0/24            │
│  ├─ GKE Cluster        │   ├─ GKE Cluster (DR)            │
│  ├─ Cloud SQL         │   ├─ Cloud SQL Replica           │
│  └─ Private Services  │   └─ Private Services            │
├─────────────────────────┴───────────────────────────────────┤
│                 Global Resources                             │
│  ├─ Cloud Load Balancer                                    │
│  ├─ Cloud CDN                                              │
│  └─ Cloud Armor (DDoS Protection)                          │
└─────────────────────────────────────────────────────────────┘
```

#### **Network Features**
- **Private Google Access**: Enables private IPs to access Google APIs
- **Cloud NAT**: Provides outbound internet for private resources
- **Firewall Rules**: Hierarchical security policies
- **VPC Peering**: Connect to other VPCs or on-premises
- **Shared VPC**: Centralized network management

### 2. Compute Services

#### **Cloud Run (Serverless Containers)**
- **Configuration**:
  ```yaml
  apiVersion: serving.knative.dev/v1
  kind: Service
  metadata:
    name: inferloop-synthdata
    annotations:
      run.googleapis.com/execution-environment: gen2
  spec:
    template:
      metadata:
        annotations:
          autoscaling.knative.dev/minScale: "0"
          autoscaling.knative.dev/maxScale: "1000"
          run.googleapis.com/cpu-throttling: "false"
      spec:
        containerConcurrency: 1000
        timeoutSeconds: 900
        serviceAccountName: synthdata-sa
        containers:
        - image: gcr.io/PROJECT_ID/synthdata:latest
          resources:
            limits:
              cpu: "4"
              memory: "8Gi"
          env:
          - name: GCS_BUCKET
            value: synthdata-storage
  ```

- **Features**:
  - Automatic HTTPS endpoints
  - Built-in load balancing
  - 0-1000 instance auto-scaling
  - 15-minute request timeout
  - Up to 32GB memory and 8 vCPUs

#### **Google Kubernetes Engine (GKE)**
- **Cluster Architecture**:
  ```
  ┌────────────────────────────────────────┐
  │          GKE Autopilot Cluster         │
  ├────────────────────────────────────────┤
  │  Control Plane (Google Managed)        │
  ├────────────────────────────────────────┤
  │  Node Pools:                           │
  │  ├─ Default: n2-standard-4 (2-10)     │
  │  ├─ GPU: nvidia-tesla-t4 (optional)   │
  │  └─ Spot: e2-standard-4 (0-20)        │
  ├────────────────────────────────────────┤
  │  Add-ons:                              │
  │  ├─ Istio Service Mesh                │
  │  ├─ Config Connector                  │
  │  └─ Workload Identity                 │
  └────────────────────────────────────────┘
  ```

- **Features**:
  - Autopilot mode for hands-off operations
  - Automatic node provisioning
  - Integrated monitoring and logging
  - Binary authorization
  - Pod security policies

#### **Cloud Functions (Event-Driven)**
- **Function Configuration**:
  ```python
  # Function with enhanced settings
  @functions_framework.http
  def process_synthetic_data(request):
      """HTTP Cloud Function for data processing."""
      
      # Configuration
      memory = 8192  # 8GB
      timeout = 540  # 9 minutes
      max_instances = 1000
      min_instances = 0
      
      # Process synthetic data generation
      return generate_synthetic_data(request)
  ```

- **Triggers**:
  - HTTP endpoints
  - Cloud Storage events
  - Pub/Sub messages
  - Firestore changes
  - Cloud Scheduler

### 3. Storage Services

#### **Cloud Storage**
- **Bucket Structure**:
  ```
  synthdata-production/
  ├── raw-data/           # Original datasets
  │   └── 2024-01-15/    # Date partitioned
  ├── synthetic-data/     # Generated data
  │   ├── sdv/           # By generator type
  │   └── ctgan/
  ├── models/            # Trained models
  └── temp/              # Temporary processing
  ```

- **Storage Classes**:
  - **Standard**: Frequently accessed data
  - **Nearline**: Data accessed once per month
  - **Coldline**: Quarterly access
  - **Archive**: Yearly access

- **Features**:
  - Object lifecycle management
  - Versioning and retention
  - Customer-managed encryption keys (CMEK)
  - Signed URLs for secure access
  - Pub/Sub notifications

#### **Firestore (NoSQL Database)**
- **Data Model**:
  ```
  /projects/{projectId}
    /jobs/{jobId}
      - status: "processing"
      - created: timestamp
      - generator: "sdv"
      - parameters: {}
      - results: {}
    /datasets/{datasetId}
      - name: "customer_data"
      - schema: {}
      - metadata: {}
  ```

- **Features**:
  - Real-time synchronization
  - Offline support
  - ACID transactions
  - Global replication
  - Automatic scaling

#### **Cloud SQL (PostgreSQL)**
- **Instance Configuration**:
  - Version: PostgreSQL 14
  - Machine type: db-n1-standard-4
  - Storage: 100GB SSD with auto-resize
  - High Availability: Regional with failover replica
  - Automated backups: 7-day retention
  - Point-in-time recovery

### 4. Security Architecture

#### **Identity and Access Management (IAM)**
```
┌─────────────────────────────────────────┐
│         IAM Hierarchy                    │
├─────────────────────────────────────────┤
│  Service Accounts:                      │
│  ├─ synthdata-cloudrun@                │
│  │  └─ roles/storage.objectAdmin       │
│  ├─ synthdata-gke@                     │
│  │  ├─ roles/container.developer       │
│  │  └─ roles/logging.logWriter         │
│  └─ synthdata-function@                │
│      └─ roles/cloudfunctions.invoker   │
├─────────────────────────────────────────┤
│  Workload Identity:                    │
│  ├─ Kubernetes SA ←→ Google SA         │
│  └─ Pod-level authentication           │
└─────────────────────────────────────────┘
```

#### **Network Security**
- **Cloud Armor**:
  - DDoS protection
  - Geographic restrictions
  - Rate limiting
  - OWASP rules

- **VPC Service Controls**:
  - API access boundaries
  - Data exfiltration prevention
  - Context-aware access

#### **Data Security**
- **Encryption**:
  - At-rest: Automatic with Google-managed keys
  - In-transit: TLS 1.3 everywhere
  - CMEK option for compliance

- **Secret Management**:
  - Secret Manager for sensitive data
  - Automatic rotation
  - Audit logging
  - Version control

### 5. Monitoring and Observability

#### **Cloud Monitoring Stack**
```
┌────────────────────────────────────────────────┐
│           Observability Platform                │
├────────────────────────────────────────────────┤
│  Metrics Collection:                           │
│  ├─ System Metrics (CPU, Memory, Network)     │
│  ├─ Application Metrics (Custom)              │
│  ├─ SLI/SLO Tracking                         │
│  └─ Cost Metrics                             │
├────────────────────────────────────────────────┤
│  Logging:                                      │
│  ├─ Cloud Logging (Centralized)              │
│  ├─ Log-based Metrics                        │
│  ├─ Error Reporting                          │
│  └─ Log Router (Export to BigQuery)          │
├────────────────────────────────────────────────┤
│  Tracing:                                      │
│  ├─ Cloud Trace (Distributed)                │
│  ├─ Latency Analysis                         │
│  └─ Service Dependencies                     │
├────────────────────────────────────────────────┤
│  Alerting:                                     │
│  ├─ Alert Policies                           │
│  ├─ Notification Channels                    │
│  └─ Incident Management                      │
└────────────────────────────────────────────────┘
```

#### **SRE Practices**
- **Service Level Objectives (SLOs)**:
  - Availability: 99.9%
  - Latency: p95 < 500ms
  - Error rate: < 0.1%

- **Error Budgets**:
  - Monthly error budget tracking
  - Automated deployment freezes
  - Postmortem culture

### 6. Deployment Patterns

#### **Continuous Deployment**
```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   GitHub    │───►│ Cloud Build  │───►│   Deploy    │
│   Push      │    │              │    │             │
└─────────────┘    └──────┬───────┘    └──────┬──────┘
                          │                    │
                    ┌─────▼─────┐        ┌────▼────┐
                    │  Tests    │        │  Stages │
                    │  - Unit   │        │  - Dev  │
                    │  - Integ  │        │  - Stg  │
                    │  - Sec    │        │  - Prod │
                    └───────────┘        └─────────┘
```

#### **Blue-Green Deployment (Cloud Run)**
- Traffic splitting: 0% → 10% → 50% → 100%
- Automatic rollback on errors
- Preview URLs for testing
- Gradual rollout with monitoring

### 7. Data Processing Architecture

#### **Batch Processing**
```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Cloud     │───►│   Dataflow   │───►│   Cloud     │
│  Storage    │    │   Pipeline   │    │  Storage    │
│  (Input)    │    │              │    │  (Output)   │
└─────────────┘    └──────────────┘    └─────────────┘
                          │
                    ┌─────▼─────┐
                    │ BigQuery  │
                    │ Analytics │
                    └───────────┘
```

#### **Real-time Processing**
- Pub/Sub for message ingestion
- Cloud Functions for processing
- Firestore for real-time updates
- BigQuery for analytics

### 8. Cost Optimization

#### **Resource Optimization**
1. **Committed Use Discounts**:
   - 1 or 3-year commitments
   - Up to 57% discount
   - Applies to Compute Engine and GKE

2. **Preemptible/Spot VMs**:
   - 80% cheaper than regular VMs
   - Perfect for batch processing
   - Automatic restart handling

3. **Autoscaling**:
   - Scale to zero when idle
   - Predictive autoscaling
   - Schedule-based scaling

#### **Storage Optimization**
- Lifecycle policies for data archival
- Nearline/Coldline for infrequent access
- Object composition for large files
- Parallel composite uploads

### 9. Multi-Region Architecture

#### **Global Load Balancing**
```
┌─────────────────────────────────────────┐
│         Global Load Balancer             │
├─────────────┬─────────────┬─────────────┤
│  us-central1│   us-east1  │ europe-west1│
├─────────────┼─────────────┼─────────────┤
│ Cloud Run   │  Cloud Run  │  Cloud Run  │
│ Instance    │  Instance   │  Instance   │
└─────────────┴─────────────┴─────────────┘
              │
        ┌─────▼─────┐
        │  Anycast  │
        │    IPs    │
        └───────────┘
```

#### **Data Replication**:
- Cloud Storage multi-region buckets
- Firestore multi-region replication
- Cloud SQL cross-region replicas
- Cloud Spanner global consistency

### 10. Disaster Recovery

#### **Backup Strategy**
- **Cloud SQL**: Automated daily backups
- **Firestore**: Daily exports to Cloud Storage
- **GKE**: Velero for cluster backup
- **Cloud Storage**: Cross-region replication

#### **Recovery Plan**
- **RTO**: 2 hours for critical services
- **RPO**: 15 minutes for data loss
- Automated failover procedures
- Regular DR testing

### 11. Compliance and Governance

#### **Compliance Features**
- **Data Residency**: Regional restrictions
- **Access Transparency**: Audit Google's access
- **Data Loss Prevention**: Automatic PII detection
- **VPC Service Controls**: Data perimeter

#### **Organizational Policies**
- Resource location constraints
- Service usage restrictions
- IAM policy constraints
- Network security policies

## Implementation Structure

### File Organization
```
deploy/gcp/
├── __init__.py
├── provider.py        # Main GCP provider
├── cli.py            # CLI commands
├── templates.py      # IaC templates
├── config.py         # Configuration
└── tests.py          # Unit tests

inferloop-infra/gcp/
├── terraform/        # Terraform modules
├── kubernetes/       # K8s manifests
└── scripts/         # Deployment scripts
```

### Key Features Implemented
1. **Multi-Service Support**: Cloud Run, GKE, Cloud Functions
2. **Complete Storage**: GCS with lifecycle, Firestore, Cloud SQL
3. **Security**: IAM, Secret Manager, VPC controls
4. **Monitoring**: Full observability stack
5. **Cost Management**: Accurate estimation and optimization

## Best Practices

### 1. **Development**
- Use Cloud Shell for quick testing
- Leverage Cloud Code IDE extensions
- Implement structured logging
- Use Error Reporting

### 2. **Security**
- Enable Binary Authorization
- Use Workload Identity
- Implement least privilege
- Regular security scanning

### 3. **Operations**
- Define SLOs for all services
- Implement proper alerting
- Use Cloud Operations suite
- Automate everything

### 4. **Cost Management**
- Set up budget alerts
- Use committed use discounts
- Right-size resources
- Clean up unused resources

## Conclusion

The GCP infrastructure design provides a modern, serverless-first architecture that leverages Google's global network and managed services. It offers excellent scalability, security, and developer experience while maintaining cost efficiency. The implementation supports multiple deployment patterns and is optimized for synthetic data generation workloads with minimal operational overhead.