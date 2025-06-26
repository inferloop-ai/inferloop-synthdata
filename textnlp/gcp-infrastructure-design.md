# Google Cloud Platform Infrastructure Design for TextNLP Synthetic Data Platform

## Executive Summary

This document outlines the Google Cloud Platform (GCP) infrastructure design for deploying the TextNLP Synthetic Data Generation platform. The architecture leverages GCP's advanced AI/ML services, global infrastructure, and cloud-native technologies to deliver a scalable, performant, and cost-effective solution.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Cloud CDN + Cloud Armor                      │
│                  (Global Edge Caching + DDoS)                   │
└─────────────────────┬───────────────────────┬──────────────────┘
                      │                       │
┌─────────────────────▼───────────────────────▼──────────────────┐
│                  Global Load Balancer                           │
│                    (Multi-region LB)                            │
└─────────────────────┬───────────────────────┬──────────────────┘
                      │                       │
        ┌─────────────▼──────────┐ ┌─────────▼─────────────┐
        │     Cloud Endpoints    │ │    Apigee API        │
        │   (OpenAPI Gateway)    │ │   Management         │
        └─────────────┬──────────┘ └─────────┬─────────────┘
                      │                       │
┌─────────────────────▼───────────────────────▼──────────────────┐
│            Google Kubernetes Engine (GKE) Autopilot             │
│  ┌────────────────────────────────────────────────────────┐   │
│  │         API Workloads (Cloud Run for Anthos)           │   │
│  ├────────────────────────────────────────────────────────┤   │
│  │    Text Generation Workloads (GPU Node Pools)          │   │
│  ├────────────────────────────────────────────────────────┤   │
│  │      Validation Workloads (Spot Instances)             │   │
│  └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│              Vertex AI + Model Garden                           │
│         (PaLM API, Custom Models, AutoML)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                        Data Layer                               │
├─────────────────┬────────────────┬──────────────────────────────┤
│   Cloud SQL     │   Memorystore  │   Cloud Storage            │
│  (PostgreSQL)   │    (Redis)     │   (Multi-region)           │
└─────────────────┴────────────────┴──────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                    Event-Driven Services                        │
├──────────┬──────────┬──────────┬──────────┬───────────────────┤
│  Pub/Sub │Workflows │ Cloud    │Dataflow  │  Cloud           │
│          │          │Functions │          │  Composer        │
└──────────┴──────────┴──────────┴──────────┴───────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                 Observability & Security                        │
├──────────┬──────────┬──────────┬──────────┬───────────────────┤
│  Cloud   │  Cloud   │ Security │  Cloud   │  Binary          │
│ Logging  │  Trace   │ Command  │   IAM    │Authorization     │
└──────────┴──────────┴──────────┴──────────┴───────────────────┘
```

## Component Details

### 1. Network Architecture

#### VPC Design
```yaml
VPC Name: textnlp-production-vpc
Mode: Custom
Regions:
  - us-central1
  - us-east1
  - europe-west1

Subnets:
  - name: gke-nodes-subnet
    region: us-central1
    ip_range: 10.0.0.0/20
    secondary_ranges:
      - name: gke-pods
        ip_range: 10.4.0.0/14
      - name: gke-services
        ip_range: 10.8.0.0/20
        
  - name: private-services-subnet
    region: us-central1
    ip_range: 10.1.0.0/24
    private_google_access: true
    
  - name: serverless-connector-subnet
    region: us-central1
    ip_range: 10.2.0.0/28
    
  - name: sql-subnet
    region: us-central1
    ip_range: 10.3.0.0/24
```

#### Firewall Rules
```yaml
FirewallRules:
  - name: allow-health-checks
    direction: INGRESS
    priority: 1000
    sourceRanges: 
      - 35.191.0.0/16
      - 130.211.0.0/22
    allow:
      - protocol: tcp
        
  - name: allow-internal
    direction: INGRESS
    priority: 1100
    sourceRanges:
      - 10.0.0.0/8
    allow:
      - protocol: tcp
      - protocol: udp
      - protocol: icmp
      
  - name: deny-all-ingress
    direction: INGRESS
    priority: 65534
    action: DENY
    rules:
      - protocol: all
```

### 2. Container Orchestration - GKE

#### GKE Autopilot Configuration

```yaml
apiVersion: container.gke.io/v1
kind: Cluster
metadata:
  name: textnlp-autopilot
spec:
  location: us-central1
  autopilot:
    enabled: true
  network: textnlp-production-vpc
  subnetwork: gke-nodes-subnet
  clusterSecondaryRangeName: gke-pods
  servicesSecondaryRangeName: gke-services
  
  privateClusterConfig:
    enablePrivateNodes: true
    enablePrivateEndpoint: false
    masterIpv4CidrBlock: 172.16.0.0/28
    
  workloadIdentityConfig:
    workloadPool: textnlp-prod.svc.id.goog
    
  addonsConfig:
    httpLoadBalancing:
      disabled: false
    cloudRunConfig:
      disabled: false
    istioConfig:
      disabled: false
      auth: AUTH_MUTUAL_TLS
    
  verticalPodAutoscaling:
    enabled: true
    
  databaseEncryption:
    state: ENCRYPTED
    keyName: projects/textnlp-prod/locations/global/keyRings/textnlp-kr/cryptoKeys/gke-key
```

#### Workload Deployments

**API Service (Cloud Run for Anthos)**
```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: textnlp-api
  annotations:
    run.googleapis.com/cpu-throttling: "false"
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "3"
        autoscaling.knative.dev/maxScale: "1000"
        autoscaling.knative.dev/target: "100"
    spec:
      serviceAccountName: textnlp-api-sa
      containers:
      - image: gcr.io/textnlp-prod/textnlp-api:latest
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
        env:
        - name: PROJECT_ID
          value: textnlp-prod
        - name: ENVIRONMENT
          value: production
        livenessProbe:
          httpGet:
            path: /health
          initialDelaySeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
          initialDelaySeconds: 5
```

**GPU Generation Workload**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: textnlp-generation-gpu
spec:
  replicas: 2
  selector:
    matchLabels:
      app: textnlp-generation
  template:
    metadata:
      labels:
        app: textnlp-generation
    spec:
      serviceAccountName: textnlp-generation-sa
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
      containers:
      - name: generation
        image: gcr.io/textnlp-prod/textnlp-generation:latest
        resources:
          requests:
            nvidia.com/gpu: 1
            cpu: "4"
            memory: "16Gi"
          limits:
            nvidia.com/gpu: 1
            cpu: "8"
            memory: "32Gi"
        volumeMounts:
        - name: model-cache
          mountPath: /models
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
```

### 3. API Management

#### Cloud Endpoints Configuration

```yaml
swagger: "2.0"
info:
  title: "TextNLP API"
  version: "1.0.0"
host: "api.textnlp.io"
x-google-endpoints:
  - name: "api.textnlp.io"
    allowCors: true
schemes:
  - "https"
security:
  - api_key: []
  - firebase: []
paths:
  /v1/generate:
    post:
      summary: "Generate synthetic text"
      operationId: "generateText"
      x-google-backend:
        address: "https://textnlp-api-abcd1234-uc.a.run.app"
        protocol: "h2"
        path_translation: "APPEND_PATH_TO_ADDRESS"
      parameters:
        - name: body
          in: body
          required: true
          schema:
            $ref: "#/definitions/GenerateRequest"
      responses:
        200:
          description: "Success"
          schema:
            $ref: "#/definitions/GenerateResponse"
      x-google-quota:
        metricCosts:
          "generate-requests": 1
          "tokens-generated": 1000
```

#### Apigee Configuration

```javascript
// API Proxy Configuration
{
  "name": "textnlp-api-proxy",
  "basepaths": ["/api/v1"],
  "virtualHosts": ["secure"],
  "policies": [
    {
      "name": "OAuth-v2",
      "type": "OAuthV2",
      "configuration": {
        "Operation": "VerifyAccessToken",
        "Scope": "read write"
      }
    },
    {
      "name": "Spike-Arrest",
      "type": "SpikeArrest",
      "configuration": {
        "Rate": "1000pm",
        "Identifier": "client_id"
      }
    },
    {
      "name": "Response-Cache",
      "type": "ResponseCache",
      "configuration": {
        "CacheKey": {
          "KeyFragment": ["request.uri", "request.queryparam.model"]
        },
        "ExpirySettings": {
          "TimeoutInSec": 300
        }
      }
    }
  ]
}
```

### 4. AI/ML Infrastructure

#### Vertex AI Configuration

**Model Endpoints**
```python
from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(project='textnlp-prod', location='us-central1')

# Deploy custom model
model = aiplatform.Model(
    display_name="textnlp-custom-llm",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-12:latest",
    serving_container_environment_variables={
        "MODEL_PATH": "gs://textnlp-models/custom-llm"
    }
)

endpoint = model.deploy(
    machine_type="n1-standard-16",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    min_replica_count=1,
    max_replica_count=10,
    traffic_percentage=100,
    deployed_model_display_name="textnlp-custom-llm-v1"
)
```

**PaLM API Integration**
```python
import vertexai
from vertexai.language_models import TextGenerationModel

vertexai.init(project="textnlp-prod", location="us-central1")

parameters = {
    "temperature": 0.7,
    "max_output_tokens": 2048,
    "top_p": 0.8,
    "top_k": 40
}

model = TextGenerationModel.from_pretrained("text-bison@001")

# Batch prediction job
batch_prediction_job = model.batch_predict(
    source_uri="gs://textnlp-data/prompts/batch_input.jsonl",
    destination_uri_prefix="gs://textnlp-data/predictions",
    model_parameters=parameters,
)
```

**Model Garden Access**
```yaml
AvailableModels:
  - model_id: gemini-pro
    endpoint_type: streaming
    quota_limit: 1000_requests_per_minute
    
  - model_id: claude-2-vertex
    endpoint_type: batch
    quota_limit: 500_requests_per_minute
    
  - model_id: llama2-70b
    endpoint_type: custom
    deployment_type: dedicated
    machine_type: a2-ultragpu-1g
```

### 5. Data Storage Architecture

#### Cloud Storage Configuration

```yaml
Buckets:
  - name: textnlp-data-prod
    location: US
    storageClass: STANDARD
    uniformBucketLevelAccess: true
    versioning: true
    lifecycleRules:
      - action:
          type: SetStorageClass
          storageClass: NEARLINE
        condition:
          age: 30
      - action:
          type: SetStorageClass
          storageClass: COLDLINE
        condition:
          age: 90
      - action:
          type: Delete
        condition:
          age: 365
          
  - name: textnlp-models-prod
    location: US-CENTRAL1
    storageClass: STANDARD
    versioning: true
    
  - name: textnlp-backups-prod
    location: NAM4
    storageClass: COLDLINE
    retentionPolicy:
      retentionPeriod: 2592000  # 30 days
```

**Object Organization**
```
textnlp-data-prod/
├── prompts/
│   ├── templates/
│   │   └── {version}/
│   ├── user-generated/
│   │   └── {user_id}/{timestamp}/
│   └── system/
├── generations/
│   ├── real-time/
│   │   └── {date}/
│   ├── batch/
│   │   └── {job_id}/
│   └── archive/
├── validation/
│   ├── metrics/
│   │   └── {date}/
│   ├── human-eval/
│   └── reports/
└── exports/
    └── {format}/
```

#### Cloud SQL Configuration

```yaml
Instance:
  name: textnlp-postgres-prod
  databaseVersion: POSTGRES_15
  region: us-central1
  tier: db-custom-8-32768
  
  settings:
    availabilityType: REGIONAL
    backupConfiguration:
      enabled: true
      startTime: "02:00"
      location: us
      pointInTimeRecoveryEnabled: true
      transactionLogRetentionDays: 7
    ipConfiguration:
      privateNetwork: projects/textnlp-prod/global/networks/textnlp-vpc
      requireSsl: true
    databaseFlags:
      - name: max_connections
        value: "1000"
      - name: shared_buffers
        value: "8GB"
    insightsConfig:
      queryInsightsEnabled: true
      queryPlansPerMinute: 10
      recordApplicationTags: true
      
  replicaConfiguration:
    - name: textnlp-postgres-replica-1
      region: us-east1
      tier: db-custom-4-16384
    - name: textnlp-postgres-replica-2
      region: europe-west1
      tier: db-custom-4-16384
```

### 6. Event-Driven Architecture

#### Pub/Sub Configuration

```yaml
Topics:
  - name: text-generation-requests
    messageRetentionDuration: 604800s  # 7 days
    schema:
      name: generation-request-schema
      type: AVRO
      definition: |
        {
          "type": "record",
          "name": "GenerationRequest",
          "fields": [
            {"name": "id", "type": "string"},
            {"name": "prompt", "type": "string"},
            {"name": "model", "type": "string"},
            {"name": "parameters", "type": "map", "values": "string"}
          ]
        }
        
Subscriptions:
  - name: generation-processor
    topic: text-generation-requests
    ackDeadlineSeconds: 600
    messageRetentionDuration: 604800s
    retryPolicy:
      minimumBackoff: 10s
      maximumBackoff: 600s
    deadLetterPolicy:
      deadLetterTopic: text-generation-dlq
      maxDeliveryAttempts: 5
```

#### Cloud Workflows

```yaml
main:
  params: [args]
  steps:
    - validateInput:
        call: sys.log
        args:
          text: ${"Processing batch job: " + args.batchId}
          
    - getPrompts:
        call: http.get
        args:
          url: ${"https://storage.googleapis.com/textnlp-data-prod/batch/" + args.batchId + "/prompts.json"}
          auth:
            type: OAuth2
        result: promptsData
        
    - processPrompts:
        parallel:
          for:
            value: prompt
            in: ${promptsData.body.prompts}
            steps:
              - generateText:
                  call: http.post
                  args:
                    url: https://api.textnlp.io/v1/generate
                    auth:
                      type: OAuth2
                    body:
                      prompt: ${prompt.text}
                      model: ${prompt.model}
                      parameters: ${prompt.parameters}
                  result: generationResult
                  
              - storeResult:
                  call: googleapis.storage.v1.objects.insert
                  args:
                    bucket: textnlp-data-prod
                    name: ${"generations/batch/" + args.batchId + "/" + prompt.id + ".json"}
                    body: ${generationResult.body}
                    
    - notifyCompletion:
        call: googleapis.pubsub.v1.projects.topics.publish
        args:
          topic: projects/textnlp-prod/topics/batch-completions
          body:
            messages:
              - data: ${base64.encode(json.encode({"batchId": args.batchId, "status": "completed"}))}
```

### 7. Serverless Components

#### Cloud Functions

```python
import functions_framework
from google.cloud import storage, pubsub_v1, firestore

@functions_framework.http
def authenticate_request(request):
    """HTTP Cloud Function for JWT validation"""
    import jwt
    from google.auth import compute_engine
    
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    
    try:
        # Verify JWT token
        payload = jwt.decode(token, get_public_key(), algorithms=['RS256'])
        
        # Check permissions in Firestore
        db = firestore.Client()
        user_doc = db.collection('users').document(payload['sub']).get()
        
        if user_doc.exists and user_doc.to_dict().get('active'):
            return {'authorized': True, 'user_id': payload['sub']}
        else:
            return {'authorized': False}, 403
            
    except Exception as e:
        return {'error': str(e)}, 401

@functions_framework.cloud_event
def process_generation_complete(cloud_event):
    """Triggered when generation completes"""
    import base64
    import json
    
    # Parse Pub/Sub message
    message = base64.b64decode(cloud_event.data["message"]["data"]).decode()
    generation_data = json.loads(message)
    
    # Trigger validation workflow
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path('textnlp-prod', 'validation-requests')
    
    validation_message = {
        'generation_id': generation_data['id'],
        'text': generation_data['generated_text'],
        'model': generation_data['model']
    }
    
    publisher.publish(
        topic_path,
        data=json.dumps(validation_message).encode('utf-8')
    )

@functions_framework.http
def batch_import_handler(request):
    """Handle batch import via HTTP trigger"""
    file_path = request.json.get('file_path')
    
    # Validate file exists
    storage_client = storage.Client()
    bucket_name, blob_name = file_path.replace('gs://', '').split('/', 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    if not blob.exists():
        return {'error': 'File not found'}, 404
        
    # Create Dataflow job for processing
    from googleapiclient.discovery import build
    
    dataflow = build('dataflow', 'v1b3')
    
    job_config = {
        'jobName': f'batch-import-{request.headers.get("X-Request-ID")}',
        'parameters': {
            'inputFile': file_path,
            'outputTopic': 'projects/textnlp-prod/topics/text-generation-requests'
        },
        'environment': {
            'tempLocation': 'gs://textnlp-temp/dataflow',
            'zone': 'us-central1-a'
        }
    }
    
    response = dataflow.projects().templates().launch(
        projectId='textnlp-prod',
        gcsPath='gs://textnlp-dataflow-templates/batch-import',
        body=job_config
    ).execute()
    
    return {'job_id': response['job']['id']}
```

#### Cloud Run Jobs

```yaml
apiVersion: run.googleapis.com/v1
kind: Job
metadata:
  name: daily-metrics-aggregation
spec:
  template:
    spec:
      parallelism: 10
      taskCount: 100
      template:
        spec:
          containers:
          - image: gcr.io/textnlp-prod/metrics-aggregator:latest
            resources:
              limits:
                cpu: "4"
                memory: "8Gi"
            env:
            - name: DATE
              value: "{{ .Date }}"
            - name: BUCKET
              value: "textnlp-data-prod"
          timeoutSeconds: 3600
          serviceAccountName: metrics-aggregator-sa
```

### 8. Security Implementation

#### Identity and Access Management (IAM)

```yaml
ServiceAccounts:
  - email: textnlp-api-sa@textnlp-prod.iam.gserviceaccount.com
    displayName: TextNLP API Service Account
    roles:
      - roles/cloudtrace.agent
      - roles/monitoring.metricWriter
      - roles/logging.logWriter
      - roles/cloudsql.client
      - roles/redis.editor
      - roles/storage.objectViewer
      - roles/pubsub.publisher
      - roles/secretmanager.secretAccessor
      
  - email: textnlp-generation-sa@textnlp-prod.iam.gserviceaccount.com
    displayName: Text Generation Service Account
    roles:
      - roles/aiplatform.user
      - roles/storage.objectAdmin
      - roles/pubsub.subscriber
      - roles/pubsub.publisher

CustomRoles:
  - name: textnlpModelInvoker
    title: TextNLP Model Invoker
    permissions:
      - aiplatform.endpoints.predict
      - aiplatform.models.get
      - storage.objects.get
      - storage.objects.create
```

#### Secret Manager

```yaml
Secrets:
  - name: db-password
    replication:
      automatic: {}
    annotations:
      environment: production
      service: postgresql
      
  - name: jwt-signing-key
    replication:
      userManaged:
        replicas:
          - location: us-central1
          - location: us-east1
          - location: europe-west1
    rotation:
      nextRotationTime: "2024-01-01T00:00:00Z"
      rotationPeriod: "2592000s"  # 30 days
      
  - name: api-keys
    replication:
      automatic: {}
    versionAliases:
      latest: 1
```

#### Binary Authorization

```yaml
apiVersion: binaryauthorization.grafeas.io/v1beta1
kind: Policy
metadata:
  name: textnlp-binary-auth-policy
spec:
  globalPolicyEvaluationMode: ENABLE
  defaultAdmissionRule:
    evaluationMode: REQUIRE_ATTESTATION
    enforcementMode: ENFORCED_BLOCK_AND_AUDIT_LOG
    requireAttestationsBy:
      - projects/textnlp-prod/attestors/prod-attestor
  clusterAdmissionRules:
    us-central1.textnlp-autopilot:
      evaluationMode: REQUIRE_ATTESTATION
      enforcementMode: ENFORCED_BLOCK_AND_AUDIT_LOG
      requireAttestationsBy:
        - projects/textnlp-prod/attestors/prod-attestor
```

### 9. Monitoring and Observability

#### Cloud Monitoring Configuration

**Custom Metrics**
```python
from google.cloud import monitoring_v3
import time

client = monitoring_v3.MetricServiceClient()
project_name = f"projects/textnlp-prod"

# Define custom metric
descriptor = monitoring_v3.MetricDescriptor(
    type="custom.googleapis.com/textnlp/tokens_generated",
    metric_kind=monitoring_v3.MetricDescriptor.MetricKind.GAUGE,
    value_type=monitoring_v3.MetricDescriptor.ValueType.INT64,
    description="Number of tokens generated",
    display_name="Tokens Generated",
    labels=[
        monitoring_v3.LabelDescriptor(
            key="model",
            value_type=monitoring_v3.LabelDescriptor.ValueType.STRING,
        ),
        monitoring_v3.LabelDescriptor(
            key="user_id",
            value_type=monitoring_v3.LabelDescriptor.ValueType.STRING,
        ),
    ],
)

# Write metric data
series = monitoring_v3.TimeSeries()
series.metric.type = "custom.googleapis.com/textnlp/tokens_generated"
series.metric.labels["model"] = "gpt-j-6b"
series.metric.labels["user_id"] = "user123"

now = time.time()
point = monitoring_v3.Point(
    {"interval": {"end_time": {"seconds": int(now)}}, "value": {"int64_value": 1024}}
)
series.points = [point]

client.create_time_series(name=project_name, time_series=[series])
```

**Alerting Policies**
```yaml
AlertingPolicies:
  - displayName: High API Latency
    conditions:
      - displayName: API latency above 1s
        conditionThreshold:
          filter: |
            resource.type = "k8s_container"
            AND metric.type = "kubernetes.io/container/request_latency"
            AND metadata.user_labels."app" = "textnlp-api"
          comparison: COMPARISON_GT
          thresholdValue: 1000
          duration: 300s
          aggregations:
            - alignmentPeriod: 60s
              perSeriesAligner: ALIGN_PERCENTILE_95
    notificationChannels:
      - projects/textnlp-prod/notificationChannels/email-oncall
      - projects/textnlp-prod/notificationChannels/pagerduty
      
  - displayName: GPU Utilization Low
    conditions:
      - displayName: GPU usage below 50%
        conditionThreshold:
          filter: |
            resource.type = "k8s_node"
            AND metric.type = "kubernetes.io/node/accelerator/gpu_utilization"
          comparison: COMPARISON_LT
          thresholdValue: 0.5
          duration: 1800s
```

**Dashboards**
```json
{
  "displayName": "TextNLP Production Dashboard",
  "gridLayout": {
    "widgets": [
      {
        "title": "Request Rate",
        "xyChart": {
          "timeSeries": [{
            "filter": "resource.type=\"k8s_container\" metric.type=\"kubernetes.io/container/request_count\"",
            "aggregation": {
              "alignmentPeriod": "60s",
              "perSeriesAligner": "ALIGN_RATE"
            }
          }]
        }
      },
      {
        "title": "Token Generation Rate",
        "xyChart": {
          "timeSeries": [{
            "filter": "metric.type=\"custom.googleapis.com/textnlp/tokens_generated\"",
            "aggregation": {
              "alignmentPeriod": "300s",
              "perSeriesAligner": "ALIGN_SUM",
              "crossSeriesReducer": "REDUCE_SUM",
              "groupByFields": ["metric.label.model"]
            }
          }]
        }
      },
      {
        "title": "Model Inference Latency",
        "xyChart": {
          "timeSeries": [{
            "filter": "resource.type=\"aiplatform.googleapis.com/Endpoint\" metric.type=\"aiplatform.googleapis.com/prediction/latencies\"",
            "aggregation": {
              "alignmentPeriod": "60s",
              "perSeriesAligner": "ALIGN_PERCENTILE_50"
            }
          }]
        }
      }
    ]
  }
}
```

### 10. Disaster Recovery

#### Backup Strategy

**Automated Backups**
```yaml
BackupPlan:
  name: textnlp-backup-plan
  retentionDays: 30
  backupVault: projects/textnlp-prod/locations/us/backupVaults/prod-vault
  
  backupPolicies:
    - resourceType: COMPUTE_ENGINE_DISK
      schedule: "0 2 * * *"  # Daily at 2 AM
      
    - resourceType: CLOUD_SQL_DATABASE
      schedule: "0 */6 * * *"  # Every 6 hours
      
    - resourceType: GCS_BUCKET
      buckets:
        - textnlp-data-prod
        - textnlp-models-prod
      schedule: "0 0 * * 0"  # Weekly
```

**Cross-Region Replication**
```yaml
ReplicationConfig:
  primaryRegion: us-central1
  secondaryRegions:
    - us-east1
    - europe-west1
    
  services:
    CloudSQL:
      replicationType: ASYNC
      failoverTarget: us-east1
      
    CloudStorage:
      replicationType: DUAL_REGION
      turboReplication: true
      
    GKE:
      multiClusterIngress: true
      configSync: enabled
```

### 11. Cost Optimization

#### Resource Optimization

**Committed Use Discounts**
```yaml
CommitmentConfig:
  - resourceType: CPU
    region: us-central1
    commitment: 1000 vCPUs
    term: 3 years
    
  - resourceType: Memory
    region: us-central1
    commitment: 4000 GB
    term: 3 years
    
  - resourceType: Local SSD
    region: us-central1
    commitment: 100 TB
    term: 1 year
```

**Spot VM Usage**
```yaml
SpotVMConfig:
  nodePool: validation-pool
  spotPercentage: 80
  onDemandBaseCapacity: 2
  maxPrice: 0.50  # 50% of on-demand price
  terminationHandler: enabled
```

### 12. CI/CD Pipeline

#### Cloud Build Configuration

```yaml
steps:
  # Build API image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/textnlp-api:$SHORT_SHA', './api']
    
  # Run tests
  - name: 'gcr.io/$PROJECT_ID/textnlp-api:$SHORT_SHA'
    args: ['pytest', '-v', '--cov=app']
    
  # Security scanning
  - name: 'gcr.io/cloud-builders/gcloud'
    args: ['container', 'images', 'scan', 'gcr.io/$PROJECT_ID/textnlp-api:$SHORT_SHA']
    
  # Push to registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/textnlp-api:$SHORT_SHA']
    
  # Deploy to GKE
  - name: 'gcr.io/cloud-builders/kubectl'
    args:
      - 'set'
      - 'image'
      - 'deployment/textnlp-api'
      - 'api=gcr.io/$PROJECT_ID/textnlp-api:$SHORT_SHA'
      - '--namespace=production'
    env:
      - 'CLOUDSDK_COMPUTE_ZONE=us-central1'
      - 'CLOUDSDK_CONTAINER_CLUSTER=textnlp-autopilot'
      
  # Create attestation
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'beta'
      - 'container'
      - 'binauthz'
      - 'attestations'
      - 'sign-and-create'
      - '--artifact-url=gcr.io/$PROJECT_ID/textnlp-api:$SHORT_SHA'
      - '--attestor=prod-attestor'
      - '--attestor-project=$PROJECT_ID'

options:
  logging: CLOUD_LOGGING_ONLY
  machineType: 'E2_HIGHCPU_8'
  
timeout: '1200s'

triggers:
  - github:
      owner: 'textnlp'
      name: 'textnlp-platform'
      push:
        branch: '^main$'
```

## Implementation Timeline

### Phase 1: Foundation (Week 1-2)
- Project setup and IAM configuration
- VPC and networking setup
- GKE Autopilot cluster creation
- Cloud SQL and Memorystore provisioning

### Phase 2: Core Services (Week 3-4)
- Container registry setup
- Initial deployments to GKE
- Cloud Endpoints configuration
- Basic monitoring setup

### Phase 3: AI Integration (Week 5-6)
- Vertex AI setup and model deployment
- PaLM API integration
- Custom model endpoints
- Batch prediction pipelines

### Phase 4: Production Hardening (Week 7-8)
- Binary authorization implementation
- Advanced monitoring and alerting
- Disaster recovery testing
- Performance optimization

## Conclusion

This GCP infrastructure design leverages Google Cloud's advanced AI/ML capabilities, global infrastructure, and cloud-native services to deliver a robust, scalable platform for text generation. The architecture emphasizes automation, security, and cost optimization while providing the flexibility to integrate cutting-edge language models and scale globally.