# Azure Infrastructure Design for TextNLP Synthetic Data Platform

## Executive Summary

This document details the Azure-specific infrastructure design for deploying the TextNLP Synthetic Data Generation platform. The architecture leverages Azure's managed services and AI capabilities to deliver a scalable, secure, and enterprise-ready solution optimized for text generation workloads.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Azure Front Door                            │
│                  (Global Load Balancer + WAF)                   │
└─────────────────────┬───────────────────────┬──────────────────┘
                      │                       │
┌─────────────────────▼───────────────────────▼──────────────────┐
│                  Azure Traffic Manager                          │
│                    (DNS-based routing)                          │
└─────────────────────┬───────────────────────┬──────────────────┘
                      │                       │
┌─────────────────────▼───────────────────────▼──────────────────┐
│          Application Gateway (Regional LB + WAF)                │
└─────────────────────┬───────────────────────┬──────────────────┘
                      │                       │
        ┌─────────────▼──────────┐ ┌─────────▼─────────────┐
        │   API Management       │ │  Event Grid           │
        │   (API Gateway)        │ │  (Event Routing)      │
        └─────────────┬──────────┘ └─────────┬─────────────┘
                      │                       │
┌─────────────────────▼───────────────────────▼──────────────────┐
│              Azure Kubernetes Service (AKS)                     │
│  ┌────────────────────────────────────────────────────────┐   │
│  │          API Pods (Horizontal Pod Autoscaler)          │   │
│  ├────────────────────────────────────────────────────────┤   │
│  │    Text Generation Pods (GPU Node Pool)                │   │
│  ├────────────────────────────────────────────────────────┤   │
│  │      Validation Pods (CPU Node Pool)                   │   │
│  └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│           Azure OpenAI Service / Azure ML                       │
│        (Managed Model Endpoints & Fine-tuning)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                        Data Layer                               │
├─────────────────┬────────────────┬──────────────────────────────┤
│ Azure Database  │  Azure Cache   │   Azure Storage             │
│ for PostgreSQL  │   for Redis    │   (Blob + Files)           │
└─────────────────┴────────────────┴──────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                    Supporting Services                          │
├──────────┬──────────┬──────────┬──────────┬───────────────────┤
│ Service  │  Logic   │ Function │  Batch   │ Container        │
│   Bus    │  Apps    │   Apps   │ Account  │ Instances        │
└──────────┴──────────┴──────────┴──────────┴───────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                 Monitoring & Security                           │
├──────────┬──────────┬──────────┬──────────┬───────────────────┤
│  Azure   │  App     │ Security │ Sentinel │  Key Vault       │
│ Monitor  │ Insights │  Center  │          │                  │
└──────────┴──────────┴──────────┴──────────┴───────────────────┘
```

## Component Details

### 1. Network Architecture

#### Virtual Network Design
```
TextNLP-Production-VNet (10.0.0.0/16)
├── Gateway Subnet (10.0.0.0/27)
│   └── Application Gateway, VPN Gateway
├── AKS System Subnet (10.0.1.0/24)
│   └── AKS System Node Pool
├── AKS User Subnet (10.0.2.0/23)
│   └── AKS User Node Pools (CPU + GPU)
├── Private Endpoint Subnet (10.0.4.0/24)
│   └── Private Endpoints for PaaS Services
├── Integration Subnet (10.0.5.0/24)
│   └── Logic Apps, Function Apps
└── Database Subnet (10.0.6.0/24)
    └── PostgreSQL Flexible Server
```

#### Network Security Groups (NSGs)

**AKS-NSG Rules**
```yaml
Inbound:
  - Priority: 100
    Name: AllowApplicationGateway
    Source: GatewaySubnet
    Destination: AKSSubnet
    Port: 80,443
    Protocol: TCP
    Action: Allow
    
  - Priority: 200
    Name: AllowAzureLoadBalancer
    Source: AzureLoadBalancer
    Destination: Any
    Port: Any
    Protocol: Any
    Action: Allow

Outbound:
  - Priority: 100
    Name: AllowInternetOutbound
    Source: Any
    Destination: Internet
    Port: 443
    Protocol: TCP
    Action: Allow
```

### 2. Container Orchestration - AKS

#### AKS Cluster Configuration

```yaml
apiVersion: 2023-05-01
name: textnlp-aks-production
location: eastus
properties:
  kubernetesVersion: "1.28"
  dnsPrefix: textnlp-prod
  
  agentPoolProfiles:
    # System Node Pool
    - name: system
      count: 3
      vmSize: Standard_D4s_v5
      osType: Linux
      mode: System
      availabilityZones: [1, 2, 3]
      enableAutoScaling: true
      minCount: 3
      maxCount: 10
      
    # CPU Node Pool for API/Validation
    - name: cpu
      count: 3
      vmSize: Standard_D8s_v5
      osType: Linux
      mode: User
      availabilityZones: [1, 2, 3]
      enableAutoScaling: true
      minCount: 3
      maxCount: 50
      nodeLabels:
        workload: cpu
        
    # GPU Node Pool for Generation
    - name: gpu
      count: 2
      vmSize: Standard_NC6s_v3
      osType: Linux
      mode: User
      availabilityZones: [1, 2, 3]
      enableAutoScaling: true
      minCount: 1
      maxCount: 10
      nodeLabels:
        workload: gpu
      nodeTaints:
        - key: gpu
          value: "true"
          effect: NoSchedule
          
  networkProfile:
    networkPlugin: azure
    networkPolicy: calico
    loadBalancerSku: standard
    serviceCidr: 10.100.0.0/16
    dnsServiceIP: 10.100.0.10
    
  addonProfiles:
    azureKeyvaultSecretsProvider:
      enabled: true
    azurepolicy:
      enabled: true
    httpApplicationRouting:
      enabled: false
    omsagent:
      enabled: true
      config:
        logAnalyticsWorkspaceResourceID: /subscriptions/.../workspaces/textnlp-logs
```

#### Kubernetes Deployments

**API Deployment**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: textnlp-api
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: textnlp-api
  template:
    metadata:
      labels:
        app: textnlp-api
        azure.workload.identity/use: "true"
    spec:
      serviceAccountName: textnlp-api-sa
      nodeSelector:
        workload: cpu
      containers:
      - name: api
        image: textnlpacr.azurecr.io/textnlp-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: AZURE_CLIENT_ID
          value: "workload-identity-client-id"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
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

**GPU Generation Service**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: textnlp-generation
  namespace: production
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
      nodeSelector:
        workload: gpu
      tolerations:
      - key: gpu
        operator: Equal
        value: "true"
        effect: NoSchedule
      containers:
      - name: generation
        image: textnlpacr.azurecr.io/textnlp-generation:latest
        resources:
          requests:
            memory: "16Gi"
            cpu: "4000m"
            nvidia.com/gpu: 1
          limits:
            memory: "32Gi"
            cpu: "8000m"
            nvidia.com/gpu: 1
```

### 3. API Management

#### API Management Configuration

```json
{
  "name": "textnlp-apim",
  "location": "eastus",
  "sku": {
    "name": "Premium",
    "capacity": 2
  },
  "properties": {
    "publisherEmail": "admin@textnlp.com",
    "publisherName": "TextNLP Platform",
    "virtualNetworkType": "Internal",
    "virtualNetworkConfiguration": {
      "subnetResourceId": "/subscriptions/.../subnets/apim-subnet"
    },
    "customProperties": {
      "Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Protocols.Tls10": "false",
      "Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Protocols.Tls11": "false",
      "Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Protocols.Ssl30": "false"
    }
  }
}
```

**API Policies**
```xml
<policies>
  <inbound>
    <base />
    <rate-limit calls="1000" renewal-period="60" />
    <quota calls="1000000" renewal-period="2592000" />
    <validate-jwt header-name="Authorization" failed-validation-httpcode="401">
      <openid-config url="https://login.microsoftonline.com/{tenant}/v2.0/.well-known/openid-configuration" />
      <audiences>
        <audience>api://textnlp</audience>
      </audiences>
    </validate-jwt>
    <cors allow-credentials="true">
      <allowed-origins>
        <origin>https://app.textnlp.com</origin>
      </allowed-origins>
      <allowed-methods>
        <method>GET</method>
        <method>POST</method>
        <method>PUT</method>
        <method>DELETE</method>
      </allowed-methods>
    </cors>
  </inbound>
  <backend>
    <base />
  </backend>
  <outbound>
    <base />
    <set-header name="X-Content-Type-Options" exists-action="override">
      <value>nosniff</value>
    </set-header>
  </outbound>
  <on-error>
    <base />
  </on-error>
</policies>
```

### 4. AI/ML Infrastructure

#### Azure OpenAI Service

**Deployment Configuration**
```json
{
  "deployments": [
    {
      "name": "gpt-4-turbo",
      "model": "gpt-4",
      "version": "turbo-2024-04-09",
      "capacity": 100,
      "raiPolicy": "default"
    },
    {
      "name": "gpt-35-turbo",
      "model": "gpt-35-turbo",
      "version": "0613",
      "capacity": 200,
      "raiPolicy": "default"
    }
  ],
  "networkAcls": {
    "defaultAction": "Deny",
    "virtualNetworkRules": [
      {
        "id": "/subscriptions/.../subnets/aks-subnet"
      }
    ],
    "ipRules": []
  }
}
```

#### Azure Machine Learning

**Compute Clusters**
```python
from azureml.core.compute import AmlCompute, ComputeTarget

# GPU cluster for model training
gpu_config = AmlCompute.provisioning_configuration(
    vm_size="Standard_NC6s_v3",
    min_nodes=0,
    max_nodes=10,
    idle_seconds_before_scaledown=300,
    vnet_resourcegroup_name="textnlp-rg",
    vnet_name="textnlp-vnet",
    subnet_name="aml-subnet"
)

# CPU cluster for batch inference
cpu_config = AmlCompute.provisioning_configuration(
    vm_size="Standard_D16s_v5",
    min_nodes=1,
    max_nodes=20,
    idle_seconds_before_scaledown=1800
)
```

**Model Endpoints**
```python
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AksWebservice

inference_config = InferenceConfig(
    runtime="python",
    entry_script="score.py",
    conda_file="environment.yml"
)

deployment_config = AksWebservice.deploy_configuration(
    cpu_cores=4,
    memory_gb=32,
    enable_app_insights=True,
    collect_model_data=True,
    auth_enabled=True,
    gpu_cores=1
)
```

### 5. Data Storage Architecture

#### Storage Account Configuration

```json
{
  "name": "textnlpprodstorage",
  "location": "eastus",
  "sku": {
    "name": "Standard_ZRS"
  },
  "kind": "StorageV2",
  "properties": {
    "accessTier": "Hot",
    "supportsHttpsTrafficOnly": true,
    "minimumTlsVersion": "TLS1_2",
    "allowBlobPublicAccess": false,
    "networkAcls": {
      "defaultAction": "Deny",
      "virtualNetworkRules": [
        {
          "id": "/subscriptions/.../subnets/aks-subnet",
          "action": "Allow"
        }
      ]
    }
  }
}
```

**Container Structure**
```
textnlp-storage/
├── prompts/
│   ├── templates/
│   ├── user-content/
│   └── system/
├── generations/
│   ├── realtime/
│   ├── batch/
│   └── archive/
├── models/
│   ├── checkpoints/
│   ├── fine-tuned/
│   └── configs/
├── validation/
│   ├── metrics/
│   ├── human-eval/
│   └── reports/
└── backups/
    ├── database/
    └── configurations/
```

**Lifecycle Management Policies**
```json
{
  "rules": [
    {
      "name": "archiveOldGenerations",
      "enabled": true,
      "type": "Lifecycle",
      "definition": {
        "filters": {
          "blobTypes": ["blockBlob"],
          "prefixMatch": ["generations/realtime"]
        },
        "actions": {
          "baseBlob": {
            "tierToCool": {
              "daysAfterModificationGreaterThan": 30
            },
            "tierToArchive": {
              "daysAfterModificationGreaterThan": 90
            },
            "delete": {
              "daysAfterModificationGreaterThan": 365
            }
          }
        }
      }
    }
  ]
}
```

#### Azure Database for PostgreSQL

**Flexible Server Configuration**
```yaml
ServerName: textnlp-postgres-prod
Version: "15"
Tier: GeneralPurpose
SkuName: Standard_D8ds_v4
StorageSizeGB: 512
BackupRetentionDays: 35
GeoRedundantBackup: Enabled
HighAvailability:
  Mode: ZoneRedundant
  StandbyAvailabilityZone: "2"

Configuration:
  max_connections: 500
  shared_buffers: 256MB
  effective_cache_size: 24GB
  work_mem: 16MB
  maintenance_work_mem: 512MB
  
SecurityConfiguration:
  - SSL: Enforced
  - MinimalTlsVersion: TLS1.2
  - PublicNetworkAccess: Disabled
  - PrivateEndpoint: Enabled
```

### 6. Message Queue and Event Processing

#### Service Bus Configuration

```json
{
  "name": "textnlp-servicebus",
  "location": "eastus",
  "sku": {
    "name": "Premium",
    "tier": "Premium",
    "capacity": 1
  },
  "properties": {
    "zoneRedundant": true,
    "encryption": {
      "keySource": "Microsoft.KeyVault",
      "keyVaultProperties": [{
        "keyName": "servicebus-key",
        "keyVaultUri": "https://textnlp-kv.vault.azure.net/"
      }]
    }
  }
}
```

**Queue Definitions**
```yaml
Queues:
  - Name: text-generation-queue
    Properties:
      MaxSizeInMegabytes: 5120
      DefaultMessageTimeToLive: P14D
      LockDuration: PT5M
      MaxDeliveryCount: 3
      EnablePartitioning: true
      
  - Name: batch-processing-queue
    Properties:
      MaxSizeInMegabytes: 10240
      DefaultMessageTimeToLive: P7D
      LockDuration: PT30M
      MaxDeliveryCount: 5
      EnableBatchedOperations: true
      
Topics:
  - Name: generation-events
    Subscriptions:
      - Name: validation-sub
        Rules:
          - Name: completed-only
            Filter: "Status = 'Completed'"
      - Name: monitoring-sub
        Rules:
          - Name: all-events
```

#### Event Grid Configuration

```json
{
  "topics": [
    {
      "name": "textnlp-events",
      "location": "eastus",
      "properties": {
        "inputSchema": "EventGridSchema",
        "publicNetworkAccess": "Disabled"
      }
    }
  ],
  "eventSubscriptions": [
    {
      "name": "generation-completed",
      "properties": {
        "destination": {
          "endpointType": "ServiceBusQueue",
          "properties": {
            "resourceId": "/subscriptions/.../queues/validation-queue"
          }
        },
        "filter": {
          "includedEventTypes": ["TextGeneration.Completed"]
        }
      }
    }
  ]
}
```

### 7. Serverless Components

#### Azure Functions

**Function App Configuration**
```json
{
  "name": "textnlp-functions",
  "location": "eastus",
  "kind": "functionapp,linux",
  "properties": {
    "serverFarmId": "textnlp-asp",
    "siteConfig": {
      "linuxFxVersion": "PYTHON|3.11",
      "appSettings": [
        {
          "name": "AzureWebJobsStorage",
          "value": "@Microsoft.KeyVault(SecretUri=...)"
        },
        {
          "name": "FUNCTIONS_WORKER_RUNTIME",
          "value": "python"
        }
      ],
      "ftpsState": "Disabled",
      "minTlsVersion": "1.2"
    },
    "httpsOnly": true
  }
}
```

**Key Functions**
```python
# Batch Processing Trigger
@app.blob_trigger(arg_name="myblob", 
                  path="prompts/batch/{name}",
                  connection="AzureWebJobsStorage")
@app.service_bus_queue_output(arg_name="msg",
                              queue_name="batch-processing-queue",
                              connection="ServiceBusConnection")
def batch_trigger(myblob: bytes, msg: func.Out[str]):
    # Parse batch file and queue individual jobs
    batch_data = json.loads(myblob)
    for item in batch_data['prompts']:
        msg.set(json.dumps({
            'prompt': item,
            'batch_id': batch_data['id'],
            'parameters': batch_data.get('parameters', {})
        }))

# Token Usage Calculator
@app.timer_trigger(schedule="0 */5 * * * *", 
                   arg_name="mytimer")
def calculate_usage(mytimer: func.TimerRequest):
    # Aggregate token usage and update metrics
    usage_data = aggregate_token_usage()
    send_to_monitoring(usage_data)
```

#### Logic Apps

**Workflow Orchestration**
```json
{
  "definition": {
    "$schema": "https://schema.management.azure.com/providers/Microsoft.Logic/schemas/2016-06-01/workflowdefinition.json#",
    "triggers": {
      "manual": {
        "type": "Request",
        "kind": "Http"
      }
    },
    "actions": {
      "Parse_Request": {
        "type": "ParseJson",
        "inputs": {
          "content": "@triggerBody()",
          "schema": {}
        }
      },
      "Generate_Text": {
        "type": "Http",
        "inputs": {
          "method": "POST",
          "uri": "https://textnlp-aks.eastus.cloudapp.azure.com/api/v1/generate",
          "authentication": {
            "type": "ManagedServiceIdentity"
          }
        }
      },
      "Validate_Output": {
        "type": "Http",
        "inputs": {
          "method": "POST",
          "uri": "https://textnlp-aks.eastus.cloudapp.azure.com/api/v1/validate"
        }
      },
      "Store_Result": {
        "type": "ApiConnection",
        "inputs": {
          "host": {
            "connection": {
              "name": "@parameters('$connections')['azureblob']['connectionId']"
            }
          },
          "method": "post",
          "path": "/v2/datasets/@{encodeURIComponent('AccountNameFromSettings')}/files"
        }
      }
    }
  }
}
```

### 8. Security Implementation

#### Azure Active Directory Integration

**App Registration**
```json
{
  "displayName": "TextNLP Platform",
  "signInAudience": "AzureADMyOrg",
  "api": {
    "requestedAccessTokenVersion": 2,
    "oauth2PermissionScopes": [
      {
        "id": "generate.read",
        "value": "TextGeneration.Read",
        "type": "User",
        "adminConsentDisplayName": "Read text generations"
      },
      {
        "id": "generate.write",
        "value": "TextGeneration.Write",
        "type": "User",
        "adminConsentDisplayName": "Create text generations"
      }
    ]
  },
  "requiredResourceAccess": [
    {
      "resourceAppId": "00000003-0000-0000-c000-000000000000",
      "resourceAccess": [
        {
          "id": "user.read",
          "type": "Scope"
        }
      ]
    }
  ]
}
```

#### Key Vault Configuration

**Secrets Structure**
```yaml
Secrets:
  - database-connection-string
  - redis-connection-string
  - openai-api-key
  - storage-account-key
  - servicebus-connection-string
  - jwt-signing-key
  
Certificates:
  - ssl-certificate
  - client-authentication-cert
  
Keys:
  - data-encryption-key
  - backup-encryption-key
```

**Access Policies**
```json
{
  "accessPolicies": [
    {
      "tenantId": "tenant-id",
      "objectId": "aks-managed-identity",
      "permissions": {
        "secrets": ["get", "list"],
        "certificates": ["get", "list"],
        "keys": ["decrypt", "encrypt"]
      }
    }
  ]
}
```

### 9. Monitoring and Observability

#### Azure Monitor Configuration

**Log Analytics Workspace**
```json
{
  "name": "textnlp-logs",
  "location": "eastus",
  "properties": {
    "retentionInDays": 90,
    "features": {
      "searchVersion": 2,
      "enableLogAccessUsingOnlyResourcePermissions": true
    },
    "workspaceCapping": {
      "dailyQuotaGb": 100
    }
  }
}
```

**Application Insights**
```yaml
Components:
  - Name: textnlp-api-insights
    Type: web
    RetentionInDays: 90
    IngestionMode: LogAnalytics
    PublicNetworkAccessForIngestion: Disabled
    
  - Name: textnlp-generation-insights  
    Type: other
    RetentionInDays: 90
    SamplingPercentage: 100
```

**Monitoring Dashboards**
```json
{
  "name": "TextNLP-Production-Dashboard",
  "dashboardPanes": [
    {
      "title": "API Performance",
      "visualization": "timechart",
      "query": "requests | summarize avg(duration), percentile(duration, 95), percentile(duration, 99) by bin(timestamp, 5m)"
    },
    {
      "title": "Generation Metrics",
      "visualization": "barchart",
      "query": "customMetrics | where name == 'TokensGenerated' | summarize sum(value) by bin(timestamp, 1h)"
    },
    {
      "title": "Error Rate",
      "visualization": "linechart",
      "query": "requests | where success == false | summarize count() by bin(timestamp, 5m), resultCode"
    }
  ]
}
```

#### Distributed Tracing

**Application Map Configuration**
- API Management → AKS API Service
- AKS API Service → Azure OpenAI
- AKS API Service → PostgreSQL
- AKS API Service → Redis Cache
- AKS API Service → Blob Storage
- AKS API Service → Service Bus

### 10. Disaster Recovery

#### Backup Strategy

**Azure Backup Configuration**
```yaml
BackupPolicies:
  - Name: Database-Backup-Policy
    Type: PostgreSQL
    Schedule:
      Frequency: Daily
      Time: "02:00"
    Retention:
      Daily: 7
      Weekly: 4
      Monthly: 12
      Yearly: 5
    
  - Name: Storage-Backup-Policy
    Type: Blob
    Schedule:
      Frequency: Hourly
    Retention:
      PointInTime: 30 days
```

**Geo-Replication Setup**
```json
{
  "primaryRegion": "eastus",
  "secondaryRegion": "westus2",
  "replicationStrategy": {
    "database": "ActiveGeoReplication",
    "storage": "GRS-RA",
    "redis": "ActiveGeoReplication",
    "aks": "MultiRegionClusters"
  }
}
```

### 11. Cost Optimization

#### Azure Advisor Recommendations

**Cost Optimization Rules**
```yaml
Recommendations:
  - RightSizeVMs:
      Check: Underutilized VMs
      Action: Resize or shutdown
      
  - ReservedInstances:
      Coverage: 70%
      Term: 3 years
      PaymentOption: Upfront
      
  - StorageOptimization:
      EnableLifecycle: true
      ArchiveAfterDays: 90
      DeleteAfterDays: 365
      
  - AutoShutdown:
      NonProdEnvironments: true
      Schedule: "18:00-08:00"
```

#### Cost Management

**Budget Alerts**
```json
{
  "name": "textnlp-monthly-budget",
  "properties": {
    "category": "Cost",
    "amount": 50000,
    "timeGrain": "Monthly",
    "notifications": {
      "Actual_GreaterThan_80_Percent": {
        "enabled": true,
        "operator": "GreaterThan",
        "threshold": 80,
        "contactEmails": ["finance@textnlp.com"]
      }
    }
  }
}
```

### 12. Deployment Pipeline

#### Azure DevOps Pipeline

```yaml
trigger:
  branches:
    include:
    - main
  paths:
    include:
    - src/*
    - deploy/*

stages:
- stage: Build
  jobs:
  - job: BuildAndTest
    pool:
      vmImage: 'ubuntu-latest'
    steps:
    - task: Docker@2
      inputs:
        containerRegistry: 'textnlpacr'
        repository: 'textnlp-api'
        command: 'buildAndPush'
        Dockerfile: '**/Dockerfile'
        
- stage: DeployDev
  jobs:
  - deployment: DeployToAKS
    environment: 'development'
    strategy:
      runOnce:
        deploy:
          steps:
          - task: KubernetesManifest@0
            inputs:
              action: 'deploy'
              namespace: 'development'
              manifests: 'k8s/dev/*.yaml'
              
- stage: DeployProd
  condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/main'))
  jobs:
  - deployment: DeployToProduction
    environment: 'production'
    strategy:
      canary:
        increments: [10, 25, 50]
        preDeploy:
          steps:
          - task: AzureCLI@2
            inputs:
              scriptType: 'bash'
              scriptLocation: 'inlineScript'
              inlineScript: |
                az aks get-credentials -n textnlp-aks-production -g textnlp-rg
                kubectl apply -f k8s/prod/canary.yaml
```

## Implementation Timeline

### Phase 1: Foundation (Week 1-2)
- Resource group and VNet setup
- AKS cluster deployment
- PostgreSQL and Redis provisioning
- Storage account configuration

### Phase 2: Core Services (Week 3-4)
- Container registry setup
- Initial deployments to AKS
- API Management configuration
- Key Vault integration

### Phase 3: AI/ML Integration (Week 5-6)
- Azure OpenAI setup
- Azure ML workspace creation
- Model endpoint deployment
- Batch processing setup

### Phase 4: Production Readiness (Week 7-8)
- Monitoring implementation
- Security hardening
- DR testing
- Performance optimization

## Conclusion

This Azure infrastructure design provides a comprehensive, cloud-native solution for the TextNLP platform. By leveraging Azure's managed services and AI capabilities, we ensure high availability, security, and scalability while maintaining cost efficiency and operational simplicity.