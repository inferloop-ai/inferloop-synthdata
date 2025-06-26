# AWS Infrastructure Design for TextNLP Synthetic Data Platform

## Executive Summary

This document outlines the AWS-specific infrastructure design for deploying the TextNLP Synthetic Data Generation platform. The architecture leverages AWS managed services to provide a scalable, secure, and cost-effective solution for enterprise text generation workloads.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CloudFront CDN                           │
│                    (Global Edge Locations)                      │
└─────────────────────┬───────────────────────┬──────────────────┘
                      │                       │
┌─────────────────────▼───────────────────────▼──────────────────┐
│                    Route 53 (DNS Management)                    │
└─────────────────────┬───────────────────────┬──────────────────┘
                      │                       │
┌─────────────────────▼───────────────────────▼──────────────────┐
│   Application Load Balancer (Multi-AZ) + WAF + Shield          │
└─────────────────────┬───────────────────────┬──────────────────┘
                      │                       │
        ┌─────────────▼──────────┐ ┌─────────▼─────────────┐
        │   API Gateway (REST)   │ │  API Gateway (WS)     │
        │   + Lambda Authorizer  │ │  For Streaming        │
        └─────────────┬──────────┘ └─────────┬─────────────┘
                      │                       │
┌─────────────────────▼───────────────────────▼──────────────────┐
│                         ECS Fargate Cluster                     │
│  ┌────────────────────────────────────────────────────────┐   │
│  │              API Service (Auto-scaling)                 │   │
│  ├────────────────────────────────────────────────────────┤   │
│  │         Text Generation Service (GPU-enabled)          │   │
│  ├────────────────────────────────────────────────────────┤   │
│  │            Validation Service (CPU-optimized)          │   │
│  └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                    AWS Bedrock / SageMaker                      │
│         (Managed Model Endpoints for Large Models)              │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                        Data Layer                               │
├─────────────────┬────────────────┬──────────────────────────────┤
│   RDS Aurora    │  ElastiCache   │      S3 Buckets            │
│  (PostgreSQL)   │    (Redis)     │  (Storage + Backup)        │
└─────────────────┴────────────────┴──────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                    Supporting Services                          │
├──────────┬──────────┬──────────┬──────────┬───────────────────┤
│   SQS    │   SNS    │  Lambda  │ Step Func│   EventBridge    │
└──────────┴──────────┴──────────┴──────────┴───────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                 Monitoring & Security                           │
├──────────┬──────────┬──────────┬──────────┬───────────────────┤
│CloudWatch│ X-Ray    │ GuardDuty│ Config   │ Security Hub     │
└──────────┴──────────┴──────────┴──────────┴───────────────────┘
```

## Component Details

### 1. Network Architecture

#### VPC Design
```
Production VPC (10.0.0.0/16)
├── Public Subnets (Multi-AZ)
│   ├── 10.0.1.0/24 (us-east-1a) - ALB, NAT Gateway
│   ├── 10.0.2.0/24 (us-east-1b) - ALB, NAT Gateway
│   └── 10.0.3.0/24 (us-east-1c) - ALB, NAT Gateway
├── Private Subnets (Multi-AZ)
│   ├── 10.0.11.0/24 (us-east-1a) - ECS Tasks, Lambda
│   ├── 10.0.12.0/24 (us-east-1b) - ECS Tasks, Lambda
│   └── 10.0.13.0/24 (us-east-1c) - ECS Tasks, Lambda
├── Database Subnets (Multi-AZ)
│   ├── 10.0.21.0/24 (us-east-1a) - RDS Primary
│   ├── 10.0.22.0/24 (us-east-1b) - RDS Standby
│   └── 10.0.23.0/24 (us-east-1c) - Read Replicas
└── Isolated Subnets (Multi-AZ)
    ├── 10.0.31.0/24 (us-east-1a) - SageMaker Endpoints
    ├── 10.0.32.0/24 (us-east-1b) - SageMaker Endpoints
    └── 10.0.33.0/24 (us-east-1c) - SageMaker Endpoints
```

#### Security Groups
- **ALB-SG**: Ingress 80/443 from 0.0.0.0/0
- **API-SG**: Ingress from ALB-SG only
- **DB-SG**: Ingress 5432 from API-SG
- **Cache-SG**: Ingress 6379 from API-SG
- **Model-SG**: Ingress from API-SG for inference

### 2. Compute Infrastructure

#### ECS Fargate Configuration

**API Service Task Definition**
```json
{
  "family": "textnlp-api",
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [{
    "name": "api",
    "image": "ECR_REPO/textnlp-api:latest",
    "portMappings": [{"containerPort": 8000}],
    "environment": [
      {"name": "AWS_REGION", "value": "us-east-1"},
      {"name": "ENVIRONMENT", "value": "production"}
    ],
    "secrets": [
      {"name": "DB_PASSWORD", "valueFrom": "arn:aws:secretsmanager:..."}
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/textnlp-api",
        "awslogs-region": "us-east-1",
        "awslogs-stream-prefix": "api"
      }
    }
  }]
}
```

**Generation Service with GPU**
```json
{
  "family": "textnlp-generation",
  "requiresCompatibilities": ["EC2"],
  "placementConstraints": [{
    "type": "memberOf",
    "expression": "attribute:ecs.instance-type =~ g4dn.*"
  }],
  "cpu": "16384",
  "memory": "65536",
  "containerDefinitions": [{
    "name": "generation",
    "image": "ECR_REPO/textnlp-generation:latest",
    "resourceRequirements": [{
      "type": "GPU",
      "value": "1"
    }]
  }]
}
```

#### Auto-scaling Configuration

**Target Tracking Scaling**
```yaml
ServiceScalingTarget:
  MinCapacity: 2
  MaxCapacity: 100
  TargetValue: 70.0
  PredefinedMetricType: ECSServiceAverageCPUUtilization
  ScaleInCooldown: 300
  ScaleOutCooldown: 60

CustomMetrics:
  - MetricName: RequestsPerTask
    TargetValue: 1000
  - MetricName: GPUUtilization
    TargetValue: 80
```

### 3. Model Serving Infrastructure

#### SageMaker Endpoints

**Multi-Model Endpoint Configuration**
```python
endpoint_config = {
    "EndpointName": "textnlp-multi-model",
    "ProductionVariants": [{
        "VariantName": "gpt-j-6b",
        "ModelName": "gpt-j-6b-model",
        "InitialInstanceCount": 2,
        "InstanceType": "ml.g4dn.xlarge",
        "InitialVariantWeight": 0.5
    }, {
        "VariantName": "llama-7b",
        "ModelName": "llama-7b-model",
        "InitialInstanceCount": 2,
        "InstanceType": "ml.g4dn.2xlarge",
        "InitialVariantWeight": 0.5
    }]
}
```

#### AWS Bedrock Integration

**Bedrock Model Access**
```python
bedrock_config = {
    "region": "us-east-1",
    "models": [
        "anthropic.claude-v2",
        "ai21.j2-ultra-v1",
        "amazon.titan-text-express-v1"
    ],
    "inference_config": {
        "maxTokens": 4096,
        "temperature": 0.7,
        "topP": 0.9
    }
}
```

### 4. Data Storage Architecture

#### S3 Bucket Structure
```
textnlp-production/
├── prompts/
│   ├── templates/
│   ├── user-uploads/
│   └── archived/
├── generated/
│   ├── yyyy/mm/dd/
│   └── batch-jobs/
├── models/
│   ├── checkpoints/
│   ├── fine-tuned/
│   └── configs/
├── validation/
│   ├── results/
│   ├── human-eval/
│   └── metrics/
└── backups/
    ├── database/
    └── configurations/
```

**S3 Lifecycle Policies**
```json
{
  "Rules": [{
    "Id": "ArchiveOldGenerations",
    "Status": "Enabled",
    "Transitions": [{
      "Days": 30,
      "StorageClass": "STANDARD_IA"
    }, {
      "Days": 90,
      "StorageClass": "GLACIER"
    }]
  }, {
    "Id": "DeleteTempFiles",
    "Status": "Enabled",
    "Prefix": "temp/",
    "Expiration": {"Days": 7}
  }]
}
```

#### RDS Aurora Configuration

**Cluster Configuration**
```yaml
Engine: aurora-postgresql
EngineVersion: "15.4"
DBClusterIdentifier: textnlp-production
MasterUsername: textnlp_admin
DatabaseName: textnlp

Instances:
  Writer:
    DBInstanceClass: db.r6g.2xlarge
    PromotionTier: 0
  Readers:
    - DBInstanceClass: db.r6g.xlarge
      PromotionTier: 1
    - DBInstanceClass: db.r6g.xlarge
      PromotionTier: 2

Features:
  - BackupRetentionPeriod: 30
  - PreferredBackupWindow: "03:00-04:00"
  - EnableCloudwatchLogsExports: ["postgresql"]
  - DeletionProtection: true
  - StorageEncrypted: true
```

### 5. Caching Layer

#### ElastiCache Redis Configuration

**Cluster Mode Enabled**
```yaml
CacheClusterId: textnlp-redis
CacheNodeType: cache.r6g.xlarge
NumNodeGroups: 3
ReplicasPerNodeGroup: 2
Engine: redis
EngineVersion: "7.0"

Parameters:
  maxmemory-policy: allkeys-lru
  timeout: 300
  tcp-keepalive: 60
  
SecurityFeatures:
  - AtRestEncryptionEnabled: true
  - TransitEncryptionEnabled: true
  - AuthTokenEnabled: true
```

### 6. Message Queue Architecture

#### SQS Configuration

**Generation Queue**
```json
{
  "QueueName": "textnlp-generation-queue.fifo",
  "FifoQueue": true,
  "ContentBasedDeduplication": true,
  "VisibilityTimeout": 3600,
  "MessageRetentionPeriod": 1209600,
  "RedrivePolicy": {
    "deadLetterTargetArn": "arn:aws:sqs:...:textnlp-dlq",
    "maxReceiveCount": 3
  }
}
```

**Batch Processing Queue**
```json
{
  "QueueName": "textnlp-batch-queue",
  "VisibilityTimeout": 43200,
  "ReceiveMessageWaitTimeSeconds": 20,
  "RedriveAllowPolicy": {
    "redrivePermission": "byQueue",
    "sourceQueueArns": ["arn:aws:sqs:...:textnlp-generation-queue.fifo"]
  }
}
```

### 7. Serverless Components

#### Lambda Functions

**Authentication Lambda**
```python
# JWT Token Validator
def lambda_handler(event, context):
    token = event['authorizationToken']
    # Validate JWT token
    # Return IAM policy
    return {
        'principalId': user_id,
        'policyDocument': generate_policy('Allow', event['methodArn']),
        'context': {'userId': user_id}
    }
```

**Batch Trigger Lambda**
```python
# S3 Event Trigger for Batch Processing
def lambda_handler(event, context):
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        # Trigger ECS task for batch processing
        ecs.run_task(
            cluster='textnlp-cluster',
            taskDefinition='textnlp-batch-processor',
            overrides={'containerOverrides': [{'environment': [
                {'name': 'S3_BUCKET', 'value': bucket},
                {'name': 'S3_KEY', 'value': key}
            ]}]}
        )
```

### 8. API Gateway Configuration

#### REST API Design

**Resource Structure**
```
/api/v1
├── /auth
│   ├── POST /login
│   ├── POST /refresh
│   └── POST /logout
├── /generate
│   ├── POST /text
│   ├── POST /batch
│   └── GET /status/{job_id}
├── /validate
│   ├── POST /quality
│   ├── POST /similarity
│   └── GET /metrics
├── /models
│   ├── GET /list
│   ├── GET /{model_id}
│   └── POST /{model_id}/invoke
└── /admin
    ├── GET /health
    ├── GET /metrics
    └── POST /cache/clear
```

**API Gateway Settings**
```yaml
RestApiId: textnlp-api
StageName: prod
ThrottlingBurstLimit: 5000
ThrottlingRateLimit: 1000
CachingEnabled: true
CacheClusterSize: 1.6
RequestValidation: true
XrayTracingEnabled: true
```

### 9. Security Implementation

#### IAM Roles and Policies

**ECS Task Execution Role**
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "ecr:GetAuthorizationToken",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "secretsmanager:GetSecretValue",
      "kms:Decrypt"
    ],
    "Resource": ["*"]
  }]
}
```

**Application Task Role**
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject"
    ],
    "Resource": ["arn:aws:s3:::textnlp-production/*"]
  }, {
    "Effect": "Allow",
    "Action": [
      "sagemaker:InvokeEndpoint",
      "bedrock:InvokeModel"
    ],
    "Resource": ["*"]
  }, {
    "Effect": "Allow",
    "Action": [
      "sqs:SendMessage",
      "sqs:ReceiveMessage",
      "sqs:DeleteMessage"
    ],
    "Resource": ["arn:aws:sqs:*:*:textnlp-*"]
  }]
}
```

#### Secrets Management

**AWS Secrets Manager Structure**
```
/textnlp/production/
├── database/
│   ├── master-password
│   ├── read-replica-password
│   └── connection-string
├── api/
│   ├── jwt-secret
│   ├── api-keys
│   └── oauth-credentials
├── models/
│   ├── openai-api-key
│   ├── anthropic-api-key
│   └── huggingface-token
└── external/
    ├── smtp-credentials
    └── monitoring-api-keys
```

### 10. Monitoring and Observability

#### CloudWatch Dashboards

**Application Dashboard**
```json
{
  "DashboardName": "TextNLP-Production",
  "DashboardBody": {
    "widgets": [
      {
        "type": "metric",
        "properties": {
          "metrics": [
            ["AWS/ECS", "CPUUtilization", "ServiceName", "textnlp-api"],
            [".", "MemoryUtilization", ".", "."],
            ["Custom", "RequestLatency", ".", "."],
            [".", "TokensGenerated", ".", "."]
          ],
          "period": 300,
          "stat": "Average",
          "region": "us-east-1"
        }
      }
    ]
  }
}
```

#### X-Ray Tracing

**Service Map Configuration**
- API Gateway → Lambda Authorizer
- Lambda → DynamoDB (Session Store)
- API Gateway → ECS Service
- ECS Service → RDS Aurora
- ECS Service → ElastiCache
- ECS Service → SageMaker/Bedrock
- ECS Service → S3

### 11. Disaster Recovery

#### Backup Strategy

**Automated Backups**
```yaml
RDS:
  AutomatedBackups: Enabled
  BackupRetentionPeriod: 30
  BackupWindow: "03:00-04:00 UTC"
  
S3:
  CrossRegionReplication:
    Destination: us-west-2
    StorageClass: STANDARD_IA
    
EBS:
  SnapshotSchedule: Daily
  RetentionPeriod: 7
```

**Recovery Procedures**
1. **RDS Failure**: Automatic failover to standby (RTO: 1-2 minutes)
2. **AZ Failure**: Multi-AZ deployment ensures continuity
3. **Region Failure**: Manual failover to DR region (RTO: 4 hours)

### 12. Cost Optimization

#### Resource Tagging Strategy
```yaml
RequiredTags:
  - Environment: [dev, staging, production]
  - Application: textnlp
  - CostCenter: engineering
  - Owner: team-email
  - Terraform: [true, false]
```

#### Cost Optimization Measures

**Compute Optimization**
- Spot instances for batch processing (70% cost reduction)
- Reserved instances for baseline capacity (40% savings)
- Graviton instances where applicable (20% better price/performance)

**Storage Optimization**
- S3 Intelligent-Tiering for automatic cost optimization
- EBS GP3 volumes with customized IOPS
- Lifecycle policies for log rotation

**Model Serving Optimization**
- SageMaker multi-model endpoints
- Serverless inference for low-traffic models
- Batch transform for offline processing

### 13. Deployment Pipeline

#### CI/CD with AWS Services

```yaml
CodePipeline:
  Source:
    Provider: GitHub
    Repository: inferloop/textnlp
    Branch: main
    
  Build:
    Provider: CodeBuild
    ComputeType: BUILD_GENERAL1_LARGE
    Image: aws/codebuild/standard:6.0
    EnvironmentVariables:
      - ECR_REPOSITORY_URI
      - AWS_REGION
      
  Deploy:
    Provider: ECS
    ClusterName: textnlp-production
    ServiceName: textnlp-api
    FileName: imagedefinitions.json
```

#### Blue-Green Deployment

**CodeDeploy Configuration**
```json
{
  "applicationName": "textnlp",
  "deploymentGroupName": "production",
  "deploymentConfigName": "CodeDeployDefault.ECSLinear10PercentEvery1Minutes",
  "blueGreenDeploymentConfiguration": {
    "terminateBlueInstancesOnDeploymentSuccess": {
      "action": "TERMINATE",
      "terminationWaitTimeInMinutes": 5
    },
    "deploymentReadyOption": {
      "actionOnTimeout": "CONTINUE_DEPLOYMENT"
    },
    "greenFleetProvisioningOption": {
      "action": "COPY_AUTO_SCALING_GROUP"
    }
  }
}
```

## Implementation Timeline

### Phase 1: Foundation (Week 1-2)
- VPC and network setup
- ECS cluster creation
- RDS and ElastiCache deployment
- S3 bucket configuration

### Phase 2: Application (Week 3-4)
- ECS service deployment
- API Gateway configuration
- Lambda function deployment
- Basic monitoring setup

### Phase 3: Model Integration (Week 5-6)
- SageMaker endpoint setup
- Bedrock integration
- Model serving optimization
- Performance testing

### Phase 4: Production Readiness (Week 7-8)
- Security hardening
- Disaster recovery testing
- Cost optimization
- Documentation completion

## Conclusion

This AWS infrastructure design provides a robust, scalable foundation for the TextNLP platform. By leveraging managed services and following AWS best practices, we ensure high availability, security, and cost-effectiveness while maintaining the flexibility to scale based on demand.