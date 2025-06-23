# Inferloop Infrastructure

Multi-cloud infrastructure deployment and management for Inferloop Synthetic Data Platform.

## Overview

Inferloop Infrastructure provides a unified interface for deploying and managing infrastructure across multiple cloud providers and on-premise environments. It abstracts away provider-specific details while allowing access to provider-specific features when needed.

## Supported Providers

- **AWS** - Amazon Web Services (EC2, ECS, Lambda, S3, RDS, etc.)
- **GCP** - Google Cloud Platform (Compute Engine, Cloud Run, GCS, CloudSQL, etc.)
- **Azure** - Microsoft Azure (VMs, Container Instances, Functions, Blob Storage, etc.)
- **On-Premise** - Kubernetes, Docker, OpenShift, and traditional infrastructure

## Features

- 🌐 **Multi-Cloud Support** - Deploy to any major cloud provider with consistent APIs
- 🏗️ **Infrastructure as Code** - Define infrastructure using YAML/JSON configurations
- 🎯 **Template Engine** - Reusable deployment templates with variable substitution
- 🔄 **Lifecycle Management** - Full resource lifecycle management with health monitoring
- 🚀 **Async Operations** - Non-blocking operations for better performance
- 📊 **Cost Estimation** - Estimate costs before deployment
- 🔒 **Security First** - Built-in security best practices
- 📈 **Monitoring Ready** - Integrated monitoring and logging setup

## Installation

### Basic Installation

```bash
pip install inferloop-infra
```

### Provider-Specific Installation

```bash
# AWS
pip install inferloop-infra[aws]

# Google Cloud
pip install inferloop-infra[gcp]

# Azure
pip install inferloop-infra[azure]

# On-Premise
pip install inferloop-infra[onprem]

# All providers
pip install inferloop-infra[all]
```

## Quick Start

### 1. Create a Deployment Configuration

```yaml
# deployment.yaml
name: my-app-stack
version: "1.0.0"
provider: aws
region: us-east-1

resources:
  app_server:
    type: compute
    config:
      name: my-app-server
      cpu: 2
      memory: 4096
      disk_size: 30
      
  database:
    type: database
    config:
      name: my-app-db
      engine: postgresql
      engine_version: "13.7"
      instance_class: db.t3.micro
```

### 2. Deploy Infrastructure

```bash
# Deploy
inferloop-deploy deploy deployment.yaml

# Deploy with dry run
inferloop-deploy deploy deployment.yaml --dry-run

# Deploy with variables
inferloop-deploy deploy deployment.yaml --var environment=prod --var instance_type=t3.large
```

### 3. Manage Deployments

```bash
# List deployments
inferloop-deploy list

# Check status
inferloop-deploy status my-app-stack-1.0.0-20240101120000

# Update deployment
inferloop-deploy update my-app-stack-1.0.0-20240101120000 updated-deployment.yaml

# Destroy deployment
inferloop-deploy destroy my-app-stack-1.0.0-20240101120000
```

## Using Templates

### 1. List Available Templates

```bash
inferloop-deploy templates --list
```

### 2. Deploy Using Template

```bash
inferloop-deploy deploy --template synthetic-data-stack \
  --var environment=prod \
  --var instance_type=t3.large \
  --var database_instance_class=db.t3.medium
```

### 3. Show Template Details

```bash
inferloop-deploy templates --show synthetic-data-stack
```

## Python SDK Usage

```python
import asyncio
from inferloop_infra import DeploymentOrchestrator, DeploymentConfig

async def deploy_infrastructure():
    # Create orchestrator
    orchestrator = DeploymentOrchestrator()
    
    # Create deployment config
    config = DeploymentConfig(
        name="my-app",
        version="1.0.0",
        provider="aws",
        region="us-east-1",
        resources={
            "web_server": {
                "type": "container",
                "config": {
                    "name": "my-web-app",
                    "image": "nginx:latest",
                    "cpu": 512,
                    "memory": 1024,
                    "port": 80
                }
            }
        }
    )
    
    # Deploy
    status = await orchestrator.deploy(config)
    print(f"Deployment ID: {status.deployment_id}")
    print(f"Status: {status.state.value}")

# Run deployment
asyncio.run(deploy_infrastructure())
```

## Provider-Specific Features

### AWS

```python
from inferloop_infra.providers.aws import AWSProvider, AWSEC2

# Create provider
provider = AWSProvider({
    'access_key_id': 'YOUR_KEY',
    'secret_access_key': 'YOUR_SECRET',
    'region': 'us-east-1'
})

# Use specific AWS features
ec2 = AWSEC2(provider)
instances = await ec2.list()
```

### Multi-Region Deployment

```yaml
# multi-region.yaml
name: global-app
version: "1.0.0"
provider: aws

deployments:
  - region: us-east-1
    resources:
      # US East resources
      
  - region: eu-west-1
    resources:
      # EU West resources
      
  - region: ap-southeast-1
    resources:
      # Asia Pacific resources
```

## Resource Types

### Compute Resources
- Virtual Machines (EC2, Compute Engine, Azure VMs)
- Containers (ECS, Cloud Run, Container Instances)
- Serverless Functions (Lambda, Cloud Functions, Azure Functions)
- Kubernetes Clusters (EKS, GKE, AKS)

### Storage Resources
- Object Storage (S3, GCS, Blob Storage)
- Block Storage (EBS, Persistent Disks, Managed Disks)
- File Storage (EFS, Filestore, Azure Files)

### Database Resources
- Relational Databases (RDS, Cloud SQL, Azure SQL)
- NoSQL Databases (DynamoDB, Firestore, Cosmos DB)
- Cache Services (ElastiCache, Memorystore, Azure Cache)

### Networking Resources
- Virtual Networks (VPC, VPC, VNet)
- Load Balancers (ALB/NLB, Cloud Load Balancing, Azure LB)
- Content Delivery (CloudFront, Cloud CDN, Azure CDN)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     CLI Interface                        │
├─────────────────────────────────────────────────────────┤
│                Deployment Orchestrator                   │
├─────────────────────────────────────────────────────────┤
│   Template Engine  │  Lifecycle Manager  │  Monitoring   │
├─────────────────────────────────────────────────────────┤
│                  Provider Factory                        │
├─────────┬──────────┬──────────┬────────────┬───────────┤
│   AWS   │   GCP    │  Azure   │  On-Prem   │  Custom   │
├─────────┴──────────┴──────────┴────────────┴───────────┤
│              Common Abstractions Layer                   │
└─────────────────────────────────────────────────────────┘
```

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/inferloop/inferloop-infra.git
cd inferloop-infra

# Install with development dependencies
pip install -e ".[dev,all]"

# Run tests
pytest

# Run linting
black .
isort .
flake8
mypy .
```

### Adding a New Provider

1. Create provider module in `providers/` directory
2. Implement provider class inheriting from `BaseProvider`
3. Implement resource managers for each resource type
4. Register provider in `ProviderFactory`
5. Add provider-specific dependencies to `pyproject.toml`
6. Write tests for the provider

## Security Best Practices

1. **Never commit credentials** - Use environment variables or secret managers
2. **Use IAM roles** when possible instead of access keys
3. **Enable encryption** for all storage and databases
4. **Follow least privilege** principle for all permissions
5. **Enable logging and monitoring** for all resources
6. **Use private subnets** for databases and internal services
7. **Implement network segmentation** with security groups/firewalls

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- Documentation: https://inferloop-infra.readthedocs.io
- Issues: https://github.com/inferloop/inferloop-infra/issues
- Discussions: https://github.com/inferloop/inferloop-infra/discussions