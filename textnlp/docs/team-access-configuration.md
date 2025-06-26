# TextNLP Team Access and Credentials Configuration

**Date**: 2025-01-25  
**Version**: 1.0  
**Status**: FINAL

## Overview

This document outlines the team access configuration and credential management for the TextNLP platform. It covers role-based access control (RBAC), credential management, and security best practices.

## Team Roles and Permissions

### 1. Development Team Roles

#### Senior Developer
**Permissions**: Full access to development and staging environments
```yaml
role: senior-developer
permissions:
  - code:write
  - deploy:staging
  - logs:read
  - metrics:read
  - models:write
  - database:write
  - secrets:read
environments: [development, staging]
```

#### Developer
**Permissions**: Code and test access
```yaml
role: developer
permissions:
  - code:write
  - logs:read
  - metrics:read
  - models:read
  - database:read
environments: [development]
```

#### Junior Developer
**Permissions**: Limited development access
```yaml
role: junior-developer
permissions:
  - code:read
  - logs:read
  - models:read
environments: [development]
```

### 2. Operations Team Roles

#### DevOps Engineer
**Permissions**: Infrastructure and deployment access
```yaml
role: devops-engineer
permissions:
  - infrastructure:write
  - deploy:all
  - logs:write
  - metrics:write
  - secrets:write
  - monitoring:write
environments: [development, staging, production]
```

#### Site Reliability Engineer (SRE)
**Permissions**: Monitoring and incident response
```yaml
role: sre
permissions:
  - infrastructure:read
  - logs:write
  - metrics:write
  - monitoring:write
  - incidents:write
environments: [staging, production]
```

### 3. Management Roles

#### Team Lead
**Permissions**: Team management and oversight
```yaml
role: team-lead
permissions:
  - code:read
  - deploy:approve
  - team:manage
  - reports:write
  - budget:read
environments: [all]
```

#### Product Manager
**Permissions**: Product configuration and analytics
```yaml
role: product-manager
permissions:
  - analytics:read
  - features:write
  - models:read
  - reports:read
environments: [staging, production]
```

## Cloud Provider Access Configuration

### AWS Access Setup

#### 1. IAM User Creation
```bash
# Create IAM users for team members
aws iam create-user --user-name textnlp-dev-john
aws iam create-user --user-name textnlp-ops-sarah

# Attach policies
aws iam attach-user-policy \
  --user-name textnlp-dev-john \
  --policy-arn arn:aws:iam::aws:policy/TextNLPDeveloperPolicy

aws iam attach-user-policy \
  --user-name textnlp-ops-sarah \
  --policy-arn arn:aws:iam::aws:policy/TextNLPDevOpsPolicy
```

#### 2. IAM Role Configuration
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::123456789012:root"
      },
      "Action": "sts:AssumeRole",
      "Condition": {
        "StringEquals": {
          "sts:ExternalId": "textnlp-external-id"
        }
      }
    }
  ]
}
```

#### 3. Resource Permissions
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "EC2GPUAccess",
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeInstances",
        "ec2:StartInstances",
        "ec2:StopInstances"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "ec2:InstanceType": ["g4dn.*", "p3.*", "p4d.*"]
        }
      }
    },
    {
      "Sid": "S3ModelAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::textnlp-models/*",
        "arn:aws:s3:::textnlp-models"
      ]
    }
  ]
}
```

### GCP Access Setup

#### 1. Service Account Creation
```bash
# Create service accounts
gcloud iam service-accounts create textnlp-developer \
  --display-name="TextNLP Developer Account"

gcloud iam service-accounts create textnlp-devops \
  --display-name="TextNLP DevOps Account"

# Grant roles
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:textnlp-developer@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/ml.developer"

gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:textnlp-devops@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/compute.admin"
```

#### 2. Custom Role Definition
```yaml
title: "TextNLP Developer"
description: "Custom role for TextNLP developers"
stage: "GA"
includedPermissions:
- compute.instances.get
- compute.instances.list
- compute.instances.start
- compute.instances.stop
- storage.buckets.get
- storage.objects.create
- storage.objects.get
- storage.objects.list
- ml.models.create
- ml.models.get
- ml.models.list
- ml.models.predict
```

### Azure Access Setup

#### 1. Azure AD User Setup
```powershell
# Create Azure AD users
New-AzADUser -DisplayName "John Developer" `
  -UserPrincipalName "john@textnlp.onmicrosoft.com" `
  -Password $SecurePassword

# Assign roles
New-AzRoleAssignment `
  -ObjectId (Get-AzADUser -UserPrincipalName "john@textnlp.onmicrosoft.com").Id `
  -RoleDefinitionName "TextNLP Developer" `
  -ResourceGroupName "textnlp-rg"
```

#### 2. Custom Role Definition
```json
{
  "Name": "TextNLP Developer",
  "IsCustom": true,
  "Description": "Can manage TextNLP resources",
  "Actions": [
    "Microsoft.Compute/virtualMachines/read",
    "Microsoft.Compute/virtualMachines/start/action",
    "Microsoft.Compute/virtualMachines/restart/action",
    "Microsoft.Storage/storageAccounts/blobServices/containers/read",
    "Microsoft.Storage/storageAccounts/blobServices/containers/write",
    "Microsoft.MachineLearningServices/workspaces/read"
  ],
  "AssignableScopes": [
    "/subscriptions/{subscription-id}/resourceGroups/textnlp-rg"
  ]
}
```

## Credential Management

### 1. Secrets Storage Architecture

```yaml
secrets_management:
  provider: hashicorp_vault
  
  structure:
    /textnlp/
      /common/
        - jwt_secret
        - database_password
        - redis_password
      /aws/
        - access_key_id
        - secret_access_key
        - s3_bucket_name
      /gcp/
        - service_account_key
        - project_id
      /azure/
        - subscription_id
        - client_id
        - client_secret
      /api_keys/
        - starter_key
        - professional_key
        - enterprise_key
```

### 2. HashiCorp Vault Configuration

```hcl
# Enable KV secrets engine
path "secret/data/textnlp/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Developer policy
path "secret/data/textnlp/common/*" {
  capabilities = ["read"]
}

path "secret/data/textnlp/api_keys/*" {
  capabilities = ["read"]
}

# DevOps policy
path "secret/data/textnlp/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
```

### 3. Environment-Specific Secrets

```yaml
environments:
  development:
    vault_path: secret/textnlp/dev
    auto_unseal: false
    
  staging:
    vault_path: secret/textnlp/staging
    auto_unseal: true
    seal_type: awskms
    
  production:
    vault_path: secret/textnlp/prod
    auto_unseal: true
    seal_type: awskms
    mfa_required: true
```

## SSH Access Configuration

### 1. SSH Key Management

```bash
# Generate team SSH keys
ssh-keygen -t ed25519 -C "textnlp-dev-team" -f ~/.ssh/textnlp-dev
ssh-keygen -t ed25519 -C "textnlp-ops-team" -f ~/.ssh/textnlp-ops

# Add to SSH agent
ssh-add ~/.ssh/textnlp-dev
ssh-add ~/.ssh/textnlp-ops
```

### 2. Bastion Host Configuration

```yaml
bastion:
  host: bastion.textnlp.ai
  port: 22
  users:
    - username: dev-team
      key: textnlp-dev.pub
      allowed_hosts: ["10.0.1.*", "10.0.2.*"]
    - username: ops-team
      key: textnlp-ops.pub
      allowed_hosts: ["10.0.*.*"]
```

### 3. SSH Config Template

```ssh
# TextNLP Development Environment
Host textnlp-dev-gpu
    HostName 10.0.1.100
    User ubuntu
    IdentityFile ~/.ssh/textnlp-dev
    ProxyJump bastion.textnlp.ai

# TextNLP Staging Environment
Host textnlp-staging-*
    HostName %h.internal.textnlp.ai
    User ubuntu
    IdentityFile ~/.ssh/textnlp-dev
    ProxyJump bastion.textnlp.ai
```

## Database Access Configuration

### 1. PostgreSQL User Setup

```sql
-- Create role for developers
CREATE ROLE textnlp_developers;
GRANT CONNECT ON DATABASE textnlp TO textnlp_developers;
GRANT USAGE ON SCHEMA public TO textnlp_developers;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO textnlp_developers;

-- Create role for applications
CREATE ROLE textnlp_app WITH LOGIN PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE textnlp TO textnlp_app;

-- Create individual users
CREATE USER john_dev WITH LOGIN IN ROLE textnlp_developers PASSWORD 'dev_password';
CREATE USER sarah_ops WITH LOGIN SUPERUSER PASSWORD 'ops_password';
```

### 2. Connection Pooling Configuration

```yaml
pgbouncer:
  databases:
    textnlp:
      host: postgres.internal
      port: 5432
      auth_user: pgbouncer
      
  pools:
    - name: developer_pool
      mode: session
      size: 20
      users: [john_dev, jane_dev]
      
    - name: application_pool
      mode: transaction
      size: 100
      users: [textnlp_app]
```

## API Access Configuration

### 1. API Key Generation

```python
import secrets
import hashlib
import json

def generate_api_key(tier: str, user: str) -> dict:
    """Generate API key for user"""
    key = secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    
    metadata = {
        "tier": tier,
        "user": user,
        "created_at": datetime.utcnow().isoformat(),
        "permissions": get_tier_permissions(tier),
        "rate_limits": get_tier_limits(tier)
    }
    
    # Store in database
    store_api_key(key_hash, metadata)
    
    return {
        "api_key": key,
        "metadata": metadata
    }
```

### 2. OAuth2 Configuration

```yaml
oauth2:
  providers:
    - name: google
      client_id: ${GOOGLE_CLIENT_ID}
      client_secret: ${GOOGLE_CLIENT_SECRET}
      redirect_uri: https://api.textnlp.ai/auth/google/callback
      scopes: [openid, email, profile]
      
    - name: github
      client_id: ${GITHUB_CLIENT_ID}
      client_secret: ${GITHUB_CLIENT_SECRET}
      redirect_uri: https://api.textnlp.ai/auth/github/callback
      scopes: [user:email]
      
  jwt:
    algorithm: RS256
    public_key_path: /keys/jwt_public.pem
    private_key_path: /keys/jwt_private.pem
    expiration: 3600
```

## Monitoring Access Configuration

### 1. Grafana User Setup

```yaml
grafana:
  auth:
    generic_oauth:
      enabled: true
      name: TextNLP SSO
      client_id: ${GRAFANA_CLIENT_ID}
      client_secret: ${GRAFANA_CLIENT_SECRET}
      scopes: openid email profile
      auth_url: https://auth.textnlp.ai/oauth/authorize
      token_url: https://auth.textnlp.ai/oauth/token
      
  users:
    default_role: Viewer
    
  team_mappings:
    - team: Developers
      role: Viewer
      dashboards: [gpu-metrics, model-performance, api-usage]
      
    - team: DevOps
      role: Editor
      dashboards: all
      
    - team: SRE
      role: Admin
      dashboards: all
```

### 2. Prometheus Access Control

```yaml
prometheus:
  basic_auth_users:
    admin: $2y$10$... # bcrypt hash
    readonly: $2y$10$... # bcrypt hash
    
  tls_config:
    cert_file: /certs/prometheus.crt
    key_file: /certs/prometheus.key
    
  web.config:
    users:
      - username: metrics-reader
        password_hash: $2y$10$...
        permissions: [read]
      - username: metrics-admin
        password_hash: $2y$10$...
        permissions: [read, write, admin]
```

## Security Best Practices

### 1. Multi-Factor Authentication (MFA)

```yaml
mfa:
  required_for:
    - production_access
    - secret_management
    - infrastructure_changes
    
  methods:
    - type: totp
      apps: [Google Authenticator, Authy]
    - type: hardware_token
      devices: [YubiKey]
    - type: sms
      backup_only: true
```

### 2. Audit Logging

```yaml
audit:
  enabled: true
  
  events:
    - authentication
    - authorization
    - secret_access
    - model_deployment
    - configuration_change
    
  storage:
    provider: elasticsearch
    retention_days: 90
    
  alerts:
    - event: failed_authentication
      threshold: 5
      window: 5m
      
    - event: unauthorized_access
      threshold: 1
      window: 1m
```

### 3. Credential Rotation Policy

```yaml
rotation_policy:
  api_keys:
    interval: 90d
    grace_period: 14d
    
  database_passwords:
    interval: 60d
    automated: true
    
  ssh_keys:
    interval: 365d
    notification: 30d
    
  certificates:
    interval: 365d
    auto_renew: true
    renewal_threshold: 30d
```

## Onboarding Process

### 1. New Team Member Checklist

- [ ] Create cloud provider accounts (AWS/GCP/Azure)
- [ ] Generate SSH key pair
- [ ] Configure MFA
- [ ] Add to appropriate AD/LDAP groups
- [ ] Grant Vault access
- [ ] Configure VPN access
- [ ] Provide API keys
- [ ] Add to monitoring systems
- [ ] Complete security training
- [ ] Sign NDA and access agreements

### 2. Automated Onboarding Script

```bash
#!/bin/bash
# Onboard new team member

USERNAME=$1
ROLE=$2
EMAIL=$3

# Create accounts
./scripts/create_cloud_accounts.sh $USERNAME $EMAIL
./scripts/generate_ssh_keys.sh $USERNAME
./scripts/configure_vault_access.sh $USERNAME $ROLE
./scripts/setup_monitoring_access.sh $USERNAME $ROLE

# Send welcome email
./scripts/send_onboarding_email.sh $USERNAME $EMAIL

echo "Onboarding complete for $USERNAME"
```

## Offboarding Process

### 1. Access Revocation Checklist

- [ ] Disable cloud provider accounts
- [ ] Revoke SSH access
- [ ] Rotate shared secrets
- [ ] Remove from Vault policies
- [ ] Disable VPN access
- [ ] Revoke API keys
- [ ] Remove from monitoring systems
- [ ] Archive audit logs
- [ ] Update documentation

### 2. Automated Offboarding Script

```bash
#!/bin/bash
# Offboard team member

USERNAME=$1
ARCHIVE_PATH=$2

# Revoke access
./scripts/revoke_cloud_access.sh $USERNAME
./scripts/remove_ssh_keys.sh $USERNAME
./scripts/revoke_vault_access.sh $USERNAME
./scripts/disable_monitoring_access.sh $USERNAME

# Archive data
./scripts/archive_user_data.sh $USERNAME $ARCHIVE_PATH

# Rotate affected credentials
./scripts/rotate_shared_secrets.sh

echo "Offboarding complete for $USERNAME"
```

## Emergency Access Procedures

### 1. Break-Glass Access

```yaml
break_glass:
  enabled: true
  
  accounts:
    - name: emergency-admin
      permissions: all
      mfa_required: true
      notification_list: [cto@textnlp.ai, security@textnlp.ai]
      
  activation:
    approval_required: true
    approvers: [cto, security-lead]
    auto_expire: 4h
    
  audit:
    detailed_logging: true
    real_time_alerts: true
```

### 2. Incident Response Access

```yaml
incident_response:
  on_call_access:
    auto_escalate: true
    temporary_permissions: [logs:write, systems:restart]
    duration: 24h
    
  escalation_path:
    - level: 1
      team: sre
      response_time: 15m
      
    - level: 2
      team: senior-sre
      response_time: 30m
      
    - level: 3
      team: engineering-lead
      response_time: 1h
```

## Compliance and Documentation

### 1. Access Reviews

```yaml
access_reviews:
  frequency: quarterly
  
  scope:
    - cloud_accounts
    - database_access
    - api_keys
    - ssh_access
    
  process:
    - manager_review
    - security_audit
    - update_permissions
    - document_changes
```

### 2. Compliance Tracking

```yaml
compliance:
  standards: [SOC2, GDPR, HIPAA]
  
  requirements:
    - access_logs_retention: 1y
    - permission_reviews: quarterly
    - security_training: annual
    - incident_reporting: 24h
```

---

**Document Maintenance**: This document should be reviewed monthly and updated whenever team changes occur. All changes must be approved by the security team lead.