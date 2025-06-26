# Team Access Configuration - TextNLP Platform

## Overview

This document defines team access roles, permissions, and configuration procedures for the TextNLP platform deployment. It establishes a secure, role-based access control (RBAC) system across all platforms and environments.

## Team Structure and Roles

### 1. Core Team Roles

| Role | Count | Primary Responsibilities |
|------|-------|------------------------|
| **Platform Architect** | 1 | Overall system design, technical decisions |
| **DevOps Engineer** | 2 | Infrastructure automation, CI/CD |
| **ML Engineer** | 3 | Model development, training, validation |
| **Backend Developer** | 2 | API development, service integration |
| **Security Engineer** | 1 | Security policies, compliance, auditing |
| **Product Manager** | 1 | Requirements, roadmap, stakeholder coordination |
| **QA Engineer** | 2 | Testing, quality assurance, validation |

### 2. Extended Team Roles

| Role | Count | Primary Responsibilities |
|------|-------|------------------------|
| **Data Scientist** | 2 | Data analysis, model evaluation |
| **Site Reliability Engineer** | 1 | Production monitoring, incident response |
| **Technical Writer** | 1 | Documentation, user guides |
| **Business Analyst** | 1 | Requirements analysis, business logic |

## Access Level Matrix

### Platform Access Levels

| Level | Description | Privileges |
|-------|-------------|------------|
| **L0 - Read Only** | View-only access | Dashboard, metrics, documentation |
| **L1 - Developer** | Development access | Deploy to dev, view staging |
| **L2 - Senior Developer** | Extended development | Deploy to staging, limited prod access |
| **L3 - Lead** | Team lead access | All environments, team management |
| **L4 - Admin** | Administrative access | Full system access, security config |

### Environment Access Matrix

| Role | Development | Staging | Production | Administrative |
|------|-------------|---------|------------|---------------|
| Platform Architect | L4 | L4 | L4 | L4 |
| DevOps Engineer | L4 | L4 | L3 | L3 |
| ML Engineer | L3 | L2 | L1 | L0 |
| Backend Developer | L3 | L2 | L1 | L0 |
| Security Engineer | L4 | L4 | L4 | L4 |
| Product Manager | L1 | L1 | L0 | L0 |
| QA Engineer | L2 | L3 | L1 | L0 |
| Data Scientist | L2 | L1 | L0 | L0 |
| SRE | L3 | L4 | L4 | L2 |

## AWS IAM Configuration

### 1. IAM Groups and Policies

```bash
# Create IAM groups
aws iam create-group --group-name TextNLP-Platform-Architects
aws iam create-group --group-name TextNLP-DevOps-Engineers
aws iam create-group --group-name TextNLP-ML-Engineers
aws iam create-group --group-name TextNLP-Backend-Developers
aws iam create-group --group-name TextNLP-Security-Engineers
aws iam create-group --group-name TextNLP-QA-Engineers
aws iam create-group --group-name TextNLP-Read-Only
```

### 2. Platform Architect Policy
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "*",
            "Resource": "*"
        }
    ]
}
```

### 3. DevOps Engineer Policy
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ec2:*",
                "ecs:*",
                "eks:*",
                "s3:*",
                "iam:PassRole",
                "iam:CreateServiceLinkedRole",
                "cloudformation:*",
                "cloudwatch:*",
                "logs:*",
                "autoscaling:*",
                "elasticloadbalancing:*"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Deny",
            "Action": [
                "iam:DeleteRole",
                "iam:DeleteUser",
                "iam:DeletePolicy"
            ],
            "Resource": "*",
            "Condition": {
                "StringNotEquals": {
                    "aws:RequestedRegion": ["us-east-1", "us-west-2"]
                }
            }
        }
    ]
}
```

### 4. ML Engineer Policy
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::textnlp-models/*",
                "arn:aws:s3:::textnlp-datasets/*",
                "arn:aws:s3:::textnlp-dev/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "ec2:DescribeInstances",
                "ec2:DescribeImages",
                "ec2:RunInstances",
                "ec2:TerminateInstances"
            ],
            "Resource": "*",
            "Condition": {
                "StringEquals": {
                    "ec2:InstanceType": [
                        "g4dn.xlarge",
                        "g4dn.2xlarge",
                        "p3.2xlarge"
                    ]
                }
            }
        }
    ]
}
```

### 5. User Creation Script
```bash
#!/bin/bash
# create-team-users.sh

# Platform Architect
aws iam create-user --user-name john.architect
aws iam add-user-to-group --user-name john.architect --group-name TextNLP-Platform-Architects

# DevOps Engineers
aws iam create-user --user-name sarah.devops
aws iam create-user --user-name mike.devops
aws iam add-user-to-group --user-name sarah.devops --group-name TextNLP-DevOps-Engineers
aws iam add-user-to-group --user-name mike.devops --group-name TextNLP-DevOps-Engineers

# ML Engineers
aws iam create-user --user-name alex.ml
aws iam create-user --user-name priya.ml
aws iam create-user --user-name david.ml
aws iam add-user-to-group --user-name alex.ml --group-name TextNLP-ML-Engineers
aws iam add-user-to-group --user-name priya.ml --group-name TextNLP-ML-Engineers
aws iam add-user-to-group --user-name david.ml --group-name TextNLP-ML-Engineers

# Generate access keys (store securely)
for user in john.architect sarah.devops mike.devops alex.ml priya.ml david.ml; do
    aws iam create-access-key --user-name $user > ${user}-credentials.json
    echo "Created credentials for $user"
done
```

## GCP IAM Configuration

### 1. Custom Roles Creation
```bash
# Platform Architect Role
gcloud iam roles create textnlp.platformArchitect \
    --project=textnlp-prod-001 \
    --title="TextNLP Platform Architect" \
    --description="Full access to TextNLP platform" \
    --permissions="$(cat platform-architect-permissions.txt)"

# DevOps Engineer Role
gcloud iam roles create textnlp.devopsEngineer \
    --project=textnlp-prod-001 \
    --title="TextNLP DevOps Engineer" \
    --description="Infrastructure and deployment access" \
    --permissions="$(cat devops-engineer-permissions.txt)"

# ML Engineer Role
gcloud iam roles create textnlp.mlEngineer \
    --project=textnlp-prod-001 \
    --title="TextNLP ML Engineer" \
    --description="ML model development access" \
    --permissions="$(cat ml-engineer-permissions.txt)"
```

### 2. Service Account for Teams
```bash
# Create team service accounts
gcloud iam service-accounts create textnlp-devops-team \
    --display-name="TextNLP DevOps Team" \
    --project=textnlp-prod-001

gcloud iam service-accounts create textnlp-ml-team \
    --display-name="TextNLP ML Team" \
    --project=textnlp-prod-001

# Bind roles to service accounts
gcloud projects add-iam-policy-binding textnlp-prod-001 \
    --member="serviceAccount:textnlp-devops-team@textnlp-prod-001.iam.gserviceaccount.com" \
    --role="projects/textnlp-prod-001/roles/textnlp.devopsEngineer"
```

### 3. User Assignment
```bash
# Assign users to roles
gcloud projects add-iam-policy-binding textnlp-prod-001 \
    --member="user:john.architect@company.com" \
    --role="projects/textnlp-prod-001/roles/textnlp.platformArchitect"

gcloud projects add-iam-policy-binding textnlp-prod-001 \
    --member="user:sarah.devops@company.com" \
    --role="projects/textnlp-prod-001/roles/textnlp.devopsEngineer"

gcloud projects add-iam-policy-binding textnlp-prod-001 \
    --member="user:alex.ml@company.com" \
    --role="projects/textnlp-prod-001/roles/textnlp.mlEngineer"
```

## Azure RBAC Configuration

### 1. Custom Role Definitions
```json
// platform-architect-role.json
{
    "Name": "TextNLP Platform Architect",
    "Description": "Full access to TextNLP platform resources",
    "Actions": [
        "*"
    ],
    "NotActions": [],
    "AssignableScopes": [
        "/subscriptions/{subscription-id}/resourceGroups/textnlp-prod-rg"
    ]
}
```

```json
// devops-engineer-role.json
{
    "Name": "TextNLP DevOps Engineer",
    "Description": "Infrastructure and deployment access",
    "Actions": [
        "Microsoft.Compute/*",
        "Microsoft.ContainerInstance/*",
        "Microsoft.ContainerService/*",
        "Microsoft.Storage/*",
        "Microsoft.Network/*",
        "Microsoft.Resources/*"
    ],
    "NotActions": [
        "Microsoft.Authorization/*/Delete",
        "Microsoft.Authorization/*/Write"
    ],
    "AssignableScopes": [
        "/subscriptions/{subscription-id}/resourceGroups/textnlp-prod-rg"
    ]
}
```

### 2. Create Custom Roles
```bash
# Create custom roles
az role definition create --role-definition platform-architect-role.json
az role definition create --role-definition devops-engineer-role.json
az role definition create --role-definition ml-engineer-role.json

# Create Azure AD groups
az ad group create --display-name "TextNLP Platform Architects" --mail-nickname textnlp-architects
az ad group create --display-name "TextNLP DevOps Engineers" --mail-nickname textnlp-devops
az ad group create --display-name "TextNLP ML Engineers" --mail-nickname textnlp-ml
```

### 3. Assign Roles to Groups
```bash
# Get group object IDs
ARCHITECT_GROUP_ID=$(az ad group show --group "TextNLP Platform Architects" --query objectId -o tsv)
DEVOPS_GROUP_ID=$(az ad group show --group "TextNLP DevOps Engineers" --query objectId -o tsv)
ML_GROUP_ID=$(az ad group show --group "TextNLP ML Engineers" --query objectId -o tsv)

# Assign roles
az role assignment create \
    --assignee $ARCHITECT_GROUP_ID \
    --role "TextNLP Platform Architect" \
    --resource-group textnlp-prod-rg

az role assignment create \
    --assignee $DEVOPS_GROUP_ID \
    --role "TextNLP DevOps Engineer" \
    --resource-group textnlp-prod-rg
```

## Kubernetes RBAC Configuration

### 1. Namespace-based Access
```yaml
# namespaces.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: textnlp-dev
  labels:
    environment: development
---
apiVersion: v1
kind: Namespace
metadata:
  name: textnlp-staging
  labels:
    environment: staging
---
apiVersion: v1
kind: Namespace
metadata:
  name: textnlp-prod
  labels:
    environment: production
```

### 2. RBAC Roles
```yaml
# rbac-roles.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: textnlp-platform-architect
rules:
- apiGroups: ["*"]
  resources: ["*"]
  verbs: ["*"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: textnlp-devops-engineer
rules:
- apiGroups: ["apps", "extensions"]
  resources: ["deployments", "replicasets", "daemonsets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: textnlp-dev
  name: textnlp-ml-engineer
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
```

### 3. Role Bindings
```yaml
# rbac-bindings.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: textnlp-platform-architects
subjects:
- kind: User
  name: john.architect@company.com
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: textnlp-platform-architect
  apiGroup: rbac.authorization.k8s.io
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: textnlp-devops-engineers
subjects:
- kind: User
  name: sarah.devops@company.com
  apiGroup: rbac.authorization.k8s.io
- kind: User
  name: mike.devops@company.com
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: textnlp-devops-engineer
  apiGroup: rbac.authorization.k8s.io
```

## Git Repository Access

### 1. Repository Structure
```
inferloop-synthdata/
├── textnlp/                    # Main application code
├── infrastructure/             # Infrastructure as code
├── docs/                      # Documentation
├── scripts/                   # Deployment scripts
└── .github/                   # CI/CD workflows
```

### 2. Branch Protection Rules
```yaml
# .github/branch-protection.yml
main:
  required_status_checks:
    strict: true
    contexts:
      - "continuous-integration"
      - "security-scan"
      - "code-quality"
  enforce_admins: false
  required_pull_request_reviews:
    required_approving_review_count: 2
    dismiss_stale_reviews: true
    require_code_owner_reviews: true
  restrictions:
    users: []
    teams: ["textnlp-platform-architects", "textnlp-devops-engineers"]
```

### 3. Team Permissions
```yaml
# .github/teams.yml
teams:
  textnlp-platform-architects:
    permission: admin
    members:
      - john.architect
  
  textnlp-devops-engineers:
    permission: maintain
    members:
      - sarah.devops
      - mike.devops
  
  textnlp-ml-engineers:
    permission: write
    members:
      - alex.ml
      - priya.ml
      - david.ml
  
  textnlp-developers:
    permission: write
    members:
      - backend.dev1
      - backend.dev2
```

## CI/CD Access Control

### 1. GitHub Actions Secrets
```bash
# Repository secrets (admin access required)
gh secret set AWS_ACCESS_KEY_ID --body "AKIA..."
gh secret set AWS_SECRET_ACCESS_KEY --body "..."
gh secret set GCP_SERVICE_ACCOUNT_KEY --body "$(cat key.json)"
gh secret set AZURE_CREDENTIALS --body "$(cat azure-creds.json)"

# Environment secrets
gh secret set PROD_DATABASE_URL --env production
gh secret set STAGING_DATABASE_URL --env staging
```

### 2. Deployment Environments
```yaml
# .github/workflows/deploy.yml
environments:
  development:
    required_reviewers: []
    deployment_branches:
      - main
      - develop
  
  staging:
    required_reviewers:
      - textnlp-devops-engineers
    deployment_branches:
      - main
  
  production:
    required_reviewers:
      - textnlp-platform-architects
      - textnlp-devops-engineers
    deployment_branches:
      - main
```

## Monitoring and Audit Access

### 1. Logging Access
```yaml
# Log access by role
roles:
  platform_architect:
    - application_logs: read/write
    - infrastructure_logs: read/write
    - security_logs: read/write
    - audit_logs: read/write
  
  devops_engineer:
    - application_logs: read/write
    - infrastructure_logs: read/write
    - security_logs: read
    - audit_logs: read
  
  ml_engineer:
    - application_logs: read
    - infrastructure_logs: read
    - model_logs: read/write
```

### 2. Monitoring Dashboard Access
```yaml
# Grafana teams and permissions
teams:
  Platform Architects:
    role: Admin
    dashboards: ["*"]
    
  DevOps Engineers:
    role: Editor
    dashboards: ["Infrastructure", "Application", "Performance"]
    
  ML Engineers:
    role: Viewer
    dashboards: ["ML Models", "Training", "Inference"]
```

## Security Policies

### 1. Password Policy
- Minimum 12 characters
- Must include uppercase, lowercase, numbers, symbols
- Cannot reuse last 12 passwords
- Expires every 90 days
- Account lockout after 5 failed attempts

### 2. MFA Requirements
- Required for all users
- TOTP apps preferred (Google Authenticator, Authy)
- Hardware tokens for admin roles
- Backup codes generated and stored securely

### 3. Session Management
- Session timeout: 8 hours
- Idle timeout: 2 hours
- Concurrent session limit: 3
- IP restriction for production access

## Emergency Access Procedures

### 1. Break-Glass Access
```bash
# Emergency admin account (sealed)
# Username: emergency-admin
# Password: stored in sealed envelope
# MFA: hardware token in safe

# Usage procedure:
# 1. Get approval from 2 executives
# 2. Document incident ticket
# 3. Use account for emergency only
# 4. Rotate credentials immediately after use
```

### 2. Incident Response Team
| Role | Primary Contact | Backup Contact |
|------|----------------|----------------|
| Incident Commander | john.architect@company.com | sarah.devops@company.com |
| Security Lead | security.eng@company.com | john.architect@company.com |
| Communications | product.mgr@company.com | business.analyst@company.com |

## Access Review Process

### 1. Monthly Reviews
- Review all user access permissions
- Validate team membership
- Check for unused accounts
- Audit privileged access

### 2. Quarterly Reviews
- Complete access recertification
- Update role definitions
- Review and update policies
- Security training compliance check

### 3. Annual Reviews
- Full security audit
- Penetration testing
- Compliance assessment
- Policy updates

## Onboarding Checklist

### New Team Member Onboarding
- [ ] Create user accounts (AWS, GCP, Azure)
- [ ] Add to appropriate groups/teams
- [ ] Provision development environment
- [ ] Grant repository access
- [ ] Setup MFA
- [ ] Complete security training
- [ ] Review access policies
- [ ] Document access grants

### Offboarding Checklist
- [ ] Disable all user accounts
- [ ] Remove from groups/teams
- [ ] Revoke API keys and tokens
- [ ] Remove repository access
- [ ] Collect company devices
- [ ] Update emergency contacts
- [ ] Document access removal

## Completion Status

- [ ] Team roles defined
- [ ] AWS IAM configured
- [ ] GCP IAM configured
- [ ] Azure RBAC configured
- [ ] Kubernetes RBAC configured
- [ ] Git repository access configured
- [ ] CI/CD access control setup
- [ ] Monitoring access configured
- [ ] Security policies documented
- [ ] Emergency procedures defined
- [ ] Access review process established
- [ ] Team training completed