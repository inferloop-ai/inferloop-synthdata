# Account Credentials Setup Guide - TextNLP Platform

## Overview

This document provides secure account setup procedures for all cloud platforms required for TextNLP deployment. All credentials must be stored in approved secret management systems.

## Security Requirements

### Mandatory Security Practices
1. **MFA Required**: Enable multi-factor authentication on all accounts
2. **Least Privilege**: Grant minimum required permissions
3. **Rotation Policy**: Rotate credentials every 90 days
4. **Audit Logging**: Enable CloudTrail/Activity Logs
5. **Secure Storage**: Use HashiCorp Vault or cloud KMS

## AWS Account Setup

### 1. Root Account Configuration
```bash
# Never use root account for daily operations
# Enable MFA on root account immediately
# Create billing alerts
```

### 2. IAM User Creation
```bash
# Create deployment user
aws iam create-user --user-name textnlp-deployment

# Create access key
aws iam create-access-key --user-name textnlp-deployment

# Output format:
# {
#     "AccessKey": {
#         "UserName": "textnlp-deployment",
#         "AccessKeyId": "AKIA...",
#         "SecretAccessKey": "...",
#         "Status": "Active"
#     }
# }
```

### 3. IAM Policies
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
                "iam:*",
                "cloudwatch:*",
                "logs:*",
                "route53:*",
                "elasticloadbalancing:*"
            ],
            "Resource": "*"
        }
    ]
}
```

### 4. AWS CLI Configuration
```bash
# Configure AWS CLI
aws configure --profile textnlp-prod

# AWS Access Key ID [None]: AKIA...
# AWS Secret Access Key [None]: ...
# Default region name [None]: us-east-1
# Default output format [None]: json

# Test configuration
aws sts get-caller-identity --profile textnlp-prod
```

### 5. Service-Specific Roles
```bash
# ECS Task Role
aws iam create-role --role-name textnlp-ecs-task-role \
    --assume-role-policy-document file://ecs-trust-policy.json

# EKS Service Role
aws iam create-role --role-name textnlp-eks-service-role \
    --assume-role-policy-document file://eks-trust-policy.json

# Lambda Execution Role
aws iam create-role --role-name textnlp-lambda-execution-role \
    --assume-role-policy-document file://lambda-trust-policy.json
```

## GCP Account Setup

### 1. Project Creation
```bash
# Create new project
gcloud projects create textnlp-prod-001 \
    --name="TextNLP Production" \
    --organization=123456789

# Set as default project
gcloud config set project textnlp-prod-001

# Enable required APIs
gcloud services enable compute.googleapis.com
gcloud services enable container.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable storage-api.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable monitoring.googleapis.com
gcloud services enable logging.googleapis.com
```

### 2. Service Account Creation
```bash
# Create service account
gcloud iam service-accounts create textnlp-deployment \
    --display-name="TextNLP Deployment Service Account"

# Generate key
gcloud iam service-accounts keys create textnlp-key.json \
    --iam-account=textnlp-deployment@textnlp-prod-001.iam.gserviceaccount.com

# Grant necessary roles
gcloud projects add-iam-policy-binding textnlp-prod-001 \
    --member="serviceAccount:textnlp-deployment@textnlp-prod-001.iam.gserviceaccount.com" \
    --role="roles/editor"

gcloud projects add-iam-policy-binding textnlp-prod-001 \
    --member="serviceAccount:textnlp-deployment@textnlp-prod-001.iam.gserviceaccount.com" \
    --role="roles/compute.admin"
```

### 3. gcloud Configuration
```bash
# Authenticate with service account
gcloud auth activate-service-account \
    --key-file=textnlp-key.json

# Set default configuration
gcloud config set project textnlp-prod-001
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a
```

### 4. Workload Identity Setup
```bash
# Create Kubernetes service account
kubectl create serviceaccount textnlp-ksa \
    --namespace textnlp-prod

# Bind to GCP service account
gcloud iam service-accounts add-iam-policy-binding \
    textnlp-deployment@textnlp-prod-001.iam.gserviceaccount.com \
    --role roles/iam.workloadIdentityUser \
    --member "serviceAccount:textnlp-prod-001.svc.id.goog[textnlp-prod/textnlp-ksa]"
```

## Azure Account Setup

### 1. Subscription Configuration
```bash
# Login to Azure
az login

# List subscriptions
az account list --output table

# Set default subscription
az account set --subscription "TextNLP Production"
```

### 2. Resource Group Creation
```bash
# Create resource group
az group create \
    --name textnlp-prod-rg \
    --location eastus
```

### 3. Service Principal Creation
```bash
# Create service principal
az ad sp create-for-rbac \
    --name textnlp-deployment \
    --role Contributor \
    --scopes /subscriptions/{subscription-id} \
    --sdk-auth

# Output format:
# {
#   "clientId": "...",
#   "clientSecret": "...",
#   "subscriptionId": "...",
#   "tenantId": "...",
#   "activeDirectoryEndpointUrl": "...",
#   "resourceManagerEndpointUrl": "...",
#   "activeDirectoryGraphResourceId": "...",
#   "sqlManagementEndpointUrl": "...",
#   "galleryEndpointUrl": "...",
#   "managementEndpointUrl": "..."
# }
```

### 4. Azure CLI Configuration
```bash
# Login with service principal
az login --service-principal \
    --username {clientId} \
    --password {clientSecret} \
    --tenant {tenantId}

# Set defaults
az configure --defaults group=textnlp-prod-rg location=eastus
```

### 5. Key Vault Setup
```bash
# Create Key Vault
az keyvault create \
    --name textnlp-prod-kv \
    --resource-group textnlp-prod-rg \
    --location eastus

# Grant access to service principal
az keyvault set-policy \
    --name textnlp-prod-kv \
    --spn {clientId} \
    --secret-permissions get list set delete \
    --key-permissions get list create delete
```

## On-Premises Setup

### 1. LDAP/Active Directory Integration
```yaml
# ldap-config.yaml
ldap:
  server: ldap://corp.company.com:389
  base_dn: dc=company,dc=com
  bind_dn: cn=textnlp-service,ou=services,dc=company,dc=com
  bind_password: ${LDAP_BIND_PASSWORD}
  user_search_base: ou=users,dc=company,dc=com
  group_search_base: ou=groups,dc=company,dc=com
```

### 2. Service Accounts
```bash
# Create Kubernetes service account
kubectl create namespace textnlp-prod
kubectl create serviceaccount textnlp-sa -n textnlp-prod

# Create RBAC
kubectl create clusterrolebinding textnlp-admin \
    --clusterrole=cluster-admin \
    --serviceaccount=textnlp-prod:textnlp-sa
```

### 3. Certificate Management
```bash
# Generate certificates for internal CA
openssl genrsa -out ca.key 4096
openssl req -new -x509 -days 3650 -key ca.key -out ca.crt \
    -subj "/C=US/ST=CA/L=SF/O=Company/CN=TextNLP-CA"

# Create server certificate
openssl genrsa -out server.key 4096
openssl req -new -key server.key -out server.csr \
    -subj "/C=US/ST=CA/L=SF/O=Company/CN=*.textnlp.internal"
openssl x509 -req -days 365 -in server.csr -CA ca.crt \
    -CAkey ca.key -CAcreateserial -out server.crt
```

## Credential Storage

### 1. HashiCorp Vault Setup
```bash
# Initialize Vault
vault operator init -key-shares=5 -key-threshold=3

# Store AWS credentials
vault kv put secret/textnlp/aws \
    access_key_id="AKIA..." \
    secret_access_key="..."

# Store GCP credentials
vault kv put secret/textnlp/gcp \
    key_file=@textnlp-key.json

# Store Azure credentials
vault kv put secret/textnlp/azure \
    client_id="..." \
    client_secret="..." \
    tenant_id="..." \
    subscription_id="..."
```

### 2. Environment Variables Template
```bash
# .env.template
# AWS
export AWS_ACCESS_KEY_ID="vault:secret/textnlp/aws#access_key_id"
export AWS_SECRET_ACCESS_KEY="vault:secret/textnlp/aws#secret_access_key"
export AWS_DEFAULT_REGION="us-east-1"

# GCP
export GOOGLE_APPLICATION_CREDENTIALS="/secure/textnlp-key.json"
export GCP_PROJECT="textnlp-prod-001"

# Azure
export AZURE_CLIENT_ID="vault:secret/textnlp/azure#client_id"
export AZURE_CLIENT_SECRET="vault:secret/textnlp/azure#client_secret"
export AZURE_TENANT_ID="vault:secret/textnlp/azure#tenant_id"
export AZURE_SUBSCRIPTION_ID="vault:secret/textnlp/azure#subscription_id"
```

### 3. Kubernetes Secrets
```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: cloud-credentials
  namespace: textnlp-prod
type: Opaque
data:
  aws-access-key-id: <base64>
  aws-secret-access-key: <base64>
  gcp-key-file: <base64>
  azure-client-id: <base64>
  azure-client-secret: <base64>
```

## Access Control Matrix

| Environment | AWS IAM | GCP IAM | Azure RBAC | Kubernetes RBAC |
|------------|---------|---------|------------|-----------------|
| Development | PowerUser | Editor | Contributor | edit |
| Staging | PowerUser | Editor | Contributor | admin |
| Production | Custom Policy | Custom Role | Custom Role | view + deploy |

## Credential Rotation Schedule

| Credential Type | Rotation Frequency | Automation |
|----------------|-------------------|------------|
| API Keys | 90 days | Vault + Lambda |
| Service Account Keys | 180 days | Terraform |
| Database Passwords | 30 days | Vault Database Engine |
| TLS Certificates | 365 days | cert-manager |

## Audit and Compliance

### 1. Enable Audit Logging
```bash
# AWS CloudTrail
aws cloudtrail create-trail --name textnlp-audit-trail \
    --s3-bucket-name textnlp-audit-logs

# GCP Activity Logs
gcloud logging sinks create textnlp-audit-sink \
    storage.googleapis.com/textnlp-audit-logs \
    --log-filter='protoPayload.@type="type.googleapis.com/google.cloud.audit.AuditLog"'

# Azure Activity Logs
az monitor activity-log create --name textnlp-audit \
    --resource-group textnlp-prod-rg
```

### 2. Access Reviews
- Monthly review of all service accounts
- Quarterly review of IAM policies
- Annual penetration testing
- Continuous compliance monitoring

## Emergency Access Procedures

### Break-Glass Account
1. Sealed credentials in physical safe
2. Dual-control access requirement
3. Automatic alerts on usage
4. Immediate rotation after use

### Recovery Procedures
```bash
# AWS account recovery
# Contact: aws-support@company.com
# Phone: +1-800-xxx-xxxx

# GCP account recovery
# Contact: gcp-support@company.com
# Phone: +1-800-xxx-xxxx

# Azure account recovery
# Contact: azure-support@company.com
# Phone: +1-800-xxx-xxxx
```

## Completion Checklist

- [ ] AWS account created and configured
- [ ] GCP project created and configured
- [ ] Azure subscription configured
- [ ] All service accounts created
- [ ] MFA enabled on all accounts
- [ ] Credentials stored in Vault
- [ ] Audit logging enabled
- [ ] Access matrix documented
- [ ] Emergency procedures tested
- [ ] Team training completed