#!/bin/bash
# Azure Deployment Automation Script for TextNLP Platform
# Phase 2: Foundation Setup

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../network-configs"
SECURITY_DIR="${SCRIPT_DIR}/../security-configs"
GPU_DIR="${SCRIPT_DIR}/../gpu-configs"

# Azure Configuration
SUBSCRIPTION_ID="12345678-1234-1234-1234-123456789012"
RESOURCE_GROUP="textnlp-prod-rg"
LOCATION="eastus"
CLUSTER_NAME="textnlp-production"
ACR_NAME="textnlpacrprod"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Azure CLI
    if ! command -v az &> /dev/null; then
        error "Azure CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check Terraform
    if ! command -v terraform &> /dev/null; then
        error "Terraform is not installed. Please install it first."
        exit 1
    fi
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed. Please install it first."
        exit 1
    fi
    
    # Check Helm
    if ! command -v helm &> /dev/null; then
        error "Helm is not installed. Please install it first."
        exit 1
    fi
    
    # Check Azure authentication
    if ! az account show &> /dev/null; then
        error "Azure authentication not configured. Please run 'az login' first."
        exit 1
    fi
    
    # Set subscription
    az account set --subscription "${SUBSCRIPTION_ID}"
    
    success "Prerequisites check completed"
}

# Create resource group
create_resource_group() {
    log "Creating resource group..."
    
    az group create \
        --name "${RESOURCE_GROUP}" \
        --location "${LOCATION}" \
        --tags Environment=production Application=textnlp
    
    success "Resource group created"
}

# Deploy Azure network infrastructure
deploy_network() {
    log "Deploying Azure network infrastructure..."
    
    cd "${CONFIG_DIR}"
    
    # Initialize Terraform
    terraform init
    
    # Plan deployment
    terraform plan \
        -var="subscription_id=${SUBSCRIPTION_ID}" \
        -var="resource_group_name=${RESOURCE_GROUP}" \
        -var="location=${LOCATION}" \
        -out=azure-network.tfplan
    
    # Apply deployment
    terraform apply azure-network.tfplan
    
    success "Azure network infrastructure deployed"
}

# Create Azure Container Registry
create_acr() {
    log "Creating Azure Container Registry..."
    
    az acr create \
        --resource-group "${RESOURCE_GROUP}" \
        --name "${ACR_NAME}" \
        --sku Premium \
        --location "${LOCATION}" \
        --admin-enabled false \
        --public-network-enabled true \
        --allow-trusted-services true
    
    # Enable encryption with customer-managed keys
    az keyvault create \
        --name "textnlp-kv" \
        --resource-group "${RESOURCE_GROUP}" \
        --location "${LOCATION}" \
        --sku standard \
        --enable-purge-protection true \
        --enable-soft-delete true \
        --retention-days 90
    
    # Create encryption key
    az keyvault key create \
        --vault-name "textnlp-kv" \
        --name "container-key" \
        --protection software \
        --size 2048
    
    success "Azure Container Registry created"
}

# Create AKS cluster
deploy_aks() {
    log "Deploying AKS cluster..."
    
    # Get network information from Terraform output
    VNET_NAME=$(terraform output -raw vnet_name)
    SUBNET_ID=$(terraform output -raw aks_subnet_id)
    
    # Create AKS cluster
    az aks create \
        --resource-group "${RESOURCE_GROUP}" \
        --name "${CLUSTER_NAME}" \
        --location "${LOCATION}" \
        --kubernetes-version "1.28.3" \
        --node-count 3 \
        --min-count 1 \
        --max-count 10 \
        --enable-cluster-autoscaler \
        --node-vm-size "Standard_D4s_v3" \
        --node-osdisk-size 100 \
        --node-osdisk-type "Premium_LRS" \
        --network-plugin azure \
        --network-policy azure \
        --vnet-subnet-id "${SUBNET_ID}" \
        --service-cidr "10.96.0.0/12" \
        --dns-service-ip "10.96.0.10" \
        --enable-managed-identity \
        --attach-acr "${ACR_NAME}" \
        --enable-addons monitoring,azure-keyvault-secrets-provider \
        --enable-secret-rotation \
        --enable-workload-identity \
        --enable-oidc-issuer \
        --enable-azure-rbac \
        --enable-defender \
        --tags Environment=production Application=textnlp
    
    # Get cluster credentials
    az aks get-credentials \
        --resource-group "${RESOURCE_GROUP}" \
        --name "${CLUSTER_NAME}" \
        --overwrite-existing
    
    success "AKS cluster deployed"
}

# Create GPU node pool
deploy_gpu_nodepool() {
    log "Creating GPU node pool..."
    
    # Create GPU node pool
    az aks nodepool add \
        --resource-group "${RESOURCE_GROUP}" \
        --cluster-name "${CLUSTER_NAME}" \
        --name "textnlpgpu" \
        --node-count 0 \
        --min-count 0 \
        --max-count 10 \
        --enable-cluster-autoscaler \
        --node-vm-size "Standard_NC4as_T4_v3" \
        --node-osdisk-size 128 \
        --node-osdisk-type "Premium_LRS" \
        --priority Spot \
        --spot-max-price 0.5 \
        --eviction-policy Delete \
        --node-taints "nvidia.com/gpu=true:NoSchedule" \
        --labels node-type=gpu gpu-type=t4 workload=ml \
        --tags Environment=production NodeType=gpu
    
    success "GPU node pool created"
}

# Install NVIDIA GPU drivers
install_gpu_drivers() {
    log "Installing NVIDIA GPU drivers..."
    
    # Apply NVIDIA device plugin
    kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml
    
    # Wait for device plugin to be ready
    kubectl rollout status daemonset nvidia-device-plugin-daemonset -n kube-system --timeout=600s
    
    success "NVIDIA GPU drivers installed"
}

# Install NVIDIA GPU Operator
install_gpu_operator() {
    log "Installing NVIDIA GPU Operator..."
    
    # Add NVIDIA Helm repo
    helm repo add nvidia https://nvidia.github.io/gpu-operator
    helm repo update
    
    # Install GPU Operator with AKS-specific settings
    helm install gpu-operator nvidia/gpu-operator \
        --namespace gpu-operator \
        --create-namespace \
        --set driver.enabled=false \
        --set toolkit.enabled=true \
        --set devicePlugin.enabled=false \
        --set dcgm.enabled=true \
        --set dcgmExporter.enabled=true \
        --set gfd.enabled=true \
        --set migStrategy=single \
        --set nodeStatusExporter.enabled=true \
        --set validator.plugin.enabled=true \
        --set validator.operator.enabled=true \
        --set operator.defaultRuntime=containerd
    
    # Wait for GPU Operator to be ready
    log "Waiting for GPU Operator to be ready..."
    kubectl wait --for=condition=ready pod -l app=nvidia-operator-validator -n gpu-operator --timeout=600s
    
    success "NVIDIA GPU Operator installed"
}

# Configure Azure AD and RBAC
configure_aad_rbac() {
    log "Configuring Azure AD and RBAC..."
    
    # Get cluster OIDC issuer URL
    OIDC_ISSUER=$(az aks show --resource-group "${RESOURCE_GROUP}" --name "${CLUSTER_NAME}" --query "oidcIssuerProfile.issuerUrl" -o tsv)
    
    # Create managed identity for workload identity
    az identity create \
        --name "textnlp-workload-identity" \
        --resource-group "${RESOURCE_GROUP}" \
        --location "${LOCATION}"
    
    # Get managed identity details
    USER_ASSIGNED_CLIENT_ID=$(az identity show --resource-group "${RESOURCE_GROUP}" --name "textnlp-workload-identity" --query 'clientId' -o tsv)
    
    # Create federated identity credential
    az identity federated-credential create \
        --name "textnlp-federated-credential" \
        --identity-name "textnlp-workload-identity" \
        --resource-group "${RESOURCE_GROUP}" \
        --issuer "${OIDC_ISSUER}" \
        --subject "system:serviceaccount:textnlp-gpu:textnlp-gpu-sa"
    
    # Create Azure AD groups
    az ad group create \
        --display-name "TextNLP Platform Architects" \
        --mail-nickname "textnlp-platform-architects" \
        --description "Platform architects for TextNLP"
    
    az ad group create \
        --display-name "TextNLP DevOps Engineers" \
        --mail-nickname "textnlp-devops-engineers" \
        --description "DevOps engineers for TextNLP"
    
    az ad group create \
        --display-name "TextNLP ML Engineers" \
        --mail-nickname "textnlp-ml-engineers" \
        --description "ML engineers for TextNLP"
    
    # Create custom roles
    az role definition create --role-definition '{
        "Name": "TextNLP DevOps Engineer",
        "Description": "Infrastructure and deployment access",
        "Actions": [
            "Microsoft.Compute/*",
            "Microsoft.ContainerService/*",
            "Microsoft.Storage/*",
            "Microsoft.Network/*",
            "Microsoft.Resources/*",
            "Microsoft.Insights/*"
        ],
        "NotActions": [
            "Microsoft.Authorization/*/Delete",
            "Microsoft.Authorization/*/Write"
        ],
        "AssignableScopes": ["/subscriptions/'${SUBSCRIPTION_ID}'/resourceGroups/'${RESOURCE_GROUP}'"]
    }'
    
    # Apply Kubernetes RBAC
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: textnlp-gpu-sa
  namespace: textnlp-gpu
  annotations:
    azure.workload.identity/client-id: "${USER_ASSIGNED_CLIENT_ID}"
  labels:
    azure.workload.identity/use: "true"
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: textnlp-developer
rules:
- apiGroups: ["", "apps", "extensions"]
  resources: ["*"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
- apiGroups: ["batch"]
  resources: ["jobs", "cronjobs"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: textnlp-gpu-developers
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: textnlp-developer
subjects:
- kind: ServiceAccount
  name: textnlp-gpu-sa
  namespace: textnlp-gpu
EOF
    
    success "Azure AD and RBAC configured"
}

# Deploy security configurations
deploy_security() {
    log "Deploying security configurations..."
    
    # Install cert-manager for TLS certificates
    kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.2/cert-manager.yaml
    
    # Wait for cert-manager to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=cert-manager -n cert-manager --timeout=300s
    
    # Install Azure Key Vault CSI driver
    helm repo add csi-secrets-store-provider-azure https://azure.github.io/secrets-store-csi-driver-provider-azure/charts
    helm repo update
    
    helm install csi-secrets-store-provider-azure csi-secrets-store-provider-azure/csi-secrets-store-provider-azure \
        --namespace kube-system
    
    # Apply network policies
    kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: textnlp-gpu
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-api-to-gpu
  namespace: textnlp-gpu
spec:
  podSelector:
    matchLabels:
      app: gpu-worker
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: textnlp-api
    ports:
    - protocol: TCP
      port: 50051
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-dns-egress
  namespace: textnlp-gpu
spec:
  podSelector: {}
  egress:
  - to: []
    ports:
    - protocol: UDP
      port: 53
    - protocol: TCP
      port: 53
EOF
    
    # Create Azure Key Vault secret provider class
    kubectl apply -f - <<EOF
apiVersion: secrets-store.csi.x-k8s.io/v1
kind: SecretProviderClass
metadata:
  name: textnlp-secrets
  namespace: textnlp-gpu
spec:
  provider: azure
  parameters:
    usePodIdentity: "false"
    useVMManagedIdentity: "false"
    userAssignedIdentityID: "${USER_ASSIGNED_CLIENT_ID}"
    keyvaultName: "textnlp-kv"
    cloudName: ""
    objects: |
      array:
        - |
          objectName: container-key
          objectType: key
          objectVersion: ""
    tenantId: "$(az account show --query tenantId -o tsv)"
EOF
    
    success "Security configurations deployed"
}

# Deploy monitoring stack
deploy_monitoring() {
    log "Deploying monitoring stack..."
    
    # Add Prometheus Helm repo
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    # Install Prometheus stack with Azure optimizations
    helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set prometheus.prometheusSpec.retention=30d \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=500Gi \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.storageClassName=managed-premium \
        --set grafana.adminPassword=admin123 \
        --set grafana.persistence.enabled=true \
        --set grafana.persistence.size=20Gi \
        --set grafana.persistence.storageClassName=managed-premium \
        --set alertmanager.alertmanagerSpec.storage.volumeClaimTemplate.spec.resources.requests.storage=50Gi \
        --set alertmanager.alertmanagerSpec.storage.volumeClaimTemplate.spec.storageClassName=managed-premium
    
    # Wait for Prometheus to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=prometheus -n monitoring --timeout=600s
    
    success "Monitoring stack deployed"
}

# Create namespaces
create_namespaces() {
    log "Creating application namespaces..."
    
    kubectl create namespace textnlp-api --dry-run=client -o yaml | kubectl apply -f -
    kubectl create namespace textnlp-gpu --dry-run=client -o yaml | kubectl apply -f -
    kubectl create namespace textnlp-db --dry-run=client -o yaml | kubectl apply -f -
    
    # Label namespaces
    kubectl label namespace textnlp-api app=textnlp tier=api security=standard name=textnlp-api
    kubectl label namespace textnlp-gpu app=textnlp tier=gpu security=restricted name=textnlp-gpu
    kubectl label namespace textnlp-db app=textnlp tier=database security=restricted name=textnlp-db
    
    # Set up resource quotas
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ResourceQuota
metadata:
  name: textnlp-gpu-quota
  namespace: textnlp-gpu
spec:
  hard:
    requests.cpu: "40"
    requests.memory: "256Gi"
    requests.nvidia.com/gpu: "4"
    limits.cpu: "64"
    limits.memory: "512Gi"
    limits.nvidia.com/gpu: "4"
    persistentvolumeclaims: "10"
    pods: "50"
EOF
    
    success "Application namespaces created"
}

# Setup Azure Key Vault for encryption
setup_key_vault() {
    log "Setting up Azure Key Vault for encryption..."
    
    # Create additional encryption keys
    az keyvault key create \
        --vault-name "textnlp-kv" \
        --name "master-encryption-key" \
        --protection software \
        --size 4096 \
        --ops encrypt decrypt sign verify
    
    az keyvault key create \
        --vault-name "textnlp-kv" \
        --name "data-encryption-key" \
        --protection software \
        --size 2048 \
        --ops encrypt decrypt
    
    # Set access policies
    CLUSTER_IDENTITY=$(az aks show --resource-group "${RESOURCE_GROUP}" --name "${CLUSTER_NAME}" --query "identityProfile.kubeletidentity.objectId" -o tsv)
    
    az keyvault set-policy \
        --name "textnlp-kv" \
        --object-id "${CLUSTER_IDENTITY}" \
        --key-permissions get list \
        --secret-permissions get list \
        --certificate-permissions get list
    
    success "Azure Key Vault configured"
}

# Validate deployment
validate_deployment() {
    log "Validating deployment..."
    
    # Check cluster status
    kubectl cluster-info
    
    # Check nodes
    kubectl get nodes -o wide
    
    # Check GPU availability
    kubectl get nodes -o json | jq '.items[].status.capacity | select(.["nvidia.com/gpu"]?) | .["nvidia.com/gpu"]'
    
    # Check pods in system namespaces
    kubectl get pods -n kube-system
    kubectl get pods -n gpu-operator
    kubectl get pods -n monitoring
    
    # Test GPU allocation
    kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
  namespace: textnlp-gpu
spec:
  serviceAccountName: textnlp-gpu-sa
  restartPolicy: Never
  containers:
  - name: gpu-test
    image: nvidia/cuda:11.8-runtime-ubuntu20.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
  nodeSelector:
    node-type: gpu
EOF
    
    # Wait for test pod to complete and check logs
    kubectl wait --for=condition=ready pod/gpu-test -n textnlp-gpu --timeout=300s
    kubectl logs gpu-test -n textnlp-gpu
    kubectl delete pod gpu-test -n textnlp-gpu
    
    # Test Workload Identity
    kubectl run workload-identity-test \
        --image=mcr.microsoft.com/azure-cli:latest \
        --serviceaccount=textnlp-gpu-sa \
        --namespace=textnlp-gpu \
        --rm -it --restart=Never \
        -- az account show
    
    success "Deployment validation completed"
}

# Cleanup function
cleanup() {
    log "Cleaning up test resources..."
    kubectl delete pod gpu-test -n textnlp-gpu --ignore-not-found=true
    kubectl delete pod workload-identity-test -n textnlp-gpu --ignore-not-found=true
}

# Main deployment function
main() {
    log "Starting Azure deployment for TextNLP Platform Phase 2"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    check_prerequisites
    create_resource_group
    create_acr
    setup_key_vault
    create_namespaces
    deploy_network
    deploy_aks
    deploy_gpu_nodepool
    install_gpu_drivers
    install_gpu_operator
    configure_aad_rbac
    deploy_security
    deploy_monitoring
    validate_deployment
    
    success "Azure deployment completed successfully!"
    
    echo ""
    echo "=== Deployment Summary ==="
    echo "AKS Cluster: ${CLUSTER_NAME}"
    echo "Resource Group: ${RESOURCE_GROUP}"
    echo "Location: ${LOCATION}"
    echo "GPU Nodes: Ready with NVIDIA GPU Operator"
    echo "Container Registry: ${ACR_NAME}.azurecr.io"
    echo "Monitoring: Prometheus + Grafana deployed"
    echo "Security: Network policies, Azure RBAC, and Workload Identity configured"
    echo "Encryption: Azure Key Vault encryption keys created"
    echo ""
    echo "Access Information:"
    echo "- Grafana: kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80"
    echo "- Prometheus: kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090"
    echo ""
    echo "Next steps:"
    echo "1. Configure DNS entries for your domain"
    echo "2. Deploy your applications to the created namespaces"
    echo "3. Configure backup and disaster recovery with Azure Backup"
    echo "4. Set up CI/CD pipelines with Azure DevOps or GitHub Actions"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi