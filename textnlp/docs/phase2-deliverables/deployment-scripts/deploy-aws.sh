#!/bin/bash
# AWS Deployment Automation Script for TextNLP Platform
# Phase 2: Foundation Setup

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../network-configs"
SECURITY_DIR="${SCRIPT_DIR}/../security-configs"
GPU_DIR="${SCRIPT_DIR}/../gpu-configs"

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
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        error "AWS CLI is not installed. Please install it first."
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
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS credentials not configured. Please run 'aws configure' first."
        exit 1
    fi
    
    success "Prerequisites check completed"
}

# Deploy AWS network infrastructure
deploy_network() {
    log "Deploying AWS network infrastructure..."
    
    cd "${CONFIG_DIR}"
    
    # Initialize Terraform
    terraform init
    
    # Plan deployment
    terraform plan -var-file="aws-network-vars.tfvars" -out=aws-network.tfplan
    
    # Apply deployment
    terraform apply aws-network.tfplan
    
    success "AWS network infrastructure deployed"
}

# Deploy EKS cluster
deploy_eks() {
    log "Deploying EKS cluster..."
    
    # Get VPC and subnet IDs from Terraform output
    VPC_ID=$(terraform output -raw vpc_id)
    PRIVATE_SUBNET_IDS=$(terraform output -json private_subnet_ids | jq -r '.[]' | tr '\n' ',' | sed 's/,$//')
    PUBLIC_SUBNET_IDS=$(terraform output -json public_subnet_ids | jq -r '.[]' | tr '\n' ',' | sed 's/,$//')
    
    # Create EKS cluster
    aws eks create-cluster \
        --name textnlp-production \
        --version 1.28 \
        --role-arn "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/TextNLP-EKS-Service-Role" \
        --resources-vpc-config subnetIds="${PRIVATE_SUBNET_IDS},${PUBLIC_SUBNET_IDS}",endpointConfigPrivateAccess=true,endpointConfigPublicAccess=true
    
    # Wait for cluster to be active
    log "Waiting for EKS cluster to be active..."
    aws eks wait cluster-active --name textnlp-production
    
    # Update kubeconfig
    aws eks update-kubeconfig --name textnlp-production
    
    success "EKS cluster deployed"
}

# Deploy GPU node groups
deploy_gpu_nodes() {
    log "Deploying GPU node groups..."
    
    # Create GPU node group
    aws eks create-nodegroup \
        --cluster-name textnlp-production \
        --nodegroup-name textnlp-gpu-workers \
        --scaling-config minSize=1,maxSize=10,desiredSize=2 \
        --disk-size 100 \
        --instance-types g4dn.xlarge,g4dn.2xlarge,p3.2xlarge \
        --ami-type AL2_x86_64_GPU \
        --capacity-type SPOT \
        --node-role "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/TextNLP-EKS-NodeGroup-Role" \
        --subnets $(terraform output -json private_subnet_ids | jq -r '.[]' | tr '\n' ' ') \
        --labels nodeType=gpu,workload=ml,gpuType=nvidia \
        --taints key=nvidia.com/gpu,value=true,effect=NO_SCHEDULE
    
    # Wait for node group to be active
    log "Waiting for GPU node group to be active..."
    aws eks wait nodegroup-active --cluster-name textnlp-production --nodegroup-name textnlp-gpu-workers
    
    success "GPU node groups deployed"
}

# Install NVIDIA GPU Operator
install_gpu_operator() {
    log "Installing NVIDIA GPU Operator..."
    
    # Add NVIDIA Helm repo
    helm repo add nvidia https://nvidia.github.io/gpu-operator
    helm repo update
    
    # Install GPU Operator
    helm install gpu-operator nvidia/gpu-operator \
        --namespace gpu-operator \
        --create-namespace \
        --set driver.enabled=true \
        --set toolkit.enabled=true \
        --set devicePlugin.enabled=true \
        --set dcgm.enabled=true \
        --set dcgmExporter.enabled=true \
        --set gfd.enabled=true \
        --set migStrategy=single \
        --set nodeStatusExporter.enabled=true \
        --set validator.plugin.enabled=true \
        --set validator.operator.enabled=true
    
    # Wait for GPU Operator to be ready
    log "Waiting for GPU Operator to be ready..."
    kubectl wait --for=condition=ready pod -l app=nvidia-operator-validator -n gpu-operator --timeout=600s
    
    success "NVIDIA GPU Operator installed"
}

# Configure IAM and RBAC
configure_iam_rbac() {
    log "Configuring IAM and RBAC..."
    
    # Apply IAM policies (these would be created via Terraform in a real deployment)
    # For now, we'll create the basic RBAC configurations
    
    # Create service accounts
    kubectl create serviceaccount textnlp-api-sa -n textnlp-api --dry-run=client -o yaml | kubectl apply -f -
    kubectl create serviceaccount textnlp-gpu-sa -n textnlp-gpu --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply RBAC configurations
    kubectl apply -f - <<EOF
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: textnlp-developer
rules:
- apiGroups: ["", "apps", "extensions"]
  resources: ["*"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
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
    
    success "IAM and RBAC configured"
}

# Deploy security configurations
deploy_security() {
    log "Deploying security configurations..."
    
    # Install cert-manager for TLS certificates
    kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.2/cert-manager.yaml
    
    # Wait for cert-manager to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=cert-manager -n cert-manager --timeout=300s
    
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
EOF
    
    success "Security configurations deployed"
}

# Deploy monitoring stack
deploy_monitoring() {
    log "Deploying monitoring stack..."
    
    # Add Prometheus Helm repo
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    # Install Prometheus stack
    helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set prometheus.prometheusSpec.retention=30d \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=500Gi \
        --set grafana.adminPassword=admin123
    
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
    kubectl label namespace textnlp-api app=textnlp tier=api security=standard
    kubectl label namespace textnlp-gpu app=textnlp tier=gpu security=restricted
    kubectl label namespace textnlp-db app=textnlp tier=database security=restricted
    
    success "Application namespaces created"
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
EOF
    
    # Wait for test pod to complete
    kubectl wait --for=condition=ready pod/gpu-test -n textnlp-gpu --timeout=300s
    kubectl logs gpu-test -n textnlp-gpu
    kubectl delete pod gpu-test -n textnlp-gpu
    
    success "Deployment validation completed"
}

# Cleanup function
cleanup() {
    log "Cleaning up test resources..."
    kubectl delete pod gpu-test -n textnlp-gpu --ignore-not-found=true
}

# Main deployment function
main() {
    log "Starting AWS deployment for TextNLP Platform Phase 2"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    check_prerequisites
    create_namespaces
    deploy_network
    deploy_eks
    deploy_gpu_nodes
    install_gpu_operator
    configure_iam_rbac
    deploy_security
    deploy_monitoring
    validate_deployment
    
    success "AWS deployment completed successfully!"
    
    echo ""
    echo "=== Deployment Summary ==="
    echo "EKS Cluster: textnlp-production"
    echo "GPU Nodes: Ready with NVIDIA GPU Operator"
    echo "Monitoring: Prometheus + Grafana deployed"
    echo "Security: Network policies and RBAC configured"
    echo ""
    echo "Next steps:"
    echo "1. Configure DNS entries for your domain"
    echo "2. Deploy your applications to the created namespaces"
    echo "3. Configure backup and disaster recovery"
    echo "4. Set up CI/CD pipelines"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi