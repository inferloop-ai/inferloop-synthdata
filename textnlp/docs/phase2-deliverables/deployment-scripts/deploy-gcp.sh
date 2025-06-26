#!/bin/bash
# GCP Deployment Automation Script for TextNLP Platform
# Phase 2: Foundation Setup

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../network-configs"
SECURITY_DIR="${SCRIPT_DIR}/../security-configs"
GPU_DIR="${SCRIPT_DIR}/../gpu-configs"

# GCP Configuration
PROJECT_ID="textnlp-prod-001"
REGION="us-central1"
ZONE="us-central1-a"
CLUSTER_NAME="textnlp-production"

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
    
    # Check gcloud CLI
    if ! command -v gcloud &> /dev/null; then
        error "gcloud CLI is not installed. Please install it first."
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
    
    # Check GCP authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
        error "GCP authentication not configured. Please run 'gcloud auth login' first."
        exit 1
    fi
    
    # Set project
    gcloud config set project "${PROJECT_ID}"
    
    success "Prerequisites check completed"
}

# Enable required GCP APIs
enable_apis() {
    log "Enabling required GCP APIs..."
    
    gcloud services enable \
        compute.googleapis.com \
        container.googleapis.com \
        containerregistry.googleapis.com \
        cloudkms.googleapis.com \
        cloudbuild.googleapis.com \
        monitoring.googleapis.com \
        logging.googleapis.com \
        aiplatform.googleapis.com \
        ml.googleapis.com
    
    success "GCP APIs enabled"
}

# Deploy GCP network infrastructure
deploy_network() {
    log "Deploying GCP network infrastructure..."
    
    cd "${CONFIG_DIR}"
    
    # Initialize Terraform
    terraform init
    
    # Plan deployment
    terraform plan -var="project_id=${PROJECT_ID}" -var="region=${REGION}" -out=gcp-network.tfplan
    
    # Apply deployment
    terraform apply gcp-network.tfplan
    
    success "GCP network infrastructure deployed"
}

# Create GKE cluster
deploy_gke() {
    log "Deploying GKE cluster..."
    
    # Get network and subnet from Terraform output
    NETWORK_NAME=$(terraform output -raw network_name)
    SUBNET_NAME=$(terraform output -raw subnet_name)
    
    # Create GKE cluster
    gcloud container clusters create "${CLUSTER_NAME}" \
        --region="${REGION}" \
        --network="${NETWORK_NAME}" \
        --subnetwork="${SUBNET_NAME}" \
        --cluster-version="1.28.3-gke.1286000" \
        --machine-type="e2-standard-4" \
        --num-nodes=1 \
        --min-nodes=1 \
        --max-nodes=10 \
        --enable-autoscaling \
        --enable-autorepair \
        --enable-autoupgrade \
        --enable-network-policy \
        --enable-ip-alias \
        --enable-shielded-nodes \
        --shielded-secure-boot \
        --shielded-integrity-monitoring \
        --disk-type="pd-ssd" \
        --disk-size="100GB" \
        --image-type="COS_CONTAINERD" \
        --workload-pool="${PROJECT_ID}.svc.id.goog" \
        --enable-cloud-logging \
        --enable-cloud-monitoring \
        --logging=SYSTEM,WORKLOAD \
        --monitoring=SYSTEM \
        --addons=HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver
    
    # Get cluster credentials
    gcloud container clusters get-credentials "${CLUSTER_NAME}" --region="${REGION}"
    
    success "GKE cluster deployed"
}

# Create GPU node pool
deploy_gpu_nodepool() {
    log "Creating GPU node pool..."
    
    # Create GPU node pool
    gcloud container node-pools create "textnlp-gpu-pool" \
        --cluster="${CLUSTER_NAME}" \
        --region="${REGION}" \
        --machine-type="n1-standard-4" \
        --accelerator="type=nvidia-tesla-t4,count=1" \
        --num-nodes=0 \
        --min-nodes=0 \
        --max-nodes=10 \
        --enable-autoscaling \
        --enable-autorepair \
        --enable-autoupgrade \
        --preemptible \
        --disk-type="pd-ssd" \
        --disk-size="100GB" \
        --image-type="COS_CONTAINERD" \
        --node-labels="node-type=gpu,gpu-type=t4" \
        --node-taints="nvidia.com/gpu=true:NoSchedule" \
        --metadata="disable-legacy-endpoints=true" \
        --scopes="https://www.googleapis.com/auth/cloud-platform"
    
    success "GPU node pool created"
}

# Install NVIDIA GPU drivers
install_gpu_drivers() {
    log "Installing NVIDIA GPU drivers..."
    
    # Apply NVIDIA driver DaemonSet
    kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
    
    # Wait for driver installation
    log "Waiting for GPU drivers to be installed..."
    kubectl rollout status daemonset nvidia-driver-installer -n kube-system --timeout=600s
    
    success "NVIDIA GPU drivers installed"
}

# Install NVIDIA GPU Operator
install_gpu_operator() {
    log "Installing NVIDIA GPU Operator..."
    
    # Add NVIDIA Helm repo
    helm repo add nvidia https://nvidia.github.io/gpu-operator
    helm repo update
    
    # Install GPU Operator with GKE-specific settings
    helm install gpu-operator nvidia/gpu-operator \
        --namespace gpu-operator \
        --create-namespace \
        --set driver.enabled=false \
        --set toolkit.enabled=true \
        --set devicePlugin.enabled=true \
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

# Configure IAM and RBAC
configure_iam_rbac() {
    log "Configuring IAM and RBAC..."
    
    # Create service accounts
    gcloud iam service-accounts create textnlp-compute-sa \
        --display-name="TextNLP Compute Service Account" \
        --description="Service account for Compute Engine instances"
    
    gcloud iam service-accounts create textnlp-gke-sa \
        --display-name="TextNLP GKE Service Account" \
        --description="Service account for GKE clusters"
    
    # Grant IAM roles
    gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
        --member="serviceAccount:textnlp-compute-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
        --role="roles/logging.logWriter"
    
    gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
        --member="serviceAccount:textnlp-compute-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
        --role="roles/monitoring.metricWriter"
    
    gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
        --member="serviceAccount:textnlp-gke-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
        --role="roles/container.developer"
    
    # Create Kubernetes service accounts with Workload Identity
    kubectl create serviceaccount textnlp-api-sa -n textnlp-api --dry-run=client -o yaml | kubectl apply -f -
    kubectl create serviceaccount textnlp-gpu-sa -n textnlp-gpu --dry-run=client -o yaml | kubectl apply -f -
    
    # Bind Kubernetes SA to Google SA
    gcloud iam service-accounts add-iam-policy-binding \
        --role roles/iam.workloadIdentityUser \
        --member "serviceAccount:${PROJECT_ID}.svc.id.goog[textnlp-gpu/textnlp-gpu-sa]" \
        textnlp-gke-sa@${PROJECT_ID}.iam.gserviceaccount.com
    
    kubectl annotate serviceaccount textnlp-gpu-sa \
        -n textnlp-gpu \
        iam.gke.io/gcp-service-account=textnlp-gke-sa@${PROJECT_ID}.iam.gserviceaccount.com
    
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
    
    # Create Pod Security Policy
    kubectl apply -f - <<EOF
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: textnlp-restricted
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
EOF
    
    success "Security configurations deployed"
}

# Deploy monitoring stack
deploy_monitoring() {
    log "Deploying monitoring stack..."
    
    # Add Prometheus Helm repo
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    # Install Prometheus stack with GKE optimizations
    helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set prometheus.prometheusSpec.retention=30d \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=500Gi \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.storageClassName=standard-rwo \
        --set grafana.adminPassword=admin123 \
        --set grafana.persistence.enabled=true \
        --set grafana.persistence.size=20Gi \
        --set grafana.persistence.storageClassName=standard-rwo
    
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

# Setup Cloud KMS encryption
setup_kms() {
    log "Setting up Cloud KMS for encryption..."
    
    # Create key ring
    gcloud kms keyrings create textnlp-primary \
        --location="${REGION}" || true
    
    # Create encryption keys
    gcloud kms keys create master-encryption-key \
        --location="${REGION}" \
        --keyring=textnlp-primary \
        --purpose=encryption \
        --rotation-period=90d \
        --next-rotation-time=$(date -d '+90 days' -u +%Y-%m-%dT%H:%M:%SZ) || true
    
    success "Cloud KMS configured"
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
  nodeSelector:
    node-type: gpu
EOF
    
    # Wait for test pod to complete and check logs
    kubectl wait --for=condition=ready pod/gpu-test -n textnlp-gpu --timeout=300s
    kubectl logs gpu-test -n textnlp-gpu
    kubectl delete pod gpu-test -n textnlp-gpu
    
    # Check Workload Identity
    kubectl run workload-identity-test \
        --image=google/cloud-sdk:slim \
        --serviceaccount=textnlp-gpu-sa \
        --namespace=textnlp-gpu \
        --rm -it --restart=Never \
        -- gcloud auth list
    
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
    log "Starting GCP deployment for TextNLP Platform Phase 2"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    check_prerequisites
    enable_apis
    setup_kms
    create_namespaces
    deploy_network
    deploy_gke
    deploy_gpu_nodepool
    install_gpu_drivers
    install_gpu_operator
    configure_iam_rbac
    deploy_security
    deploy_monitoring
    validate_deployment
    
    success "GCP deployment completed successfully!"
    
    echo ""
    echo "=== Deployment Summary ==="
    echo "GKE Cluster: ${CLUSTER_NAME}"
    echo "Region: ${REGION}"
    echo "GPU Nodes: Ready with NVIDIA GPU Operator"
    echo "Monitoring: Prometheus + Grafana deployed"
    echo "Security: Network policies, RBAC, and Workload Identity configured"
    echo "Encryption: Cloud KMS encryption keys created"
    echo ""
    echo "Access Information:"
    echo "- Grafana: kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80"
    echo "- Prometheus: kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090"
    echo ""
    echo "Next steps:"
    echo "1. Configure DNS entries for your domain"
    echo "2. Deploy your applications to the created namespaces"
    echo "3. Configure backup and disaster recovery with GCS"
    echo "4. Set up CI/CD pipelines with Cloud Build"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi