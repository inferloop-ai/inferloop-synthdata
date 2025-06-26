#!/bin/bash
# On-Premises Deployment Automation Script for TextNLP Platform
# Phase 2: Foundation Setup

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../network-configs"
SECURITY_DIR="${SCRIPT_DIR}/../security-configs"
GPU_DIR="${SCRIPT_DIR}/../gpu-configs"

# On-Premises Configuration
CLUSTER_NAME="textnlp-production"
CLUSTER_VERSION="1.28.3"
MASTER_NODE_IP="10.100.1.10"
NODE_CIDR="10.100.1.0/24"
POD_CIDR="10.244.0.0/16"
SERVICE_CIDR="10.96.0.0/12"

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
    
    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root for system configuration"
        exit 1
    fi
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log "Installing kubectl..."
        curl -LO "https://dl.k8s.io/release/v${CLUSTER_VERSION}/bin/linux/amd64/kubectl"
        chmod +x kubectl
        mv kubectl /usr/local/bin/
    fi
    
    # Check Helm
    if ! command -v helm &> /dev/null; then
        log "Installing Helm..."
        curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log "Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sh get-docker.sh
        systemctl enable docker
        systemctl start docker
    fi
    
    # Check system requirements
    local total_mem=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $total_mem -lt 16 ]]; then
        warning "System has less than 16GB RAM. Kubernetes may not perform optimally."
    fi
    
    success "Prerequisites check completed"
}

# Install and configure containerd
install_containerd() {
    log "Installing and configuring containerd..."
    
    # Install containerd
    apt-get update
    apt-get install -y containerd.io
    
    # Configure containerd
    mkdir -p /etc/containerd
    containerd config default > /etc/containerd/config.toml
    
    # Enable systemd cgroup driver
    sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml
    
    # Restart containerd
    systemctl restart containerd
    systemctl enable containerd
    
    success "containerd installed and configured"
}

# Disable swap and configure system
configure_system() {
    log "Configuring system for Kubernetes..."
    
    # Disable swap
    swapoff -a
    sed -i '/ swap / s/^/#/' /etc/fstab
    
    # Load required kernel modules
    cat <<EOF > /etc/modules-load.d/k8s.conf
overlay
br_netfilter
EOF
    
    modprobe overlay
    modprobe br_netfilter
    
    # Configure sysctl
    cat <<EOF > /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF
    
    sysctl --system
    
    success "System configured for Kubernetes"
}

# Install kubeadm, kubelet, and kubectl
install_kubernetes() {
    log "Installing Kubernetes components..."
    
    # Add Kubernetes repository
    apt-get update
    apt-get install -y apt-transport-https ca-certificates curl gpg
    
    curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.28/deb/Release.key | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
    echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.28/deb/ /' > /etc/apt/sources.list.d/kubernetes.list
    
    # Install Kubernetes components
    apt-get update
    apt-get install -y kubelet kubeadm kubectl
    apt-mark hold kubelet kubeadm kubectl
    
    # Configure kubelet
    cat <<EOF > /etc/default/kubelet
KUBELET_EXTRA_ARGS=--cgroup-driver=systemd
EOF
    
    systemctl enable kubelet
    
    success "Kubernetes components installed"
}

# Initialize Kubernetes cluster
init_cluster() {
    log "Initializing Kubernetes cluster..."
    
    # Create kubeadm config
    cat <<EOF > /tmp/kubeadm-config.yaml
apiVersion: kubeadm.k8s.io/v1beta3
kind: ClusterConfiguration
kubernetesVersion: v${CLUSTER_VERSION}
clusterName: ${CLUSTER_NAME}
controlPlaneEndpoint: ${MASTER_NODE_IP}:6443
networking:
  podSubnet: ${POD_CIDR}
  serviceSubnet: ${SERVICE_CIDR}
etcd:
  local:
    serverCertSANs:
    - ${MASTER_NODE_IP}
    peerCertSANs:
    - ${MASTER_NODE_IP}
apiServer:
  certSANs:
  - ${MASTER_NODE_IP}
  extraArgs:
    encryption-provider-config: /etc/kubernetes/encryption-config.yaml
controllerManager:
  extraArgs:
    bind-address: 0.0.0.0
scheduler:
  extraArgs:
    bind-address: 0.0.0.0
---
apiVersion: kubeadm.k8s.io/v1beta3
kind: InitConfiguration
localAPIEndpoint:
  advertiseAddress: ${MASTER_NODE_IP}
  bindPort: 6443
---
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
cgroupDriver: systemd
containerRuntimeEndpoint: unix:///var/run/containerd/containerd.sock
EOF
    
    # Create encryption config for etcd
    cat <<EOF > /etc/kubernetes/encryption-config.yaml
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
  - resources:
      - secrets
    providers:
      - aescbc:
          keys:
            - name: key1
              secret: $(head -c 32 /dev/urandom | base64)
      - identity: {}
EOF
    
    # Initialize cluster
    kubeadm init --config=/tmp/kubeadm-config.yaml --upload-certs
    
    # Configure kubectl for root user
    mkdir -p /root/.kube
    cp -i /etc/kubernetes/admin.conf /root/.kube/config
    chown root:root /root/.kube/config
    
    # Configure kubectl for regular users
    mkdir -p /home/*/. kube
    cp -i /etc/kubernetes/admin.conf /home/*/.kube/config
    chown -R $(stat -c '%U:%G' /home/*) /home/*/.kube/ 2>/dev/null || true
    
    success "Kubernetes cluster initialized"
}

# Install Calico CNI
install_calico() {
    log "Installing Calico CNI..."
    
    # Download Calico manifest
    curl -O https://raw.githubusercontent.com/projectcalico/calico/v3.26.3/manifests/tigera-operator.yaml
    curl -O https://raw.githubusercontent.com/projectcalico/calico/v3.26.3/manifests/custom-resources.yaml
    
    # Apply Calico operator
    kubectl apply -f tigera-operator.yaml
    
    # Configure Calico with custom CIDR
    sed -i "s|cidr: 192.168.0.0/16|cidr: ${POD_CIDR}|g" custom-resources.yaml
    
    # Apply Calico configuration
    kubectl apply -f custom-resources.yaml
    
    # Wait for Calico to be ready
    kubectl wait --for=condition=ready pod -l k8s-app=calico-node -n calico-system --timeout=600s
    
    success "Calico CNI installed"
}

# Remove master node taint (for single-node testing)
remove_master_taint() {
    log "Removing master node taint for workload scheduling..."
    
    kubectl taint nodes --all node-role.kubernetes.io/control-plane- || true
    kubectl taint nodes --all node-role.kubernetes.io/master- || true
    
    success "Master node taint removed"
}

# Install MetalLB for load balancing
install_metallb() {
    log "Installing MetalLB load balancer..."
    
    # Install MetalLB
    kubectl apply -f https://raw.githubusercontent.com/metallb/metallb/v0.13.12/config/manifests/metallb-native.yaml
    
    # Wait for MetalLB to be ready
    kubectl wait --for=condition=ready pod -l app=metallb -n metallb-system --timeout=600s
    
    # Configure address pool
    kubectl apply -f - <<EOF
apiVersion: metallb.io/v1beta1
kind: IPAddressPool
metadata:
  name: textnlp-pool
  namespace: metallb-system
spec:
  addresses:
  - 10.100.1.100-10.100.1.200
---
apiVersion: metallb.io/v1beta1
kind: L2Advertisement
metadata:
  name: l2-advertisement
  namespace: metallb-system
spec:
  ipAddressPools:
  - textnlp-pool
EOF
    
    success "MetalLB installed and configured"
}

# Install NGINX Ingress Controller
install_nginx_ingress() {
    log "Installing NGINX Ingress Controller..."
    
    # Add NGINX Helm repo
    helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
    helm repo update
    
    # Install NGINX Ingress
    helm install ingress-nginx ingress-nginx/ingress-nginx \
        --namespace ingress-nginx \
        --create-namespace \
        --set controller.service.type=LoadBalancer \
        --set controller.service.loadBalancerIP=10.100.1.100 \
        --set controller.metrics.enabled=true \
        --set controller.metrics.serviceMonitor.enabled=true
    
    # Wait for ingress controller to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=ingress-nginx -n ingress-nginx --timeout=600s
    
    success "NGINX Ingress Controller installed"
}

# Setup GPU support
setup_gpu_support() {
    log "Setting up GPU support..."
    
    # Check if NVIDIA GPUs are present
    if ! lspci | grep -i nvidia &> /dev/null; then
        warning "No NVIDIA GPUs detected. Skipping GPU setup."
        return 0
    fi
    
    # Install NVIDIA drivers (if not already installed)
    if ! command -v nvidia-smi &> /dev/null; then
        log "Installing NVIDIA drivers..."
        apt-get update
        apt-get install -y nvidia-driver-525 nvidia-utils-525
        
        # Restart may be required for driver installation
        warning "NVIDIA drivers installed. System reboot may be required."
    fi
    
    # Install NVIDIA Container Toolkit
    if ! command -v nvidia-container-runtime &> /dev/null; then
        log "Installing NVIDIA Container Toolkit..."
        
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | apt-key add -
        curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        
        apt-get update
        apt-get install -y nvidia-container-toolkit
        
        # Configure containerd
        nvidia-ctk runtime configure --runtime=containerd
        systemctl restart containerd
    fi
    
    # Install NVIDIA GPU Operator
    helm repo add nvidia https://nvidia.github.io/gpu-operator
    helm repo update
    
    helm install gpu-operator nvidia/gpu-operator \
        --namespace gpu-operator \
        --create-namespace \
        --set driver.enabled=false \
        --set toolkit.enabled=false \
        --set devicePlugin.enabled=true \
        --set dcgm.enabled=true \
        --set dcgmExporter.enabled=true \
        --set gfd.enabled=true \
        --set migStrategy=single \
        --set nodeStatusExporter.enabled=true \
        --set validator.plugin.enabled=true \
        --set validator.operator.enabled=true
    
    # Wait for GPU Operator to be ready
    kubectl wait --for=condition=ready pod -l app=nvidia-operator-validator -n gpu-operator --timeout=600s
    
    success "GPU support configured"
}

# Install and configure local storage
setup_local_storage() {
    log "Setting up local storage..."
    
    # Install local path provisioner
    kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.24/deploy/local-path-storage.yaml
    
    # Set as default storage class
    kubectl patch storageclass local-path -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
    
    # Create fast-ssd storage class for high-performance workloads
    kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
  annotations:
    storageclass.kubernetes.io/is-default-class: "false"
provisioner: rancher.io/local-path
parameters:
  nodePath: /opt/local-path-provisioner
volumeBindingMode: WaitForFirstConsumer
reclaimPolicy: Delete
EOF
    
    # Create directories for local storage
    mkdir -p /opt/local-path-provisioner
    chmod 755 /opt/local-path-provisioner
    
    success "Local storage configured"
}

# Setup NFS for shared storage
setup_nfs_storage() {
    log "Setting up NFS storage..."
    
    # Install NFS server
    apt-get update
    apt-get install -y nfs-kernel-server
    
    # Create NFS export directories
    mkdir -p /storage/kubernetes
    mkdir -p /storage/models
    mkdir -p /storage/datasets
    
    # Configure NFS exports
    cat <<EOF >> /etc/exports
/storage/kubernetes 10.100.1.0/24(rw,sync,no_subtree_check,no_root_squash)
/storage/models 10.100.1.0/24(rw,sync,no_subtree_check,no_root_squash)
/storage/datasets 10.100.1.0/24(rw,sync,no_subtree_check,no_root_squash)
EOF
    
    # Export NFS shares
    exportfs -a
    systemctl enable nfs-kernel-server
    systemctl start nfs-kernel-server
    
    # Install NFS CSI driver
    helm repo add csi-driver-nfs https://raw.githubusercontent.com/kubernetes-csi/csi-driver-nfs/master/charts
    helm repo update
    
    helm install csi-driver-nfs csi-driver-nfs/csi-driver-nfs \
        --namespace kube-system
    
    # Create NFS storage class
    kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: nfs-shared
provisioner: nfs.csi.k8s.io
parameters:
  server: ${MASTER_NODE_IP}
  share: /storage/kubernetes
  mountPermissions: "755"
reclaimPolicy: Delete
volumeBindingMode: Immediate
allowVolumeExpansion: true
EOF
    
    success "NFS storage configured"
}

# Install monitoring stack
install_monitoring() {
    log "Installing monitoring stack..."
    
    # Add Prometheus Helm repo
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    # Install Prometheus stack
    helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set prometheus.prometheusSpec.retention=30d \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=100Gi \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.storageClassName=local-path \
        --set grafana.adminPassword=admin123 \
        --set grafana.persistence.enabled=true \
        --set grafana.persistence.size=10Gi \
        --set grafana.persistence.storageClassName=local-path \
        --set alertmanager.alertmanagerSpec.storage.volumeClaimTemplate.spec.resources.requests.storage=10Gi \
        --set alertmanager.alertmanagerSpec.storage.volumeClaimTemplate.spec.storageClassName=local-path
    
    # Wait for Prometheus to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=prometheus -n monitoring --timeout=600s
    
    success "Monitoring stack installed"
}

# Configure security
configure_security() {
    log "Configuring security..."
    
    # Install cert-manager
    kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.2/cert-manager.yaml
    
    # Wait for cert-manager to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=cert-manager -n cert-manager --timeout=300s
    
    # Create self-signed CA issuer
    kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: selfsigned-issuer
spec:
  selfSigned: {}
---
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: selfsigned-ca
  namespace: cert-manager
spec:
  isCA: true
  commonName: textnlp-ca
  secretName: root-secret
  privateKey:
    algorithm: ECDSA
    size: 256
  issuerRef:
    name: selfsigned-issuer
    kind: ClusterIssuer
    group: cert-manager.io
---
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: ca-issuer
spec:
  ca:
    secretName: root-secret
EOF
    
    # Enable Pod Security Standards
    kubectl label namespace kube-system pod-security.kubernetes.io/enforce=privileged
    kubectl label namespace kube-public pod-security.kubernetes.io/enforce=baseline
    kubectl label namespace default pod-security.kubernetes.io/enforce=restricted
    
    success "Security configured"
}

# Create application namespaces
create_namespaces() {
    log "Creating application namespaces..."
    
    kubectl create namespace textnlp-api --dry-run=client -o yaml | kubectl apply -f -
    kubectl create namespace textnlp-gpu --dry-run=client -o yaml | kubectl apply -f -
    kubectl create namespace textnlp-db --dry-run=client -o yaml | kubectl apply -f -
    
    # Label namespaces
    kubectl label namespace textnlp-api app=textnlp tier=api security=standard
    kubectl label namespace textnlp-gpu app=textnlp tier=gpu security=restricted
    kubectl label namespace textnlp-db app=textnlp tier=database security=restricted
    
    # Apply Pod Security Standards
    kubectl label namespace textnlp-api pod-security.kubernetes.io/enforce=baseline
    kubectl label namespace textnlp-gpu pod-security.kubernetes.io/enforce=restricted
    kubectl label namespace textnlp-db pod-security.kubernetes.io/enforce=restricted
    
    # Create resource quotas
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ResourceQuota
metadata:
  name: textnlp-gpu-quota
  namespace: textnlp-gpu
spec:
  hard:
    requests.cpu: "20"
    requests.memory: "128Gi"
    requests.nvidia.com/gpu: "2"
    limits.cpu: "32"
    limits.memory: "256Gi"
    limits.nvidia.com/gpu: "2"
    persistentvolumeclaims: "10"
    pods: "50"
EOF
    
    success "Application namespaces created"
}

# Validate deployment
validate_deployment() {
    log "Validating deployment..."
    
    # Check cluster status
    kubectl cluster-info
    
    # Check nodes
    kubectl get nodes -o wide
    
    # Check system pods
    kubectl get pods -n kube-system
    kubectl get pods -n calico-system
    kubectl get pods -n metallb-system
    kubectl get pods -n ingress-nginx
    kubectl get pods -n monitoring
    
    # Check storage classes
    kubectl get storageclass
    
    # Check GPU availability (if present)
    if kubectl get nodes -o json | jq '.items[].status.capacity | select(.["nvidia.com/gpu"]?)' | grep -q nvidia; then
        log "GPU nodes detected"
        kubectl get nodes -o json | jq '.items[].status.capacity | select(.["nvidia.com/gpu"]?) | .["nvidia.com/gpu"]'
        
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
EOF
        
        kubectl wait --for=condition=ready pod/gpu-test -n textnlp-gpu --timeout=300s
        kubectl logs gpu-test -n textnlp-gpu
        kubectl delete pod gpu-test -n textnlp-gpu
    fi
    
    # Test ingress controller
    curl -I http://10.100.1.100 || warning "Ingress controller not responding"
    
    success "Deployment validation completed"
}

# Generate join command for additional nodes
generate_join_command() {
    log "Generating join command for worker nodes..."
    
    echo ""
    echo "=== Worker Node Join Command ==="
    kubeadm token create --print-join-command
    echo ""
    
    success "Join command generated"
}

# Main deployment function
main() {
    log "Starting on-premises deployment for TextNLP Platform Phase 2"
    
    check_prerequisites
    configure_system
    install_containerd
    install_kubernetes
    init_cluster
    install_calico
    remove_master_taint
    install_metallb
    install_nginx_ingress
    setup_gpu_support
    setup_local_storage
    setup_nfs_storage
    install_monitoring
    configure_security
    create_namespaces
    validate_deployment
    generate_join_command
    
    success "On-premises deployment completed successfully!"
    
    echo ""
    echo "=== Deployment Summary ==="
    echo "Kubernetes Cluster: ${CLUSTER_NAME}"
    echo "Master Node IP: ${MASTER_NODE_IP}"
    echo "Pod CIDR: ${POD_CIDR}"
    echo "Service CIDR: ${SERVICE_CIDR}"
    echo "Load Balancer IP Range: 10.100.1.100-10.100.1.200"
    echo "GPU Support: $(lspci | grep -i nvidia > /dev/null && echo "Enabled" || echo "Not Available")"
    echo "Storage: Local Path Provisioner + NFS"
    echo "Monitoring: Prometheus + Grafana"
    echo ""
    echo "Access Information:"
    echo "- Cluster: kubectl cluster-info"
    echo "- Grafana: kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80"
    echo "- Prometheus: kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090"
    echo ""
    echo "Next steps:"
    echo "1. Add worker nodes using the join command above"
    echo "2. Configure DNS entries for your domain"
    echo "3. Deploy your applications to the created namespaces"
    echo "4. Configure backup and disaster recovery"
    echo "5. Set up CI/CD pipelines"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi