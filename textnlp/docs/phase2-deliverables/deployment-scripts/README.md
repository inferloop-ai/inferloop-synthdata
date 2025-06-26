# TextNLP Platform Phase 2 Deployment Scripts

This directory contains automated deployment scripts for setting up the TextNLP Platform foundation infrastructure across different cloud providers and on-premises environments.

## Overview

The deployment scripts automate the complete setup of:
- Network infrastructure
- Kubernetes clusters with GPU support
- Security configurations (IAM/RBAC, encryption, network policies)
- Monitoring and observability stack
- Storage provisioning
- Load balancing and ingress

## Available Deployment Scripts

### 1. AWS Deployment (`deploy-aws.sh`)

Deploys TextNLP platform on Amazon Web Services with:
- EKS cluster with GPU node groups
- VPC with public/private subnets
- NVIDIA GPU Operator
- IAM roles and policies
- Network security policies
- Prometheus/Grafana monitoring

**Prerequisites:**
- AWS CLI configured with appropriate credentials
- Terraform installed
- kubectl and helm installed

**Usage:**
```bash
sudo chmod +x deploy-aws.sh
./deploy-aws.sh
```

### 2. GCP Deployment (`deploy-gcp.sh`)

Deploys TextNLP platform on Google Cloud Platform with:
- GKE cluster with GPU node pools
- VPC with custom subnets
- NVIDIA GPU drivers and operator
- IAM service accounts with Workload Identity
- Cloud KMS encryption
- Network policies and monitoring

**Prerequisites:**
- gcloud CLI authenticated
- Project created and billing enabled
- Terraform installed
- kubectl and helm installed

**Usage:**
```bash
sudo chmod +x deploy-gcp.sh
./deploy-gcp.sh
```

### 3. Azure Deployment (`deploy-azure.sh`)

Deploys TextNLP platform on Microsoft Azure with:
- AKS cluster with GPU node pools
- Virtual Network with subnets
- Azure Container Registry
- Azure AD integration with RBAC
- Azure Key Vault encryption
- Network security and monitoring

**Prerequisites:**
- Azure CLI authenticated
- Subscription with appropriate permissions
- Terraform installed
- kubectl and helm installed

**Usage:**
```bash
sudo chmod +x deploy-azure.sh
./deploy-azure.sh
```

### 4. On-Premises Deployment (`deploy-onprem.sh`)

Sets up a complete Kubernetes cluster on bare metal/VMs with:
- Kubernetes cluster with kubeadm
- Calico CNI networking
- MetalLB load balancer
- Local and NFS storage
- NVIDIA GPU support
- Self-signed certificates
- Monitoring stack

**Prerequisites:**
- Ubuntu 20.04+ or RHEL 8+ system
- Root access
- Minimum 16GB RAM, 4 CPU cores
- Network connectivity between nodes
- Optional: NVIDIA GPUs for ML workloads

**Usage:**
```bash
sudo chmod +x deploy-onprem.sh
sudo ./deploy-onprem.sh
```

## Configuration Files

Each deployment script references configuration files in the `../network-configs`, `../security-configs`, and `../gpu-configs` directories:

- **Network Configs**: YAML and Terraform files for network infrastructure
- **Security Configs**: IAM/RBAC, encryption, and security policies
- **GPU Configs**: GPU access control and resource management

## Post-Deployment Steps

After running any deployment script, complete these steps:

### 1. Verify Deployment
```bash
kubectl cluster-info
kubectl get nodes -o wide
kubectl get pods --all-namespaces
```

### 2. Test GPU Availability (if applicable)
```bash
kubectl run gpu-test --image=nvidia/cuda:11.8-runtime-ubuntu20.04 --rm -it --restart=Never -- nvidia-smi
```

### 3. Access Monitoring
```bash
# Grafana
kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80

# Prometheus
kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090
```

### 4. Deploy Applications
Deploy your TextNLP applications to the created namespaces:
- `textnlp-api`: API services
- `textnlp-gpu`: GPU-intensive ML workloads
- `textnlp-db`: Database services

## Customization

### Environment Variables
Set these environment variables before running scripts to customize the deployment:

```bash
# AWS
export AWS_REGION="us-west-2"
export CLUSTER_NAME="my-textnlp-cluster"

# GCP
export PROJECT_ID="my-project-123"
export REGION="us-central1"

# Azure
export SUBSCRIPTION_ID="your-subscription-id"
export RESOURCE_GROUP="my-textnlp-rg"

# On-Premises
export MASTER_NODE_IP="192.168.1.100"
export POD_CIDR="172.16.0.0/16"
```

### Resource Sizing
Modify the scripts to adjust resource allocation:
- Node instance types/sizes
- Storage volumes
- Memory and CPU limits
- GPU quotas

### Security Configuration
Customize security settings:
- Network CIDR ranges
- RBAC policies
- Encryption keys
- Certificate authorities

## Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   chmod +x deploy-*.sh
   # For on-premises deployment
   sudo ./deploy-onprem.sh
   ```

2. **Network Connectivity**
   - Ensure firewall rules allow required ports
   - Check DNS resolution
   - Verify subnet routing

3. **GPU Not Detected**
   - Install NVIDIA drivers: `sudo apt install nvidia-driver-525`
   - Restart system after driver installation
   - Verify with `nvidia-smi`

4. **Storage Issues**
   - Check disk space: `df -h`
   - Verify storage class configuration
   - Test PVC creation

### Debug Mode
Enable debug logging:
```bash
export DEBUG=true
./deploy-aws.sh
```

### Log Collection
Collect deployment logs:
```bash
./deploy-aws.sh 2>&1 | tee deployment.log
```

## Security Considerations

### Credentials Management
- Never commit credentials to version control
- Use environment variables or secure credential stores
- Rotate credentials regularly
- Follow principle of least privilege

### Network Security
- All deployments include network policies
- Default deny-all with specific allow rules
- Encrypted communication (TLS/mTLS)
- Private subnets for sensitive workloads

### Access Control
- RBAC configured for all platforms
- Service accounts with minimal permissions
- Multi-factor authentication where available
- Regular access reviews

## Support and Maintenance

### Monitoring
- Prometheus metrics collection
- Grafana dashboards
- Alert manager notifications
- Log aggregation

### Backup and Recovery
- Regular etcd backups (cloud managed or Velero)
- Persistent volume snapshots
- Configuration backup
- Disaster recovery procedures

### Updates
- Kubernetes version updates
- Security patches
- Certificate rotation
- Dependency updates

## Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/)
- [Terraform Documentation](https://developer.hashicorp.com/terraform)
- [Helm Documentation](https://helm.sh/docs/)

For support, please refer to the main TextNLP documentation or contact the platform team.