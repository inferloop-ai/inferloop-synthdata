#!/bin/bash

# TSIoT Kubernetes Deployment Script
# Deploys the Time Series IoT Synthetic Data Generation Platform to Kubernetes

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
K8S_MANIFESTS_DIR="$PROJECT_ROOT/deployments/kubernetes"
LOG_DIR="$PROJECT_ROOT/logs/deploy"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

# Print usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --environment ENV       Target environment (development|staging|production) [default: development]
    --namespace NAMESPACE   Kubernetes namespace [default: tsiot]
    --version VERSION       Application version to deploy [default: latest]
    --replicas COUNT        Number of replicas [default: 1]
    --wait-timeout DURATION Wait timeout for deployment [default: 10m]
    --context CONTEXT       Kubernetes context to use
    --force                 Force deployment even if resources exist
    --dry-run               Show what would be deployed without executing
    --apply-monitoring      Apply monitoring resources (Prometheus, Grafana)
    --apply-ingress         Apply ingress resources
    --skip-validation       Skip resource validation
    --rolling-update        Perform rolling update
    -v, --verbose           Enable verbose output
    -h, --help              Show this help message

Examples:
    $0                                      # Deploy to development environment
    $0 --environment staging --replicas 3  # Deploy to staging with 3 replicas
    $0 --dry-run --apply-monitoring        # Dry run with monitoring resources
    $0 --rolling-update --version v1.2.3   # Rolling update to specific version

Environment Variables:
    KUBECONFIG              Kubernetes configuration file
    TSIOT_REGISTRY          Container registry URL
    TSIOT_IMAGE_TAG         Container image tag
    KUBECTL_CONTEXT         Default kubectl context
EOF
}

# Parse command line arguments
ENVIRONMENT="development"
NAMESPACE="tsiot"
VERSION="latest"
REPLICAS=1
WAIT_TIMEOUT="10m"
CONTEXT=""
FORCE=false
DRY_RUN=false
APPLY_MONITORING=false
APPLY_INGRESS=false
SKIP_VALIDATION=false
ROLLING_UPDATE=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --replicas)
            REPLICAS="$2"
            shift 2
            ;;
        --wait-timeout)
            WAIT_TIMEOUT="$2"
            shift 2
            ;;
        --context)
            CONTEXT="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --apply-monitoring)
            APPLY_MONITORING=true
            shift
            ;;
        --apply-ingress)
            APPLY_INGRESS=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --rolling-update)
            ROLLING_UPDATE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Set environment variables
export TSIOT_REGISTRY="${TSIOT_REGISTRY:-ghcr.io/inferloop}"
export TSIOT_IMAGE_TAG="${TSIOT_IMAGE_TAG:-$VERSION}"
export KUBECTL_CONTEXT="${KUBECTL_CONTEXT:-$CONTEXT}"

# Create necessary directories
create_directories() {
    mkdir -p "$LOG_DIR"
}

# Setup kubectl context
setup_kubectl_context() {
    if [[ -n "$CONTEXT" ]]; then
        log_info "Using kubectl context: $CONTEXT"
        kubectl config use-context "$CONTEXT"
    fi
    
    # Verify cluster access
    if ! kubectl cluster-info >/dev/null 2>&1; then
        log_error "Cannot access Kubernetes cluster"
        exit 1
    fi
    
    log_info "Connected to cluster: $(kubectl config current-context)"
}

# Create namespace if not exists
create_namespace() {
    log_info "Setting up namespace: $NAMESPACE"
    
    if kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        log_info "Namespace $NAMESPACE already exists"
    else
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "Dry run: Would create namespace $NAMESPACE"
        else
            kubectl create namespace "$NAMESPACE"
            log_success "Created namespace: $NAMESPACE"
        fi
    fi
    
    # Label namespace for monitoring
    if [[ "$DRY_RUN" == "false" ]]; then
        kubectl label namespace "$NAMESPACE" \
            environment="$ENVIRONMENT" \
            app.kubernetes.io/name=tsiot \
            --overwrite
    fi
}

# Apply base resources
apply_base_resources() {
    log_info "Applying base Kubernetes resources..."
    
    local base_dir="$K8S_MANIFESTS_DIR/base"
    
    if [[ ! -d "$base_dir" ]]; then
        log_error "Base manifests directory not found: $base_dir"
        exit 1
    fi
    
    # Process and apply each manifest
    for manifest in "$base_dir"/*.yaml; do
        if [[ -f "$manifest" ]]; then
            apply_manifest "$manifest"
        fi
    done
}

# Apply environment-specific resources
apply_environment_resources() {
    log_info "Applying environment-specific resources for: $ENVIRONMENT"
    
    local overlay_dir="$K8S_MANIFESTS_DIR/overlays/$ENVIRONMENT"
    
    if [[ -d "$overlay_dir" ]]; then
        for manifest in "$overlay_dir"/*.yaml; do
            if [[ -f "$manifest" ]]; then
                apply_manifest "$manifest"
            fi
        done
    else
        log_warning "No environment-specific overlays found for: $ENVIRONMENT"
    fi
}

# Apply monitoring resources
apply_monitoring_resources() {
    if [[ "$APPLY_MONITORING" == "false" ]]; then
        return 0
    fi
    
    log_info "Applying monitoring resources..."
    
    local monitoring_dir="$K8S_MANIFESTS_DIR/monitoring"
    
    if [[ -d "$monitoring_dir" ]]; then
        for manifest in "$monitoring_dir"/*.yaml; do
            if [[ -f "$manifest" ]]; then
                apply_manifest "$manifest"
            fi
        done
    else
        log_warning "Monitoring manifests directory not found: $monitoring_dir"
    fi
}

# Apply ingress resources
apply_ingress_resources() {
    if [[ "$APPLY_INGRESS" == "false" ]]; then
        return 0
    fi
    
    log_info "Applying ingress resources..."
    
    # Check if ingress controller is available
    if ! kubectl get ingressclass >/dev/null 2>&1; then
        log_warning "No ingress controller found, skipping ingress setup"
        return 0
    fi
    
    local base_dir="$K8S_MANIFESTS_DIR/base"
    local ingress_manifest="$base_dir/ingress.yaml"
    
    if [[ -f "$ingress_manifest" ]]; then
        apply_manifest "$ingress_manifest"
    else
        log_warning "Ingress manifest not found: $ingress_manifest"
    fi
}

# Process and apply manifest
apply_manifest() {
    local manifest="$1"
    local processed_manifest
    
    log_info "Processing manifest: $(basename "$manifest")"
    
    # Create a temporary processed manifest
    processed_manifest=$(mktemp)
    
    # Substitute environment variables
    envsubst < "$manifest" > "$processed_manifest"
    
    # Additional processing for dynamic values
    sed -i.bak \
        -e "s/{{NAMESPACE}}/$NAMESPACE/g" \
        -e "s/{{ENVIRONMENT}}/$ENVIRONMENT/g" \
        -e "s/{{VERSION}}/$VERSION/g" \
        -e "s/{{REPLICAS}}/$REPLICAS/g" \
        -e "s|{{REGISTRY}}|$TSIOT_REGISTRY|g" \
        -e "s/{{IMAGE_TAG}}/$TSIOT_IMAGE_TAG/g" \
        "$processed_manifest"
    
    # Validate manifest if not skipping validation
    if [[ "$SKIP_VALIDATION" == "false" ]]; then
        if ! kubectl apply --dry-run=client -f "$processed_manifest" >/dev/null 2>&1; then
            log_error "Manifest validation failed: $(basename "$manifest")"
            rm -f "$processed_manifest" "$processed_manifest.bak"
            exit 1
        fi
    fi
    
    # Apply or dry run
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run: kubectl apply -f $(basename "$manifest")"
        kubectl apply --dry-run=client -f "$processed_manifest"
    else
        if [[ "$VERBOSE" == "true" ]]; then
            kubectl apply -f "$processed_manifest" -v=6
        else
            kubectl apply -f "$processed_manifest"
        fi
    fi
    
    # Cleanup
    rm -f "$processed_manifest" "$processed_manifest.bak"
}

# Update deployment images for rolling update
update_deployment_images() {
    if [[ "$ROLLING_UPDATE" == "false" ]]; then
        return 0
    fi
    
    log_info "Performing rolling update to version: $VERSION"
    
    # Update server deployment
    kubectl set image deployment/tsiot-server \
        server="$TSIOT_REGISTRY/tsiot-server:$TSIOT_IMAGE_TAG" \
        -n "$NAMESPACE"
    
    # Update worker deployment
    kubectl set image deployment/tsiot-worker \
        worker="$TSIOT_REGISTRY/tsiot-worker:$TSIOT_IMAGE_TAG" \
        -n "$NAMESPACE"
    
    # Update CLI deployment (if exists)
    if kubectl get deployment tsiot-cli -n "$NAMESPACE" >/dev/null 2>&1; then
        kubectl set image deployment/tsiot-cli \
            cli="$TSIOT_REGISTRY/tsiot-cli:$TSIOT_IMAGE_TAG" \
            -n "$NAMESPACE"
    fi
}

# Wait for deployments to be ready
wait_for_deployments() {
    log_info "Waiting for deployments to be ready..."
    
    local deployments=("tsiot-server" "tsiot-worker")
    
    for deployment in "${deployments[@]}"; do
        if kubectl get deployment "$deployment" -n "$NAMESPACE" >/dev/null 2>&1; then
            log_info "Waiting for deployment: $deployment"
            
            if [[ "$DRY_RUN" == "false" ]]; then
                kubectl rollout status deployment/"$deployment" \
                    -n "$NAMESPACE" \
                    --timeout="$WAIT_TIMEOUT"
            fi
        fi
    done
}

# Verify deployment health
verify_deployment_health() {
    log_info "Verifying deployment health..."
    
    # Check pod status
    local failed_pods
    failed_pods=$(kubectl get pods -n "$NAMESPACE" \
        --field-selector=status.phase!=Running \
        --no-headers 2>/dev/null | wc -l)
    
    if [[ "$failed_pods" -gt 0 ]]; then
        log_warning "Found $failed_pods pods not in Running state"
        kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running
    fi
    
    # Check service endpoints
    local services
    services=$(kubectl get services -n "$NAMESPACE" -o name 2>/dev/null)
    
    for service in $services; do
        local service_name
        service_name=$(echo "$service" | cut -d'/' -f2)
        
        local endpoints
        endpoints=$(kubectl get endpoints "$service_name" -n "$NAMESPACE" \
            -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null | wc -w)
        
        if [[ "$endpoints" -eq 0 ]]; then
            log_warning "Service $service_name has no endpoints"
        else
            log_info "Service $service_name has $endpoints endpoint(s)"
        fi
    done
}

# Get deployment status
get_deployment_status() {
    log_info "Deployment status:"
    
    echo "Deployments:"
    kubectl get deployments -n "$NAMESPACE" -o wide
    
    echo ""
    echo "Services:"
    kubectl get services -n "$NAMESPACE" -o wide
    
    echo ""
    echo "Pods:"
    kubectl get pods -n "$NAMESPACE" -o wide
    
    if [[ "$APPLY_INGRESS" == "true" ]]; then
        echo ""
        echo "Ingresses:"
        kubectl get ingresses -n "$NAMESPACE" -o wide
    fi
}

# Scale deployment
scale_deployment() {
    if [[ "$REPLICAS" -eq 1 ]]; then
        return 0
    fi
    
    log_info "Scaling deployments to $REPLICAS replicas..."
    
    # Scale server deployment
    if kubectl get deployment tsiot-server -n "$NAMESPACE" >/dev/null 2>&1; then
        kubectl scale deployment tsiot-server --replicas="$REPLICAS" -n "$NAMESPACE"
    fi
    
    # Scale worker deployment
    if kubectl get deployment tsiot-worker -n "$NAMESPACE" >/dev/null 2>&1; then
        kubectl scale deployment tsiot-worker --replicas="$REPLICAS" -n "$NAMESPACE"
    fi
}

# Clean up failed deployments
cleanup_failed_deployments() {
    if [[ "$FORCE" == "false" ]]; then
        return 0
    fi
    
    log_info "Cleaning up failed deployments..."
    
    # Delete failed pods
    kubectl delete pods --field-selector=status.phase=Failed -n "$NAMESPACE" --ignore-not-found=true
    
    # Delete evicted pods
    kubectl delete pods --field-selector=status.phase=Succeeded -n "$NAMESPACE" --ignore-not-found=true
}

# Generate deployment report
generate_deployment_report() {
    log_info "Generating Kubernetes deployment report..."
    
    local report_file="$LOG_DIR/k8s-deployment-report-$(date +%Y%m%d-%H%M%S).json"
    
    cat > "$report_file" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "deployment_type": "kubernetes",
    "cluster": "$(kubectl config current-context)",
    "namespace": "$NAMESPACE",
    "environment": "$ENVIRONMENT",
    "version": "$VERSION",
    "replicas": $REPLICAS,
    "configuration": {
        "registry": "$TSIOT_REGISTRY",
        "image_tag": "$TSIOT_IMAGE_TAG",
        "apply_monitoring": $APPLY_MONITORING,
        "apply_ingress": $APPLY_INGRESS,
        "rolling_update": $ROLLING_UPDATE
    },
    "resources": {
        "deployments": $(kubectl get deployments -n "$NAMESPACE" -o json | jq '.items | length'),
        "services": $(kubectl get services -n "$NAMESPACE" -o json | jq '.items | length'),
        "pods": $(kubectl get pods -n "$NAMESPACE" -o json | jq '.items | length'),
        "configmaps": $(kubectl get configmaps -n "$NAMESPACE" -o json | jq '.items | length'),
        "secrets": $(kubectl get secrets -n "$NAMESPACE" -o json | jq '.items | length')
    }
}
EOF
    
    log_success "Kubernetes deployment report generated: $report_file"
}

# Main execution
main() {
    log_info "Starting Kubernetes deployment"
    log_info "Environment: $ENVIRONMENT"
    log_info "Namespace: $NAMESPACE"
    log_info "Version: $VERSION"
    log_info "Replicas: $REPLICAS"
    
    create_directories
    
    # Verify kubectl is available
    if ! command -v kubectl >/dev/null 2>&1; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Setup kubectl context
    setup_kubectl_context
    
    # Create namespace
    create_namespace
    
    # Clean up if forced
    cleanup_failed_deployments
    
    # Apply resources in order
    apply_base_resources
    apply_environment_resources
    apply_monitoring_resources
    apply_ingress_resources
    
    # Perform rolling update if requested
    update_deployment_images
    
    # Scale deployments
    scale_deployment
    
    # Wait for deployments to be ready
    if [[ "$DRY_RUN" == "false" ]]; then
        wait_for_deployments
        verify_deployment_health
    fi
    
    # Show deployment status
    get_deployment_status
    
    # Generate report
    generate_deployment_report
    
    log_success "Kubernetes deployment completed successfully"
}

# Execute main function
main "$@"