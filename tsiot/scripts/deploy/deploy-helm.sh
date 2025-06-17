#!/bin/bash

# TSIoT Helm Deployment Script
# Deploys the Time Series IoT Synthetic Data Generation Platform using Helm

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HELM_CHART_DIR="$PROJECT_ROOT/deployments/helm"
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
    --release-name NAME     Helm release name [default: tsiot]
    --version VERSION       Application version to deploy [default: latest]
    --chart-version VERSION Helm chart version [default: latest from chart]
    --values-file FILE      Custom values file
    --set KEY=VALUE         Set values (can be used multiple times)
    --replicas COUNT        Number of replicas [default: 1]
    --wait-timeout DURATION Wait timeout for deployment [default: 10m]
    --force                 Force deployment even if release exists
    --dry-run               Show what would be deployed without executing
    --upgrade               Upgrade existing release
    --install               Install new release
    --atomic                Atomic operation (rollback on failure)
    --create-namespace      Create namespace if it doesn't exist
    --dependency-update     Update chart dependencies before deployment
    --verify                Verify chart integrity
    --history               Show release history
    --rollback VERSION      Rollback to specific version
    -v, --verbose           Enable verbose output
    -h, --help              Show this help message

Examples:
    $0                                          # Install/upgrade with default settings
    $0 --environment staging --replicas 3      # Deploy to staging with 3 replicas
    $0 --set server.resources.requests.cpu=500m # Override CPU requests
    $0 --rollback 2                            # Rollback to version 2
    $0 --dry-run --upgrade                     # Dry run upgrade
    $0 --values-file custom-values.yaml        # Use custom values file

Environment Variables:
    HELM_NAMESPACE          Default Helm namespace
    TSIOT_REGISTRY          Container registry URL
    TSIOT_IMAGE_TAG         Container image tag
    HELM_VALUES_FILE        Default values file
EOF
}

# Parse command line arguments
ENVIRONMENT="development"
NAMESPACE="tsiot"
RELEASE_NAME="tsiot"
VERSION="latest"
CHART_VERSION=""
VALUES_FILE=""
SET_VALUES=()
REPLICAS=1
WAIT_TIMEOUT="10m"
FORCE=false
DRY_RUN=false
UPGRADE=false
INSTALL=false
ATOMIC=true
CREATE_NAMESPACE=true
DEPENDENCY_UPDATE=false
VERIFY=false
HISTORY=false
ROLLBACK_VERSION=""
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
        --release-name)
            RELEASE_NAME="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --chart-version)
            CHART_VERSION="$2"
            shift 2
            ;;
        --values-file)
            VALUES_FILE="$2"
            shift 2
            ;;
        --set)
            SET_VALUES+=("$2")
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
        --force)
            FORCE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --upgrade)
            UPGRADE=true
            shift
            ;;
        --install)
            INSTALL=true
            shift
            ;;
        --atomic)
            ATOMIC=true
            shift
            ;;
        --create-namespace)
            CREATE_NAMESPACE=true
            shift
            ;;
        --dependency-update)
            DEPENDENCY_UPDATE=true
            shift
            ;;
        --verify)
            VERIFY=true
            shift
            ;;
        --history)
            HISTORY=true
            shift
            ;;
        --rollback)
            ROLLBACK_VERSION="$2"
            shift 2
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
export HELM_NAMESPACE="${HELM_NAMESPACE:-$NAMESPACE}"
export TSIOT_REGISTRY="${TSIOT_REGISTRY:-ghcr.io/inferloop}"
export TSIOT_IMAGE_TAG="${TSIOT_IMAGE_TAG:-$VERSION}"

# Create necessary directories
create_directories() {
    mkdir -p "$LOG_DIR"
}

# Check Helm and kubectl availability
check_tools() {
    if ! command -v helm >/dev/null 2>&1; then
        log_error "Helm is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v kubectl >/dev/null 2>&1; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check Helm version
    local helm_version
    helm_version=$(helm version --short | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+')
    log_info "Using Helm version: $helm_version"
    
    # Verify cluster access
    if ! kubectl cluster-info >/dev/null 2>&1; then
        log_error "Cannot access Kubernetes cluster"
        exit 1
    fi
    
    log_info "Connected to cluster: $(kubectl config current-context)"
}

# Validate chart
validate_chart() {
    log_info "Validating Helm chart..."
    
    if [[ ! -f "$HELM_CHART_DIR/Chart.yaml" ]]; then
        log_error "Helm chart not found: $HELM_CHART_DIR/Chart.yaml"
        exit 1
    fi
    
    # Lint the chart
    if ! helm lint "$HELM_CHART_DIR" >/dev/null 2>&1; then
        log_error "Helm chart validation failed"
        helm lint "$HELM_CHART_DIR"
        exit 1
    fi
    
    log_success "Helm chart validation passed"
}

# Update chart dependencies
update_dependencies() {
    if [[ "$DEPENDENCY_UPDATE" == "false" ]]; then
        return 0
    fi
    
    log_info "Updating chart dependencies..."
    
    if [[ -f "$HELM_CHART_DIR/Chart.lock" ]]; then
        rm "$HELM_CHART_DIR/Chart.lock"
    fi
    
    cd "$HELM_CHART_DIR"
    helm dependency update
    
    log_success "Chart dependencies updated"
}

# Prepare values file
prepare_values_file() {
    log_info "Preparing values file..."
    
    local values_file
    
    if [[ -n "$VALUES_FILE" ]]; then
        values_file="$VALUES_FILE"
    else
        values_file="$HELM_CHART_DIR/values-$ENVIRONMENT.yaml"
        
        # Fallback to default values if environment-specific file doesn't exist
        if [[ ! -f "$values_file" ]]; then
            values_file="$HELM_CHART_DIR/values.yaml"
        fi
    fi
    
    if [[ ! -f "$values_file" ]]; then
        log_error "Values file not found: $values_file"
        exit 1
    fi
    
    export HELM_VALUES_FILE="$values_file"
    log_info "Using values file: $values_file"
}

# Build Helm command
build_helm_command() {
    local action="$1"
    local cmd=("helm" "$action")
    
    case "$action" in
        "install"|"upgrade")
            cmd+=("$RELEASE_NAME" "$HELM_CHART_DIR")
            
            # Add values file
            if [[ -n "$HELM_VALUES_FILE" ]]; then
                cmd+=("--values" "$HELM_VALUES_FILE")
            fi
            
            # Add set values
            for set_value in "${SET_VALUES[@]}"; do
                cmd+=("--set" "$set_value")
            done
            
            # Add common options
            cmd+=("--namespace" "$NAMESPACE")
            cmd+=("--timeout" "$WAIT_TIMEOUT")
            cmd+=("--wait")
            
            # Add chart version if specified
            if [[ -n "$CHART_VERSION" ]]; then
                cmd+=("--version" "$CHART_VERSION")
            fi
            
            # Add environment-specific options
            cmd+=("--set" "global.environment=$ENVIRONMENT")
            cmd+=("--set" "global.version=$VERSION")
            cmd+=("--set" "global.registry=$TSIOT_REGISTRY")
            cmd+=("--set" "global.imageTag=$TSIOT_IMAGE_TAG")
            cmd+=("--set" "global.replicas=$REPLICAS")
            
            # Add flags
            if [[ "$CREATE_NAMESPACE" == "true" ]]; then
                cmd+=("--create-namespace")
            fi
            
            if [[ "$ATOMIC" == "true" ]]; then
                cmd+=("--atomic")
            fi
            
            if [[ "$DRY_RUN" == "true" ]]; then
                cmd+=("--dry-run")
            fi
            
            if [[ "$FORCE" == "true" ]] && [[ "$action" == "upgrade" ]]; then
                cmd+=("--force")
            fi
            
            if [[ "$VERIFY" == "true" ]]; then
                cmd+=("--verify")
            fi
            ;;
            
        "rollback")
            cmd+=("$RELEASE_NAME" "$ROLLBACK_VERSION")
            cmd+=("--namespace" "$NAMESPACE")
            cmd+=("--timeout" "$WAIT_TIMEOUT")
            cmd+=("--wait")
            
            if [[ "$ATOMIC" == "true" ]]; then
                cmd+=("--atomic")
            fi
            ;;
    esac
    
    echo "${cmd[@]}"
}

# Check if release exists
release_exists() {
    helm list -n "$NAMESPACE" -q | grep -q "^$RELEASE_NAME$"
}

# Show release history
show_history() {
    if [[ "$HISTORY" == "false" ]]; then
        return 0
    fi
    
    log_info "Showing release history for: $RELEASE_NAME"
    
    if release_exists; then
        helm history "$RELEASE_NAME" -n "$NAMESPACE"
    else
        log_warning "Release $RELEASE_NAME does not exist"
    fi
}

# Perform rollback
perform_rollback() {
    if [[ -z "$ROLLBACK_VERSION" ]]; then
        return 0
    fi
    
    log_info "Rolling back release $RELEASE_NAME to version $ROLLBACK_VERSION"
    
    if ! release_exists; then
        log_error "Release $RELEASE_NAME does not exist"
        exit 1
    fi
    
    local cmd
    cmd=$(build_helm_command "rollback")
    
    if [[ "$VERBOSE" == "true" ]]; then
        log_info "Executing: $cmd"
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run: $cmd"
    else
        eval "$cmd"
        log_success "Rollback completed successfully"
    fi
}

# Install or upgrade release
install_or_upgrade_release() {
    if [[ -n "$ROLLBACK_VERSION" ]]; then
        return 0  # Skip if rollback is requested
    fi
    
    local action
    local cmd
    
    # Determine action
    if release_exists; then
        if [[ "$INSTALL" == "true" ]]; then
            log_error "Release $RELEASE_NAME already exists. Use --upgrade or --force"
            exit 1
        fi
        action="upgrade"
    else
        if [[ "$UPGRADE" == "true" ]]; then
            log_error "Release $RELEASE_NAME does not exist. Use --install"
            exit 1
        fi
        action="install"
    fi
    
    log_info "Performing Helm $action for release: $RELEASE_NAME"
    
    cmd=$(build_helm_command "$action")
    
    if [[ "$VERBOSE" == "true" ]]; then
        log_info "Executing: $cmd"
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run: $cmd"
        eval "$cmd"
    else
        eval "$cmd"
        log_success "Helm $action completed successfully"
    fi
}

# Get release status
get_release_status() {
    log_info "Getting release status..."
    
    if release_exists; then
        echo "Release Status:"
        helm status "$RELEASE_NAME" -n "$NAMESPACE"
        
        echo ""
        echo "Release Values:"
        helm get values "$RELEASE_NAME" -n "$NAMESPACE"
        
        echo ""
        echo "Kubernetes Resources:"
        kubectl get all -l "app.kubernetes.io/instance=$RELEASE_NAME" -n "$NAMESPACE"
    else
        log_warning "Release $RELEASE_NAME does not exist"
    fi
}

# Verify deployment
verify_deployment() {
    if [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    
    log_info "Verifying deployment..."
    
    # Check release status
    if ! helm status "$RELEASE_NAME" -n "$NAMESPACE" | grep -q "STATUS: deployed"; then
        log_error "Release is not in deployed status"
        return 1
    fi
    
    # Check pod readiness
    local ready_pods
    ready_pods=$(kubectl get pods -l "app.kubernetes.io/instance=$RELEASE_NAME" -n "$NAMESPACE" \
        --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l)
    
    local total_pods
    total_pods=$(kubectl get pods -l "app.kubernetes.io/instance=$RELEASE_NAME" -n "$NAMESPACE" \
        --no-headers 2>/dev/null | wc -l)
    
    if [[ "$ready_pods" -eq "$total_pods" ]] && [[ "$total_pods" -gt 0 ]]; then
        log_success "All pods are ready ($ready_pods/$total_pods)"
    else
        log_warning "Not all pods are ready ($ready_pods/$total_pods)"
        return 1
    fi
    
    # Check service endpoints
    local services
    services=$(kubectl get services -l "app.kubernetes.io/instance=$RELEASE_NAME" -n "$NAMESPACE" \
        -o name 2>/dev/null)
    
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
    
    log_success "Deployment verification completed"
}

# Generate deployment report
generate_deployment_report() {
    log_info "Generating Helm deployment report..."
    
    local report_file="$LOG_DIR/helm-deployment-report-$(date +%Y%m%d-%H%M%S).json"
    local release_revision=""
    local release_status=""
    
    if release_exists; then
        release_revision=$(helm list -n "$NAMESPACE" -o json | jq -r ".[] | select(.name==\"$RELEASE_NAME\") | .revision")
        release_status=$(helm list -n "$NAMESPACE" -o json | jq -r ".[] | select(.name==\"$RELEASE_NAME\") | .status")
    fi
    
    cat > "$report_file" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "deployment_type": "helm",
    "cluster": "$(kubectl config current-context)",
    "release": {
        "name": "$RELEASE_NAME",
        "namespace": "$NAMESPACE",
        "revision": "$release_revision",
        "status": "$release_status"
    },
    "configuration": {
        "environment": "$ENVIRONMENT",
        "version": "$VERSION",
        "chart_version": "$CHART_VERSION",
        "replicas": $REPLICAS,
        "values_file": "$HELM_VALUES_FILE",
        "set_values": $(printf '%s\n' "${SET_VALUES[@]}" | jq -R . | jq -s .),
        "registry": "$TSIOT_REGISTRY",
        "image_tag": "$TSIOT_IMAGE_TAG"
    },
    "options": {
        "atomic": $ATOMIC,
        "create_namespace": $CREATE_NAMESPACE,
        "dependency_update": $DEPENDENCY_UPDATE,
        "verify": $VERIFY,
        "force": $FORCE,
        "dry_run": $DRY_RUN
    }
}
EOF
    
    log_success "Helm deployment report generated: $report_file"
}

# Cleanup function
cleanup() {
    # Remove temporary files if any
    if [[ -n "${TEMP_VALUES_FILE:-}" ]] && [[ -f "$TEMP_VALUES_FILE" ]]; then
        rm -f "$TEMP_VALUES_FILE"
    fi
}

# Trap cleanup on exit
trap cleanup EXIT

# Main execution
main() {
    log_info "Starting Helm deployment"
    log_info "Release: $RELEASE_NAME"
    log_info "Namespace: $NAMESPACE"
    log_info "Environment: $ENVIRONMENT"
    log_info "Version: $VERSION"
    
    create_directories
    check_tools
    validate_chart
    update_dependencies
    prepare_values_file
    
    # Show history if requested
    show_history
    
    # Perform rollback if requested
    perform_rollback
    
    # Install or upgrade release
    install_or_upgrade_release
    
    # Verify deployment
    verify_deployment
    
    # Show status
    get_release_status
    
    # Generate report
    generate_deployment_report
    
    log_success "Helm deployment completed successfully"
}

# Execute main function
main "$@"