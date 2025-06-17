#!/bin/bash

# TSIoT Rollback Script
# Performs rollback operations for the Time Series IoT Synthetic Data Generation Platform

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
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
    --method METHOD         Deployment method (kubernetes|helm|docker-compose|docker) [default: auto-detect]
    --environment ENV       Target environment (development|staging|production) [default: development]
    --namespace NAMESPACE   Kubernetes namespace [default: tsiot]
    --release-name NAME     Helm release name [default: tsiot]
    --version VERSION       Target version to rollback to
    --revision REVISION     Target revision to rollback to (for Helm/K8s)
    --steps STEPS           Number of steps to rollback [default: 1]
    --confirm               Skip confirmation prompt
    --list-history          List deployment history
    --check-status          Check current deployment status
    --backup-data           Backup data before rollback
    --restore-data          Restore data after rollback
    --wait-timeout DURATION Wait timeout for rollback [default: 10m]
    --force                 Force rollback even if health checks fail
    --dry-run               Show what would be rolled back without executing
    -v, --verbose           Enable verbose output
    -h, --help              Show this help message

Examples:
    $0                                          # Auto-detect and rollback 1 step
    $0 --method helm --revision 2              # Rollback Helm release to revision 2
    $0 --method kubernetes --steps 2           # Rollback K8s deployment 2 steps
    $0 --version v1.2.3 --backup-data         # Rollback to specific version with data backup
    $0 --list-history                          # List deployment history
    $0 --check-status                          # Check current status

Environment Variables:
    TSIOT_DEPLOY_METHOD     Default deployment method
    KUBECONFIG              Kubernetes configuration file
    HELM_NAMESPACE          Helm deployment namespace
EOF
}

# Parse command line arguments
METHOD=""
ENVIRONMENT="development"
NAMESPACE="tsiot"
RELEASE_NAME="tsiot"
VERSION=""
REVISION=""
STEPS=1
CONFIRM=false
LIST_HISTORY=false
CHECK_STATUS=false
BACKUP_DATA=false
RESTORE_DATA=false
WAIT_TIMEOUT="10m"
FORCE=false
DRY_RUN=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            METHOD="$2"
            shift 2
            ;;
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
        --revision)
            REVISION="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --confirm)
            CONFIRM=true
            shift
            ;;
        --list-history)
            LIST_HISTORY=true
            shift
            ;;
        --check-status)
            CHECK_STATUS=true
            shift
            ;;
        --backup-data)
            BACKUP_DATA=true
            shift
            ;;
        --restore-data)
            RESTORE_DATA=true
            shift
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
export TSIOT_DEPLOY_METHOD="${TSIOT_DEPLOY_METHOD:-$METHOD}"
export HELM_NAMESPACE="${HELM_NAMESPACE:-$NAMESPACE}"

# Create necessary directories
create_directories() {
    mkdir -p "$LOG_DIR"
}

# Auto-detect deployment method
auto_detect_method() {
    if [[ -n "$METHOD" ]]; then
        return 0
    fi
    
    log_info "Auto-detecting deployment method..."
    
    # Check for Helm release
    if command -v helm >/dev/null 2>&1; then
        if helm list -n "$NAMESPACE" | grep -q "$RELEASE_NAME"; then
            METHOD="helm"
            log_info "Detected Helm deployment"
            return 0
        fi
    fi
    
    # Check for Kubernetes deployment
    if command -v kubectl >/dev/null 2>&1; then
        if kubectl get deployment -n "$NAMESPACE" | grep -q "tsiot"; then
            METHOD="kubernetes"
            log_info "Detected Kubernetes deployment"
            return 0
        fi
    fi
    
    # Check for Docker Compose
    if command -v docker-compose >/dev/null 2>&1; then
        if [[ -f "$PROJECT_ROOT/deployments/docker/docker-compose.yml" ]]; then
            cd "$PROJECT_ROOT/deployments/docker"
            if docker-compose ps | grep -q "tsiot"; then
                METHOD="docker-compose"
                log_info "Detected Docker Compose deployment"
                return 0
            fi
        fi
    fi
    
    # Check for Docker containers
    if command -v docker >/dev/null 2>&1; then
        if docker ps | grep -q "tsiot"; then
            METHOD="docker"
            log_info "Detected Docker deployment"
            return 0
        fi
    fi
    
    log_error "Could not auto-detect deployment method"
    log_error "Please specify --method explicitly"
    exit 1
}

# Check current deployment status
check_deployment_status() {
    log_info "Checking current deployment status..."
    
    case "$METHOD" in
        "helm")
            check_helm_status
            ;;
        "kubernetes")
            check_kubernetes_status
            ;;
        "docker-compose")
            check_docker_compose_status
            ;;
        "docker")
            check_docker_status
            ;;
    esac
}

# Check Helm status
check_helm_status() {
    if ! command -v helm >/dev/null 2>&1; then
        log_error "Helm is not installed"
        exit 1
    fi
    
    if helm list -n "$NAMESPACE" | grep -q "$RELEASE_NAME"; then
        echo "Helm Release Status:"
        helm status "$RELEASE_NAME" -n "$NAMESPACE"
        
        echo ""
        echo "Release History:"
        helm history "$RELEASE_NAME" -n "$NAMESPACE"
    else
        log_warning "Helm release $RELEASE_NAME not found in namespace $NAMESPACE"
    fi
}

# Check Kubernetes status
check_kubernetes_status() {
    if ! command -v kubectl >/dev/null 2>&1; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    echo "Deployments:"
    kubectl get deployments -n "$NAMESPACE" -l app=tsiot -o wide
    
    echo ""
    echo "ReplicaSets:"
    kubectl get replicasets -n "$NAMESPACE" -l app=tsiot -o wide
    
    echo ""
    echo "Pods:"
    kubectl get pods -n "$NAMESPACE" -l app=tsiot -o wide
    
    echo ""
    echo "Rollout History:"
    for deployment in $(kubectl get deployments -n "$NAMESPACE" -l app=tsiot -o name); do
        echo "History for $deployment:"
        kubectl rollout history "$deployment" -n "$NAMESPACE"
        echo ""
    done
}

# Check Docker Compose status
check_docker_compose_status() {
    if ! command -v docker-compose >/dev/null 2>&1; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    local compose_file="$PROJECT_ROOT/deployments/docker/docker-compose.yml"
    
    if [[ -f "$compose_file" ]]; then
        cd "$PROJECT_ROOT/deployments/docker"
        echo "Docker Compose Services:"
        docker-compose ps
    else
        log_warning "Docker Compose file not found: $compose_file"
    fi
}

# Check Docker status
check_docker_status() {
    if ! command -v docker >/dev/null 2>&1; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    echo "TSIoT Containers:"
    docker ps -a --filter "name=tsiot" --format "table {{.Names}}\\t{{.Image}}\\t{{.Status}}\\t{{.Ports}}"
}

# List deployment history
list_deployment_history() {
    log_info "Listing deployment history..."
    
    case "$METHOD" in
        "helm")
            list_helm_history
            ;;
        "kubernetes")
            list_kubernetes_history
            ;;
        "docker-compose")
            list_docker_compose_history
            ;;
        "docker")
            list_docker_history
            ;;
    esac
}

# List Helm history
list_helm_history() {
    if helm list -n "$NAMESPACE" | grep -q "$RELEASE_NAME"; then
        echo "Helm Release History for $RELEASE_NAME:"
        helm history "$RELEASE_NAME" -n "$NAMESPACE" --max 20
    else
        log_warning "Helm release $RELEASE_NAME not found"
    fi
}

# List Kubernetes history
list_kubernetes_history() {
    echo "Kubernetes Deployment History:"
    for deployment in $(kubectl get deployments -n "$NAMESPACE" -l app=tsiot -o name); do
        deployment_name=$(echo "$deployment" | cut -d'/' -f2)
        echo ""
        echo "History for $deployment_name:"
        kubectl rollout history "$deployment" -n "$NAMESPACE"
    done
}

# List Docker Compose history
list_docker_compose_history() {
    log_info "Docker Compose deployment history (from logs):"
    
    # Show recent container creation events
    if command -v docker >/dev/null 2>&1; then
        docker images --filter "reference=*tsiot*" --format "table {{.Repository}}:{{.Tag}}\\t{{.CreatedAt}}\\t{{.Size}}"
    fi
}

# List Docker history
list_docker_history() {
    log_info "Docker container history:"
    
    # Show recent TSIoT containers
    docker ps -a --filter "name=tsiot" --format "table {{.Names}}\\t{{.Image}}\\t{{.CreatedAt}}\\t{{.Status}}"
    
    echo ""
    echo "Available TSIoT Images:"
    docker images --filter "reference=*tsiot*" --format "table {{.Repository}}:{{.Tag}}\\t{{.CreatedAt}}\\t{{.Size}}"
}

# Backup data before rollback
backup_data() {
    if [[ "$BACKUP_DATA" == "false" ]]; then
        return 0
    fi
    
    log_info "Backing up data before rollback..."
    
    if [[ -f "$PROJECT_ROOT/scripts/data/backup-data.sh" ]]; then
        "$PROJECT_ROOT/scripts/data/backup-data.sh" \
            --environment "$ENVIRONMENT" \
            --backup-type "pre-rollback" \
            --tag "rollback-$(date +%Y%m%d-%H%M%S)"
    else
        log_warning "Backup script not found, skipping data backup"
    fi
}

# Restore data after rollback
restore_data() {
    if [[ "$RESTORE_DATA" == "false" ]]; then
        return 0
    fi
    
    log_info "Restoring data after rollback..."
    
    if [[ -f "$PROJECT_ROOT/scripts/data/restore-data.sh" ]]; then
        "$PROJECT_ROOT/scripts/data/restore-data.sh" \
            --environment "$ENVIRONMENT" \
            --backup-type "pre-rollback"
    else
        log_warning "Restore script not found, skipping data restore"
    fi
}

# Confirm rollback operation
confirm_rollback() {
    if [[ "$CONFIRM" == "true" ]] || [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    
    echo ""
    log_warning "You are about to perform a rollback operation:"
    echo "  Method: $METHOD"
    echo "  Environment: $ENVIRONMENT"
    echo "  Namespace: $NAMESPACE"
    
    if [[ -n "$VERSION" ]]; then
        echo "  Target Version: $VERSION"
    fi
    
    if [[ -n "$REVISION" ]]; then
        echo "  Target Revision: $REVISION"
    else
        echo "  Steps back: $STEPS"
    fi
    
    echo ""
    read -p "Are you sure you want to proceed? (yes/no): " -r
    
    if [[ ! "$REPLY" =~ ^[Yy][Ee][Ss]$ ]]; then
        log_info "Rollback cancelled by user"
        exit 0
    fi
}

# Perform Helm rollback
rollback_helm() {
    log_info "Performing Helm rollback..."
    
    if ! helm list -n "$NAMESPACE" | grep -q "$RELEASE_NAME"; then
        log_error "Helm release $RELEASE_NAME not found"
        exit 1
    fi
    
    local rollback_cmd=("helm" "rollback" "$RELEASE_NAME")
    
    if [[ -n "$REVISION" ]]; then
        rollback_cmd+=("$REVISION")
    else
        # Calculate target revision
        local current_revision
        current_revision=$(helm list -n "$NAMESPACE" -o json | jq -r ".[] | select(.name==\"$RELEASE_NAME\") | .revision")
        local target_revision=$((current_revision - STEPS))
        
        if [[ $target_revision -lt 1 ]]; then
            log_error "Cannot rollback $STEPS steps from revision $current_revision"
            exit 1
        fi
        
        rollback_cmd+=("$target_revision")
    fi
    
    rollback_cmd+=("--namespace" "$NAMESPACE")
    rollback_cmd+=("--timeout" "$WAIT_TIMEOUT")
    rollback_cmd+=("--wait")
    
    if [[ "$FORCE" == "true" ]]; then
        rollback_cmd+=("--force")
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run: ${rollback_cmd[*]}"
    else
        "${rollback_cmd[@]}"
        log_success "Helm rollback completed"
    fi
}

# Perform Kubernetes rollback
rollback_kubernetes() {
    log_info "Performing Kubernetes rollback..."
    
    local deployments
    deployments=$(kubectl get deployments -n "$NAMESPACE" -l app=tsiot -o name)
    
    if [[ -z "$deployments" ]]; then
        log_error "No TSIoT deployments found in namespace $NAMESPACE"
        exit 1
    fi
    
    for deployment in $deployments; do
        local deployment_name
        deployment_name=$(echo "$deployment" | cut -d'/' -f2)
        
        log_info "Rolling back deployment: $deployment_name"
        
        if [[ -n "$REVISION" ]]; then
            if [[ "$DRY_RUN" == "true" ]]; then
                log_info "Dry run: kubectl rollout undo $deployment --to-revision=$REVISION -n $NAMESPACE"
            else
                kubectl rollout undo "$deployment" --to-revision="$REVISION" -n "$NAMESPACE"
            fi
        else
            # Rollback specified number of steps
            for ((i=1; i<=STEPS; i++)); do
                if [[ "$DRY_RUN" == "true" ]]; then
                    log_info "Dry run: kubectl rollout undo $deployment -n $NAMESPACE"
                else
                    kubectl rollout undo "$deployment" -n "$NAMESPACE"
                fi
            done
        fi
        
        # Wait for rollback to complete
        if [[ "$DRY_RUN" == "false" ]]; then
            kubectl rollout status "$deployment" -n "$NAMESPACE" --timeout="$WAIT_TIMEOUT"
        fi
    done
    
    log_success "Kubernetes rollback completed"
}

# Perform Docker Compose rollback
rollback_docker_compose() {
    log_info "Performing Docker Compose rollback..."
    
    local compose_file="$PROJECT_ROOT/deployments/docker/docker-compose.yml"
    
    if [[ ! -f "$compose_file" ]]; then
        log_error "Docker Compose file not found: $compose_file"
        exit 1
    fi
    
    cd "$PROJECT_ROOT/deployments/docker"
    
    if [[ -n "$VERSION" ]]; then
        # Set image tag for rollback
        export TSIOT_IMAGE_TAG="$VERSION"
        log_info "Rolling back to version: $VERSION"
    else
        log_warning "No specific version specified for Docker Compose rollback"
        log_warning "You may need to manually specify the target image version"
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run: docker-compose pull && docker-compose up -d"
        docker-compose config
    else
        # Pull the target images
        docker-compose pull
        
        # Restart services with new images
        docker-compose up -d
        
        log_success "Docker Compose rollback completed"
    fi
}

# Perform Docker rollback
rollback_docker() {
    log_info "Performing Docker rollback..."
    
    local container_name="tsiot-$ENVIRONMENT"
    
    if [[ -z "$VERSION" ]]; then
        log_error "Version must be specified for Docker rollback"
        exit 1
    fi
    
    local image_name="ghcr.io/inferloop/tsiot:$VERSION"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run: Stop $container_name, start with $image_name"
    else
        # Stop current container
        if docker ps -q -f name="$container_name" | grep -q .; then
            log_info "Stopping current container: $container_name"
            docker stop "$container_name"
            docker rm "$container_name"
        fi
        
        # Start container with target version
        log_info "Starting container with version: $VERSION"
        docker run -d \
            --name "$container_name" \
            --env TSIOT_ENV="$ENVIRONMENT" \
            --env TSIOT_VERSION="$VERSION" \
            -p 8080:8080 \
            "$image_name"
        
        log_success "Docker rollback completed"
    fi
}

# Verify rollback success
verify_rollback() {
    if [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    
    log_info "Verifying rollback success..."
    
    # Wait a moment for services to stabilize
    sleep 10
    
    case "$METHOD" in
        "helm")
            verify_helm_rollback
            ;;
        "kubernetes")
            verify_kubernetes_rollback
            ;;
        "docker-compose")
            verify_docker_compose_rollback
            ;;
        "docker")
            verify_docker_rollback
            ;;
    esac
}

# Verify Helm rollback
verify_helm_rollback() {
    if helm status "$RELEASE_NAME" -n "$NAMESPACE" | grep -q "STATUS: deployed"; then
        log_success "Helm release is in deployed status"
    else
        log_error "Helm release is not in deployed status"
        return 1
    fi
}

# Verify Kubernetes rollback
verify_kubernetes_rollback() {
    local all_ready=true
    
    for deployment in $(kubectl get deployments -n "$NAMESPACE" -l app=tsiot -o name); do
        if ! kubectl get "$deployment" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Available")].status}' | grep -q "True"; then
            all_ready=false
            break
        fi
    done
    
    if [[ "$all_ready" == "true" ]]; then
        log_success "All deployments are available"
    else
        log_error "Some deployments are not available"
        return 1
    fi
}

# Verify Docker Compose rollback
verify_docker_compose_rollback() {
    cd "$PROJECT_ROOT/deployments/docker"
    
    local running_services
    running_services=$(docker-compose ps --services --filter "status=running" | wc -l)
    
    if [[ "$running_services" -gt 0 ]]; then
        log_success "Docker Compose services are running"
    else
        log_error "No Docker Compose services are running"
        return 1
    fi
}

# Verify Docker rollback
verify_docker_rollback() {
    local container_name="tsiot-$ENVIRONMENT"
    
    if docker ps --filter "name=$container_name" --filter "status=running" | grep -q "$container_name"; then
        log_success "Docker container is running"
    else
        log_error "Docker container is not running"
        return 1
    fi
}

# Generate rollback report
generate_rollback_report() {
    log_info "Generating rollback report..."
    
    local report_file="$LOG_DIR/rollback-report-$(date +%Y%m%d-%H%M%S).json"
    
    cat > "$report_file" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "rollback": {
        "method": "$METHOD",
        "environment": "$ENVIRONMENT",
        "namespace": "$NAMESPACE",
        "release_name": "$RELEASE_NAME",
        "target_version": "$VERSION",
        "target_revision": "$REVISION",
        "steps": $STEPS,
        "backup_data": $BACKUP_DATA,
        "restore_data": $RESTORE_DATA,
        "force": $FORCE,
        "dry_run": $DRY_RUN
    }
}
EOF
    
    log_success "Rollback report generated: $report_file"
}

# Main execution
main() {
    create_directories
    
    # Handle special operations first
    if [[ "$CHECK_STATUS" == "true" ]]; then
        auto_detect_method
        check_deployment_status
        exit 0
    fi
    
    if [[ "$LIST_HISTORY" == "true" ]]; then
        auto_detect_method
        list_deployment_history
        exit 0
    fi
    
    # Auto-detect method if not specified
    auto_detect_method
    
    log_info "Starting rollback operation"
    log_info "Method: $METHOD"
    log_info "Environment: $ENVIRONMENT"
    
    # Confirm operation
    confirm_rollback
    
    # Backup data if requested
    backup_data
    
    # Perform rollback based on method
    case "$METHOD" in
        "helm")
            rollback_helm
            ;;
        "kubernetes")
            rollback_kubernetes
            ;;
        "docker-compose")
            rollback_docker_compose
            ;;
        "docker")
            rollback_docker
            ;;
        *)
            log_error "Unsupported rollback method: $METHOD"
            exit 1
            ;;
    esac
    
    # Verify rollback
    verify_rollback
    
    # Restore data if requested
    restore_data
    
    # Generate report
    generate_rollback_report
    
    log_success "Rollback completed successfully"
}

# Execute main function
main "$@"