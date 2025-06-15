#!/bin/bash

# TSIoT Main Deployment Script
# Orchestrates deployment of the Time Series IoT Synthetic Data Generation Platform

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs/deploy"
DEPLOY_CONFIG_DIR="$PROJECT_ROOT/configs/environments"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Deployment methods
DEPLOYMENT_METHODS=(
    "docker"
    "kubernetes"
    "helm"
    "docker-compose"
)

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
    --method METHOD         Deployment method (docker|kubernetes|helm|docker-compose) [default: docker-compose]
    --environment ENV       Target environment (development|staging|production) [default: development]
    --namespace NAMESPACE   Kubernetes namespace [default: tsiot]
    --version VERSION       Application version to deploy [default: latest]
    --config-file FILE      Custom configuration file
    --replicas COUNT        Number of replicas for scaling [default: 1]
    --wait-timeout DURATION Wait timeout for deployment readiness [default: 10m]
    --force                 Force deployment even if target exists
    --dry-run               Show what would be deployed without executing
    --rollback              Rollback to previous version
    --health-check          Perform health check after deployment
    --migrate-db            Run database migrations
    --seed-data             Seed initial data
    -v, --verbose           Enable verbose output
    -h, --help              Show this help message

Examples:
    $0                                          # Deploy using docker-compose to development
    $0 --method kubernetes --environment staging   # Deploy to Kubernetes staging
    $0 --method helm --replicas 3              # Deploy with Helm using 3 replicas
    $0 --rollback --environment production     # Rollback production deployment
    $0 --dry-run --method kubernetes           # Dry run Kubernetes deployment

Environment Variables:
    TSIOT_DEPLOY_ENV        Deployment environment
    TSIOT_REGISTRY          Container registry URL
    TSIOT_IMAGE_TAG         Container image tag
    KUBECONFIG              Kubernetes configuration file
    HELM_NAMESPACE          Helm deployment namespace
    DATABASE_URL            Database connection URL
EOF
}

# Parse command line arguments
METHOD="docker-compose"
ENVIRONMENT="development"
NAMESPACE="tsiot"
VERSION="latest"
CONFIG_FILE=""
REPLICAS=1
WAIT_TIMEOUT="10m"
FORCE=false
DRY_RUN=false
ROLLBACK=false
HEALTH_CHECK=true
MIGRATE_DB=false
SEED_DATA=false
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
        --version)
            VERSION="$2"
            shift 2
            ;;
        --config-file)
            CONFIG_FILE="$2"
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
        --rollback)
            ROLLBACK=true
            shift
            ;;
        --health-check)
            HEALTH_CHECK=true
            shift
            ;;
        --migrate-db)
            MIGRATE_DB=true
            shift
            ;;
        --seed-data)
            SEED_DATA=true
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

# Validate deployment method
if [[ ! " ${DEPLOYMENT_METHODS[@]} " =~ " ${METHOD} " ]]; then
    log_error "Invalid deployment method: $METHOD"
    log_error "Valid methods: ${DEPLOYMENT_METHODS[*]}"
    exit 1
fi

# Set environment variables
export TSIOT_DEPLOY_ENV="${TSIOT_DEPLOY_ENV:-$ENVIRONMENT}"
export TSIOT_REGISTRY="${TSIOT_REGISTRY:-ghcr.io/inferloop}"
export TSIOT_IMAGE_TAG="${TSIOT_IMAGE_TAG:-$VERSION}"
export HELM_NAMESPACE="${HELM_NAMESPACE:-$NAMESPACE}"

# Create necessary directories
create_directories() {
    mkdir -p "$LOG_DIR"
}

# Load configuration
load_configuration() {
    local config_file
    
    if [[ -n "$CONFIG_FILE" ]]; then
        config_file="$CONFIG_FILE"
    else
        config_file="$DEPLOY_CONFIG_DIR/$ENVIRONMENT.yaml"
    fi
    
    if [[ -f "$config_file" ]]; then
        log_info "Loading configuration from: $config_file"
        export TSIOT_CONFIG_FILE="$config_file"
    else
        log_warning "Configuration file not found: $config_file"
        log_warning "Using default configuration"
    fi
}

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."
    
    # Check required tools
    check_required_tools
    
    # Validate configuration
    validate_configuration
    
    # Check deployment target
    check_deployment_target
    
    # Verify image availability
    verify_image_availability
    
    log_success "Pre-deployment checks completed"
}

# Check required tools
check_required_tools() {
    local required_tools=()
    
    case "$METHOD" in
        "docker")
            required_tools+=("docker")
            ;;
        "kubernetes")
            required_tools+=("kubectl")
            ;;
        "helm")
            required_tools+=("helm" "kubectl")
            ;;
        "docker-compose")
            required_tools+=("docker" "docker-compose")
            ;;
    esac
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            log_error "$tool is not installed or not in PATH"
            exit 1
        fi
    done
}

# Validate configuration
validate_configuration() {
    log_info "Validating deployment configuration..."
    
    # Check environment-specific configuration
    case "$ENVIRONMENT" in
        "development"|"staging"|"production")
            log_info "Environment: $ENVIRONMENT"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
    
    # Validate replicas
    if [[ ! "$REPLICAS" =~ ^[0-9]+$ ]] || [[ "$REPLICAS" -lt 1 ]]; then
        log_error "Invalid replicas count: $REPLICAS"
        exit 1
    fi
}

# Check deployment target
check_deployment_target() {
    case "$METHOD" in
        "kubernetes"|"helm")
            check_kubernetes_access
            ;;
        "docker"|"docker-compose")
            check_docker_access
            ;;
    esac
}

# Check Kubernetes access
check_kubernetes_access() {
    log_info "Checking Kubernetes access..."
    
    if ! kubectl cluster-info >/dev/null 2>&1; then
        log_error "Cannot access Kubernetes cluster"
        exit 1
    fi
    
    # Check namespace
    if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        log_info "Creating namespace: $NAMESPACE"
        if [[ "$DRY_RUN" == "false" ]]; then
            kubectl create namespace "$NAMESPACE"
        fi
    fi
}

# Check Docker access
check_docker_access() {
    log_info "Checking Docker access..."
    
    if ! docker info >/dev/null 2>&1; then
        log_error "Cannot access Docker daemon"
        exit 1
    fi
}

# Verify image availability
verify_image_availability() {
    log_info "Verifying container image availability..."
    
    local image_name="$TSIOT_REGISTRY/tsiot:$TSIOT_IMAGE_TAG"
    
    case "$METHOD" in
        "docker"|"docker-compose")
            if ! docker pull "$image_name" >/dev/null 2>&1; then
                log_warning "Image not available in registry: $image_name"
                log_info "Building image locally..."
                build_local_image
            fi
            ;;
        "kubernetes"|"helm")
            # For K8s, we'll let the cluster pull the image
            log_info "Image will be pulled by cluster: $image_name"
            ;;
    esac
}

# Build local image
build_local_image() {
    log_info "Building local Docker image..."
    
    if [[ -f "$PROJECT_ROOT/Dockerfile" ]]; then
        docker build -t "$TSIOT_REGISTRY/tsiot:$TSIOT_IMAGE_TAG" "$PROJECT_ROOT"
    else
        log_error "Dockerfile not found in project root"
        exit 1
    fi
}

# Run database migrations
run_database_migrations() {
    if [[ "$MIGRATE_DB" == "false" ]]; then
        return 0
    fi
    
    log_info "Running database migrations..."
    
    # Use the migration tool or script
    if [[ -f "$PROJECT_ROOT/scripts/data/migrate-data.sh" ]]; then
        "$PROJECT_ROOT/scripts/data/migrate-data.sh" --environment "$ENVIRONMENT"
    else
        log_warning "Migration script not found, skipping migrations"
    fi
}

# Seed initial data
seed_initial_data() {
    if [[ "$SEED_DATA" == "false" ]]; then
        return 0
    fi
    
    log_info "Seeding initial data..."
    
    # Use the data seeding script
    if [[ -f "$PROJECT_ROOT/scripts/data/generate-sample-data.sh" ]]; then
        "$PROJECT_ROOT/scripts/data/generate-sample-data.sh" --environment "$ENVIRONMENT"
    else
        log_warning "Data seeding script not found, skipping data seeding"
    fi
}

# Deploy using Docker Compose
deploy_docker_compose() {
    log_info "Deploying using Docker Compose..."
    
    local compose_file="$PROJECT_ROOT/deployments/docker/docker-compose.yml"
    local env_file="$PROJECT_ROOT/deployments/docker/.env.$ENVIRONMENT"
    
    if [[ ! -f "$compose_file" ]]; then
        log_error "Docker Compose file not found: $compose_file"
        exit 1
    fi
    
    cd "$PROJECT_ROOT/deployments/docker"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run: docker-compose config"
        docker-compose config
        return 0
    fi
    
    # Set environment file if it exists
    if [[ -f "$env_file" ]]; then
        export ENV_FILE="$env_file"
    fi
    
    # Deploy services
    docker-compose up -d --scale server="$REPLICAS"
    
    log_success "Docker Compose deployment completed"
}

# Deploy using Kubernetes
deploy_kubernetes() {
    log_info "Deploying using Kubernetes..."
    
    "$SCRIPT_DIR/deploy-k8s.sh" \
        --environment "$ENVIRONMENT" \
        --namespace "$NAMESPACE" \
        --version "$VERSION" \
        --replicas "$REPLICAS" \
        --wait-timeout "$WAIT_TIMEOUT" \
        $([ "$FORCE" == "true" ] && echo "--force") \
        $([ "$DRY_RUN" == "true" ] && echo "--dry-run") \
        $([ "$VERBOSE" == "true" ] && echo "--verbose")
}

# Deploy using Helm
deploy_helm() {
    log_info "Deploying using Helm..."
    
    "$SCRIPT_DIR/deploy-helm.sh" \
        --environment "$ENVIRONMENT" \
        --namespace "$NAMESPACE" \
        --version "$VERSION" \
        --replicas "$REPLICAS" \
        --wait-timeout "$WAIT_TIMEOUT" \
        $([ "$FORCE" == "true" ] && echo "--force") \
        $([ "$DRY_RUN" == "true" ] && echo "--dry-run") \
        $([ "$VERBOSE" == "true" ] && echo "--verbose")
}

# Deploy using Docker
deploy_docker() {
    log_info "Deploying using Docker..."
    
    local image_name="$TSIOT_REGISTRY/tsiot:$TSIOT_IMAGE_TAG"
    local container_name="tsiot-$ENVIRONMENT"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run: docker run $image_name"
        return 0
    fi
    
    # Stop existing container if running
    if docker ps -q -f name="$container_name" | grep -q .; then
        if [[ "$FORCE" == "true" ]]; then
            log_info "Stopping existing container: $container_name"
            docker stop "$container_name"
            docker rm "$container_name"
        else
            log_error "Container $container_name is already running. Use --force to replace."
            exit 1
        fi
    fi
    
    # Run new container
    docker run -d \
        --name "$container_name" \
        --env TSIOT_ENV="$ENVIRONMENT" \
        --env TSIOT_VERSION="$VERSION" \
        -p 8080:8080 \
        "$image_name"
    
    log_success "Docker deployment completed"
}

# Perform rollback
perform_rollback() {
    log_info "Performing rollback..."
    
    "$SCRIPT_DIR/rollback.sh" \
        --method "$METHOD" \
        --environment "$ENVIRONMENT" \
        --namespace "$NAMESPACE" \
        $([ "$VERBOSE" == "true" ] && echo "--verbose")
}

# Wait for deployment readiness
wait_for_readiness() {
    log_info "Waiting for deployment readiness..."
    
    local timeout_seconds
    timeout_seconds=$(echo "$WAIT_TIMEOUT" | sed 's/m/*60/' | sed 's/s//' | bc)
    local elapsed=0
    local check_interval=10
    
    while [[ $elapsed -lt $timeout_seconds ]]; do
        if check_deployment_health; then
            log_success "Deployment is ready"
            return 0
        fi
        
        log_info "Waiting for readiness... ($elapsed/${timeout_seconds}s)"
        sleep $check_interval
        elapsed=$((elapsed + check_interval))
    done
    
    log_error "Deployment did not become ready within timeout"
    return 1
}

# Check deployment health
check_deployment_health() {
    case "$METHOD" in
        "docker-compose")
            check_docker_compose_health
            ;;
        "kubernetes")
            check_kubernetes_health
            ;;
        "helm")
            check_helm_health
            ;;
        "docker")
            check_docker_health
            ;;
    esac
}

# Check Docker Compose health
check_docker_compose_health() {
    cd "$PROJECT_ROOT/deployments/docker"
    local running_services
    running_services=$(docker-compose ps --services --filter "status=running" | wc -l)
    local total_services
    total_services=$(docker-compose config --services | wc -l)
    
    [[ "$running_services" -eq "$total_services" ]]
}

# Check Kubernetes health
check_kubernetes_health() {
    kubectl get pods -n "$NAMESPACE" -l app=tsiot --field-selector=status.phase=Running | grep -q "Running"
}

# Check Helm health
check_helm_health() {
    helm status tsiot -n "$NAMESPACE" | grep -q "STATUS: deployed"
}

# Check Docker health
check_docker_health() {
    local container_name="tsiot-$ENVIRONMENT"
    docker ps --filter "name=$container_name" --filter "status=running" | grep -q "$container_name"
}

# Perform health check
perform_health_check() {
    if [[ "$HEALTH_CHECK" == "false" ]]; then
        return 0
    fi
    
    log_info "Performing post-deployment health check..."
    
    # Determine the service URL based on deployment method
    local service_url
    case "$METHOD" in
        "docker-compose"|"docker")
            service_url="http://localhost:8080"
            ;;
        "kubernetes"|"helm")
            # For K8s, you might need to use port-forward or ingress
            service_url="http://localhost:8080"  # Assuming port-forward or local access
            ;;
    esac
    
    # Wait a moment for services to initialize
    sleep 10
    
    # Check health endpoint
    if curl -f -s "$service_url/health" >/dev/null; then
        log_success "Health check passed"
    else
        log_warning "Health check failed - service may still be starting"
        return 1
    fi
}

# Generate deployment report
generate_deployment_report() {
    log_info "Generating deployment report..."
    
    local report_file="$LOG_DIR/deployment-report-$(date +%Y%m%d-%H%M%S).json"
    
    cat > "$report_file" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "deployment": {
        "method": "$METHOD",
        "environment": "$ENVIRONMENT",
        "namespace": "$NAMESPACE",
        "version": "$VERSION",
        "replicas": $REPLICAS,
        "force": $FORCE,
        "dry_run": $DRY_RUN,
        "rollback": $ROLLBACK
    },
    "configuration": {
        "config_file": "$CONFIG_FILE",
        "registry": "$TSIOT_REGISTRY",
        "image_tag": "$TSIOT_IMAGE_TAG"
    },
    "options": {
        "migrate_db": $MIGRATE_DB,
        "seed_data": $SEED_DATA,
        "health_check": $HEALTH_CHECK,
        "wait_timeout": "$WAIT_TIMEOUT"
    }
}
EOF
    
    log_success "Deployment report generated: $report_file"
}

# Main execution
main() {
    log_info "Starting TSIoT deployment"
    log_info "Method: $METHOD"
    log_info "Environment: $ENVIRONMENT"
    log_info "Version: $VERSION"
    log_info "Replicas: $REPLICAS"
    
    create_directories
    load_configuration
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    if [[ "$ROLLBACK" == "true" ]]; then
        perform_rollback
        exit $?
    fi
    
    # Run pre-deployment checks
    pre_deployment_checks
    
    # Run database migrations if requested
    run_database_migrations
    
    # Deploy based on method
    case "$METHOD" in
        "docker-compose")
            deploy_docker_compose
            ;;
        "kubernetes")
            deploy_kubernetes
            ;;
        "helm")
            deploy_helm
            ;;
        "docker")
            deploy_docker
            ;;
    esac
    
    # Wait for readiness
    if [[ "$DRY_RUN" == "false" ]]; then
        wait_for_readiness
        perform_health_check
        
        # Seed data if requested
        seed_initial_data
    fi
    
    generate_deployment_report
    log_success "Deployment completed successfully"
}

# Execute main function
main "$@"