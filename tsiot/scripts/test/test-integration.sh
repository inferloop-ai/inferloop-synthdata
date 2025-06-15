#!/bin/bash

# TSIoT Integration Test Runner
# Runs integration tests for the Time Series IoT Synthetic Data Generation Platform

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs/tests"
REPORT_DIR="$PROJECT_ROOT/reports/tests"
DOCKER_COMPOSE_FILE="$PROJECT_ROOT/deployments/docker/docker-compose.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Service configuration
REQUIRED_SERVICES=(
    "postgres"
    "redis"
    "kafka"
    "influxdb"
    "timescaledb"
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
    --setup                 Setup required services for integration tests
    --teardown              Teardown services after tests
    --check-services        Check if required services are running
    --reset-data            Reset test databases to clean state
    -v, --verbose           Enable verbose output
    -t, --timeout DURATION  Test timeout duration [default: 20m]
    --tags TAGS             Run only tests with specified tags
    --service SERVICE       Test only specific service integration
    -h, --help              Show this help message

Services:
    storage                 Storage layer integration (postgres, redis, influxdb)
    messaging               Messaging layer integration (kafka, mqtt)
    generators              Generator service integration
    validators              Validation service integration
    api                     API layer integration
    workflows               Workflow engine integration

Examples:
    $0 --setup              # Setup services and run all integration tests
    $0 --service storage    # Run only storage integration tests
    $0 --teardown           # Teardown services after tests
    $0 --check-services     # Check if services are ready

Environment Variables:
    TSIOT_INTEGRATION_ENV   Integration test environment
    TSIOT_DB_HOST          Database host for tests
    TSIOT_KAFKA_BROKERS    Kafka brokers for tests
    INTEGRATION_TIMEOUT     Global integration test timeout
EOF
}

# Parse command line arguments
SETUP=false
TEARDOWN=false
CHECK_SERVICES=false
RESET_DATA=false
VERBOSE=false
TEST_TIMEOUT="20m"
TEST_TAGS=""
SERVICE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --setup)
            SETUP=true
            shift
            ;;
        --teardown)
            TEARDOWN=true
            shift
            ;;
        --check-services)
            CHECK_SERVICES=true
            shift
            ;;
        --reset-data)
            RESET_DATA=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -t|--timeout)
            TEST_TIMEOUT="$2"
            shift 2
            ;;
        --tags)
            TEST_TAGS="$2"
            shift 2
            ;;
        --service)
            SERVICE="$2"
            shift 2
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
export TSIOT_INTEGRATION_ENV="${TSIOT_INTEGRATION_ENV:-test}"
export TSIOT_DB_HOST="${TSIOT_DB_HOST:-localhost}"
export TSIOT_KAFKA_BROKERS="${TSIOT_KAFKA_BROKERS:-localhost:9092}"
export INTEGRATION_TIMEOUT="${INTEGRATION_TIMEOUT:-$TEST_TIMEOUT}"

# Create necessary directories
create_directories() {
    mkdir -p "$LOG_DIR" "$REPORT_DIR"
}

# Check if Docker is available
check_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v docker-compose >/dev/null 2>&1; then
        log_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
}

# Setup required services
setup_services() {
    log_info "Setting up required services for integration tests..."
    
    check_docker
    
    # Start services using docker-compose
    if [[ -f "$DOCKER_COMPOSE_FILE" ]]; then
        log_info "Starting services with docker-compose..."
        cd "$PROJECT_ROOT/deployments/docker"
        docker-compose up -d --wait "${REQUIRED_SERVICES[@]}"
        
        # Wait for services to be ready
        wait_for_services
        
        if [[ "$RESET_DATA" == "true" ]]; then
            reset_test_data
        fi
        
        log_success "Services are ready for integration tests"
    else
        log_error "Docker compose file not found: $DOCKER_COMPOSE_FILE"
        exit 1
    fi
}

# Wait for services to be ready
wait_for_services() {
    log_info "Waiting for services to be ready..."
    
    local max_attempts=30
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        local all_ready=true
        
        # Check PostgreSQL
        if ! docker-compose exec -T postgres pg_isready -h localhost -p 5432 >/dev/null 2>&1; then
            all_ready=false
        fi
        
        # Check Redis
        if ! docker-compose exec -T redis redis-cli ping >/dev/null 2>&1; then
            all_ready=false
        fi
        
        # Check Kafka
        if ! docker-compose exec -T kafka kafka-topics.sh --bootstrap-server localhost:9092 --list >/dev/null 2>&1; then
            all_ready=false
        fi
        
        # Check InfluxDB
        if ! curl -f http://localhost:8086/health >/dev/null 2>&1; then
            all_ready=false
        fi
        
        if [[ "$all_ready" == "true" ]]; then
            log_success "All services are ready"
            return 0
        fi
        
        log_info "Waiting for services... (attempt $((attempt + 1))/$max_attempts)"
        sleep 5
        ((attempt++))
    done
    
    log_error "Services failed to become ready within timeout"
    return 1
}

# Check if services are running
check_services() {
    log_info "Checking if required services are running..."
    
    local all_running=true
    
    for service in "${REQUIRED_SERVICES[@]}"; do
        if ! docker-compose ps "$service" | grep -q "Up"; then
            log_warning "Service $service is not running"
            all_running=false
        fi
    done
    
    if [[ "$all_running" == "true" ]]; then
        log_success "All required services are running"
        return 0
    else
        log_error "Some required services are not running"
        return 1
    fi
}

# Reset test data
reset_test_data() {
    log_info "Resetting test databases to clean state..."
    
    # Reset PostgreSQL test database
    docker-compose exec -T postgres psql -U postgres -c "DROP DATABASE IF EXISTS tsiot_test;"
    docker-compose exec -T postgres psql -U postgres -c "CREATE DATABASE tsiot_test;"
    
    # Reset Redis test database
    docker-compose exec -T redis redis-cli -n 1 FLUSHDB
    
    # Reset InfluxDB test database
    curl -X POST "http://localhost:8086/query" --data-urlencode "q=DROP DATABASE tsiot_test" >/dev/null 2>&1 || true
    curl -X POST "http://localhost:8086/query" --data-urlencode "q=CREATE DATABASE tsiot_test" >/dev/null 2>&1
    
    # Reset Kafka topics
    docker-compose exec -T kafka kafka-topics.sh --bootstrap-server localhost:9092 --delete --topic tsiot-test >/dev/null 2>&1 || true
    docker-compose exec -T kafka kafka-topics.sh --bootstrap-server localhost:9092 --create --topic tsiot-test --partitions 3 --replication-factor 1 >/dev/null 2>&1
    
    log_success "Test databases reset completed"
}

# Teardown services
teardown_services() {
    log_info "Tearing down integration test services..."
    
    if [[ -f "$DOCKER_COMPOSE_FILE" ]]; then
        cd "$PROJECT_ROOT/deployments/docker"
        docker-compose down -v
        log_success "Services teardown completed"
    else
        log_warning "Docker compose file not found: $DOCKER_COMPOSE_FILE"
    fi
}

# Build test flags
build_test_flags() {
    local flags=()
    
    flags+=("-tags" "integration")
    flags+=("-timeout" "$TEST_TIMEOUT")
    
    if [[ "$VERBOSE" == "true" ]]; then
        flags+=("-v")
    fi
    
    if [[ -n "$TEST_TAGS" ]]; then
        flags+=("-tags" "integration,$TEST_TAGS")
    fi
    
    echo "${flags[@]}"
}

# Run storage integration tests
run_storage_tests() {
    log_info "Running storage integration tests..."
    
    local test_flags
    test_flags=$(build_test_flags)
    
    # shellcheck disable=SC2086
    go test ${test_flags[@]} \
        "$PROJECT_ROOT/tests/integration/storage/..." | tee "$LOG_DIR/integration-storage.log"
}

# Run messaging integration tests
run_messaging_tests() {
    log_info "Running messaging integration tests..."
    
    local test_flags
    test_flags=$(build_test_flags)
    
    # shellcheck disable=SC2086
    go test ${test_flags[@]} \
        "$PROJECT_ROOT/tests/integration/messaging/..." | tee "$LOG_DIR/integration-messaging.log"
}

# Run generator integration tests
run_generator_tests() {
    log_info "Running generator integration tests..."
    
    local test_flags
    test_flags=$(build_test_flags)
    
    # shellcheck disable=SC2086
    go test ${test_flags[@]} \
        "$PROJECT_ROOT/tests/integration/generators/..." | tee "$LOG_DIR/integration-generators.log"
}

# Run validator integration tests
run_validator_tests() {
    log_info "Running validator integration tests..."
    
    local test_flags
    test_flags=$(build_test_flags)
    
    # shellcheck disable=SC2086
    go test ${test_flags[@]} \
        "$PROJECT_ROOT/tests/integration/validators/..." | tee "$LOG_DIR/integration-validators.log"
}

# Run API integration tests
run_api_tests() {
    log_info "Running API integration tests..."
    
    local test_flags
    test_flags=$(build_test_flags)
    
    # shellcheck disable=SC2086
    go test ${test_flags[@]} \
        "$PROJECT_ROOT/tests/integration/api/..." | tee "$LOG_DIR/integration-api.log"
}

# Run workflow integration tests
run_workflow_tests() {
    log_info "Running workflow integration tests..."
    
    local test_flags
    test_flags=$(build_test_flags)
    
    # shellcheck disable=SC2086
    go test ${test_flags[@]} \
        "$PROJECT_ROOT/tests/integration/workflows/..." | tee "$LOG_DIR/integration-workflows.log"
}

# Run all integration tests
run_all_tests() {
    log_info "Running all integration tests..."
    
    run_storage_tests
    run_messaging_tests
    run_generator_tests
    run_validator_tests
    run_api_tests
    run_workflow_tests
}

# Generate integration test report
generate_report() {
    log_info "Generating integration test report..."
    
    local report_file="$REPORT_DIR/integration-test-report.json"
    cat > "$report_file" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "test_type": "integration",
    "environment": "$TSIOT_INTEGRATION_ENV",
    "timeout": "$TEST_TIMEOUT",
    "service": "$SERVICE",
    "tags": "$TEST_TAGS",
    "services_status": {
        "postgres": "$(docker-compose ps postgres | grep -q "Up" && echo "running" || echo "stopped")",
        "redis": "$(docker-compose ps redis | grep -q "Up" && echo "running" || echo "stopped")",
        "kafka": "$(docker-compose ps kafka | grep -q "Up" && echo "running" || echo "stopped")",
        "influxdb": "$(docker-compose ps influxdb | grep -q "Up" && echo "running" || echo "stopped")"
    },
    "logs": {
        "storage": "$LOG_DIR/integration-storage.log",
        "messaging": "$LOG_DIR/integration-messaging.log",
        "generators": "$LOG_DIR/integration-generators.log",
        "validators": "$LOG_DIR/integration-validators.log",
        "api": "$LOG_DIR/integration-api.log",
        "workflows": "$LOG_DIR/integration-workflows.log"
    }
}
EOF
    
    log_success "Integration test report generated: $report_file"
}

# Main execution
main() {
    create_directories
    
    cd "$PROJECT_ROOT"
    
    if [[ "$CHECK_SERVICES" == "true" ]]; then
        check_services
        exit $?
    fi
    
    if [[ "$SETUP" == "true" ]]; then
        setup_services
    fi
    
    if [[ "$TEARDOWN" == "true" ]]; then
        teardown_services
        exit 0
    fi
    
    # Verify Go is available
    if ! command -v go >/dev/null 2>&1; then
        log_error "Go is not installed or not in PATH"
        exit 1
    fi
    
    # Ensure services are running
    if ! check_services; then
        log_error "Required services are not running. Use --setup to start them."
        exit 1
    fi
    
    log_info "Starting integration tests"
    log_info "Environment: $TSIOT_INTEGRATION_ENV"
    log_info "Timeout: $TEST_TIMEOUT"
    
    if [[ -n "$SERVICE" ]]; then
        case "$SERVICE" in
            "storage")
                run_storage_tests
                ;;
            "messaging")
                run_messaging_tests
                ;;
            "generators")
                run_generator_tests
                ;;
            "validators")
                run_validator_tests
                ;;
            "api")
                run_api_tests
                ;;
            "workflows")
                run_workflow_tests
                ;;
            *)
                log_error "Unknown service: $SERVICE"
                exit 1
                ;;
        esac
    else
        run_all_tests
    fi
    
    generate_report
    log_success "Integration tests completed successfully"
}

# Execute main function
main "$@"