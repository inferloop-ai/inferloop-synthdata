#!/bin/bash

# TSIoT End-to-End Test Runner
# Runs comprehensive end-to-end tests for the Time Series IoT Synthetic Data Generation Platform

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs/tests"
REPORT_DIR="$PROJECT_ROOT/reports/tests"
FIXTURES_DIR="$PROJECT_ROOT/tests/fixtures"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test scenarios
TEST_SCENARIOS=(
    "data_generation_workflow"
    "privacy_preservation_pipeline"
    "real_time_streaming"
    "batch_processing"
    "multi_tenant_isolation"
    "disaster_recovery"
    "performance_scaling"
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
    --scenario SCENARIO     Run specific E2E scenario
    --environment ENV       Target environment (local|staging|production) [default: local]
    --browser BROWSER       Browser for UI tests (chrome|firefox|safari) [default: chrome]
    --headless              Run browser tests in headless mode
    --parallel              Run scenarios in parallel where possible
    --timeout DURATION      Test timeout duration [default: 45m]
    --cleanup               Clean up test data after completion
    --skip-setup            Skip environment setup
    --capture-logs          Capture detailed logs and screenshots
    -v, --verbose           Enable verbose output
    -h, --help              Show this help message

Scenarios:
    data_generation_workflow    Complete data generation workflow
    privacy_preservation_pipeline  Privacy-preserving synthetic data generation
    real_time_streaming         Real-time data streaming and processing
    batch_processing           Large-scale batch data processing
    multi_tenant_isolation     Multi-tenant data isolation validation
    disaster_recovery          Disaster recovery and failover scenarios
    performance_scaling        Performance and scaling validation

Examples:
    $0                                      # Run all E2E scenarios
    $0 --scenario data_generation_workflow  # Run specific scenario
    $0 --environment staging --headless     # Run against staging headlessly
    $0 --parallel --capture-logs            # Run in parallel with logging

Environment Variables:
    TSIOT_E2E_ENV           E2E test environment
    TSIOT_API_BASE_URL      Base URL for API tests
    TSIOT_UI_BASE_URL       Base URL for UI tests
    SELENIUM_HUB_URL        Selenium Grid hub URL
    E2E_TIMEOUT             Global E2E test timeout
    E2E_PARALLEL_WORKERS    Number of parallel workers
EOF
}

# Parse command line arguments
SCENARIO=""
ENVIRONMENT="local"
BROWSER="chrome"
HEADLESS=false
PARALLEL=false
TEST_TIMEOUT="45m"
CLEANUP=false
SKIP_SETUP=false
CAPTURE_LOGS=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --scenario)
            SCENARIO="$2"
            shift 2
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --browser)
            BROWSER="$2"
            shift 2
            ;;
        --headless)
            HEADLESS=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --timeout)
            TEST_TIMEOUT="$2"
            shift 2
            ;;
        --cleanup)
            CLEANUP=true
            shift
            ;;
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        --capture-logs)
            CAPTURE_LOGS=true
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
export TSIOT_E2E_ENV="${TSIOT_E2E_ENV:-$ENVIRONMENT}"
export E2E_TIMEOUT="${E2E_TIMEOUT:-$TEST_TIMEOUT}"
export E2E_PARALLEL_WORKERS="${E2E_PARALLEL_WORKERS:-3}"

# Set environment-specific URLs
case "$ENVIRONMENT" in
    "local")
        export TSIOT_API_BASE_URL="${TSIOT_API_BASE_URL:-http://localhost:8080}"
        export TSIOT_UI_BASE_URL="${TSIOT_UI_BASE_URL:-http://localhost:3000}"
        ;;
    "staging")
        export TSIOT_API_BASE_URL="${TSIOT_API_BASE_URL:-https://api-staging.tsiot.com}"
        export TSIOT_UI_BASE_URL="${TSIOT_UI_BASE_URL:-https://staging.tsiot.com}"
        ;;
    "production")
        export TSIOT_API_BASE_URL="${TSIOT_API_BASE_URL:-https://api.tsiot.com}"
        export TSIOT_UI_BASE_URL="${TSIOT_UI_BASE_URL:-https://tsiot.com}"
        ;;
esac

# Create necessary directories
create_directories() {
    mkdir -p "$LOG_DIR" "$REPORT_DIR" "$REPORT_DIR/screenshots" "$REPORT_DIR/videos"
}

# Setup test environment
setup_environment() {
    if [[ "$SKIP_SETUP" == "true" ]]; then
        log_info "Skipping environment setup"
        return 0
    fi
    
    log_info "Setting up E2E test environment: $ENVIRONMENT"
    
    case "$ENVIRONMENT" in
        "local")
            setup_local_environment
            ;;
        "staging"|"production")
            setup_remote_environment
            ;;
    esac
}

# Setup local test environment
setup_local_environment() {
    log_info "Setting up local test environment..."
    
    # Start required services
    if [[ -f "$PROJECT_ROOT/deployments/docker/docker-compose.yml" ]]; then
        cd "$PROJECT_ROOT/deployments/docker"
        docker-compose up -d
        
        # Wait for services to be ready
        log_info "Waiting for services to start..."
        sleep 30
        
        # Verify services are running
        if ! curl -f "$TSIOT_API_BASE_URL/health" >/dev/null 2>&1; then
            log_error "API service is not responding"
            exit 1
        fi
        
        log_success "Local environment is ready"
    else
        log_error "Docker compose file not found"
        exit 1
    fi
}

# Setup remote test environment
setup_remote_environment() {
    log_info "Verifying remote test environment: $ENVIRONMENT"
    
    # Check API availability
    if ! curl -f "$TSIOT_API_BASE_URL/health" >/dev/null 2>&1; then
        log_error "API service is not available at $TSIOT_API_BASE_URL"
        exit 1
    fi
    
    # Check UI availability (if applicable)
    if [[ -n "$TSIOT_UI_BASE_URL" ]]; then
        if ! curl -f "$TSIOT_UI_BASE_URL" >/dev/null 2>&1; then
            log_warning "UI service may not be available at $TSIOT_UI_BASE_URL"
        fi
    fi
    
    log_success "Remote environment verification completed"
}

# Setup test data
setup_test_data() {
    log_info "Setting up test data..."
    
    # Create test users and API keys
    create_test_users
    
    # Load test datasets
    load_test_datasets
    
    # Setup test configurations
    setup_test_configurations
    
    log_success "Test data setup completed"
}

# Create test users
create_test_users() {
    log_info "Creating test users..."
    
    # Create API test user
    curl -X POST "$TSIOT_API_BASE_URL/auth/users" \
        -H "Content-Type: application/json" \
        -d '{
            "username": "e2e-test-user",
            "email": "e2e-test@example.com",
            "password": "E2ETestPassword123!",
            "roles": ["user", "generator"]
        }' >/dev/null 2>&1 || true
    
    # Create admin test user
    curl -X POST "$TSIOT_API_BASE_URL/auth/users" \
        -H "Content-Type: application/json" \
        -d '{
            "username": "e2e-admin-user",
            "email": "e2e-admin@example.com",
            "password": "E2EAdminPassword123!",
            "roles": ["admin", "user", "generator"]
        }' >/dev/null 2>&1 || true
}

# Load test datasets
load_test_datasets() {
    log_info "Loading test datasets..."
    
    if [[ -d "$FIXTURES_DIR/timeseries" ]]; then
        for dataset in "$FIXTURES_DIR/timeseries"/*.json; do
            if [[ -f "$dataset" ]]; then
                curl -X POST "$TSIOT_API_BASE_URL/datasets" \
                    -H "Content-Type: application/json" \
                    -d "@$dataset" >/dev/null 2>&1 || true
            fi
        done
    fi
}

# Setup test configurations
setup_test_configurations() {
    log_info "Setting up test configurations..."
    
    if [[ -d "$FIXTURES_DIR/configs" ]]; then
        for config in "$FIXTURES_DIR/configs"/*.json; do
            if [[ -f "$config" ]]; then
                curl -X POST "$TSIOT_API_BASE_URL/configurations" \
                    -H "Content-Type: application/json" \
                    -d "@$config" >/dev/null 2>&1 || true
            fi
        done
    fi
}

# Run data generation workflow scenario
run_data_generation_workflow() {
    log_info "Running data generation workflow scenario..."
    
    local scenario_log="$LOG_DIR/e2e-data-generation-workflow.log"
    
    # Execute Go test for this scenario
    go test -tags e2e -timeout "$TEST_TIMEOUT" -v \
        "$PROJECT_ROOT/tests/e2e/scenarios/" \
        -run TestDataGenerationWorkflow \
        -environment="$ENVIRONMENT" \
        -api-url="$TSIOT_API_BASE_URL" \
        -capture-logs="$CAPTURE_LOGS" 2>&1 | tee "$scenario_log"
}

# Run privacy preservation pipeline scenario
run_privacy_preservation_pipeline() {
    log_info "Running privacy preservation pipeline scenario..."
    
    local scenario_log="$LOG_DIR/e2e-privacy-preservation.log"
    
    go test -tags e2e -timeout "$TEST_TIMEOUT" -v \
        "$PROJECT_ROOT/tests/e2e/scenarios/" \
        -run TestPrivacyPreservationPipeline \
        -environment="$ENVIRONMENT" \
        -api-url="$TSIOT_API_BASE_URL" \
        -capture-logs="$CAPTURE_LOGS" 2>&1 | tee "$scenario_log"
}

# Run real-time streaming scenario
run_real_time_streaming() {
    log_info "Running real-time streaming scenario..."
    
    local scenario_log="$LOG_DIR/e2e-real-time-streaming.log"
    
    go test -tags e2e -timeout "$TEST_TIMEOUT" -v \
        "$PROJECT_ROOT/tests/e2e/scenarios/" \
        -run TestRealTimeStreaming \
        -environment="$ENVIRONMENT" \
        -api-url="$TSIOT_API_BASE_URL" \
        -capture-logs="$CAPTURE_LOGS" 2>&1 | tee "$scenario_log"
}

# Run batch processing scenario
run_batch_processing() {
    log_info "Running batch processing scenario..."
    
    local scenario_log="$LOG_DIR/e2e-batch-processing.log"
    
    go test -tags e2e -timeout "$TEST_TIMEOUT" -v \
        "$PROJECT_ROOT/tests/e2e/scenarios/" \
        -run TestBatchProcessing \
        -environment="$ENVIRONMENT" \
        -api-url="$TSIOT_API_BASE_URL" \
        -capture-logs="$CAPTURE_LOGS" 2>&1 | tee "$scenario_log"
}

# Run multi-tenant isolation scenario
run_multi_tenant_isolation() {
    log_info "Running multi-tenant isolation scenario..."
    
    local scenario_log="$LOG_DIR/e2e-multi-tenant-isolation.log"
    
    go test -tags e2e -timeout "$TEST_TIMEOUT" -v \
        "$PROJECT_ROOT/tests/e2e/scenarios/" \
        -run TestMultiTenantIsolation \
        -environment="$ENVIRONMENT" \
        -api-url="$TSIOT_API_BASE_URL" \
        -capture-logs="$CAPTURE_LOGS" 2>&1 | tee "$scenario_log"
}

# Run disaster recovery scenario
run_disaster_recovery() {
    log_info "Running disaster recovery scenario..."
    
    local scenario_log="$LOG_DIR/e2e-disaster-recovery.log"
    
    go test -tags e2e -timeout "$TEST_TIMEOUT" -v \
        "$PROJECT_ROOT/tests/e2e/scenarios/" \
        -run TestDisasterRecovery \
        -environment="$ENVIRONMENT" \
        -api-url="$TSIOT_API_BASE_URL" \
        -capture-logs="$CAPTURE_LOGS" 2>&1 | tee "$scenario_log"
}

# Run performance scaling scenario
run_performance_scaling() {
    log_info "Running performance scaling scenario..."
    
    local scenario_log="$LOG_DIR/e2e-performance-scaling.log"
    
    go test -tags e2e -timeout "$TEST_TIMEOUT" -v \
        "$PROJECT_ROOT/tests/e2e/scenarios/" \
        -run TestPerformanceScaling \
        -environment="$ENVIRONMENT" \
        -api-url="$TSIOT_API_BASE_URL" \
        -capture-logs="$CAPTURE_LOGS" 2>&1 | tee "$scenario_log"
}

# Run all scenarios
run_all_scenarios() {
    log_info "Running all E2E scenarios..."
    
    if [[ "$PARALLEL" == "true" ]]; then
        run_scenarios_parallel
    else
        run_scenarios_sequential
    fi
}

# Run scenarios sequentially
run_scenarios_sequential() {
    for scenario in "${TEST_SCENARIOS[@]}"; do
        case "$scenario" in
            "data_generation_workflow")
                run_data_generation_workflow
                ;;
            "privacy_preservation_pipeline")
                run_privacy_preservation_pipeline
                ;;
            "real_time_streaming")
                run_real_time_streaming
                ;;
            "batch_processing")
                run_batch_processing
                ;;
            "multi_tenant_isolation")
                run_multi_tenant_isolation
                ;;
            "disaster_recovery")
                run_disaster_recovery
                ;;
            "performance_scaling")
                run_performance_scaling
                ;;
        esac
    done
}

# Run scenarios in parallel
run_scenarios_parallel() {
    log_info "Running scenarios in parallel with $E2E_PARALLEL_WORKERS workers"
    
    local pids=()
    local worker_count=0
    
    for scenario in "${TEST_SCENARIOS[@]}"; do
        # Wait if we've reached the worker limit
        while [[ $worker_count -ge $E2E_PARALLEL_WORKERS ]]; do
            wait -n
            worker_count=$((worker_count - 1))
        done
        
        # Start scenario in background
        case "$scenario" in
            "data_generation_workflow")
                run_data_generation_workflow &
                ;;
            "privacy_preservation_pipeline")
                run_privacy_preservation_pipeline &
                ;;
            "real_time_streaming")
                run_real_time_streaming &
                ;;
            "batch_processing")
                run_batch_processing &
                ;;
            "multi_tenant_isolation")
                run_multi_tenant_isolation &
                ;;
            "disaster_recovery")
                run_disaster_recovery &
                ;;
            "performance_scaling")
                run_performance_scaling &
                ;;
        esac
        
        pids+=($!)
        worker_count=$((worker_count + 1))
    done
    
    # Wait for all scenarios to complete
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
}

# Cleanup test environment
cleanup_environment() {
    if [[ "$CLEANUP" == "false" ]]; then
        log_info "Skipping cleanup (use --cleanup to enable)"
        return 0
    fi
    
    log_info "Cleaning up test environment..."
    
    # Clean up test data
    cleanup_test_data
    
    # Stop local services if running locally
    if [[ "$ENVIRONMENT" == "local" ]]; then
        cleanup_local_services
    fi
    
    log_success "Cleanup completed"
}

# Cleanup test data
cleanup_test_data() {
    log_info "Cleaning up test data..."
    
    # Delete test users
    curl -X DELETE "$TSIOT_API_BASE_URL/auth/users/e2e-test-user" >/dev/null 2>&1 || true
    curl -X DELETE "$TSIOT_API_BASE_URL/auth/users/e2e-admin-user" >/dev/null 2>&1 || true
    
    # Delete test datasets
    curl -X DELETE "$TSIOT_API_BASE_URL/datasets?prefix=e2e-test" >/dev/null 2>&1 || true
    
    # Delete test configurations
    curl -X DELETE "$TSIOT_API_BASE_URL/configurations?prefix=e2e-test" >/dev/null 2>&1 || true
}

# Cleanup local services
cleanup_local_services() {
    log_info "Stopping local services..."
    
    if [[ -f "$PROJECT_ROOT/deployments/docker/docker-compose.yml" ]]; then
        cd "$PROJECT_ROOT/deployments/docker"
        docker-compose down -v
    fi
}

# Generate E2E test report
generate_report() {
    log_info "Generating E2E test report..."
    
    local report_file="$REPORT_DIR/e2e-test-report.json"
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    
    cat > "$report_file" << EOF
{
    "timestamp": "$timestamp",
    "test_type": "e2e",
    "environment": "$ENVIRONMENT",
    "browser": "$BROWSER",
    "headless": $HEADLESS,
    "parallel": $PARALLEL,
    "timeout": "$TEST_TIMEOUT",
    "scenario": "$SCENARIO",
    "capture_logs": $CAPTURE_LOGS,
    "api_base_url": "$TSIOT_API_BASE_URL",
    "ui_base_url": "$TSIOT_UI_BASE_URL",
    "logs": {
        "data_generation_workflow": "$LOG_DIR/e2e-data-generation-workflow.log",
        "privacy_preservation": "$LOG_DIR/e2e-privacy-preservation.log",
        "real_time_streaming": "$LOG_DIR/e2e-real-time-streaming.log",
        "batch_processing": "$LOG_DIR/e2e-batch-processing.log",
        "multi_tenant_isolation": "$LOG_DIR/e2e-multi-tenant-isolation.log",
        "disaster_recovery": "$LOG_DIR/e2e-disaster-recovery.log",
        "performance_scaling": "$LOG_DIR/e2e-performance-scaling.log"
    },
    "artifacts": {
        "screenshots": "$REPORT_DIR/screenshots",
        "videos": "$REPORT_DIR/videos"
    }
}
EOF
    
    log_success "E2E test report generated: $report_file"
}

# Main execution
main() {
    log_info "Starting TSIoT E2E tests"
    log_info "Environment: $ENVIRONMENT"
    log_info "Browser: $BROWSER"
    log_info "Headless: $HEADLESS"
    log_info "Parallel: $PARALLEL"
    log_info "Timeout: $TEST_TIMEOUT"
    
    create_directories
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Verify Go is available
    if ! command -v go >/dev/null 2>&1; then
        log_error "Go is not installed or not in PATH"
        exit 1
    fi
    
    # Setup environment
    setup_environment
    
    # Setup test data
    setup_test_data
    
    # Run scenarios
    if [[ -n "$SCENARIO" ]]; then
        case "$SCENARIO" in
            "data_generation_workflow")
                run_data_generation_workflow
                ;;
            "privacy_preservation_pipeline")
                run_privacy_preservation_pipeline
                ;;
            "real_time_streaming")
                run_real_time_streaming
                ;;
            "batch_processing")
                run_batch_processing
                ;;
            "multi_tenant_isolation")
                run_multi_tenant_isolation
                ;;
            "disaster_recovery")
                run_disaster_recovery
                ;;
            "performance_scaling")
                run_performance_scaling
                ;;
            *)
                log_error "Unknown scenario: $SCENARIO"
                exit 1
                ;;
        esac
    else
        run_all_scenarios
    fi
    
    # Generate report
    generate_report
    
    # Cleanup
    cleanup_environment
    
    log_success "E2E tests completed successfully"
}

# Trap cleanup on exit
trap cleanup_environment EXIT

# Execute main function
main "$@"