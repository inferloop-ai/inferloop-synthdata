#!/bin/bash

# TSIoT Test Runner Script
# Comprehensive test execution for the Time Series IoT Synthetic Data Generation Platform

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs/tests"
REPORT_DIR="$PROJECT_ROOT/reports/tests"

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

# Create necessary directories
create_directories() {
    mkdir -p "$LOG_DIR" "$REPORT_DIR"
}

# Print usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -t, --type TYPE         Test type to run (unit|integration|e2e|performance|all) [default: all]
    -c, --coverage          Generate code coverage report
    -v, --verbose           Enable verbose output
    -p, --parallel          Run tests in parallel where possible
    -f, --fail-fast         Stop on first test failure
    -o, --output FORMAT     Output format (text|json|junit) [default: text]
    --timeout DURATION      Test timeout duration [default: 30m]
    --exclude PATTERN       Exclude tests matching pattern
    --tags TAGS             Run only tests with specified tags
    -h, --help              Show this help message

Examples:
    $0                      # Run all tests
    $0 -t unit              # Run only unit tests
    $0 -t integration -c    # Run integration tests with coverage
    $0 -p -f                # Run tests in parallel, fail fast
    $0 --tags "slow"        # Run only tests tagged as 'slow'

Environment Variables:
    TSIOT_TEST_ENV          Test environment (development|staging|production)
    TSIOT_TEST_CONFIG       Path to test configuration file
    TSIOT_LOG_LEVEL         Log level (debug|info|warn|error)
    GO_TEST_TIMEOUT         Global test timeout
    TEST_PARALLEL_JOBS      Number of parallel test jobs
EOF
}

# Parse command line arguments
TEST_TYPE="all"
COVERAGE=false
VERBOSE=false
PARALLEL=false
FAIL_FAST=false
OUTPUT_FORMAT="text"
TEST_TIMEOUT="30m"
EXCLUDE_PATTERN=""
TEST_TAGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -f|--fail-fast)
            FAIL_FAST=true
            shift
            ;;
        -o|--output)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        --timeout)
            TEST_TIMEOUT="$2"
            shift 2
            ;;
        --exclude)
            EXCLUDE_PATTERN="$2"
            shift 2
            ;;
        --tags)
            TEST_TAGS="$2"
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

# Validate test type
if [[ ! "$TEST_TYPE" =~ ^(unit|integration|e2e|performance|all)$ ]]; then
    log_error "Invalid test type: $TEST_TYPE"
    usage
    exit 1
fi

# Set environment variables
export TSIOT_TEST_ENV="${TSIOT_TEST_ENV:-development}"
export TSIOT_LOG_LEVEL="${TSIOT_LOG_LEVEL:-info}"
export GO_TEST_TIMEOUT="${GO_TEST_TIMEOUT:-$TEST_TIMEOUT}"
export TEST_PARALLEL_JOBS="${TEST_PARALLEL_JOBS:-4}"

# Build Go test flags
build_test_flags() {
    local flags=()
    
    if [[ "$VERBOSE" == "true" ]]; then
        flags+=("-v")
    fi
    
    if [[ "$PARALLEL" == "true" ]]; then
        flags+=("-parallel" "$TEST_PARALLEL_JOBS")
    fi
    
    if [[ "$FAIL_FAST" == "true" ]]; then
        flags+=("-failfast")
    fi
    
    flags+=("-timeout" "$TEST_TIMEOUT")
    
    if [[ -n "$TEST_TAGS" ]]; then
        flags+=("-tags" "$TEST_TAGS")
    fi
    
    case "$OUTPUT_FORMAT" in
        "json")
            flags+=("-json")
            ;;
        "junit")
            # Will be handled by gotestsum if available
            ;;
    esac
    
    echo "${flags[@]}"
}

# Run unit tests
run_unit_tests() {
    log_info "Running unit tests..."
    
    local test_flags
    test_flags=$(build_test_flags)
    
    local coverage_flags=()
    if [[ "$COVERAGE" == "true" ]]; then
        coverage_flags+=("-coverprofile=$REPORT_DIR/unit-coverage.out")
        coverage_flags+=("-covermode=atomic")
    fi
    
    local exclude_flags=()
    if [[ -n "$EXCLUDE_PATTERN" ]]; then
        exclude_flags+=("-skip" "$EXCLUDE_PATTERN")
    fi
    
    # Run unit tests
    if [[ "$OUTPUT_FORMAT" == "junit" ]] && command -v gotestsum >/dev/null 2>&1; then
        gotestsum --junitfile "$REPORT_DIR/unit-tests.xml" --format testname -- \
            ${test_flags[@]} ${coverage_flags[@]} ${exclude_flags[@]} \
            "$PROJECT_ROOT/internal/..." "$PROJECT_ROOT/pkg/..."
    else
        # shellcheck disable=SC2086
        go test ${test_flags[@]} ${coverage_flags[@]} ${exclude_flags[@]} \
            "$PROJECT_ROOT/internal/..." "$PROJECT_ROOT/pkg/..." | tee "$LOG_DIR/unit-tests.log"
    fi
    
    if [[ "$COVERAGE" == "true" && -f "$REPORT_DIR/unit-coverage.out" ]]; then
        go tool cover -html="$REPORT_DIR/unit-coverage.out" -o "$REPORT_DIR/unit-coverage.html"
        go tool cover -func="$REPORT_DIR/unit-coverage.out" | tee "$REPORT_DIR/unit-coverage.txt"
        
        # Extract coverage percentage
        local coverage_pct
        coverage_pct=$(go tool cover -func="$REPORT_DIR/unit-coverage.out" | grep "total:" | awk '{print $3}')
        log_info "Unit test coverage: $coverage_pct"
    fi
}

# Run integration tests
run_integration_tests() {
    log_info "Running integration tests..."
    
    # Check if required services are running
    if ! "$SCRIPT_DIR/test-integration.sh" --check-services; then
        log_warning "Starting required services for integration tests"
        "$SCRIPT_DIR/test-integration.sh" --setup
    fi
    
    "$SCRIPT_DIR/test-integration.sh" "$@"
}

# Run end-to-end tests
run_e2e_tests() {
    log_info "Running end-to-end tests..."
    "$SCRIPT_DIR/test-e2e.sh" "$@"
}

# Run performance tests
run_performance_tests() {
    log_info "Running performance tests..."
    "$SCRIPT_DIR/test-performance.sh" "$@"
}

# Generate test summary
generate_summary() {
    log_info "Generating test summary..."
    
    local summary_file="$REPORT_DIR/test-summary.json"
    cat > "$summary_file" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "test_type": "$TEST_TYPE",
    "environment": "$TSIOT_TEST_ENV",
    "coverage_enabled": $COVERAGE,
    "parallel_execution": $PARALLEL,
    "fail_fast": $FAIL_FAST,
    "timeout": "$TEST_TIMEOUT",
    "exclude_pattern": "$EXCLUDE_PATTERN",
    "tags": "$TEST_TAGS",
    "reports": {
        "log_directory": "$LOG_DIR",
        "report_directory": "$REPORT_DIR"
    }
}
EOF
    
    log_success "Test summary generated: $summary_file"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up test artifacts..."
    
    # Clean up temporary files
    find "$PROJECT_ROOT" -name "*.test" -type f -delete 2>/dev/null || true
    find "$PROJECT_ROOT" -name "coverage.tmp" -type f -delete 2>/dev/null || true
    
    # Compress old logs
    if [[ -d "$LOG_DIR" ]]; then
        find "$LOG_DIR" -name "*.log" -mtime +7 -exec gzip {} \; 2>/dev/null || true
    fi
}

# Trap cleanup on exit
trap cleanup EXIT

# Main execution
main() {
    log_info "Starting TSIoT test execution"
    log_info "Test type: $TEST_TYPE"
    log_info "Environment: $TSIOT_TEST_ENV"
    log_info "Coverage: $COVERAGE"
    log_info "Parallel: $PARALLEL"
    log_info "Output format: $OUTPUT_FORMAT"
    
    create_directories
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Verify Go is available
    if ! command -v go >/dev/null 2>&1; then
        log_error "Go is not installed or not in PATH"
        exit 1
    fi
    
    # Download dependencies if needed
    log_info "Ensuring dependencies are up to date..."
    go mod download
    go mod tidy
    
    case "$TEST_TYPE" in
        "unit")
            run_unit_tests
            ;;
        "integration")
            run_integration_tests
            ;;
        "e2e")
            run_e2e_tests
            ;;
        "performance")
            run_performance_tests
            ;;
        "all")
            run_unit_tests
            run_integration_tests
            run_e2e_tests
            run_performance_tests
            ;;
    esac
    
    generate_summary
    log_success "Test execution completed successfully"
}

# Execute main function
main "$@"