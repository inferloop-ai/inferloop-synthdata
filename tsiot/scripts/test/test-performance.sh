#!/bin/bash

# TSIoT Performance Test Runner
# Runs performance and load tests for the Time Series IoT Synthetic Data Generation Platform

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs/tests"
REPORT_DIR="$PROJECT_ROOT/reports/tests"
BENCHMARKS_DIR="$PROJECT_ROOT/tests/benchmarks"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Performance test types
PERFORMANCE_TESTS=(
    "cpu_intensive"
    "memory_usage"
    "load_testing"
    "stress_testing"
    "throughput"
    "latency"
    "concurrent_users"
    "data_generation_scale"
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
    --test-type TYPE        Performance test type to run
    --duration DURATION     Test duration [default: 5m]
    --load-level LEVEL      Load level (light|medium|heavy) [default: medium]
    --concurrent-users NUM  Number of concurrent users [default: 50]
    --data-size SIZE        Data size for tests (small|medium|large) [default: medium]
    --cpu-profile           Enable CPU profiling
    --memory-profile        Enable memory profiling
    --trace-profile         Enable execution tracing
    --benchmark             Run Go benchmarks
    --load-test             Run load tests with external tools
    --report-format FORMAT  Report format (json|html|csv) [default: json]
    -v, --verbose           Enable verbose output
    -h, --help              Show this help message

Test Types:
    cpu_intensive           CPU-intensive workload tests
    memory_usage           Memory usage and garbage collection tests
    load_testing           HTTP load testing with multiple clients
    stress_testing         Stress testing under extreme conditions
    throughput             Data throughput measurement
    latency                Response latency measurement
    concurrent_users       Concurrent user simulation
    data_generation_scale  Large-scale data generation performance

Examples:
    $0                                      # Run all performance tests
    $0 --test-type cpu_intensive            # Run CPU tests only
    $0 --load-level heavy --duration 10m   # Heavy load for 10 minutes
    $0 --benchmark --cpu-profile            # Run benchmarks with profiling
    $0 --concurrent-users 100 --load-test  # Load test with 100 users

Environment Variables:
    TSIOT_PERF_ENV          Performance test environment
    TSIOT_TARGET_URL        Target URL for load tests
    PERF_TEST_TIMEOUT       Global performance test timeout
    GOMAXPROCS              Go runtime GOMAXPROCS setting
    GOGC                    Go garbage collector tuning
EOF
}

# Parse command line arguments
TEST_TYPE=""
DURATION="5m"
LOAD_LEVEL="medium"
CONCURRENT_USERS=50
DATA_SIZE="medium"
CPU_PROFILE=false
MEMORY_PROFILE=false
TRACE_PROFILE=false
BENCHMARK=false
LOAD_TEST=false
REPORT_FORMAT="json"
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --test-type)
            TEST_TYPE="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --load-level)
            LOAD_LEVEL="$2"
            shift 2
            ;;
        --concurrent-users)
            CONCURRENT_USERS="$2"
            shift 2
            ;;
        --data-size)
            DATA_SIZE="$2"
            shift 2
            ;;
        --cpu-profile)
            CPU_PROFILE=true
            shift
            ;;
        --memory-profile)
            MEMORY_PROFILE=true
            shift
            ;;
        --trace-profile)
            TRACE_PROFILE=true
            shift
            ;;
        --benchmark)
            BENCHMARK=true
            shift
            ;;
        --load-test)
            LOAD_TEST=true
            shift
            ;;
        --report-format)
            REPORT_FORMAT="$2"
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
export TSIOT_PERF_ENV="${TSIOT_PERF_ENV:-performance}"
export TSIOT_TARGET_URL="${TSIOT_TARGET_URL:-http://localhost:8080}"
export PERF_TEST_TIMEOUT="${PERF_TEST_TIMEOUT:-30m}"

# Optimize Go runtime for performance tests
export GOMAXPROCS="${GOMAXPROCS:-$(nproc)}"
export GOGC="${GOGC:-100}"

# Create necessary directories
create_directories() {
    mkdir -p "$LOG_DIR" "$REPORT_DIR" "$REPORT_DIR/profiles" "$REPORT_DIR/benchmarks"
}

# Check required tools
check_tools() {
    local tools=("go")
    
    if [[ "$LOAD_TEST" == "true" ]]; then
        tools+=("hey" "wrk" "vegeta")
    fi
    
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            log_warning "$tool is not available, some tests may be skipped"
        fi
    done
}

# Build benchmark flags
build_benchmark_flags() {
    local flags=()
    
    flags+=("-bench=.")
    flags+=("-benchtime=$DURATION")
    flags+=("-timeout=$PERF_TEST_TIMEOUT")
    
    if [[ "$VERBOSE" == "true" ]]; then
        flags+=("-v")
    fi
    
    if [[ "$CPU_PROFILE" == "true" ]]; then
        flags+=("-cpuprofile=$REPORT_DIR/profiles/cpu.prof")
    fi
    
    if [[ "$MEMORY_PROFILE" == "true" ]]; then
        flags+=("-memprofile=$REPORT_DIR/profiles/mem.prof")
    fi
    
    if [[ "$TRACE_PROFILE" == "true" ]]; then
        flags+=("-trace=$REPORT_DIR/profiles/trace.out")
    fi
    
    echo "${flags[@]}"
}

# Run CPU intensive tests
run_cpu_intensive_tests() {
    log_info "Running CPU intensive performance tests..."
    
    local test_log="$LOG_DIR/perf-cpu-intensive.log"
    local flags
    flags=$(build_benchmark_flags)
    
    # shellcheck disable=SC2086
    go test ${flags[@]} \
        "$PROJECT_ROOT/tests/benchmarks/cpu/" \
        -run=^$ 2>&1 | tee "$test_log"
    
    # Analyze CPU usage
    if [[ "$CPU_PROFILE" == "true" ]]; then
        analyze_cpu_profile
    fi
}

# Run memory usage tests
run_memory_usage_tests() {
    log_info "Running memory usage performance tests..."
    
    local test_log="$LOG_DIR/perf-memory-usage.log"
    local flags
    flags=$(build_benchmark_flags)
    
    # shellcheck disable=SC2086
    go test ${flags[@]} \
        "$PROJECT_ROOT/tests/benchmarks/memory/" \
        -run=^$ 2>&1 | tee "$test_log"
    
    # Analyze memory usage
    if [[ "$MEMORY_PROFILE" == "true" ]]; then
        analyze_memory_profile
    fi
}

# Run load testing
run_load_testing() {
    log_info "Running load tests with $CONCURRENT_USERS concurrent users..."
    
    local test_log="$LOG_DIR/perf-load-testing.log"
    
    # Determine load parameters based on load level
    local requests_per_second
    local total_requests
    
    case "$LOAD_LEVEL" in
        "light")
            requests_per_second=10
            total_requests=1000
            ;;
        "medium")
            requests_per_second=50
            total_requests=5000
            ;;
        "heavy")
            requests_per_second=100
            total_requests=10000
            ;;
    esac
    
    # Run load tests with different tools
    if command -v hey >/dev/null 2>&1; then
        run_hey_load_test "$requests_per_second" "$total_requests" 2>&1 | tee -a "$test_log"
    fi
    
    if command -v wrk >/dev/null 2>&1; then
        run_wrk_load_test 2>&1 | tee -a "$test_log"
    fi
    
    if command -v vegeta >/dev/null 2>&1; then
        run_vegeta_load_test "$requests_per_second" 2>&1 | tee -a "$test_log"
    fi
}

# Run hey load test
run_hey_load_test() {
    local rps="$1"
    local total="$2"
    
    log_info "Running hey load test: $rps RPS, $total total requests"
    
    hey -q "$rps" -n "$total" -c "$CONCURRENT_USERS" \
        -o csv "$TSIOT_TARGET_URL/health" > "$REPORT_DIR/hey-results.csv"
    
    hey -q "$rps" -n "$total" -c "$CONCURRENT_USERS" \
        "$TSIOT_TARGET_URL/health"
}

# Run wrk load test
run_wrk_load_test() {
    log_info "Running wrk load test for $DURATION"
    
    wrk -t "$CONCURRENT_USERS" -c "$CONCURRENT_USERS" -d "$DURATION" \
        --script="$BENCHMARKS_DIR/load/api-test.lua" \
        "$TSIOT_TARGET_URL" > "$REPORT_DIR/wrk-results.txt"
}

# Run vegeta load test
run_vegeta_load_test() {
    local rps="$1"
    
    log_info "Running vegeta load test: $rps RPS for $DURATION"
    
    echo "GET $TSIOT_TARGET_URL/health" | \
    vegeta attack -rate="$rps" -duration="$DURATION" | \
    vegeta report > "$REPORT_DIR/vegeta-results.txt"
    
    echo "GET $TSIOT_TARGET_URL/health" | \
    vegeta attack -rate="$rps" -duration="$DURATION" | \
    vegeta report -type=json > "$REPORT_DIR/vegeta-results.json"
}

# Run stress testing
run_stress_testing() {
    log_info "Running stress tests..."
    
    local test_log="$LOG_DIR/perf-stress-testing.log"
    local flags
    flags=$(build_benchmark_flags)
    
    # Override settings for stress testing
    export GOGC=50  # More frequent GC
    export GOMAXPROCS=1  # Single core stress
    
    # shellcheck disable=SC2086
    go test ${flags[@]} \
        "$PROJECT_ROOT/tests/benchmarks/stress/" \
        -run=^$ 2>&1 | tee "$test_log"
    
    # Reset settings
    export GOMAXPROCS="${GOMAXPROCS:-$(nproc)}"
    export GOGC="${GOGC:-100}"
}

# Run throughput tests
run_throughput_tests() {
    log_info "Running throughput performance tests..."
    
    local test_log="$LOG_DIR/perf-throughput.log"
    local flags
    flags=$(build_benchmark_flags)
    
    # shellcheck disable=SC2086
    go test ${flags[@]} \
        "$PROJECT_ROOT/tests/benchmarks/throughput/" \
        -run=^$ \
        -data-size="$DATA_SIZE" 2>&1 | tee "$test_log"
}

# Run latency tests
run_latency_tests() {
    log_info "Running latency performance tests..."
    
    local test_log="$LOG_DIR/perf-latency.log"
    local flags
    flags=$(build_benchmark_flags)
    
    # shellcheck disable=SC2086
    go test ${flags[@]} \
        "$PROJECT_ROOT/tests/benchmarks/latency/" \
        -run=^$ 2>&1 | tee "$test_log"
}

# Run concurrent users tests
run_concurrent_users_tests() {
    log_info "Running concurrent users performance tests..."
    
    local test_log="$LOG_DIR/perf-concurrent-users.log"
    local flags
    flags=$(build_benchmark_flags)
    
    # shellcheck disable=SC2086
    go test ${flags[@]} \
        "$PROJECT_ROOT/tests/benchmarks/concurrent/" \
        -run=^$ \
        -concurrent-users="$CONCURRENT_USERS" 2>&1 | tee "$test_log"
}

# Run data generation scale tests
run_data_generation_scale_tests() {
    log_info "Running data generation scale performance tests..."
    
    local test_log="$LOG_DIR/perf-data-generation-scale.log"
    local flags
    flags=$(build_benchmark_flags)
    
    # shellcheck disable=SC2086
    go test ${flags[@]} \
        "$PROJECT_ROOT/tests/benchmarks/generation/" \
        -run=^$ \
        -data-size="$DATA_SIZE" 2>&1 | tee "$test_log"
}

# Analyze CPU profile
analyze_cpu_profile() {
    if [[ -f "$REPORT_DIR/profiles/cpu.prof" ]]; then
        log_info "Analyzing CPU profile..."
        
        go tool pprof -top "$REPORT_DIR/profiles/cpu.prof" > "$REPORT_DIR/profiles/cpu-top.txt"
        go tool pprof -svg "$REPORT_DIR/profiles/cpu.prof" > "$REPORT_DIR/profiles/cpu-graph.svg"
    fi
}

# Analyze memory profile
analyze_memory_profile() {
    if [[ -f "$REPORT_DIR/profiles/mem.prof" ]]; then
        log_info "Analyzing memory profile..."
        
        go tool pprof -top "$REPORT_DIR/profiles/mem.prof" > "$REPORT_DIR/profiles/memory-top.txt"
        go tool pprof -svg "$REPORT_DIR/profiles/mem.prof" > "$REPORT_DIR/profiles/memory-graph.svg"
    fi
}

# Analyze trace
analyze_trace() {
    if [[ -f "$REPORT_DIR/profiles/trace.out" ]]; then
        log_info "Trace analysis available at: go tool trace $REPORT_DIR/profiles/trace.out"
    fi
}

# Run all performance tests
run_all_tests() {
    log_info "Running all performance tests..."
    
    for test in "${PERFORMANCE_TESTS[@]}"; do
        case "$test" in
            "cpu_intensive")
                run_cpu_intensive_tests
                ;;
            "memory_usage")
                run_memory_usage_tests
                ;;
            "load_testing")
                if [[ "$LOAD_TEST" == "true" ]]; then
                    run_load_testing
                fi
                ;;
            "stress_testing")
                run_stress_testing
                ;;
            "throughput")
                run_throughput_tests
                ;;
            "latency")
                run_latency_tests
                ;;
            "concurrent_users")
                run_concurrent_users_tests
                ;;
            "data_generation_scale")
                run_data_generation_scale_tests
                ;;
        esac
    done
}

# Generate performance report
generate_report() {
    log_info "Generating performance test report..."
    
    local report_file="$REPORT_DIR/performance-test-report.$REPORT_FORMAT"
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    
    case "$REPORT_FORMAT" in
        "json")
            generate_json_report "$report_file" "$timestamp"
            ;;
        "html")
            generate_html_report "$report_file" "$timestamp"
            ;;
        "csv")
            generate_csv_report "$report_file" "$timestamp"
            ;;
    esac
    
    log_success "Performance test report generated: $report_file"
}

# Generate JSON report
generate_json_report() {
    local report_file="$1"
    local timestamp="$2"
    
    cat > "$report_file" << EOF
{
    "timestamp": "$timestamp",
    "test_type": "performance",
    "environment": "$TSIOT_PERF_ENV",
    "configuration": {
        "duration": "$DURATION",
        "load_level": "$LOAD_LEVEL",
        "concurrent_users": $CONCURRENT_USERS,
        "data_size": "$DATA_SIZE",
        "cpu_profile": $CPU_PROFILE,
        "memory_profile": $MEMORY_PROFILE,
        "trace_profile": $TRACE_PROFILE,
        "benchmark": $BENCHMARK,
        "load_test": $LOAD_TEST
    },
    "runtime": {
        "gomaxprocs": "$GOMAXPROCS",
        "gogc": "$GOGC"
    },
    "logs": {
        "cpu_intensive": "$LOG_DIR/perf-cpu-intensive.log",
        "memory_usage": "$LOG_DIR/perf-memory-usage.log",
        "load_testing": "$LOG_DIR/perf-load-testing.log",
        "stress_testing": "$LOG_DIR/perf-stress-testing.log",
        "throughput": "$LOG_DIR/perf-throughput.log",
        "latency": "$LOG_DIR/perf-latency.log",
        "concurrent_users": "$LOG_DIR/perf-concurrent-users.log",
        "data_generation_scale": "$LOG_DIR/perf-data-generation-scale.log"
    },
    "profiles": {
        "cpu": "$REPORT_DIR/profiles/cpu.prof",
        "memory": "$REPORT_DIR/profiles/mem.prof",
        "trace": "$REPORT_DIR/profiles/trace.out"
    },
    "load_test_results": {
        "hey": "$REPORT_DIR/hey-results.csv",
        "wrk": "$REPORT_DIR/wrk-results.txt",
        "vegeta": "$REPORT_DIR/vegeta-results.json"
    }
}
EOF
}

# Generate HTML report
generate_html_report() {
    local report_file="$1"
    local timestamp="$2"
    
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>TSIoT Performance Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }
        .section { margin: 20px 0; }
        .metric { background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-left: 4px solid #007cba; }
        pre { background-color: #f5f5f5; padding: 10px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="header">
        <h1>TSIoT Performance Test Report</h1>
        <p>Generated on: $timestamp</p>
        <p>Environment: $TSIOT_PERF_ENV</p>
    </div>
    
    <div class="section">
        <h2>Test Configuration</h2>
        <div class="metric">Duration: $DURATION</div>
        <div class="metric">Load Level: $LOAD_LEVEL</div>
        <div class="metric">Concurrent Users: $CONCURRENT_USERS</div>
        <div class="metric">Data Size: $DATA_SIZE</div>
    </div>
    
    <div class="section">
        <h2>Runtime Configuration</h2>
        <div class="metric">GOMAXPROCS: $GOMAXPROCS</div>
        <div class="metric">GOGC: $GOGC</div>
    </div>
    
    <div class="section">
        <h2>Test Results</h2>
        <p>Detailed results are available in the log files:</p>
        <ul>
            <li><a href="$LOG_DIR/perf-cpu-intensive.log">CPU Intensive Tests</a></li>
            <li><a href="$LOG_DIR/perf-memory-usage.log">Memory Usage Tests</a></li>
            <li><a href="$LOG_DIR/perf-load-testing.log">Load Testing</a></li>
            <li><a href="$LOG_DIR/perf-stress-testing.log">Stress Testing</a></li>
            <li><a href="$LOG_DIR/perf-throughput.log">Throughput Tests</a></li>
            <li><a href="$LOG_DIR/perf-latency.log">Latency Tests</a></li>
        </ul>
    </div>
</body>
</html>
EOF
}

# Generate CSV report
generate_csv_report() {
    local report_file="$1"
    
    cat > "$report_file" << EOF
timestamp,test_type,environment,duration,load_level,concurrent_users,data_size,cpu_profile,memory_profile,trace_profile
$(date -u +%Y-%m-%dT%H:%M:%SZ),performance,$TSIOT_PERF_ENV,$DURATION,$LOAD_LEVEL,$CONCURRENT_USERS,$DATA_SIZE,$CPU_PROFILE,$MEMORY_PROFILE,$TRACE_PROFILE
EOF
}

# Main execution
main() {
    log_info "Starting TSIoT performance tests"
    log_info "Test type: ${TEST_TYPE:-all}"
    log_info "Duration: $DURATION"
    log_info "Load level: $LOAD_LEVEL"
    log_info "Concurrent users: $CONCURRENT_USERS"
    log_info "Data size: $DATA_SIZE"
    
    create_directories
    check_tools
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Verify Go is available
    if ! command -v go >/dev/null 2>&1; then
        log_error "Go is not installed or not in PATH"
        exit 1
    fi
    
    # Run specific test or all tests
    if [[ -n "$TEST_TYPE" ]]; then
        case "$TEST_TYPE" in
            "cpu_intensive")
                run_cpu_intensive_tests
                ;;
            "memory_usage")
                run_memory_usage_tests
                ;;
            "load_testing")
                run_load_testing
                ;;
            "stress_testing")
                run_stress_testing
                ;;
            "throughput")
                run_throughput_tests
                ;;
            "latency")
                run_latency_tests
                ;;
            "concurrent_users")
                run_concurrent_users_tests
                ;;
            "data_generation_scale")
                run_data_generation_scale_tests
                ;;
            *)
                log_error "Unknown test type: $TEST_TYPE"
                exit 1
                ;;
        esac
    else
        run_all_tests
    fi
    
    # Analyze profiles if enabled
    if [[ "$CPU_PROFILE" == "true" ]]; then
        analyze_cpu_profile
    fi
    
    if [[ "$MEMORY_PROFILE" == "true" ]]; then
        analyze_memory_profile
    fi
    
    if [[ "$TRACE_PROFILE" == "true" ]]; then
        analyze_trace
    fi
    
    generate_report
    log_success "Performance tests completed successfully"
}

# Execute main function
main "$@"