#!/bin/bash

# TSIoT Health Check Script
# Performs comprehensive health checks for the Time Series IoT platform

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"; }

# Configuration
API_URL="${API_URL:-http://localhost:8080}"
TIMEOUT="${TIMEOUT:-30}"
VERBOSE=false

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --api-url URL      API endpoint URL [default: $API_URL]
    --timeout SECONDS  Request timeout [default: $TIMEOUT]
    --format FORMAT    Output format (text|json) [default: text]
    -v, --verbose      Enable verbose output
    -h, --help         Show this help

Examples:
    $0                              # Basic health check
    $0 --api-url http://prod:8080   # Check production
    $0 --format json               # JSON output
EOF
}

# Parse arguments
FORMAT="text"
while [[ $# -gt 0 ]]; do
    case $1 in
        --api-url) API_URL="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --format) FORMAT="$2"; shift 2 ;;
        -v|--verbose) VERBOSE=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) log_error "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# Health check functions
check_api_health() {
    local status="unknown"
    local response_time=0
    local error=""
    
    if command -v curl >/dev/null; then
        local start_time=$(date +%s.%N)
        if response=$(curl -s --max-time "$TIMEOUT" "$API_URL/health" 2>&1); then
            local end_time=$(date +%s.%N)
            response_time=$(echo "$end_time - $start_time" | bc)
            
            if echo "$response" | grep -q "healthy\|ok"; then
                status="healthy"
            else
                status="unhealthy"
                error="Invalid response: $response"
            fi
        else
            status="unreachable"
            error="Connection failed: $response"
        fi
    else
        status="error"
        error="curl command not available"
    fi
    
    echo "$status|$response_time|$error"
}

check_database_health() {
    local status="unknown"
    local error=""
    
    # Check if database is accessible
    if command -v pg_isready >/dev/null; then
        if pg_isready -h "${DB_HOST:-localhost}" -p "${DB_PORT:-5432}" >/dev/null 2>&1; then
            status="healthy"
        else
            status="unreachable"
            error="PostgreSQL not responding"
        fi
    else
        status="unknown"
        error="pg_isready not available"
    fi
    
    echo "$status|$error"
}

check_system_resources() {
    local cpu_usage=0
    local memory_usage=0
    local disk_usage=0
    
    # CPU usage
    if command -v top >/dev/null; then
        cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//' || echo "0")
    fi
    
    # Memory usage
    if command -v free >/dev/null; then
        memory_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    elif command -v vm_stat >/dev/null; then
        # macOS
        memory_usage=$(vm_stat | awk '/Pages free:/{free=$3} /Pages active:/{active=$3} /Pages inactive:/{inactive=$3} /Pages speculative:/{spec=$3} /Pages wired down:/{wired=$4} END {total=free+active+inactive+spec+wired; used=active+inactive+wired; printf "%.1f", used/total*100}')
    fi
    
    # Disk usage
    if command -v df >/dev/null; then
        disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    fi
    
    echo "$cpu_usage|$memory_usage|$disk_usage"
}

# Main health check
main() {
    log_info "Starting TSIoT health check"
    
    # API Health
    log_info "Checking API health..."
    IFS='|' read -r api_status response_time api_error <<< "$(check_api_health)"
    
    # Database Health
    log_info "Checking database connectivity..."
    IFS='|' read -r db_status db_error <<< "$(check_database_health)"
    
    # System Resources
    log_info "Checking system resources..."
    IFS='|' read -r cpu_usage memory_usage disk_usage <<< "$(check_system_resources)"
    
    # Generate report
    case "$FORMAT" in
        "json")
            cat << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "overall_status": "$([ "$api_status" = "healthy" ] && echo "healthy" || echo "unhealthy")",
    "api": {
        "status": "$api_status",
        "response_time": "$response_time",
        "url": "$API_URL",
        "error": "$api_error"
    },
    "database": {
        "status": "$db_status",
        "error": "$db_error"
    },
    "system": {
        "cpu_usage": "$cpu_usage",
        "memory_usage": "$memory_usage",
        "disk_usage": "$disk_usage"
    }
}
EOF
            ;;
        *)
            echo "TSIoT Health Check Report"
            echo "========================"
            echo "Timestamp: $(date)"
            echo ""
            echo "API Health:"
            echo "  Status: $api_status"
            echo "  URL: $API_URL"
            echo "  Response Time: ${response_time}s"
            [[ -n "$api_error" ]] && echo "  Error: $api_error"
            echo ""
            echo "Database:"
            echo "  Status: $db_status"
            [[ -n "$db_error" ]] && echo "  Error: $db_error"
            echo ""
            echo "System Resources:"
            echo "  CPU Usage: ${cpu_usage}%"
            echo "  Memory Usage: ${memory_usage}%"
            echo "  Disk Usage: ${disk_usage}%"
            ;;
    esac
    
    # Exit with error if unhealthy
    if [[ "$api_status" != "healthy" ]]; then
        exit 1
    fi
}

main "$@"