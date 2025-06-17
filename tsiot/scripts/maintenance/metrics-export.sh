#!/bin/bash

# TSIoT Metrics Export Script
# Exports metrics and monitoring data

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXPORT_DIR="$PROJECT_ROOT/exports/metrics"

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
METRICS_URL="${METRICS_URL:-http://localhost:9090}"
TIME_RANGE="24h"
FORMAT="json"

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --metrics-url URL   Prometheus metrics URL [default: $METRICS_URL]
    --time-range RANGE  Time range for export [default: $TIME_RANGE]
    --format FORMAT     Export format (json|csv|prometheus) [default: $FORMAT]
    --output-dir DIR    Output directory [default: $EXPORT_DIR]
    --compress          Compress exported files
    -h, --help          Show this help

Examples:
    $0                              # Export last 24h metrics
    $0 --time-range 7d --compress   # Export last 7 days, compressed
    $0 --format csv                 # Export as CSV
EOF
}

COMPRESS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --metrics-url) METRICS_URL="$2"; shift 2 ;;
        --time-range) TIME_RANGE="$2"; shift 2 ;;
        --format) FORMAT="$2"; shift 2 ;;
        --output-dir) EXPORT_DIR="$2"; shift 2 ;;
        --compress) COMPRESS=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) log_error "Unknown option: $1"; usage; exit 1 ;;
    esac
done

export_prometheus_metrics() {
    log_info "Exporting Prometheus metrics..."
    
    local output_file="$EXPORT_DIR/prometheus_metrics_$(date +%Y%m%d_%H%M%S).$FORMAT"
    
    # Common metrics to export
    local metrics=(
        "tsiot_requests_total"
        "tsiot_request_duration_seconds"
        "tsiot_active_connections"
        "tsiot_cpu_usage"
        "tsiot_memory_usage"
        "tsiot_disk_usage"
    )
    
    case "$FORMAT" in
        "json")
            {
                echo '{"metrics": ['
                local first=true
                for metric in "${metrics[@]}"; do
                    if [[ "$first" == "true" ]]; then
                        first=false
                    else
                        echo ','
                    fi
                    
                    if command -v curl >/dev/null; then
                        curl -s "$METRICS_URL/api/v1/query?query=$metric" || echo '{}'
                    fi
                done
                echo ']}
            } > "$output_file"
            ;;
        "csv")
            {
                echo "metric,timestamp,value,labels"
                for metric in "${metrics[@]}"; do
                    if command -v curl >/dev/null; then
                        curl -s "$METRICS_URL/api/v1/query?query=$metric" | \
                            jq -r '.data.result[] | [.metric.__name__, .value[0], .value[1], (.metric | del(.__name__) | to_entries | map("\(.key)=\(.value)") | join(","))] | @csv' 2>/dev/null || true
                    fi
                done
            } > "$output_file"
            ;;
        "prometheus")
            if command -v curl >/dev/null; then
                curl -s "$METRICS_URL/metrics" > "$output_file"
            fi
            ;;
    esac
    
    if [[ -f "$output_file" ]]; then
        if [[ "$COMPRESS" == "true" ]]; then
            gzip "$output_file"
            output_file="${output_file}.gz"
        fi
        
        log_success "Metrics exported: $output_file"
        echo "$output_file"
    else
        log_error "Failed to export metrics"
        return 1
    fi
}

export_system_metrics() {
    log_info "Exporting system metrics..."
    
    local output_file="$EXPORT_DIR/system_metrics_$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$output_file" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "system": {
        "hostname": "$(hostname)",
        "uptime": "$(uptime)",
        "load_average": "$(uptime | awk -F'load average:' '{print $2}')",
        "memory": $(free -m 2>/dev/null | awk 'NR==2{printf "{\"total\":%s,\"used\":%s,\"free\":%s}", $2,$3,$4}' || echo '{}'),
        "disk": $(df -h / | awk 'NR==2{printf "{\"size\":\"%s\",\"used\":\"%s\",\"available\":\"%s\",\"use_percent\":\"%s\"}", $2,$3,$4,$5}' || echo '{}')
    }
}
EOF
    
    if [[ "$COMPRESS" == "true" ]]; then
        gzip "$output_file"
        output_file="${output_file}.gz"
    fi
    
    log_success "System metrics exported: $output_file"
    echo "$output_file"
}

main() {
    log_info "Starting metrics export"
    log_info "Metrics URL: $METRICS_URL"
    log_info "Time range: $TIME_RANGE"
    log_info "Format: $FORMAT"
    log_info "Output directory: $EXPORT_DIR"
    
    # Create export directory
    mkdir -p "$EXPORT_DIR"
    
    local exported_files=()
    
    # Export Prometheus metrics
    if exported_file=$(export_prometheus_metrics 2>/dev/null); then
        exported_files+=("$exported_file")
    fi
    
    # Export system metrics
    if exported_file=$(export_system_metrics); then
        exported_files+=("$exported_file")
    fi
    
    # Create export manifest
    local manifest_file="$EXPORT_DIR/export_manifest_$(date +%Y%m%d_%H%M%S).json"
    cat > "$manifest_file" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "time_range": "$TIME_RANGE",
    "format": "$FORMAT",
    "compressed": $COMPRESS,
    "files": $(printf '%s\n' "${exported_files[@]}" | jq -R . | jq -s .)
}
EOF
    
    log_success "Metrics export completed"
    log_info "Manifest: $manifest_file"
    log_info "Files exported: ${#exported_files[@]}"
}

main "$@"