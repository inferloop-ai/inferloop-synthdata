#!/bin/bash

# TSIoT Cleanup Script
# Cleans up temporary files, logs, and old data

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

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --type TYPE        Cleanup type (logs|cache|temp|docker|all) [default: all]
    --age DAYS         Delete files older than N days [default: 30]
    --dry-run          Show what would be deleted without executing
    --force            Force cleanup without confirmation
    -v, --verbose      Enable verbose output
    -h, --help         Show this help

Examples:
    $0                     # Clean all with 30-day retention
    $0 --type logs --age 7 # Clean logs older than 7 days
    $0 --dry-run           # Preview cleanup actions
EOF
}

# Configuration
CLEANUP_TYPE="all"
AGE_DAYS=30
DRY_RUN=false
FORCE=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --type) CLEANUP_TYPE="$2"; shift 2 ;;
        --age) AGE_DAYS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --force) FORCE=true; shift ;;
        -v|--verbose) VERBOSE=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) log_error "Unknown option: $1"; usage; exit 1 ;;
    esac
done

cleanup_logs() {
    log_info "Cleaning up log files older than $AGE_DAYS days..."
    
    local log_dirs=(
        "$PROJECT_ROOT/logs"
        "/var/log/tsiot"
        "/tmp/tsiot-logs"
    )
    
    for dir in "${log_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            if [[ "$DRY_RUN" == "true" ]]; then
                find "$dir" -name "*.log" -type f -mtime +"$AGE_DAYS" -print
            else
                find "$dir" -name "*.log" -type f -mtime +"$AGE_DAYS" -delete
            fi
        fi
    done
    
    log_success "Log cleanup completed"
}

cleanup_cache() {
    log_info "Cleaning up cache files..."
    
    local cache_dirs=(
        "$PROJECT_ROOT/.cache"
        "$PROJECT_ROOT/tmp"
        "/tmp/tsiot"
    )
    
    for dir in "${cache_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            if [[ "$DRY_RUN" == "true" ]]; then
                echo "Would remove: $dir/*"
            else
                rm -rf "$dir"/*
            fi
        fi
    done
    
    log_success "Cache cleanup completed"
}

cleanup_temp() {
    log_info "Cleaning up temporary files..."
    
    local temp_patterns=(
        "$PROJECT_ROOT/*.tmp"
        "$PROJECT_ROOT/**/*.bak"
        "$PROJECT_ROOT/**/*~"
    )
    
    for pattern in "${temp_patterns[@]}"; do
        if [[ "$DRY_RUN" == "true" ]]; then
            find $pattern -type f 2>/dev/null || true
        else
            find $pattern -type f -delete 2>/dev/null || true
        fi
    done
    
    log_success "Temporary files cleanup completed"
}

cleanup_docker() {
    if ! command -v docker >/dev/null; then
        log_warning "Docker not available, skipping Docker cleanup"
        return 0
    fi
    
    log_info "Cleaning up Docker resources..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "Would run: docker system prune -f"
        echo "Would run: docker volume prune -f"
    else
        docker system prune -f
        docker volume prune -f
    fi
    
    log_success "Docker cleanup completed"
}

main() {
    log_info "Starting cleanup process"
    log_info "Type: $CLEANUP_TYPE"
    log_info "Age threshold: $AGE_DAYS days"
    log_info "Dry run: $DRY_RUN"
    
    if [[ "$FORCE" == "false" ]] && [[ "$DRY_RUN" == "false" ]]; then
        read -p "Are you sure you want to proceed with cleanup? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Cleanup cancelled"
            exit 0
        fi
    fi
    
    case "$CLEANUP_TYPE" in
        "logs") cleanup_logs ;;
        "cache") cleanup_cache ;;
        "temp") cleanup_temp ;;
        "docker") cleanup_docker ;;
        "all")
            cleanup_logs
            cleanup_cache
            cleanup_temp
            cleanup_docker
            ;;
        *) log_error "Unknown cleanup type: $CLEANUP_TYPE"; exit 1 ;;
    esac
    
    log_success "Cleanup completed successfully"
}

main "$@"