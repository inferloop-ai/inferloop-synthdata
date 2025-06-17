#!/bin/bash

# TSIoT Log Rotation Script
# Rotates and manages log files

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"

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
MAX_SIZE="100M"
KEEP_ROTATIONS=5
COMPRESS=true

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --max-size SIZE    Maximum log file size [default: $MAX_SIZE]
    --keep NUM         Number of rotations to keep [default: $KEEP_ROTATIONS]
    --no-compress      Don't compress rotated logs
    --log-dir DIR      Log directory [default: $LOG_DIR]
    -h, --help         Show this help

Examples:
    $0                           # Rotate with defaults
    $0 --max-size 50M --keep 10  # Custom size and retention
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-size) MAX_SIZE="$2"; shift 2 ;;
        --keep) KEEP_ROTATIONS="$2"; shift 2 ;;
        --no-compress) COMPRESS=false; shift ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) log_error "Unknown option: $1"; usage; exit 1 ;;
    esac
done

rotate_log() {
    local log_file="$1"
    local base_name="$(basename "$log_file" .log)"
    local dir_name="$(dirname "$log_file")"
    
    if [[ ! -f "$log_file" ]]; then
        return 0
    fi
    
    local file_size
    file_size=$(stat -f%z "$log_file" 2>/dev/null || stat -c%s "$log_file" 2>/dev/null || echo 0)
    local max_bytes
    max_bytes=$(echo "$MAX_SIZE" | sed 's/M/*1024*1024/' | sed 's/K/*1024/' | bc)
    
    if [[ $file_size -gt $max_bytes ]]; then
        log_info "Rotating log file: $log_file ($(du -h "$log_file" | cut -f1))"
        
        # Shift existing rotated files
        for ((i=KEEP_ROTATIONS-1; i>=1; i--)); do
            local old_file="$dir_name/${base_name}.log.$i"
            local new_file="$dir_name/${base_name}.log.$((i+1))"
            
            if [[ "$COMPRESS" == "true" ]]; then
                old_file="${old_file}.gz"
                new_file="${new_file}.gz"
            fi
            
            if [[ -f "$old_file" ]]; then
                mv "$old_file" "$new_file"
            fi
        done
        
        # Move current log to .1
        local rotated_file="$dir_name/${base_name}.log.1"
        mv "$log_file" "$rotated_file"
        
        # Compress if enabled
        if [[ "$COMPRESS" == "true" ]]; then
            gzip "$rotated_file"
        fi
        
        # Create new empty log file
        touch "$log_file"
        
        # Remove old rotations beyond keep limit
        local old_file="$dir_name/${base_name}.log.$((KEEP_ROTATIONS+1))"
        if [[ "$COMPRESS" == "true" ]]; then
            old_file="${old_file}.gz"
        fi
        
        if [[ -f "$old_file" ]]; then
            rm -f "$old_file"
        fi
        
        log_success "Rotated: $log_file"
    fi
}

main() {
    log_info "Starting log rotation"
    log_info "Log directory: $LOG_DIR"
    log_info "Max size: $MAX_SIZE"
    log_info "Keep rotations: $KEEP_ROTATIONS"
    log_info "Compress: $COMPRESS"
    
    if [[ ! -d "$LOG_DIR" ]]; then
        log_warning "Log directory does not exist: $LOG_DIR"
        exit 0
    fi
    
    # Find and rotate log files
    local rotated_count=0
    
    while IFS= read -r -d '' log_file; do
        rotate_log "$log_file"
        ((rotated_count++))
    done < <(find "$LOG_DIR" -name "*.log" -type f -print0)
    
    log_success "Log rotation completed. Processed $rotated_count log files."
}

main "$@"