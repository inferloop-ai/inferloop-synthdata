#!/bin/bash

# TSIOT CLI - Batch Processing Script
# This script demonstrates batch processing capabilities for large-scale synthetic data generation

set -e

# Configuration
TSIOT_CLI=${TSIOT_CLI:-"./bin/tsiot"}
CONFIG_DIR=${CONFIG_DIR:-"./batch_configs"}
OUTPUT_DIR=${OUTPUT_DIR:-"./batch_output"}
NUM_WORKERS=${NUM_WORKERS:-4}
BATCH_SIZE=${BATCH_SIZE:-10}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Setup directories
setup_directories() {
    log_info "Setting up directories..."
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/logs"
    mkdir -p "$OUTPUT_DIR/metrics"
    mkdir -p "$OUTPUT_DIR/data"
    log_success "Directories created"
}

# Generate batch configuration files
generate_batch_configs() {
    log_info "Generating batch configuration files..."
    
    # IoT Sensor Data Configuration
    cat > "$CONFIG_DIR/iot_sensors.yaml" << EOF
batch:
  name: "iot_sensors_batch"
  description: "Simulated IoT sensor data for various devices"
  output_dir: "$OUTPUT_DIR/data/iot_sensors"
  parallel_workers: $NUM_WORKERS
  series:
    - name: "temperature_sensor_1"
      type: "arima"
      length: 8760  # 1 year hourly data
      parameters:
        ar_params: [0.8, -0.2]
        ma_params: [0.3]
        mean: 22.5
        variance: 16
        seasonal: true
        seasonal_period: 24
    - name: "humidity_sensor_1"
      type: "lstm"
      length: 8760
      parameters:
        trend: 0.02
        seasonality: 24
        noise: 0.05
        base_value: 65
    - name: "pressure_sensor_1"
      type: "random_walk"
      length: 8760
      parameters:
        drift: 0.001
        volatility: 0.5
        initial_value: 1013.25
    - name: "vibration_sensor_1"
      type: "arima"
      length: 8760
      parameters:
        ar_params: [0.6, -0.1]
        ma_params: [0.2, -0.1]
        mean: 0.5
        variance: 0.25
EOF

    # Financial Data Configuration
    cat > "$CONFIG_DIR/financial_data.yaml" << EOF
batch:
  name: "financial_markets_batch"
  description: "Synthetic financial market data"
  output_dir: "$OUTPUT_DIR/data/financial"
  parallel_workers: $NUM_WORKERS
  series:
    - name: "stock_price_tech"
      type: "lstm"
      length: 2520  # 10 years daily data
      parameters:
        trend: 0.08
        volatility: 0.25
        base_value: 100
        complexity: "high"
    - name: "stock_price_energy"
      type: "lstm"
      length: 2520
      parameters:
        trend: 0.05
        volatility: 0.30
        base_value: 50
        complexity: "medium"
    - name: "crypto_btc"
      type: "random_walk"
      length: 2520
      parameters:
        drift: 0.15
        volatility: 0.80
        initial_value: 50000
    - name: "forex_eurusd"
      type: "arima"
      length: 2520
      parameters:
        ar_params: [0.95, -0.05]
        ma_params: [0.1]
        mean: 1.1
        variance: 0.01
EOF

    # Web Analytics Configuration
    cat > "$CONFIG_DIR/web_analytics.yaml" << EOF
batch:
  name: "web_analytics_batch"
  description: "Web traffic and user behavior metrics"
  output_dir: "$OUTPUT_DIR/data/web_analytics"
  parallel_workers: $NUM_WORKERS
  series:
    - name: "page_views_daily"
      type: "lstm"
      length: 365
      parameters:
        trend: 0.1
        seasonality: 7  # Weekly pattern
        noise: 0.2
        base_value: 10000
    - name: "unique_visitors"
      type: "arima"
      length: 365
      parameters:
        ar_params: [0.7, -0.1]
        ma_params: [0.2]
        seasonal: true
        seasonal_period: 7
        mean: 5000
        variance: 1000000
    - name: "conversion_rate"
      type: "lstm"
      length: 365
      parameters:
        trend: 0.02
        seasonality: 7
        noise: 0.05
        base_value: 0.025
EOF

    log_success "Batch configuration files generated"
}

# Execute batch processing
execute_batch() {
    local config_file=$1
    local batch_name=$(basename "$config_file" .yaml)
    
    log_info "Executing batch: $batch_name"
    
    # Start time
    start_time=$(date +%s)
    
    # Execute batch with progress monitoring
    $TSIOT_CLI generate batch \
        --config "$config_file" \
        --concurrent $NUM_WORKERS \
        --progress \
        --log-file "$OUTPUT_DIR/logs/${batch_name}.log" \
        --metrics-file "$OUTPUT_DIR/metrics/${batch_name}_metrics.json" || {
        log_error "Batch execution failed: $batch_name"
        return 1
    }
    
    # End time and duration
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    log_success "Batch completed: $batch_name (${duration}s)"
    
    # Generate batch summary
    generate_batch_summary "$batch_name" "$duration"
}

# Generate batch summary
generate_batch_summary() {
    local batch_name=$1
    local duration=$2
    
    log_info "Generating summary for batch: $batch_name"
    
    # Count generated files
    local data_dir="$OUTPUT_DIR/data/${batch_name}"
    local file_count=0
    local total_size=0
    
    if [ -d "$data_dir" ]; then
        file_count=$(find "$data_dir" -type f \( -name "*.json" -o -name "*.csv" \) | wc -l)
        total_size=$(du -sh "$data_dir" | cut -f1)
    fi
    
    # Create summary report
    cat > "$OUTPUT_DIR/metrics/${batch_name}_summary.md" << EOF
# Batch Processing Summary: $batch_name

## Execution Details
- **Start Time**: $(date -d "@$(($(date +%s) - duration))" '+%Y-%m-%d %H:%M:%S')
- **End Time**: $(date '+%Y-%m-%d %H:%M:%S')
- **Duration**: ${duration} seconds
- **Workers**: $NUM_WORKERS

## Output Statistics
- **Files Generated**: $file_count
- **Total Size**: $total_size
- **Output Directory**: $data_dir

## Log Files
- **Execution Log**: $OUTPUT_DIR/logs/${batch_name}.log
- **Metrics**: $OUTPUT_DIR/metrics/${batch_name}_metrics.json

## Generated Files
EOF

    # List generated files
    if [ -d "$data_dir" ]; then
        echo "### Data Files" >> "$OUTPUT_DIR/metrics/${batch_name}_summary.md"
        find "$data_dir" -type f \( -name "*.json" -o -name "*.csv" \) -exec basename {} \; | sort >> "$OUTPUT_DIR/metrics/${batch_name}_summary.md"
    fi
}

# Monitor batch progress
monitor_progress() {
    log_info "Starting batch progress monitor..."
    
    # Monitor log files for progress
    while [ ! -f "$OUTPUT_DIR/.batch_complete" ]; do
        # Count completed series across all batches
        completed_series=0
        for log_file in "$OUTPUT_DIR/logs"/*.log; do
            if [ -f "$log_file" ]; then
                completed_series=$((completed_series + $(grep -c "Series completed" "$log_file" 2>/dev/null || echo 0)))
            fi
        done
        
        echo -ne "\rSeries completed: $completed_series"
        sleep 2
    done
    echo ""
}

# Validate generated data
validate_batch_data() {
    log_info "Validating generated batch data..."
    
    local validation_errors=0
    
    for data_dir in "$OUTPUT_DIR/data"/*; do
        if [ -d "$data_dir" ]; then
            local batch_name=$(basename "$data_dir")
            log_info "Validating batch: $batch_name"
            
            # Validate each file in the batch
            for data_file in "$data_dir"/*.{json,csv}; do
                if [ -f "$data_file" ]; then
                    $TSIOT_CLI validate \
                        --input "$data_file" \
                        --tests "basic,statistical" \
                        --output "$data_file.validation" \
                        --silent || {
                        log_warning "Validation failed for $(basename "$data_file")"
                        validation_errors=$((validation_errors + 1))
                    }
                fi
            done
        fi
    done
    
    if [ $validation_errors -eq 0 ]; then
        log_success "All batch data validated successfully"
    else
        log_warning "$validation_errors validation errors found"
    fi
}

# Generate aggregated analytics
generate_analytics() {
    log_info "Generating aggregated analytics..."
    
    # Create analytics report
    cat > "$OUTPUT_DIR/batch_analytics.html" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>TSIOT Batch Processing Analytics</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .batch { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .metrics { display: flex; gap: 20px; }
        .metric { background: #e8f4fd; padding: 10px; border-radius: 3px; text-align: center; }
        table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>TSIOT Batch Processing Analytics</h1>
        <p>Generated on: $(date)</p>
    </div>
EOF

    # Add batch summaries
    for summary_file in "$OUTPUT_DIR/metrics"/*_summary.md; do
        if [ -f "$summary_file" ]; then
            local batch_name=$(basename "$summary_file" _summary.md)
            echo "<div class=\"batch\">" >> "$OUTPUT_DIR/batch_analytics.html"
            echo "<h2>Batch: $batch_name</h2>" >> "$OUTPUT_DIR/batch_analytics.html"
            
            # Convert markdown to basic HTML
            sed 's/^# /\<h3\>/; s/^## /\<h4\>/; s/^### /\<h5\>/; s/^\*\* \(.*\)\*\*:/\<strong\>\1:\<\/strong\>/; s/^- /\<li\>/; /^$/d' "$summary_file" >> "$OUTPUT_DIR/batch_analytics.html"
            
            echo "</div>" >> "$OUTPUT_DIR/batch_analytics.html"
        fi
    done
    
    echo "</body></html>" >> "$OUTPUT_DIR/batch_analytics.html"
    
    log_success "Analytics report generated: $OUTPUT_DIR/batch_analytics.html"
}

# Cleanup old batch data
cleanup_old_data() {
    local days_to_keep=${1:-7}
    
    log_info "Cleaning up data older than $days_to_keep days..."
    
    # Find and remove old directories
    find "$OUTPUT_DIR" -type d -name "*_20*" -mtime +$days_to_keep -exec rm -rf {} + 2>/dev/null || true
    
    # Clean up old log files
    find "$OUTPUT_DIR/logs" -name "*.log" -mtime +$days_to_keep -delete 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Export to cloud storage (example with AWS S3)
export_to_cloud() {
    local bucket_name=$1
    
    if [ -z "$bucket_name" ]; then
        log_info "No cloud storage bucket specified, skipping export"
        return 0
    fi
    
    log_info "Exporting batch data to S3 bucket: $bucket_name"
    
    # Check if AWS CLI is available
    if ! command -v aws &> /dev/null; then
        log_warning "AWS CLI not found, skipping cloud export"
        return 1
    fi
    
    # Upload data with timestamp prefix
    local timestamp=$(date +%Y%m%d_%H%M%S)
    aws s3 sync "$OUTPUT_DIR" "s3://$bucket_name/tsiot-batch/$timestamp/" \
        --exclude "*.log" \
        --exclude "batch_analytics.html" || {
        log_error "Failed to export to S3"
        return 1
    }
    
    log_success "Data exported to s3://$bucket_name/tsiot-batch/$timestamp/"
}

# Main execution function
main() {
    echo "============================================================"
    echo "    TSIOT CLI - Batch Processing Script                   "
    echo "============================================================"
    echo ""
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --cli-path)
                TSIOT_CLI="$2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --workers)
                NUM_WORKERS="$2"
                shift 2
                ;;
            --batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --cleanup-days)
                CLEANUP_DAYS="$2"
                shift 2
                ;;
            --s3-bucket)
                S3_BUCKET="$2"
                shift 2
                ;;
            --skip-validation)
                SKIP_VALIDATION=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --cli-path PATH       Path to TSIOT CLI binary"
                echo "  --output-dir DIR      Output directory for batch data"
                echo "  --workers NUM         Number of parallel workers"
                echo "  --batch-size NUM      Batch size for processing"
                echo "  --cleanup-days NUM    Days to keep old data (default: 7)"
                echo "  --s3-bucket BUCKET    S3 bucket for cloud export"
                echo "  --skip-validation     Skip data validation step"
                echo "  --help                Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Check prerequisites
    if [ ! -f "$TSIOT_CLI" ]; then
        log_error "TSIOT CLI not found at $TSIOT_CLI"
        exit 1
    fi
    
    # Setup and execute
    setup_directories
    generate_batch_configs
    
    log_info "Starting batch processing with $NUM_WORKERS workers..."
    
    # Start progress monitor in background
    monitor_progress &
    MONITOR_PID=$!
    
    # Execute all batches in parallel
    batch_pids=()
    for config_file in "$CONFIG_DIR"/*.yaml; do
        if [ -f "$config_file" ]; then
            execute_batch "$config_file" &
            batch_pids+=($!)
        fi
    done
    
    # Wait for all batches to complete
    for pid in "${batch_pids[@]}"; do
        wait $pid || log_warning "Batch process $pid failed"
    done
    
    # Signal monitor to stop
    touch "$OUTPUT_DIR/.batch_complete"
    kill $MONITOR_PID 2>/dev/null || true
    
    log_success "All batches completed"
    
    # Post-processing
    if [ "$SKIP_VALIDATION" != "true" ]; then
        validate_batch_data
    fi
    
    generate_analytics
    
    # Cleanup if specified
    if [ -n "$CLEANUP_DAYS" ]; then
        cleanup_old_data "$CLEANUP_DAYS"
    fi
    
    # Export to cloud if specified
    if [ -n "$S3_BUCKET" ]; then
        export_to_cloud "$S3_BUCKET"
    fi
    
    # Final summary
    echo ""
    echo "============================================================"
    log_success "Batch processing completed successfully!"
    echo "============================================================"
    echo ""
    echo "📁 Output directory: $OUTPUT_DIR"
    echo "📊 Analytics report: $OUTPUT_DIR/batch_analytics.html"
    echo "📋 Summary files: $OUTPUT_DIR/metrics/*_summary.md"
    echo ""
    echo "Generated batches:"
    for config_file in "$CONFIG_DIR"/*.yaml; do
        if [ -f "$config_file" ]; then
            local batch_name=$(basename "$config_file" .yaml)
            echo "  - $batch_name"
        fi
    done
    echo ""
    
    # Cleanup temp files
    rm -f "$OUTPUT_DIR/.batch_complete"
}

# Cleanup on exit
cleanup() {
    kill $MONITOR_PID 2>/dev/null || true
    rm -f "$OUTPUT_DIR/.batch_complete"
}

trap cleanup EXIT

# Run main function
main "$@"