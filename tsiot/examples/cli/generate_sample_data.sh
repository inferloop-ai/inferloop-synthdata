#!/bin/bash

# TSIOT CLI - Generate Sample Data Script
# This script demonstrates how to use the TSIOT CLI to generate various types of synthetic time series data

set -e

# Configuration
TSIOT_CLI=${TSIOT_CLI:-"./bin/tsiot"}
OUTPUT_DIR=${OUTPUT_DIR:-"./sample_data"}
API_URL=${API_URL:-"http://localhost:8080"}
API_KEY=${API_KEY:-""}

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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if TSIOT CLI exists
    if [ ! -f "$TSIOT_CLI" ]; then
        log_error "TSIOT CLI not found at $TSIOT_CLI"
        log_info "Please build the CLI first: go build -o bin/tsiot ./cmd/cli"
        exit 1
    fi
    
    # Check if jq is available (for JSON processing)
    if ! command -v jq &> /dev/null; then
        log_warning "jq not found. JSON output will not be formatted."
        JQ_AVAILABLE=false
    else
        JQ_AVAILABLE=true
    fi
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    log_success "Prerequisites check completed"
}

# Test API connectivity
test_api_connectivity() {
    log_info "Testing API connectivity..."
    
    if [ -n "$API_KEY" ]; then
        AUTH_HEADER="Authorization: Bearer $API_KEY"
    else
        AUTH_HEADER=""
    fi
    
    if curl -s -f -H "$AUTH_HEADER" "$API_URL/health" > /dev/null; then
        log_success "API is accessible at $API_URL"
    else
        log_error "Cannot connect to API at $API_URL"
        log_info "Please ensure TSIOT server is running"
        exit 1
    fi
}

# Generate ARIMA time series
generate_arima_series() {
    log_info "Generating ARIMA time series..."
    
    $TSIOT_CLI generate \
        --type arima \
        --length 1000 \
        --output "$OUTPUT_DIR/arima_series.json" \
        --format json \
        --parameters '{
            "ar_params": [0.5, -0.3, 0.1],
            "ma_params": [0.2, -0.1],
            "mean": 100,
            "variance": 25,
            "seasonal": true,
            "seasonal_period": 24
        }' || {
        log_error "Failed to generate ARIMA series"
        return 1
    }
    
    log_success "ARIMA series saved to $OUTPUT_DIR/arima_series.json"
}

# Generate LSTM time series
generate_lstm_series() {
    log_info "Generating LSTM time series..."
    
    $TSIOT_CLI generate \
        --type lstm \
        --length 500 \
        --output "$OUTPUT_DIR/lstm_series.csv" \
        --format csv \
        --parameters '{
            "trend": 0.1,
            "seasonality": 168,
            "noise": 0.05,
            "complexity": "medium"
        }' || {
        log_error "Failed to generate LSTM series"
        return 1
    }
    
    log_success "LSTM series saved to $OUTPUT_DIR/lstm_series.csv"
}

# Generate Random Walk series
generate_random_walk() {
    log_info "Generating Random Walk series..."
    
    $TSIOT_CLI generate \
        --type random_walk \
        --length 2000 \
        --output "$OUTPUT_DIR/random_walk.json" \
        --format json \
        --parameters '{
            "drift": 0.02,
            "volatility": 1.5,
            "initial_value": 100
        }' || {
        log_error "Failed to generate Random Walk series"
        return 1
    }
    
    log_success "Random Walk series saved to $OUTPUT_DIR/random_walk.json"
}

# Generate multiple series with different parameters
generate_batch_series() {
    log_info "Generating batch of time series..."
    
    # Create batch configuration
    cat > "$OUTPUT_DIR/batch_config.yaml" << EOF
batch:
  output_dir: "$OUTPUT_DIR/batch"
  series:
    - name: "financial_stock"
      type: "lstm"
      length: 252
      parameters:
        trend: 0.08
        volatility: 0.2
        seasonality: 0
    - name: "iot_sensor"
      type: "arima"
      length: 1440
      parameters:
        ar_params: [0.7, -0.2]
        ma_params: [0.3]
        seasonal: true
        seasonal_period: 60
    - name: "web_traffic"
      type: "lstm"
      length: 168
      parameters:
        trend: 0.05
        seasonality: 24
        noise: 0.1
EOF
    
    $TSIOT_CLI generate batch \
        --config "$OUTPUT_DIR/batch_config.yaml" \
        --concurrent 3 || {
        log_error "Failed to generate batch series"
        return 1
    }
    
    log_success "Batch series generated in $OUTPUT_DIR/batch/"
}

# Validate generated data
validate_generated_data() {
    log_info "Validating generated data..."
    
    for file in "$OUTPUT_DIR"/*.json "$OUTPUT_DIR"/*.csv; do
        if [ -f "$file" ]; then
            $TSIOT_CLI validate \
                --input "$file" \
                --tests "basic,statistical,quality" \
                --output "$file.validation" || {
                log_warning "Validation failed for $file"
                continue
            }
            log_success "Validated $(basename "$file")"
        fi
    done
}

# Analyze generated data
analyze_data() {
    log_info "Analyzing generated data..."
    
    # Generate analysis report
    $TSIOT_CLI analyze \
        --input "$OUTPUT_DIR" \
        --output "$OUTPUT_DIR/analysis_report.html" \
        --format html \
        --include-plots || {
        log_warning "Analysis failed"
        return 1
    }
    
    log_success "Analysis report saved to $OUTPUT_DIR/analysis_report.html"
}

# Export to different formats
export_data() {
    log_info "Exporting data to different formats..."
    
    # Convert JSON to CSV
    if [ -f "$OUTPUT_DIR/arima_series.json" ]; then
        $TSIOT_CLI convert \
            --input "$OUTPUT_DIR/arima_series.json" \
            --output "$OUTPUT_DIR/arima_series.csv" \
            --format csv || log_warning "Failed to convert ARIMA series to CSV"
    fi
    
    # Convert to Parquet
    if [ -f "$OUTPUT_DIR/lstm_series.csv" ]; then
        $TSIOT_CLI convert \
            --input "$OUTPUT_DIR/lstm_series.csv" \
            --output "$OUTPUT_DIR/lstm_series.parquet" \
            --format parquet || log_warning "Failed to convert to Parquet"
    fi
    
    log_success "Data exported to multiple formats"
}

# Stream data to Kafka (if available)
stream_to_kafka() {
    log_info "Streaming data to Kafka (if configured)..."
    
    if [ -n "$KAFKA_BROKERS" ]; then
        $TSIOT_CLI stream \
            --input "$OUTPUT_DIR/lstm_series.csv" \
            --kafka-brokers "$KAFKA_BROKERS" \
            --topic "tsiot-timeseries" \
            --rate 10 \
            --duration 60s || {
            log_warning "Failed to stream to Kafka"
            return 1
        }
        log_success "Data streamed to Kafka topic 'tsiot-timeseries'"
    else
        log_info "Kafka not configured, skipping streaming"
    fi
}

# Generate summary report
generate_summary() {
    log_info "Generating summary report..."
    
    cat > "$OUTPUT_DIR/generation_summary.md" << EOF
# TSIOT Data Generation Summary

Generated on: $(date)

## Files Generated

EOF

    # List all generated files
    for file in "$OUTPUT_DIR"/*; do
        if [ -f "$file" ] && [ "$(basename "$file")" != "generation_summary.md" ]; then
            size=$(du -h "$file" | cut -f1)
            echo "- $(basename "$file") ($size)" >> "$OUTPUT_DIR/generation_summary.md"
        fi
    done
    
    cat >> "$OUTPUT_DIR/generation_summary.md" << EOF

## Commands Used

\`\`\`bash
# ARIMA Series
$TSIOT_CLI generate --type arima --length 1000 --output arima_series.json

# LSTM Series  
$TSIOT_CLI generate --type lstm --length 500 --output lstm_series.csv

# Random Walk
$TSIOT_CLI generate --type random_walk --length 2000 --output random_walk.json

# Batch Generation
$TSIOT_CLI generate batch --config batch_config.yaml

# Validation
$TSIOT_CLI validate --input data.json --tests basic,statistical,quality

# Analysis
$TSIOT_CLI analyze --input . --output analysis_report.html
\`\`\`

## Next Steps

1. Examine the generated data files
2. Review validation results
3. Open analysis_report.html in a browser
4. Use the data for your machine learning experiments

For more information, visit: https://docs.tsiot.io
EOF
    
    log_success "Summary report saved to $OUTPUT_DIR/generation_summary.md"
}

# Main execution
main() {
    echo "=================================================="
    echo "    TSIOT CLI - Sample Data Generation Script    "
    echo "=================================================="
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
            --api-url)
                API_URL="$2"
                shift 2
                ;;
            --api-key)
                API_KEY="$2"
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
                echo "  --cli-path PATH      Path to TSIOT CLI binary (default: ./bin/tsiot)"
                echo "  --output-dir DIR     Output directory (default: ./sample_data)"
                echo "  --api-url URL        API URL (default: http://localhost:8080)"
                echo "  --api-key KEY        API key for authentication"
                echo "  --skip-validation    Skip data validation step"
                echo "  --help               Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Execute steps
    check_prerequisites
    test_api_connectivity
    
    log_info "Starting data generation process..."
    
    # Generate different types of series
    generate_arima_series
    generate_lstm_series
    generate_random_walk
    generate_batch_series
    
    # Validate and analyze
    if [ "$SKIP_VALIDATION" != "true" ]; then
        validate_generated_data
    fi
    
    analyze_data
    export_data
    stream_to_kafka
    generate_summary
    
    echo ""
    echo "=================================================="
    log_success "Data generation completed successfully!"
    echo "=================================================="
    echo ""
    echo "📁 Output directory: $OUTPUT_DIR"
    echo "📋 Summary report: $OUTPUT_DIR/generation_summary.md"
    echo "📊 Analysis report: $OUTPUT_DIR/analysis_report.html"
    echo ""
    echo "Next steps:"
    echo "1. cd $OUTPUT_DIR"
    echo "2. ls -la  # View generated files"
    echo "3. open analysis_report.html  # View analysis"
    echo ""
}

# Run main function
main "$@"