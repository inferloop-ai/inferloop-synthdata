#!/bin/bash

# TSIoT Basic CLI Commands
# ========================
# This script demonstrates the fundamental CLI operations for TSIoT
# synthetic IoT time series data generation and analysis.

# Set up output directories
OUTPUT_DIR="basic_cli_output"
mkdir -p "$OUTPUT_DIR"

echo "=€ TSIoT Basic CLI Commands Demonstration"
echo "=========================================="

# Basic Data Generation
echo ""
echo "=Ê 1. Basic Data Generation"
echo "----------------------------"

# Generate simple temperature data
echo "Generating temperature sensor data..."
tsiot generate \
    --generator statistical \
    --sensor-type temperature \
    --count 1440 \
    --frequency 1m \
    --start-time "2023-01-01T00:00:00Z" \
    --end-time "2023-01-01T23:59:00Z" \
    --output "$OUTPUT_DIR/temperature_basic.json"

if [ $? -eq 0 ]; then
    echo " Successfully generated temperature data"
else
    echo "L Failed to generate temperature data"
fi

# Generate humidity data
echo "Generating humidity sensor data..."
tsiot generate \
    --generator statistical \
    --sensor-type humidity \
    --count 720 \
    --frequency 2m \
    --output "$OUTPUT_DIR/humidity_basic.json"

if [ $? -eq 0 ]; then
    echo " Successfully generated humidity data"
else
    echo "L Failed to generate humidity data"
fi

# Generate pressure data with custom parameters
echo "Generating pressure sensor data with custom parameters..."
tsiot generate \
    --generator statistical \
    --sensor-type pressure \
    --count 288 \
    --frequency 5m \
    --start-time "2023-01-01T00:00:00Z" \
    --output "$OUTPUT_DIR/pressure_basic.json"

if [ $? -eq 0 ]; then
    echo " Successfully generated pressure data"
else
    echo "L Failed to generate pressure data"
fi

# Basic Data Analysis
echo ""
echo "= 2. Basic Data Analysis"
echo "-------------------------"

# Analyze temperature data
echo "Analyzing temperature data..."
tsiot analyze \
    --input "$OUTPUT_DIR/temperature_basic.json" \
    --analysis basic_stats \
    --output "$OUTPUT_DIR/temperature_analysis.json"

if [ $? -eq 0 ]; then
    echo " Successfully analyzed temperature data"
    
    # Display basic statistics if jq is available
    if command -v jq &> /dev/null; then
        echo "=È Basic Statistics:"
        jq '.basic_statistics' "$OUTPUT_DIR/temperature_analysis.json" 2>/dev/null || echo "   (Analysis results saved to file)"
    else
        echo "   Analysis results saved to $OUTPUT_DIR/temperature_analysis.json"
    fi
else
    echo "L Failed to analyze temperature data"
fi

# Multi-type analysis
echo "Performing comprehensive analysis..."
tsiot analyze \
    --input "$OUTPUT_DIR/temperature_basic.json" \
    --analysis basic_stats,trend,seasonality \
    --output "$OUTPUT_DIR/temperature_comprehensive.json"

if [ $? -eq 0 ]; then
    echo " Successfully performed comprehensive analysis"
else
    echo "L Failed to perform comprehensive analysis"
fi

# Basic Data Validation
echo ""
echo " 3. Basic Data Validation"
echo "----------------------------"

# Validate synthetic data quality
echo "Validating temperature data quality..."
tsiot validate \
    --synthetic "$OUTPUT_DIR/temperature_basic.json" \
    --validators statistical,distributional \
    --threshold 0.8 \
    --output "$OUTPUT_DIR/temperature_validation.json"

if [ $? -eq 0 ]; then
    echo " Successfully validated temperature data"
    
    # Display validation results if jq is available
    if command -v jq &> /dev/null; then
        echo "<¯ Validation Results:"
        jq '{overall_quality_score, passed, validators_run}' "$OUTPUT_DIR/temperature_validation.json" 2>/dev/null || echo "   (Validation results saved to file)"
    else
        echo "   Validation results saved to $OUTPUT_DIR/temperature_validation.json"
    fi
else
    echo "L Failed to validate temperature data"
fi

# Cross-validation with reference data
echo "Cross-validating humidity vs temperature data..."
tsiot validate \
    --synthetic "$OUTPUT_DIR/humidity_basic.json" \
    --reference "$OUTPUT_DIR/temperature_basic.json" \
    --validators statistical \
    --threshold 0.7 \
    --output "$OUTPUT_DIR/cross_validation.json"

if [ $? -eq 0 ]; then
    echo " Successfully performed cross-validation"
else
    echo "L Failed to perform cross-validation"
fi

# Basic Data Migration
echo ""
echo "=æ 4. Basic Data Migration"
echo "--------------------------"

# Migrate temperature data (copy with potential format conversion)
echo "Migrating temperature data..."
tsiot migrate \
    --source "$OUTPUT_DIR/temperature_basic.json" \
    --destination "$OUTPUT_DIR/temperature_migrated.json" \
    --batch-size 500

if [ $? -eq 0 ]; then
    echo " Successfully migrated temperature data"
else
    echo "L Failed to migrate temperature data"
fi

# Dry run migration
echo "Performing dry run migration..."
tsiot migrate \
    --source "$OUTPUT_DIR/humidity_basic.json" \
    --destination "$OUTPUT_DIR/humidity_dryrun.json" \
    --batch-size 100 \
    --dry-run

if [ $? -eq 0 ]; then
    echo " Successfully completed dry run migration"
else
    echo "L Failed to complete dry run migration"
fi

# File Operations and Utilities
echo ""
echo "=Á 5. File Operations and Utilities"
echo "------------------------------------"

# Check generated files
echo "Generated files:"
for file in "$OUTPUT_DIR"/*.json; do
    if [ -f "$file" ]; then
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "unknown")
        echo "  =Ä $(basename "$file") (${size} bytes)"
    fi
done

# Basic file inspection using standard tools
echo ""
echo "= Quick Data Inspection:"
echo "--------------------------"

if [ -f "$OUTPUT_DIR/temperature_basic.json" ]; then
    echo "Temperature data sample:"
    if command -v jq &> /dev/null; then
        echo "  Total points: $(jq '.points | length' "$OUTPUT_DIR/temperature_basic.json" 2>/dev/null || echo "unknown")"
        echo "  Sensor type: $(jq -r '.sensor_type' "$OUTPUT_DIR/temperature_basic.json" 2>/dev/null || echo "unknown")"
        echo "  First few points:"
        jq '.points[0:3]' "$OUTPUT_DIR/temperature_basic.json" 2>/dev/null || echo "    (jq not available for detailed inspection)"
    else
        echo "  (Install jq for detailed JSON inspection)"
        head -n 10 "$OUTPUT_DIR/temperature_basic.json"
    fi
fi

# Clean up demonstration
echo ""
echo ">ù 6. Clean Up Operations"
echo "-------------------------"

# List all generated files before cleanup
echo "Files that would be cleaned up:"
ls -la "$OUTPUT_DIR"/ 2>/dev/null || echo "No files found"

# Optional: Uncomment the next line to actually clean up generated files
# rm -rf "$OUTPUT_DIR"
echo "=¡ To clean up generated files, run: rm -rf $OUTPUT_DIR"

# Help and Information
echo ""
echo "S 7. Getting Help"
echo "------------------"

echo "For command help, use:"
echo "  tsiot --help                    # General help"
echo "  tsiot generate --help           # Generation options"
echo "  tsiot analyze --help            # Analysis options"
echo "  tsiot validate --help           # Validation options"
echo "  tsiot migrate --help            # Migration options"

echo ""
echo "=Ú Additional Examples:"
echo "  # Generate with specific time range"
echo "  tsiot generate --start-time '2023-06-01T00:00:00Z' --end-time '2023-06-01T12:00:00Z'"
echo ""
echo "  # Analyze with custom analysis types"
echo "  tsiot analyze --analysis 'basic_stats,trend,seasonality,anomalies'"
echo ""
echo "  # Validate with multiple validators"
echo "  tsiot validate --validators 'statistical,distributional,temporal' --threshold 0.9"
echo ""
echo "  # Migrate with larger batches"
echo "  tsiot migrate --batch-size 1000 --source data.json --destination migrated.json"

echo ""
echo "<‰ Basic CLI commands demonstration complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Try the advanced-workflows.sh script for complex operations"
echo "  2. Check the automation-scripts.sh for batch processing examples"
echo "  3. Explore the Jupyter notebooks for interactive analysis"
echo "  4. Read the documentation for detailed parameter explanations"
echo ""
echo "Output files are in: $OUTPUT_DIR/"
echo "Remember to clean up when done: rm -rf $OUTPUT_DIR"