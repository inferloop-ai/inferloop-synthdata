#!/bin/bash

# TSIoT Advanced Workflows
# =========================
# This script demonstrates advanced data processing workflows using TSIoT,
# including multi-sensor generation, complex analysis pipelines, and 
# comprehensive validation workflows.

# Configuration
OUTPUT_DIR="advanced_workflows_output"
TEMP_DIR="$OUTPUT_DIR/temp"
ANALYSIS_DIR="$OUTPUT_DIR/analysis"
VALIDATION_DIR="$OUTPUT_DIR/validation"
COMPARISON_DIR="$OUTPUT_DIR/comparison"

# Create directory structure
mkdir -p "$OUTPUT_DIR" "$TEMP_DIR" "$ANALYSIS_DIR" "$VALIDATION_DIR" "$COMPARISON_DIR"

echo "=€ TSIoT Advanced Workflows Demonstration"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"

# Advanced Multi-Sensor Data Generation
echo ""
echo "<í 1. Multi-Sensor Industrial Facility Simulation"
echo "=================================================="

# Define sensor configurations for an industrial facility
declare -A SENSORS=(
    ["temperature"]="sensor-type temperature count 2880 frequency 30s"
    ["humidity"]="sensor-type humidity count 2880 frequency 30s"
    ["pressure"]="sensor-type pressure count 1440 frequency 1m"
    ["co2"]="sensor-type co2 count 720 frequency 2m"
    ["vibration"]="sensor-type vibration count 5760 frequency 15s"
)

declare -A GENERATORS=(
    ["statistical"]="Basic statistical model"
    ["timegan"]="Advanced temporal patterns"
    ["wgan-gp"]="High-fidelity generation"
)

echo "Generating multi-sensor facility data..."

# Generate data for each sensor type with different generators
for sensor in "${!SENSORS[@]}"; do
    echo ""
    echo "=Ê Generating $sensor sensor data..."
    
    # Split sensor configuration
    IFS=' ' read -ra PARAMS <<< "${SENSORS[$sensor]}"
    
    for generator in "${!GENERATORS[@]}"; do
        echo "  Using $generator generator..."
        
        output_file="$TEMP_DIR/${sensor}_${generator}.json"
        
        tsiot generate \
            --generator "$generator" \
            --${PARAMS[0]} "${PARAMS[1]}" \
            --${PARAMS[2]} "${PARAMS[3]}" \
            --${PARAMS[4]} "${PARAMS[5]}" \
            --start-time "2023-01-01T00:00:00Z" \
            --output "$output_file" \
            2>/dev/null
        
        if [ $? -eq 0 ]; then
            echo "     Generated with $generator"
        else
            echo "    L Failed with $generator"
        fi
    done
done

echo ""
echo "=È Generated files summary:"
ls -la "$TEMP_DIR"/*.json 2>/dev/null | wc -l | xargs echo "  Total files:"

# Advanced Analysis Pipeline
echo ""
echo "=, 2. Comprehensive Analysis Pipeline"
echo "====================================="

# Define analysis configurations
declare -A ANALYSIS_CONFIGS=(
    ["basic"]="basic_stats"
    ["temporal"]="basic_stats,trend,seasonality"
    ["comprehensive"]="basic_stats,trend,seasonality,anomalies"
    ["pattern"]="seasonality,anomalies,pattern_analysis"
)

echo "Running analysis pipeline on generated data..."

for sensor in "temperature" "humidity" "pressure"; do
    for generator in "statistical" "timegan"; do
        input_file="$TEMP_DIR/${sensor}_${generator}.json"
        
        if [ -f "$input_file" ]; then
            echo ""
            echo "= Analyzing $sensor data ($generator generator)..."
            
            for analysis_name in "${!ANALYSIS_CONFIGS[@]}"; do
                analysis_types="${ANALYSIS_CONFIGS[$analysis_name]}"
                output_file="$ANALYSIS_DIR/${sensor}_${generator}_${analysis_name}.json"
                
                echo "  Running $analysis_name analysis..."
                
                tsiot analyze \
                    --input "$input_file" \
                    --analysis "$analysis_types" \
                    --output "$output_file" \
                    2>/dev/null
                
                if [ $? -eq 0 ]; then
                    echo "     $analysis_name completed"
                else
                    echo "    L $analysis_name failed"
                fi
            done
        fi
    done
done

# Cross-Generator Comparison Workflow
echo ""
echo "– 3. Cross-Generator Comparison Workflow"
echo "=========================================="

echo "Comparing generator performance across sensor types..."

# Function to perform cross-generator validation
compare_generators() {
    local sensor=$1
    echo ""
    echo "=, Comparing generators for $sensor sensor..."
    
    # Use statistical as reference, compare others against it
    reference_file="$TEMP_DIR/${sensor}_statistical.json"
    
    if [ -f "$reference_file" ]; then
        for generator in "timegan" "wgan-gp"; do
            synthetic_file="$TEMP_DIR/${sensor}_${generator}.json"
            
            if [ -f "$synthetic_file" ]; then
                echo "  Validating $generator vs statistical..."
                
                output_file="$COMPARISON_DIR/${sensor}_${generator}_vs_statistical.json"
                
                tsiot validate \
                    --synthetic "$synthetic_file" \
                    --reference "$reference_file" \
                    --validators "statistical,distributional,temporal" \
                    --threshold 0.7 \
                    --output "$output_file" \
                    2>/dev/null
                
                if [ $? -eq 0 ] && command -v jq &> /dev/null; then
                    score=$(jq -r '.overall_quality_score // "N/A"' "$output_file" 2>/dev/null)
                    passed=$(jq -r '.passed // false' "$output_file" 2>/dev/null)
                    echo "    Quality Score: $score, Passed: $passed"
                elif [ $? -eq 0 ]; then
                    echo "     Validation completed (install jq for detailed results)"
                else
                    echo "    L Validation failed"
                fi
            fi
        done
    else
        echo "    Reference file not found: $reference_file"
    fi
}

# Compare generators for each sensor type
for sensor in "temperature" "humidity" "pressure"; do
    compare_generators "$sensor"
done

# Quality Assessment Workflow
echo ""
echo "=Ê 4. Comprehensive Quality Assessment"
echo "======================================"

echo "Performing quality assessment across all generated data..."

# Function to assess data quality
assess_quality() {
    local file=$1
    local name=$2
    
    echo "  =Ë Assessing $name..."
    
    # Self-validation (no reference data)
    output_file="$VALIDATION_DIR/${name}_quality.json"
    
    tsiot validate \
        --synthetic "$file" \
        --validators "statistical,distributional" \
        --threshold 0.8 \
        --output "$output_file" \
        2>/dev/null
    
    if [ $? -eq 0 ]; then
        if command -v jq &> /dev/null; then
            score=$(jq -r '.overall_quality_score // "N/A"' "$output_file" 2>/dev/null)
            validators=$(jq -r '.validators_run // "N/A"' "$output_file" 2>/dev/null)
            echo "    Quality Score: $score (Validators: $validators)"
        else
            echo "     Quality assessment completed"
        fi
    else
        echo "    L Quality assessment failed"
    fi
}

# Assess quality for all generated files
echo ""
echo "<¯ Individual Quality Scores:"
for file in "$TEMP_DIR"/*.json; do
    if [ -f "$file" ]; then
        filename=$(basename "$file" .json)
        assess_quality "$file" "$filename"
    fi
done

# Data Migration and Format Conversion Workflow
echo ""
echo "=æ 5. Data Migration and Format Workflows"
echo "=========================================="

echo "Demonstrating data migration capabilities..."

# Create migration scenarios
echo ""
echo "= Migration Scenarios:"

# Scenario 1: Batch migration with size optimization
echo "  Scenario 1: Optimized batch migration..."
if [ -f "$TEMP_DIR/temperature_statistical.json" ]; then
    tsiot migrate \
        --source "$TEMP_DIR/temperature_statistical.json" \
        --destination "$OUTPUT_DIR/temperature_optimized.json" \
        --batch-size 1000 \
        2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "     Optimized migration completed"
        
        # Compare file sizes
        original_size=$(stat -f%z "$TEMP_DIR/temperature_statistical.json" 2>/dev/null || stat -c%s "$TEMP_DIR/temperature_statistical.json" 2>/dev/null || echo "0")
        migrated_size=$(stat -f%z "$OUTPUT_DIR/temperature_optimized.json" 2>/dev/null || stat -c%s "$OUTPUT_DIR/temperature_optimized.json" 2>/dev/null || echo "0")
        
        echo "    Original: $original_size bytes, Migrated: $migrated_size bytes"
    else
        echo "    L Optimized migration failed"
    fi
fi

# Scenario 2: Dry run for large datasets
echo "  Scenario 2: Dry run validation..."
if [ -f "$TEMP_DIR/humidity_timegan.json" ]; then
    tsiot migrate \
        --source "$TEMP_DIR/humidity_timegan.json" \
        --destination "$OUTPUT_DIR/humidity_dryrun.json" \
        --batch-size 500 \
        --dry-run \
        2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "     Dry run completed successfully"
    else
        echo "    L Dry run failed"
    fi
fi

# Automated Workflow Pipeline
echo ""
echo "> 6. Automated Workflow Pipeline"
echo "=================================="

echo "Running end-to-end automated pipeline..."

# Function for complete workflow
run_complete_workflow() {
    local sensor_type=$1
    local generator_type=$2
    local workflow_id="${sensor_type}_${generator_type}_$(date +%s)"
    
    echo ""
    echo "= Workflow: $workflow_id"
    echo "  Sensor: $sensor_type, Generator: $generator_type"
    
    # Step 1: Generate data
    echo "  Step 1: Data Generation..."
    gen_file="$TEMP_DIR/workflow_${workflow_id}.json"
    
    tsiot generate \
        --generator "$generator_type" \
        --sensor-type "$sensor_type" \
        --count 1440 \
        --frequency 1m \
        --start-time "2023-01-01T00:00:00Z" \
        --output "$gen_file" \
        2>/dev/null
    
    if [ $? -ne 0 ]; then
        echo "    L Generation failed"
        return 1
    fi
    echo "     Generated"
    
    # Step 2: Analyze data
    echo "  Step 2: Data Analysis..."
    analysis_file="$ANALYSIS_DIR/workflow_${workflow_id}_analysis.json"
    
    tsiot analyze \
        --input "$gen_file" \
        --analysis "basic_stats,trend,seasonality" \
        --output "$analysis_file" \
        2>/dev/null
    
    if [ $? -ne 0 ]; then
        echo "    L Analysis failed"
        return 1
    fi
    echo "     Analyzed"
    
    # Step 3: Validate quality
    echo "  Step 3: Quality Validation..."
    validation_file="$VALIDATION_DIR/workflow_${workflow_id}_validation.json"
    
    tsiot validate \
        --synthetic "$gen_file" \
        --validators "statistical,distributional" \
        --threshold 0.75 \
        --output "$validation_file" \
        2>/dev/null
    
    if [ $? -ne 0 ]; then
        echo "    L Validation failed"
        return 1
    fi
    echo "     Validated"
    
    # Step 4: Migrate if quality is good
    if command -v jq &> /dev/null; then
        quality_score=$(jq -r '.overall_quality_score // 0' "$validation_file" 2>/dev/null)
        passed=$(jq -r '.passed // false' "$validation_file" 2>/dev/null)
        
        if [ "$passed" = "true" ]; then
            echo "  Step 4: Data Migration (Quality Score: $quality_score)..."
            migrate_file="$OUTPUT_DIR/workflow_${workflow_id}_final.json"
            
            tsiot migrate \
                --source "$gen_file" \
                --destination "$migrate_file" \
                --batch-size 500 \
                2>/dev/null
            
            if [ $? -eq 0 ]; then
                echo "     Migrated (Workflow Complete)"
                echo "  <‰ Workflow $workflow_id completed successfully!"
                return 0
            else
                echo "    L Migration failed"
                return 1
            fi
        else
            echo "    Quality threshold not met (Score: $quality_score)"
            echo "  = Workflow $workflow_id completed with quality issues"
            return 1
        fi
    else
        echo "  Step 4: Migration (Quality check skipped - jq not available)..."
        migrate_file="$OUTPUT_DIR/workflow_${workflow_id}_final.json"
        
        tsiot migrate \
            --source "$gen_file" \
            --destination "$migrate_file" \
            --batch-size 500 \
            2>/dev/null
        
        if [ $? -eq 0 ]; then
            echo "     Migrated"
            echo "  <‰ Workflow $workflow_id completed!"
            return 0
        else
            echo "    L Migration failed"
            return 1
        fi
    fi
}

# Run automated workflows for different configurations
echo "Running automated workflows:"

WORKFLOW_CONFIGS=(
    "temperature statistical"
    "humidity timegan"
    "pressure statistical"
)

successful_workflows=0
total_workflows=${#WORKFLOW_CONFIGS[@]}

for config in "${WORKFLOW_CONFIGS[@]}"; do
    read -r sensor generator <<< "$config"
    if run_complete_workflow "$sensor" "$generator"; then
        ((successful_workflows++))
    fi
done

echo ""
echo "=Ê Workflow Summary: $successful_workflows/$total_workflows workflows completed successfully"

# Performance Analysis
echo ""
echo "¡ 7. Performance Analysis"
echo "========================="

echo "Analyzing generation and processing performance..."

# Function to time operations
time_operation() {
    local operation=$1
    local description=$2
    shift 2
    
    echo "  Timing: $description..."
    start_time=$(date +%s.%N)
    
    "$@" 2>/dev/null
    local exit_code=$?
    
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "N/A")
    
    if [ $exit_code -eq 0 ]; then
        echo "     Completed in ${duration}s"
    else
        echo "    L Failed after ${duration}s"
    fi
    
    return $exit_code
}

# Time different operations
echo ""
echo "=P Operation Timing Tests:"

# Test generation performance
time_operation "generation" "Small dataset generation (100 points)" \
    tsiot generate --generator statistical --sensor-type temperature --count 100 --output "$TEMP_DIR/perf_small.json"

time_operation "generation" "Medium dataset generation (1000 points)" \
    tsiot generate --generator statistical --sensor-type temperature --count 1000 --output "$TEMP_DIR/perf_medium.json"

# Test analysis performance
if [ -f "$TEMP_DIR/perf_medium.json" ]; then
    time_operation "analysis" "Comprehensive analysis" \
        tsiot analyze --input "$TEMP_DIR/perf_medium.json" --analysis "basic_stats,trend,seasonality,anomalies" --output "$TEMP_DIR/perf_analysis.json"
fi

# Test validation performance
if [ -f "$TEMP_DIR/perf_medium.json" ]; then
    time_operation "validation" "Multi-validator validation" \
        tsiot validate --synthetic "$TEMP_DIR/perf_medium.json" --validators "statistical,distributional,temporal" --threshold 0.8 --output "$TEMP_DIR/perf_validation.json"
fi

# Final Summary and Cleanup Options
echo ""
echo "=Ë 8. Workflow Summary and Cleanup"
echo "==================================="

echo "Advanced workflows completed!"
echo ""
echo "=Á Generated Artifacts:"

# Count files in each directory
temp_files=$(ls "$TEMP_DIR"/*.json 2>/dev/null | wc -l)
analysis_files=$(ls "$ANALYSIS_DIR"/*.json 2>/dev/null | wc -l)
validation_files=$(ls "$VALIDATION_DIR"/*.json 2>/dev/null | wc -l)
comparison_files=$(ls "$COMPARISON_DIR"/*.json 2>/dev/null | wc -l)
output_files=$(ls "$OUTPUT_DIR"/*.json 2>/dev/null | wc -l)

echo "  =Ê Temporary files: $temp_files"
echo "  =È Analysis results: $analysis_files"
echo "   Validation reports: $validation_files"
echo "  – Comparison results: $comparison_files"
echo "  =æ Final outputs: $output_files"

# Calculate total disk usage
if command -v du &> /dev/null; then
    total_size=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1 || echo "unknown")
    echo "  =¾ Total disk usage: $total_size"
fi

echo ""
echo ">ù Cleanup Options:"
echo "  =Ñ Clean temporary files: rm -rf $TEMP_DIR"
echo "  =Ñ Clean all generated files: rm -rf $OUTPUT_DIR"
echo "  =æ Archive results: tar -czf advanced_workflows_$(date +%Y%m%d_%H%M%S).tar.gz $OUTPUT_DIR"

echo ""
echo "<¯ Key Workflow Insights:"
echo "   Multi-sensor data generation across different generators"
echo "   Comprehensive analysis pipeline with multiple analysis types"
echo "   Cross-generator performance comparison and validation"
echo "   Automated quality assessment workflow"
echo "   Advanced data migration and format conversion"
echo "   End-to-end pipeline automation with quality gates"
echo "   Performance timing and optimization analysis"

echo ""
echo "=€ Next Steps:"
echo "  1. Analyze the generated quality reports for insights"
echo "  2. Customize workflows for your specific use cases"
echo "  3. Integrate workflows into CI/CD pipelines"
echo "  4. Scale up data generation for production workloads"
echo "  5. Implement real-time monitoring of data quality"

echo ""
echo "<‰ Advanced workflows demonstration complete!"
echo "============================================="