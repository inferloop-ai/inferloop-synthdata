#!/bin/bash

# TSIoT Automation Scripts
# ========================
# This script provides automated batch processing, scheduled workflows,
# and comprehensive automation capabilities for TSIoT synthetic data operations.

# Configuration
AUTOMATION_DIR="automation_output"
SCHEDULE_DIR="$AUTOMATION_DIR/scheduled"
BATCH_DIR="$AUTOMATION_DIR/batch"
LOGS_DIR="$AUTOMATION_DIR/logs"
CONFIG_DIR="$AUTOMATION_DIR/config"
ARCHIVE_DIR="$AUTOMATION_DIR/archive"

# Create directory structure
mkdir -p "$AUTOMATION_DIR" "$SCHEDULE_DIR" "$BATCH_DIR" "$LOGS_DIR" "$CONFIG_DIR" "$ARCHIVE_DIR"

# Logging functions
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1" | tee -a "$LOGS_DIR/automation.log"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$LOGS_DIR/automation.log" >&2
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $1" | tee -a "$LOGS_DIR/automation.log"
}

echo "= TSIoT Automation Scripts"
echo "==========================="
log_info "Starting TSIoT automation demonstration"

# 1. Batch Data Generation Automation
echo ""
echo "=Ê 1. Batch Data Generation Automation"
echo "======================================="

# Function for batch generation
batch_generate() {
    local config_name=$1
    local sensor_types=("${@:2}")
    
    log_info "Starting batch generation: $config_name"
    
    local batch_dir="$BATCH_DIR/${config_name}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$batch_dir"
    
    local successful_generations=0
    local total_generations=${#sensor_types[@]}
    
    # Generate data for each sensor type
    for sensor_type in "${sensor_types[@]}"; do
        log_info "Generating $sensor_type data..."
        
        local output_file="$batch_dir/${sensor_type}_batch.json"
        
        # Generate with error handling and retries
        local max_retries=3
        local retry_count=0
        local generation_success=false
        
        while [ $retry_count -lt $max_retries ] && [ "$generation_success" = false ]; do
            if tsiot generate \
                --generator statistical \
                --sensor-type "$sensor_type" \
                --count 1440 \
                --frequency 1m \
                --start-time "2023-01-01T00:00:00Z" \
                --output "$output_file" \
                2>>"$LOGS_DIR/generation_errors.log"; then
                
                log_success "Generated $sensor_type data (attempt $((retry_count + 1)))"
                ((successful_generations++))
                generation_success=true
            else
                ((retry_count++))
                log_error "Failed to generate $sensor_type data (attempt $retry_count)"
                if [ $retry_count -lt $max_retries ]; then
                    log_info "Retrying in 2 seconds..."
                    sleep 2
                fi
            fi
        done
        
        if [ "$generation_success" = false ]; then
            log_error "Failed to generate $sensor_type data after $max_retries attempts"
        fi
    done
    
    log_info "Batch generation complete: $successful_generations/$total_generations successful"
    echo "$batch_dir"
}

# Define batch configurations
declare -A BATCH_CONFIGS=(
    ["industrial_sensors"]="temperature humidity pressure co2 vibration"
    ["environmental_sensors"]="temperature humidity light_intensity uv_index air_quality"
    ["smart_home_sensors"]="temperature humidity motion door_sensor energy_consumption"
)

echo "Running batch generation for different sensor configurations..."

# Execute batch generations
for config_name in "${!BATCH_CONFIGS[@]}"; do
    echo ""
    echo "<í Processing $config_name configuration..."
    
    # Convert space-separated string to array
    IFS=' ' read -ra sensors <<< "${BATCH_CONFIGS[$config_name]}"
    
    batch_output_dir=$(batch_generate "$config_name" "${sensors[@]}")
    
    # Generate summary report
    summary_file="$batch_output_dir/batch_summary.json"
    
    {
        echo "{"
        echo "  \"batch_name\": \"$config_name\","
        echo "  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
        echo "  \"sensors\": ["
        
        first=true
        for sensor in "${sensors[@]}"; do
            if [ "$first" = false ]; then echo ","; fi
            first=false
            
            sensor_file="$batch_output_dir/${sensor}_batch.json"
            if [ -f "$sensor_file" ]; then
                file_size=$(stat -f%z "$sensor_file" 2>/dev/null || stat -c%s "$sensor_file" 2>/dev/null || echo "0")
                echo -n "    {\"sensor\": \"$sensor\", \"status\": \"success\", \"file_size\": $file_size}"
            else
                echo -n "    {\"sensor\": \"$sensor\", \"status\": \"failed\", \"file_size\": 0}"
            fi
        done
        
        echo ""
        echo "  ]"
        echo "}"
    } > "$summary_file"
    
    log_success "Batch summary saved: $summary_file"
done

# 2. Scheduled Workflow Automation
echo ""
echo "ð 2. Scheduled Workflow Automation"
echo "==================================="

# Function to simulate scheduled workflow
simulate_scheduled_workflow() {
    local workflow_name=$1
    local schedule_interval=$2
    local max_iterations=$3
    
    log_info "Starting scheduled workflow: $workflow_name (every $schedule_interval, max $max_iterations iterations)"
    
    local workflow_dir="$SCHEDULE_DIR/${workflow_name}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$workflow_dir"
    
    for ((i=1; i<=max_iterations; i++)); do
        log_info "Scheduled workflow iteration $i/$max_iterations"
        
        local iteration_dir="$workflow_dir/iteration_$i"
        mkdir -p "$iteration_dir"
        
        # Generate data
        local data_file="$iteration_dir/sensor_data.json"
        if tsiot generate \
            --generator statistical \
            --sensor-type temperature \
            --count 144 \
            --frequency 10m \
            --start-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            --output "$data_file" \
            2>>"$LOGS_DIR/scheduled_errors.log"; then
            
            log_success "Generated data for iteration $i"
            
            # Analyze data
            local analysis_file="$iteration_dir/analysis.json"
            if tsiot analyze \
                --input "$data_file" \
                --analysis "basic_stats,trend" \
                --output "$analysis_file" \
                2>>"$LOGS_DIR/scheduled_errors.log"; then
                
                log_success "Analyzed data for iteration $i"
                
                # Validate data
                local validation_file="$iteration_dir/validation.json"
                if tsiot validate \
                    --synthetic "$data_file" \
                    --validators "statistical" \
                    --threshold 0.8 \
                    --output "$validation_file" \
                    2>>"$LOGS_DIR/scheduled_errors.log"; then
                    
                    log_success "Validated data for iteration $i"
                else
                    log_error "Failed to validate data for iteration $i"
                fi
            else
                log_error "Failed to analyze data for iteration $i"
            fi
        else
            log_error "Failed to generate data for iteration $i"
        fi
        
        # Simulate schedule interval (reduced for demo)
        if [ $i -lt $max_iterations ]; then
            log_info "Waiting for next scheduled run..."
            sleep 2  # In real usage, this would be much longer
        fi
    done
    
    log_success "Scheduled workflow completed: $workflow_name"
}

# Define scheduled workflows
declare -A SCHEDULED_WORKFLOWS=(
    ["hourly_monitoring"]="1h 3"
    ["daily_analysis"]="1d 2"
    ["weekly_report"]="1w 1"
)

echo "Simulating scheduled workflows..."

for workflow_name in "${!SCHEDULED_WORKFLOWS[@]}"; do
    echo ""
    echo "=Å Running $workflow_name workflow..."
    
    IFS=' ' read -ra schedule_config <<< "${SCHEDULED_WORKFLOWS[$workflow_name]}"
    interval="${schedule_config[0]}"
    iterations="${schedule_config[1]}"
    
    simulate_scheduled_workflow "$workflow_name" "$interval" "$iterations"
done

# 3. Pipeline Automation with Dependencies
echo ""
echo "= 3. Pipeline Automation with Dependencies"
echo "==========================================="

# Function for pipeline stage
execute_pipeline_stage() {
    local stage_name=$1
    local input_file=$2
    local output_file=$3
    local stage_type=$4
    shift 4
    local stage_args=("$@")
    
    log_info "Executing pipeline stage: $stage_name"
    
    case $stage_type in
        "generate")
            if tsiot generate "${stage_args[@]}" --output "$output_file" 2>>"$LOGS_DIR/pipeline_errors.log"; then
                log_success "Stage $stage_name completed"
                return 0
            else
                log_error "Stage $stage_name failed"
                return 1
            fi
            ;;
        "analyze")
            if [ -f "$input_file" ] && tsiot analyze --input "$input_file" "${stage_args[@]}" --output "$output_file" 2>>"$LOGS_DIR/pipeline_errors.log"; then
                log_success "Stage $stage_name completed"
                return 0
            else
                log_error "Stage $stage_name failed (input: $input_file)"
                return 1
            fi
            ;;
        "validate")
            if [ -f "$input_file" ] && tsiot validate --synthetic "$input_file" "${stage_args[@]}" --output "$output_file" 2>>"$LOGS_DIR/pipeline_errors.log"; then
                log_success "Stage $stage_name completed"
                return 0
            else
                log_error "Stage $stage_name failed (input: $input_file)"
                return 1
            fi
            ;;
        "migrate")
            if [ -f "$input_file" ] && tsiot migrate --source "$input_file" --destination "$output_file" "${stage_args[@]}" 2>>"$LOGS_DIR/pipeline_errors.log"; then
                log_success "Stage $stage_name completed"
                return 0
            else
                log_error "Stage $stage_name failed (input: $input_file)"
                return 1
            fi
            ;;
        *)
            log_error "Unknown stage type: $stage_type"
            return 1
            ;;
    esac
}

# Define and execute complex pipeline
run_complex_pipeline() {
    local pipeline_name=$1
    local pipeline_dir="$BATCH_DIR/pipeline_${pipeline_name}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$pipeline_dir"
    
    log_info "Starting complex pipeline: $pipeline_name"
    
    # Stage 1: Generate raw data
    local raw_data="$pipeline_dir/01_raw_data.json"
    if execute_pipeline_stage "data_generation" "" "$raw_data" "generate" \
        --generator "statistical" --sensor-type "temperature" --count 1440 --frequency 1m; then
        
        # Stage 2: Basic analysis
        local basic_analysis="$pipeline_dir/02_basic_analysis.json"
        if execute_pipeline_stage "basic_analysis" "$raw_data" "$basic_analysis" "analyze" \
            --analysis "basic_stats"; then
            
            # Stage 3: Comprehensive analysis
            local comprehensive_analysis="$pipeline_dir/03_comprehensive_analysis.json"
            if execute_pipeline_stage "comprehensive_analysis" "$raw_data" "$comprehensive_analysis" "analyze" \
                --analysis "basic_stats,trend,seasonality"; then
                
                # Stage 4: Quality validation
                local validation_results="$pipeline_dir/04_validation.json"
                if execute_pipeline_stage "quality_validation" "$raw_data" "$validation_results" "validate" \
                    --validators "statistical,distributional" --threshold 0.8; then
                    
                    # Stage 5: Data migration/archival
                    local archived_data="$pipeline_dir/05_archived_data.json"
                    if execute_pipeline_stage "data_archival" "$raw_data" "$archived_data" "migrate" \
                        --batch-size 500; then
                        
                        log_success "Complex pipeline completed successfully: $pipeline_name"
                        
                        # Generate pipeline summary
                        {
                            echo "{"
                            echo "  \"pipeline_name\": \"$pipeline_name\","
                            echo "  \"completion_time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
                            echo "  \"stages_completed\": 5,"
                            echo "  \"status\": \"success\","
                            echo "  \"files\": ["
                            echo "    \"01_raw_data.json\","
                            echo "    \"02_basic_analysis.json\","
                            echo "    \"03_comprehensive_analysis.json\","
                            echo "    \"04_validation.json\","
                            echo "    \"05_archived_data.json\""
                            echo "  ]"
                            echo "}"
                        } > "$pipeline_dir/pipeline_summary.json"
                        
                        return 0
                    fi
                fi
            fi
        fi
    fi
    
    log_error "Complex pipeline failed: $pipeline_name"
    return 1
}

echo "Running complex pipeline with dependencies..."
run_complex_pipeline "production_workflow"

# 4. Automated Quality Monitoring
echo ""
echo "= 4. Automated Quality Monitoring"
echo "=================================="

# Function for continuous quality monitoring
monitor_data_quality() {
    local monitoring_session=$1
    local check_interval=$2
    local max_checks=$3
    
    log_info "Starting quality monitoring session: $monitoring_session"
    
    local monitoring_dir="$BATCH_DIR/monitoring_${monitoring_session}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$monitoring_dir"
    
    local quality_threshold=0.75
    local failed_checks=0
    local max_failed_checks=2
    
    for ((check=1; check<=max_checks; check++)); do
        log_info "Quality check $check/$max_checks"
        
        # Generate test data
        local test_data="$monitoring_dir/check_${check}_data.json"
        if tsiot generate \
            --generator statistical \
            --sensor-type humidity \
            --count 288 \
            --frequency 5m \
            --output "$test_data" \
            2>>"$LOGS_DIR/monitoring_errors.log"; then
            
            # Validate quality
            local quality_report="$monitoring_dir/check_${check}_quality.json"
            if tsiot validate \
                --synthetic "$test_data" \
                --validators "statistical,distributional" \
                --threshold $quality_threshold \
                --output "$quality_report" \
                2>>"$LOGS_DIR/monitoring_errors.log"; then
                
                # Check quality score
                if command -v jq &> /dev/null; then
                    local quality_score=$(jq -r '.overall_quality_score // 0' "$quality_report" 2>/dev/null)
                    local passed=$(jq -r '.passed // false' "$quality_report" 2>/dev/null)
                    
                    if [ "$passed" = "true" ]; then
                        log_success "Quality check $check passed (score: $quality_score)"
                        failed_checks=0  # Reset failed counter on success
                    else
                        log_error "Quality check $check failed (score: $quality_score)"
                        ((failed_checks++))
                        
                        if [ $failed_checks -ge $max_failed_checks ]; then
                            log_error "Maximum failed checks reached ($failed_checks/$max_failed_checks)"
                            log_error "Quality monitoring session terminated early"
                            return 1
                        fi
                    fi
                else
                    log_success "Quality check $check completed (jq not available for detailed analysis)"
                fi
            else
                log_error "Failed to validate data quality for check $check"
                ((failed_checks++))
            fi
        else
            log_error "Failed to generate test data for check $check"
            ((failed_checks++))
        fi
        
        # Wait between checks (reduced for demo)
        if [ $check -lt $max_checks ]; then
            log_info "Waiting $check_interval before next check..."
            sleep 1  # In real usage, this would be much longer
        fi
    done
    
    log_success "Quality monitoring session completed: $monitoring_session"
    
    # Generate monitoring summary
    {
        echo "{"
        echo "  \"session_name\": \"$monitoring_session\","
        echo "  \"total_checks\": $max_checks,"
        echo "  \"failed_checks\": $failed_checks,"
        echo "  \"quality_threshold\": $quality_threshold,"
        echo "  \"completion_time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
        echo "  \"status\": \"completed\""
        echo "}"
    } > "$monitoring_dir/monitoring_summary.json"
    
    return 0
}

echo "Starting automated quality monitoring..."
monitor_data_quality "continuous_monitoring" "30s" 5

# 5. Automated Cleanup and Archival
echo ""
echo "=Ä 5. Automated Cleanup and Archival"
echo "===================================="

# Function for automated cleanup
automated_cleanup() {
    log_info "Starting automated cleanup and archival"
    
    local cleanup_dir="$ARCHIVE_DIR/cleanup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$cleanup_dir"
    
    # Find old files (simulate with current files for demo)
    log_info "Identifying files for cleanup..."
    
    local files_to_archive=()
    local files_to_delete=()
    
    # Collect files for archival (older than 1 minute for demo)
    while IFS= read -r -d '' file; do
        if [ -f "$file" ] && [[ "$file" == *.json ]]; then
            # Get file age in seconds (simplified for demo)
            local file_time=$(stat -f%m "$file" 2>/dev/null || stat -c%Y "$file" 2>/dev/null || echo "0")
            local current_time=$(date +%s)
            local age=$((current_time - file_time))
            
            # Archive files older than 60 seconds (demo threshold)
            if [ $age -gt 60 ]; then
                files_to_archive+=("$file")
            fi
        fi
    done < <(find "$BATCH_DIR" -name "*.json" -print0 2>/dev/null)
    
    # Archive old files
    if [ ${#files_to_archive[@]} -gt 0 ]; then
        log_info "Archiving ${#files_to_archive[@]} old files..."
        
        local archive_file="$cleanup_dir/archived_files_$(date +%Y%m%d_%H%M%S).tar.gz"
        
        if tar -czf "$archive_file" "${files_to_archive[@]}" 2>>"$LOGS_DIR/cleanup_errors.log"; then
            log_success "Created archive: $archive_file"
            
            # Verify archive and remove original files
            if tar -tzf "$archive_file" >/dev/null 2>&1; then
                for file in "${files_to_archive[@]}"; do
                    if rm "$file" 2>>"$LOGS_DIR/cleanup_errors.log"; then
                        log_success "Removed archived file: $(basename "$file")"
                    else
                        log_error "Failed to remove archived file: $(basename "$file")"
                    fi
                done
            else
                log_error "Archive verification failed, keeping original files"
            fi
        else
            log_error "Failed to create archive"
        fi
    else
        log_info "No files found for archival"
    fi
    
    # Cleanup empty directories
    log_info "Cleaning up empty directories..."
    find "$BATCH_DIR" -type d -empty -delete 2>/dev/null || true
    find "$SCHEDULE_DIR" -type d -empty -delete 2>/dev/null || true
    
    # Generate cleanup report
    {
        echo "{"
        echo "  \"cleanup_time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
        echo "  \"files_archived\": ${#files_to_archive[@]},"
        echo "  \"archive_location\": \"$cleanup_dir\","
        echo "  \"total_disk_saved\": \"$(du -sh "$cleanup_dir" 2>/dev/null | cut -f1 || echo "unknown")\""
        echo "}"
    } > "$cleanup_dir/cleanup_report.json"
    
    log_success "Automated cleanup completed"
}

automated_cleanup

# 6. Configuration Management
echo ""
echo "™ 6. Configuration Management"
echo "=============================="

# Function to create automation configurations
create_automation_configs() {
    log_info "Creating automation configuration templates"
    
    # Batch generation config
    cat > "$CONFIG_DIR/batch_generation.json" << 'EOF'
{
  "batch_configs": {
    "industrial_sensors": {
      "sensors": ["temperature", "humidity", "pressure", "co2", "vibration"],
      "generator": "statistical",
      "count": 1440,
      "frequency": "1m",
      "quality_threshold": 0.8
    },
    "environmental_sensors": {
      "sensors": ["temperature", "humidity", "light_intensity", "uv_index"],
      "generator": "timegan",
      "count": 720,
      "frequency": "2m",
      "quality_threshold": 0.85
    }
  }
}
EOF
    
    # Monitoring config
    cat > "$CONFIG_DIR/monitoring.json" << 'EOF'
{
  "monitoring": {
    "check_interval": "30s",
    "quality_threshold": 0.75,
    "max_failed_checks": 3,
    "alert_thresholds": {
      "low_quality": 0.6,
      "critical_quality": 0.4
    }
  }
}
EOF
    
    # Cleanup config
    cat > "$CONFIG_DIR/cleanup.json" << 'EOF'
{
  "cleanup": {
    "archive_age_days": 7,
    "delete_age_days": 30,
    "max_archive_size_gb": 10,
    "compression": "gzip",
    "verify_archives": true
  }
}
EOF
    
    # Pipeline config
    cat > "$CONFIG_DIR/pipeline.json" << 'EOF'
{
  "pipelines": {
    "standard_workflow": {
      "stages": [
        {
          "name": "data_generation",
          "type": "generate",
          "params": {
            "generator": "statistical",
            "count": 1440,
            "frequency": "1m"
          }
        },
        {
          "name": "quality_analysis",
          "type": "analyze",
          "params": {
            "analysis": "basic_stats,trend,seasonality"
          }
        },
        {
          "name": "validation",
          "type": "validate",
          "params": {
            "validators": "statistical,distributional",
            "threshold": 0.8
          }
        }
      ]
    }
  }
}
EOF
    
    log_success "Configuration templates created in $CONFIG_DIR"
    
    # List created configs
    echo "=Ý Created configuration files:"
    for config_file in "$CONFIG_DIR"/*.json; do
        if [ -f "$config_file" ]; then
            echo "   $(basename "$config_file")"
        fi
    done
}

create_automation_configs

# 7. Monitoring and Alerting Simulation
echo ""
echo "=¨ 7. Monitoring and Alerting Simulation"
echo "========================================"

# Function for alert simulation
simulate_alerts() {
    log_info "Simulating monitoring and alerting system"
    
    local alert_log="$LOGS_DIR/alerts.log"
    
    # Simulate different alert scenarios
    local scenarios=("quality_degradation" "generation_failure" "high_resource_usage")
    
    for scenario in "${scenarios[@]}"; do
        echo ""
        echo "= Simulating $scenario scenario..."
        
        case $scenario in
            "quality_degradation")
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALERT: Quality score dropped below threshold (0.65 < 0.75)" >> "$alert_log"
                log_error "ALERT: Data quality degradation detected"
                ;;
            "generation_failure")
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALERT: Data generation failed for sensor type: pressure" >> "$alert_log"
                log_error "ALERT: Data generation failure detected"
                ;;
            "high_resource_usage")
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: High disk usage detected (85% full)" >> "$alert_log"
                log_error "WARNING: High resource usage detected"
                ;;
        esac
        
        # Simulate alert resolution
        sleep 1
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: Alert resolved for $scenario" >> "$alert_log"
        log_success "Alert resolved: $scenario"
    done
    
    echo ""
    echo "=Ê Alert Summary:"
    if [ -f "$alert_log" ]; then
        grep -c "ALERT" "$alert_log" | xargs echo "  Total alerts:"
        grep -c "WARNING" "$alert_log" | xargs echo "  Total warnings:"
        grep -c "resolved" "$alert_log" | xargs echo "  Resolved incidents:"
    fi
}

simulate_alerts

# 8. Performance Benchmarking Automation
echo ""
echo "¡ 8. Performance Benchmarking Automation"
echo "========================================"

# Function for automated benchmarking
run_performance_benchmarks() {
    log_info "Running automated performance benchmarks"
    
    local benchmark_dir="$BATCH_DIR/benchmarks_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$benchmark_dir"
    
    # Define benchmark scenarios
    declare -A BENCHMARK_SCENARIOS=(
        ["small_dataset"]="100"
        ["medium_dataset"]="1000"
        ["large_dataset"]="5000"
    )
    
    local benchmark_results="$benchmark_dir/benchmark_results.json"
    echo "{" > "$benchmark_results"
    echo "  \"benchmark_timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"," >> "$benchmark_results"
    echo "  \"results\": [" >> "$benchmark_results"
    
    local first_result=true
    
    for scenario in "${!BENCHMARK_SCENARIOS[@]}"; do
        local count="${BENCHMARK_SCENARIOS[$scenario]}"
        
        echo ""
        echo "<Ã Benchmarking $scenario (${count} points)..."
        
        # Time the generation operation
        local start_time=$(date +%s.%N)
        local test_file="$benchmark_dir/${scenario}_test.json"
        
        if tsiot generate \
            --generator statistical \
            --sensor-type temperature \
            --count "$count" \
            --frequency 1m \
            --output "$test_file" \
            2>>"$LOGS_DIR/benchmark_errors.log"; then
            
            local end_time=$(date +%s.%N)
            local duration=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "N/A")
            
            # Get file size
            local file_size=$(stat -f%z "$test_file" 2>/dev/null || stat -c%s "$test_file" 2>/dev/null || echo "0")
            
            log_success "Benchmark $scenario completed in ${duration}s (${file_size} bytes)"
            
            # Add to results JSON
            if [ "$first_result" = false ]; then
                echo "," >> "$benchmark_results"
            fi
            first_result=false
            
            cat >> "$benchmark_results" << EOF
    {
      "scenario": "$scenario",
      "data_points": $count,
      "duration_seconds": $duration,
      "file_size_bytes": $file_size,
      "status": "success"
    }
EOF
        else
            log_error "Benchmark $scenario failed"
            
            if [ "$first_result" = false ]; then
                echo "," >> "$benchmark_results"
            fi
            first_result=false
            
            cat >> "$benchmark_results" << EOF
    {
      "scenario": "$scenario",
      "data_points": $count,
      "duration_seconds": null,
      "file_size_bytes": 0,
      "status": "failed"
    }
EOF
        fi
    done
    
    echo "" >> "$benchmark_results"
    echo "  ]" >> "$benchmark_results"
    echo "}" >> "$benchmark_results"
    
    log_success "Performance benchmarks completed, results saved to $benchmark_results"
}

run_performance_benchmarks

# 9. Final Summary and Reporting
echo ""
echo "=Ë 9. Final Summary and Reporting"
echo "================================="

# Generate comprehensive automation report
generate_automation_report() {
    log_info "Generating comprehensive automation report"
    
    local report_file="$AUTOMATION_DIR/automation_report_$(date +%Y%m%d_%H%M%S).json"
    
    # Count generated files
    local total_files=$(find "$AUTOMATION_DIR" -name "*.json" | wc -l)
    local log_files=$(find "$LOGS_DIR" -name "*.log" | wc -l)
    local config_files=$(find "$CONFIG_DIR" -name "*.json" | wc -l)
    
    # Calculate disk usage
    local total_size="unknown"
    if command -v du &> /dev/null; then
        total_size=$(du -sh "$AUTOMATION_DIR" 2>/dev/null | cut -f1 || echo "unknown")
    fi
    
    # Generate report
    cat > "$report_file" << EOF
{
  "automation_report": {
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "summary": {
      "total_json_files": $total_files,
      "log_files": $log_files,
      "config_files": $config_files,
      "total_disk_usage": "$total_size"
    },
    "completed_automations": [
      "batch_data_generation",
      "scheduled_workflows",
      "pipeline_automation",
      "quality_monitoring",
      "cleanup_and_archival",
      "configuration_management",
      "monitoring_and_alerting",
      "performance_benchmarking"
    ],
    "directory_structure": {
      "automation_output": "$AUTOMATION_DIR",
      "scheduled_workflows": "$SCHEDULE_DIR",
      "batch_processing": "$BATCH_DIR",
      "logs": "$LOGS_DIR",
      "configurations": "$CONFIG_DIR",
      "archives": "$ARCHIVE_DIR"
    }
  }
}
EOF
    
    log_success "Automation report generated: $report_file"
    
    # Display summary
    echo ""
    echo "<¯ Automation Summary:"
    echo "======================"
    echo "   Batch data generation: Multiple sensor configurations"
    echo "   Scheduled workflows: Hourly, daily, and weekly automation"
    echo "   Pipeline automation: End-to-end processing with dependencies"
    echo "   Quality monitoring: Continuous data quality assessment"
    echo "   Cleanup and archival: Automated file management"
    echo "   Configuration management: Template-based automation configs"
    echo "   Monitoring and alerting: Simulated alert scenarios"
    echo "   Performance benchmarking: Automated performance testing"
    echo ""
    echo "=Ê Statistics:"
    echo "  Total files generated: $total_files"
    echo "  Log files created: $log_files"
    echo "  Configuration templates: $config_files"
    echo "  Total disk usage: $total_size"
    echo ""
    echo "=Á Output directory: $AUTOMATION_DIR"
    echo "=Ä Detailed report: $report_file"
}

generate_automation_report

# Cleanup instructions
echo ""
echo ">ù Cleanup Instructions:"
echo "========================"
echo "  =Â Review generated files: ls -la $AUTOMATION_DIR"
echo "  =Ë Check logs: cat $LOGS_DIR/automation.log"
echo "  =Ä Archive everything: tar -czf automation_backup_$(date +%Y%m%d).tar.gz $AUTOMATION_DIR"
echo "  =Ñ Clean up when done: rm -rf $AUTOMATION_DIR"

echo ""
echo "=¡ Next Steps for Production:"
echo "============================="
echo "  1. Implement real cron jobs for scheduled workflows"
echo "  2. Set up monitoring dashboards with real alerting"
echo "  3. Configure automated backup and disaster recovery"
echo "  4. Implement resource monitoring and auto-scaling"
echo "  5. Add integration with existing CI/CD pipelines"
echo "  6. Set up centralized logging and metrics collection"
echo "  7. Implement automated testing and validation gates"

echo ""
echo " TSIoT Automation Scripts Complete!"
echo "======================================"
log_info "TSIoT automation demonstration completed successfully"

# Create final automation summary script
cat > "$AUTOMATION_DIR/run_automation.sh" << 'EOF'
#!/bin/bash
# TSIoT Automation Runner
# This script can be used to run specific automation tasks

case "$1" in
    "batch")
        echo "Running batch generation..."
        # Add batch generation commands here
        ;;
    "monitor")
        echo "Starting quality monitoring..."
        # Add monitoring commands here
        ;;
    "cleanup")
        echo "Running cleanup tasks..."
        # Add cleanup commands here
        ;;
    *)
        echo "Usage: $0 {batch|monitor|cleanup}"
        echo "  batch   - Run batch data generation"
        echo "  monitor - Start quality monitoring"
        echo "  cleanup - Run cleanup and archival"
        ;;
esac
EOF

chmod +x "$AUTOMATION_DIR/run_automation.sh"
log_success "Created automation runner script: $AUTOMATION_DIR/run_automation.sh"