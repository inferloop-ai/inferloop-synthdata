#!/bin/bash

# TSIoT Linting Script
# Runs linting and static analysis for the Time Series IoT Synthetic Data Generation Platform

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs/dev"
REPORT_DIR="$PROJECT_ROOT/reports/lint"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Linting tools configuration
GOLANGCI_LINT_VERSION="1.54.2"
ESLINT_CONFIG="$PROJECT_ROOT/.eslintrc.js"
PYLINT_CONFIG="$PROJECT_ROOT/.pylintrc"

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
    --language LANG         Language to lint (go|javascript|python|rust|all) [default: all]
    --severity LEVEL        Minimum severity level (info|warning|error) [default: warning]
    --fix                   Automatically fix issues where possible
    --format FORMAT         Output format (text|json|junit|sarif) [default: text]
    --output-file FILE      Write results to file
    --config-file FILE      Use custom configuration file
    --exclude PATTERN       Exclude files/directories matching pattern
    --include-only PATTERN  Include only files/directories matching pattern
    --disable-rules RULES   Disable specific rules (comma-separated)
    --enable-rules RULES    Enable specific rules (comma-separated)
    --strict                Treat warnings as errors
    --parallel              Run linters in parallel
    --cache                 Use cache for faster runs
    --no-cache              Disable cache
    --baseline FILE         Use baseline file to ignore existing issues
    -v, --verbose           Enable verbose output
    -h, --help              Show this help message

Languages:
    go                      Go source files (*.go)
    javascript              JavaScript/TypeScript files (*.js, *.ts, *.jsx, *.tsx)
    python                  Python files (*.py)
    rust                    Rust files (*.rs)
    all                     All supported languages

Examples:
    $0                              # Lint all languages
    $0 --language go --fix          # Lint and fix Go files
    $0 --format json --strict       # JSON output, warnings as errors
    $0 --exclude "vendor/*"         # Exclude vendor directory
    $0 --parallel --cache           # Fast parallel linting with cache

Environment Variables:
    GOLANGCI_LINT_CONFIG    Custom golangci-lint configuration file
    ESLINT_CONFIG           Custom ESLint configuration file
    PYLINT_CONFIG           Custom Pylint configuration file
    LINT_CACHE_DIR          Directory for lint cache files
EOF
}

# Parse command line arguments
LANGUAGE="all"
SEVERITY="warning"
FIX=false
FORMAT="text"
OUTPUT_FILE=""
CONFIG_FILE=""
EXCLUDE_PATTERN=""
INCLUDE_ONLY=""
DISABLE_RULES=""
ENABLE_RULES=""
STRICT=false
PARALLEL=false
USE_CACHE=true
BASELINE_FILE=""
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --language)
            LANGUAGE="$2"
            shift 2
            ;;
        --severity)
            SEVERITY="$2"
            shift 2
            ;;
        --fix)
            FIX=true
            shift
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --output-file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --config-file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --exclude)
            EXCLUDE_PATTERN="$2"
            shift 2
            ;;
        --include-only)
            INCLUDE_ONLY="$2"
            shift 2
            ;;
        --disable-rules)
            DISABLE_RULES="$2"
            shift 2
            ;;
        --enable-rules)
            ENABLE_RULES="$2"
            shift 2
            ;;
        --strict)
            STRICT=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --cache)
            USE_CACHE=true
            shift
            ;;
        --no-cache)
            USE_CACHE=false
            shift
            ;;
        --baseline)
            BASELINE_FILE="$2"
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

# Validate language
if [[ ! "$LANGUAGE" =~ ^(go|javascript|python|rust|all)$ ]]; then
    log_error "Invalid language: $LANGUAGE"
    usage
    exit 1
fi

# Set environment variables
export GOLANGCI_LINT_CONFIG="${GOLANGCI_LINT_CONFIG:-$PROJECT_ROOT/.golangci.yml}"
export ESLINT_CONFIG="${ESLINT_CONFIG:-$PROJECT_ROOT/.eslintrc.js}"
export PYLINT_CONFIG="${PYLINT_CONFIG:-$PROJECT_ROOT/.pylintrc}"
export LINT_CACHE_DIR="${LINT_CACHE_DIR:-$PROJECT_ROOT/.cache/lint}"

# Create necessary directories
create_directories() {
    mkdir -p "$LOG_DIR" "$REPORT_DIR"
    
    if [[ "$USE_CACHE" == "true" ]]; then
        mkdir -p "$LINT_CACHE_DIR"
    fi
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check required tools
check_required_tools() {
    local missing_tools=()
    
    case "$LANGUAGE" in
        "go"|"all")
            if ! command_exists golangci-lint; then
                missing_tools+=("golangci-lint")
            fi
            ;;
    esac
    
    case "$LANGUAGE" in
        "javascript"|"all")
            if ! command_exists eslint && ! command_exists npx; then
                missing_tools+=("eslint or npx")
            fi
            ;;
    esac
    
    case "$LANGUAGE" in
        "python"|"all")
            if ! command_exists pylint && ! command_exists flake8; then
                missing_tools+=("pylint or flake8")
            fi
            ;;
    esac
    
    case "$LANGUAGE" in
        "rust"|"all")
            if ! command_exists cargo; then
                missing_tools+=("cargo (Rust)")
            fi
            ;;
    esac
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please run ./scripts/dev/setup-dev.sh to install dependencies"
        exit 1
    fi
}

# Build common flags for linters
build_common_flags() {
    local flags=()
    
    if [[ "$VERBOSE" == "true" ]]; then
        flags+=("--verbose")
    fi
    
    if [[ -n "$EXCLUDE_PATTERN" ]]; then
        flags+=("--exclude=$EXCLUDE_PATTERN")
    fi
    
    echo "${flags[@]}"
}

# Lint Go files
lint_go() {
    log_info "Linting Go files..."
    
    if ! command_exists golangci-lint; then
        log_error "golangci-lint not found"
        return 1
    fi
    
    cd "$PROJECT_ROOT"
    
    local flags=()
    local config_flag=""
    
    # Configuration file
    if [[ -n "$CONFIG_FILE" ]]; then
        config_flag="--config=$CONFIG_FILE"
    elif [[ -f "$GOLANGCI_LINT_CONFIG" ]]; then
        config_flag="--config=$GOLANGCI_LINT_CONFIG"
    fi
    
    if [[ -n "$config_flag" ]]; then
        flags+=("$config_flag")
    fi
    
    # Output format
    case "$FORMAT" in
        "json")
            flags+=("--out-format=json")
            ;;
        "junit")
            flags+=("--out-format=junit-xml")
            ;;
        "sarif")
            flags+=("--out-format=sarif")
            ;;
    esac
    
    # Fix issues
    if [[ "$FIX" == "true" ]]; then
        flags+=("--fix")
    fi
    
    # Severity level
    case "$SEVERITY" in
        "error")
            flags+=("--severity=error")
            ;;
        "warning")
            flags+=("--severity=warning")
            ;;
    esac
    
    # Cache settings
    if [[ "$USE_CACHE" == "true" ]]; then
        flags+=("--cache")
    else\n        flags+=(\"--no-cache\")\n    fi\n    \n    # Disable/enable rules\n    if [[ -n \"$DISABLE_RULES\" ]]; then\n        IFS=',' read -ra rules <<< \"$DISABLE_RULES\"\n        for rule in \"${rules[@]}\"; do\n            flags+=(\"--disable=$rule\")\n        done\n    fi\n    \n    if [[ -n \"$ENABLE_RULES\" ]]; then\n        IFS=',' read -ra rules <<< \"$ENABLE_RULES\"\n        for rule in \"${rules[@]}\"; do\n            flags+=(\"--enable=$rule\")\n        done\n    fi\n    \n    # Include/exclude patterns\n    if [[ -n \"$INCLUDE_ONLY\" ]]; then\n        flags+=(\"--path-prefix=$INCLUDE_ONLY\")\n    fi\n    \n    # Output file\n    local output_file=\"$REPORT_DIR/golangci-lint.${FORMAT}\"\n    if [[ -n \"$OUTPUT_FILE\" ]]; then\n        output_file=\"$OUTPUT_FILE\"\n    fi\n    \n    # Run golangci-lint\n    local exit_code=0\n    \n    if [[ \"$FORMAT\" != \"text\" ]] || [[ -n \"$OUTPUT_FILE\" ]]; then\n        golangci-lint run \"${flags[@]}\" > \"$output_file\" 2>&1 || exit_code=$?\n        \n        # Also show output if verbose\n        if [[ \"$VERBOSE\" == \"true\" ]]; then\n            cat \"$output_file\"\n        fi\n    else\n        golangci-lint run \"${flags[@]}\" 2>&1 | tee \"$LOG_DIR/golangci-lint.log\" || exit_code=$?\n    fi\n    \n    if [[ $exit_code -eq 0 ]]; then\n        log_success \"Go linting completed successfully\"\n    else\n        log_error \"Go linting found issues (exit code: $exit_code)\"\n        if [[ \"$STRICT\" == \"true\" ]]; then\n            return $exit_code\n        fi\n    fi\n    \n    return 0\n}\n\n# Lint JavaScript/TypeScript files\nlint_javascript() {\n    log_info \"Linting JavaScript/TypeScript files...\"\n    \n    # Find JavaScript/TypeScript project directories\n    local js_dirs=()\n    \n    if [[ -f \"$PROJECT_ROOT/web/dashboard/package.json\" ]]; then\n        js_dirs+=(\"$PROJECT_ROOT/web/dashboard\")\n    fi\n    \n    if [[ -f \"$PROJECT_ROOT/examples/web-dashboard/package.json\" ]]; then\n        js_dirs+=(\"$PROJECT_ROOT/examples/web-dashboard\")\n    fi\n    \n    if [[ ${#js_dirs[@]} -eq 0 ]]; then\n        log_info \"No JavaScript/TypeScript projects found\"\n        return 0\n    fi\n    \n    local overall_exit_code=0\n    \n    for dir in \"${js_dirs[@]}\"; do\n        log_info \"Linting JavaScript in: $(basename \"$dir\")\"\n        \n        cd \"$dir\"\n        \n        local flags=()\n        local eslint_cmd=\"eslint\"\n        \n        # Use npx if eslint is not globally available\n        if ! command_exists eslint && command_exists npx; then\n            eslint_cmd=\"npx eslint\"\n        fi\n        \n        # Configuration\n        if [[ -n \"$CONFIG_FILE\" ]]; then\n            flags+=(\"--config=$CONFIG_FILE\")\n        elif [[ -f \".eslintrc.js\" ]] || [[ -f \".eslintrc.json\" ]]; then\n            # Use project-specific config\n            :\n        elif [[ -f \"$ESLINT_CONFIG\" ]]; then\n            flags+=(\"--config=$ESLINT_CONFIG\")\n        fi\n        \n        # Output format\n        case \"$FORMAT\" in\n            \"json\")\n                flags+=(\"--format=json\")\n                ;;\n            \"junit\")\n                flags+=(\"--format=junit\")\n                ;;\n        esac\n        \n        # Fix issues\n        if [[ \"$FIX\" == \"true\" ]]; then\n            flags+=(\"--fix\")\n        fi\n        \n        # Cache settings\n        if [[ \"$USE_CACHE\" == \"true\" ]]; then\n            flags+=(\"--cache\")\n            flags+=(\"--cache-location=$LINT_CACHE_DIR/eslint\")\n        else\n            flags+=(\"--no-eslintrc\")\n        fi\n        \n        # File patterns\n        local patterns=(\"src/**/*.{js,jsx,ts,tsx}\" \"*.{js,jsx,ts,tsx}\")\n        \n        if [[ -n \"$INCLUDE_ONLY\" ]]; then\n            patterns=(\"$INCLUDE_ONLY\")\n        fi\n        \n        # Exclude patterns\n        if [[ -n \"$EXCLUDE_PATTERN\" ]]; then\n            flags+=(\"--ignore-pattern=$EXCLUDE_PATTERN\")\n        fi\n        \n        # Output file\n        local output_file=\"$REPORT_DIR/eslint-$(basename \"$dir\").${FORMAT}\"\n        if [[ -n \"$OUTPUT_FILE\" ]]; then\n            output_file=\"$OUTPUT_FILE\"\n        fi\n        \n        # Run ESLint\n        local exit_code=0\n        \n        if [[ \"$FORMAT\" != \"text\" ]] || [[ -n \"$OUTPUT_FILE\" ]]; then\n            $eslint_cmd \"${flags[@]}\" \"${patterns[@]}\" > \"$output_file\" 2>&1 || exit_code=$?\n        else\n            $eslint_cmd \"${flags[@]}\" \"${patterns[@]}\" 2>&1 | tee \"$LOG_DIR/eslint-$(basename \"$dir\").log\" || exit_code=$?\n        fi\n        \n        if [[ $exit_code -eq 0 ]]; then\n            log_success \"JavaScript linting completed for $(basename \"$dir\")\"\n        else\n            log_error \"JavaScript linting found issues in $(basename \"$dir\") (exit code: $exit_code)\"\n            overall_exit_code=$exit_code\n        fi\n    done\n    \n    if [[ $overall_exit_code -eq 0 ]]; then\n        log_success \"JavaScript/TypeScript linting completed successfully\"\n    elif [[ \"$STRICT\" == \"true\" ]]; then\n        return $overall_exit_code\n    fi\n    \n    return 0\n}\n\n# Lint Python files\nlint_python() {\n    log_info \"Linting Python files...\"\n    \n    # Find Python project directories\n    local python_dirs=()\n    \n    if find \"$PROJECT_ROOT\" -name \"*.py\" -type f | grep -q .; then\n        python_dirs+=(\"$PROJECT_ROOT\")\n    fi\n    \n    if [[ -f \"$PROJECT_ROOT/internal/sdk/python/setup.py\" ]]; then\n        python_dirs+=(\"$PROJECT_ROOT/internal/sdk/python\")\n    fi\n    \n    if [[ ${#python_dirs[@]} -eq 0 ]]; then\n        log_info \"No Python files found\"\n        return 0\n    fi\n    \n    local overall_exit_code=0\n    \n    # Activate virtual environment if available\n    if [[ -f \"$PROJECT_ROOT/.dev/venv/bin/activate\" ]]; then\n        source \"$PROJECT_ROOT/.dev/venv/bin/activate\"\n    fi\n    \n    for dir in \"${python_dirs[@]}\"; do\n        log_info \"Linting Python in: $(basename \"$dir\")\"\n        \n        cd \"$dir\"\n        \n        # Use pylint if available, otherwise flake8\n        if command_exists pylint; then\n            lint_python_with_pylint \"$dir\"\n        elif command_exists flake8; then\n            lint_python_with_flake8 \"$dir\"\n        else\n            log_warning \"No Python linter available\"\n            continue\n        fi\n        \n        local exit_code=$?\n        if [[ $exit_code -ne 0 ]]; then\n            overall_exit_code=$exit_code\n        fi\n    done\n    \n    if [[ $overall_exit_code -eq 0 ]]; then\n        log_success \"Python linting completed successfully\"\n    elif [[ \"$STRICT\" == \"true\" ]]; then\n        return $overall_exit_code\n    fi\n    \n    return 0\n}\n\n# Lint Python with pylint\nlint_python_with_pylint() {\n    local dir=\"$1\"\n    local flags=()\n    \n    # Configuration\n    if [[ -n \"$CONFIG_FILE\" ]]; then\n        flags+=(\"--rcfile=$CONFIG_FILE\")\n    elif [[ -f \"$dir/.pylintrc\" ]]; then\n        flags+=(\"--rcfile=$dir/.pylintrc\")\n    elif [[ -f \"$PYLINT_CONFIG\" ]]; then\n        flags+=(\"--rcfile=$PYLINT_CONFIG\")\n    fi\n    \n    # Output format\n    case \"$FORMAT\" in\n        \"json\")\n            flags+=(\"--output-format=json\")\n            ;;\n        \"junit\")\n            flags+=(\"--output-format=junit\")\n            ;;\n    esac\n    \n    # Disable/enable rules\n    if [[ -n \"$DISABLE_RULES\" ]]; then\n        flags+=(\"--disable=$DISABLE_RULES\")\n    fi\n    \n    if [[ -n \"$ENABLE_RULES\" ]]; then\n        flags+=(\"--enable=$ENABLE_RULES\")\n    fi\n    \n    # Find Python files\n    local python_files\n    if [[ -n \"$INCLUDE_ONLY\" ]]; then\n        python_files=$(find . -path \"$INCLUDE_ONLY\" -name \"*.py\" -type f)\n    else\n        python_files=$(find . -name \"*.py\" -type f)\n    fi\n    \n    if [[ -n \"$EXCLUDE_PATTERN\" ]]; then\n        python_files=$(echo \"$python_files\" | grep -v \"$EXCLUDE_PATTERN\")\n    fi\n    \n    if [[ -z \"$python_files\" ]]; then\n        log_info \"No Python files to lint in $(basename \"$dir\")\"\n        return 0\n    fi\n    \n    # Output file\n    local output_file=\"$REPORT_DIR/pylint-$(basename \"$dir\").${FORMAT}\"\n    if [[ -n \"$OUTPUT_FILE\" ]]; then\n        output_file=\"$OUTPUT_FILE\"\n    fi\n    \n    # Run pylint\n    local exit_code=0\n    \n    if [[ \"$FORMAT\" != \"text\" ]] || [[ -n \"$OUTPUT_FILE\" ]]; then\n        # shellcheck disable=SC2086\n        pylint \"${flags[@]}\" $python_files > \"$output_file\" 2>&1 || exit_code=$?\n    else\n        # shellcheck disable=SC2086\n        pylint \"${flags[@]}\" $python_files 2>&1 | tee \"$LOG_DIR/pylint-$(basename \"$dir\").log\" || exit_code=$?\n    fi\n    \n    if [[ $exit_code -eq 0 ]]; then\n        log_success \"Pylint completed for $(basename \"$dir\")\"\n    else\n        log_error \"Pylint found issues in $(basename \"$dir\") (exit code: $exit_code)\"\n    fi\n    \n    return $exit_code\n}\n\n# Lint Python with flake8\nlint_python_with_flake8() {\n    local dir=\"$1\"\n    local flags=()\n    \n    # Configuration\n    if [[ -n \"$CONFIG_FILE\" ]]; then\n        flags+=(\"--config=$CONFIG_FILE\")\n    elif [[ -f \"$dir/.flake8\" ]]; then\n        flags+=(\"--config=$dir/.flake8\")\n    elif [[ -f \"$dir/setup.cfg\" ]]; then\n        flags+=(\"--config=$dir/setup.cfg\")\n    fi\n    \n    # Output format\n    case \"$FORMAT\" in\n        \"json\")\n            flags+=(\"--format=json\")\n            ;;\n    esac\n    \n    # Exclude patterns\n    if [[ -n \"$EXCLUDE_PATTERN\" ]]; then\n        flags+=(\"--exclude=$EXCLUDE_PATTERN\")\n    fi\n    \n    # Target directory\n    local target=\".\"\n    if [[ -n \"$INCLUDE_ONLY\" ]]; then\n        target=\"$INCLUDE_ONLY\"\n    fi\n    \n    # Output file\n    local output_file=\"$REPORT_DIR/flake8-$(basename \"$dir\").${FORMAT}\"\n    if [[ -n \"$OUTPUT_FILE\" ]]; then\n        output_file=\"$OUTPUT_FILE\"\n    fi\n    \n    # Run flake8\n    local exit_code=0\n    \n    if [[ \"$FORMAT\" != \"text\" ]] || [[ -n \"$OUTPUT_FILE\" ]]; then\n        flake8 \"${flags[@]}\" \"$target\" > \"$output_file\" 2>&1 || exit_code=$?\n    else\n        flake8 \"${flags[@]}\" \"$target\" 2>&1 | tee \"$LOG_DIR/flake8-$(basename \"$dir\").log\" || exit_code=$?\n    fi\n    \n    if [[ $exit_code -eq 0 ]]; then\n        log_success \"Flake8 completed for $(basename \"$dir\")\"\n    else\n        log_error \"Flake8 found issues in $(basename \"$dir\") (exit code: $exit_code)\"\n    fi\n    \n    return $exit_code\n}\n\n# Lint Rust files\nlint_rust() {\n    log_info \"Linting Rust files...\"\n    \n    # Find Rust project directories\n    local rust_dirs=()\n    \n    if [[ -f \"$PROJECT_ROOT/internal/sdk/rust/Cargo.toml\" ]]; then\n        rust_dirs+=(\"$PROJECT_ROOT/internal/sdk/rust\")\n    fi\n    \n    if [[ ${#rust_dirs[@]} -eq 0 ]]; then\n        log_info \"No Rust projects found\"\n        return 0\n    fi\n    \n    local overall_exit_code=0\n    \n    for dir in \"${rust_dirs[@]}\"; do\n        log_info \"Linting Rust in: $(basename \"$dir\")\"\n        \n        cd \"$dir\"\n        \n        # Run clippy (Rust linter)\n        local clippy_flags=()\n        \n        if [[ \"$VERBOSE\" == \"true\" ]]; then\n            clippy_flags+=(\"--verbose\")\n        fi\n        \n        if [[ \"$FIX\" == \"true\" ]]; then\n            clippy_flags+=(\"--fix\")\n        fi\n        \n        # Severity level\n        case \"$SEVERITY\" in\n            \"error\")\n                clippy_flags+=(\"--\" \"-D\" \"warnings\")\n                ;;\n        esac\n        \n        # Run clippy\n        local exit_code=0\n        \n        cargo clippy \"${clippy_flags[@]}\" 2>&1 | tee \"$LOG_DIR/clippy-$(basename \"$dir\").log\" || exit_code=$?\n        \n        if [[ $exit_code -eq 0 ]]; then\n            log_success \"Rust linting completed for $(basename \"$dir\")\"\n        else\n            log_error \"Rust linting found issues in $(basename \"$dir\") (exit code: $exit_code)\"\n            overall_exit_code=$exit_code\n        fi\n        \n        # Run rustfmt for formatting\n        if [[ \"$FIX\" == \"true\" ]]; then\n            cargo fmt\n            log_info \"Rust formatting applied for $(basename \"$dir\")\"\n        fi\n    done\n    \n    if [[ $overall_exit_code -eq 0 ]]; then\n        log_success \"Rust linting completed successfully\"\n    elif [[ \"$STRICT\" == \"true\" ]]; then\n        return $overall_exit_code\n    fi\n    \n    return 0\n}\n\n# Run linters in parallel\nrun_parallel_linting() {\n    log_info \"Running linters in parallel...\"\n    \n    local pids=()\n    local results=()\n    \n    case \"$LANGUAGE\" in\n        \"go\")\n            lint_go &\n            pids+=($!)\n            results+=(\"go\")\n            ;;\n        \"javascript\")\n            lint_javascript &\n            pids+=($!)\n            results+=(\"javascript\")\n            ;;\n        \"python\")\n            lint_python &\n            pids+=($!)\n            results+=(\"python\")\n            ;;\n        \"rust\")\n            lint_rust &\n            pids+=($!)\n            results+=(\"rust\")\n            ;;\n        \"all\")\n            lint_go &\n            pids+=($!)\n            results+=(\"go\")\n            \n            lint_javascript &\n            pids+=($!)\n            results+=(\"javascript\")\n            \n            lint_python &\n            pids+=($!)\n            results+=(\"python\")\n            \n            lint_rust &\n            pids+=($!)\n            results+=(\"rust\")\n            ;;\n    esac\n    \n    # Wait for all processes and collect results\n    local overall_exit_code=0\n    \n    for i in \"${!pids[@]}\"; do\n        local pid=${pids[$i]}\n        local lang=${results[$i]}\n        \n        if wait \"$pid\"; then\n            log_success \"$lang linting completed successfully\"\n        else\n            local exit_code=$?\n            log_error \"$lang linting failed (exit code: $exit_code)\"\n            overall_exit_code=$exit_code\n        fi\n    done\n    \n    return $overall_exit_code\n}\n\n# Run linters sequentially\nrun_sequential_linting() {\n    local overall_exit_code=0\n    \n    case \"$LANGUAGE\" in\n        \"go\")\n            lint_go || overall_exit_code=$?\n            ;;\n        \"javascript\")\n            lint_javascript || overall_exit_code=$?\n            ;;\n        \"python\")\n            lint_python || overall_exit_code=$?\n            ;;\n        \"rust\")\n            lint_rust || overall_exit_code=$?\n            ;;\n        \"all\")\n            lint_go || overall_exit_code=$?\n            lint_javascript || overall_exit_code=$?\n            lint_python || overall_exit_code=$?\n            lint_rust || overall_exit_code=$?\n            ;;\n    esac\n    \n    return $overall_exit_code\n}\n\n# Generate linting summary\ngenerate_summary() {\n    log_info \"Generating linting summary...\"\n    \n    local summary_file=\"$REPORT_DIR/lint-summary.json\"\n    \n    cat > \"$summary_file\" << EOF\n{\n    \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\n    \"linting\": {\n        \"language\": \"$LANGUAGE\",\n        \"severity\": \"$SEVERITY\",\n        \"format\": \"$FORMAT\",\n        \"fix\": $FIX,\n        \"strict\": $STRICT,\n        \"parallel\": $PARALLEL,\n        \"cache\": $USE_CACHE\n    },\n    \"reports\": {\n        \"directory\": \"$REPORT_DIR\",\n        \"logs\": \"$LOG_DIR\"\n    },\n    \"tools\": {\n        \"golangci-lint\": $(command_exists golangci-lint && echo \"true\" || echo \"false\"),\n        \"eslint\": $(command_exists eslint && echo \"true\" || echo \"false\"),\n        \"pylint\": $(command_exists pylint && echo \"true\" || echo \"false\"),\n        \"flake8\": $(command_exists flake8 && echo \"true\" || echo \"false\"),\n        \"cargo\": $(command_exists cargo && echo \"true\" || echo \"false\")\n    }\n}\nEOF\n    \n    log_success \"Linting summary generated: $summary_file\"\n}\n\n# Main execution\nmain() {\n    log_info \"Starting linting process\"\n    log_info \"Language: $LANGUAGE\"\n    log_info \"Severity: $SEVERITY\"\n    log_info \"Format: $FORMAT\"\n    log_info \"Fix: $FIX\"\n    log_info \"Parallel: $PARALLEL\"\n    \n    create_directories\n    check_required_tools\n    \n    # Change to project root\n    cd \"$PROJECT_ROOT\"\n    \n    # Run linting\n    local exit_code=0\n    \n    if [[ \"$PARALLEL\" == \"true\" ]]; then\n        run_parallel_linting || exit_code=$?\n    else\n        run_sequential_linting || exit_code=$?\n    fi\n    \n    # Generate summary\n    generate_summary\n    \n    if [[ $exit_code -eq 0 ]]; then\n        log_success \"Linting completed successfully\"\n    else\n        log_error \"Linting completed with issues\"\n        if [[ \"$STRICT\" == \"true\" ]]; then\n            exit $exit_code\n        fi\n    fi\n    \n    return $exit_code\n}\n\n# Execute main function\nmain \"$@\""}, {"replace_all": false}]