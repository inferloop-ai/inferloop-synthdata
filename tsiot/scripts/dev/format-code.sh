#!/bin/bash

# TSIoT Code Formatting Script
# Formats code for the Time Series IoT Synthetic Data Generation Platform

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs/dev"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
    --language LANG         Language to format (go|javascript|python|rust|yaml|json|all) [default: all]
    --check                 Only check formatting without making changes
    --diff                  Show diff of formatting changes
    --write                 Write changes to files (default)
    --backup                Create backup files before formatting
    --exclude PATTERN       Exclude files/directories matching pattern
    --include-only PATTERN  Include only files/directories matching pattern
    --config-file FILE      Use custom configuration file
    --line-length LENGTH    Set maximum line length [default: language-specific]
    --tab-width WIDTH       Set tab width [default: language-specific]
    --use-tabs              Use tabs instead of spaces
    --sort-imports          Sort imports (where applicable)
    --remove-unused         Remove unused imports (where applicable)
    --parallel              Format files in parallel
    -v, --verbose           Enable verbose output
    -h, --help              Show this help message

Languages:
    go                      Go source files (*.go)
    javascript              JavaScript/TypeScript files (*.js, *.ts, *.jsx, *.tsx)
    python                  Python files (*.py)
    rust                    Rust files (*.rs)
    yaml                    YAML files (*.yml, *.yaml)
    json                    JSON files (*.json)
    markdown                Markdown files (*.md)
    all                     All supported languages

Examples:
    $0                              # Format all files
    $0 --language go --check        # Check Go formatting without changes
    $0 --diff --exclude "vendor/*"  # Show formatting diff, exclude vendor
    $0 --backup --write             # Format with backup
    $0 --parallel                   # Fast parallel formatting

Environment Variables:
    GO_FMT_FLAGS            Additional flags for gofmt/goimports
    PRETTIER_CONFIG         Prettier configuration file
    BLACK_CONFIG            Black configuration file
    RUSTFMT_CONFIG          rustfmt configuration file
EOF
}

# Parse command line arguments
LANGUAGE="all"
CHECK_ONLY=false
SHOW_DIFF=false
WRITE_CHANGES=true
CREATE_BACKUP=false
EXCLUDE_PATTERN=""
INCLUDE_ONLY=""
CONFIG_FILE=""
LINE_LENGTH=""
TAB_WIDTH=""
USE_TABS=false
SORT_IMPORTS=true
REMOVE_UNUSED=true
PARALLEL=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --language)
            LANGUAGE="$2"
            shift 2
            ;;
        --check)
            CHECK_ONLY=true
            WRITE_CHANGES=false
            shift
            ;;
        --diff)
            SHOW_DIFF=true
            shift
            ;;
        --write)
            WRITE_CHANGES=true
            CHECK_ONLY=false
            shift
            ;;
        --backup)
            CREATE_BACKUP=true
            shift
            ;;
        --exclude)
            EXCLUDE_PATTERN="$2"
            shift 2
            ;;
        --include-only)
            INCLUDE_ONLY="$2"
            shift 2
            ;;
        --config-file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --line-length)
            LINE_LENGTH="$2"
            shift 2
            ;;
        --tab-width)
            TAB_WIDTH="$2"
            shift 2
            ;;
        --use-tabs)
            USE_TABS=true
            shift
            ;;
        --sort-imports)
            SORT_IMPORTS=true
            shift
            ;;
        --remove-unused)
            REMOVE_UNUSED=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
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
if [[ ! "$LANGUAGE" =~ ^(go|javascript|python|rust|yaml|json|markdown|all)$ ]]; then
    log_error "Invalid language: $LANGUAGE"
    usage
    exit 1
fi

# Set environment variables
export GO_FMT_FLAGS="${GO_FMT_FLAGS:-}"
export PRETTIER_CONFIG="${PRETTIER_CONFIG:-$PROJECT_ROOT/.prettierrc}"
export BLACK_CONFIG="${BLACK_CONFIG:-$PROJECT_ROOT/pyproject.toml}"
export RUSTFMT_CONFIG="${RUSTFMT_CONFIG:-$PROJECT_ROOT/rustfmt.toml}"

# Create necessary directories
create_directories() {
    mkdir -p "$LOG_DIR"
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
            if ! command_exists gofmt; then
                missing_tools+=("gofmt")
            fi
            ;;
    esac
    
    case "$LANGUAGE" in
        "javascript"|"all")
            if ! command_exists prettier && ! command_exists npx; then
                missing_tools+=("prettier or npx")
            fi
            ;;
    esac
    
    case "$LANGUAGE" in
        "python"|"all")
            if ! command_exists black && ! command_exists autopep8; then
                missing_tools+=("black or autopep8")
            fi
            ;;
    esac
    
    case "$LANGUAGE" in
        "rust"|"all")
            if ! command_exists rustfmt; then
                missing_tools+=("rustfmt")
            fi
            ;;
    esac
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_warning "Missing formatting tools: ${missing_tools[*]}"
        log_warning "Some formatting operations may be skipped"
    fi
}

# Find files to format
find_files() {
    local file_patterns="$1"
    local files=()
    
    IFS='|' read -ra patterns <<< "$file_patterns"
    
    for pattern in "${patterns[@]}"; do
        if [[ -n "$INCLUDE_ONLY" ]]; then
            # Find files matching both the pattern and include filter
            while IFS= read -r -d '' file; do
                if [[ "$file" == *"$INCLUDE_ONLY"* ]]; then
                    files+=("$file")
                fi
            done < <(find "$PROJECT_ROOT" -name "$pattern" -type f -print0 2>/dev/null)
        else
            # Find all files matching the pattern
            while IFS= read -r -d '' file; do
                files+=("$file")
            done < <(find "$PROJECT_ROOT" -name "$pattern" -type f -print0 2>/dev/null)
        fi
    done
    
    # Apply exclude pattern
    if [[ -n "$EXCLUDE_PATTERN" ]]; then
        local filtered_files=()
        for file in "${files[@]}"; do
            if [[ ! "$file" =~ $EXCLUDE_PATTERN ]]; then
                filtered_files+=("$file")
            fi
        done
        files=("${filtered_files[@]}")
    fi
    
    printf '%s\n' "${files[@]}"
}

# Create backup of file
create_file_backup() {
    local file="$1"
    
    if [[ "$CREATE_BACKUP" == "true" ]]; then
        cp "$file" "$file.bak"
        if [[ "$VERBOSE" == "true" ]]; then
            log_info "Created backup: $file.bak"
        fi
    fi
}

# Format Go files
format_go() {
    log_info "Formatting Go files..."
    
    if ! command_exists gofmt; then
        log_warning "gofmt not found, skipping Go formatting"
        return 0
    fi
    
    # Find Go files
    local go_files
    mapfile -t go_files < <(find_files "*.go")
    
    if [[ ${#go_files[@]} -eq 0 ]]; then
        log_info "No Go files found"
        return 0
    fi
    
    log_info "Found ${#go_files[@]} Go files to format"
    
    local formatted_count=0
    local error_count=0
    
    for file in "${go_files[@]}"; do
        if [[ "$VERBOSE" == "true" ]]; then
            log_info "Formatting: $file"
        fi
        
        # Create backup if requested
        create_file_backup "$file"
        
        # Format with goimports if available, otherwise gofmt
        local fmt_cmd="gofmt"
        if command_exists goimports; then
            fmt_cmd="goimports"
        fi
        
        # Build format command
        local fmt_flags=()
        
        if [[ "$USE_TABS" == "false" ]]; then
            fmt_flags+=("-tabs=false")
        fi
        
        if [[ -n "$TAB_WIDTH" ]]; then
            fmt_flags+=("-tabwidth=$TAB_WIDTH")
        fi
        
        if [[ "$CHECK_ONLY" == "true" ]]; then
            # Check if file needs formatting
            if ! $fmt_cmd "${fmt_flags[@]}" -d "$file" | diff -q "$file" - >/dev/null 2>&1; then
                if [[ "$SHOW_DIFF" == "true" ]]; then
                    echo "=== $file ==="
                    $fmt_cmd "${fmt_flags[@]}" -d "$file"
                    echo ""
                fi
                ((formatted_count++))
            fi
        else
            # Format file
            local temp_file
            temp_file=$(mktemp)
            
            if $fmt_cmd "${fmt_flags[@]}" "$file" > "$temp_file" 2>/dev/null; then
                if [[ "$SHOW_DIFF" == "true" ]]; then
                    if ! diff -q "$file" "$temp_file" >/dev/null 2>&1; then
                        echo "=== $file ===\"\n                        diff -u \"$file\" \"$temp_file\" || true\n                        echo \"\"\n                    fi\n                fi\n                \n                if [[ \"$WRITE_CHANGES\" == \"true\" ]]; then\n                    mv \"$temp_file\" \"$file\"\n                    ((formatted_count++))\n                else\n                    rm \"$temp_file\"\n                fi\n            else\n                log_error \"Failed to format: $file\"\n                rm \"$temp_file\"\n                ((error_count++))\n            fi\n        fi\n    done\n    \n    if [[ $error_count -eq 0 ]]; then\n        if [[ \"$CHECK_ONLY\" == \"true\" ]]; then\n            log_success \"Go format check completed: $formatted_count files need formatting\"\n        else\n            log_success \"Go formatting completed: $formatted_count files formatted\"\n        fi\n    else\n        log_warning \"Go formatting completed with $error_count errors\"\n    fi\n    \n    return $error_count\n}\n\n# Format JavaScript/TypeScript files\nformat_javascript() {\n    log_info \"Formatting JavaScript/TypeScript files...\"\n    \n    # Check for prettier\n    local prettier_cmd=\"prettier\"\n    if ! command_exists prettier && command_exists npx; then\n        prettier_cmd=\"npx prettier\"\n    elif ! command_exists prettier; then\n        log_warning \"prettier not found, skipping JavaScript/TypeScript formatting\"\n        return 0\n    fi\n    \n    # Find JavaScript/TypeScript files\n    local js_files\n    mapfile -t js_files < <(find_files \"*.js|*.jsx|*.ts|*.tsx\")\n    \n    if [[ ${#js_files[@]} -eq 0 ]]; then\n        log_info \"No JavaScript/TypeScript files found\"\n        return 0\n    fi\n    \n    log_info \"Found ${#js_files[@]} JavaScript/TypeScript files to format\"\n    \n    # Build prettier flags\n    local prettier_flags=()\n    \n    if [[ -n \"$CONFIG_FILE\" ]]; then\n        prettier_flags+=(\"--config=$CONFIG_FILE\")\n    elif [[ -f \"$PRETTIER_CONFIG\" ]]; then\n        prettier_flags+=(\"--config=$PRETTIER_CONFIG\")\n    fi\n    \n    if [[ -n \"$LINE_LENGTH\" ]]; then\n        prettier_flags+=(\"--print-width=$LINE_LENGTH\")\n    fi\n    \n    if [[ -n \"$TAB_WIDTH\" ]]; then\n        prettier_flags+=(\"--tab-width=$TAB_WIDTH\")\n    fi\n    \n    if [[ \"$USE_TABS\" == \"true\" ]]; then\n        prettier_flags+=(\"--use-tabs\")\n    fi\n    \n    if [[ \"$CHECK_ONLY\" == \"true\" ]]; then\n        prettier_flags+=(\"--check\")\n    fi\n    \n    if [[ \"$WRITE_CHANGES\" == \"true\" ]] && [[ \"$CHECK_ONLY\" == \"false\" ]]; then\n        prettier_flags+=(\"--write\")\n    fi\n    \n    # Process files\n    local formatted_count=0\n    local error_count=0\n    \n    if [[ \"$PARALLEL\" == \"true\" ]] && [[ ${#js_files[@]} -gt 10 ]]; then\n        # Process in batches for parallel execution\n        local batch_size=10\n        for ((i=0; i<${#js_files[@]}; i+=batch_size)); do\n            local batch=(\"${js_files[@]:i:batch_size}\")\n            \n            for file in \"${batch[@]}\"; do\n                {\n                    format_js_file \"$file\" \"$prettier_cmd\" \"${prettier_flags[@]}\"\n                } &\n            done\n            \n            wait\n        done\n    else\n        # Sequential processing\n        for file in \"${js_files[@]}\"; do\n            if format_js_file \"$file\" \"$prettier_cmd\" \"${prettier_flags[@]}\"; then\n                ((formatted_count++))\n            else\n                ((error_count++))\n            fi\n        done\n    fi\n    \n    if [[ $error_count -eq 0 ]]; then\n        if [[ \"$CHECK_ONLY\" == \"true\" ]]; then\n            log_success \"JavaScript/TypeScript format check completed\"\n        else\n            log_success \"JavaScript/TypeScript formatting completed\"\n        fi\n    else\n        log_warning \"JavaScript/TypeScript formatting completed with $error_count errors\"\n    fi\n    \n    return $error_count\n}\n\n# Format single JavaScript file\nformat_js_file() {\n    local file=\"$1\"\n    local prettier_cmd=\"$2\"\n    shift 2\n    local prettier_flags=(\"$@\")\n    \n    if [[ \"$VERBOSE\" == \"true\" ]]; then\n        log_info \"Formatting: $file\"\n    fi\n    \n    # Create backup if requested\n    create_file_backup \"$file\"\n    \n    # Format file\n    if $prettier_cmd \"${prettier_flags[@]}\" \"$file\" >/dev/null 2>&1; then\n        return 0\n    else\n        log_error \"Failed to format: $file\"\n        return 1\n    fi\n}\n\n# Format Python files\nformat_python() {\n    log_info \"Formatting Python files...\"\n    \n    # Determine Python formatter\n    local python_formatter=\"\"\n    if command_exists black; then\n        python_formatter=\"black\"\n    elif command_exists autopep8; then\n        python_formatter=\"autopep8\"\n    else\n        log_warning \"No Python formatter found (black, autopep8), skipping Python formatting\"\n        return 0\n    fi\n    \n    # Find Python files\n    local py_files\n    mapfile -t py_files < <(find_files \"*.py\")\n    \n    if [[ ${#py_files[@]} -eq 0 ]]; then\n        log_info \"No Python files found\"\n        return 0\n    fi\n    \n    log_info \"Found ${#py_files[@]} Python files to format with $python_formatter\"\n    \n    # Activate virtual environment if available\n    if [[ -f \"$PROJECT_ROOT/.dev/venv/bin/activate\" ]]; then\n        source \"$PROJECT_ROOT/.dev/venv/bin/activate\"\n    fi\n    \n    local formatted_count=0\n    local error_count=0\n    \n    case \"$python_formatter\" in\n        \"black\")\n            format_python_with_black \"${py_files[@]}\"\n            ;;\n        \"autopep8\")\n            format_python_with_autopep8 \"${py_files[@]}\"\n            ;;\n    esac\n    \n    return $?\n}\n\n# Format Python with Black\nformat_python_with_black() {\n    local files=(\"$@\")\n    local black_flags=()\n    \n    # Configuration\n    if [[ -n \"$CONFIG_FILE\" ]]; then\n        black_flags+=(\"--config=$CONFIG_FILE\")\n    elif [[ -f \"$BLACK_CONFIG\" ]]; then\n        # Black reads from pyproject.toml automatically\n        :\n    fi\n    \n    # Line length\n    if [[ -n \"$LINE_LENGTH\" ]]; then\n        black_flags+=(\"--line-length=$LINE_LENGTH\")\n    fi\n    \n    # Check only mode\n    if [[ \"$CHECK_ONLY\" == \"true\" ]]; then\n        black_flags+=(\"--check\")\n    fi\n    \n    # Show diff\n    if [[ \"$SHOW_DIFF\" == \"true\" ]]; then\n        black_flags+=(\"--diff\")\n    fi\n    \n    # Verbose mode\n    if [[ \"$VERBOSE\" == \"true\" ]]; then\n        black_flags+=(\"--verbose\")\n    fi\n    \n    # Process files\n    local error_count=0\n    \n    for file in \"${files[@]}\"; do\n        # Create backup if requested\n        create_file_backup \"$file\"\n        \n        if ! black \"${black_flags[@]}\" \"$file\"; then\n            ((error_count++))\n        fi\n    done\n    \n    if [[ $error_count -eq 0 ]]; then\n        if [[ \"$CHECK_ONLY\" == \"true\" ]]; then\n            log_success \"Python format check completed with Black\"\n        else\n            log_success \"Python formatting completed with Black\"\n        fi\n    else\n        log_warning \"Python formatting completed with $error_count errors\"\n    fi\n    \n    return $error_count\n}\n\n# Format Python with autopep8\nformat_python_with_autopep8() {\n    local files=(\"$@\")\n    local autopep8_flags=()\n    \n    # Configuration\n    if [[ -n \"$CONFIG_FILE\" ]]; then\n        autopep8_flags+=(\"--global-config=$CONFIG_FILE\")\n    fi\n    \n    # Line length\n    if [[ -n \"$LINE_LENGTH\" ]]; then\n        autopep8_flags+=(\"--max-line-length=$LINE_LENGTH\")\n    fi\n    \n    # In-place editing\n    if [[ \"$WRITE_CHANGES\" == \"true\" ]] && [[ \"$CHECK_ONLY\" == \"false\" ]]; then\n        autopep8_flags+=(\"--in-place\")\n    fi\n    \n    # Aggressive mode\n    autopep8_flags+=(\"--aggressive\" \"--aggressive\")\n    \n    # Process files\n    local formatted_count=0\n    local error_count=0\n    \n    for file in \"${files[@]}\"; do\n        if [[ \"$VERBOSE\" == \"true\" ]]; then\n            log_info \"Formatting: $file\"\n        fi\n        \n        # Create backup if requested\n        create_file_backup \"$file\"\n        \n        if [[ \"$CHECK_ONLY\" == \"true\" ]]; then\n            # Check if file needs formatting\n            local temp_file\n            temp_file=$(mktemp)\n            \n            if autopep8 \"${autopep8_flags[@]}\" \"$file\" > \"$temp_file\" 2>/dev/null; then\n                if ! diff -q \"$file\" \"$temp_file\" >/dev/null 2>&1; then\n                    if [[ \"$SHOW_DIFF\" == \"true\" ]]; then\n                        echo \"=== $file ===\"\n                        diff -u \"$file\" \"$temp_file\" || true\n                        echo \"\"\n                    fi\n                    ((formatted_count++))\n                fi\n                rm \"$temp_file\"\n            else\n                log_error \"Failed to check: $file\"\n                rm \"$temp_file\"\n                ((error_count++))\n            fi\n        else\n            # Format file\n            if autopep8 \"${autopep8_flags[@]}\" \"$file\" >/dev/null 2>&1; then\n                ((formatted_count++))\n            else\n                log_error \"Failed to format: $file\"\n                ((error_count++))\n            fi\n        fi\n    done\n    \n    if [[ $error_count -eq 0 ]]; then\n        if [[ \"$CHECK_ONLY\" == \"true\" ]]; then\n            log_success \"Python format check completed: $formatted_count files need formatting\"\n        else\n            log_success \"Python formatting completed: $formatted_count files formatted\"\n        fi\n    else\n        log_warning \"Python formatting completed with $error_count errors\"\n    fi\n    \n    return $error_count\n}\n\n# Format Rust files\nformat_rust() {\n    log_info \"Formatting Rust files...\"\n    \n    if ! command_exists rustfmt; then\n        log_warning \"rustfmt not found, skipping Rust formatting\"\n        return 0\n    fi\n    \n    # Find Rust project directories\n    local rust_dirs=()\n    \n    if [[ -f \"$PROJECT_ROOT/internal/sdk/rust/Cargo.toml\" ]]; then\n        rust_dirs+=(\"$PROJECT_ROOT/internal/sdk/rust\")\n    fi\n    \n    if [[ ${#rust_dirs[@]} -eq 0 ]]; then\n        log_info \"No Rust projects found\"\n        return 0\n    fi\n    \n    local error_count=0\n    \n    for dir in \"${rust_dirs[@]}\"; do\n        log_info \"Formatting Rust in: $(basename \"$dir\")\"\n        \n        cd \"$dir\"\n        \n        # Build rustfmt flags\n        local rustfmt_flags=()\n        \n        if [[ -n \"$CONFIG_FILE\" ]]; then\n            rustfmt_flags+=(\"--config-path=$CONFIG_FILE\")\n        elif [[ -f \"$RUSTFMT_CONFIG\" ]]; then\n            rustfmt_flags+=(\"--config-path=$RUSTFMT_CONFIG\")\n        fi\n        \n        if [[ \"$CHECK_ONLY\" == \"true\" ]]; then\n            rustfmt_flags+=(\"--check\")\n        fi\n        \n        if [[ \"$VERBOSE\" == \"true\" ]]; then\n            rustfmt_flags+=(\"--verbose\")\n        fi\n        \n        # Format using cargo fmt\n        if cargo fmt \"${rustfmt_flags[@]}\" 2>&1 | tee \"$LOG_DIR/rustfmt-$(basename \"$dir\").log\"; then\n            log_success \"Rust formatting completed for $(basename \"$dir\")\"\n        else\n            log_error \"Rust formatting failed for $(basename \"$dir\")\"\n            ((error_count++))\n        fi\n    done\n    \n    return $error_count\n}\n\n# Format YAML files\nformat_yaml() {\n    log_info \"Formatting YAML files...\"\n    \n    # Check for yamlfmt or prettier\n    local yaml_formatter=\"\"\n    if command_exists yamlfmt; then\n        yaml_formatter=\"yamlfmt\"\n    elif command_exists prettier; then\n        yaml_formatter=\"prettier\"\n    elif command_exists npx; then\n        yaml_formatter=\"npx prettier\"\n    else\n        log_warning \"No YAML formatter found, skipping YAML formatting\"\n        return 0\n    fi\n    \n    # Find YAML files\n    local yaml_files\n    mapfile -t yaml_files < <(find_files \"*.yml|*.yaml\")\n    \n    if [[ ${#yaml_files[@]} -eq 0 ]]; then\n        log_info \"No YAML files found\"\n        return 0\n    fi\n    \n    log_info \"Found ${#yaml_files[@]} YAML files to format with $yaml_formatter\"\n    \n    local error_count=0\n    \n    case \"$yaml_formatter\" in\n        \"yamlfmt\")\n            format_yaml_with_yamlfmt \"${yaml_files[@]}\"\n            error_count=$?\n            ;;\n        \"prettier\"|\"npx prettier\")\n            format_yaml_with_prettier \"$yaml_formatter\" \"${yaml_files[@]}\"\n            error_count=$?\n            ;;\n    esac\n    \n    return $error_count\n}\n\n# Format YAML with yamlfmt\nformat_yaml_with_yamlfmt() {\n    local files=(\"$@\")\n    local yamlfmt_flags=()\n    \n    if [[ \"$VERBOSE\" == \"true\" ]]; then\n        yamlfmt_flags+=(\"-v\")\n    fi\n    \n    if [[ \"$SHOW_DIFF\" == \"true\" ]]; then\n        yamlfmt_flags+=(\"-d\")\n    fi\n    \n    local error_count=0\n    \n    for file in \"${files[@]}\"; do\n        # Create backup if requested\n        create_file_backup \"$file\"\n        \n        if yamlfmt \"${yamlfmt_flags[@]}\" \"$file\"; then\n            if [[ \"$VERBOSE\" == \"true\" ]]; then\n                log_info \"Formatted: $file\"\n            fi\n        else\n            log_error \"Failed to format: $file\"\n            ((error_count++))\n        fi\n    done\n    \n    return $error_count\n}\n\n# Format YAML with prettier\nformat_yaml_with_prettier() {\n    local prettier_cmd=\"$1\"\n    shift\n    local files=(\"$@\")\n    \n    local prettier_flags=(\"--parser=yaml\")\n    \n    if [[ \"$CHECK_ONLY\" == \"true\" ]]; then\n        prettier_flags+=(\"--check\")\n    elif [[ \"$WRITE_CHANGES\" == \"true\" ]]; then\n        prettier_flags+=(\"--write\")\n    fi\n    \n    local error_count=0\n    \n    for file in \"${files[@]}\"; do\n        # Create backup if requested\n        create_file_backup \"$file\"\n        \n        if $prettier_cmd \"${prettier_flags[@]}\" \"$file\" >/dev/null 2>&1; then\n            if [[ \"$VERBOSE\" == \"true\" ]]; then\n                log_info \"Formatted: $file\"\n            fi\n        else\n            log_error \"Failed to format: $file\"\n            ((error_count++))\n        fi\n    done\n    \n    return $error_count\n}\n\n# Format JSON files\nformat_json() {\n    log_info \"Formatting JSON files...\"\n    \n    # Find JSON files\n    local json_files\n    mapfile -t json_files < <(find_files \"*.json\")\n    \n    if [[ ${#json_files[@]} -eq 0 ]]; then\n        log_info \"No JSON files found\"\n        return 0\n    fi\n    \n    log_info \"Found ${#json_files[@]} JSON files to format\"\n    \n    local error_count=0\n    \n    for file in \"${json_files[@]}\"; do\n        if [[ \"$VERBOSE\" == \"true\" ]]; then\n            log_info \"Formatting: $file\"\n        fi\n        \n        # Create backup if requested\n        create_file_backup \"$file\"\n        \n        # Format JSON using jq or python\n        if command_exists jq; then\n            format_json_with_jq \"$file\"\n        elif command_exists python3; then\n            format_json_with_python \"$file\"\n        else\n            log_warning \"No JSON formatter available (jq, python3)\"\n            continue\n        fi\n        \n        local exit_code=$?\n        if [[ $exit_code -ne 0 ]]; then\n            ((error_count++))\n        fi\n    done\n    \n    if [[ $error_count -eq 0 ]]; then\n        log_success \"JSON formatting completed\"\n    else\n        log_warning \"JSON formatting completed with $error_count errors\"\n    fi\n    \n    return $error_count\n}\n\n# Format JSON with jq\nformat_json_with_jq() {\n    local file=\"$1\"\n    local temp_file\n    temp_file=$(mktemp)\n    \n    if jq '.' \"$file\" > \"$temp_file\" 2>/dev/null; then\n        if [[ \"$CHECK_ONLY\" == \"true\" ]]; then\n            if ! diff -q \"$file\" \"$temp_file\" >/dev/null 2>&1; then\n                if [[ \"$SHOW_DIFF\" == \"true\" ]]; then\n                    echo \"=== $file ===\"\n                    diff -u \"$file\" \"$temp_file\" || true\n                    echo \"\"\n                fi\n            fi\n        elif [[ \"$WRITE_CHANGES\" == \"true\" ]]; then\n            mv \"$temp_file\" \"$file\"\n        fi\n        \n        rm -f \"$temp_file\"\n        return 0\n    else\n        log_error \"Invalid JSON in: $file\"\n        rm -f \"$temp_file\"\n        return 1\n    fi\n}\n\n# Format JSON with Python\nformat_json_with_python() {\n    local file=\"$1\"\n    local temp_file\n    temp_file=$(mktemp)\n    \n    if python3 -m json.tool \"$file\" > \"$temp_file\" 2>/dev/null; then\n        if [[ \"$CHECK_ONLY\" == \"true\" ]]; then\n            if ! diff -q \"$file\" \"$temp_file\" >/dev/null 2>&1; then\n                if [[ \"$SHOW_DIFF\" == \"true\" ]]; then\n                    echo \"=== $file ===\"\n                    diff -u \"$file\" \"$temp_file\" || true\n                    echo \"\"\n                fi\n            fi\n        elif [[ \"$WRITE_CHANGES\" == \"true\" ]]; then\n            mv \"$temp_file\" \"$file\"\n        fi\n        \n        rm -f \"$temp_file\"\n        return 0\n    else\n        log_error \"Invalid JSON in: $file\"\n        rm -f \"$temp_file\"\n        return 1\n    fi\n}\n\n# Format Markdown files\nformat_markdown() {\n    log_info \"Formatting Markdown files...\"\n    \n    # Check for prettier\n    local prettier_cmd=\"prettier\"\n    if ! command_exists prettier && command_exists npx; then\n        prettier_cmd=\"npx prettier\"\n    elif ! command_exists prettier; then\n        log_warning \"prettier not found, skipping Markdown formatting\"\n        return 0\n    fi\n    \n    # Find Markdown files\n    local md_files\n    mapfile -t md_files < <(find_files \"*.md\")\n    \n    if [[ ${#md_files[@]} -eq 0 ]]; then\n        log_info \"No Markdown files found\"\n        return 0\n    fi\n    \n    log_info \"Found ${#md_files[@]} Markdown files to format\"\n    \n    local prettier_flags=(\"--parser=markdown\")\n    \n    if [[ \"$CHECK_ONLY\" == \"true\" ]]; then\n        prettier_flags+=(\"--check\")\n    elif [[ \"$WRITE_CHANGES\" == \"true\" ]]; then\n        prettier_flags+=(\"--write\")\n    fi\n    \n    if [[ -n \"$LINE_LENGTH\" ]]; then\n        prettier_flags+=(\"--print-width=$LINE_LENGTH\")\n    fi\n    \n    local error_count=0\n    \n    for file in \"${md_files[@]}\"; do\n        # Create backup if requested\n        create_file_backup \"$file\"\n        \n        if $prettier_cmd \"${prettier_flags[@]}\" \"$file\" >/dev/null 2>&1; then\n            if [[ \"$VERBOSE\" == \"true\" ]]; then\n                log_info \"Formatted: $file\"\n            fi\n        else\n            log_error \"Failed to format: $file\"\n            ((error_count++))\n        fi\n    done\n    \n    if [[ $error_count -eq 0 ]]; then\n        log_success \"Markdown formatting completed\"\n    else\n        log_warning \"Markdown formatting completed with $error_count errors\"\n    fi\n    \n    return $error_count\n}\n\n# Run formatters in parallel\nrun_parallel_formatting() {\n    log_info \"Running formatters in parallel...\"\n    \n    local pids=()\n    local results=()\n    \n    case \"$LANGUAGE\" in\n        \"go\")\n            format_go &\n            pids+=($!)\n            results+=(\"go\")\n            ;;\n        \"javascript\")\n            format_javascript &\n            pids+=($!)\n            results+=(\"javascript\")\n            ;;\n        \"python\")\n            format_python &\n            pids+=($!)\n            results+=(\"python\")\n            ;;\n        \"rust\")\n            format_rust &\n            pids+=($!)\n            results+=(\"rust\")\n            ;;\n        \"yaml\")\n            format_yaml &\n            pids+=($!)\n            results+=(\"yaml\")\n            ;;\n        \"json\")\n            format_json &\n            pids+=($!)\n            results+=(\"json\")\n            ;;\n        \"markdown\")\n            format_markdown &\n            pids+=($!)\n            results+=(\"markdown\")\n            ;;\n        \"all\")\n            format_go &\n            pids+=($!)\n            results+=(\"go\")\n            \n            format_javascript &\n            pids+=($!)\n            results+=(\"javascript\")\n            \n            format_python &\n            pids+=($!)\n            results+=(\"python\")\n            \n            format_rust &\n            pids+=($!)\n            results+=(\"rust\")\n            \n            format_yaml &\n            pids+=($!)\n            results+=(\"yaml\")\n            \n            format_json &\n            pids+=($!)\n            results+=(\"json\")\n            \n            format_markdown &\n            pids+=($!)\n            results+=(\"markdown\")\n            ;;\n    esac\n    \n    # Wait for all processes and collect results\n    local overall_error_count=0\n    \n    for i in \"${!pids[@]}\"; do\n        local pid=${pids[$i]}\n        local lang=${results[$i]}\n        \n        if wait \"$pid\"; then\n            log_success \"$lang formatting completed successfully\"\n        else\n            local exit_code=$?\n            log_error \"$lang formatting failed (exit code: $exit_code)\"\n            overall_error_count=$((overall_error_count + exit_code))\n        fi\n    done\n    \n    return $overall_error_count\n}\n\n# Run formatters sequentially\nrun_sequential_formatting() {\n    local overall_error_count=0\n    \n    case \"$LANGUAGE\" in\n        \"go\")\n            format_go\n            overall_error_count=$((overall_error_count + $?))\n            ;;\n        \"javascript\")\n            format_javascript\n            overall_error_count=$((overall_error_count + $?))\n            ;;\n        \"python\")\n            format_python\n            overall_error_count=$((overall_error_count + $?))\n            ;;\n        \"rust\")\n            format_rust\n            overall_error_count=$((overall_error_count + $?))\n            ;;\n        \"yaml\")\n            format_yaml\n            overall_error_count=$((overall_error_count + $?))\n            ;;\n        \"json\")\n            format_json\n            overall_error_count=$((overall_error_count + $?))\n            ;;\n        \"markdown\")\n            format_markdown\n            overall_error_count=$((overall_error_count + $?))\n            ;;\n        \"all\")\n            format_go\n            overall_error_count=$((overall_error_count + $?))\n            \n            format_javascript\n            overall_error_count=$((overall_error_count + $?))\n            \n            format_python\n            overall_error_count=$((overall_error_count + $?))\n            \n            format_rust\n            overall_error_count=$((overall_error_count + $?))\n            \n            format_yaml\n            overall_error_count=$((overall_error_count + $?))\n            \n            format_json\n            overall_error_count=$((overall_error_count + $?))\n            \n            format_markdown\n            overall_error_count=$((overall_error_count + $?))\n            ;;\n    esac\n    \n    return $overall_error_count\n}\n\n# Main execution\nmain() {\n    log_info \"Starting code formatting\"\n    log_info \"Language: $LANGUAGE\"\n    log_info \"Check only: $CHECK_ONLY\"\n    log_info \"Show diff: $SHOW_DIFF\"\n    log_info \"Write changes: $WRITE_CHANGES\"\n    log_info \"Parallel: $PARALLEL\"\n    \n    create_directories\n    check_required_tools\n    \n    # Change to project root\n    cd \"$PROJECT_ROOT\"\n    \n    # Run formatting\n    local error_count=0\n    \n    if [[ \"$PARALLEL\" == \"true\" ]]; then\n        run_parallel_formatting\n        error_count=$?\n    else\n        run_sequential_formatting\n        error_count=$?\n    fi\n    \n    if [[ $error_count -eq 0 ]]; then\n        if [[ \"$CHECK_ONLY\" == \"true\" ]]; then\n            log_success \"Code format check completed successfully\"\n        else\n            log_success \"Code formatting completed successfully\"\n        fi\n    else\n        if [[ \"$CHECK_ONLY\" == \"true\" ]]; then\n            log_warning \"Code format check completed with issues\"\n        else\n            log_warning \"Code formatting completed with $error_count errors\"\n        fi\n        exit $error_count\n    fi\n}\n\n# Execute main function\nmain \"$@\""}, {"replace_all": false}]