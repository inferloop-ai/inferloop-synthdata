#!/bin/bash

# TSIoT Dependencies Installation Script
# Installs and manages dependencies for the Time Series IoT Synthetic Data Generation Platform

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
    --type TYPE             Dependency type (go|node|python|all) [default: all]
    --environment ENV       Environment (development|production) [default: development]
    --update                Update existing dependencies
    --clean                 Clean dependency cache before installation
    --verify                Verify dependencies after installation
    --offline               Install from cache only (no network)
    --force                 Force reinstallation of all dependencies
    --skip-optional         Skip optional dependencies
    --parallel              Install dependencies in parallel where possible
    -v, --verbose           Enable verbose output
    -h, --help              Show this help message

Dependency Types:
    go                      Go modules and tools
    node                    Node.js packages (for web dashboard)
    python                  Python packages (for SDK and notebooks)
    rust                    Rust dependencies (for Rust SDK)
    all                     All dependency types

Examples:
    $0                      # Install all dependencies
    $0 --type go            # Install only Go dependencies
    $0 --update --verify    # Update and verify all dependencies
    $0 --clean --force      # Clean install all dependencies

Environment Variables:
    GO_VERSION              Go version to use
    NODE_VERSION            Node.js version to use
    PYTHON_VERSION          Python version to use
    GOPROXY                 Go module proxy
    NPM_REGISTRY            npm registry URL
EOF
}

# Parse command line arguments
DEPENDENCY_TYPE="all"
ENVIRONMENT="development"
UPDATE=false
CLEAN=false
VERIFY=false
OFFLINE=false
FORCE=false
SKIP_OPTIONAL=false
PARALLEL=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            DEPENDENCY_TYPE="$2"
            shift 2
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --update)
            UPDATE=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --verify)
            VERIFY=true
            shift
            ;;
        --offline)
            OFFLINE=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --skip-optional)
            SKIP_OPTIONAL=true
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

# Validate dependency type
if [[ ! "$DEPENDENCY_TYPE" =~ ^(go|node|python|rust|all)$ ]]; then
    log_error "Invalid dependency type: $DEPENDENCY_TYPE"
    usage
    exit 1
fi

# Set environment variables
export GO_VERSION="${GO_VERSION:-1.21}"
export NODE_VERSION="${NODE_VERSION:-18}"
export PYTHON_VERSION="${PYTHON_VERSION:-3.9}"
export GOPROXY="${GOPROXY:-https://proxy.golang.org,direct}"
export NPM_REGISTRY="${NPM_REGISTRY:-https://registry.npmjs.org/}"

# Create necessary directories
create_directories() {
    mkdir -p "$LOG_DIR"
    mkdir -p "$PROJECT_ROOT/.cache"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Clean dependency caches
clean_caches() {
    if [[ "$CLEAN" == "false" ]]; then
        return 0
    fi
    
    log_info "Cleaning dependency caches..."
    
    # Go module cache
    if [[ "$DEPENDENCY_TYPE" == "go" ]] || [[ "$DEPENDENCY_TYPE" == "all" ]]; then
        if command_exists go; then
            go clean -modcache
            log_info "Go module cache cleaned"
        fi
    fi
    
    # Node.js cache
    if [[ "$DEPENDENCY_TYPE" == "node" ]] || [[ "$DEPENDENCY_TYPE" == "all" ]]; then
        if command_exists npm; then
            npm cache clean --force
            log_info "npm cache cleaned"
        fi
        
        if command_exists yarn; then
            yarn cache clean
            log_info "Yarn cache cleaned"
        fi
    fi
    
    # Python cache
    if [[ "$DEPENDENCY_TYPE" == "python" ]] || [[ "$DEPENDENCY_TYPE" == "all" ]]; then
        if command_exists pip; then
            pip cache purge
            log_info "pip cache cleaned"
        fi
    fi
    
    # Project cache
    rm -rf "$PROJECT_ROOT/.cache"
    mkdir -p "$PROJECT_ROOT/.cache"
    
    log_success "Dependency caches cleaned"
}

# Install Go dependencies
install_go_dependencies() {
    log_info "Installing Go dependencies..."
    
    if ! command_exists go; then
        log_error "Go is not installed. Please run setup-dev.sh first."
        exit 1
    fi
    
    cd "$PROJECT_ROOT"
    
    # Clean go.sum if force reinstall
    if [[ "$FORCE" == "true" ]]; then
        rm -f go.sum
    fi
    
    # Download and verify modules
    local go_flags=()
    
    if [[ "$VERBOSE" == "true" ]]; then
        go_flags+=("-v")
    fi
    
    if [[ "$OFFLINE" == "true" ]]; then
        export GOPROXY="off"
    fi
    
    log_info "Downloading Go modules..."
    go mod download "${go_flags[@]}"
    
    if [[ "$UPDATE" == "true" ]]; then
        log_info "Updating Go modules..."
        go get -u ./...
        go mod tidy
    fi
    
    # Install development tools
    if [[ "$ENVIRONMENT" == "development" ]]; then
        install_go_dev_tools
    fi
    
    log_success "Go dependencies installed"
}

# Install Go development tools
install_go_dev_tools() {
    log_info "Installing Go development tools..."
    
    local tools=(
        "github.com/golangci/golangci-lint/cmd/golangci-lint@latest"
        "golang.org/x/tools/cmd/goimports@latest"
        "github.com/swaggo/swag/cmd/swag@latest"
        "github.com/cosmtrek/air@latest"
        "github.com/go-delve/delve/cmd/dlv@latest"
        "gotest.tools/gotestsum@latest"
        "github.com/golang/mock/mockgen@latest"
        "github.com/google/wire/cmd/wire@latest"
    )
    
    if [[ "$SKIP_OPTIONAL" == "false" ]]; then
        tools+=(
            "github.com/securecodewarrior/sast-scan@latest"
            "github.com/mikefarah/yq/v4@latest"
            "github.com/pressly/goose/v3/cmd/goose@latest"
        )
    fi
    
    local failed_tools=()
    
    for tool in "${tools[@]}"; do
        local tool_name
        tool_name=$(basename "${tool%@*}")\n        \n        if [[ \"$FORCE\" == \"true\" ]] || ! command_exists \"$tool_name\"; then\n            log_info \"Installing $tool_name...\"\n            \n            if [[ \"$PARALLEL\" == \"true\" ]]; then\n                go install \"$tool\" &\n            else\n                if ! go install \"$tool\"; then\n                    failed_tools+=(\"$tool_name\")\n                    log_warning \"Failed to install $tool_name\"\n                fi\n            fi\n        else\n            log_info \"$tool_name is already installed\"\n        fi\n    done\n    \n    # Wait for parallel installations\n    if [[ \"$PARALLEL\" == \"true\" ]]; then\n        wait\n        \n        # Check if tools are available after parallel installation\n        for tool in \"${tools[@]}\"; do\n            local tool_name\n            tool_name=$(basename \"${tool%@*}\")\n            \n            if ! command_exists \"$tool_name\"; then\n                failed_tools+=(\"$tool_name\")\n            fi\n        done\n    fi\n    \n    if [[ ${#failed_tools[@]} -gt 0 ]]; then\n        log_warning \"Failed to install some tools: ${failed_tools[*]}\"\n    else\n        log_success \"Go development tools installed\"\n    fi\n}\n\n# Install Node.js dependencies\ninstall_node_dependencies() {\n    log_info \"Installing Node.js dependencies...\"\n    \n    if ! command_exists node; then\n        log_warning \"Node.js is not installed. Skipping Node.js dependencies.\"\n        return 0\n    fi\n    \n    # Check for package.json files\n    local package_dirs=()\n    \n    if [[ -f \"$PROJECT_ROOT/web/dashboard/package.json\" ]]; then\n        package_dirs+=(\"$PROJECT_ROOT/web/dashboard\")\n    fi\n    \n    if [[ -f \"$PROJECT_ROOT/examples/web-dashboard/package.json\" ]]; then\n        package_dirs+=(\"$PROJECT_ROOT/examples/web-dashboard\")\n    fi\n    \n    if [[ ${#package_dirs[@]} -eq 0 ]]; then\n        log_info \"No Node.js projects found, skipping\"\n        return 0\n    fi\n    \n    # Determine package manager\n    local package_manager=\"npm\"\n    if command_exists yarn && [[ -f \"yarn.lock\" ]]; then\n        package_manager=\"yarn\"\n    fi\n    \n    log_info \"Using package manager: $package_manager\"\n    \n    for dir in \"${package_dirs[@]}\"; do\n        log_info \"Installing dependencies in: $(basename \"$dir\")\"\n        \n        cd \"$dir\"\n        \n        # Remove node_modules if force install\n        if [[ \"$FORCE\" == \"true\" ]]; then\n            rm -rf node_modules package-lock.json yarn.lock\n        fi\n        \n        # Install dependencies\n        case \"$package_manager\" in\n            \"npm\")\n                local npm_flags=()\n                \n                if [[ \"$ENVIRONMENT\" == \"development\" ]]; then\n                    npm_flags+=(\"--include=dev\")\n                fi\n                \n                if [[ \"$OFFLINE\" == \"true\" ]]; then\n                    npm_flags+=(\"--offline\")\n                fi\n                \n                if [[ \"$VERBOSE\" == \"true\" ]]; then\n                    npm_flags+=(\"--verbose\")\n                fi\n                \n                npm install \"${npm_flags[@]}\"\n                \n                if [[ \"$UPDATE\" == \"true\" ]]; then\n                    npm update\n                fi\n                ;;\n            \"yarn\")\n                local yarn_flags=()\n                \n                if [[ \"$ENVIRONMENT\" == \"production\" ]]; then\n                    yarn_flags+=(\"--production\")\n                fi\n                \n                if [[ \"$OFFLINE\" == \"true\" ]]; then\n                    yarn_flags+=(\"--offline\")\n                fi\n                \n                if [[ \"$VERBOSE\" == \"true\" ]]; then\n                    yarn_flags+=(\"--verbose\")\n                fi\n                \n                yarn install \"${yarn_flags[@]}\"\n                \n                if [[ \"$UPDATE\" == \"true\" ]]; then\n                    yarn upgrade\n                fi\n                ;;\n        esac\n    done\n    \n    log_success \"Node.js dependencies installed\"\n}\n\n# Install Python dependencies\ninstall_python_dependencies() {\n    log_info \"Installing Python dependencies...\"\n    \n    local python_cmd=\"python3\"\n    \n    if ! command_exists \"$python_cmd\"; then\n        if command_exists python; then\n            python_cmd=\"python\"\n        else\n            log_warning \"Python is not installed. Skipping Python dependencies.\"\n            return 0\n        fi\n    fi\n    \n    # Check for Python project directories\n    local python_dirs=()\n    \n    if [[ -f \"$PROJECT_ROOT/requirements.txt\" ]]; then\n        python_dirs+=(\"$PROJECT_ROOT\")\n    fi\n    \n    if [[ -f \"$PROJECT_ROOT/internal/sdk/python/requirements.txt\" ]]; then\n        python_dirs+=(\"$PROJECT_ROOT/internal/sdk/python\")\n    fi\n    \n    if [[ -f \"$PROJECT_ROOT/examples/python-sdk/requirements.txt\" ]]; then\n        python_dirs+=(\"$PROJECT_ROOT/examples/python-sdk\")\n    fi\n    \n    if [[ ${#python_dirs[@]} -eq 0 ]]; then\n        log_info \"No Python projects found, skipping\"\n        return 0\n    fi\n    \n    # Setup virtual environment for development\n    if [[ \"$ENVIRONMENT\" == \"development\" ]]; then\n        setup_python_venv\n    fi\n    \n    for dir in \"${python_dirs[@]}\"; do\n        log_info \"Installing Python dependencies in: $(basename \"$dir\")\"\n        \n        cd \"$dir\"\n        \n        # Activate virtual environment if available\n        if [[ -f \"$PROJECT_ROOT/.dev/venv/bin/activate\" ]]; then\n            source \"$PROJECT_ROOT/.dev/venv/bin/activate\"\n        fi\n        \n        # Install requirements\n        local pip_flags=()\n        \n        if [[ \"$VERBOSE\" == \"true\" ]]; then\n            pip_flags+=(\"--verbose\")\n        fi\n        \n        if [[ \"$OFFLINE\" == \"true\" ]]; then\n            pip_flags+=(\"--no-index\" \"--find-links\" \"$PROJECT_ROOT/.cache/pip\")\n        fi\n        \n        if [[ \"$FORCE\" == \"true\" ]]; then\n            pip_flags+=(\"--force-reinstall\")\n        fi\n        \n        if [[ \"$UPDATE\" == \"true\" ]]; then\n            pip_flags+=(\"--upgrade\")\n        fi\n        \n        # Install from requirements.txt\n        if [[ -f \"requirements.txt\" ]]; then\n            pip install \"${pip_flags[@]}\" -r requirements.txt\n        fi\n        \n        # Install development requirements\n        if [[ \"$ENVIRONMENT\" == \"development\" ]] && [[ -f \"requirements-dev.txt\" ]]; then\n            pip install \"${pip_flags[@]}\" -r requirements-dev.txt\n        fi\n        \n        # Install package in editable mode if setup.py exists\n        if [[ -f \"setup.py\" ]] || [[ -f \"pyproject.toml\" ]]; then\n            pip install \"${pip_flags[@]}\" -e .\n        fi\n    done\n    \n    log_success \"Python dependencies installed\"\n}\n\n# Setup Python virtual environment\nsetup_python_venv() {\n    local venv_dir=\"$PROJECT_ROOT/.dev/venv\"\n    \n    if [[ ! -d \"$venv_dir\" ]] || [[ \"$FORCE\" == \"true\" ]]; then\n        log_info \"Creating Python virtual environment...\"\n        \n        rm -rf \"$venv_dir\"\n        python3 -m venv \"$venv_dir\"\n        \n        # Activate and upgrade pip\n        source \"$venv_dir/bin/activate\"\n        pip install --upgrade pip setuptools wheel\n        \n        log_success \"Python virtual environment created\"\n    fi\n}\n\n# Install Rust dependencies\ninstall_rust_dependencies() {\n    log_info \"Installing Rust dependencies...\"\n    \n    if ! command_exists cargo; then\n        log_warning \"Rust is not installed. Skipping Rust dependencies.\"\n        return 0\n    fi\n    \n    # Check for Cargo.toml files\n    local rust_dirs=()\n    \n    if [[ -f \"$PROJECT_ROOT/internal/sdk/rust/Cargo.toml\" ]]; then\n        rust_dirs+=(\"$PROJECT_ROOT/internal/sdk/rust\")\n    fi\n    \n    if [[ ${#rust_dirs[@]} -eq 0 ]]; then\n        log_info \"No Rust projects found, skipping\"\n        return 0\n    fi\n    \n    for dir in \"${rust_dirs[@]}\"; do\n        log_info \"Installing Rust dependencies in: $(basename \"$dir\")\"\n        \n        cd \"$dir\"\n        \n        # Build dependencies\n        local cargo_flags=()\n        \n        if [[ \"$VERBOSE\" == \"true\" ]]; then\n            cargo_flags+=(\"--verbose\")\n        fi\n        \n        if [[ \"$OFFLINE\" == \"true\" ]]; then\n            cargo_flags+=(\"--offline\")\n        fi\n        \n        if [[ \"$UPDATE\" == \"true\" ]]; then\n            cargo update\n        fi\n        \n        cargo build \"${cargo_flags[@]}\"\n    done\n    \n    log_success \"Rust dependencies installed\"\n}\n\n# Verify dependencies\nverify_dependencies() {\n    if [[ \"$VERIFY\" == \"false\" ]]; then\n        return 0\n    fi\n    \n    log_info \"Verifying dependencies...\"\n    \n    local verification_failed=false\n    \n    # Verify Go dependencies\n    if [[ \"$DEPENDENCY_TYPE\" == \"go\" ]] || [[ \"$DEPENDENCY_TYPE\" == \"all\" ]]; then\n        cd \"$PROJECT_ROOT\"\n        \n        if ! go mod verify; then\n            log_error \"Go module verification failed\"\n            verification_failed=true\n        fi\n        \n        if ! go build ./...; then\n            log_error \"Go build verification failed\"\n            verification_failed=true\n        fi\n    fi\n    \n    # Verify Node.js dependencies\n    if [[ \"$DEPENDENCY_TYPE\" == \"node\" ]] || [[ \"$DEPENDENCY_TYPE\" == \"all\" ]]; then\n        if [[ -f \"$PROJECT_ROOT/web/dashboard/package.json\" ]]; then\n            cd \"$PROJECT_ROOT/web/dashboard\"\n            \n            if command_exists npm; then\n                if ! npm ls >/dev/null 2>&1; then\n                    log_warning \"Node.js dependency verification has warnings\"\n                fi\n            fi\n        fi\n    fi\n    \n    # Verify Python dependencies\n    if [[ \"$DEPENDENCY_TYPE\" == \"python\" ]] || [[ \"$DEPENDENCY_TYPE\" == \"all\" ]]; then\n        if [[ -f \"$PROJECT_ROOT/.dev/venv/bin/activate\" ]]; then\n            source \"$PROJECT_ROOT/.dev/venv/bin/activate\"\n            \n            if ! pip check >/dev/null 2>&1; then\n                log_warning \"Python dependency verification has conflicts\"\n            fi\n        fi\n    fi\n    \n    if [[ \"$verification_failed\" == \"true\" ]]; then\n        log_error \"Dependency verification failed\"\n        exit 1\n    else\n        log_success \"Dependency verification completed\"\n    fi\n}\n\n# Generate dependency report\ngenerate_dependency_report() {\n    log_info \"Generating dependency report...\"\n    \n    local report_file=\"$LOG_DIR/dependency-report-$(date +%Y%m%d-%H%M%S).json\"\n    \n    cat > \"$report_file\" << EOF\n{\n    \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\n    \"dependency_installation\": {\n        \"type\": \"$DEPENDENCY_TYPE\",\n        \"environment\": \"$ENVIRONMENT\",\n        \"options\": {\n            \"update\": $UPDATE,\n            \"clean\": $CLEAN,\n            \"verify\": $VERIFY,\n            \"offline\": $OFFLINE,\n            \"force\": $FORCE,\n            \"skip_optional\": $SKIP_OPTIONAL,\n            \"parallel\": $PARALLEL\n        }\n    },\n    \"go\": {\n        \"version\": \"$(command_exists go && go version | grep -oE 'go[0-9]+\\.[0-9]+\\.[0-9]+' || echo 'not installed')\",\n        \"modules\": \"$(cd \"$PROJECT_ROOT\" && go list -m all 2>/dev/null | wc -l || echo 0)\"\n    },\n    \"node\": {\n        \"version\": \"$(command_exists node && node --version || echo 'not installed')\",\n        \"packages\": \"$(find \"$PROJECT_ROOT\" -name node_modules -type d | wc -l)\"\n    },\n    \"python\": {\n        \"version\": \"$(command_exists python3 && python3 --version | grep -oE '[0-9]+\\.[0-9]+\\.[0-9]+' || echo 'not installed')\",\n        \"venv\": $([ -d \"$PROJECT_ROOT/.dev/venv\" ] && echo \"true\" || echo \"false\")\n    }\n}\nEOF\n    \n    log_success \"Dependency report generated: $report_file\"\n}\n\n# Main execution\nmain() {\n    log_info \"Starting dependency installation\"\n    log_info \"Type: $DEPENDENCY_TYPE\"\n    log_info \"Environment: $ENVIRONMENT\"\n    log_info \"Update: $UPDATE\"\n    log_info \"Clean: $CLEAN\"\n    log_info \"Verify: $VERIFY\"\n    \n    create_directories\n    \n    # Clean caches if requested\n    clean_caches\n    \n    # Install dependencies based on type\n    case \"$DEPENDENCY_TYPE\" in\n        \"go\")\n            install_go_dependencies\n            ;;\n        \"node\")\n            install_node_dependencies\n            ;;\n        \"python\")\n            install_python_dependencies\n            ;;\n        \"rust\")\n            install_rust_dependencies\n            ;;\n        \"all\")\n            install_go_dependencies\n            install_node_dependencies\n            install_python_dependencies\n            install_rust_dependencies\n            ;;\n    esac\n    \n    # Verify dependencies\n    verify_dependencies\n    \n    # Generate report\n    generate_dependency_report\n    \n    log_success \"Dependency installation completed successfully\"\n}\n\n# Execute main function\nmain \"$@\""}, {"replace_all": false}]