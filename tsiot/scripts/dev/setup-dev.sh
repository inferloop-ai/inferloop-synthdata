#!/bin/bash

# TSIoT Development Environment Setup Script
# Sets up the development environment for the Time Series IoT Synthetic Data Generation Platform

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

# Required tools and versions
REQUIRED_GO_VERSION="1.21"
REQUIRED_NODE_VERSION="18"
REQUIRED_PYTHON_VERSION="3.9"

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
    --full                  Full setup including all dependencies
    --minimal               Minimal setup (Go tools only)
    --skip-docker           Skip Docker setup
    --skip-database         Skip database setup
    --skip-nodejs           Skip Node.js setup
    --skip-python           Skip Python setup
    --skip-tools            Skip development tools installation
    --skip-hooks            Skip Git hooks setup
    --reinstall             Reinstall existing tools
    --update                Update existing installations
    --check-only            Only check current setup without installing
    -v, --verbose           Enable verbose output
    -h, --help              Show this help message

Examples:
    $0                      # Full development setup
    $0 --minimal            # Minimal Go development setup
    $0 --check-only         # Check current development setup
    $0 --update             # Update existing development tools

Environment Variables:
    TSIOT_DEV_MODE          Development mode (local|docker|hybrid)
    GO_VERSION              Preferred Go version
    NODE_VERSION            Preferred Node.js version
    PYTHON_VERSION          Preferred Python version
EOF
}

# Parse command line arguments
FULL_SETUP=true
MINIMAL_SETUP=false
SKIP_DOCKER=false
SKIP_DATABASE=false
SKIP_NODEJS=false
SKIP_PYTHON=false
SKIP_TOOLS=false
SKIP_HOOKS=false
REINSTALL=false
UPDATE=false
CHECK_ONLY=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            FULL_SETUP=true
            MINIMAL_SETUP=false
            shift
            ;;
        --minimal)
            MINIMAL_SETUP=true
            FULL_SETUP=false
            shift
            ;;
        --skip-docker)
            SKIP_DOCKER=true
            shift
            ;;
        --skip-database)
            SKIP_DATABASE=true
            shift
            ;;
        --skip-nodejs)
            SKIP_NODEJS=true
            shift
            ;;
        --skip-python)
            SKIP_PYTHON=true
            shift
            ;;
        --skip-tools)
            SKIP_TOOLS=true
            shift
            ;;
        --skip-hooks)
            SKIP_HOOKS=true
            shift
            ;;
        --reinstall)
            REINSTALL=true
            shift
            ;;
        --update)
            UPDATE=true
            shift
            ;;
        --check-only)
            CHECK_ONLY=true
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

# Set environment variables
export TSIOT_DEV_MODE="${TSIOT_DEV_MODE:-local}"
export GO_VERSION="${GO_VERSION:-$REQUIRED_GO_VERSION}"
export NODE_VERSION="${NODE_VERSION:-$REQUIRED_NODE_VERSION}"
export PYTHON_VERSION="${PYTHON_VERSION:-$REQUIRED_PYTHON_VERSION}"

# Create necessary directories
create_directories() {
    mkdir -p "$LOG_DIR"
    mkdir -p "$PROJECT_ROOT/.dev"
    mkdir -p "$PROJECT_ROOT/.dev/bin"
    mkdir -p "$PROJECT_ROOT/.dev/cache"
}

# Detect operating system
detect_os() {
    case "$(uname -s)" in
        Darwin*)
            OS="macos"
            ;;
        Linux*)
            OS="linux"
            ;;
        CYGWIN*|MINGW*|MSYS*)
            OS="windows"
            ;;
        *)
            log_error "Unsupported operating system: $(uname -s)"
            exit 1
            ;;
    esac
    
    log_info "Detected OS: $OS"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Compare versions
version_ge() {
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

# Check Go installation
check_go() {
    log_info "Checking Go installation..."
    
    if command_exists go; then
        local go_version
        go_version=$(go version | grep -oE 'go[0-9]+\.[0-9]+' | sed 's/go//')
        
        if version_ge "$go_version" "$REQUIRED_GO_VERSION"; then
            log_success "Go $go_version is installed (required: $REQUIRED_GO_VERSION+)"
            return 0
        else
            log_warning "Go $go_version is installed but version $REQUIRED_GO_VERSION+ is required"
            return 1
        fi
    else
        log_warning "Go is not installed"
        return 1
    fi
}

# Install Go
install_go() {
    if [[ "$CHECK_ONLY" == "true" ]]; then
        return 0
    fi
    
    log_info "Installing Go $GO_VERSION..."
    
    case "$OS" in
        "macos")
            if command_exists brew; then
                brew install go
            else
                log_error "Homebrew not found. Please install Go manually from https://golang.org/dl/"
                exit 1
            fi
            ;;
        "linux")
            local go_archive="go${GO_VERSION}.linux-amd64.tar.gz"
            local go_url="https://dl.google.com/go/$go_archive"
            
            wget -O "/tmp/$go_archive" "$go_url"
            sudo rm -rf /usr/local/go
            sudo tar -C /usr/local -xzf "/tmp/$go_archive"
            
            # Add Go to PATH if not already there
            if ! echo "$PATH" | grep -q "/usr/local/go/bin"; then
                echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
                export PATH=$PATH:/usr/local/go/bin
            fi
            ;;
        "windows")
            log_error "Please install Go manually from https://golang.org/dl/"
            exit 1
            ;;
    esac
    
    log_success "Go installation completed"
}

# Setup Go environment
setup_go_environment() {
    log_info "Setting up Go environment..."
    
    # Set up GOPATH and GOROOT if needed
    if [[ -z "${GOPATH:-}" ]]; then
        export GOPATH="$HOME/go"
        echo "export GOPATH=$HOME/go" >> ~/.bashrc
    fi
    
    # Add GOPATH/bin to PATH
    if ! echo "$PATH" | grep -q "$GOPATH/bin"; then
        export PATH="$PATH:$GOPATH/bin"
        echo 'export PATH=$PATH:$GOPATH/bin' >> ~/.bashrc
    fi
    
    # Enable Go modules
    export GO111MODULE=on
    
    log_success "Go environment setup completed"
}

# Install Go development tools
install_go_tools() {
    if [[ "$SKIP_TOOLS" == "true" ]] || [[ "$CHECK_ONLY" == "true" ]]; then
        return 0
    fi
    
    log_info "Installing Go development tools..."
    
    local tools=(
        "github.com/golangci/golangci-lint/cmd/golangci-lint@latest"
        "golang.org/x/tools/cmd/goimports@latest"
        "golang.org/x/tools/cmd/godoc@latest"
        "github.com/go-delve/delve/cmd/dlv@latest"
        "github.com/swaggo/swag/cmd/swag@latest"
        "github.com/cosmtrek/air@latest"
        "github.com/golang/mock/mockgen@latest"
        "gotest.tools/gotestsum@latest"
        "github.com/mikefarah/yq/v4@latest"
        "github.com/google/wire/cmd/wire@latest"
    )\n    \n    for tool in \"${tools[@]}\"; do\n        local tool_name\n        tool_name=$(basename \"${tool%@*}\")\n        \n        if [[ \"$REINSTALL\" == \"true\" ]] || ! command_exists \"$tool_name\"; then\n            log_info \"Installing $tool_name...\"\n            go install \"$tool\"\n        else\n            log_info \"$tool_name is already installed\"\n        fi\n    done\n    \n    log_success \"Go tools installation completed\"\n}\n\n# Check Node.js installation\ncheck_nodejs() {\n    if [[ \"$SKIP_NODEJS\" == \"true\" ]] || [[ \"$MINIMAL_SETUP\" == \"true\" ]]; then\n        return 0\n    fi\n    \n    log_info \"Checking Node.js installation...\"\n    \n    if command_exists node; then\n        local node_version\n        node_version=$(node --version | sed 's/v//')\n        \n        if version_ge \"$node_version\" \"$REQUIRED_NODE_VERSION\"; then\n            log_success \"Node.js $node_version is installed (required: $REQUIRED_NODE_VERSION+)\"\n            return 0\n        else\n            log_warning \"Node.js $node_version is installed but version $REQUIRED_NODE_VERSION+ is required\"\n            return 1\n        fi\n    else\n        log_warning \"Node.js is not installed\"\n        return 1\n    fi\n}\n\n# Install Node.js\ninstall_nodejs() {\n    if [[ \"$SKIP_NODEJS\" == \"true\" ]] || [[ \"$MINIMAL_SETUP\" == \"true\" ]] || [[ \"$CHECK_ONLY\" == \"true\" ]]; then\n        return 0\n    fi\n    \n    log_info \"Installing Node.js $NODE_VERSION...\"\n    \n    case \"$OS\" in\n        \"macos\")\n            if command_exists brew; then\n                brew install node\n            else\n                log_error \"Homebrew not found. Please install Node.js manually\"\n                exit 1\n            fi\n            ;;\n        \"linux\")\n            # Install using NodeSource repository\n            curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | sudo -E bash -\n            sudo apt-get install -y nodejs\n            ;;\n        \"windows\")\n            log_error \"Please install Node.js manually from https://nodejs.org/\"\n            exit 1\n            ;;\n    esac\n    \n    log_success \"Node.js installation completed\"\n}\n\n# Check Python installation\ncheck_python() {\n    if [[ \"$SKIP_PYTHON\" == \"true\" ]] || [[ \"$MINIMAL_SETUP\" == \"true\" ]]; then\n        return 0\n    fi\n    \n    log_info \"Checking Python installation...\"\n    \n    for cmd in python3 python; do\n        if command_exists \"$cmd\"; then\n            local python_version\n            python_version=$(\"$cmd\" --version 2>&1 | grep -oE '[0-9]+\\.[0-9]+' | head -1)\n            \n            if version_ge \"$python_version\" \"$REQUIRED_PYTHON_VERSION\"; then\n                log_success \"Python $python_version is installed (required: $REQUIRED_PYTHON_VERSION+)\"\n                export PYTHON_CMD=\"$cmd\"\n                return 0\n            fi\n        fi\n    done\n    \n    log_warning \"Python $REQUIRED_PYTHON_VERSION+ is not installed\"\n    return 1\n}\n\n# Install Python\ninstall_python() {\n    if [[ \"$SKIP_PYTHON\" == \"true\" ]] || [[ \"$MINIMAL_SETUP\" == \"true\" ]] || [[ \"$CHECK_ONLY\" == \"true\" ]]; then\n        return 0\n    fi\n    \n    log_info \"Installing Python $PYTHON_VERSION...\"\n    \n    case \"$OS\" in\n        \"macos\")\n            if command_exists brew; then\n                brew install python@3.9\n            else\n                log_error \"Homebrew not found. Please install Python manually\"\n                exit 1\n            fi\n            ;;\n        \"linux\")\n            sudo apt-get update\n            sudo apt-get install -y python3 python3-pip python3-venv\n            ;;\n        \"windows\")\n            log_error \"Please install Python manually from https://python.org/\"\n            exit 1\n            ;;\n    esac\n    \n    log_success \"Python installation completed\"\n}\n\n# Setup Python environment\nsetup_python_environment() {\n    if [[ \"$SKIP_PYTHON\" == \"true\" ]] || [[ \"$MINIMAL_SETUP\" == \"true\" ]] || [[ \"$CHECK_ONLY\" == \"true\" ]]; then\n        return 0\n    fi\n    \n    log_info \"Setting up Python development environment...\"\n    \n    local python_cmd=\"${PYTHON_CMD:-python3}\"\n    \n    # Create virtual environment for development\n    if [[ ! -d \"$PROJECT_ROOT/.dev/venv\" ]]; then\n        \"$python_cmd\" -m venv \"$PROJECT_ROOT/.dev/venv\"\n    fi\n    \n    # Activate virtual environment and install packages\n    source \"$PROJECT_ROOT/.dev/venv/bin/activate\"\n    \n    pip install --upgrade pip\n    pip install -r \"$PROJECT_ROOT/requirements-dev.txt\" 2>/dev/null || {\n        # Install common development packages if requirements file doesn't exist\n        pip install pytest black flake8 mypy jupyter pandas numpy matplotlib\n    }\n    \n    log_success \"Python environment setup completed\"\n}\n\n# Check Docker installation\ncheck_docker() {\n    if [[ \"$SKIP_DOCKER\" == \"true\" ]] || [[ \"$MINIMAL_SETUP\" == \"true\" ]]; then\n        return 0\n    fi\n    \n    log_info \"Checking Docker installation...\"\n    \n    if command_exists docker; then\n        if docker info >/dev/null 2>&1; then\n            log_success \"Docker is installed and running\"\n            \n            if command_exists docker-compose; then\n                log_success \"Docker Compose is available\"\n            else\n                log_warning \"Docker Compose is not installed\"\n                return 1\n            fi\n            return 0\n        else\n            log_warning \"Docker is installed but not running\"\n            return 1\n        fi\n    else\n        log_warning \"Docker is not installed\"\n        return 1\n    fi\n}\n\n# Install Docker\ninstall_docker() {\n    if [[ \"$SKIP_DOCKER\" == \"true\" ]] || [[ \"$MINIMAL_SETUP\" == \"true\" ]] || [[ \"$CHECK_ONLY\" == \"true\" ]]; then\n        return 0\n    fi\n    \n    log_info \"Installing Docker...\"\n    \n    case \"$OS\" in\n        \"macos\")\n            if command_exists brew; then\n                brew install --cask docker\n                log_info \"Please start Docker Desktop manually\"\n            else\n                log_error \"Please install Docker Desktop manually from https://docker.com/\"\n                exit 1\n            fi\n            ;;\n        \"linux\")\n            # Install Docker using official script\n            curl -fsSL https://get.docker.com -o get-docker.sh\n            sudo sh get-docker.sh\n            \n            # Add user to docker group\n            sudo usermod -aG docker \"$USER\"\n            \n            # Install Docker Compose\n            sudo curl -L \"https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)\" -o /usr/local/bin/docker-compose\n            sudo chmod +x /usr/local/bin/docker-compose\n            \n            log_info \"Please log out and back in for Docker group changes to take effect\"\n            ;;\n        \"windows\")\n            log_error \"Please install Docker Desktop manually from https://docker.com/\"\n            exit 1\n            ;;\n    esac\n    \n    log_success \"Docker installation completed\"\n}\n\n# Setup development databases\nsetup_databases() {\n    if [[ \"$SKIP_DATABASE\" == \"true\" ]] || [[ \"$MINIMAL_SETUP\" == \"true\" ]] || [[ \"$CHECK_ONLY\" == \"true\" ]]; then\n        return 0\n    fi\n    \n    log_info \"Setting up development databases...\"\n    \n    local compose_file=\"$PROJECT_ROOT/deployments/docker/docker-compose.dev.yml\"\n    \n    if [[ -f \"$compose_file\" ]]; then\n        cd \"$PROJECT_ROOT/deployments/docker\"\n        docker-compose -f docker-compose.dev.yml up -d postgres redis influxdb kafka\n        \n        log_info \"Waiting for databases to be ready...\"\n        sleep 10\n        \n        log_success \"Development databases are running\"\n    else\n        log_warning \"Development Docker Compose file not found: $compose_file\"\n    fi\n}\n\n# Setup Git hooks\nsetup_git_hooks() {\n    if [[ \"$SKIP_HOOKS\" == \"true\" ]] || [[ \"$CHECK_ONLY\" == \"true\" ]]; then\n        return 0\n    fi\n    \n    log_info \"Setting up Git hooks...\"\n    \n    local hooks_dir=\"$PROJECT_ROOT/.git/hooks\"\n    \n    # Pre-commit hook\n    cat > \"$hooks_dir/pre-commit\" << 'EOF'\n#!/bin/bash\nset -e\n\n# Run linting\necho \"Running linters...\"\nscripts/dev/lint.sh\n\n# Run tests\necho \"Running tests...\"\nscripts/test/test.sh --type unit --timeout 5m\n\necho \"Pre-commit checks passed\"\nEOF\n    \n    # Pre-push hook\n    cat > \"$hooks_dir/pre-push\" << 'EOF'\n#!/bin/bash\nset -e\n\n# Run full test suite\necho \"Running full test suite...\"\nscripts/test/test.sh --timeout 10m\n\necho \"Pre-push checks passed\"\nEOF\n    \n    # Make hooks executable\n    chmod +x \"$hooks_dir/pre-commit\"\n    chmod +x \"$hooks_dir/pre-push\"\n    \n    log_success \"Git hooks setup completed\"\n}\n\n# Create development configuration\ncreate_dev_config() {\n    if [[ \"$CHECK_ONLY\" == \"true\" ]]; then\n        return 0\n    fi\n    \n    log_info \"Creating development configuration...\"\n    \n    # Create .env.development file\n    cat > \"$PROJECT_ROOT/.env.development\" << EOF\n# TSIoT Development Environment Configuration\nTSIOT_ENV=development\nTSIOT_LOG_LEVEL=debug\nTSIOT_DEBUG=true\n\n# Database Configuration\nDATABASE_URL=postgres://tsiot:password@localhost:5432/tsiot_dev\nREDIS_URL=redis://localhost:6379/0\nINFLUXDB_URL=http://localhost:8086\nINFLUXDB_DATABASE=tsiot_dev\n\n# Kafka Configuration\nKAFKA_BROKERS=localhost:9092\n\n# API Configuration\nAPI_PORT=8080\nAPI_HOST=localhost\n\n# Development Tools\nHOT_RELOAD=true\nPROFILING_ENABLED=true\nEOF\n    \n    # Create VS Code settings\n    mkdir -p \"$PROJECT_ROOT/.vscode\"\n    \n    cat > \"$PROJECT_ROOT/.vscode/settings.json\" << 'EOF'\n{\n    \"go.toolsManagement.checkForUpdates\": \"local\",\n    \"go.useLanguageServer\": true,\n    \"go.gopath\": \"\",\n    \"go.goroot\": \"\",\n    \"go.lintOnSave\": \"package\",\n    \"go.formatTool\": \"goimports\",\n    \"go.lintTool\": \"golangci-lint\",\n    \"go.testFlags\": [\"-v\"],\n    \"go.testTimeout\": \"30s\",\n    \"go.coverOnSave\": true,\n    \"go.coverageDecorator\": \"gutter\",\n    \"files.eol\": \"\\n\",\n    \"files.trimTrailingWhitespace\": true,\n    \"files.insertFinalNewline\": true,\n    \"editor.formatOnSave\": true,\n    \"editor.codeActionsOnSave\": {\n        \"source.organizeImports\": true\n    }\n}\nEOF\n    \n    cat > \"$PROJECT_ROOT/.vscode/launch.json\" << 'EOF'\n{\n    \"version\": \"0.2.0\",\n    \"configurations\": [\n        {\n            \"name\": \"Launch TSIoT Server\",\n            \"type\": \"go\",\n            \"request\": \"launch\",\n            \"mode\": \"auto\",\n            \"program\": \"${workspaceFolder}/cmd/server\",\n            \"env\": {\n                \"TSIOT_ENV\": \"development\"\n            },\n            \"args\": []\n        },\n        {\n            \"name\": \"Launch TSIoT CLI\",\n            \"type\": \"go\",\n            \"request\": \"launch\",\n            \"mode\": \"auto\",\n            \"program\": \"${workspaceFolder}/cmd/cli\",\n            \"env\": {\n                \"TSIOT_ENV\": \"development\"\n            },\n            \"args\": [\"--help\"]\n        }\n    ]\n}\nEOF\n    \n    log_success \"Development configuration created\"\n}\n\n# Generate development documentation\ngenerate_dev_docs() {\n    if [[ \"$CHECK_ONLY\" == \"true\" ]]; then\n        return 0\n    fi\n    \n    log_info \"Generating development documentation...\"\n    \n    cat > \"$PROJECT_ROOT/DEV_SETUP.md\" << 'EOF'\n# TSIoT Development Setup\n\nThis document contains information about the development environment setup.\n\n## Quick Start\n\n1. Run the development setup script:\n   ```bash\n   ./scripts/dev/setup-dev.sh\n   ```\n\n2. Start the development environment:\n   ```bash\n   docker-compose -f deployments/docker/docker-compose.dev.yml up -d\n   ```\n\n3. Run the application:\n   ```bash\n   go run cmd/server/main.go\n   ```\n\n## Development Tools\n\n- **Go**: Primary development language\n- **golangci-lint**: Code linting\n- **air**: Hot reload for development\n- **delve**: Debugging\n- **swag**: API documentation generation\n\n## Database Access\n\n- **PostgreSQL**: `localhost:5432` (tsiot/password)\n- **Redis**: `localhost:6379`\n- **InfluxDB**: `localhost:8086`\n- **Kafka**: `localhost:9092`\n\n## Useful Commands\n\n- `make dev`: Start development server with hot reload\n- `make test`: Run all tests\n- `make lint`: Run linters\n- `make build`: Build the application\n- `make docs`: Generate API documentation\n\n## VS Code Setup\n\nThe development setup includes VS Code configuration for:\n- Go language support\n- Debugging configuration\n- Code formatting and linting\n- Test integration\n\n## Environment Variables\n\nDevelopment environment variables are configured in `.env.development`.\nEOF\n    \n    log_success \"Development documentation generated\"\n}\n\n# Check current setup\ncheck_current_setup() {\n    log_info \"Checking current development setup...\"\n    \n    echo \"=== System Information ===\"\n    echo \"OS: $OS\"\n    echo \"Architecture: $(uname -m)\"\n    echo \"\"\n    \n    echo \"=== Go Environment ===\"\n    if check_go; then\n        echo \"Go version: $(go version)\"\n        echo \"GOPATH: ${GOPATH:-not set}\"\n        echo \"GO111MODULE: ${GO111MODULE:-not set}\"\n    else\n        echo \"Go: Not properly installed\"\n    fi\n    echo \"\"\n    \n    if [[ \"$MINIMAL_SETUP\" == \"false\" ]]; then\n        echo \"=== Node.js Environment ===\"\n        if check_nodejs; then\n            echo \"Node.js version: $(node --version)\"\n            echo \"npm version: $(npm --version)\"\n        else\n            echo \"Node.js: Not properly installed\"\n        fi\n        echo \"\"\n        \n        echo \"=== Python Environment ===\"\n        if check_python; then\n            echo \"Python version: $(${PYTHON_CMD:-python3} --version)\"\n            if [[ -f \"$PROJECT_ROOT/.dev/venv/bin/activate\" ]]; then\n                echo \"Virtual environment: Available\"\n            else\n                echo \"Virtual environment: Not created\"\n            fi\n        else\n            echo \"Python: Not properly installed\"\n        fi\n        echo \"\"\n        \n        echo \"=== Docker Environment ===\"\n        if check_docker; then\n            echo \"Docker version: $(docker --version)\"\n            echo \"Docker Compose version: $(docker-compose --version)\"\n        else\n            echo \"Docker: Not properly installed or running\"\n        fi\n        echo \"\"\n    fi\n    \n    echo \"=== Development Tools ===\"\n    local tools=(\"golangci-lint\" \"goimports\" \"air\" \"dlv\" \"swag\")\n    for tool in \"${tools[@]}\"; do\n        if command_exists \"$tool\"; then\n            echo \"$tool: Available\"\n        else\n            echo \"$tool: Not installed\"\n        fi\n    done\n    echo \"\"\n    \n    echo \"=== Project Files ===\"\n    if [[ -f \"$PROJECT_ROOT/.env.development\" ]]; then\n        echo \"Development config: Available\"\n    else\n        echo \"Development config: Not created\"\n    fi\n    \n    if [[ -d \"$PROJECT_ROOT/.vscode\" ]]; then\n        echo \"VS Code config: Available\"\n    else\n        echo \"VS Code config: Not created\"\n    fi\n    \n    if [[ -f \"$PROJECT_ROOT/.git/hooks/pre-commit\" ]]; then\n        echo \"Git hooks: Installed\"\n    else\n        echo \"Git hooks: Not installed\"\n    fi\n}\n\n# Main execution\nmain() {\n    log_info \"Starting TSIoT development environment setup\"\n    log_info \"Setup mode: $([ \"$MINIMAL_SETUP\" == \"true\" ] && echo \"minimal\" || echo \"full\")\"\n    \n    create_directories\n    detect_os\n    \n    if [[ \"$CHECK_ONLY\" == \"true\" ]]; then\n        check_current_setup\n        exit 0\n    fi\n    \n    # Install Go and tools\n    if ! check_go; then\n        install_go\n    fi\n    \n    setup_go_environment\n    install_go_tools\n    \n    # Install additional tools for full setup\n    if [[ \"$FULL_SETUP\" == \"true\" ]]; then\n        if ! check_nodejs; then\n            install_nodejs\n        fi\n        \n        if ! check_python; then\n            install_python\n        fi\n        setup_python_environment\n        \n        if ! check_docker; then\n            install_docker\n        fi\n        \n        setup_databases\n    fi\n    \n    # Setup development environment\n    setup_git_hooks\n    create_dev_config\n    generate_dev_docs\n    \n    log_success \"Development environment setup completed!\"\n    \n    echo \"\"\n    log_info \"Next steps:\"\n    echo \"  1. Restart your terminal or run: source ~/.bashrc\"\n    echo \"  2. Start development databases: docker-compose -f deployments/docker/docker-compose.dev.yml up -d\"\n    echo \"  3. Install project dependencies: ./scripts/dev/install-deps.sh\"\n    echo \"  4. Run the application: go run cmd/server/main.go\"\n    echo \"\"\n    echo \"For more information, see DEV_SETUP.md\"\n}\n\n# Execute main function\nmain \"$@\""}, {"replace_all": false}]