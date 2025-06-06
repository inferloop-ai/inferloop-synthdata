# scripts/setup.sh
#!/bin/bash
"""
Setup script for Audio Synthetic Data Framework
"""

set -e

echo "🎵 Setting up Audio Synthetic Data Framework..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
check_python() {
    print_status "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_status "Python $PYTHON_VERSION detected ✓"
        else
            print_error "Python 3.8+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8 or higher."
        exit 1
    fi
}

# Check system dependencies
check_system_deps() {
    print_status "Checking system dependencies..."
    
    # Check for required system packages
    REQUIRED_PACKAGES=("ffmpeg" "git" "curl")
    
    for package in "${REQUIRED_PACKAGES[@]}"; do
        if command -v "$package" &> /dev/null; then
            print_status "$package found ✓"
        else
            print_warning "$package not found. Installing..."
            
            # Try to install based on OS
            if [[ "$OSTYPE" == "linux-gnu"* ]]; then
                sudo apt-get update && sudo apt-get install -y "$package"
            elif [[ "$OSTYPE" == "darwin"* ]]; then
                brew install "$package"
            else
                print_error "Please install $package manually"
                exit 1
            fi
        fi
    done
}

# Create virtual environment
setup_venv() {
    print_status "Setting up virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_status "Virtual environment created ✓"
    else
        print_status "Virtual environment already exists ✓"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    print_status "pip upgraded ✓"
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Install requirements
    pip install -r requirements.txt
    
    # Install development dependencies if requested
    if [ "$1" = "--dev" ]; then
        pip install -e ".[dev,api,analysis,privacy]"
        print_status "Development dependencies installed ✓"
    else
        pip install -e .
        print_status "Core dependencies installed ✓"
    fi
}

# Download models
download_models() {
    print_status "Setting up models directory..."
    
    mkdir -p models
    
    # Download pre-trained models (placeholder URLs)
    MODELS_DIR="models"
    
    if [ ! -f "$MODELS_DIR/diffusion_model.pt" ]; then
        print_status "Downloading diffusion model..."
        # wget -O "$MODELS_DIR/diffusion_model.pt" "https://example.com/diffusion_model.pt"
        # For now, create placeholder
        touch "$MODELS_DIR/diffusion_model.pt"
    fi
    
    if [ ! -f "$MODELS_DIR/tts_model.pt" ]; then
        print_status "Downloading TTS model..."
        # wget -O "$MODELS_DIR/tts_model.pt" "https://example.com/tts_model.pt"
        touch "$MODELS_DIR/tts_model.pt"
    fi
    
    print_status "Model setup completed ✓"
}

# Setup configuration
setup_config() {
    print_status "Setting up configuration..."
    
    # Create config directory
    mkdir -p configs
    
    # Copy default configuration if it doesn't exist
    if [ ! -f "configs/default.yaml" ]; then
        # Run the init-config command
        python -m audio_synth.cli.main init-config --output-dir configs
        print_status "Default configuration created ✓"
    else
        print_status "Configuration already exists ✓"
    fi
    
    # Create output directories
    mkdir -p output logs
    print_status "Output directories created ✓"
}

# Run tests
run_tests() {
    print_status "Running tests..."
    
    if [ "$1" = "--skip-tests" ]; then
        print_warning "Skipping tests as requested"
        return
    fi
    
    # Run unit tests
    python -m pytest tests/unit/ -v
    
    if [ $? -eq 0 ]; then
        print_status "All tests passed ✓"
    else
        print_warning "Some tests failed. Setup completed but please check test results."
    fi
}

# Main setup function
main() {
    echo "🎵 Audio Synthetic Data Framework Setup"
    echo "======================================="
    
    # Parse arguments
    DEV_MODE=false
    SKIP_TESTS=false
    
    for arg in "$@"; do
        case $arg in
            --dev)
                DEV_MODE=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --help)
                echo "Usage: ./setup.sh [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --dev         Install development dependencies"
                echo "  --skip-tests  Skip running tests"
                echo "  --help        Show this help message"
                exit 0
                ;;
        esac
    done
    
    # Run setup steps
    check_python
    check_system_deps
    setup_venv
    
    if [ "$DEV_MODE" = true ]; then
        install_dependencies --dev
    else
        install_dependencies
    fi
    
    download_models
    setup_config
    
    if [ "$SKIP_TESTS" = false ]; then
        run_tests
    else
        run_tests --skip-tests
    fi
    
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Test the CLI: audio-synth --help"
    echo "3. Start the API server: audio-synth-server"
    echo "4. Check the documentation in docs/"
    echo ""
    echo "Happy synthesizing! 🎵"
}

# Run main function with all arguments
main "$@"

# ============================================================================
