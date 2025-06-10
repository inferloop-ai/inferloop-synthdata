#!/bin/bash

# Development Environment Setup Script

set -e

echo "🚀 Setting up Enterprise Video Synthesis Pipeline development environment..."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
echo "✅ Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is required but not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is required but not installed. Please install Docker Compose first."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    print_warning "Python 3 is recommended for local development."
    print_info "You can still use the Docker-based development environment."
else
    print_success "Python 3 found: $(python3 --version)"
fi

print_success "Prerequisites check passed!"

# Setup Python virtual environment (optional)
if command -v python3 &> /dev/null; then
    echo "🐍 Setting up Python virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Created Python virtual environment"
    else
        print_info "Virtual environment already exists"
    fi
    
    # Activate and install dependencies
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    print_success "Installed Python dependencies"
else
    print_warning "Skipping Python setup (Python not found)"
fi

# Copy environment configuration
echo "⚙️ Setting up environment configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    print_success "Created .env file from template"
    print_warning "Please update .env file with your specific configuration"
else
    print_info ".env file already exists"
fi

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p {data/raw,data/processed,data/generated,logs,tmp}
print_success "Created data directories"

# Initialize Docker images
echo "🐳 Pulling Docker images..."
docker-compose pull
print_success "Docker images pulled"

# Setup git hooks (if in git repo)
if [ -d .git ]; then
    echo "🔧 Setting up git hooks..."
    if [ -f scripts/setup/pre-commit-hook.sh ]; then
        cp scripts/setup/pre-commit-hook.sh .git/hooks/pre-commit
        chmod +x .git/hooks/pre-commit
        print_success "Git hooks installed"
    fi
fi

# Create sample data (optional)
echo "📊 Setting up sample data..."
mkdir -p data/samples
# Add sample files here if needed

print_success "Development environment setup complete!"
echo ""
echo "🚀 Next steps:"
echo "  1. Update .env file with your configuration"
echo "  2. Run 'make deploy' to start the local stack"
echo "  3. Visit http://localhost:8080 for the API documentation"
echo "  4. Check service health with 'make status'"
echo ""
echo "📚 Available commands:"
echo "  make help     - Show all available commands"
echo "  make build    - Build all services"
echo "  make start    - Start all services"
echo "  make stop     - Stop all services"
echo "  make logs     - View service logs"
echo "  make status   - Check service status"
