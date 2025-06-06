#!/bin/bash
# Setup script for the CodeAPI DSL project

set -e

echo "Setting up development environment..."

# Define paths
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$BASE_DIR")"
VENV_DIR="$ROOT_DIR/.venv"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install requirements
echo "Installing dependencies..."
pip install -r "$ROOT_DIR/requirements.txt"

# Setup development mode
echo "Setting up development mode..."
pip install -e "$ROOT_DIR"

# Create necessary directories if they don't exist
mkdir -p "$ROOT_DIR/models"
mkdir -p "$ROOT_DIR/output"

echo "Setup completed successfully!"
