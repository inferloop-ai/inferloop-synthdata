# Installation Guide

This guide covers all installation methods for the Inferloop Synthetic Data SDK.

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 4GB RAM
- 2GB available disk space

### Recommended Requirements
- Python 3.9 or higher
- 8GB RAM
- 10GB available disk space
- GPU support for deep learning models (optional)

### Operating Systems
- Linux (Ubuntu 18.04+, CentOS 7+, RHEL 7+)
- macOS 10.15+ (Catalina or later)
- Windows 10/11

## Installation Methods

### 1. Quick Installation (Recommended)

```bash
# Install with all features
pip install inferloop-synthetic[all]
```

### 2. Basic Installation

```bash
# Core functionality only
pip install inferloop-synthetic
```

### 3. Custom Installation

Choose specific libraries based on your needs:

```bash
# SDV only
pip install inferloop-synthetic[sdv]

# CTGAN only
pip install inferloop-synthetic[ctgan]

# YData only
pip install inferloop-synthetic[ydata]

# Multiple libraries
pip install inferloop-synthetic[sdv,ctgan]

# Development tools
pip install inferloop-synthetic[dev]

# All features
pip install inferloop-synthetic[all]
```

### 4. Development Installation

For contributors and developers:

```bash
# Clone repository
git clone https://github.com/inferloop/inferloop-synthetic.git
cd inferloop-synthetic

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows

# Install in editable mode
pip install -e ".[dev,all]"
```

## Platform-Specific Instructions

### Ubuntu/Debian

```bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3.9 python3.9-pip python3.9-venv

# Install system dependencies
sudo apt install build-essential python3.9-dev

# Install the SDK
pip3.9 install inferloop-synthetic[all]
```

### CentOS/RHEL

```bash
# Install Python 3.9
sudo yum install python39 python39-pip

# Install development tools
sudo yum groupinstall "Development Tools"
sudo yum install python39-devel

# Install the SDK
pip3.9 install inferloop-synthetic[all]
```

### macOS

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.9

# Install the SDK
pip3.9 install inferloop-synthetic[all]
```

### Windows

1. Download Python 3.9+ from [python.org](https://www.python.org/downloads/)
2. Install Python with "Add Python to PATH" checked
3. Open Command Prompt as Administrator
4. Install the SDK:

```cmd
pip install inferloop-synthetic[all]
```

## Docker Installation

### Using Pre-built Images

```bash
# Pull the latest image
docker pull inferloop/synthetic-data:latest

# Run container
docker run -it -p 8000:8000 inferloop/synthetic-data:latest
```

### Build from Source

```bash
# Clone repository
git clone https://github.com/inferloop/inferloop-synthetic.git
cd inferloop-synthetic

# Build Docker image
docker build -t inferloop-synthetic .

# Run container
docker run -it -p 8000:8000 inferloop-synthetic
```

## GPU Support (Optional)

For accelerated deep learning models:

### NVIDIA GPU Support

```bash
# Install CUDA toolkit (version 11.8 recommended)
# Visit: https://developer.nvidia.com/cuda-toolkit

# Install cuDNN
# Visit: https://developer.nvidia.com/cudnn

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]

# Install the SDK
pip install inferloop-synthetic[all]
```

### AMD GPU Support (ROCm)

```bash
# Install ROCm (Linux only)
# Follow instructions at: https://docs.amd.com/

# Install PyTorch for ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2

# Install the SDK
pip install inferloop-synthetic[all]
```

## Verification

Verify your installation:

```bash
# Check installation
python -c "import inferloop_synthetic; print(inferloop_synthetic.__version__)"

# Run CLI help
inferloop-synthetic --help

# Test basic functionality
python -c "
from inferloop_synthetic.sdk import GeneratorFactory
print('Installation successful!')
"
```

## Virtual Environment Setup

### Using venv (Recommended)

```bash
# Create virtual environment
python -m venv inferloop-env

# Activate environment
source inferloop-env/bin/activate  # Linux/macOS
# OR
inferloop-env\Scripts\activate     # Windows

# Install SDK
pip install inferloop-synthetic[all]

# Deactivate when done
deactivate
```

### Using conda

```bash
# Create conda environment
conda create -n inferloop python=3.9

# Activate environment
conda activate inferloop

# Install SDK
pip install inferloop-synthetic[all]

# Deactivate when done
conda deactivate
```

## Troubleshooting

### Common Issues

#### Permission Denied (Linux/macOS)
```bash
# Use user installation
pip install --user inferloop-synthetic[all]
```

#### Dependency Conflicts
```bash
# Create fresh environment
python -m venv fresh-env
source fresh-env/bin/activate
pip install inferloop-synthetic[all]
```

#### Windows Long Path Issues
```bash
# Enable long paths in Windows
# Run as Administrator in PowerShell:
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

#### Memory Issues During Installation
```bash
# Increase pip cache and use no-cache option
pip install --no-cache-dir inferloop-synthetic[all]
```

### Dependency Issues

If you encounter dependency conflicts:

```bash
# Update pip
pip install --upgrade pip setuptools wheel

# Install with no dependencies first
pip install --no-deps inferloop-synthetic

# Install dependencies manually
pip install -r requirements.txt
```

### GPU Issues

```bash
# Check CUDA installation
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check TensorFlow GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Environment Variables

Optional environment variables for configuration:

```bash
# Set cache directory
export INFERLOOP_CACHE_DIR=/path/to/cache

# Set log level
export INFERLOOP_LOG_LEVEL=INFO

# Set GPU memory fraction
export INFERLOOP_GPU_MEMORY_FRACTION=0.8

# Set number of workers
export INFERLOOP_NUM_WORKERS=4
```

## Next Steps

After successful installation:

1. Read the [Quick Start Guide](quick-start.md)
2. Try the [First Dataset Tutorial](first-dataset.md)
3. Explore [SDK Usage Examples](sdk-usage.md)
4. Check out [Configuration Options](configuration.md)

## Getting Help

If you encounter issues:

1. Check our [Troubleshooting Guide](troubleshooting.md)
2. Search [GitHub Issues](https://github.com/inferloop/inferloop-synthetic/issues)
3. Create a new issue with:
   - Your operating system
   - Python version
   - Installation method used
   - Complete error message
   - Steps to reproduce