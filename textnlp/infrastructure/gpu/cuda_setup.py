"""
CUDA and cuDNN Setup and Configuration for TextNLP
Manages CUDA/cuDNN dependencies across different environments
"""

import os
import subprocess
import platform
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class CUDAVersion(Enum):
    """Supported CUDA versions"""
    CUDA_11_2 = "11.2"
    CUDA_11_4 = "11.4"
    CUDA_11_6 = "11.6"
    CUDA_11_7 = "11.7"
    CUDA_11_8 = "11.8"
    CUDA_12_0 = "12.0"
    CUDA_12_1 = "12.1"


class CuDNNVersion(Enum):
    """Supported cuDNN versions"""
    CUDNN_8_2 = "8.2"
    CUDNN_8_4 = "8.4"
    CUDNN_8_6 = "8.6"
    CUDNN_8_7 = "8.7"
    CUDNN_8_8 = "8.8"
    CUDNN_8_9 = "8.9"


@dataclass
class CUDARequirements:
    """CUDA requirements for different models/frameworks"""
    framework: str
    min_cuda_version: CUDAVersion
    recommended_cuda_version: CUDAVersion
    min_cudnn_version: CuDNNVersion
    recommended_cudnn_version: CuDNNVersion
    compute_capabilities: List[str]  # e.g., ["7.0", "7.5", "8.0", "8.6"]


@dataclass
class CUDAEnvironment:
    """Current CUDA environment configuration"""
    cuda_version: Optional[str]
    cudnn_version: Optional[str]
    gpu_count: int
    gpu_names: List[str]
    compute_capabilities: List[str]
    driver_version: Optional[str]
    available_memory_mb: List[int]


class CUDASetupManager:
    """Manages CUDA/cuDNN setup and configuration"""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self.requirements = self._initialize_requirements()
        
    def _initialize_requirements(self) -> Dict[str, CUDARequirements]:
        """Initialize CUDA requirements for different frameworks"""
        return {
            "pytorch": CUDARequirements(
                framework="pytorch",
                min_cuda_version=CUDAVersion.CUDA_11_2,
                recommended_cuda_version=CUDAVersion.CUDA_11_8,
                min_cudnn_version=CuDNNVersion.CUDNN_8_2,
                recommended_cudnn_version=CuDNNVersion.CUDNN_8_7,
                compute_capabilities=["7.0", "7.5", "8.0", "8.6", "8.9"]
            ),
            "tensorflow": CUDARequirements(
                framework="tensorflow",
                min_cuda_version=CUDAVersion.CUDA_11_2,
                recommended_cuda_version=CUDAVersion.CUDA_11_8,
                min_cudnn_version=CuDNNVersion.CUDNN_8_2,
                recommended_cudnn_version=CuDNNVersion.CUDNN_8_6,
                compute_capabilities=["7.0", "7.5", "8.0", "8.6"]
            ),
            "jax": CUDARequirements(
                framework="jax",
                min_cuda_version=CUDAVersion.CUDA_11_4,
                recommended_cuda_version=CUDAVersion.CUDA_11_8,
                min_cudnn_version=CuDNNVersion.CUDNN_8_2,
                recommended_cudnn_version=CuDNNVersion.CUDNN_8_8,
                compute_capabilities=["7.0", "7.5", "8.0", "8.6", "8.9"]
            )
        }
    
    def detect_environment(self) -> CUDAEnvironment:
        """Detect current CUDA environment"""
        cuda_version = self._get_cuda_version()
        cudnn_version = self._get_cudnn_version()
        gpu_info = self._get_gpu_info()
        driver_version = self._get_driver_version()
        
        return CUDAEnvironment(
            cuda_version=cuda_version,
            cudnn_version=cudnn_version,
            gpu_count=gpu_info["count"],
            gpu_names=gpu_info["names"],
            compute_capabilities=gpu_info["compute_capabilities"],
            driver_version=driver_version,
            available_memory_mb=gpu_info["memory"]
        )
    
    def _get_cuda_version(self) -> Optional[str]:
        """Get installed CUDA version"""
        try:
            # Try nvcc first
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse version from output
            for line in result.stdout.split('\n'):
                if "release" in line:
                    parts = line.split("release")[-1].strip()
                    version = parts.split(",")[0].strip()
                    return version
                    
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Try reading from cuda directory
        cuda_paths = [
            "/usr/local/cuda/version.txt",
            "/usr/local/cuda/version.json",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v*\\version.txt"
        ]
        
        for path in cuda_paths:
            if Path(path).exists():
                try:
                    with open(path, 'r') as f:
                        content = f.read()
                        if "CUDA Version" in content:
                            return content.split("CUDA Version")[-1].strip()
                except Exception:
                    pass
        
        return None
    
    def _get_cudnn_version(self) -> Optional[str]:
        """Get installed cuDNN version"""
        # Check common cuDNN header locations
        cudnn_headers = [
            "/usr/include/cudnn_version.h",
            "/usr/local/cuda/include/cudnn_version.h",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v*\\include\\cudnn_version.h"
        ]
        
        for header_path in cudnn_headers:
            if Path(header_path).exists():
                try:
                    with open(header_path, 'r') as f:
                        content = f.read()
                        
                    # Parse version from header
                    major = minor = patch = None
                    for line in content.split('\n'):
                        if "#define CUDNN_MAJOR" in line:
                            major = line.split()[-1]
                        elif "#define CUDNN_MINOR" in line:
                            minor = line.split()[-1]
                        elif "#define CUDNN_PATCHLEVEL" in line:
                            patch = line.split()[-1]
                    
                    if major and minor:
                        return f"{major}.{minor}.{patch}" if patch else f"{major}.{minor}"
                        
                except Exception:
                    pass
        
        return None
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information using nvidia-smi"""
        gpu_info = {
            "count": 0,
            "names": [],
            "compute_capabilities": [],
            "memory": []
        }
        
        try:
            # Get GPU count and names
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True
            )
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(', ')
                    gpu_info["count"] += 1
                    gpu_info["names"].append(parts[0])
                    gpu_info["compute_capabilities"].append(parts[1])
                    # Convert memory from MiB to MB
                    memory_str = parts[2].replace(' MiB', '')
                    gpu_info["memory"].append(int(memory_str))
                    
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("nvidia-smi not found or failed")
            
        return gpu_info
    
    def _get_driver_version(self) -> Optional[str]:
        """Get NVIDIA driver version"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    
    def validate_environment(self, framework: str) -> Dict[str, Any]:
        """Validate CUDA environment for a specific framework"""
        if framework not in self.requirements:
            return {"valid": False, "error": f"Unknown framework: {framework}"}
        
        env = self.detect_environment()
        req = self.requirements[framework]
        
        issues = []
        warnings = []
        
        # Check CUDA version
        if not env.cuda_version:
            issues.append("CUDA not detected")
        else:
            cuda_version = env.cuda_version.split('.')[0:2]
            cuda_version_str = '.'.join(cuda_version)
            
            if cuda_version_str < req.min_cuda_version.value:
                issues.append(
                    f"CUDA {cuda_version_str} is below minimum required {req.min_cuda_version.value}"
                )
            elif cuda_version_str < req.recommended_cuda_version.value:
                warnings.append(
                    f"CUDA {cuda_version_str} is below recommended {req.recommended_cuda_version.value}"
                )
        
        # Check cuDNN version
        if not env.cudnn_version:
            issues.append("cuDNN not detected")
        else:
            cudnn_version = env.cudnn_version.split('.')[0:2]
            cudnn_version_str = '.'.join(cudnn_version)
            
            if cudnn_version_str < req.min_cudnn_version.value:
                issues.append(
                    f"cuDNN {cudnn_version_str} is below minimum required {req.min_cudnn_version.value}"
                )
            elif cudnn_version_str < req.recommended_cudnn_version.value:
                warnings.append(
                    f"cuDNN {cudnn_version_str} is below recommended {req.recommended_cudnn_version.value}"
                )
        
        # Check compute capabilities
        if env.compute_capabilities:
            supported_caps = set(env.compute_capabilities)
            required_caps = set(req.compute_capabilities)
            
            if not supported_caps.intersection(required_caps):
                issues.append(
                    f"No supported compute capabilities. Found: {supported_caps}, "
                    f"Required one of: {required_caps}"
                )
        
        # Check GPU availability
        if env.gpu_count == 0:
            issues.append("No GPUs detected")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "environment": env.__dict__,
            "requirements": {
                "min_cuda": req.min_cuda_version.value,
                "recommended_cuda": req.recommended_cuda_version.value,
                "min_cudnn": req.min_cudnn_version.value,
                "recommended_cudnn": req.recommended_cudnn_version.value,
                "compute_capabilities": req.compute_capabilities
            }
        }
    
    def generate_dockerfile_cuda_base(self, cuda_version: CUDAVersion, 
                                    cudnn_version: CuDNNVersion,
                                    base_image: str = "nvidia/cuda") -> str:
        """Generate Dockerfile base for CUDA setup"""
        cuda_ver = cuda_version.value
        cudnn_ver = cudnn_version.value
        
        dockerfile = f"""# CUDA base image for TextNLP
FROM {base_image}:{cuda_ver}-cudnn{cudnn_ver.split('.')[0]}-devel-ubuntu20.04

# Set environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${{CUDA_HOME}}/bin:${{PATH}}
ENV LD_LIBRARY_PATH=${{CUDA_HOME}}/lib64:${{LD_LIBRARY_PATH}}

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    cmake \\
    git \\
    wget \\
    python3-dev \\
    python3-pip \\
    && rm -rf /var/lib/apt/lists/*

# Verify CUDA installation
RUN nvcc --version && \\
    python3 -c "import subprocess; print(subprocess.check_output(['nvidia-smi']).decode())"

# Install Python ML frameworks with CUDA support
RUN pip3 install --upgrade pip && \\
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu{cuda_ver.replace('.', '')} && \\
    pip3 install tensorflow-gpu jax[cuda{cuda_ver.replace('.', '')}] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install additional GPU utilities
RUN pip3 install \\
    nvidia-ml-py3 \\
    gpustat \\
    py3nvml \\
    cupy-cuda{cuda_ver.replace('.', '')}

# Set up workspace
WORKDIR /workspace

# Health check for GPU
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \\
    CMD nvidia-smi || exit 1
"""
        return dockerfile
    
    def generate_setup_script(self, platform_type: str = "linux") -> str:
        """Generate setup script for CUDA/cuDNN installation"""
        if platform_type == "linux":
            return self._generate_linux_setup_script()
        elif platform_type == "windows":
            return self._generate_windows_setup_script()
        else:
            raise ValueError(f"Unsupported platform: {platform_type}")
    
    def _generate_linux_setup_script(self) -> str:
        """Generate Linux CUDA setup script"""
        script = """#!/bin/bash
# TextNLP CUDA/cuDNN Setup Script for Linux

set -e

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
NC='\\033[0m' # No Color

echo -e "${GREEN}TextNLP CUDA/cuDNN Setup Script${NC}"
echo "=================================="

# Function to check if running with sudo
check_sudo() {
    if [ "$EUID" -ne 0 ]; then 
        echo -e "${RED}Please run with sudo${NC}"
        exit 1
    fi
}

# Function to detect OS
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$NAME
        VER=$VERSION_ID
    else
        echo -e "${RED}Cannot detect OS${NC}"
        exit 1
    fi
    echo -e "${GREEN}Detected OS: $OS $VER${NC}"
}

# Function to install CUDA
install_cuda() {
    local cuda_version=$1
    echo -e "${YELLOW}Installing CUDA ${cuda_version}...${NC}"
    
    # Add NVIDIA package repository
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
    dpkg -i cuda-keyring_1.0-1_all.deb
    apt-get update
    
    # Install CUDA
    apt-get install -y cuda-${cuda_version}
    
    # Set up environment variables
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    
    echo -e "${GREEN}CUDA ${cuda_version} installed successfully${NC}"
}

# Function to install cuDNN
install_cudnn() {
    local cudnn_version=$1
    echo -e "${YELLOW}Installing cuDNN ${cudnn_version}...${NC}"
    
    # Download cuDNN (requires NVIDIA developer account)
    echo -e "${YELLOW}Please download cuDNN ${cudnn_version} from https://developer.nvidia.com/cudnn${NC}"
    echo "Place the downloaded file in /tmp/cudnn.tar.gz and press Enter"
    read -p "Press Enter when ready..."
    
    if [ -f /tmp/cudnn.tar.gz ]; then
        tar -xzvf /tmp/cudnn.tar.gz -C /usr/local/
        echo -e "${GREEN}cuDNN ${cudnn_version} installed successfully${NC}"
    else
        echo -e "${RED}cuDNN tar file not found at /tmp/cudnn.tar.gz${NC}"
        exit 1
    fi
}

# Function to verify installation
verify_installation() {
    echo -e "${YELLOW}Verifying installation...${NC}"
    
    # Check CUDA
    if command -v nvcc &> /dev/null; then
        cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        echo -e "${GREEN}CUDA ${cuda_version} detected${NC}"
    else
        echo -e "${RED}CUDA not detected${NC}"
    fi
    
    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        echo -e "${GREEN}${gpu_count} GPU(s) detected${NC}"
        nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
    else
        echo -e "${RED}nvidia-smi not found${NC}"
    fi
}

# Main installation flow
main() {
    check_sudo
    detect_os
    
    # Default versions
    CUDA_VERSION="11-8"
    CUDNN_VERSION="8.7"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --cuda)
                CUDA_VERSION="$2"
                shift 2
                ;;
            --cudnn)
                CUDNN_VERSION="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Install CUDA
    install_cuda $CUDA_VERSION
    
    # Install cuDNN
    install_cudnn $CUDNN_VERSION
    
    # Verify installation
    verify_installation
    
    echo -e "${GREEN}Setup complete! Please restart your shell or run 'source ~/.bashrc'${NC}"
}

# Run main function
main "$@"
"""
        return script
    
    def _generate_windows_setup_script(self) -> str:
        """Generate Windows CUDA setup script"""
        script = """@echo off
REM TextNLP CUDA/cuDNN Setup Script for Windows

echo TextNLP CUDA/cuDNN Setup Script
echo ==================================

REM Check for admin privileges
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Administrator privileges required
    echo Please run this script as Administrator
    pause
    exit /b 1
)

REM Default versions
set CUDA_VERSION=11.8
set CUDNN_VERSION=8.7

REM Parse arguments
:parse_args
if "%~1"=="" goto :main
if "%~1"=="--cuda" (
    set CUDA_VERSION=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--cudnn" (
    set CUDNN_VERSION=%~2
    shift
    shift
    goto :parse_args
)
shift
goto :parse_args

:main
echo Installing CUDA %CUDA_VERSION% and cuDNN %CUDNN_VERSION%

REM Download CUDA installer
echo Downloading CUDA installer...
set CUDA_URL=https://developer.download.nvidia.com/compute/cuda/%CUDA_VERSION%/Prod/local_installers/cuda_%CUDA_VERSION%_windows.exe
powershell -Command "Invoke-WebRequest -Uri '%CUDA_URL%' -OutFile 'cuda_installer.exe'"

REM Install CUDA silently
echo Installing CUDA...
cuda_installer.exe -s

REM Set environment variables
setx CUDA_PATH "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v%CUDA_VERSION%" /M
setx PATH "%PATH%;%CUDA_PATH%\\bin;%CUDA_PATH%\\libnvvp" /M

REM Download cuDNN
echo.
echo Please download cuDNN %CUDNN_VERSION% from https://developer.nvidia.com/cudnn
echo Extract the contents to %CUDA_PATH%
pause

REM Verify installation
echo.
echo Verifying installation...
nvcc --version
nvidia-smi

echo.
echo Setup complete! Please restart your command prompt.
pause
"""
        return script
    
    def generate_environment_yaml(self, framework: str, 
                                cuda_version: CUDAVersion,
                                env_name: str = "textnlp") -> str:
        """Generate conda environment.yaml file"""
        cuda_ver = cuda_version.value.replace('.', '')
        
        yaml_content = f"""name: {env_name}
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pip
  - numpy
  - scipy
  - scikit-learn
  - pandas
  - matplotlib
  - jupyter
  - ipython
"""
        
        if framework == "pytorch":
            yaml_content += f"""  - pytorch::pytorch>=2.0.0
  - pytorch::torchvision
  - pytorch::torchaudio
  - pytorch::pytorch-cuda={cuda_version.value}
"""
        elif framework == "tensorflow":
            yaml_content += f"""  - tensorflow-gpu>=2.12.0
  - cudatoolkit={cuda_version.value}
  - cudnn>=8.2
"""
        
        yaml_content += """  - pip:
    - transformers>=4.30.0
    - accelerate>=0.20.0
    - datasets>=2.12.0
    - tokenizers>=0.13.0
    - sentencepiece
    - bitsandbytes
    - nvidia-ml-py3
    - gpustat
    - wandb
    - peft
    - trl
"""
        
        return yaml_content
    
    def check_framework_compatibility(self, framework: str) -> Dict[str, Any]:
        """Check if current environment is compatible with framework"""
        env = self.detect_environment()
        validation = self.validate_environment(framework)
        
        # Additional framework-specific checks
        compatibility = {
            "framework": framework,
            "compatible": validation["valid"],
            "cuda_environment": validation["environment"],
            "issues": validation["issues"],
            "warnings": validation["warnings"],
            "recommendations": []
        }
        
        # Add recommendations
        if not validation["valid"]:
            if not env.cuda_version:
                compatibility["recommendations"].append(
                    f"Install CUDA {self.requirements[framework].recommended_cuda_version.value}"
                )
            if not env.cudnn_version:
                compatibility["recommendations"].append(
                    f"Install cuDNN {self.requirements[framework].recommended_cudnn_version.value}"
                )
        
        # Check for specific GPU models
        if env.gpu_names:
            for gpu_name in env.gpu_names:
                if "T4" in gpu_name:
                    compatibility["recommendations"].append(
                        "T4 GPU detected - recommended for inference workloads"
                    )
                elif "V100" in gpu_name:
                    compatibility["recommendations"].append(
                        "V100 GPU detected - good for training and inference"
                    )
                elif "A100" in gpu_name:
                    compatibility["recommendations"].append(
                        "A100 GPU detected - optimal for large model training"
                    )
        
        return compatibility


class CUDADockerBuilder:
    """Helper class to build Docker images with CUDA support"""
    
    def __init__(self, cuda_manager: CUDASetupManager):
        self.cuda_manager = cuda_manager
    
    def build_textnlp_image(self, 
                          cuda_version: CUDAVersion,
                          cudnn_version: CuDNNVersion,
                          tag: str = "textnlp:latest") -> str:
        """Build complete TextNLP Docker image with CUDA"""
        dockerfile = f"""{self.cuda_manager.generate_dockerfile_cuda_base(cuda_version, cudnn_version)}

# Install TextNLP specific dependencies
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

# Copy TextNLP code
COPY . /workspace/textnlp/

# Install TextNLP
WORKDIR /workspace/textnlp
RUN pip3 install -e .

# Set up model cache directory
ENV TRANSFORMERS_CACHE=/models
RUN mkdir -p /models

# Expose API port
EXPOSE 8000

# Default command
CMD ["python3", "-m", "textnlp.api.app"]
"""
        
        # Save Dockerfile
        dockerfile_path = Path("Dockerfile.cuda")
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile)
        
        # Build command
        build_cmd = f"docker build -f {dockerfile_path} -t {tag} ."
        
        return build_cmd
    
    def generate_docker_compose(self, services: List[str]) -> str:
        """Generate docker-compose.yml for GPU services"""
        compose = """version: '3.8'

services:
"""
        
        for service in services:
            if service == "inference":
                compose += """  inference:
    image: textnlp:latest-cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - CUDA_VISIBLE_DEVICES=0
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models
      - ./data:/data
    command: ["python3", "-m", "textnlp.api.app", "--gpu"]
    
"""
            elif service == "training":
                compose += """  training:
    image: textnlp:latest-cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./models:/models
      - ./data:/data
      - ./checkpoints:/checkpoints
    command: ["python3", "-m", "textnlp.train", "--distributed"]
    
"""
        
        compose += """volumes:
  models:
  data:
  checkpoints:
"""
        
        return compose


# Utility functions
def setup_cuda_environment(framework: str = "pytorch", 
                         cuda_version: str = "11.8",
                         cudnn_version: str = "8.7") -> Dict[str, Any]:
    """Quick setup function for CUDA environment"""
    manager = CUDASetupManager()
    
    # Validate current environment
    validation = manager.validate_environment(framework)
    
    if not validation["valid"]:
        logger.warning(f"Environment validation failed: {validation['issues']}")
        
        # Generate setup script
        if platform.system().lower() == "linux":
            script = manager.generate_setup_script("linux")
            script_path = Path("setup_cuda.sh")
            with open(script_path, 'w') as f:
                f.write(script)
            os.chmod(script_path, 0o755)
            logger.info(f"Setup script generated: {script_path}")
        
        # Generate conda environment
        env_yaml = manager.generate_environment_yaml(
            framework,
            CUDAVersion(f"CUDA_{cuda_version.replace('.', '_')}"),
            "textnlp"
        )
        
        with open("environment.yaml", 'w') as f:
            f.write(env_yaml)
        logger.info("Conda environment file generated: environment.yaml")
    
    return validation


# Example usage
if __name__ == "__main__":
    # Create CUDA setup manager
    manager = CUDASetupManager()
    
    # Detect current environment
    env = manager.detect_environment()
    print(f"Current CUDA Environment:")
    print(json.dumps(env.__dict__, indent=2))
    
    # Validate for PyTorch
    validation = manager.validate_environment("pytorch")
    print(f"\nPyTorch Compatibility:")
    print(json.dumps(validation, indent=2))
    
    # Generate Dockerfile
    dockerfile = manager.generate_dockerfile_cuda_base(
        CUDAVersion.CUDA_11_8,
        CuDNNVersion.CUDNN_8_7
    )
    print(f"\nGenerated Dockerfile preview:")
    print(dockerfile[:500] + "...")
    
    # Generate setup script
    setup_script = manager.generate_setup_script("linux")
    print(f"\nGenerated setup script preview:")
    print(setup_script[:500] + "...")