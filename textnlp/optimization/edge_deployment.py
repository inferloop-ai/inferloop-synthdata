"""
Edge Deployment System for TextNLP
Optimized deployment configurations for edge devices with resource constraints
"""

import os
import json
import logging
import platform
import subprocess
import time
import psutil
import shutil
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import yaml
import tempfile

try:
    import tensorrt as trt
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False

try:
    import openvino as ov
    from openvino.tools import mo
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False

try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False

logger = logging.getLogger(__name__)


class EdgeDeviceType(Enum):
    """Types of edge devices"""
    RASPBERRY_PI = "raspberry_pi"
    JETSON_NANO = "jetson_nano"
    JETSON_XAVIER = "jetson_xavier"
    INTEL_NUC = "intel_nuc"
    ARM_CORTEX = "arm_cortex"
    MOBILE_CPU = "mobile_cpu"
    EDGE_TPU = "edge_tpu"
    GENERIC_ARM = "generic_arm"
    GENERIC_X86 = "generic_x86"


class DeploymentStrategy(Enum):
    """Deployment strategies for edge devices"""
    LIGHTWEIGHT = "lightweight"  # Minimal models, basic features
    BALANCED = "balanced"  # Balance between features and performance
    FEATURE_RICH = "feature_rich"  # Full features, higher resource usage
    CUSTOM = "custom"  # Custom configuration


@dataclass
class EdgeDeviceSpec:
    """Specifications for an edge device"""
    device_type: EdgeDeviceType
    cpu_cores: int
    memory_gb: float
    storage_gb: float
    has_gpu: bool = False
    gpu_memory_gb: float = 0.0
    has_npu: bool = False  # Neural Processing Unit
    has_tpu: bool = False  # Tensor Processing Unit
    architecture: str = "arm64"  # arm64, x86_64, armv7l
    os_type: str = "linux"  # linux, android, ios
    power_budget_watts: float = 10.0
    thermal_limit_celsius: float = 85.0
    network_bandwidth_mbps: float = 100.0


@dataclass
class EdgeDeploymentConfig:
    """Configuration for edge deployment"""
    device_spec: EdgeDeviceSpec
    strategy: DeploymentStrategy
    
    # Model optimization settings
    quantization_bits: int = 8  # 4, 8, 16
    pruning_ratio: float = 0.3
    enable_distillation: bool = True
    target_latency_ms: float = 500.0
    max_memory_usage_mb: float = 512.0
    
    # Runtime settings
    batch_size: int = 1
    max_sequence_length: int = 256
    enable_kv_cache: bool = True
    use_dynamic_batching: bool = False
    
    # Deployment options
    container_runtime: str = "docker"  # docker, podman, none
    enable_monitoring: bool = True
    enable_auto_scaling: bool = False
    enable_model_switching: bool = True
    
    # Network settings
    enable_model_caching: bool = True
    enable_result_caching: bool = True
    cache_size_mb: float = 128.0
    
    # Security settings
    enable_encryption: bool = True
    enable_secure_boot: bool = False
    
    # Maintenance settings
    enable_auto_update: bool = True
    update_check_interval_hours: int = 24
    log_retention_days: int = 7


@dataclass
class EdgeDeploymentResult:
    """Result of edge deployment"""
    success: bool
    deployment_time: float
    model_size_mb: float
    memory_usage_mb: float
    startup_time_ms: float
    inference_latency_ms: float
    throughput_tokens_per_second: float
    power_consumption_watts: float
    deployment_path: str
    config_used: Dict[str, Any]
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class EdgeDeviceProfiler:
    """Profile edge device capabilities"""
    
    def __init__(self):
        self.device_info = {}
        self._profile_device()
    
    def _profile_device(self):
        """Profile the current device"""
        
        # Basic system info
        self.device_info = {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "disk_usage": shutil.disk_usage("/"),
        }
        
        # CPU benchmarking
        self.device_info["cpu_benchmark"] = self._benchmark_cpu()
        
        # Memory benchmarking
        self.device_info["memory_benchmark"] = self._benchmark_memory()
        
        # GPU detection
        self._detect_gpu()
        
        # Special hardware detection
        self._detect_special_hardware()
        
        logger.info(f"Device profiling completed: {self.device_info['machine']} with {self.device_info['cpu_count']} cores")
    
    def _benchmark_cpu(self) -> Dict[str, float]:
        """Benchmark CPU performance"""
        
        # Simple matrix multiplication benchmark
        size = 1000
        start_time = time.time()
        
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)
        c = np.dot(a, b)
        
        cpu_time = time.time() - start_time
        
        # CPU frequency test
        try:
            cpu_freq = psutil.cpu_freq()
            max_freq = cpu_freq.max if cpu_freq else 0
        except:
            max_freq = 0
        
        return {
            "matrix_mult_time": cpu_time,
            "gflops_estimate": (2 * size**3) / (cpu_time * 1e9),
            "max_frequency_mhz": max_freq
        }
    
    def _benchmark_memory(self) -> Dict[str, float]:
        """Benchmark memory performance"""
        
        # Memory bandwidth test
        size = 10 * 1024 * 1024  # 10MB
        data = np.random.bytes(size)
        
        start_time = time.time()
        copied_data = bytearray(data)
        copy_time = time.time() - start_time
        
        bandwidth_mbps = (size / copy_time) / (1024 * 1024)
        
        return {
            "bandwidth_mbps": bandwidth_mbps,
            "copy_latency_ms": copy_time * 1000
        }
    
    def _detect_gpu(self):
        """Detect GPU capabilities"""
        
        self.device_info["gpu_available"] = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            self.device_info["gpu_count"] = torch.cuda.device_count()
            self.device_info["gpu_name"] = torch.cuda.get_device_name(0)
            self.device_info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            self.device_info["gpu_count"] = 0
            self.device_info["gpu_name"] = None
            self.device_info["gpu_memory_gb"] = 0
    
    def _detect_special_hardware(self):
        """Detect special hardware accelerators"""
        
        # Check for Intel OpenVINO devices
        self.device_info["has_openvino"] = HAS_OPENVINO
        if HAS_OPENVINO:
            try:
                core = ov.Core()
                self.device_info["openvino_devices"] = core.available_devices
            except:
                self.device_info["openvino_devices"] = []
        
        # Check for TensorRT
        self.device_info["has_tensorrt"] = HAS_TENSORRT
        
        # Check for ONNX Runtime
        self.device_info["has_onnxruntime"] = HAS_ONNXRUNTIME
        if HAS_ONNXRUNTIME:
            self.device_info["onnx_providers"] = ort.get_available_providers()
        
        # Check for Edge TPU (Coral)
        self.device_info["has_edge_tpu"] = self._check_edge_tpu()
    
    def _check_edge_tpu(self) -> bool:
        """Check for Google Coral Edge TPU"""
        try:
            import tflite_runtime.interpreter as tflite
            # Try to create interpreter with Edge TPU delegate
            tflite.Interpreter(model_path="", experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
            return True
        except:
            return False
    
    def suggest_device_type(self) -> EdgeDeviceType:
        """Suggest the most appropriate device type"""
        
        machine = self.device_info["machine"].lower()
        memory_gb = self.device_info["memory_total_gb"]
        cpu_count = self.device_info["cpu_count"]
        
        # Jetson devices
        if "tegra" in platform.platform().lower() or "jetson" in platform.platform().lower():
            if memory_gb >= 8:
                return EdgeDeviceType.JETSON_XAVIER
            else:
                return EdgeDeviceType.JETSON_NANO
        
        # Raspberry Pi
        if "arm" in machine and memory_gb <= 8 and cpu_count <= 4:
            return EdgeDeviceType.RASPBERRY_PI
        
        # Intel NUC or similar
        if "x86_64" in machine and memory_gb >= 4:
            return EdgeDeviceType.INTEL_NUC
        
        # Generic classifications
        if "arm" in machine:
            return EdgeDeviceType.GENERIC_ARM
        else:
            return EdgeDeviceType.GENERIC_X86
    
    def create_device_spec(self) -> EdgeDeviceSpec:
        """Create device specification from profiling"""
        
        device_type = self.suggest_device_type()
        
        return EdgeDeviceSpec(
            device_type=device_type,
            cpu_cores=self.device_info["cpu_count"],
            memory_gb=self.device_info["memory_total_gb"],
            storage_gb=self.device_info["disk_usage"].total / (1024**3),
            has_gpu=self.device_info["gpu_available"],
            gpu_memory_gb=self.device_info["gpu_memory_gb"],
            has_npu=bool(self.device_info.get("openvino_devices")),
            has_tpu=self.device_info.get("has_edge_tpu", False),
            architecture=self.device_info["machine"],
            power_budget_watts=self._estimate_power_budget(device_type),
            thermal_limit_celsius=self._estimate_thermal_limit(device_type)
        )
    
    def _estimate_power_budget(self, device_type: EdgeDeviceType) -> float:
        """Estimate power budget based on device type"""
        
        power_budgets = {
            EdgeDeviceType.RASPBERRY_PI: 5.0,
            EdgeDeviceType.JETSON_NANO: 10.0,
            EdgeDeviceType.JETSON_XAVIER: 30.0,
            EdgeDeviceType.INTEL_NUC: 15.0,
            EdgeDeviceType.ARM_CORTEX: 3.0,
            EdgeDeviceType.MOBILE_CPU: 2.0,
            EdgeDeviceType.EDGE_TPU: 4.0,
            EdgeDeviceType.GENERIC_ARM: 8.0,
            EdgeDeviceType.GENERIC_X86: 20.0
        }
        
        return power_budgets.get(device_type, 10.0)
    
    def _estimate_thermal_limit(self, device_type: EdgeDeviceType) -> float:
        """Estimate thermal limit based on device type"""
        
        thermal_limits = {
            EdgeDeviceType.RASPBERRY_PI: 80.0,
            EdgeDeviceType.JETSON_NANO: 85.0,
            EdgeDeviceType.JETSON_XAVIER: 90.0,
            EdgeDeviceType.INTEL_NUC: 85.0,
            EdgeDeviceType.ARM_CORTEX: 75.0,
            EdgeDeviceType.MOBILE_CPU: 70.0,
            EdgeDeviceType.EDGE_TPU: 80.0,
            EdgeDeviceType.GENERIC_ARM: 80.0,
            EdgeDeviceType.GENERIC_X86: 85.0
        }
        
        return thermal_limits.get(device_type, 85.0)


class EdgeModelOptimizer:
    """Optimize models specifically for edge deployment"""
    
    def __init__(self, config: EdgeDeploymentConfig):
        self.config = config
        self.device_spec = config.device_spec
    
    def optimize_model_for_edge(self, model_name_or_path: str, 
                               output_dir: str) -> Dict[str, Any]:
        """Optimize model specifically for edge deployment"""
        
        logger.info(f"Optimizing {model_name_or_path} for {self.device_spec.device_type.value}")
        
        optimization_results = {
            "original_size": 0,
            "optimized_size": 0,
            "optimization_techniques": [],
            "performance_gain": 0,
            "memory_reduction": 0
        }
        
        # Load original model
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        original_size = self._get_model_size(model)
        optimization_results["original_size"] = original_size
        
        # Apply optimization based on device capabilities
        optimized_model = model
        
        # 1. Quantization
        if self.config.quantization_bits < 16:
            optimized_model = self._apply_quantization(optimized_model)
            optimization_results["optimization_techniques"].append("quantization")
        
        # 2. Pruning
        if self.config.pruning_ratio > 0:
            optimized_model = self._apply_pruning(optimized_model)
            optimization_results["optimization_techniques"].append("pruning")
        
        # 3. Knowledge distillation for very constrained devices
        if (self.device_spec.memory_gb < 2 and self.config.enable_distillation):
            optimized_model = self._apply_distillation(optimized_model, tokenizer)
            optimization_results["optimization_techniques"].append("distillation")
        
        # 4. Convert to edge-optimized format
        if self.device_spec.has_tpu:
            optimized_model = self._convert_for_edge_tpu(optimized_model, tokenizer, output_dir)
            optimization_results["optimization_techniques"].append("edge_tpu_conversion")
        elif self.device_spec.has_npu and HAS_OPENVINO:
            optimized_model = self._convert_for_openvino(optimized_model, tokenizer, output_dir)
            optimization_results["optimization_techniques"].append("openvino_conversion")
        elif HAS_ONNXRUNTIME:
            optimized_model = self._convert_to_onnx(optimized_model, tokenizer, output_dir)
            optimization_results["optimization_techniques"].append("onnx_conversion")
        
        # Calculate final metrics
        final_size = self._get_model_size(optimized_model)
        optimization_results["optimized_size"] = final_size
        optimization_results["memory_reduction"] = ((original_size - final_size) / original_size) * 100
        
        # Save optimized model
        self._save_optimized_model(optimized_model, tokenizer, output_dir)
        
        return optimization_results
    
    def _apply_quantization(self, model: nn.Module) -> nn.Module:
        """Apply quantization appropriate for the edge device"""
        
        if self.config.quantization_bits == 8:
            # INT8 quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model.cpu(),
                {nn.Linear},
                dtype=torch.qint8
            )
        elif self.config.quantization_bits == 4:
            # INT4 quantization (more aggressive for very constrained devices)
            # This is a simplified implementation
            quantized_model = self._apply_int4_quantization(model)
        else:
            quantized_model = model
        
        return quantized_model
    
    def _apply_int4_quantization(self, model: nn.Module) -> nn.Module:
        """Apply INT4 quantization for very constrained devices"""
        
        # Simplified INT4 quantization
        # In practice, this would use specialized libraries
        logger.warning("INT4 quantization is experimental")
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Quantize weights to 4 bits
                weight = module.weight.data
                scale = weight.abs().max() / 7  # 4-bit signed range: -8 to 7
                quantized_weight = torch.round(weight / scale).clamp(-8, 7)
                
                # Store quantized weights (simplified)
                module.weight.data = quantized_weight * scale
        
        return model
    
    def _apply_pruning(self, model: nn.Module) -> nn.Module:
        """Apply structured or unstructured pruning"""
        
        import torch.nn.utils.prune as prune
        
        # Identify layers to prune
        layers_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layers_to_prune.append((module, 'weight'))
        
        # Apply magnitude-based pruning
        prune.global_unstructured(
            layers_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.config.pruning_ratio,
        )
        
        # Remove pruning reparameterization
        for module, param_name in layers_to_prune:
            prune.remove(module, param_name)
        
        return model
    
    def _apply_distillation(self, model: nn.Module, tokenizer) -> nn.Module:
        """Apply knowledge distillation to create smaller model"""
        
        # This is a simplified placeholder for knowledge distillation
        # In practice, this would train a smaller student model
        
        logger.info("Applying knowledge distillation (simplified)")
        
        # For very constrained devices, we might want to use a much smaller model
        if self.device_spec.memory_gb < 1:
            # Use a very small model architecture
            config = model.config
            config.n_layer = max(2, config.n_layer // 4)  # Reduce layers significantly
            config.n_head = max(1, config.n_head // 2)   # Reduce attention heads
            config.n_embd = max(128, config.n_embd // 2)  # Reduce embedding size
            
            # Create smaller model with same config class
            smaller_model = model.__class__(config)
            
            # Copy some weights (simplified distillation)
            # In practice, this would be a proper training process
            return smaller_model
        
        return model
    
    def _convert_for_edge_tpu(self, model: nn.Module, tokenizer, output_dir: str):
        """Convert model for Google Coral Edge TPU"""
        
        logger.info("Converting model for Edge TPU")
        
        try:
            # Convert to TensorFlow Lite format first
            import tensorflow as tf
            
            # This is a placeholder - actual implementation would:
            # 1. Convert PyTorch to ONNX
            # 2. Convert ONNX to TensorFlow
            # 3. Convert TensorFlow to TFLite
            # 4. Compile for Edge TPU using edgetpu_compiler
            
            output_path = Path(output_dir) / "model_edgetpu.tflite"
            logger.info(f"Edge TPU model would be saved to {output_path}")
            
            return model  # Return original for now
            
        except Exception as e:
            logger.error(f"Edge TPU conversion failed: {e}")
            return model
    
    def _convert_for_openvino(self, model: nn.Module, tokenizer, output_dir: str):
        """Convert model for Intel OpenVINO"""
        
        if not HAS_OPENVINO:
            logger.warning("OpenVINO not available")
            return model
        
        logger.info("Converting model for OpenVINO")
        
        try:
            # Convert to ONNX first
            onnx_path = Path(output_dir) / "model.onnx"
            self._export_to_onnx(model, tokenizer, str(onnx_path))
            
            # Convert ONNX to OpenVINO IR
            core = ov.Core()
            ov_model = core.read_model(str(onnx_path))
            
            # Optimize for specific device
            compiled_model = core.compile_model(ov_model, "CPU")
            
            # Save OpenVINO model
            ov_path = Path(output_dir) / "model_openvino.xml"
            ov.save_model(ov_model, str(ov_path))
            
            logger.info(f"OpenVINO model saved to {ov_path}")
            
            return compiled_model
            
        except Exception as e:
            logger.error(f"OpenVINO conversion failed: {e}")
            return model
    
    def _convert_to_onnx(self, model: nn.Module, tokenizer, output_dir: str):
        """Convert model to ONNX format"""
        
        onnx_path = Path(output_dir) / "model.onnx"
        self._export_to_onnx(model, tokenizer, str(onnx_path))
        
        # Load with ONNX Runtime
        if HAS_ONNXRUNTIME:
            # Choose providers based on device capabilities
            providers = ['CPUExecutionProvider']
            if self.device_spec.has_gpu:
                providers.insert(0, 'CUDAExecutionProvider')
            
            session = ort.InferenceSession(str(onnx_path), providers=providers)
            return session
        
        return model
    
    def _export_to_onnx(self, model: nn.Module, tokenizer, output_path: str):
        """Export PyTorch model to ONNX"""
        
        # Prepare dummy input
        dummy_input = tokenizer(
            "Hello world", 
            return_tensors="pt", 
            max_length=self.config.max_sequence_length,
            padding="max_length"
        )
        
        # Export to ONNX
        torch.onnx.export(
            model.cpu(),
            tuple(dummy_input.values()),
            output_path,
            input_names=list(dummy_input.keys()),
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "logits": {0: "batch_size"}
            },
            opset_version=11,
            do_constant_folding=True
        )
        
        logger.info(f"Model exported to ONNX: {output_path}")
    
    def _get_model_size(self, model) -> float:
        """Get model size in MB"""
        
        if hasattr(model, 'get_memory_footprint'):
            return model.get_memory_footprint() / (1024 * 1024)
        elif hasattr(model, 'parameters'):
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            return param_size / (1024 * 1024)
        else:
            return 0.0
    
    def _save_optimized_model(self, model, tokenizer, output_dir: str):
        """Save optimized model and configuration"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(output_path / "model")
        else:
            torch.save(model, output_path / "model.pt")
        
        # Save tokenizer
        if hasattr(tokenizer, 'save_pretrained'):
            tokenizer.save_pretrained(output_path / "tokenizer")
        
        # Save edge deployment config
        config_dict = {
            "device_type": self.device_spec.device_type.value,
            "quantization_bits": self.config.quantization_bits,
            "pruning_ratio": self.config.pruning_ratio,
            "max_sequence_length": self.config.max_sequence_length,
            "batch_size": self.config.batch_size,
            "target_latency_ms": self.config.target_latency_ms,
            "optimization_timestamp": time.time()
        }
        
        with open(output_path / "edge_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)


class EdgeDeploymentManager:
    """Manage edge deployment process"""
    
    def __init__(self, config: EdgeDeploymentConfig):
        self.config = config
        self.profiler = EdgeDeviceProfiler()
    
    def deploy_model(self, model_name_or_path: str, 
                    deployment_dir: str) -> EdgeDeploymentResult:
        """Deploy model to edge device"""
        
        start_time = time.time()
        deployment_path = Path(deployment_dir)
        deployment_path.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Starting edge deployment for {model_name_or_path}")
            
            # 1. Optimize model for edge
            optimizer = EdgeModelOptimizer(self.config)
            optimization_results = optimizer.optimize_model_for_edge(
                model_name_or_path, 
                str(deployment_path / "optimized_model")
            )
            
            # 2. Create deployment configuration
            deployment_config = self._create_deployment_config(deployment_path)
            
            # 3. Create container or standalone deployment
            if self.config.container_runtime != "none":
                self._create_container_deployment(deployment_path)
            else:
                self._create_standalone_deployment(deployment_path)
            
            # 4. Create monitoring and management scripts
            if self.config.enable_monitoring:
                self._create_monitoring_setup(deployment_path)
            
            # 5. Benchmark deployed model
            performance_metrics = self._benchmark_deployment(deployment_path)
            
            # 6. Create maintenance scripts
            self._create_maintenance_scripts(deployment_path)
            
            deployment_time = time.time() - start_time
            
            return EdgeDeploymentResult(
                success=True,
                deployment_time=deployment_time,
                model_size_mb=optimization_results["optimized_size"],
                memory_usage_mb=performance_metrics.get("memory_usage_mb", 0),
                startup_time_ms=performance_metrics.get("startup_time_ms", 0),
                inference_latency_ms=performance_metrics.get("inference_latency_ms", 0),
                throughput_tokens_per_second=performance_metrics.get("throughput", 0),
                power_consumption_watts=performance_metrics.get("power_consumption", 0),
                deployment_path=str(deployment_path),
                config_used=deployment_config,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            logger.error(f"Edge deployment failed: {e}")
            return EdgeDeploymentResult(
                success=False,
                deployment_time=time.time() - start_time,
                model_size_mb=0,
                memory_usage_mb=0,
                startup_time_ms=0,
                inference_latency_ms=0,
                throughput_tokens_per_second=0,
                power_consumption_watts=0,
                deployment_path=str(deployment_path),
                config_used={},
                error_message=str(e)
            )
    
    def _create_deployment_config(self, deployment_path: Path) -> Dict[str, Any]:
        """Create deployment configuration files"""
        
        config = {
            "deployment": {
                "device_type": self.config.device_spec.device_type.value,
                "strategy": self.config.strategy.value,
                "container_runtime": self.config.container_runtime,
                "enable_monitoring": self.config.enable_monitoring
            },
            "model": {
                "quantization_bits": self.config.quantization_bits,
                "batch_size": self.config.batch_size,
                "max_sequence_length": self.config.max_sequence_length,
                "enable_kv_cache": self.config.enable_kv_cache
            },
            "runtime": {
                "target_latency_ms": self.config.target_latency_ms,
                "max_memory_usage_mb": self.config.max_memory_usage_mb,
                "enable_dynamic_batching": self.config.use_dynamic_batching
            },
            "hardware": {
                "cpu_cores": self.config.device_spec.cpu_cores,
                "memory_gb": self.config.device_spec.memory_gb,
                "has_gpu": self.config.device_spec.has_gpu,
                "power_budget_watts": self.config.device_spec.power_budget_watts
            }
        }
        
        # Save as YAML for better readability
        with open(deployment_path / "deployment_config.yaml", 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        # Also save as JSON for programmatic access
        with open(deployment_path / "deployment_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        return config
    
    def _create_container_deployment(self, deployment_path: Path):
        """Create containerized deployment"""
        
        # Create Dockerfile optimized for edge device
        dockerfile_content = self._generate_edge_dockerfile()
        
        with open(deployment_path / "Dockerfile", 'w') as f:
            f.write(dockerfile_content)
        
        # Create docker-compose for easy management
        compose_content = self._generate_docker_compose()
        
        with open(deployment_path / "docker-compose.yml", 'w') as f:
            f.write(compose_content)
        
        # Create build and run scripts
        self._create_container_scripts(deployment_path)
    
    def _generate_edge_dockerfile(self) -> str:
        """Generate Dockerfile optimized for edge deployment"""
        
        # Choose base image based on device architecture
        if self.config.device_spec.architecture == "arm64":
            base_image = "python:3.9-slim-arm64"
        elif "arm" in self.config.device_spec.architecture:
            base_image = "python:3.9-slim-arm32v7"
        else:
            base_image = "python:3.9-slim"
        
        dockerfile = f"""
# Optimized Dockerfile for {self.config.device_spec.device_type.value}
FROM {base_image}

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy optimized model and application
COPY optimized_model/ ./model/
COPY app/ ./app/
COPY deployment_config.yaml .

# Set environment variables for optimization
ENV PYTHONPATH=/app
ENV OMP_NUM_THREADS={self.config.device_spec.cpu_cores}
ENV MKL_NUM_THREADS={self.config.device_spec.cpu_cores}

# Create non-root user for security
RUN useradd -m -u 1000 edgeuser && chown -R edgeuser:edgeuser /app
USER edgeuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
  CMD python app/health_check.py

# Expose port
EXPOSE 8080

# Start application
CMD ["python", "app/main.py"]
"""
        return dockerfile.strip()
    
    def _generate_docker_compose(self) -> str:
        """Generate docker-compose.yml for edge deployment"""
        
        # Calculate resource limits based on device specs
        memory_limit = f"{int(self.config.device_spec.memory_gb * 0.8)}g"
        cpu_limit = str(self.config.device_spec.cpu_cores)
        
        compose = f"""
version: '3.8'

services:
  textnlp-edge:
    build: .
    container_name: textnlp-edge
    restart: unless-stopped
    ports:
      - "8080:8080"
    environment:
      - DEVICE_TYPE={self.config.device_spec.device_type.value}
      - LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
      - ./cache:/app/cache
    deploy:
      resources:
        limits:
          cpus: '{cpu_limit}'
          memory: {memory_limit}
        reservations:
          memory: 256M
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
"""
        
        # Add GPU support if available
        if self.config.device_spec.has_gpu:
            compose += """
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
"""
        
        return compose.strip()
    
    def _create_container_scripts(self, deployment_path: Path):
        """Create container management scripts"""
        
        # Build script
        build_script = """#!/bin/bash
set -e

echo "Building TextNLP Edge container..."
docker build -t textnlp-edge .

echo "Build completed successfully!"
"""
        
        with open(deployment_path / "build.sh", 'w') as f:
            f.write(build_script)
        
        # Run script
        run_script = """#!/bin/bash
set -e

echo "Starting TextNLP Edge deployment..."
docker-compose up -d

echo "Deployment started! Check status with: docker-compose ps"
echo "View logs with: docker-compose logs -f"
echo "Stop with: docker-compose down"
"""
        
        with open(deployment_path / "run.sh", 'w') as f:
            f.write(run_script)
        
        # Make scripts executable
        (deployment_path / "build.sh").chmod(0o755)
        (deployment_path / "run.sh").chmod(0o755)
    
    def _create_standalone_deployment(self, deployment_path: Path):
        """Create standalone deployment without containers"""
        
        # Create Python virtual environment setup script
        venv_script = f"""#!/bin/bash
set -e

echo "Setting up Python virtual environment for edge deployment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt

echo "Virtual environment setup completed!"
echo "Activate with: source venv/bin/activate"
echo "Run with: python app/main.py"
"""
        
        with open(deployment_path / "setup_venv.sh", 'w') as f:
            f.write(venv_script)
        
        # Create system service file for automatic startup
        service_content = self._generate_systemd_service(deployment_path)
        
        with open(deployment_path / "textnlp-edge.service", 'w') as f:
            f.write(service_content)
        
        # Create installation script
        install_script = f"""#!/bin/bash
set -e

echo "Installing TextNLP Edge as system service..."

# Create service user
sudo useradd -r -s /bin/false textnlp-edge || true

# Copy service file
sudo cp textnlp-edge.service /etc/systemd/system/

# Set permissions
sudo chown -R textnlp-edge:textnlp-edge {deployment_path}

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable textnlp-edge
sudo systemctl start textnlp-edge

echo "Service installed and started!"
echo "Check status with: sudo systemctl status textnlp-edge"
echo "View logs with: sudo journalctl -u textnlp-edge -f"
"""
        
        with open(deployment_path / "install_service.sh", 'w') as f:
            f.write(install_script)
        
        # Make scripts executable
        (deployment_path / "setup_venv.sh").chmod(0o755)
        (deployment_path / "install_service.sh").chmod(0o755)
    
    def _generate_systemd_service(self, deployment_path: Path) -> str:
        """Generate systemd service file"""
        
        return f"""
[Unit]
Description=TextNLP Edge Service
After=network.target

[Service]
Type=simple
User=textnlp-edge
Group=textnlp-edge
WorkingDirectory={deployment_path}
Environment=PATH={deployment_path}/venv/bin
ExecStart={deployment_path}/venv/bin/python app/main.py
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal
SyslogIdentifier=textnlp-edge

# Resource limits
MemoryLimit={int(self.config.max_memory_usage_mb)}M
CPUQuota={self.config.device_spec.cpu_cores * 100}%

[Install]
WantedBy=multi-user.target
"""
    
    def _create_monitoring_setup(self, deployment_path: Path):
        """Create monitoring and alerting setup"""
        
        # Create monitoring script
        monitoring_script = """#!/usr/bin/env python3
import psutil
import time
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collect_metrics():
    \"\"\"Collect system metrics\"\"\"
    metrics = {
        'timestamp': time.time(),
        'cpu_usage': psutil.cpu_percent(interval=1),
        'memory_usage': psutil.virtual_memory().percent,
        'memory_available_mb': psutil.virtual_memory().available / (1024 * 1024),
        'disk_usage': psutil.disk_usage('/').percent,
        'load_average': psutil.getloadavg(),
        'temperature': get_temperature()
    }
    return metrics

def get_temperature():
    \"\"\"Get CPU temperature if available\"\"\"
    try:
        # Try different temperature sources
        for temp_file in ['/sys/class/thermal/thermal_zone0/temp', 
                         '/sys/class/hwmon/hwmon0/temp1_input']:
            try:
                with open(temp_file, 'r') as f:
                    temp = int(f.read().strip())
                    if temp > 1000:  # Convert from millidegrees
                        temp = temp / 1000
                    return temp
            except:
                continue
        return None
    except:
        return None

def main():
    metrics_file = Path('logs/metrics.jsonl')
    metrics_file.parent.mkdir(exist_ok=True)
    
    while True:
        try:
            metrics = collect_metrics()
            
            # Log metrics
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(metrics) + '\\n')
            
            # Check for alerts
            if metrics['cpu_usage'] > 80:
                logger.warning(f"High CPU usage: {metrics['cpu_usage']:.1f}%")
            
            if metrics['memory_usage'] > 85:
                logger.warning(f"High memory usage: {metrics['memory_usage']:.1f}%")
            
            if metrics['temperature'] and metrics['temperature'] > 75:
                logger.warning(f"High temperature: {metrics['temperature']:.1f}°C")
            
            time.sleep(30)  # Collect metrics every 30 seconds
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
"""
        
        with open(deployment_path / "monitor.py", 'w') as f:
            f.write(monitoring_script)
        
        (deployment_path / "monitor.py").chmod(0o755)
    
    def _create_maintenance_scripts(self, deployment_path: Path):
        """Create maintenance and update scripts"""
        
        # Update script
        update_script = """#!/bin/bash
set -e

echo "Starting TextNLP Edge update..."

# Backup current deployment
backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"
cp -r optimized_model/ "$backup_dir/"
cp deployment_config.yaml "$backup_dir/"

echo "Backup created in $backup_dir"

# Check for updates (placeholder)
echo "Checking for model updates..."

# Download and validate new model if available
# This would connect to your model repository

echo "Update check completed"
"""
        
        with open(deployment_path / "update.sh", 'w') as f:
            f.write(update_script)
        
        # Health check script
        health_script = """#!/usr/bin/env python3
import requests
import sys
import json

def check_health():
    try:
        response = requests.get('http://localhost:8080/health', timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"Service is healthy: {health_data}")
            return True
        else:
            print(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check error: {e}")
        return False

if __name__ == "__main__":
    if check_health():
        sys.exit(0)
    else:
        sys.exit(1)
"""
        
        with open(deployment_path / "health_check.py", 'w') as f:
            f.write(health_script)
        
        # Make scripts executable
        (deployment_path / "update.sh").chmod(0o755)
        (deployment_path / "health_check.py").chmod(0o755)
    
    def _benchmark_deployment(self, deployment_path: Path) -> Dict[str, float]:
        """Benchmark the deployed model"""
        
        # This would run actual benchmarks on the deployed model
        # For now, return estimated metrics based on device specs
        
        # Estimate performance based on device capabilities
        cpu_performance_factor = self.config.device_spec.cpu_cores / 4.0  # Normalize to 4 cores
        memory_performance_factor = self.config.device_spec.memory_gb / 4.0  # Normalize to 4GB
        
        base_latency = 200  # Base latency in ms
        estimated_latency = base_latency / (cpu_performance_factor * 0.7 + memory_performance_factor * 0.3)
        
        return {
            "startup_time_ms": 5000,
            "inference_latency_ms": estimated_latency,
            "memory_usage_mb": self.config.max_memory_usage_mb * 0.8,
            "throughput": 1000 / estimated_latency,  # tokens per second
            "power_consumption": self.config.device_spec.power_budget_watts * 0.6
        }


# Example usage and configuration presets
def create_device_configs() -> Dict[str, EdgeDeploymentConfig]:
    """Create predefined configurations for common edge devices"""
    
    configs = {}
    
    # Raspberry Pi 4 Configuration
    rpi4_spec = EdgeDeviceSpec(
        device_type=EdgeDeviceType.RASPBERRY_PI,
        cpu_cores=4,
        memory_gb=4.0,
        storage_gb=32.0,
        has_gpu=False,
        architecture="arm64",
        power_budget_watts=5.0
    )
    
    configs["raspberry_pi_4"] = EdgeDeploymentConfig(
        device_spec=rpi4_spec,
        strategy=DeploymentStrategy.LIGHTWEIGHT,
        quantization_bits=8,
        pruning_ratio=0.4,
        max_sequence_length=128,
        batch_size=1,
        target_latency_ms=1000.0,
        max_memory_usage_mb=512.0
    )
    
    # Jetson Nano Configuration
    jetson_nano_spec = EdgeDeviceSpec(
        device_type=EdgeDeviceType.JETSON_NANO,
        cpu_cores=4,
        memory_gb=4.0,
        storage_gb=64.0,
        has_gpu=True,
        gpu_memory_gb=2.0,
        architecture="arm64",
        power_budget_watts=10.0
    )
    
    configs["jetson_nano"] = EdgeDeploymentConfig(
        device_spec=jetson_nano_spec,
        strategy=DeploymentStrategy.BALANCED,
        quantization_bits=8,
        pruning_ratio=0.2,
        max_sequence_length=256,
        batch_size=2,
        target_latency_ms=500.0,
        max_memory_usage_mb=1024.0
    )
    
    # Intel NUC Configuration
    intel_nuc_spec = EdgeDeviceSpec(
        device_type=EdgeDeviceType.INTEL_NUC,
        cpu_cores=8,
        memory_gb=16.0,
        storage_gb=256.0,
        has_gpu=False,
        has_npu=True,
        architecture="x86_64",
        power_budget_watts=15.0
    )
    
    configs["intel_nuc"] = EdgeDeploymentConfig(
        device_spec=intel_nuc_spec,
        strategy=DeploymentStrategy.FEATURE_RICH,
        quantization_bits=8,
        pruning_ratio=0.1,
        max_sequence_length=512,
        batch_size=4,
        target_latency_ms=200.0,
        max_memory_usage_mb=2048.0,
        use_dynamic_batching=True
    )
    
    return configs


if __name__ == "__main__":
    # Example usage
    print("TextNLP Edge Deployment System")
    print("=" * 50)
    
    # Profile current device
    profiler = EdgeDeviceProfiler()
    device_spec = profiler.create_device_spec()
    
    print(f"Detected device: {device_spec.device_type.value}")
    print(f"CPU cores: {device_spec.cpu_cores}")
    print(f"Memory: {device_spec.memory_gb:.1f} GB")
    print(f"Architecture: {device_spec.architecture}")
    print(f"GPU available: {device_spec.has_gpu}")
    
    # Create appropriate configuration
    config = EdgeDeploymentConfig(
        device_spec=device_spec,
        strategy=DeploymentStrategy.BALANCED
    )
    
    # Deploy model
    deployment_manager = EdgeDeploymentManager(config)
    
    print("\nStarting deployment...")
    result = deployment_manager.deploy_model(
        model_name_or_path="gpt2",  # Use small model for testing
        deployment_dir="./edge_deployment"
    )
    
    if result.success:
        print(f"Deployment successful!")
        print(f"Deployment time: {result.deployment_time:.2f}s")
        print(f"Model size: {result.model_size_mb:.1f} MB")
        print(f"Expected latency: {result.inference_latency_ms:.1f} ms")
        print(f"Deployment path: {result.deployment_path}")
    else:
        print(f"Deployment failed: {result.error_message}")