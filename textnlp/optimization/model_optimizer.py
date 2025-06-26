"""
Model Optimization for TextNLP
Advanced model optimization including INT8 quantization, pruning, and distillation
"""

import os
import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
import time
import numpy as np
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoConfig,
    BitsAndBytesConfig, TrainingArguments
)
from optimum.intel.openvino import OVModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

try:
    import intel_extension_for_pytorch as ipex
    HAS_IPEX = True
except ImportError:
    HAS_IPEX = False

try:
    from neural_compressor import Quantization
    HAS_NEURAL_COMPRESSOR = True
except ImportError:
    HAS_NEURAL_COMPRESSOR = False

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of model optimizations"""
    INT8_QUANTIZATION = "int8_quantization"
    INT4_QUANTIZATION = "int4_quantization"
    DYNAMIC_QUANTIZATION = "dynamic_quantization"
    PRUNING = "pruning"
    DISTILLATION = "distillation"
    ONNX_OPTIMIZATION = "onnx_optimization"
    OPENVINO_OPTIMIZATION = "openvino_optimization"
    TORCH_COMPILE = "torch_compile"
    TENSOR_RT = "tensor_rt"


@dataclass
class OptimizationConfig:
    """Configuration for model optimization"""
    optimization_type: OptimizationType
    target_accuracy_loss: float = 0.05  # Max acceptable accuracy loss
    target_speedup: float = 2.0  # Target speedup multiplier
    quantization_approach: str = "dynamic"  # "dynamic", "static", "qat"
    calibration_dataset_size: int = 100
    output_dir: str = "optimized_models"
    save_original: bool = True
    benchmark_iterations: int = 100
    
    # Quantization specific
    quantization_config: Dict[str, Any] = None
    
    # Pruning specific
    pruning_ratio: float = 0.5
    pruning_structured: bool = False
    
    # Distillation specific
    teacher_model: str = None
    temperature: float = 4.0
    alpha: float = 0.7


@dataclass
class OptimizationResult:
    """Result of model optimization"""
    original_model_size: float  # MB
    optimized_model_size: float  # MB
    size_reduction: float  # Percentage
    original_latency: float  # ms
    optimized_latency: float  # ms
    speedup: float  # Multiplier
    accuracy_loss: float  # Percentage
    optimization_time: float  # seconds
    success: bool
    error_message: str = ""
    metadata: Dict[str, Any] = None


class ModelOptimizer:
    """Advanced model optimization system"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Model optimizer initialized with {config.optimization_type.value}")
    
    def optimize_model(self, model_name_or_path: str, 
                      tokenizer_name_or_path: Optional[str] = None,
                      calibration_data: Optional[List[str]] = None) -> OptimizationResult:
        """Optimize a model with the specified optimization type"""
        
        start_time = time.time()
        
        try:
            # Load original model and tokenizer
            logger.info(f"Loading model: {model_name_or_path}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            tokenizer_path = tokenizer_name_or_path or model_name_or_path
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Benchmark original model
            original_metrics = self._benchmark_model(model, tokenizer)
            
            # Apply optimization based on type
            if self.config.optimization_type == OptimizationType.INT8_QUANTIZATION:
                optimized_model = self._apply_int8_quantization(model, tokenizer, calibration_data)
            elif self.config.optimization_type == OptimizationType.INT4_QUANTIZATION:
                optimized_model = self._apply_int4_quantization(model, tokenizer)
            elif self.config.optimization_type == OptimizationType.DYNAMIC_QUANTIZATION:
                optimized_model = self._apply_dynamic_quantization(model)
            elif self.config.optimization_type == OptimizationType.PRUNING:
                optimized_model = self._apply_pruning(model, calibration_data)
            elif self.config.optimization_type == OptimizationType.ONNX_OPTIMIZATION:
                optimized_model = self._convert_to_onnx(model, tokenizer)
            elif self.config.optimization_type == OptimizationType.OPENVINO_OPTIMIZATION:
                optimized_model = self._convert_to_openvino(model, tokenizer)
            elif self.config.optimization_type == OptimizationType.TORCH_COMPILE:
                optimized_model = self._apply_torch_compile(model)
            else:
                raise ValueError(f"Unsupported optimization type: {self.config.optimization_type}")
            
            # Benchmark optimized model
            optimized_metrics = self._benchmark_model(optimized_model, tokenizer)
            
            # Calculate optimization results
            result = self._calculate_optimization_result(
                original_metrics, optimized_metrics, start_time
            )
            
            # Save optimized model
            if result.success:
                self._save_optimized_model(optimized_model, tokenizer, model_name_or_path)
                logger.info(f"Optimization completed successfully. Speedup: {result.speedup:.2f}x")
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return OptimizationResult(
                original_model_size=0.0,
                optimized_model_size=0.0,
                size_reduction=0.0,
                original_latency=0.0,
                optimized_latency=0.0,
                speedup=0.0,
                accuracy_loss=0.0,
                optimization_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _apply_int8_quantization(self, model: nn.Module, tokenizer, 
                                calibration_data: Optional[List[str]] = None) -> nn.Module:
        """Apply INT8 quantization using various backends"""
        
        if HAS_NEURAL_COMPRESSOR and calibration_data:
            # Use Intel Neural Compressor for static quantization
            return self._apply_neural_compressor_quantization(model, tokenizer, calibration_data)
        elif HAS_IPEX:
            # Use Intel Extension for PyTorch
            return self._apply_ipex_quantization(model)
        else:
            # Fallback to PyTorch dynamic quantization
            return self._apply_pytorch_quantization(model)
    
    def _apply_neural_compressor_quantization(self, model: nn.Module, tokenizer,
                                            calibration_data: List[str]) -> nn.Module:
        """Apply quantization using Intel Neural Compressor"""
        
        if not HAS_NEURAL_COMPRESSOR:
            raise ImportError("Intel Neural Compressor not available")
        
        logger.info("Applying Neural Compressor INT8 quantization")
        
        # Prepare calibration dataset
        def calibration_dataloader():
            for text in calibration_data[:self.config.calibration_dataset_size]:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                yield inputs
        
        # Configure quantization
        from neural_compressor.config import PostTrainingQuantConfig
        from neural_compressor import quantization
        
        config = PostTrainingQuantConfig(
            approach="static",
            backend="pytorch",
            calibration_sampling_size=len(calibration_data),
            accuracy_criterion={
                "relative": self.config.target_accuracy_loss,
                "higher_is_better": True
            }
        )
        
        # Apply quantization
        quantized_model = quantization.fit(
            model=model,
            conf=config,
            calib_dataloader=calibration_dataloader(),
            eval_func=lambda model: self._evaluate_model_accuracy(model, tokenizer, calibration_data)
        )
        
        return quantized_model
    
    def _apply_ipex_quantization(self, model: nn.Module) -> nn.Module:
        """Apply quantization using Intel Extension for PyTorch"""
        
        if not HAS_IPEX:
            raise ImportError("Intel Extension for PyTorch not available")
        
        logger.info("Applying IPEX INT8 quantization")
        
        # Convert model to Intel Extension format
        model = model.to("cpu")  # IPEX works on CPU
        
        # Apply dynamic quantization
        qconfig = ipex.quantization.default_dynamic_qconfig
        prepared_model = ipex.quantization.prepare(model, qconfig, example_inputs=None)
        quantized_model = ipex.quantization.convert(prepared_model)
        
        return quantized_model
    
    def _apply_pytorch_quantization(self, model: nn.Module) -> nn.Module:
        """Apply PyTorch dynamic quantization"""
        
        logger.info("Applying PyTorch dynamic quantization")
        
        # Apply dynamic quantization to linear layers
        quantized_model = torch.quantization.quantize_dynamic(
            model.cpu(),
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=torch.qint8
        )
        
        return quantized_model
    
    def _apply_int4_quantization(self, model: nn.Module, tokenizer) -> nn.Module:
        """Apply INT4 quantization using BitsAndBytesConfig"""
        
        logger.info("Applying INT4 quantization with BitsAndBytesConfig")
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Reload model with quantization config
        model_name = model.config._name_or_path if hasattr(model.config, '_name_or_path') else "unknown"
        quantized_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        return quantized_model
    
    def _apply_dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization"""
        
        logger.info("Applying dynamic quantization")
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model.cpu(),
            {nn.Linear},
            dtype=torch.qint8
        )
        
        return quantized_model
    
    def _apply_pruning(self, model: nn.Module, calibration_data: Optional[List[str]]) -> nn.Module:
        """Apply neural network pruning"""
        
        logger.info(f"Applying pruning with ratio {self.config.pruning_ratio}")
        
        import torch.nn.utils.prune as prune
        
        # Identify prunable modules
        modules_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                modules_to_prune.append((module, 'weight'))
        
        # Apply magnitude-based pruning
        if self.config.pruning_structured:
            # Structured pruning
            for module, param_name in modules_to_prune:
                prune.ln_structured(
                    module, 
                    name=param_name, 
                    amount=self.config.pruning_ratio,
                    n=2, 
                    dim=0
                )
        else:
            # Unstructured pruning
            prune.global_unstructured(
                modules_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=self.config.pruning_ratio,
            )
        
        # Remove pruning reparameterization
        for module, param_name in modules_to_prune:
            prune.remove(module, param_name)
        
        return model
    
    def _convert_to_onnx(self, model: nn.Module, tokenizer) -> Any:
        """Convert model to ONNX format with optimization"""
        
        logger.info("Converting to ONNX format")
        
        # Prepare dummy input
        dummy_input = tokenizer("Hello world", return_tensors="pt", max_length=512, padding="max_length")
        
        # Export to ONNX
        onnx_path = self.output_dir / "model.onnx"
        torch.onnx.export(
            model.cpu(),
            tuple(dummy_input.values()),
            onnx_path,
            input_names=list(dummy_input.keys()),
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=11,
            do_constant_folding=True
        )
        
        # Apply ONNX quantization
        quantized_onnx_path = self.output_dir / "model_quantized.onnx"
        quantize_dynamic(
            str(onnx_path),
            str(quantized_onnx_path),
            weight_type=QuantType.QInt8
        )
        
        # Load optimized ONNX model
        optimized_model = ORTModelForCausalLM.from_pretrained(
            str(quantized_onnx_path.parent),
            file_name=quantized_onnx_path.name
        )
        
        return optimized_model
    
    def _convert_to_openvino(self, model: nn.Module, tokenizer) -> Any:
        """Convert model to OpenVINO format"""
        
        logger.info("Converting to OpenVINO format")
        
        try:
            # First convert to ONNX, then to OpenVINO
            onnx_model = self._convert_to_onnx(model, tokenizer)
            
            # Convert ONNX to OpenVINO
            openvino_model = OVModelForCausalLM.from_pretrained(
                str(self.output_dir),
                export=True
            )
            
            return openvino_model
            
        except Exception as e:
            logger.warning(f"OpenVINO conversion failed: {e}")
            # Fallback to ONNX
            return self._convert_to_onnx(model, tokenizer)
    
    def _apply_torch_compile(self, model: nn.Module) -> nn.Module:
        """Apply PyTorch 2.0 compilation"""
        
        logger.info("Applying PyTorch compilation")
        
        if hasattr(torch, 'compile'):
            # PyTorch 2.0+ compilation
            compiled_model = torch.compile(
                model,
                mode="reduce-overhead",  # or "default", "max-autotune"
                dynamic=True
            )
            return compiled_model
        else:
            logger.warning("PyTorch compile not available, returning original model")
            return model
    
    def _benchmark_model(self, model, tokenizer, num_iterations: int = None) -> Dict[str, float]:
        """Benchmark model performance"""
        
        iterations = num_iterations or self.config.benchmark_iterations
        
        # Prepare test input
        test_input = "The quick brown fox jumps over the lazy dog. " * 10
        inputs = tokenizer(test_input, return_tensors="pt", max_length=512, truncation=True)
        
        if hasattr(model, 'to'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Warm up
        for _ in range(5):
            with torch.no_grad():
                _ = model(**inputs)
        
        # Benchmark
        latencies = []
        for _ in range(iterations):
            start_time = time.time()
            with torch.no_grad():
                outputs = model(**inputs)
            latencies.append((time.time() - start_time) * 1000)  # Convert to ms
        
        # Calculate model size
        if hasattr(model, 'get_memory_footprint'):
            model_size = model.get_memory_footprint() / 1024 / 1024  # MB
        else:
            # Estimate model size
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            model_size = param_size / 1024 / 1024  # MB
        
        return {
            "latency_mean": np.mean(latencies),
            "latency_std": np.std(latencies),
            "model_size": model_size
        }
    
    def _evaluate_model_accuracy(self, model, tokenizer, test_data: List[str]) -> float:
        """Evaluate model accuracy (simplified perplexity-based)"""
        
        total_loss = 0.0
        num_samples = 0
        
        model.eval()
        with torch.no_grad():
            for text in test_data[:50]:  # Limit for speed
                inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
                if hasattr(model, 'to'):
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item()
                num_samples += 1
        
        # Return perplexity (lower is better, so we return negative for "higher is better")
        perplexity = torch.exp(torch.tensor(total_loss / num_samples))
        return -perplexity.item()
    
    def _calculate_optimization_result(self, original_metrics: Dict[str, float],
                                     optimized_metrics: Dict[str, float],
                                     start_time: float) -> OptimizationResult:
        """Calculate optimization results"""
        
        size_reduction = ((original_metrics["model_size"] - optimized_metrics["model_size"]) / 
                         original_metrics["model_size"]) * 100
        
        speedup = original_metrics["latency_mean"] / optimized_metrics["latency_mean"]
        
        # Simplified accuracy loss calculation (would need proper evaluation in practice)
        accuracy_loss = 0.0  # Placeholder - would calculate actual accuracy difference
        
        success = (
            speedup >= self.config.target_speedup * 0.8 and  # 80% of target speedup
            accuracy_loss <= self.config.target_accuracy_loss
        )
        
        return OptimizationResult(
            original_model_size=original_metrics["model_size"],
            optimized_model_size=optimized_metrics["model_size"],
            size_reduction=size_reduction,
            original_latency=original_metrics["latency_mean"],
            optimized_latency=optimized_metrics["latency_mean"],
            speedup=speedup,
            accuracy_loss=accuracy_loss,
            optimization_time=time.time() - start_time,
            success=success,
            metadata={
                "original_latency_std": original_metrics["latency_std"],
                "optimized_latency_std": optimized_metrics["latency_std"]
            }
        )
    
    def _save_optimized_model(self, model, tokenizer, original_model_name: str):
        """Save optimized model and metadata"""
        
        # Create model-specific output directory
        model_dir = self.output_dir / f"{Path(original_model_name).name}_{self.config.optimization_type.value}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(model_dir)
        else:
            # For custom optimized models, save state dict
            torch.save(model.state_dict(), model_dir / "pytorch_model.bin")
        
        # Save tokenizer
        tokenizer.save_pretrained(model_dir)
        
        # Save optimization config
        config_path = model_dir / "optimization_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "optimization_type": self.config.optimization_type.value,
                "target_accuracy_loss": self.config.target_accuracy_loss,
                "target_speedup": self.config.target_speedup,
                "quantization_approach": self.config.quantization_approach,
                "optimization_timestamp": time.time()
            }, f, indent=2)
        
        logger.info(f"Optimized model saved to {model_dir}")


class OptimizationPipeline:
    """Pipeline for running multiple optimization strategies"""
    
    def __init__(self, output_dir: str = "optimization_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def run_optimization_suite(self, model_name: str, 
                             calibration_data: Optional[List[str]] = None) -> List[OptimizationResult]:
        """Run multiple optimization strategies and compare results"""
        
        optimization_types = [
            OptimizationType.DYNAMIC_QUANTIZATION,
            OptimizationType.INT8_QUANTIZATION,
            OptimizationType.TORCH_COMPILE,
            OptimizationType.PRUNING
        ]
        
        if torch.cuda.is_available():
            optimization_types.append(OptimizationType.INT4_QUANTIZATION)
        
        results = []
        
        for opt_type in optimization_types:
            logger.info(f"Running optimization: {opt_type.value}")
            
            config = OptimizationConfig(
                optimization_type=opt_type,
                output_dir=str(self.output_dir / opt_type.value),
                target_speedup=1.5,
                calibration_dataset_size=50 if calibration_data else 0
            )
            
            optimizer = ModelOptimizer(config)
            
            try:
                result = optimizer.optimize_model(
                    model_name, 
                    calibration_data=calibration_data
                )
                results.append(result)
                
                if result.success:
                    logger.info(f"{opt_type.value}: {result.speedup:.2f}x speedup, "
                              f"{result.size_reduction:.1f}% size reduction")
                else:
                    logger.warning(f"{opt_type.value} optimization failed: {result.error_message}")
                    
            except Exception as e:
                logger.error(f"Failed to run {opt_type.value}: {e}")
                results.append(OptimizationResult(
                    original_model_size=0, optimized_model_size=0, size_reduction=0,
                    original_latency=0, optimized_latency=0, speedup=0,
                    accuracy_loss=0, optimization_time=0, success=False,
                    error_message=str(e)
                ))
        
        self.results = results
        self._save_comparison_report()
        
        return results
    
    def _save_comparison_report(self):
        """Save optimization comparison report"""
        
        report = {
            "timestamp": time.time(),
            "optimizations": []
        }
        
        for i, result in enumerate(self.results):
            report["optimizations"].append({
                "optimization_type": list(OptimizationType)[i].value,
                "success": result.success,
                "speedup": result.speedup,
                "size_reduction": result.size_reduction,
                "accuracy_loss": result.accuracy_loss,
                "optimization_time": result.optimization_time,
                "error_message": result.error_message
            })
        
        # Find best optimization
        successful_results = [r for r in self.results if r.success]
        if successful_results:
            best_result = max(successful_results, key=lambda x: x.speedup)
            report["best_optimization"] = {
                "type": list(OptimizationType)[self.results.index(best_result)].value,
                "speedup": best_result.speedup,
                "size_reduction": best_result.size_reduction
            }
        
        report_path = self.output_dir / "optimization_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Optimization report saved to {report_path}")


# Example usage
if __name__ == "__main__":
    # Example calibration data
    calibration_data = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming many industries.",
        "Natural language processing enables computers to understand text.",
        "Optimization techniques can significantly improve model performance.",
        "Quantization reduces model size while maintaining accuracy."
    ] * 20  # Repeat to get more samples
    
    # Run optimization suite
    pipeline = OptimizationPipeline("optimization_results")
    
    # Test with a small model (replace with actual model)
    model_name = "gpt2"  # or path to your model
    
    results = pipeline.run_optimization_suite(model_name, calibration_data)
    
    print("\nOptimization Results Summary:")
    print("-" * 60)
    for i, result in enumerate(results):
        opt_type = list(OptimizationType)[i].value
        if result.success:
            print(f"{opt_type:25}: {result.speedup:.2f}x speedup, {result.size_reduction:.1f}% smaller")
        else:
            print(f"{opt_type:25}: FAILED - {result.error_message}")