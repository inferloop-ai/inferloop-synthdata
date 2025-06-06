# scripts/benchmark.py
"""
Performance benchmarking script for Audio Synthetic Data Framework
"""

import time
import psutil
import torch
import torchaudio
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from audio_synth.sdk.client import AudioSynthSDK
from audio_synth.core.utils.config import AudioConfig, GenerationConfig

class PerformanceBenchmark:
    """Performance benchmarking suite"""
    
    def __init__(self, config_path: str = None):
        self.sdk = AudioSynthSDK(config_path)
        self.results = {}
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        import platform
        
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total / (1024**3),  # GB
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    
    def benchmark_generation_methods(self, 
                                   methods: List[str] = None,
                                   num_samples: int = 10,
                                   duration: float = 5.0) -> Dict[str, Any]:
        """Benchmark different generation methods"""
        
        methods = methods or ["diffusion", "tts", "gan", "vae"]
        results = {}
        
        print(f"🔬 Benchmarking generation methods...")
        print(f"Methods: {methods}")
        print(f"Samples per method: {num_samples}")
        print(f"Duration: {duration}s")
        print("-" * 50)
        
        for method in methods:
            print(f"\n📊 Benchmarking {method.upper()}...")
            
            method_results = {
                "times": [],
                "memory_usage": [],
                "quality_scores": [],
                "errors": 0
            }
            
            for i in range(num_samples):
                try:
                    # Monitor performance
                    start_time = time.time()
                    start_memory = psutil.virtual_memory().used / (1024**2)  # MB
                    
                    # Generate audio
                    result = self.sdk.generate_and_validate(
                        method=method,
                        prompt=f"Benchmark test {i+1}",
                        num_samples=1,
                        validators=["quality"]
                    )
                    
                    # Record metrics
                    end_time = time.time()
                    end_memory = psutil.virtual_memory().used / (1024**2)  # MB
                    
                    generation_time = end_time - start_time
                    memory_delta = end_memory - start_memory
                    
                    method_results["times"].append(generation_time)
                    method_results["memory_usage"].append(memory_delta)
                    
                    # Extract quality score
                    if result["validation"]["quality"]:
                        quality_metrics = result["validation"]["quality"][0]
                        avg_quality = sum(quality_metrics.values()) / len(quality_metrics)
                        method_results["quality_scores"].append(avg_quality)
                    
                    print(f"  Sample {i+1}/{num_samples}: {generation_time:.2f}s, {memory_delta:.1f}MB")
                    
                except Exception as e:
                    method_results["errors"] += 1
                    print(f"  Sample {i+1}/{num_samples}: ERROR - {str(e)}")
            
            # Calculate statistics
            if method_results["times"]:
                method_results["stats"] = {
                    "mean_time": np.mean(method_results["times"]),
                    "std_time": np.std(method_results["times"]),
                    "min_time": np.min(method_results["times"]),
                    "max_time": np.max(method_results["times"]),
                    "mean_memory": np.mean(method_results["memory_usage"]),
                    "std_memory": np.std(method_results["memory_usage"]),
                    "mean_quality": np.mean(method_results["quality_scores"]) if method_results["quality_scores"] else 0,
                    "success_rate": (num_samples - method_results["errors"]) / num_samples
                }
            
            results[method] = method_results
            
            # Print summary
            if method_results["times"]:
                stats = method_results["stats"]
                print(f"  ✅ {method.upper()} Summary:")
                print(f"     Average time: {stats['mean_time']:.2f}±{stats['std_time']:.2f}s")
                print(f"     Average memory: {stats['mean_memory']:.1f}±{stats['std_memory']:.1f}MB")
                print(f"     Success rate: {stats['success_rate']:.1%}")
                if stats['mean_quality'] > 0:
                    print(f"     Average quality: {stats['mean_quality']:.3f}")
            else:
                print(f"  ❌ {method.upper()}: All samples failed")
        
        return results
    
    def benchmark_batch_sizes(self, 
                             method: str = "diffusion",
                             batch_sizes: List[int] = None,
                             duration: float = 2.0) -> Dict[str, Any]:
        """Benchmark different batch sizes"""
        
        batch_sizes = batch_sizes or [1, 2, 4, 8, 16]
        results = {}
        
        print(f"\n🔬 Benchmarking batch sizes for {method.upper()}...")
        print(f"Batch sizes: {batch_sizes}")
        print(f"Duration: {duration}s")
        print("-" * 50)
        
        for batch_size in batch_sizes:
            print(f"\n📊 Testing batch size: {batch_size}")
            
            try:
                # Monitor performance
                start_time = time.time()
                start_memory = psutil.virtual_memory().used / (1024**2)  # MB
                
                # Generate batch
                prompts = [f"Batch test {i}" for i in range(batch_size)]
                audios = self.sdk.generate(
                    method=method,
                    prompt=prompts[0],  # Use first prompt
                    num_samples=batch_size
                )
                
                # Record metrics
                end_time = time.time()
                end_memory = psutil.virtual_memory().used / (1024**2)  # MB
                
                total_time = end_time - start_time
                memory_delta = end_memory - start_memory
                time_per_sample = total_time / batch_size
                memory_per_sample = memory_delta / batch_size
                
                results[batch_size] = {
                    "total_time": total_time,
                    "time_per_sample": time_per_sample,
                    "memory_delta": memory_delta,
                    "memory_per_sample": memory_per_sample,
                    "throughput": batch_size / total_time,  # samples per second
                    "success": True
                }
                
                print(f"  ✅ Batch {batch_size}: {total_time:.2f}s total, {time_per_sample:.2f}s/sample")
                print(f"     Memory: {memory_delta:.1f}MB total, {memory_per_sample:.1f}MB/sample")
                print(f"     Throughput: {batch_size / total_time:.2f} samples/second")
                
            except Exception as e:
                results[batch_size] = {
                    "error": str(e),
                    "success": False
                }
                print(f"  ❌ Batch {batch_size}: ERROR - {str(e)}")
        
        return results
    
    def benchmark_validation_performance(self, 
                                       num_samples: int = 20,
                                       validators: List[str] = None) -> Dict[str, Any]:
        """Benchmark validation performance"""
        
        validators = validators or ["quality", "privacy", "fairness"]
        results = {}
        
        print(f"\n🔬 Benchmarking validation performance...")
        print(f"Validators: {validators}")
        print(f"Samples: {num_samples}")
        print("-" * 50)
        
        # Generate test audio samples
        print("Generating test audio samples...")
        test_audios = []
        test_metadata = []
        
        for i in range(num_samples):
            audio = self.sdk.generate(
                method="diffusion",
                prompt=f"Validation test {i}",
                num_samples=1
            )[0]
            test_audios.append(audio)
            test_metadata.append({
                "sample_id": i,
                "demographics": {
                    "gender": ["male", "female", "other"][i % 3],
                    "age_group": ["adult", "child", "elderly"][i % 3]
                }
            })
        
        # Benchmark each validator
        for validator in validators:
            print(f"\n📊 Testing {validator} validator...")
            
            try:
                start_time = time.time()
                start_memory = psutil.virtual_memory().used / (1024**2)  # MB
                
                # Run validation
                validation_results = self.sdk.validate(
                    audios=test_audios,
                    metadata=test_metadata,
                    validators=[validator]
                )
                
                end_time = time.time()
                end_memory = psutil.virtual_memory().used / (1024**2)  # MB
                
                total_time = end_time - start_time
                memory_delta = end_memory - start_memory
                time_per_sample = total_time / num_samples
                
                results[validator] = {
                    "total_time": total_time,
                    "time_per_sample": time_per_sample,
                    "memory_delta": memory_delta,
                    "throughput": num_samples / total_time,
                    "success": True
                }
                
                print(f"  ✅ {validator}: {total_time:.2f}s total, {time_per_sample:.3f}s/sample")
                print(f"     Throughput: {num_samples / total_time:.1f} samples/second")
                
            except Exception as e:
                results[validator] = {
                    "error": str(e),
                    "success": False
                }
                print(f"  ❌ {validator}: ERROR - {str(e)}")
        
        return results
    
    def stress_test(self, 
                   duration_minutes: int = 5,
                   concurrent_requests: int = 3) -> Dict[str, Any]:
        """Run stress test"""
        
        print(f"\n🔥 Running stress test...")
        print(f"Duration: {duration_minutes} minutes")
        print(f"Concurrent requests: {concurrent_requests}")
        print("-" * 50)
        
        import threading
        import queue
        
        results_queue = queue.Queue()
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        def worker(worker_id):
            """Worker thread for stress testing"""
            worker_results = {
                "requests": 0,
                "successes": 0,
                "failures": 0,
                "times": [],
                "errors": []
            }
            
            while time.time() < end_time:
                try:
                    request_start = time.time()
                    
                    # Generate audio
                    _ = self.sdk.generate(
                        method="diffusion",
                        prompt=f"Stress test worker {worker_id}",
                        num_samples=1
                    )
                    
                    request_time = time.time() - request_start
                    worker_results["times"].append(request_time)
                    worker_results["successes"] += 1
                    
                except Exception as e:
                    worker_results["failures"] += 1
                    worker_results["errors"].append(str(e))
                
                worker_results["requests"] += 1
                
                # Small delay to prevent overwhelming
                time.sleep(0.1)
            
            results_queue.put(worker_results)
        
        # Start worker threads
        threads = []
        for i in range(concurrent_requests):
            thread = threading.Thread(target=worker, args=(i,))
            thread.start()
            threads.append(thread)
        
        # Monitor system resources during test
        resource_monitor = []
        monitor_start = time.time()
        
        while time.time() < end_time:
            current_time = time.time() - monitor_start
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            resource_monitor.append({
                "time": current_time,
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent
            })
            
            time.sleep(1)  # Monitor every second
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        all_results = []
        while not results_queue.empty():
            all_results.append(results_queue.get())
        
        # Aggregate results
        total_requests = sum(r["requests"] for r in all_results)
        total_successes = sum(r["successes"] for r in all_results)
        total_failures = sum(r["failures"] for r in all_results)
        all_times = [t for r in all_results for t in r["times"]]
        
        stress_results = {
            "duration_minutes": duration_minutes,
            "concurrent_requests": concurrent_requests,
            "total_requests": total_requests,
            "total_successes": total_successes,
            "total_failures": total_failures,
            "success_rate": total_successes / total_requests if total_requests > 0 else 0,
            "requests_per_minute": total_requests / duration_minutes,
            "average_response_time": np.mean(all_times) if all_times else 0,
            "max_response_time": np.max(all_times) if all_times else 0,
            "resource_monitor": resource_monitor
        }
        
        print(f"\n✅ Stress test completed!")
        print(f"   Total requests: {total_requests}")
        print(f"   Success rate: {stress_results['success_rate']:.1%}")
        print(f"   Requests per minute: {stress_results['requests_per_minute']:.1f}")
        print(f"   Average response time: {stress_results['average_response_time']:.2f}s")
        
        return stress_results
    
    def run_full_benchmark(self, output_dir: str = "./benchmark_results") -> Dict[str, Any]:
        """Run complete benchmark suite"""
        
        print("🚀 Starting Full Benchmark Suite")
        print("=" * 50)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize results
        full_results = {
            "timestamp": timestamp,
            "system_info": self.system_info,
            "benchmarks": {}
        }
        
        # 1. Generation methods benchmark
        print("\n1️⃣ GENERATION METHODS BENCHMARK")
        full_results["benchmarks"]["generation_methods"] = self.benchmark_generation_methods()
        
        # 2. Batch sizes benchmark
        print("\n2️⃣ BATCH SIZES BENCHMARK")
        full_results["benchmarks"]["batch_sizes"] = self.benchmark_batch_sizes()
        
        # 3. Validation performance benchmark
        print("\n3️⃣ VALIDATION PERFORMANCE BENCHMARK")
        full_results["benchmarks"]["validation_performance"] = self.benchmark_validation_performance()
        
        # 4. Stress test (optional - commented out for regular runs)
        # print("\n4️⃣ STRESS TEST")
        # full_results["benchmarks"]["stress_test"] = self.stress_test(duration_minutes=2)
        
        # Save results
        results_file = output_path / f"benchmark_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(full_results, f, indent=2, default=str)
        
        # Generate plots
        self.generate_benchmark_plots(full_results, output_path, timestamp)
        
        # Generate report
        self.generate_benchmark_report(full_results, output_path, timestamp)
        
        print(f"\n🎉 Benchmark completed!")
        print(f"Results saved to: {output_path}")
        print(f"View the report: {output_path}/benchmark_report_{timestamp}.html")
        
        return full_results
    
    def generate_benchmark_plots(self, results: Dict[str, Any], output_path: Path, timestamp: str):
        """Generate benchmark visualization plots"""
        
        plt.style.use('seaborn-v0_8')
        
        # 1. Generation Methods Performance
        gen_results = results["benchmarks"]["generation_methods"]
        methods = []
        mean_times = []
        mean_qualities = []
        
        for method, data in gen_results.items():
            if "stats" in data:
                methods.append(method.upper())
                mean_times.append(data["stats"]["mean_time"])
                mean_qualities.append(data["stats"]["mean_quality"])
        
        if methods:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Generation time comparison
            bars1 = ax1.bar(methods, mean_times, color='skyblue')
            ax1.set_title('Generation Time by Method')
            ax1.set_ylabel('Time (seconds)')
            ax1.set_xlabel('Method')
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}s', ha='center', va='bottom')
            
            # Quality comparison
            bars2 = ax2.bar(methods, mean_qualities, color='lightgreen')
            ax2.set_title('Quality Score by Method')
            ax2.set_ylabel('Quality Score')
            ax2.set_xlabel('Method')
            
            # Add value labels on bars
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_path / f"generation_methods_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Batch Size Performance
        batch_results = results["benchmarks"]["batch_sizes"]
        batch_sizes = []
        throughputs = []
        
        for batch_size, data in batch_results.items():
            if data.get("success", False):
                batch_sizes.append(int(batch_size))
                throughputs.append(data["throughput"])
        
        if batch_sizes:
            plt.figure(figsize=(10, 6))
            plt.plot(batch_sizes, throughputs, marker='o', linewidth=2, markersize=8)
            plt.title('Throughput vs Batch Size')
            plt.xlabel('Batch Size')
            plt.ylabel('Throughput (samples/second)')
            plt.grid(True, alpha=0.3)
            
            # Add value labels
            for x, y in zip(batch_sizes, throughputs):
                plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
            
            plt.savefig(output_path / f"batch_performance_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Validation Performance
        val_results = results["benchmarks"]["validation_performance"]
        validators = []
        val_times = []
        
        for validator, data in val_results.items():
            if data.get("success", False):
                validators.append(validator.upper())
                val_times.append(data["time_per_sample"])
        
        if validators:
            plt.figure(figsize=(8, 6))
            bars = plt.bar(validators, val_times, color='orange')
            plt.title('Validation Time per Sample')
            plt.ylabel('Time (seconds)')
            plt.xlabel('Validator')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}s', ha='center', va='bottom')
            
            plt.savefig(output_path / f"validation_performance_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📊 Benchmark plots saved to {output_path}")
    
    def generate_benchmark_report(self, results: Dict[str, Any], output_path: Path, timestamp: str):
        """Generate HTML benchmark report"""
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Audio Synthesis Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin: 10px 0; }}
        .good {{ color: green; font-weight: bold; }}
        .warning {{ color: orange; font-weight: bold; }}
        .error {{ color: red; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .chart {{ text-align: center; margin: 20px 0; }}
        img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎵 Audio Synthesis Benchmark Report</h1>
        <p><strong>Generated:</strong> {results['timestamp']}</p>
        <p><strong>System:</strong> {results['system_info']['platform']}</p>
        <p><strong>CPU:</strong> {results['system_info']['cpu_count']} cores</p>
        <p><strong>Memory:</strong> {results['system_info']['memory_total']:.1f} GB</p>
        <p><strong>CUDA:</strong> {'Available' if results['system_info']['cuda_available'] else 'Not Available'}</p>
    </div>
    
    <div class="section">
        <h2>📊 Generation Methods Performance</h2>
        {self._generate_generation_methods_html(results['benchmarks']['generation_methods'])}
        <div class="chart">
            <img src="generation_methods_{timestamp}.png" alt="Generation Methods Performance">
        </div>
    </div>
    
    <div class="section">
        <h2>📈 Batch Size Performance</h2>
        {self._generate_batch_size_html(results['benchmarks']['batch_sizes'])}
        <div class="chart">
            <img src="batch_performance_{timestamp}.png" alt="Batch Performance">
        </div>
    </div>
    
    <div class="section">
        <h2>🔍 Validation Performance</h2>
        {self._generate_validation_html(results['benchmarks']['validation_performance'])}
        <div class="chart">
            <img src="validation_performance_{timestamp}.png" alt="Validation Performance">
        </div>
    </div>
    
    <div class="section">
        <h2>📋 Summary & Recommendations</h2>
        {self._generate_recommendations_html(results)}
    </div>
</body>
</html>
"""
        
        report_file = output_path / f"benchmark_report_{timestamp}.html"
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        print(f"📋 Benchmark report saved to {report_file}")
    
    def _generate_generation_methods_html(self, gen_results: Dict[str, Any]) -> str:
        """Generate HTML for generation methods results"""
        
        html = "<table><tr><th>Method</th><th>Avg Time (s)</th><th>Success Rate</th><th>Avg Quality</th><th>Status</th></tr>"
        
        for method, data in gen_results.items():
            if "stats" in data:
                stats = data["stats"]
                status = "good" if stats["success_rate"] > 0.8 else "warning" if stats["success_rate"] > 0.5 else "error"
                
                html += f"""<tr>
                    <td>{method.upper()}</td>
                    <td>{stats['mean_time']:.2f} ± {stats['std_time']:.2f}</td>
                    <td class="{status}">{stats['success_rate']:.1%}</td>
                    <td>{stats['mean_quality']:.3f}</td>
                    <td class="{status}">{'✅ Good' if status == 'good' else '⚠️ Warning' if status == 'warning' else '❌ Poor'}</td>
                </tr>"""
            else:
                html += f"""<tr>
                    <td>{method.upper()}</td>
                    <td colspan="4" class="error">❌ Failed</td>
                </tr>"""
        
        html += "</table>"
        return html
    
    def _generate_batch_size_html(self, batch_results: Dict[str, Any]) -> str:
        """Generate HTML for batch size results"""
        
        html = "<table><tr><th>Batch Size</th><th>Throughput (samples/s)</th><th>Time per Sample (s)</th><th>Memory per Sample (MB)</th></tr>"
        
        for batch_size, data in batch_results.items():
            if data.get("success", False):
                html += f"""<tr>
                    <td>{batch_size}</td>
                    <td>{data['throughput']:.2f}</td>
                    <td>{data['time_per_sample']:.2f}</td>
                    <td>{data['memory_per_sample']:.1f}</td>
                </tr>"""
            else:
                html += f"""<tr>
                    <td>{batch_size}</td>
                    <td colspan="3" class="error">❌ Failed</td>
                </tr>"""
        
        html += "</table>"
        return html
    
    def _generate_validation_html(self, val_results: Dict[str, Any]) -> str:
        """Generate HTML for validation results"""
        
        html = "<table><tr><th>Validator</th><th>Time per Sample (s)</th><th>Throughput (samples/s)</th><th>Status</th></tr>"
        
        for validator, data in val_results.items():
            if data.get("success", False):
                status = "good" if data["time_per_sample"] < 0.1 else "warning" if data["time_per_sample"] < 0.5 else "error"
                
                html += f"""<tr>
                    <td>{validator.upper()}</td>
                    <td>{data['time_per_sample']:.3f}</td>
                    <td>{data['throughput']:.1f}</td>
                    <td class="{status}">{'✅ Fast' if status == 'good' else '⚠️ Moderate' if status == 'warning' else '🐌 Slow'}</td>
                </tr>"""
            else:
                html += f"""<tr>
                    <td>{validator.upper()}</td>
                    <td colspan="3" class="error">❌ Failed</td>
                </tr>"""
        
        html += "</table>"
        return html
    
    def _generate_recommendations_html(self, results: Dict[str, Any]) -> str:
        """Generate recommendations based on benchmark results"""
        
        recommendations = []
        
        # Analyze generation methods
        gen_results = results["benchmarks"]["generation_methods"]
        fastest_method = None
        fastest_time = float('inf')
        
        for method, data in gen_results.items():
            if "stats" in data and data["stats"]["mean_time"] < fastest_time:
                fastest_time = data["stats"]["mean_time"]
                fastest_method = method
        
        if fastest_method:
            recommendations.append(f"🚀 <strong>Fastest Method:</strong> {fastest_method.upper()} ({fastest_time:.2f}s average)")
        
        # Analyze batch performance
        batch_results = results["benchmarks"]["batch_sizes"]
        best_throughput = 0
        best_batch_size = 1
        
        for batch_size, data in batch_results.items():
            if data.get("success", False) and data["throughput"] > best_throughput:
                best_throughput = data["throughput"]
                best_batch_size = batch_size
        
        recommendations.append(f"📦 <strong>Optimal Batch Size:</strong> {best_batch_size} (throughput: {best_throughput:.2f} samples/s)")
        
        # System recommendations
        system_info = results["system_info"]
        if not system_info["cuda_available"]:
            recommendations.append("🔧 <strong>Hardware Recommendation:</strong> Consider using GPU acceleration for better performance")
        
        if system_info["memory_total"] < 8:
            recommendations.append("💾 <strong>Memory Recommendation:</strong> Consider upgrading to at least 8GB RAM for better performance")
        
        html = "<div class='metric'>"
        for rec in recommendations:
            html += f"<p>{rec}</p>"
        html += "</div>"
        
        return html

def main():
    """Main benchmarking script"""
    
    parser = argparse.ArgumentParser(description="Audio Synthesis Performance Benchmark")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--output", type=str, default="./benchmark_results", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark with fewer samples")
    parser.add_argument("--methods", nargs="+", default=["diffusion", "tts"], help="Methods to benchmark")
    parser.add_argument("--stress", action="store_true", help="Include stress test")
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = PerformanceBenchmark(args.config)
    
    if args.quick:
        # Quick benchmark
        print("🏃‍♂️ Running quick benchmark...")
        results = {}
        results["generation_methods"] = benchmark.benchmark_generation_methods(
            methods=args.methods, num_samples=3, duration=2.0
        )
        results["batch_sizes"] = benchmark.benchmark_batch_sizes(
            method=args.methods[0], batch_sizes=[1, 2, 4], duration=1.0
        )
        
        # Save quick results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / f"quick_benchmark_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Quick benchmark completed! Results: {results_file}")
    
    else:
        # Full benchmark
        results = benchmark.run_full_benchmark(args.output)
        
        if args.stress:
            print("\n🔥 Running additional stress test...")
            stress_results = benchmark.stress_test(duration_minutes=2)
            results["benchmarks"]["stress_test"] = stress_results

if __name__ == "__main__":
    main()

# ============================================================================
