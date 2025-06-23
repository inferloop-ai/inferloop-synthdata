"""
Performance benchmarking for synthetic data generators
"""

import time
import tracemalloc
import psutil
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import pandas as pd
import numpy as np
from pathlib import Path
import concurrent.futures
import matplotlib.pyplot as plt
import seaborn as sns

from .base import SyntheticDataConfig, BaseSyntheticGenerator
from .factory import GeneratorFactory
from .validator import SyntheticDataValidator


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    generator_type: str
    model_type: str
    dataset_name: str
    dataset_rows: int
    dataset_columns: int
    num_samples: int
    
    # Timing metrics
    fit_time: float
    generate_time: float
    total_time: float
    
    # Memory metrics
    peak_memory_mb: float
    memory_delta_mb: float
    
    # Quality metrics
    quality_score: float
    basic_stats_score: float
    distribution_score: float
    correlation_score: float
    privacy_score: float
    utility_score: float
    
    # System info
    cpu_count: int
    cpu_usage_avg: float
    
    # Additional metadata
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'generator_type': self.generator_type,
            'model_type': self.model_type,
            'dataset_name': self.dataset_name,
            'dataset_rows': self.dataset_rows,
            'dataset_columns': self.dataset_columns,
            'num_samples': self.num_samples,
            'fit_time': self.fit_time,
            'generate_time': self.generate_time,
            'total_time': self.total_time,
            'peak_memory_mb': self.peak_memory_mb,
            'memory_delta_mb': self.memory_delta_mb,
            'quality_score': self.quality_score,
            'basic_stats_score': self.basic_stats_score,
            'distribution_score': self.distribution_score,
            'correlation_score': self.correlation_score,
            'privacy_score': self.privacy_score,
            'utility_score': self.utility_score,
            'cpu_count': self.cpu_count,
            'cpu_usage_avg': self.cpu_usage_avg,
            'timestamp': self.timestamp.isoformat(),
            'error': self.error,
            'config': self.config
        }


class GeneratorBenchmark:
    """Benchmark different synthetic data generators"""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / 'benchmarks'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []
    
    def benchmark_generator(self,
                          data: pd.DataFrame,
                          generator_type: str,
                          model_type: str,
                          num_samples: Optional[int] = None,
                          dataset_name: str = "dataset",
                          config_overrides: Optional[Dict[str, Any]] = None) -> BenchmarkResult:
        """Benchmark a single generator"""
        
        # Default number of samples to dataset size
        if num_samples is None:
            num_samples = len(data)
        
        # Create configuration
        config = SyntheticDataConfig(
            generator_type=generator_type,
            model_type=model_type,
            num_samples=num_samples
        )
        
        # Apply overrides
        if config_overrides:
            for key, value in config_overrides.items():
                setattr(config, key, value)
        
        # Initialize metrics
        fit_time = 0.0
        generate_time = 0.0
        peak_memory_mb = 0.0
        memory_delta_mb = 0.0
        cpu_usage_samples = []
        error_msg = None
        
        # Quality metrics defaults
        quality_metrics = {
            'quality_score': 0.0,
            'basic_stats_score': 0.0,
            'distribution_score': 0.0,
            'correlation_score': 0.0,
            'privacy_score': 0.0,
            'utility_score': 0.0
        }
        
        try:
            # Start memory tracking
            tracemalloc.start()
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create generator
            generator = GeneratorFactory.create_generator(config)
            
            # Benchmark fitting
            print(f"Benchmarking {generator_type}/{model_type} on {dataset_name}...")
            
            # Monitor CPU usage in background
            cpu_monitor_stop = False
            def monitor_cpu():
                while not cpu_monitor_stop:
                    cpu_usage_samples.append(psutil.cpu_percent(interval=0.1))
            
            cpu_thread = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            cpu_future = cpu_thread.submit(monitor_cpu)
            
            # Fit model
            start_time = time.time()
            generator.fit(data)
            fit_time = time.time() - start_time
            
            # Generate synthetic data
            start_time = time.time()
            result = generator.generate(num_samples)
            generate_time = time.time() - start_time
            
            # Stop CPU monitoring
            cpu_monitor_stop = True
            cpu_future.result()
            
            # Get memory usage
            current_memory = process.memory_info().rss / 1024 / 1024
            peak_memory_mb = max(tracemalloc.get_traced_memory()[1] / 1024 / 1024, 
                                current_memory)
            memory_delta_mb = current_memory - initial_memory
            
            # Validate quality
            print("Evaluating quality metrics...")
            validator = SyntheticDataValidator(data, result.data)
            validation_results = validator.validate_all()
            
            quality_metrics = {
                'quality_score': validation_results['overall_quality'],
                'basic_stats_score': validation_results['basic_stats']['score'],
                'distribution_score': validation_results['distribution_similarity']['score'],
                'correlation_score': validation_results['correlation_preservation']['score'],
                'privacy_score': validation_results['privacy_metrics']['score'],
                'utility_score': validation_results['utility_metrics']['score']
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error benchmarking {generator_type}/{model_type}: {error_msg}")
        
        finally:
            tracemalloc.stop()
        
        # Create result
        result = BenchmarkResult(
            generator_type=generator_type,
            model_type=model_type,
            dataset_name=dataset_name,
            dataset_rows=len(data),
            dataset_columns=len(data.columns),
            num_samples=num_samples,
            fit_time=fit_time,
            generate_time=generate_time,
            total_time=fit_time + generate_time,
            peak_memory_mb=peak_memory_mb,
            memory_delta_mb=memory_delta_mb,
            cpu_count=psutil.cpu_count(),
            cpu_usage_avg=np.mean(cpu_usage_samples) if cpu_usage_samples else 0.0,
            error=error_msg,
            config=config.to_dict(),
            **quality_metrics
        )
        
        self.results.append(result)
        return result
    
    def benchmark_all_generators(self,
                               data: pd.DataFrame,
                               dataset_name: str = "dataset",
                               num_samples: Optional[int] = None,
                               generators: Optional[List[Tuple[str, str]]] = None) -> List[BenchmarkResult]:
        """Benchmark all available generators"""
        
        if generators is None:
            # Default generators to benchmark
            generators = [
                ('sdv', 'gaussian_copula'),
                ('sdv', 'ctgan'),
                ('sdv', 'copula_gan'),
                ('sdv', 'tvae'),
                ('ctgan', 'ctgan'),
                ('ctgan', 'tvae'),
                ('ydata', 'wgan_gp'),
                ('ydata', 'cramer_gan'),
                ('ydata', 'dragan')
            ]
        
        results = []
        for generator_type, model_type in generators:
            try:
                result = self.benchmark_generator(
                    data=data,
                    generator_type=generator_type,
                    model_type=model_type,
                    num_samples=num_samples,
                    dataset_name=dataset_name
                )
                results.append(result)
            except Exception as e:
                print(f"Failed to benchmark {generator_type}/{model_type}: {str(e)}")
        
        return results
    
    def benchmark_dataset_sizes(self,
                              data: pd.DataFrame,
                              generator_type: str,
                              model_type: str,
                              sizes: List[int],
                              dataset_name: str = "dataset") -> List[BenchmarkResult]:
        """Benchmark generator with different dataset sizes"""
        
        results = []
        for size in sizes:
            if size > len(data):
                print(f"Skipping size {size} (larger than dataset)")
                continue
            
            # Sample data
            sample_data = data.sample(n=size, random_state=42)
            
            result = self.benchmark_generator(
                data=sample_data,
                generator_type=generator_type,
                model_type=model_type,
                dataset_name=f"{dataset_name}_n{size}"
            )
            results.append(result)
        
        return results
    
    def save_results(self, filename: Optional[str] = None):
        """Save benchmark results to file"""
        
        if filename is None:
            filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.output_dir / filename
        
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
                'platform': {
                    'system': os.uname().sysname if hasattr(os, 'uname') else os.name,
                    'release': os.uname().release if hasattr(os, 'uname') else '',
                    'machine': os.uname().machine if hasattr(os, 'uname') else ''
                }
            },
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to: {filepath}")
    
    def generate_report(self) -> str:
        """Generate a text report of benchmark results"""
        
        if not self.results:
            return "No benchmark results available"
        
        report = []
        report.append("=" * 80)
        report.append("SYNTHETIC DATA GENERATOR BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Group results by dataset
        from collections import defaultdict
        by_dataset = defaultdict(list)
        for result in self.results:
            by_dataset[result.dataset_name].append(result)
        
        for dataset_name, dataset_results in by_dataset.items():
            report.append(f"\nDataset: {dataset_name}")
            report.append("-" * 40)
            
            # Sort by total time
            dataset_results.sort(key=lambda r: r.total_time if r.error is None else float('inf'))
            
            for result in dataset_results:
                report.append(f"\n{result.generator_type}/{result.model_type}:")
                
                if result.error:
                    report.append(f"  ERROR: {result.error}")
                else:
                    report.append(f"  Dataset size: {result.dataset_rows} rows × {result.dataset_columns} columns")
                    report.append(f"  Generated samples: {result.num_samples}")
                    report.append(f"  Timing:")
                    report.append(f"    - Fit time: {result.fit_time:.2f}s")
                    report.append(f"    - Generate time: {result.generate_time:.2f}s")
                    report.append(f"    - Total time: {result.total_time:.2f}s")
                    report.append(f"  Memory:")
                    report.append(f"    - Peak memory: {result.peak_memory_mb:.1f} MB")
                    report.append(f"    - Memory delta: {result.memory_delta_mb:.1f} MB")
                    report.append(f"  Quality scores:")
                    report.append(f"    - Overall: {result.quality_score:.3f}")
                    report.append(f"    - Basic stats: {result.basic_stats_score:.3f}")
                    report.append(f"    - Distribution: {result.distribution_score:.3f}")
                    report.append(f"    - Correlation: {result.correlation_score:.3f}")
                    report.append(f"    - Privacy: {result.privacy_score:.3f}")
                    report.append(f"    - Utility: {result.utility_score:.3f}")
                    report.append(f"  CPU usage: {result.cpu_usage_avg:.1f}%")
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)
    
    def plot_results(self, save_plots: bool = True):
        """Generate visualization of benchmark results"""
        
        if not self.results:
            print("No results to plot")
            return
        
        # Convert results to DataFrame
        df = pd.DataFrame([r.to_dict() for r in self.results if r.error is None])
        
        if df.empty:
            print("No successful results to plot")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Synthetic Data Generator Benchmark Results', fontsize=16)
        
        # Helper function to create generator labels
        def get_label(row):
            return f"{row['generator_type']}/{row['model_type']}"
        
        df['generator'] = df.apply(get_label, axis=1)
        
        # 1. Total time comparison
        ax = axes[0, 0]
        df_time = df.groupby('generator')['total_time'].mean().sort_values()
        df_time.plot(kind='barh', ax=ax)
        ax.set_xlabel('Time (seconds)')
        ax.set_title('Average Total Generation Time')
        
        # 2. Memory usage comparison
        ax = axes[0, 1]
        df_memory = df.groupby('generator')['peak_memory_mb'].mean().sort_values()
        df_memory.plot(kind='barh', ax=ax, color='orange')
        ax.set_xlabel('Memory (MB)')
        ax.set_title('Average Peak Memory Usage')
        
        # 3. Quality score comparison
        ax = axes[0, 2]
        df_quality = df.groupby('generator')['quality_score'].mean().sort_values()
        df_quality.plot(kind='barh', ax=ax, color='green')
        ax.set_xlabel('Quality Score')
        ax.set_title('Average Quality Score')
        ax.set_xlim(0, 1)
        
        # 4. Time breakdown (fit vs generate)
        ax = axes[1, 0]
        df_time_breakdown = df.groupby('generator')[['fit_time', 'generate_time']].mean()
        df_time_breakdown.plot(kind='bar', ax=ax, stacked=True)
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Time Breakdown: Fit vs Generate')
        ax.legend(['Fit', 'Generate'])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 5. Quality metrics radar chart
        ax = axes[1, 1]
        quality_metrics = ['basic_stats_score', 'distribution_score', 
                          'correlation_score', 'privacy_score', 'utility_score']
        
        # Select top 5 generators by overall quality
        top_generators = df.groupby('generator')['quality_score'].mean().nlargest(5).index
        
        angles = np.linspace(0, 2 * np.pi, len(quality_metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        ax = plt.subplot(2, 3, 5, projection='polar')
        
        for generator in top_generators:
            generator_data = df[df['generator'] == generator][quality_metrics].mean()
            values = generator_data.tolist()
            values += values[:1]
            
            ax.plot(angles, values, linewidth=2, label=generator)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['Basic\nStats', 'Distribution', 'Correlation', 
                           'Privacy', 'Utility'])
        ax.set_ylim(0, 1)
        ax.set_title('Quality Metrics Comparison\n(Top 5 Generators)', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 6. Efficiency plot (quality vs time)
        ax = axes[1, 2]
        ax.scatter(df.groupby('generator')['total_time'].mean(),
                  df.groupby('generator')['quality_score'].mean())
        
        for generator in df['generator'].unique():
            gen_data = df[df['generator'] == generator]
            ax.annotate(generator,
                       (gen_data['total_time'].mean(), gen_data['quality_score'].mean()),
                       fontsize=8, rotation=45, ha='right')
        
        ax.set_xlabel('Total Time (seconds)')
        ax.set_ylabel('Quality Score')
        ax.set_title('Efficiency: Quality vs Time')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / f"benchmark_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to: {plot_path}")
        
        plt.show()
    
    def compare_scalability(self, 
                          data: pd.DataFrame,
                          generators: List[Tuple[str, str]],
                          sizes: List[int],
                          save_plot: bool = True):
        """Compare scalability of different generators"""
        
        scalability_results = []
        
        for generator_type, model_type in generators:
            print(f"\nTesting scalability for {generator_type}/{model_type}")
            
            for size in sizes:
                if size > len(data):
                    continue
                
                sample_data = data.sample(n=size, random_state=42)
                
                result = self.benchmark_generator(
                    data=sample_data,
                    generator_type=generator_type,
                    model_type=model_type,
                    dataset_name=f"scalability_n{size}"
                )
                
                if result.error is None:
                    scalability_results.append({
                        'generator': f"{generator_type}/{model_type}",
                        'size': size,
                        'time': result.total_time,
                        'memory': result.peak_memory_mb,
                        'quality': result.quality_score
                    })
        
        # Plot scalability
        if scalability_results:
            df_scale = pd.DataFrame(scalability_results)
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Generator Scalability Comparison', fontsize=16)
            
            # Time scalability
            for generator in df_scale['generator'].unique():
                gen_data = df_scale[df_scale['generator'] == generator]
                ax1.plot(gen_data['size'], gen_data['time'], marker='o', label=generator)
            
            ax1.set_xlabel('Dataset Size (rows)')
            ax1.set_ylabel('Time (seconds)')
            ax1.set_title('Time Scalability')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Memory scalability
            for generator in df_scale['generator'].unique():
                gen_data = df_scale[df_scale['generator'] == generator]
                ax2.plot(gen_data['size'], gen_data['memory'], marker='o', label=generator)
            
            ax2.set_xlabel('Dataset Size (rows)')
            ax2.set_ylabel('Memory (MB)')
            ax2.set_title('Memory Scalability')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Quality vs size
            for generator in df_scale['generator'].unique():
                gen_data = df_scale[df_scale['generator'] == generator]
                ax3.plot(gen_data['size'], gen_data['quality'], marker='o', label=generator)
            
            ax3.set_xlabel('Dataset Size (rows)')
            ax3.set_ylabel('Quality Score')
            ax3.set_title('Quality vs Dataset Size')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1)
            
            plt.tight_layout()
            
            if save_plot:
                plot_path = self.output_dir / f"scalability_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Scalability plot saved to: {plot_path}")
            
            plt.show()


def run_standard_benchmark(data: pd.DataFrame,
                          output_dir: Optional[str] = None,
                          generators: Optional[List[Tuple[str, str]]] = None) -> GeneratorBenchmark:
    """Run a standard benchmark suite"""
    
    benchmark = GeneratorBenchmark(output_dir)
    
    # Run benchmarks
    print("Running standard benchmark suite...")
    benchmark.benchmark_all_generators(data, generators=generators)
    
    # Save results
    benchmark.save_results()
    
    # Generate report
    report = benchmark.generate_report()
    report_path = benchmark.output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to: {report_path}")
    
    # Generate plots
    benchmark.plot_results()
    
    return benchmark