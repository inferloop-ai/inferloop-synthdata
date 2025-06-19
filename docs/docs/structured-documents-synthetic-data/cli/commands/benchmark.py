#!/usr/bin/env python3
"""
Benchmark command for performance testing and evaluation.

Provides CLI interface for running performance benchmarks,
quality assessments, and system performance evaluations.
"""

import asyncio
import click
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ...core import get_logger
from ...quality import create_quality_engine
from ...generation import create_layout_engine
from ...privacy import create_privacy_engine
from ..utils import OutputFormatter, ProgressTracker


logger = get_logger(__name__)


@click.command(name='benchmark')
@click.option(
    '--type', '-t',
    type=click.Choice([
        'generation', 'validation', 'export', 'privacy', 
        'quality', 'end-to-end', 'all'
    ]),
    default='all',
    help='Type of benchmark to run'
)
@click.option(
    '--dataset',
    type=click.Path(exists=True, path_type=Path),
    help='Dataset path for benchmarking'
)
@click.option(
    '--iterations',
    type=int,
    default=10,
    help='Number of iterations to run'
)
@click.option(
    '--document-count',
    type=int,
    default=100,
    help='Number of documents to process'
)
@click.option(
    '--output-file',
    type=click.Path(path_type=Path),
    help='Output file for benchmark results'
)
@click.option(
    '--format',
    type=click.Choice(['json', 'csv', 'table']),
    default='table',
    help='Output format for results'
)
@click.option(
    '--compare-baseline',
    type=click.Path(exists=True, path_type=Path),
    help='Baseline results file for comparison'
)
@click.option(
    '--parallel/--sequential',
    default=True,
    help='Enable parallel processing'
)
@click.option(
    '--warm-up',
    type=int,
    default=3,
    help='Number of warm-up iterations'
)
@click.option(
    '--detailed',
    is_flag=True,
    help='Include detailed metrics'
)
@click.option(
    '--profile',
    is_flag=True,
    help='Enable performance profiling'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
def benchmark_command(
    type: str,
    dataset: Optional[Path],
    iterations: int,
    document_count: int,
    output_file: Optional[Path],
    format: str,
    compare_baseline: Optional[Path],
    parallel: bool,
    warm_up: int,
    detailed: bool,
    profile: bool,
    verbose: bool
):
    """
    Run performance benchmarks and evaluations.
    
    Examples:
        # Run generation benchmark
        synthdata benchmark -t generation --document-count 1000
        
        # Benchmark with dataset
        synthdata benchmark -t validation --dataset ./test_data/
        
        # Compare with baseline
        synthdata benchmark -t all --compare-baseline baseline.json
        
        # Detailed profiling
        synthdata benchmark -t end-to-end --profile --detailed
    """
    asyncio.run(_benchmark_async(
        type=type,
        dataset=dataset,
        iterations=iterations,
        document_count=document_count,
        output_file=output_file,
        format=format,
        compare_baseline=compare_baseline,
        parallel=parallel,
        warm_up=warm_up,
        detailed=detailed,
        profile=profile,
        verbose=verbose
    ))


async def _benchmark_async(
    type: str,
    dataset: Optional[Path],
    iterations: int,
    document_count: int,
    output_file: Optional[Path],
    format: str,
    compare_baseline: Optional[Path],
    parallel: bool,
    warm_up: int,
    detailed: bool,
    profile: bool,
    verbose: bool
):
    """Async implementation of benchmark command"""
    formatter = OutputFormatter(verbose=verbose)
    
    try:
        # Load baseline for comparison if provided
        baseline_results = None
        if compare_baseline:
            with open(compare_baseline) as f:
                baseline_results = json.load(f)
            formatter.info(f"Loaded baseline from {compare_baseline}")
        
        # Initialize profiler if requested
        if profile:
            import cProfile
            profiler = cProfile.Profile()
            profiler.enable()
        
        # Determine benchmarks to run
        benchmarks = []
        if type == 'all':
            benchmarks = [
                'generation', 'validation', 'export', 
                'privacy', 'quality', 'end-to-end'
            ]
        else:
            benchmarks = [type]
        
        formatter.info(f"Running {len(benchmarks)} benchmark(s)")
        
        # Run benchmarks
        all_results = {}
        overall_progress = ProgressTracker(
            total=len(benchmarks), 
            desc="Running benchmarks"
        )
        
        for benchmark_type in benchmarks:
            formatter.info(f"\nRunning {benchmark_type} benchmark...")
            
            # Warm-up runs
            if warm_up > 0:
                formatter.info(f"Performing {warm_up} warm-up iterations")
                for _ in range(warm_up):
                    await _run_single_benchmark(
                        benchmark_type=benchmark_type,
                        dataset=dataset,
                        document_count=min(10, document_count),
                        parallel=parallel,
                        detailed=False
                    )
            
            # Actual benchmark runs
            benchmark_results = []
            for i in range(iterations):
                result = await _run_single_benchmark(
                    benchmark_type=benchmark_type,
                    dataset=dataset,
                    document_count=document_count,
                    parallel=parallel,
                    detailed=detailed
                )
                benchmark_results.append(result)
                
                if verbose:
                    formatter.info(
                        f"Iteration {i+1}/{iterations}: "
                        f"{result['duration']:.2f}s, "
                        f"{result['throughput']:.1f} docs/s"
                    )
            
            # Aggregate results
            aggregated = _aggregate_benchmark_results(benchmark_results)
            all_results[benchmark_type] = aggregated
            overall_progress.update(1)
        
        overall_progress.close()
        
        # Stop profiler if enabled
        if profile:
            profiler.disable()
            profile_output = f"benchmark_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prof"
            profiler.dump_stats(profile_output)
            formatter.info(f"Profile saved to {profile_output}")
        
        # Format and display results
        formatted_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'iterations': iterations,
                'document_count': document_count,
                'parallel': parallel,
                'detailed': detailed
            },
            'results': all_results
        }
        
        if format == 'json':
            output_data = json.dumps(formatted_results, indent=2)
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(output_data)
                formatter.success(f"Results saved to {output_file}")
            else:
                formatter.output(output_data)
                
        elif format == 'table':
            _display_table_results(
                formatter, all_results, baseline_results
            )
            
        elif format == 'csv':
            _save_csv_results(all_results, output_file, formatter)
        
        # Summary
        formatter.success("\nBenchmark completed successfully")
        
    except KeyboardInterrupt:
        formatter.warning("\nBenchmark interrupted by user")
    except Exception as e:
        formatter.error(f"Benchmark failed: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise


async def _run_single_benchmark(
    benchmark_type: str,
    dataset: Optional[Path],
    document_count: int,
    parallel: bool,
    detailed: bool
) -> Dict:
    """Run a single benchmark iteration"""
    start_time = time.time()
    result = {
        'type': benchmark_type,
        'start_time': start_time,
        'document_count': document_count,
        'parallel': parallel
    }
    
    try:
        if benchmark_type == 'generation':
            result.update(await _benchmark_generation(
                document_count, parallel, detailed
            ))
            
        elif benchmark_type == 'validation':
            result.update(await _benchmark_validation(
                dataset, document_count, detailed
            ))
            
        elif benchmark_type == 'export':
            result.update(await _benchmark_export(
                dataset, document_count, detailed
            ))
            
        elif benchmark_type == 'privacy':
            result.update(await _benchmark_privacy(
                document_count, detailed
            ))
            
        elif benchmark_type == 'quality':
            result.update(await _benchmark_quality(
                dataset, document_count, detailed
            ))
            
        elif benchmark_type == 'end-to-end':
            result.update(await _benchmark_end_to_end(
                document_count, parallel, detailed
            ))
        
        # Calculate duration and throughput
        end_time = time.time()
        duration = end_time - start_time
        result.update({
            'end_time': end_time,
            'duration': duration,
            'throughput': document_count / duration if duration > 0 else 0
        })
        
        return result
        
    except Exception as e:
        end_time = time.time()
        result.update({
            'end_time': end_time,
            'duration': end_time - start_time,
            'error': str(e),
            'throughput': 0
        })
        return result


async def _benchmark_generation(document_count: int, parallel: bool, 
                               detailed: bool) -> Dict:
    """Benchmark document generation"""
    layout_engine = create_layout_engine()
    
    documents_generated = 0
    
    if parallel:
        tasks = [
            layout_engine.generate_document(
                domain='general',
                format='pdf'
            )
            for _ in range(document_count)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        documents_generated = sum(1 for r in results if not isinstance(r, Exception))
    else:
        for _ in range(document_count):
            try:
                await layout_engine.generate_document(
                    domain='general',
                    format='pdf'
                )
                documents_generated += 1
            except Exception:
                pass
    
    return {
        'documents_generated': documents_generated,
        'success_rate': documents_generated / document_count
    }


async def _benchmark_validation(dataset: Optional[Path], document_count: int, 
                               detailed: bool) -> Dict:
    """Benchmark document validation"""
    quality_engine = create_quality_engine()
    
    # Create mock documents for validation
    documents = [
        {'id': i, 'content': f'Mock document {i}'}
        for i in range(document_count)
    ]
    
    validations_completed = 0
    for doc in documents:
        try:
            await quality_engine.validate_document(doc['id'])
            validations_completed += 1
        except Exception:
            pass
    
    return {
        'validations_completed': validations_completed,
        'success_rate': validations_completed / document_count
    }


async def _benchmark_export(dataset: Optional[Path], document_count: int, 
                           detailed: bool) -> Dict:
    """Benchmark document export"""
    # Mock export process
    exports_completed = 0
    
    for i in range(document_count):
        try:
            # Simulate export processing
            await asyncio.sleep(0.001)  # Small delay to simulate work
            exports_completed += 1
        except Exception:
            pass
    
    return {
        'exports_completed': exports_completed,
        'success_rate': exports_completed / document_count
    }


async def _benchmark_privacy(document_count: int, detailed: bool) -> Dict:
    """Benchmark privacy protection"""
    privacy_engine = create_privacy_engine()
    
    privacy_applications = 0
    
    for i in range(document_count):
        try:
            # Mock document
            doc = {'id': i, 'content': f'Sensitive data {i}'}
            await privacy_engine.apply_protection(doc, level='medium')
            privacy_applications += 1
        except Exception:
            pass
    
    return {
        'privacy_applications': privacy_applications,
        'success_rate': privacy_applications / document_count
    }


async def _benchmark_quality(dataset: Optional[Path], document_count: int, 
                            detailed: bool) -> Dict:
    """Benchmark quality assessment"""
    quality_engine = create_quality_engine()
    
    quality_assessments = 0
    
    for i in range(document_count):
        try:
            doc_id = f'doc_{i}'
            await quality_engine.get_metrics(doc_id)
            quality_assessments += 1
        except Exception:
            pass
    
    return {
        'quality_assessments': quality_assessments,
        'success_rate': quality_assessments / document_count
    }


async def _benchmark_end_to_end(document_count: int, parallel: bool, 
                               detailed: bool) -> Dict:
    """Benchmark complete end-to-end pipeline"""
    layout_engine = create_layout_engine()
    privacy_engine = create_privacy_engine()
    quality_engine = create_quality_engine()
    
    completed_pipelines = 0
    
    for i in range(document_count):
        try:
            # Generate
            doc = await layout_engine.generate_document(
                domain='general',
                format='pdf'
            )
            
            # Apply privacy
            doc = await privacy_engine.apply_protection(doc, level='medium')
            
            # Validate quality
            await quality_engine.validate_document(doc.get('id', f'doc_{i}'))
            
            completed_pipelines += 1
        except Exception:
            pass
    
    return {
        'completed_pipelines': completed_pipelines,
        'success_rate': completed_pipelines / document_count
    }


def _aggregate_benchmark_results(results: List[Dict]) -> Dict:
    """Aggregate multiple benchmark iterations"""
    if not results:
        return {}
    
    durations = [r['duration'] for r in results if 'duration' in r]
    throughputs = [r['throughput'] for r in results if 'throughput' in r]
    
    aggregated = {
        'iterations': len(results),
        'duration': {
            'mean': sum(durations) / len(durations) if durations else 0,
            'min': min(durations) if durations else 0,
            'max': max(durations) if durations else 0
        },
        'throughput': {
            'mean': sum(throughputs) / len(throughputs) if throughputs else 0,
            'min': min(throughputs) if throughputs else 0,
            'max': max(throughputs) if throughputs else 0
        }
    }
    
    # Aggregate other metrics
    for key in results[0].keys():
        if key not in ['duration', 'throughput', 'start_time', 'end_time', 'type']:
            values = [r.get(key, 0) for r in results if isinstance(r.get(key), (int, float))]
            if values:
                aggregated[key] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values)
                }
    
    return aggregated


def _display_table_results(formatter: OutputFormatter, results: Dict, 
                          baseline: Optional[Dict]):
    """Display benchmark results in table format"""
    formatter.output("\nBenchmark Results:")
    formatter.output("=" * 80)
    
    for benchmark_type, data in results.items():
        formatter.output(f"\n{benchmark_type.upper()} BENCHMARK:")
        formatter.output("-" * 40)
        
        if 'duration' in data:
            formatter.output(
                f"Duration (avg): {data['duration']['mean']:.2f}s "
                f"(min: {data['duration']['min']:.2f}s, "
                f"max: {data['duration']['max']:.2f}s)"
            )
        
        if 'throughput' in data:
            formatter.output(
                f"Throughput (avg): {data['throughput']['mean']:.1f} docs/s "
                f"(min: {data['throughput']['min']:.1f}, "
                f"max: {data['throughput']['max']:.1f})"
            )
        
        # Display comparison with baseline if available
        if baseline and benchmark_type in baseline:
            baseline_data = baseline[benchmark_type]
            if 'throughput' in data and 'throughput' in baseline_data:
                improvement = (
                    data['throughput']['mean'] / baseline_data['throughput']['mean'] - 1
                ) * 100
                formatter.output(
                    f"vs Baseline: {improvement:+.1f}% throughput change"
                )


def _save_csv_results(results: Dict, output_file: Optional[Path], 
                     formatter: OutputFormatter):
    """Save results to CSV format"""
    import csv
    
    if not output_file:
        output_file = Path(f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Benchmark', 'Duration_Mean', 'Duration_Min', 'Duration_Max',
            'Throughput_Mean', 'Throughput_Min', 'Throughput_Max'
        ])
        
        for benchmark_type, data in results.items():
            row = [benchmark_type]
            
            if 'duration' in data:
                row.extend([
                    data['duration']['mean'],
                    data['duration']['min'],
                    data['duration']['max']
                ])
            else:
                row.extend([0, 0, 0])
            
            if 'throughput' in data:
                row.extend([
                    data['throughput']['mean'],
                    data['throughput']['min'],
                    data['throughput']['max']
                ])
            else:
                row.extend([0, 0, 0])
            
            writer.writerow(row)
    
    formatter.success(f"CSV results saved to {output_file}")


def create_benchmark_command():
    """Factory function to create benchmark command"""
    return benchmark_command


__all__ = ['benchmark_command', 'create_benchmark_command']