"""
CLI commands for performance benchmarking
"""

import time
from pathlib import Path
from typing import Optional, List
import json

import typer
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.panel import Panel
from rich import box

from sdk.benchmark import GeneratorBenchmark, run_standard_benchmark

app = typer.Typer(help="Performance benchmarking commands")
console = Console()


@app.command()
def run(
    data_file: Path = typer.Argument(..., help="Input data file for benchmarking"),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory for results"),
    generators: Optional[str] = typer.Option(None, help="Comma-separated list of generator:model pairs"),
    sizes: Optional[str] = typer.Option(None, help="Comma-separated list of dataset sizes to test"),
    num_samples: Optional[int] = typer.Option(None, help="Number of samples to generate"),
    save_plots: bool = typer.Option(True, help="Save benchmark plots"),
    quick: bool = typer.Option(False, help="Quick benchmark with fewer generators")
):
    """Run performance benchmarks on synthetic data generators"""
    
    if not data_file.exists():
        console.print(f"[red]Error: File not found: {data_file}[/red]")
        raise typer.Exit(1)
    
    # Load data
    console.print(f"[green]Loading data from: {data_file}[/green]")
    try:
        data = pd.read_csv(data_file)
        console.print(f"Loaded {len(data)} rows, {len(data.columns)} columns")
    except Exception as e:
        console.print(f"[red]Error loading data: {str(e)}[/red]")
        raise typer.Exit(1)
    
    # Parse generators
    if generators:
        generator_list = []
        for gen_spec in generators.split(','):
            if ':' in gen_spec:
                gen_type, model_type = gen_spec.split(':', 1)
                generator_list.append((gen_type.strip(), model_type.strip()))
            else:
                console.print(f"[yellow]Invalid generator spec: {gen_spec} (use format: generator:model)[/yellow]")
        
        if not generator_list:
            console.print("[red]No valid generators specified[/red]")
            raise typer.Exit(1)
    elif quick:
        # Quick benchmark with fewer generators
        generator_list = [
            ('sdv', 'gaussian_copula'),
            ('sdv', 'ctgan'),
            ('ctgan', 'ctgan'),
            ('ydata', 'wgan_gp')
        ]
    else:
        generator_list = None  # Use all generators
    
    # Create benchmark
    benchmark = GeneratorBenchmark(output_dir)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        
        if sizes:
            # Test scalability with different sizes
            size_list = [int(s.strip()) for s in sizes.split(',')]
            console.print(f"\n[cyan]Testing scalability with sizes: {size_list}[/cyan]")
            
            task = progress.add_task(
                f"Benchmarking scalability...",
                total=len(generator_list or []) * len(size_list)
            )
            
            completed = 0
            for gen_type, model_type in (generator_list or [('sdv', 'gaussian_copula')]):
                results = benchmark.benchmark_dataset_sizes(
                    data=data,
                    generator_type=gen_type,
                    model_type=model_type,
                    sizes=size_list,
                    dataset_name=data_file.stem
                )
                completed += len(size_list)
                progress.update(task, completed=completed)
            
            # Plot scalability comparison
            if save_plots and generator_list and len(generator_list) > 1:
                benchmark.compare_scalability(
                    data=data,
                    generators=generator_list,
                    sizes=size_list,
                    save_plot=True
                )
        
        else:
            # Standard benchmark
            if generator_list:
                total = len(generator_list)
            else:
                total = 9  # Default number of generators
            
            task = progress.add_task(
                f"Benchmarking {total} generators...",
                total=total
            )
            
            results = benchmark.benchmark_all_generators(
                data=data,
                dataset_name=data_file.stem,
                num_samples=num_samples,
                generators=generator_list
            )
            
            progress.update(task, completed=total)
    
    # Save results
    benchmark.save_results()
    
    # Generate and save report
    report = benchmark.generate_report()
    report_path = benchmark.output_dir / f"benchmark_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Display summary
    console.print("\n[bold]Benchmark Summary[/bold]")
    
    # Create summary table
    summary_table = Table(box=box.ROUNDED)
    summary_table.add_column("Generator", style="cyan")
    summary_table.add_column("Model", style="blue")
    summary_table.add_column("Time (s)", style="yellow")
    summary_table.add_column("Memory (MB)", style="magenta")
    summary_table.add_column("Quality", style="green")
    summary_table.add_column("Status", style="red")
    
    for result in benchmark.results:
        status = "✓" if result.error is None else "✗"
        summary_table.add_row(
            result.generator_type,
            result.model_type,
            f"{result.total_time:.2f}" if result.error is None else "-",
            f"{result.peak_memory_mb:.1f}" if result.error is None else "-",
            f"{result.quality_score:.3f}" if result.error is None else "-",
            status
        )
    
    console.print(summary_table)
    
    # Best performers
    successful_results = [r for r in benchmark.results if r.error is None]
    if successful_results:
        console.print("\n[bold]Best Performers:[/bold]")
        
        # Fastest
        fastest = min(successful_results, key=lambda r: r.total_time)
        console.print(f"  🏃 Fastest: {fastest.generator_type}/{fastest.model_type} ({fastest.total_time:.2f}s)")
        
        # Most memory efficient
        min_memory = min(successful_results, key=lambda r: r.peak_memory_mb)
        console.print(f"  💾 Most memory efficient: {min_memory.generator_type}/{min_memory.model_type} ({min_memory.peak_memory_mb:.1f} MB)")
        
        # Best quality
        best_quality = max(successful_results, key=lambda r: r.quality_score)
        console.print(f"  ⭐ Best quality: {best_quality.generator_type}/{best_quality.model_type} ({best_quality.quality_score:.3f})")
    
    # Generate plots
    if save_plots:
        console.print("\n[green]Generating benchmark plots...[/green]")
        benchmark.plot_results(save_plots=True)
    
    console.print(f"\n[green]✓ Benchmark complete![/green]")
    console.print(f"  Results saved to: {benchmark.output_dir}")
    console.print(f"  Report: {report_path}")


@app.command()
def compare(
    results_files: List[Path] = typer.Argument(..., help="Benchmark result files to compare"),
    output_file: Optional[Path] = typer.Option(None, help="Output comparison file"),
    metric: str = typer.Option("quality_score", help="Metric to compare")
):
    """Compare multiple benchmark results"""
    
    all_results = []
    
    # Load all result files
    for results_file in results_files:
        if not results_file.exists():
            console.print(f"[red]Error: File not found: {results_file}[/red]")
            continue
        
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
                
            for result in data.get('results', []):
                result['source_file'] = results_file.stem
                all_results.append(result)
                
        except Exception as e:
            console.print(f"[red]Error loading {results_file}: {str(e)}[/red]")
    
    if not all_results:
        console.print("[red]No results to compare[/red]")
        raise typer.Exit(1)
    
    # Create comparison table
    df = pd.DataFrame(all_results)
    
    # Filter successful results
    df = df[df['error'].isna()]
    
    if df.empty:
        console.print("[yellow]No successful results to compare[/yellow]")
        raise typer.Exit(1)
    
    # Create comparison table
    comparison_table = Table(title=f"Benchmark Comparison - {metric}", box=box.ROUNDED)
    comparison_table.add_column("Generator/Model", style="cyan")
    
    # Add columns for each source file
    source_files = df['source_file'].unique()
    for source in source_files:
        comparison_table.add_column(source, style="yellow")
    
    # Group by generator/model
    generators = df.apply(lambda x: f"{x['generator_type']}/{x['model_type']}", axis=1).unique()
    
    for generator in sorted(generators):
        gen_type, model_type = generator.split('/')
        row = [generator]
        
        for source in source_files:
            source_data = df[(df['source_file'] == source) & 
                           (df['generator_type'] == gen_type) & 
                           (df['model_type'] == model_type)]
            
            if not source_data.empty:
                value = source_data.iloc[0][metric]
                if isinstance(value, float):
                    row.append(f"{value:.3f}")
                else:
                    row.append(str(value))
            else:
                row.append("-")
        
        comparison_table.add_row(*row)
    
    console.print(comparison_table)
    
    # Save comparison
    if output_file:
        comparison_data = {
            'metric': metric,
            'source_files': [str(f) for f in results_files],
            'comparison': df.to_dict('records')
        }
        
        with open(output_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        console.print(f"\n[green]Comparison saved to: {output_file}[/green]")


@app.command()
def quick_test(
    rows: int = typer.Option(1000, help="Number of rows for test dataset"),
    columns: int = typer.Option(10, help="Number of columns for test dataset"),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory")
):
    """Run a quick benchmark test with synthetic test data"""
    
    console.print(f"[green]Generating test dataset ({rows} rows, {columns} columns)...[/green]")
    
    # Generate test data
    import numpy as np
    
    data = pd.DataFrame({
        f'num_{i}': np.random.randn(rows) for i in range(columns // 2)
    })
    
    # Add categorical columns
    for i in range(columns // 2):
        data[f'cat_{i}'] = np.random.choice(['A', 'B', 'C', 'D'], size=rows)
    
    # Run quick benchmark
    console.print("\n[cyan]Running quick benchmark test...[/cyan]")
    
    generators = [
        ('sdv', 'gaussian_copula'),
        ('sdv', 'ctgan'),
        ('ctgan', 'ctgan')
    ]
    
    benchmark = GeneratorBenchmark(output_dir)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Running quick test...", total=len(generators))
        
        for i, (gen_type, model_type) in enumerate(generators):
            benchmark.benchmark_generator(
                data=data,
                generator_type=gen_type,
                model_type=model_type,
                num_samples=rows,
                dataset_name="test_data"
            )
            progress.update(task, completed=i + 1)
    
    # Display results
    console.print("\n[bold]Quick Test Results:[/bold]")
    
    results_table = Table(box=box.SIMPLE)
    results_table.add_column("Generator", style="cyan")
    results_table.add_column("Time (s)", style="yellow")
    results_table.add_column("Memory (MB)", style="magenta")
    results_table.add_column("Quality", style="green")
    
    for result in benchmark.results:
        if result.error is None:
            results_table.add_row(
                f"{result.generator_type}/{result.model_type}",
                f"{result.total_time:.2f}",
                f"{result.peak_memory_mb:.1f}",
                f"{result.quality_score:.3f}"
            )
    
    console.print(results_table)
    
    # Save results
    benchmark.save_results("quick_test_results.json")
    console.print(f"\n[green]Results saved to: {benchmark.output_dir}/quick_test_results.json[/green]")


@app.command()
def report(
    results_file: Path = typer.Argument(..., help="Benchmark results file"),
    format: str = typer.Option("text", help="Report format (text, markdown, json)"),
    output_file: Optional[Path] = typer.Option(None, help="Output file for report")
):
    """Generate a report from benchmark results"""
    
    if not results_file.exists():
        console.print(f"[red]Error: File not found: {results_file}[/red]")
        raise typer.Exit(1)
    
    # Load results
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        console.print(f"[red]Error loading results: {str(e)}[/red]")
        raise typer.Exit(1)
    
    # Create benchmark object and load results
    benchmark = GeneratorBenchmark()
    
    # Convert results back to BenchmarkResult objects
    from sdk.benchmark import BenchmarkResult
    from datetime import datetime
    
    for result_data in data.get('results', []):
        result = BenchmarkResult(
            generator_type=result_data['generator_type'],
            model_type=result_data['model_type'],
            dataset_name=result_data['dataset_name'],
            dataset_rows=result_data['dataset_rows'],
            dataset_columns=result_data['dataset_columns'],
            num_samples=result_data['num_samples'],
            fit_time=result_data['fit_time'],
            generate_time=result_data['generate_time'],
            total_time=result_data['total_time'],
            peak_memory_mb=result_data['peak_memory_mb'],
            memory_delta_mb=result_data['memory_delta_mb'],
            quality_score=result_data['quality_score'],
            basic_stats_score=result_data['basic_stats_score'],
            distribution_score=result_data['distribution_score'],
            correlation_score=result_data['correlation_score'],
            privacy_score=result_data['privacy_score'],
            utility_score=result_data['utility_score'],
            cpu_count=result_data['cpu_count'],
            cpu_usage_avg=result_data['cpu_usage_avg'],
            timestamp=datetime.fromisoformat(result_data['timestamp']),
            error=result_data['error'],
            config=result_data['config']
        )
        benchmark.results.append(result)
    
    # Generate report based on format
    if format == "text":
        report_content = benchmark.generate_report()
    elif format == "markdown":
        # Generate markdown report
        report_content = generate_markdown_report(data, benchmark.results)
    elif format == "json":
        # Return formatted JSON
        report_content = json.dumps(data, indent=2)
    else:
        console.print(f"[red]Unknown format: {format}[/red]")
        raise typer.Exit(1)
    
    # Save or display report
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_content)
        console.print(f"[green]Report saved to: {output_file}[/green]")
    else:
        console.print(report_content)


def generate_markdown_report(data: dict, results: list) -> str:
    """Generate a markdown report from benchmark results"""
    
    report = []
    report.append("# Synthetic Data Generator Benchmark Report\n")
    report.append(f"Generated: {data.get('timestamp', 'Unknown')}\n")
    
    # System info
    if 'system_info' in data:
        report.append("## System Information\n")
        sys_info = data['system_info']
        report.append(f"- CPU Count: {sys_info.get('cpu_count', 'Unknown')}")
        report.append(f"- Memory: {sys_info.get('memory_total_gb', 0):.1f} GB")
        if 'platform' in sys_info:
            report.append(f"- Platform: {sys_info['platform'].get('system', 'Unknown')} "
                         f"{sys_info['platform'].get('release', '')}")
        report.append("")
    
    # Results summary
    report.append("## Results Summary\n")
    
    # Group by dataset
    from collections import defaultdict
    by_dataset = defaultdict(list)
    for result in results:
        by_dataset[result.dataset_name].append(result)
    
    for dataset_name, dataset_results in by_dataset.items():
        report.append(f"### Dataset: {dataset_name}\n")
        
        # Create results table
        report.append("| Generator | Model | Time (s) | Memory (MB) | Quality | Status |")
        report.append("|-----------|-------|----------|-------------|---------|--------|")
        
        for result in sorted(dataset_results, key=lambda r: r.total_time if r.error is None else float('inf')):
            if result.error is None:
                report.append(f"| {result.generator_type} | {result.model_type} | "
                            f"{result.total_time:.2f} | {result.peak_memory_mb:.1f} | "
                            f"{result.quality_score:.3f} | ✓ |")
            else:
                report.append(f"| {result.generator_type} | {result.model_type} | "
                            f"- | - | - | ✗ |")
        
        report.append("")
    
    # Best performers
    successful = [r for r in results if r.error is None]
    if successful:
        report.append("## Best Performers\n")
        
        fastest = min(successful, key=lambda r: r.total_time)
        report.append(f"- **Fastest**: {fastest.generator_type}/{fastest.model_type} "
                     f"({fastest.total_time:.2f}s)")
        
        min_memory = min(successful, key=lambda r: r.peak_memory_mb)
        report.append(f"- **Most Memory Efficient**: {min_memory.generator_type}/{min_memory.model_type} "
                     f"({min_memory.peak_memory_mb:.1f} MB)")
        
        best_quality = max(successful, key=lambda r: r.quality_score)
        report.append(f"- **Best Quality**: {best_quality.generator_type}/{best_quality.model_type} "
                     f"({best_quality.quality_score:.3f})")
    
    return "\n".join(report)


if __name__ == "__main__":
    app()