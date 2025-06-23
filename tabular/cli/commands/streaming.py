"""
CLI commands for streaming operations
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from sdk.streaming import StreamingSyntheticGenerator, StreamingValidator, create_streaming_generator
from sdk.base import SyntheticDataConfig, GeneratorType, ModelType

app = typer.Typer(help="Streaming operations for large datasets")
console = Console()


@app.command()
def generate(
    input_file: Path = typer.Argument(..., help="Input CSV file path"),
    output_file: Path = typer.Argument(..., help="Output file path"),
    generator_type: str = typer.Option("sdv", help="Generator type"),
    model_type: str = typer.Option("gaussian_copula", help="Model type"),
    chunk_size: int = typer.Option(10000, help="Chunk size for processing"),
    sample_ratio: float = typer.Option(1.0, help="Sampling ratio (0-1]"),
    output_format: str = typer.Option("csv", help="Output format (csv/parquet)"),
    parallel_chunks: int = typer.Option(None, help="Number of parallel chunks"),
    config_file: Optional[Path] = typer.Option(None, help="Configuration file")
):
    """Generate synthetic data using streaming for large files"""
    
    # Validate inputs
    if not input_file.exists():
        console.print(f"[red]Error: Input file not found: {input_file}[/red]")
        raise typer.Exit(1)
    
    if output_format not in ['csv', 'parquet']:
        console.print(f"[red]Error: Invalid output format. Use 'csv' or 'parquet'[/red]")
        raise typer.Exit(1)
    
    # Load configuration
    if config_file:
        try:
            with open(config_file) as f:
                config_dict = json.load(f)
            
            generator_type = config_dict.get('generator_type', generator_type)
            model_type = config_dict.get('model_type', model_type)
            model_params = config_dict.get('model_params', {})
        except Exception as e:
            console.print(f"[red]Error loading config: {str(e)}[/red]")
            raise typer.Exit(1)
    else:
        model_params = {}
    
    # Add parallel chunks to model params
    if parallel_chunks:
        model_params['parallel_chunks'] = parallel_chunks
    
    # Create generator configuration
    try:
        config = SyntheticDataConfig(
            generator_type=GeneratorType(generator_type),
            model_type=ModelType(model_type),
            num_samples=1000,  # Per chunk
            model_params=model_params
        )
    except Exception as e:
        console.print(f"[red]Error creating configuration: {str(e)}[/red]")
        raise typer.Exit(1)
    
    # Create streaming generator
    streaming_gen = create_streaming_generator(config, chunk_size)
    
    # Generate with progress tracking
    console.print(f"[green]Starting streaming generation...[/green]")
    console.print(f"Input: {input_file}")
    console.print(f"Output: {output_file}")
    console.print(f"Chunk size: {chunk_size:,} rows")
    console.print(f"Sample ratio: {sample_ratio}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        # Estimate total chunks
        total_rows = sum(1 for _ in open(input_file)) - 1
        total_chunks = (total_rows + chunk_size - 1) // chunk_size
        
        task = progress.add_task("Processing chunks...", total=total_chunks)
        
        def update_progress(chunks_done, total):
            progress.update(task, completed=chunks_done)
        
        # Run generation
        try:
            result = streaming_gen.generate_streaming(
                input_path=str(input_file),
                output_path=str(output_file),
                sample_ratio=sample_ratio,
                output_format=output_format,
                progress_callback=update_progress
            )
            
            progress.update(task, completed=total_chunks)
            
        except Exception as e:
            console.print(f"[red]Error during generation: {str(e)}[/red]")
            raise typer.Exit(1)
    
    # Display results
    console.print(f"\n[green]✓ Generation completed successfully![/green]")
    console.print(f"Output saved to: {output_file}")
    console.print(f"Total rows generated: {result.metadata['total_rows']:,}")
    console.print(f"Chunks processed: {result.metadata['chunks_processed']}")


@app.command()
def validate(
    original_file: Path = typer.Argument(..., help="Original data file"),
    synthetic_file: Path = typer.Argument(..., help="Synthetic data file"),
    sample_size: int = typer.Option(10000, help="Sample size for validation"),
    output_file: Optional[Path] = typer.Option(None, help="Save results to file")
):
    """Validate synthetic data using streaming sampling"""
    
    # Validate inputs
    for file_path, name in [(original_file, "Original"), (synthetic_file, "Synthetic")]:
        if not file_path.exists():
            console.print(f"[red]Error: {name} file not found: {file_path}[/red]")
            raise typer.Exit(1)
    
    console.print(f"[green]Validating synthetic data...[/green]")
    console.print(f"Original: {original_file}")
    console.print(f"Synthetic: {synthetic_file}")
    console.print(f"Sample size: {sample_size:,} rows")
    
    # Create validator
    validator = StreamingValidator()
    
    with console.status("Running validation..."):
        try:
            results = validator.validate_streaming(
                original_path=str(original_file),
                synthetic_path=str(synthetic_file),
                sample_size=sample_size
            )
        except Exception as e:
            console.print(f"[red]Error during validation: {str(e)}[/red]")
            raise typer.Exit(1)
    
    # Display results
    console.print("\n[green]Validation Results:[/green]")
    
    # Statistical similarity
    if 'statistical_similarity' in results:
        table = Table(title="Statistical Similarity")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="magenta")
        
        stats = results['statistical_similarity']
        table.add_row("Overall Score", f"{stats.get('overall_score', 0):.3f}")
        
        console.print(table)
    
    # Privacy metrics
    if 'privacy_metrics' in results:
        table = Table(title="Privacy Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        privacy = results['privacy_metrics']
        for key, value in privacy.items():
            if isinstance(value, float):
                table.add_row(key.replace('_', ' ').title(), f"{value:.3f}")
        
        console.print(table)
    
    # Utility metrics
    if 'utility_metrics' in results:
        table = Table(title="Utility Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        utility = results['utility_metrics']
        for key, value in utility.items():
            if isinstance(value, float):
                table.add_row(key.replace('_', ' ').title(), f"{value:.3f}")
        
        console.print(table)
    
    # Streaming metrics
    if 'streaming_metrics' in results:
        stream_metrics = results['streaming_metrics']
        console.print(f"\n[yellow]Streaming Info:[/yellow]")
        console.print(f"Validation method: {stream_metrics['validation_method']}")
        console.print(f"Sample ratio: {stream_metrics['sample_ratio']:.2%}")
    
    # Save results if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[green]Results saved to: {output_file}[/green]")


@app.command()
def estimate_chunks(
    file_path: Path = typer.Argument(..., help="CSV file to analyze"),
    chunk_size: int = typer.Option(10000, help="Chunk size"),
    target_memory_mb: int = typer.Option(100, help="Target memory per chunk (MB)")
):
    """Estimate optimal chunk size and count for a file"""
    
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[green]Analyzing file: {file_path}[/green]")
    
    with console.status("Calculating..."):
        # Count total rows
        total_rows = sum(1 for _ in open(file_path)) - 1
        
        # Estimate optimal chunk size
        from sdk.streaming import StreamingDataProcessor
        processor = StreamingDataProcessor()
        optimal_chunk_size = processor.estimate_chunk_size(
            str(file_path), 
            target_memory_mb
        )
        
        # Calculate chunks
        chunks_with_default = (total_rows + chunk_size - 1) // chunk_size
        chunks_with_optimal = (total_rows + optimal_chunk_size - 1) // optimal_chunk_size
    
    # Display results
    table = Table(title="File Analysis")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Total Rows", f"{total_rows:,}")
    table.add_row("File Size", f"{file_path.stat().st_size / (1024*1024):.2f} MB")
    table.add_row("", "")
    table.add_row("Default Chunk Size", f"{chunk_size:,}")
    table.add_row("Default Chunks", f"{chunks_with_default:,}")
    table.add_row("", "")
    table.add_row("Optimal Chunk Size", f"{optimal_chunk_size:,}")
    table.add_row("Optimal Chunks", f"{chunks_with_optimal:,}")
    table.add_row("Target Memory/Chunk", f"{target_memory_mb} MB")
    
    console.print(table)
    
    # Recommendations
    console.print("\n[yellow]Recommendations:[/yellow]")
    if optimal_chunk_size < chunk_size:
        console.print(f"• Consider using smaller chunks ({optimal_chunk_size:,}) for memory efficiency")
    elif optimal_chunk_size > chunk_size * 2:
        console.print(f"• You can use larger chunks ({optimal_chunk_size:,}) for faster processing")
    else:
        console.print(f"• Current chunk size ({chunk_size:,}) is reasonable")


@app.command()
def monitor(
    job_id: str = typer.Argument(..., help="Streaming job ID to monitor"),
    api_url: str = typer.Option("http://localhost:8000", help="API server URL")
):
    """Monitor a running streaming job"""
    
    import requests
    import time
    
    console.print(f"[green]Monitoring job: {job_id}[/green]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("Processing...", total=100)
        
        while True:
            try:
                # Get job status
                response = requests.get(f"{api_url}/streaming/jobs/{job_id}")
                
                if response.status_code == 404:
                    console.print(f"[red]Error: Job {job_id} not found[/red]")
                    raise typer.Exit(1)
                
                status_data = response.json()
                
                # Update progress
                if 'progress' in status_data and 'percentage' in status_data['progress']:
                    progress.update(
                        task, 
                        completed=status_data['progress']['percentage'],
                        description=f"Status: {status_data['status']}"
                    )
                
                # Check completion
                if status_data['status'] == 'completed':
                    progress.update(task, completed=100)
                    console.print("\n[green]✓ Job completed successfully![/green]")
                    
                    if 'result' in status_data and status_data['result']:
                        result = status_data['result']
                        console.print(f"Output: {result.get('output_path', 'N/A')}")
                        console.print(f"Total rows: {result.get('total_rows', 'N/A'):,}")
                        
                        if 'download_url' in result:
                            console.print(f"Download URL: {api_url}{result['download_url']}")
                    break
                
                elif status_data['status'] == 'failed':
                    console.print(f"\n[red]✗ Job failed: {status_data.get('error', 'Unknown error')}[/red]")
                    raise typer.Exit(1)
                
                time.sleep(1)
                
            except requests.RequestException as e:
                console.print(f"[red]Error connecting to API: {str(e)}[/red]")
                raise typer.Exit(1)
            except KeyboardInterrupt:
                console.print("\n[yellow]Monitoring cancelled[/yellow]")
                raise typer.Exit(0)


if __name__ == "__main__":
    app()