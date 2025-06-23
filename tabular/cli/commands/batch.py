"""
CLI commands for batch processing
"""

import os
import json
import time
from pathlib import Path
from typing import Optional, List
import concurrent.futures

import typer
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich import box

from sdk.batch import BatchProcessor, BatchBuilder, create_batch_config_file, load_batch_from_config
from sdk.base import SyntheticDataConfig
from sdk.progress import console_progress_callback

app = typer.Typer(help="Batch processing commands")
console = Console()


@app.command()
def process(
    config_file: Path = typer.Argument(..., help="Batch configuration file"),
    max_workers: int = typer.Option(4, help="Maximum parallel workers"),
    fail_fast: bool = typer.Option(False, help="Stop on first failure"),
    cache_models: bool = typer.Option(True, help="Cache models between datasets"),
    output_report: Optional[Path] = typer.Option(None, help="Save processing report")
):
    """Process multiple datasets from configuration file"""
    
    if not config_file.exists():
        console.print(f"[red]Error: Configuration file not found: {config_file}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[green]Loading batch configuration from: {config_file}[/green]")
    
    try:
        # Load datasets from config
        datasets = load_batch_from_config(str(config_file))
        console.print(f"Found [cyan]{len(datasets)}[/cyan] datasets to process")
    except Exception as e:
        console.print(f"[red]Error loading configuration: {str(e)}[/red]")
        raise typer.Exit(1)
    
    # Create processor
    processor = BatchProcessor(
        max_workers=max_workers,
        fail_fast=fail_fast,
        cache_models=cache_models
    )
    
    # Progress tracking
    progress_info = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        # Main task
        main_task = progress.add_task(
            f"Processing {len(datasets)} datasets...",
            total=len(datasets)
        )
        
        # Dataset tasks
        dataset_tasks = {}
        for dataset in datasets:
            task_id = progress.add_task(
                f"[dim]{dataset.id}[/dim]",
                total=100,
                visible=False
            )
            dataset_tasks[dataset.id] = task_id
        
        def progress_callback(dataset_id: str, info):
            """Update progress display"""
            task_id = dataset_tasks.get(dataset_id)
            if task_id is not None:
                progress.update(
                    task_id,
                    completed=info.percentage,
                    description=f"{dataset_id}: {info.message}",
                    visible=True
                )
                
                # Update progress info
                progress_info[dataset_id] = info
                
                # Update main task
                completed = sum(
                    1 for info in progress_info.values()
                    if info.stage.value in ['completed', 'failed', 'cancelled']
                )
                progress.update(main_task, completed=completed)
        
        # Process batch
        try:
            result = processor.process_batch(datasets, progress_callback)
        except KeyboardInterrupt:
            console.print("\n[yellow]Batch processing cancelled by user[/yellow]")
            processor.cancel()
            raise typer.Exit(1)
    
    # Display results
    console.print("\n[bold]Batch Processing Results[/bold]")
    
    # Summary table
    summary_table = Table(box=box.SIMPLE)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="yellow")
    
    summary_table.add_row("Total Datasets", str(result.total_datasets))
    summary_table.add_row("Successful", f"[green]{result.successful}[/green]")
    summary_table.add_row("Failed", f"[red]{result.failed}[/red]")
    summary_table.add_row("Processing Time", f"{result.processing_time:.2f}s")
    summary_table.add_row("Status", f"[bold]{result.status.value}[/bold]")
    
    console.print(summary_table)
    
    # Errors table if any
    if result.errors:
        console.print("\n[red]Errors:[/red]")
        error_table = Table(box=box.ROUNDED)
        error_table.add_column("Dataset", style="cyan")
        error_table.add_column("Error", style="red")
        
        for dataset_id, error in result.errors.items():
            error_table.add_row(dataset_id, error)
        
        console.print(error_table)
    
    # Save report if requested
    if output_report:
        report_data = {
            'summary': result.to_dict(),
            'datasets': [
                {
                    'id': dataset.id,
                    'input': dataset.input_path,
                    'output': dataset.output_path,
                    'success': dataset.id not in result.errors,
                    'error': result.errors.get(dataset.id)
                }
                for dataset in datasets
            ]
        }
        
        with open(output_report, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        console.print(f"\n[green]Report saved to: {output_report}[/green]")


@app.command()
def create_config(
    output_file: Path = typer.Argument(..., help="Output configuration file"),
    input_dir: Optional[Path] = typer.Option(None, help="Input directory to scan"),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory for results"),
    pattern: str = typer.Option("*.csv", help="File pattern to match"),
    generator_type: str = typer.Option("sdv", help="Default generator type"),
    model_type: str = typer.Option("gaussian_copula", help="Default model type"),
    num_samples: int = typer.Option(1000, help="Default number of samples")
):
    """Create batch configuration file"""
    
    datasets = []
    
    if input_dir and output_dir:
        # Scan directory for files
        if not input_dir.exists():
            console.print(f"[red]Error: Input directory not found: {input_dir}[/red]")
            raise typer.Exit(1)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(f"[green]Scanning {input_dir} for {pattern} files...[/green]")
        
        files = list(input_dir.glob(pattern))
        
        if not files:
            console.print(f"[yellow]No files found matching pattern: {pattern}[/yellow]")
        else:
            console.print(f"Found [cyan]{len(files)}[/cyan] files")
            
            for i, file_path in enumerate(files):
                output_path = output_dir / file_path.name
                
                datasets.append({
                    'id': file_path.stem,
                    'input_path': str(file_path),
                    'output_path': str(output_path),
                    'priority': 0,
                    'config': {}
                })
    
    # Create configuration
    default_config = {
        'generator_type': generator_type,
        'model_type': model_type,
        'num_samples': num_samples
    }
    
    create_batch_config_file(
        str(output_file),
        datasets,
        default_config
    )
    
    console.print(f"\n[green]✓ Configuration file created: {output_file}[/green]")
    console.print(f"  Default generator: {generator_type}")
    console.print(f"  Default model: {model_type}")
    console.print(f"  Datasets: {len(datasets)}")
    
    if not datasets:
        console.print("\n[yellow]Note: No datasets added. Edit the configuration file to add datasets.[/yellow]")


@app.command()
def validate(
    config_file: Path = typer.Argument(..., help="Batch configuration file to validate")
):
    """Validate batch configuration file"""
    
    if not config_file.exists():
        console.print(f"[red]Error: Configuration file not found: {config_file}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[green]Validating configuration: {config_file}[/green]")
    
    try:
        # Load configuration
        datasets = load_batch_from_config(str(config_file))
        
        # Validation results
        issues = []
        warnings = []
        
        # Check each dataset
        for dataset in datasets:
            # Check input file exists
            if not Path(dataset.input_path).exists():
                issues.append(f"Dataset '{dataset.id}': Input file not found: {dataset.input_path}")
            
            # Check output directory exists
            output_dir = Path(dataset.output_path).parent
            if not output_dir.exists():
                warnings.append(f"Dataset '{dataset.id}': Output directory does not exist: {output_dir}")
            
            # Validate configuration
            try:
                if isinstance(dataset.config, dict):
                    SyntheticDataConfig(**dataset.config)
            except Exception as e:
                issues.append(f"Dataset '{dataset.id}': Invalid configuration: {str(e)}")
        
        # Display results
        if not issues and not warnings:
            console.print("[green]✓ Configuration is valid![/green]")
            console.print(f"  Datasets: {len(datasets)}")
        else:
            if issues:
                console.print("\n[red]Issues found:[/red]")
                for issue in issues:
                    console.print(f"  • {issue}")
            
            if warnings:
                console.print("\n[yellow]Warnings:[/yellow]")
                for warning in warnings:
                    console.print(f"  • {warning}")
            
            if issues:
                raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"[red]Error loading configuration: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def quick(
    input_files: List[Path] = typer.Argument(..., help="Input CSV files"),
    output_dir: Path = typer.Option("./synthetic", help="Output directory"),
    generator_type: str = typer.Option("sdv", help="Generator type"),
    model_type: str = typer.Option("gaussian_copula", help="Model type"),
    num_samples: int = typer.Option(1000, help="Number of samples per dataset"),
    max_workers: int = typer.Option(4, help="Maximum parallel workers")
):
    """Quick batch processing without configuration file"""
    
    # Validate inputs
    for file_path in input_files:
        if not file_path.exists():
            console.print(f"[red]Error: File not found: {file_path}[/red]")
            raise typer.Exit(1)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[green]Processing {len(input_files)} files...[/green]")
    
    # Build batch
    builder = BatchBuilder()
    builder.set_default_config({
        'generator_type': generator_type,
        'model_type': model_type,
        'num_samples': num_samples
    })
    
    for file_path in input_files:
        output_path = output_dir / f"synthetic_{file_path.name}"
        builder.add_dataset(
            input_path=str(file_path),
            output_path=str(output_path),
            dataset_id=file_path.stem
        )
    
    datasets = builder.build()
    
    # Process batch
    processor = BatchProcessor(max_workers=max_workers)
    
    # Simple progress display
    console.print("\nProcessing:")
    with console.status("Working...") as status:
        completed = 0
        
        def progress_callback(dataset_id: str, info):
            nonlocal completed
            status.update(f"Processing {dataset_id}: {info.message}")
            
            if info.stage.value in ['completed', 'failed']:
                completed += 1
                console.print(f"  [{completed}/{len(datasets)}] {dataset_id}: {info.stage.value}")
        
        result = processor.process_batch(datasets, progress_callback)
    
    # Summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Successful: [green]{result.successful}[/green]")
    console.print(f"  Failed: [red]{result.failed}[/red]")
    console.print(f"  Time: {result.processing_time:.2f}s")
    console.print(f"  Output: {output_dir}")


@app.command()
def directory(
    input_dir: Path = typer.Argument(..., help="Input directory"),
    output_dir: Path = typer.Argument(..., help="Output directory"),
    pattern: str = typer.Option("*.csv", help="File pattern"),
    recursive: bool = typer.Option(False, help="Search recursively"),
    generator_type: str = typer.Option("sdv", help="Generator type"),
    model_type: str = typer.Option("gaussian_copula", help="Model type"),
    num_samples: int = typer.Option(1000, help="Number of samples"),
    max_workers: int = typer.Option(4, help="Maximum parallel workers")
):
    """Process all files in a directory"""
    
    if not input_dir.exists():
        console.print(f"[red]Error: Input directory not found: {input_dir}[/red]")
        raise typer.Exit(1)
    
    # Create builder
    builder = BatchBuilder()
    builder.set_default_config({
        'generator_type': generator_type,
        'model_type': model_type,
        'num_samples': num_samples
    })
    
    # Add directory
    builder.add_directory(
        str(input_dir),
        str(output_dir),
        pattern=pattern,
        recursive=recursive
    )
    
    datasets = builder.build()
    
    if not datasets:
        console.print(f"[yellow]No files found matching pattern: {pattern}[/yellow]")
        raise typer.Exit(0)
    
    console.print(f"[green]Found {len(datasets)} files to process[/green]")
    
    # Show files
    file_table = Table(title="Files to Process", box=box.SIMPLE)
    file_table.add_column("Dataset ID", style="cyan")
    file_table.add_column("Input File", style="yellow")
    
    for dataset in datasets[:10]:  # Show first 10
        file_table.add_row(dataset.id, Path(dataset.input_path).name)
    
    if len(datasets) > 10:
        file_table.add_row("...", f"... and {len(datasets) - 10} more")
    
    console.print(file_table)
    
    # Confirm
    if not typer.confirm("\nProceed with batch processing?"):
        console.print("[yellow]Cancelled[/yellow]")
        raise typer.Exit(0)
    
    # Process
    processor = BatchProcessor(max_workers=max_workers, cache_models=True)
    
    with console.status("Processing...") as status:
        completed = 0
        
        def progress_callback(dataset_id: str, info):
            nonlocal completed
            if info.stage.value in ['completed', 'failed']:
                completed += 1
                status.update(f"Processing... [{completed}/{len(datasets)}]")
        
        result = processor.process_batch(datasets, progress_callback)
    
    # Results
    console.print(f"\n[bold]Batch processing completed![/bold]")
    console.print(f"  Total: {result.total_datasets}")
    console.print(f"  Successful: [green]{result.successful}[/green]")
    console.print(f"  Failed: [red]{result.failed}[/red]")
    console.print(f"  Time: {result.processing_time:.2f}s")
    console.print(f"  Output: {output_dir}")


if __name__ == "__main__":
    app()