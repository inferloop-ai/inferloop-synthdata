"""
CLI commands for model versioning and rollback
"""

import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import typer
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.prompt import Confirm

from sdk.versioning import get_version_manager, VersionedGenerator
from sdk.factory import GeneratorFactory
from sdk.base import SyntheticDataConfig

app = typer.Typer(help="Model versioning and rollback commands")
console = Console()


@app.command()
def save(
    model_id: str = typer.Argument(..., help="Model ID"),
    input_file: Path = typer.Argument(..., help="Training data file"),
    generator_type: str = typer.Option("sdv", help="Generator type"),
    model_type: str = typer.Option("gaussian_copula", help="Model type"),
    description: str = typer.Option("", help="Version description"),
    tags: Optional[List[str]] = typer.Option(None, help="Tags for version"),
    num_samples: int = typer.Option(1000, help="Number of samples for validation"),
    set_active: bool = typer.Option(True, help="Set as active version")
):
    """Save a new model version"""
    
    if not input_file.exists():
        console.print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[green]Training model on: {input_file}[/green]")
    
    try:
        # Load data
        data = pd.read_csv(input_file)
        
        # Create generator
        config = SyntheticDataConfig(
            generator_type=generator_type,
            model_type=model_type,
            num_samples=num_samples
        )
        generator = GeneratorFactory.create_generator(config)
        
        # Train model
        with console.status("Training model..."):
            generator.fit(data)
        
        # Calculate validation metrics
        console.print("Calculating validation metrics...")
        synthetic_data = generator.generate()
        
        # Basic metrics
        metrics = {
            'row_count': len(synthetic_data.data),
            'column_count': len(synthetic_data.data.columns),
            'training_rows': len(data),
            'quality_score': synthetic_data.metadata.get('quality_score', 0.0)
        }
        
        # Save version
        manager = get_version_manager()
        version = manager.save_model(
            generator=generator,
            data=data,
            metrics=metrics,
            tags=tags or [],
            description=description,
            set_active=set_active
        )
        
        console.print(f"\n[green]✓ Model version saved![/green]")
        console.print(f"  Model ID: {model_id}")
        console.print(f"  Version: {version.version_number}")
        console.print(f"  Version ID: {version.version_id}")
        
        if set_active:
            console.print(f"  Status: [bold]Active[/bold]")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def list(
    model_id: Optional[str] = typer.Argument(None, help="Model ID (show all if not specified)"),
    show_tags: bool = typer.Option(False, help="Show tags"),
    show_metrics: bool = typer.Option(False, help="Show metrics")
):
    """List models and versions"""
    
    manager = get_version_manager()
    
    if model_id:
        # Show versions for specific model
        try:
            versions = manager.list_versions(model_id)
            active_version = manager.get_active_version(model_id)
            
            if not versions:
                console.print(f"[yellow]No versions found for model: {model_id}[/yellow]")
                return
            
            # Create table
            table = Table(title=f"Model: {model_id}", box=box.ROUNDED)
            table.add_column("Version", style="cyan")
            table.add_column("Created", style="yellow")
            table.add_column("Generator", style="blue")
            table.add_column("Model Type", style="magenta")
            table.add_column("Status", style="green")
            
            if show_metrics:
                table.add_column("Quality Score", style="yellow")
                table.add_column("Rows", style="cyan")
            
            if show_tags:
                table.add_column("Tags", style="dim")
            
            for version in versions:
                row = [
                    str(version.version_number),
                    version.created_at.strftime("%Y-%m-%d %H:%M"),
                    version.generator_type,
                    version.model_type,
                    "[bold]Active[/bold]" if version.is_active else ""
                ]
                
                if show_metrics:
                    row.extend([
                        f"{version.metrics.get('quality_score', 0):.3f}",
                        str(version.metrics.get('row_count', 0))
                    ])
                
                if show_tags:
                    row.append(", ".join(version.tags))
                
                table.add_row(*row)
            
            console.print(table)
            
            # Show active version details
            if active_version:
                console.print(f"\n[green]Active version: {active_version.version_number}[/green]")
                if active_version.description:
                    console.print(f"Description: {active_version.description}")
            
        except ValueError as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            raise typer.Exit(1)
    
    else:
        # Show all models
        models = manager.list_models()
        
        if not models:
            console.print("[yellow]No models found[/yellow]")
            return
        
        table = Table(title="Available Models", box=box.ROUNDED)
        table.add_column("Model ID", style="cyan")
        table.add_column("Versions", style="yellow")
        table.add_column("Active", style="green")
        table.add_column("Latest", style="blue")
        table.add_column("Generator", style="magenta")
        
        for model_id in models:
            versions = manager.list_versions(model_id)
            active = manager.get_active_version(model_id)
            
            if versions:
                latest = max(v.version_number for v in versions)
                table.add_row(
                    model_id,
                    str(len(versions)),
                    str(active.version_number) if active else "-",
                    str(latest),
                    versions[0].generator_type
                )
        
        console.print(table)


@app.command()
def activate(
    model_id: str = typer.Argument(..., help="Model ID"),
    version: int = typer.Argument(..., help="Version number to activate")
):
    """Activate a specific model version"""
    
    manager = get_version_manager()
    
    try:
        # Get current active version
        current_active = manager.get_active_version(model_id)
        
        # Confirm if different version is active
        if current_active and current_active.version_number != version:
            if not Confirm.ask(
                f"Current active version is {current_active.version_number}. "
                f"Activate version {version} instead?"
            ):
                console.print("[yellow]Activation cancelled[/yellow]")
                return
        
        # Activate version
        manager.set_active_version(model_id, version)
        
        console.print(f"[green]✓ Version {version} activated for model {model_id}[/green]")
        
    except ValueError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def rollback(
    model_id: str = typer.Argument(..., help="Model ID"),
    version: int = typer.Argument(..., help="Version to rollback to")
):
    """Rollback to a previous model version"""
    
    manager = get_version_manager()
    
    try:
        # Get version info
        versions = manager.list_versions(model_id)
        target_version = next((v for v in versions if v.version_number == version), None)
        
        if not target_version:
            console.print(f"[red]Error: Version {version} not found[/red]")
            raise typer.Exit(1)
        
        # Show version details
        console.print(f"\n[bold]Rolling back to version {version}:[/bold]")
        console.print(f"  Created: {target_version.created_at.strftime('%Y-%m-%d %H:%M')}")
        console.print(f"  Generator: {target_version.generator_type}")
        console.print(f"  Model Type: {target_version.model_type}")
        
        if target_version.description:
            console.print(f"  Description: {target_version.description}")
        
        if target_version.tags:
            console.print(f"  Tags: {', '.join(target_version.tags)}")
        
        # Confirm
        if not Confirm.ask("\nProceed with rollback?"):
            console.print("[yellow]Rollback cancelled[/yellow]")
            return
        
        # Perform rollback
        manager.rollback(model_id, version)
        
        console.print(f"\n[green]✓ Successfully rolled back to version {version}[/green]")
        
    except ValueError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def compare(
    model_id: str = typer.Argument(..., help="Model ID"),
    version1: int = typer.Argument(..., help="First version"),
    version2: int = typer.Argument(..., help="Second version")
):
    """Compare two model versions"""
    
    manager = get_version_manager()
    
    try:
        comparison = manager.compare_versions(model_id, version1, version2)
        
        console.print(f"\n[bold]Comparing versions {version1} and {version2}:[/bold]\n")
        
        # Time difference
        time_diff = abs(comparison['created_at_diff'])
        days = int(time_diff // 86400)
        hours = int((time_diff % 86400) // 3600)
        
        console.print(f"Time between versions: {days} days, {hours} hours")
        
        # Configuration changes
        if comparison['config_changes']:
            console.print("\n[yellow]Configuration Changes:[/yellow]")
            for key, change in comparison['config_changes'].items():
                if 'added' in change:
                    console.print(f"  + {key}: {change['added']}")
                elif 'removed' in change:
                    console.print(f"  - {key}: {change['removed']}")
                else:
                    console.print(f"  ~ {key}: {change['old']} → {change['new']}")
        
        # Metrics comparison
        if comparison['metrics_comparison']:
            console.print("\n[cyan]Metrics Comparison:[/cyan]")
            
            metrics_table = Table(box=box.SIMPLE)
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column(f"Version {version1}", style="yellow")
            metrics_table.add_column(f"Version {version2}", style="green")
            metrics_table.add_column("Change", style="magenta")
            
            for metric, values in comparison['metrics_comparison'].items():
                v1 = values.get('v1', '-')
                v2 = values.get('v2', '-')
                
                if v1 != '-' and v2 != '-':
                    change = f"{values['diff']:+.3f} ({values['pct_change']:+.1f}%)"
                else:
                    change = '-'
                
                metrics_table.add_row(
                    metric,
                    f"{v1:.3f}" if isinstance(v1, (int, float)) else str(v1),
                    f"{v2:.3f}" if isinstance(v2, (int, float)) else str(v2),
                    change
                )
            
            console.print(metrics_table)
        
        # Tag changes
        if comparison['tags_added'] or comparison['tags_removed']:
            console.print("\n[blue]Tag Changes:[/blue]")
            for tag in comparison['tags_added']:
                console.print(f"  + {tag}")
            for tag in comparison['tags_removed']:
                console.print(f"  - {tag}")
        
    except ValueError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def delete(
    model_id: str = typer.Argument(..., help="Model ID"),
    version: int = typer.Argument(..., help="Version to delete")
):
    """Delete a model version"""
    
    manager = get_version_manager()
    
    try:
        # Get version info
        versions = manager.list_versions(model_id)
        target_version = next((v for v in versions if v.version_number == version), None)
        
        if not target_version:
            console.print(f"[red]Error: Version {version} not found[/red]")
            raise typer.Exit(1)
        
        if target_version.is_active:
            console.print("[red]Error: Cannot delete active version[/red]")
            console.print("Please activate a different version first")
            raise typer.Exit(1)
        
        # Show warning
        console.print(f"\n[yellow]Warning: This will permanently delete version {version}[/yellow]")
        console.print(f"  Created: {target_version.created_at.strftime('%Y-%m-%d %H:%M')}")
        
        if not Confirm.ask("\nAre you sure you want to delete this version?"):
            console.print("[yellow]Deletion cancelled[/yellow]")
            return
        
        # Delete version
        manager.delete_version(model_id, version)
        
        console.print(f"\n[green]✓ Version {version} deleted[/green]")
        
    except ValueError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def tag(
    model_id: str = typer.Argument(..., help="Model ID"),
    version: int = typer.Argument(..., help="Version number"),
    tags: List[str] = typer.Argument(..., help="Tags to add")
):
    """Add tags to a model version"""
    
    manager = get_version_manager()
    
    try:
        manager.tag_version(model_id, version, tags)
        
        console.print(f"[green]✓ Tags added to version {version}:[/green]")
        for tag in tags:
            console.print(f"  • {tag}")
        
    except ValueError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def export(
    model_id: str = typer.Argument(..., help="Model ID"),
    version: int = typer.Argument(..., help="Version to export"),
    output_path: Path = typer.Argument(..., help="Output file path")
):
    """Export a model version to file"""
    
    manager = get_version_manager()
    
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with console.status(f"Exporting version {version}..."):
            export_file = manager.export_version(model_id, version, str(output_path))
        
        console.print(f"[green]✓ Model exported to: {export_file}[/green]")
        
        # Show file size
        file_size = Path(export_file).stat().st_size / (1024 * 1024)
        console.print(f"  File size: {file_size:.2f} MB")
        
    except ValueError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def import_(
    model_id: str = typer.Argument(..., help="Model ID"),
    import_file: Path = typer.Argument(..., help="File to import"),
    set_active: bool = typer.Option(False, help="Set as active version")
):
    """Import a model version from file"""
    
    if not import_file.exists():
        console.print(f"[red]Error: File not found: {import_file}[/red]")
        raise typer.Exit(1)
    
    if not import_file.suffix == '.zip':
        console.print("[red]Error: Only ZIP files are supported[/red]")
        raise typer.Exit(1)
    
    manager = get_version_manager()
    
    try:
        with console.status("Importing model..."):
            version = manager.import_version(model_id, str(import_file), set_active)
        
        console.print(f"[green]✓ Model imported successfully![/green]")
        console.print(f"  Model ID: {model_id}")
        console.print(f"  Version: {version.version_number}")
        console.print(f"  Version ID: {version.version_id}")
        
        if set_active:
            console.print(f"  Status: [bold]Active[/bold]")
        
    except Exception as e:
        console.print(f"[red]Error importing model: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def generate(
    model_id: str = typer.Argument(..., help="Model ID"),
    output_file: Path = typer.Argument(..., help="Output file path"),
    version: Optional[int] = typer.Option(None, help="Version (uses active if not specified)"),
    num_samples: int = typer.Option(1000, help="Number of samples to generate")
):
    """Generate data using a versioned model"""
    
    manager = get_version_manager()
    
    try:
        # Load model
        console.print(f"[green]Loading model {model_id}...[/green]")
        
        if version:
            console.print(f"  Using version: {version}")
        else:
            active = manager.get_active_version(model_id)
            if active:
                console.print(f"  Using active version: {active.version_number}")
            else:
                console.print("  Using latest version")
        
        generator = manager.load_model(model_id, version)
        
        # Update num_samples in config
        generator.config.num_samples = num_samples
        
        # Generate data
        with console.status(f"Generating {num_samples} samples..."):
            result = generator.generate()
        
        # Save output
        output_file.parent.mkdir(parents=True, exist_ok=True)
        result.save(str(output_file))
        
        console.print(f"\n[green]✓ Synthetic data generated![/green]")
        console.print(f"  Output: {output_file}")
        console.print(f"  Samples: {len(result.data)}")
        console.print(f"  Columns: {len(result.data.columns)}")
        
        if result.metadata.get('quality_score'):
            console.print(f"  Quality Score: {result.metadata['quality_score']:.3f}")
        
    except ValueError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def cleanup(
    keep_versions: int = typer.Option(5, help="Number of versions to keep per model"),
    dry_run: bool = typer.Option(True, help="Show what would be deleted without deleting")
):
    """Clean up old model versions"""
    
    manager = get_version_manager()
    models = manager.list_models()
    
    if not models:
        console.print("[yellow]No models found[/yellow]")
        return
    
    cleanup_summary = []
    total_to_delete = 0
    
    for model_id in models:
        versions = manager.list_versions(model_id)
        active_version = manager.get_active_version(model_id)
        
        # Sort by version number descending
        versions.sort(key=lambda v: v.version_number, reverse=True)
        
        # Determine versions to delete
        versions_to_delete = []
        kept_count = 0
        
        for version in versions:
            # Always keep active version
            if version.is_active:
                continue
            
            # Keep requested number of versions
            if kept_count < keep_versions:
                kept_count += 1
                continue
            
            versions_to_delete.append(version)
        
        if versions_to_delete:
            cleanup_summary.append({
                'model_id': model_id,
                'total': len(versions),
                'to_delete': versions_to_delete,
                'to_keep': len(versions) - len(versions_to_delete)
            })
            total_to_delete += len(versions_to_delete)
    
    if not cleanup_summary:
        console.print("[green]No old versions to clean up[/green]")
        return
    
    # Display summary
    console.print(f"\n[bold]Cleanup Summary:[/bold]")
    console.print(f"Models to clean: {len(cleanup_summary)}")
    console.print(f"Versions to delete: {total_to_delete}")
    
    # Show details
    for item in cleanup_summary:
        console.print(f"\n[cyan]{item['model_id']}:[/cyan]")
        console.print(f"  Current versions: {item['total']}")
        console.print(f"  Keep: {item['to_keep']}")
        console.print(f"  Delete: {len(item['to_delete'])}")
        
        # Show versions to delete
        console.print("  Versions to delete:")
        for v in item['to_delete'][:5]:  # Show first 5
            console.print(f"    • v{v.version_number} ({v.created_at.strftime('%Y-%m-%d')})")
        if len(item['to_delete']) > 5:
            console.print(f"    • ... and {len(item['to_delete']) - 5} more")
    
    if dry_run:
        console.print("\n[yellow]Dry run mode - no deletions performed[/yellow]")
        console.print("Run without --dry-run to actually delete versions")
    else:
        # Confirm deletion
        if not Confirm.ask(f"\nDelete {total_to_delete} old versions?"):
            console.print("[yellow]Cleanup cancelled[/yellow]")
            return
        
        # Perform cleanup
        deleted = 0
        errors = 0
        
        with console.status("Cleaning up...") as status:
            for item in cleanup_summary:
                for version in item['to_delete']:
                    try:
                        manager.delete_version(item['model_id'], version.version_number)
                        deleted += 1
                        status.update(f"Deleted {deleted}/{total_to_delete} versions...")
                    except Exception:
                        errors += 1
        
        console.print(f"\n[green]✓ Cleanup completed![/green]")
        console.print(f"  Deleted: {deleted}")
        if errors:
            console.print(f"  Errors: [red]{errors}[/red]")


if __name__ == "__main__":
    app()