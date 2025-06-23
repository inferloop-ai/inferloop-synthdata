"""
CLI commands for data profiling
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import typer
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box
from rich.progress import track

from sdk.profiler import DataProfiler, DatasetProfile

app = typer.Typer(help="Data profiling and analysis commands")
console = Console()


@app.command()
def analyze(
    file_path: Path = typer.Argument(..., help="Path to data file (CSV or Parquet)"),
    output: Optional[Path] = typer.Option(None, help="Save profile to file"),
    sample_size: Optional[int] = typer.Option(None, help="Sample size for large files"),
    detect_patterns: bool = typer.Option(True, help="Detect data patterns"),
    detect_distributions: bool = typer.Option(True, help="Detect statistical distributions"),
    format: str = typer.Option("table", help="Output format: table, json, report")
):
    """Analyze dataset and generate comprehensive profile"""
    
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[green]Analyzing dataset: {file_path}[/green]")
    
    # Load data
    with console.status("Loading data..."):
        try:
            if file_path.suffix.lower() == '.csv':
                if sample_size:
                    df = pd.read_csv(file_path, nrows=sample_size)
                else:
                    df = pd.read_csv(file_path)
            elif file_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(file_path)
                if sample_size and len(df) > sample_size:
                    df = df.sample(n=sample_size)
            else:
                console.print(f"[red]Error: Unsupported file format: {file_path.suffix}[/red]")
                raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error loading file: {str(e)}[/red]")
            raise typer.Exit(1)
    
    # Create profiler
    profiler = DataProfiler(
        detect_patterns=detect_patterns,
        detect_distributions=detect_distributions
    )
    
    # Generate profile
    with console.status("Profiling dataset..."):
        profile = profiler.profile_dataset(df, name=file_path.name)
    
    # Display results based on format
    if format == "table":
        display_profile_table(profile)
    elif format == "json":
        if output:
            profiler.export_profile(profile, str(output))
            console.print(f"[green]Profile saved to: {output}[/green]")
        else:
            console.print(json.dumps(profile.to_dict(), indent=2, default=str))
    elif format == "report":
        report = profiler.generate_report(profile)
        if output:
            with open(output, 'w') as f:
                f.write(report)
            console.print(f"[green]Report saved to: {output}[/green]")
        else:
            console.print(report)


@app.command()
def summary(
    file_path: Path = typer.Argument(..., help="Path to data file"),
    columns: Optional[List[str]] = typer.Option(None, help="Specific columns to profile")
):
    """Quick summary of dataset"""
    
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)
    
    # Load data
    with console.status("Loading data..."):
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            console.print(f"[red]Error: Unsupported file format[/red]")
            raise typer.Exit(1)
    
    # Filter columns if specified
    if columns:
        df = df[columns]
    
    # Display summary
    console.print(Panel.fit(f"[bold]Dataset Summary: {file_path.name}[/bold]"))
    
    # Basic info
    info_table = Table(show_header=False, box=box.SIMPLE)
    info_table.add_column("Metric", style="cyan")
    info_table.add_column("Value", style="yellow")
    
    info_table.add_row("Shape", f"{df.shape[0]:,} rows × {df.shape[1]} columns")
    info_table.add_row("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    info_table.add_row("Duplicates", f"{df.duplicated().sum():,}")
    
    console.print(info_table)
    console.print()
    
    # Column info
    col_table = Table(title="Column Information", box=box.ROUNDED)
    col_table.add_column("Column", style="cyan")
    col_table.add_column("Type", style="green")
    col_table.add_column("Non-Null", style="yellow")
    col_table.add_column("Null %", style="red")
    col_table.add_column("Unique", style="magenta")
    
    for col in df.columns:
        non_null = df[col].count()
        null_pct = (df[col].isnull().sum() / len(df)) * 100
        unique = df[col].nunique()
        
        col_table.add_row(
            col,
            str(df[col].dtype),
            f"{non_null:,}",
            f"{null_pct:.1f}%",
            f"{unique:,}"
        )
    
    console.print(col_table)


@app.command()
def compare(
    file1: Path = typer.Argument(..., help="First data file"),
    file2: Path = typer.Argument(..., help="Second data file"),
    output: Optional[Path] = typer.Option(None, help="Save comparison to file")
):
    """Compare two datasets"""
    
    for file_path in [file1, file2]:
        if not file_path.exists():
            console.print(f"[red]Error: File not found: {file_path}[/red]")
            raise typer.Exit(1)
    
    console.print(f"[green]Comparing datasets...[/green]")
    
    # Load datasets
    with console.status("Loading datasets..."):
        df1 = pd.read_csv(file1) if file1.suffix == '.csv' else pd.read_parquet(file1)
        df2 = pd.read_csv(file2) if file2.suffix == '.csv' else pd.read_parquet(file2)
    
    # Generate profiles
    profiler = DataProfiler()
    
    with console.status("Profiling first dataset..."):
        profile1 = profiler.profile_dataset(df1, name=file1.name)
    
    with console.status("Profiling second dataset..."):
        profile2 = profiler.profile_dataset(df2, name=file2.name)
    
    # Compare profiles
    comparison = profiler.compare_profiles(profile1, profile2)
    
    # Display comparison
    console.print(Panel.fit("[bold]Dataset Comparison[/bold]"))
    
    # Basic comparison
    basic_table = Table(show_header=False, box=box.SIMPLE)
    basic_table.add_column("Metric", style="cyan")
    basic_table.add_column(file1.name, style="yellow")
    basic_table.add_column(file2.name, style="green")
    
    basic_table.add_row("Shape", 
                       f"{profile1.shape[0]:,} × {profile1.shape[1]}", 
                       f"{profile2.shape[0]:,} × {profile2.shape[1]}")
    basic_table.add_row("Memory", 
                       f"{profile1.memory_usage_mb:.2f} MB", 
                       f"{profile2.memory_usage_mb:.2f} MB")
    basic_table.add_row("Columns Match", 
                       "", 
                       "✓" if comparison['columns_match'] else "✗")
    
    console.print(basic_table)
    console.print()
    
    # Column differences
    if not comparison['columns_match']:
        if comparison['profile1_only']:
            console.print(f"[yellow]Columns only in {file1.name}:[/yellow]")
            console.print(", ".join(comparison['profile1_only']))
        
        if comparison['profile2_only']:
            console.print(f"[yellow]Columns only in {file2.name}:[/yellow]")
            console.print(", ".join(comparison['profile2_only']))
    
    # Save comparison if requested
    if output:
        with open(output, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        console.print(f"\n[green]Comparison saved to: {output}[/green]")


@app.command()
def quality(
    file_path: Path = typer.Argument(..., help="Path to data file"),
    rules_file: Optional[Path] = typer.Option(None, help="Quality rules JSON file"),
    max_nulls: float = typer.Option(10.0, help="Max null percentage allowed"),
    max_duplicates: float = typer.Option(5.0, help="Max duplicate percentage allowed")
):
    """Check data quality against rules"""
    
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)
    
    # Load quality rules
    rules = {
        "max_null_percentage": max_nulls,
        "max_duplicate_percentage": max_duplicates,
        "required_columns": [],
        "column_types": {},
        "value_ranges": {}
    }
    
    if rules_file and rules_file.exists():
        with open(rules_file) as f:
            custom_rules = json.load(f)
            rules.update(custom_rules)
    
    # Load data
    with console.status("Loading data..."):
        df = pd.read_csv(file_path) if file_path.suffix == '.csv' else pd.read_parquet(file_path)
    
    # Profile data
    profiler = DataProfiler()
    profile = profiler.profile_dataset(df, name=file_path.name)
    
    # Check quality
    issues = []
    
    # Check nulls
    for col_name, col_profile in profile.column_profiles.items():
        if col_profile.null_percentage > rules["max_null_percentage"]:
            issues.append({
                "type": "High Null %",
                "column": col_name,
                "value": f"{col_profile.null_percentage:.1f}%",
                "threshold": f"{rules['max_null_percentage']}%"
            })
    
    # Check duplicates
    if profile.duplicates_percentage > rules["max_duplicate_percentage"]:
        issues.append({
            "type": "High Duplicates",
            "column": "Dataset",
            "value": f"{profile.duplicates_percentage:.1f}%",
            "threshold": f"{rules['max_duplicate_percentage']}%"
        })
    
    # Display results
    if issues:
        console.print("[red]❌ Data quality issues found:[/red]\n")
        
        issues_table = Table(box=box.ROUNDED)
        issues_table.add_column("Issue Type", style="red")
        issues_table.add_column("Column", style="yellow")
        issues_table.add_column("Value", style="cyan")
        issues_table.add_column("Threshold", style="green")
        
        for issue in issues:
            issues_table.add_row(
                issue["type"],
                issue["column"],
                issue["value"],
                issue["threshold"]
            )
        
        console.print(issues_table)
    else:
        console.print("[green]✓ All data quality checks passed![/green]")
    
    # Quality score
    quality_score = max(0, 1.0 - (len(issues) / 10.0))
    console.print(f"\n[bold]Quality Score: {quality_score:.2%}[/bold]")


@app.command()
def outliers(
    file_path: Path = typer.Argument(..., help="Path to data file"),
    columns: Optional[List[str]] = typer.Option(None, help="Columns to check for outliers"),
    method: str = typer.Option("iqr", help="Outlier detection method: iqr or zscore")
):
    """Detect outliers in numerical columns"""
    
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)
    
    # Load data
    with console.status("Loading data..."):
        df = pd.read_csv(file_path) if file_path.suffix == '.csv' else pd.read_parquet(file_path)
    
    # Select numerical columns
    if columns:
        num_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not num_cols:
        console.print("[yellow]No numerical columns found for outlier detection[/yellow]")
        return
    
    # Create profiler
    profiler = DataProfiler(outlier_method=method)
    
    # Detect outliers
    console.print(f"[green]Detecting outliers using {method.upper()} method...[/green]\n")
    
    outlier_table = Table(title="Outlier Detection Results", box=box.ROUNDED)
    outlier_table.add_column("Column", style="cyan")
    outlier_table.add_column("Outliers", style="red")
    outlier_table.add_column("Percentage", style="yellow")
    outlier_table.add_column("Min", style="green")
    outlier_table.add_column("Max", style="green")
    outlier_table.add_column("Mean", style="blue")
    
    total_outliers = 0
    
    for col in track(num_cols, description="Analyzing columns..."):
        profile = profiler.profile_column(df[col])
        
        if profile.outliers_count > 0:
            outlier_table.add_row(
                col,
                f"{profile.outliers_count:,}",
                f"{profile.outliers_percentage:.1f}%",
                f"{profile.min:.2f}" if profile.min else "N/A",
                f"{profile.max:.2f}" if profile.max else "N/A",
                f"{profile.mean:.2f}" if profile.mean else "N/A"
            )
            total_outliers += profile.outliers_count
    
    console.print(outlier_table)
    console.print(f"\n[bold]Total outliers found: {total_outliers:,}[/bold]")


def display_profile_table(profile: DatasetProfile):
    """Display profile in rich table format"""
    
    # Overview panel
    overview_text = f"""[bold]Dataset: {profile.name}[/bold]
Shape: {profile.shape[0]:,} rows × {profile.shape[1]} columns
Memory: {profile.memory_usage_mb:.2f} MB
Duplicates: {profile.duplicates_count:,} ({profile.duplicates_percentage:.1f}%)
    """
    
    console.print(Panel(overview_text, title="Overview", box=box.ROUNDED))
    
    # Column types
    types_table = Table(title="Column Types", box=box.SIMPLE)
    types_table.add_column("Type", style="cyan")
    types_table.add_column("Count", style="yellow")
    types_table.add_column("Columns", style="green")
    
    types_data = [
        ("Numerical", len(profile.numerical_columns), ", ".join(profile.numerical_columns[:3]) + ("..." if len(profile.numerical_columns) > 3 else "")),
        ("Categorical", len(profile.categorical_columns), ", ".join(profile.categorical_columns[:3]) + ("..." if len(profile.categorical_columns) > 3 else "")),
        ("Datetime", len(profile.datetime_columns), ", ".join(profile.datetime_columns[:3]) + ("..." if len(profile.datetime_columns) > 3 else "")),
        ("Constant", len(profile.constant_columns), ", ".join(profile.constant_columns[:3]) + ("..." if len(profile.constant_columns) > 3 else ""))
    ]
    
    for type_name, count, cols in types_data:
        if count > 0:
            types_table.add_row(type_name, str(count), cols)
    
    console.print(types_table)
    console.print()
    
    # Potential keys
    if profile.potential_keys:
        console.print(Panel(", ".join(profile.potential_keys), title="Potential Primary Keys", box=box.ROUNDED))
    
    # Detailed column profiles
    col_table = Table(title="Column Profiles", box=box.ROUNDED)
    col_table.add_column("Column", style="cyan")
    col_table.add_column("Type", style="green")
    col_table.add_column("Nulls", style="yellow")
    col_table.add_column("Unique", style="magenta")
    col_table.add_column("Stats", style="blue")
    
    for col_name, col_profile in list(profile.column_profiles.items())[:20]:  # Show first 20
        stats = ""
        if col_profile.mean is not None:
            stats = f"μ={col_profile.mean:.1f}, σ={col_profile.std:.1f}"
        elif col_profile.mode is not None:
            stats = f"mode={col_profile.mode}"
        
        col_table.add_row(
            col_name[:30],
            col_profile.dtype[:10],
            f"{col_profile.null_percentage:.1f}%",
            f"{col_profile.unique_percentage:.1f}%",
            stats
        )
    
    if len(profile.column_profiles) > 20:
        col_table.add_row("...", "...", "...", "...", "...")
    
    console.print(col_table)


if __name__ == "__main__":
    app()