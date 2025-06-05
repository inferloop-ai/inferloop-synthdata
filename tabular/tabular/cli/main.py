# inferloop-synthetic/cli/main.py
"""
Inferloop Synthetic Data CLI
"""

import typer
import pandas as pd
import json
import yaml
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..sdk import GeneratorFactory, SyntheticDataConfig, SyntheticDataValidator

app = typer.Typer(
    name="inferloop-synthetic",
    help="🚀 Inferloop Synthetic Data Generation CLI",
    no_args_is_help=True
)
console = Console()


@app.command()
def generate(
    data_path: Path = typer.Argument(..., help="Path to input data file (CSV, JSON, Parquet)"),
    output_path: Path = typer.Argument(..., help="Path for output synthetic data"),
    generator_type: str = typer.Option("sdv", help="Generator type (sdv, ctgan, ydata)"),
    model_type: str = typer.Option("gaussian_copula", help="Model type"),
    num_samples: int = typer.Option(1000, help="Number of synthetic samples to generate"),
    config_file: Optional[Path] = typer.Option(None, help="Path to configuration file"),
    categorical_columns: Optional[str] = typer.Option(None, help="Comma-separated categorical columns"),
    continuous_columns: Optional[str] = typer.Option(None, help="Comma-separated continuous columns"),
    epochs: int = typer.Option(300, help="Training epochs"),
    batch_size: int = typer.Option(500, help="Batch size"),
    validate: bool = typer.Option(True, help="Validate generated data"),
    save_metadata: bool = typer.Option(True, help="Save generation metadata"),
    verbose: bool = typer.Option(False, help="Verbose output")
):
    """Generate synthetic data from input dataset"""
    
    try:
        # Load data
        console.print(f"📊 Loading data from {data_path}...")
        if data_path.suffix.lower() == '.csv':
            data = pd.read_csv(data_path)
        elif data_path.suffix.lower() == '.json':
            data = pd.read_json(data_path)
        elif data_path.suffix.lower() == '.parquet':
            data = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        console.print(f"✅ Loaded {len(data)} rows, {len(data.columns)} columns")
        
        # Create configuration
        if config_file and config_file.exists():
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    config_dict = yaml.safe_load(f)
                else:
                    config_dict = json.load(f)
            config = SyntheticDataConfig(**config_dict)
        else:
            # Parse column specifications
            cat_cols = categorical_columns.split(',') if categorical_columns else None
            cont_cols = continuous_columns.split(',') if continuous_columns else None
            
            config = SyntheticDataConfig(
                generator_type=generator_type,
                model_type=model_type,
                num_samples=num_samples,
                categorical_columns=cat_cols,
                continuous_columns=cont_cols,
                epochs=epochs,
                batch_size=batch_size
            )
        
        # Create generator
        console.print(f"🤖 Creating {generator_type} generator with {model_type} model...")
        generator = GeneratorFactory.create_generator(config)
        
        # Generate data with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            # Fit model
            fit_task = progress.add_task("🏋️ Training model...", total=None)
            generator.fit(data)
            progress.update(fit_task, completed=True)
            
            # Generate synthetic data
            gen_task = progress.add_task("🎲 Generating synthetic data...", total=None)
            result = generator.generate(num_samples)
            progress.update(gen_task, completed=True)
        
        # Save results
        console.print(f"💾 Saving synthetic data to {output_path}...")
        result.save(output_path, include_metadata=save_metadata)
        
        # Validation
        if validate:
            console.print("🔍 Validating synthetic data quality...")
            validator = SyntheticDataValidator(data, result.synthetic_data)
            validation_results = validator.validate_all()
            
            # Display validation results
            validation_table = Table(title="Validation Results")
            validation_table.add_column("Metric", style="cyan")
            validation_table.add_column("Score", style="green")
            
            validation_table.add_row("Overall Quality", f"{validation_results['overall_quality']:.3f}")
            validation_table.add_row("Basic Statistics", f"{validation_results['basic_stats']['score']:.3f}")
            validation_table.add_row("Distribution Similarity", f"{validation_results['distribution_similarity']['score']:.3f}")
            validation_table.add_row("Correlation Preservation", f"{validation_results['correlation_preservation']['score']:.3f}")
            validation_table.add_row("Privacy Score", f"{validation_results['privacy_metrics']['score']:.3f}")
            validation_table.add_row("Utility Score", f"{validation_results['utility_metrics']['score']:.3f}")
            
            console.print(validation_table)
        
        # Summary
        console.print(f"\n✨ Successfully generated {len(result.synthetic_data)} synthetic samples!")
        console.print(f"📊 Generation time: {result.generation_time:.2f} seconds")
        console.print(f"📂 Output saved to: {output_path}")
        
        if save_metadata:
            metadata_path = output_path.with_suffix('.metadata.json')
            console.print(f"📋 Metadata saved to: {metadata_path}")
    
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def validate(
    real_data_path: Path = typer.Argument(..., help="Path to real/original data"),
    synthetic_data_path: Path = typer.Argument(..., help="Path to synthetic data"),
    output_report: Optional[Path] = typer.Option(None, help="Path to save validation report"),
    detailed: bool = typer.Option(False, help="Show detailed validation results")
):
    """Validate synthetic data quality against real data"""
    
    try:
        # Load datasets
        console.print("📊 Loading datasets...")
        real_data = pd.read_csv(real_data_path)
        synthetic_data = pd.read_csv(synthetic_data_path)
        
        console.print(f"✅ Real data: {len(real_data)} rows")
        console.print(f"✅ Synthetic data: {len(synthetic_data)} rows")
        
        # Run validation
        console.print("🔍 Running validation...")
        validator = SyntheticDataValidator(real_data, synthetic_data)
        results = validator.validate_all()
        
        # Display results
        validation_table = Table(title="Validation Results", show_header=True)
        validation_table.add_column("Metric Category", style="cyan")
        validation_table.add_column("Score", style="green")
        validation_table.add_column("Status", style="yellow")
        
        categories = [
            ("Overall Quality", results['overall_quality']),
            ("Basic Statistics", results['basic_stats']['score']),
            ("Distribution Similarity", results['distribution_similarity']['score']),
            ("Correlation Preservation", results['correlation_preservation']['score']),
            ("Privacy Metrics", results['privacy_metrics']['score']),
            ("Utility Metrics", results['utility_metrics']['score'])
        ]
        
        for category, score in categories:
            status = "🟢 Excellent" if score >= 0.8 else "🟡 Good" if score >= 0.6 else "🔴 Needs Improvement"
            validation_table.add_row(category, f"{score:.3f}", status)
        
        console.print(validation_table)
        
        # Detailed results
        if detailed:
            console.print("\n📊 Detailed Results:")
            console.print(json.dumps(results, indent=2, default=str))
        
        # Save report
        if output_report:
            report_text = validator.generate_report()
            with open(output_report, 'w') as f:
                f.write(report_text)
                f.write("\n\nDetailed Results:\n")
                f.write(json.dumps(results, indent=2, default=str))
            console.print(f"📄 Report saved to: {output_report}")
    
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def info():
    """Show information about available generators and models"""
    
    # Available generators
    generators_table = Table(title="Available Generators")
    generators_table.add_column("Generator", style="cyan")
    generators_table.add_column("Models", style="green")
    generators_table.add_column("Description", style="yellow")
    
    generator_info = {
        "sdv": {
            "models": ["gaussian_copula", "ctgan", "copula_gan", "tvae"],
            "description": "Synthetic Data Vault - Comprehensive tabular data synthesis"
        },
        "ctgan": {
            "models": ["ctgan", "tvae"],
            "description": "CTGAN - Conditional Tabular GAN for tabular data"
        },
        "ydata": {
            "models": ["wgan_gp", "cramer_gan", "dragan"],
            "description": "YData Synthetic - Advanced GAN-based synthesis"
        }
    }
    
    for gen_name, info in generator_info.items():
        generators_table.add_row(
            gen_name,
            ", ".join(info["models"]),
            info["description"]
        )
    
    console.print(generators_table)
    
    # Usage examples
    console.print("\n💡 Usage Examples:")
    console.print("1. Basic generation:")
    console.print("   inferloop-synthetic generate data.csv output.csv --generator-type sdv")
    console.print("\n2. With specific model:")
    console.print("   inferloop-synthetic generate data.csv output.csv --generator-type ctgan --model-type ctgan")
    console.print("\n3. Validation:")
    console.print("   inferloop-synthetic validate real_data.csv synthetic_data.csv")


@app.command()
def create_config(
    output_path: Path = typer.Argument(..., help="Path for output configuration file"),
    generator_type: str = typer.Option("sdv", help="Generator type"),
    model_type: str = typer.Option("gaussian_copula", help="Model type"),
    interactive: bool = typer.Option(False, help="Interactive configuration creation")
):
    """Create a configuration file template"""
    
    if interactive:
        # Interactive configuration creation
        console.print("🛠️ Interactive Configuration Creation")
        generator_type = typer.prompt("Generator type", default=generator_type)
        model_type = typer.prompt("Model type", default=model_type)
        num_samples = typer.prompt("Number of samples", default=1000, type=int)
        epochs = typer.prompt("Training epochs", default=300, type=int)
        batch_size = typer.prompt("Batch size", default=500, type=int)
    else:
        num_samples = 1000
        epochs = 300
        batch_size = 500
    
    # Create configuration
    config = {
        "generator_type": generator_type,
        "model_type": model_type,
        "num_samples": num_samples,
        "epochs": epochs,
        "batch_size": batch_size,
        "categorical_columns": [],
        "continuous_columns": [],
        "hyperparameters": {},
        "validate_output": True,
        "quality_threshold": 0.8
    }
    
    # Save configuration
    with open(output_path, 'w') as f:
        if output_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        else:
            json.dump(config, f, indent=2)
    
    console.print(f"✅ Configuration saved to: {output_path}")


if __name__ == "__main__":
    app()


# 