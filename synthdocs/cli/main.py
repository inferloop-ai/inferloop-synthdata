#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main CLI entry point for Structured Documents Synthetic Data Generator
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich.panel import Panel
from rich import print as rprint

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from structured_docs_synth.core import (
    get_logger,
    get_config,
    list_document_types,
    get_document_type_config,
    DocumentGenerationError,
    ValidationError
)
from structured_docs_synth.generation.engines import get_template_engine, PDFGenerator, DOCXGenerator
from structured_docs_synth.delivery.api import create_app
from structured_docs_synth.privacy import PIIDetector

console = Console()


@click.group()
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--config', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def cli(ctx, debug, config):
    """Structured Documents Synthetic Data Generator CLI"""
    
    # Ensure context object exists
    ctx.ensure_object(dict)
    ctx.obj['debug'] = debug
    ctx.obj['config_file'] = config
    
    if debug:
        console.print("[yellow]Debug mode enabled[/yellow]")


@cli.command()
def version():
    """Show version information"""
    rprint(Panel.fit("Structured Documents Synthetic Data Generator\nVersion: 1.0.0", title="Version"))


@cli.command('list-types')
def list_types():
    """List available document types"""
    try:
        doc_types = list_document_types()
        
        table = Table(title="Available Document Types")
        table.add_column("Type", style="cyan")
        table.add_column("Name", style="magenta")
        table.add_column("Description", style="green")
        table.add_column("Formats", style="yellow")
        
        for doc_type in doc_types:
            config = get_document_type_config(doc_type)
            table.add_row(
                doc_type,
                config['name'],
                config['description'],
                ', '.join(config['supported_formats'])
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error listing document types: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('document_type')
def fields(document_type):
    """Show required and optional fields for a document type"""
    try:
        if document_type not in list_document_types():
            console.print(f"[red]Unknown document type: {document_type}[/red]")
            console.print("Use 'list-types' to see available types")
            sys.exit(1)
        
        config = get_document_type_config(document_type)
        
        table = Table(title=f"Fields for {config['name']}")
        table.add_column("Field", style="cyan")
        table.add_column("Required", style="magenta")
        table.add_column("Type", style="green")
        
        for field in config['required_fields']:
            table.add_row(field, "Yes", "Required")
        
        for field in config['optional_fields']:
            table.add_row(field, "No", "Optional")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error getting fields: {str(e)}[/red]")
        sys.exit(1)


@cli.command('sample-data')
@click.argument('document_type')
@click.option('--count', default=1, help='Number of sample data sets to generate')
@click.option('--output', type=click.Path(), help='Output file path (JSON)')
def sample_data(document_type, count, output):
    """Generate sample data for a document type"""
    try:
        if document_type not in list_document_types():
            console.print(f"[red]Unknown document type: {document_type}[/red]")
            sys.exit(1)
        
        template_engine = get_template_engine()
        
        with Progress() as progress:
            task = progress.add_task(f"Generating sample data for {document_type}...", total=count)
            
            sample_data_list = []
            for i in range(count):
                sample_data = template_engine.generate_sample_data(document_type)
                sample_data_list.append(sample_data)
                progress.update(task, advance=1)
        
        if output:
            import json
            with open(output, 'w') as f:
                json.dump(sample_data_list[0] if count == 1 else sample_data_list, f, indent=2)
            console.print(f"[green]Sample data saved to {output}[/green]")
        else:
            import json
            if count == 1:
                console.print(json.dumps(sample_data_list[0], indent=2))
            else:
                console.print(json.dumps(sample_data_list, indent=2))
        
    except Exception as e:
        console.print(f"[red]Error generating sample data: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('document_type')
@click.option('--data', type=click.Path(exists=True), help='JSON file with document data')
@click.option('--format', 'output_format', default='pdf', type=click.Choice(['pdf', 'docx']), help='Output format')
@click.option('--output', type=click.Path(), help='Output file path')
@click.option('--batch', is_flag=True, help='Generate batch from data array')
def generate(document_type, data, output_format, output, batch):
    """Generate documents from data"""
    try:
        if document_type not in list_document_types():
            console.print(f"[red]Unknown document type: {document_type}[/red]")
            sys.exit(1)
        
        # Load data
        if data:
            import json
            with open(data, 'r') as f:
                doc_data = json.load(f)
        else:
            # Generate sample data
            template_engine = get_template_engine()
            doc_data = template_engine.generate_sample_data(document_type)
            console.print("[yellow]No data provided, using generated sample data[/yellow]")
        
        # Initialize generator
        if output_format == 'pdf':
            generator = PDFGenerator()
        else:
            generator = DOCXGenerator()
        
        if batch and isinstance(doc_data, list):
            # Batch generation
            with Progress() as progress:
                task = progress.add_task(f"Generating {len(doc_data)} documents...", total=len(doc_data))
                
                if output_format == 'pdf':
                    file_paths = generator.generate_batch(document_type, doc_data)
                else:
                    file_paths = generator.generate_batch(document_type, doc_data)
                
                progress.update(task, advance=len(doc_data))
            
            console.print(f"[green]Generated {len(file_paths)} documents[/green]")
            for path in file_paths:
                console.print(f"  📄 {path}")
        
        else:
            # Single document generation
            if isinstance(doc_data, list):
                doc_data = doc_data[0]
            
            with console.status(f"Generating {document_type} document..."):
                if output_format == 'pdf':
                    file_path = generator.generate_pdf(
                        document_type=document_type,
                        data=doc_data,
                        output_filename=output
                    )
                else:
                    file_path = generator.generate_docx(
                        document_type=document_type,
                        data=doc_data,
                        output_filename=output
                    )
            
            console.print(f"[green]Document generated: {file_path}[/green]")
            
            # Show file info
            file_size = file_path.stat().st_size
            console.print(f"  📊 Size: {file_size:,} bytes")
        
    except ValidationError as e:
        console.print(f"[red]Validation error: {e.message}[/red]")
        if hasattr(e, 'validation_errors'):
            for error in e.validation_errors:
                console.print(f"  • {error}")
        sys.exit(1)
    except DocumentGenerationError as e:
        console.print(f"[red]Generation error: {e.message}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error generating document: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('document_type')
@click.option('--data', type=click.Path(exists=True), required=True, help='JSON file with document data')
def validate(document_type, data):
    """Validate document data without generating"""
    try:
        if document_type not in list_document_types():
            console.print(f"[red]Unknown document type: {document_type}[/red]")
            sys.exit(1)
        
        # Load data
        import json
        with open(data, 'r') as f:
            doc_data = json.load(f)
        
        # Validate
        template_engine = get_template_engine()
        validation_errors = template_engine.validate_template_data(document_type, doc_data)
        
        if validation_errors:
            console.print(f"[red]Validation failed for {document_type}:[/red]")
            for error in validation_errors:
                console.print(f"  ❌ {error}")
            sys.exit(1)
        else:
            console.print(f"[green]✓ Data is valid for {document_type}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error validating data: {str(e)}[/red]")
        sys.exit(1)


@cli.command('detect-pii')
@click.option('--data', type=click.Path(exists=True), required=True, help='JSON file with document data')
@click.option('--output', type=click.Path(), help='Output file for PII report (JSON)')
def detect_pii(data, output):
    """Detect PII in document data"""
    try:
        # Load data
        import json
        with open(data, 'r') as f:
            doc_data = json.load(f)
        
        # Initialize PII detector
        pii_detector = PIIDetector()
        
        with console.status("Scanning for PII..."):
            # Detect PII
            detection_results = pii_detector.detect_pii_in_document(doc_data)
        
        if detection_results:
            # Generate report
            report = pii_detector.generate_pii_report(detection_results)
            
            console.print(f"[red]🚨 PII DETECTED[/red]")
            console.print(f"Risk Level: [bold]{report['summary']['overall_risk_level']}[/bold]")
            console.print(f"Total Matches: {report['summary']['total_pii_matches']}")
            console.print(f"Affected Fields: {report['summary']['total_fields_with_pii']}")
            
            # Create table for PII details
            table = Table(title="PII Detection Results")
            table.add_column("Field", style="cyan")
            table.add_column("PII Type", style="magenta")
            table.add_column("Value", style="yellow")
            table.add_column("Risk", style="red")
            table.add_column("Confidence", style="green")
            
            for field_name, field_result in report['field_details'].items():
                for match in field_result['matches']:
                    table.add_row(
                        field_name,
                        match['type'],
                        match['value'],
                        field_result['risk_level'],
                        f"{match['confidence']:.2f}"
                    )
            
            console.print(table)
            
            # Show recommendations
            if report['recommendations']:
                console.print("\n💡 [bold]Recommendations:[/bold]")
                for i, rec in enumerate(report['recommendations'], 1):
                    console.print(f"   {i}. {rec}")
            
            # Save report if requested
            if output:
                with open(output, 'w') as f:
                    json.dump(report, f, indent=2)
                console.print(f"\n[green]Full report saved to {output}[/green]")
            
        else:
            console.print("[green]✅ No PII detected - data appears safe[/green]")
        
    except Exception as e:
        console.print(f"[red]Error detecting PII: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host address')
@click.option('--port', default=8000, help='Port number')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(host, port, reload):
    """Start the API server"""
    try:
        import uvicorn
        console.print(Panel(f"Starting API server at http://{host}:{port}", title="Server"))
        console.print("📖 API Documentation: http://localhost:8000/docs")
        console.print("🔍 Interactive API: http://localhost:8000/redoc")
        
        uvicorn.run(
            "structured_docs_synth.delivery.api.rest_api:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped[/yellow]")
    except Exception as e:
        console.print(f"[red]Error starting server: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
def config():
    """Show current configuration"""
    try:
        config = get_config()
        
        table = Table(title="Current Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        # Generation settings
        table.add_row("Output Directory", config.generation.output_dir)
        table.add_row("Default Document Type", config.generation.default_document_type)
        table.add_row("Max Batch Size", str(config.generation.max_documents_per_batch))
        
        # API settings  
        table.add_row("API Host", config.api.host)
        table.add_row("API Port", str(config.api.port))
        table.add_row("API Debug", str(config.api.debug))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error getting configuration: {str(e)}[/red]")
        sys.exit(1)


if __name__ == '__main__':
    cli()