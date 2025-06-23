"""
Unified CLI for multi-cloud deployment
"""

import asyncio
import typer
import yaml
import json
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.syntax import Syntax

from ..common.orchestration import (
    DeploymentOrchestrator,
    DeploymentConfig,
    DeploymentState,
    ProviderFactory,
    TemplateEngine
)


app = typer.Typer(
    name="inferloop-deploy",
    help="🚀 Deploy Inferloop infrastructure across cloud providers",
    rich_markup_mode="rich"
)

console = Console()


@app.command()
def deploy(
    config_file: Path = typer.Argument(..., help="Deployment configuration file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Perform dry run without actual deployment"),
    provider: Optional[str] = typer.Option(None, "--provider", help="Override provider in config"),
    region: Optional[str] = typer.Option(None, "--region", help="Override region in config"),
    var: List[str] = typer.Option([], "--var", help="Set template variables (format: key=value)"),
    template: Optional[str] = typer.Option(None, "--template", help="Use deployment template"),
    output_format: str = typer.Option("table", "--output", "-o", help="Output format: table, json, yaml")
):
    """Deploy infrastructure from configuration file"""
    try:
        # Parse variables
        variables = {}
        for var_str in var:
            if '=' not in var_str:
                console.print(f"[red]Invalid variable format: {var_str} (expected key=value)[/red]")
                raise typer.Exit(1)
            key, value = var_str.split('=', 1)
            variables[key] = value
        
        # Load configuration
        if template:
            # Use template engine
            template_engine = TemplateEngine()
            config_data = template_engine.render_template(template, variables)
        else:
            # Load from file
            with open(config_file, 'r') as f:
                if config_file.suffix in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
        
        # Create deployment config
        config = DeploymentConfig(**config_data)
        
        # Override provider/region if specified
        if provider:
            config.provider = provider
        if region:
            config.region = region
        
        # Set dry run
        config.dry_run = dry_run
        
        # Display configuration
        console.print(Panel(
            f"[bold]Deployment Configuration[/bold]\n"
            f"Name: {config.name}\n"
            f"Version: {config.version}\n"
            f"Provider: {config.provider}\n"
            f"Region: {config.region}\n"
            f"Strategy: {config.strategy.value}\n"
            f"Resources: {len(config.resources)}",
            title="📋 Configuration"
        ))
        
        if dry_run:
            console.print("[yellow]Running in dry-run mode - no resources will be created[/yellow]")
        
        # Confirm deployment
        if not dry_run and not typer.confirm("Do you want to proceed with deployment?"):
            console.print("[yellow]Deployment cancelled[/yellow]")
            raise typer.Exit(0)
        
        # Execute deployment
        orchestrator = DeploymentOrchestrator()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Deploying infrastructure...", total=None)
            
            # Run async deployment
            status = asyncio.run(orchestrator.deploy(config))
            
            progress.update(task, completed=True)
        
        # Display results
        if output_format == "json":
            console.print_json(data={
                'deployment_id': status.deployment_id,
                'state': status.state.value,
                'resources': status.resources,
                'errors': status.errors
            })
        elif output_format == "yaml":
            console.print(yaml.dump({
                'deployment_id': status.deployment_id,
                'state': status.state.value,
                'resources': status.resources,
                'errors': status.errors
            }))
        else:
            _display_deployment_status(status)
        
        if status.state == DeploymentState.FAILED:
            console.print(f"[red]Deployment failed: {', '.join(status.errors)}[/red]")
            raise typer.Exit(1)
        
        console.print(f"[green]✓ Deployment completed successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def update(
    deployment_id: str = typer.Argument(..., help="Deployment ID to update"),
    config_file: Path = typer.Argument(..., help="Updated configuration file"),
    var: List[str] = typer.Option([], "--var", help="Set template variables (format: key=value)")
):
    """Update existing deployment"""
    try:
        # Parse variables
        variables = {}
        for var_str in var:
            key, value = var_str.split('=', 1)
            variables[key] = value
        
        # Load configuration
        with open(config_file, 'r') as f:
            if config_file.suffix in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)
        
        config = DeploymentConfig(**config_data)
        
        # Execute update
        orchestrator = DeploymentOrchestrator()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Updating deployment...", total=None)
            
            status = asyncio.run(orchestrator.update(deployment_id, config))
            
            progress.update(task, completed=True)
        
        _display_deployment_status(status)
        
        if status.state == DeploymentState.FAILED:
            console.print(f"[red]Update failed: {', '.join(status.errors)}[/red]")
            raise typer.Exit(1)
        
        console.print(f"[green]✓ Update completed successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def destroy(
    deployment_id: str = typer.Argument(..., help="Deployment ID to destroy"),
    force: bool = typer.Option(False, "--force", "-f", help="Force destroy without confirmation")
):
    """Destroy deployment and all resources"""
    try:
        orchestrator = DeploymentOrchestrator()
        
        # Get deployment status
        status = asyncio.run(orchestrator.get_status(deployment_id))
        
        console.print(Panel(
            f"[bold red]WARNING: This will destroy all resources![/bold red]\n"
            f"Deployment: {status.name} v{status.version}\n"
            f"Provider: {status.provider}\n"
            f"Region: {status.region}\n"
            f"Resources: {len(status.resources)}",
            title="⚠️  Destroy Confirmation"
        ))
        
        if not force and not typer.confirm("Are you sure you want to destroy this deployment?"):
            console.print("[yellow]Destroy operation cancelled[/yellow]")
            raise typer.Exit(0)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Destroying resources...", total=None)
            
            success = asyncio.run(orchestrator.delete(deployment_id))
            
            progress.update(task, completed=True)
        
        if success:
            console.print(f"[green]✓ Deployment destroyed successfully![/green]")
        else:
            console.print(f"[red]Failed to destroy deployment[/red]")
            raise typer.Exit(1)
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def list(
    provider: Optional[str] = typer.Option(None, "--provider", help="Filter by provider"),
    state: Optional[str] = typer.Option(None, "--state", help="Filter by state"),
    output_format: str = typer.Option("table", "--output", "-o", help="Output format: table, json, yaml")
):
    """List all deployments"""
    try:
        orchestrator = DeploymentOrchestrator()
        
        # Convert state string to enum if provided
        state_enum = None
        if state:
            try:
                state_enum = DeploymentState(state)
            except ValueError:
                console.print(f"[red]Invalid state: {state}[/red]")
                raise typer.Exit(1)
        
        deployments = asyncio.run(orchestrator.list_deployments(provider, state_enum))
        
        if output_format == "json":
            console.print_json(data=[
                {
                    'deployment_id': d.deployment_id,
                    'name': d.name,
                    'version': d.version,
                    'provider': d.provider,
                    'region': d.region,
                    'state': d.state.value,
                    'started_at': d.started_at.isoformat()
                }
                for d in deployments
            ])
        elif output_format == "yaml":
            console.print(yaml.dump([
                {
                    'deployment_id': d.deployment_id,
                    'name': d.name,
                    'version': d.version,
                    'provider': d.provider,
                    'region': d.region,
                    'state': d.state.value,
                    'started_at': d.started_at.isoformat()
                }
                for d in deployments
            ]))
        else:
            table = Table(title="Deployments")
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="magenta")
            table.add_column("Version")
            table.add_column("Provider")
            table.add_column("Region")
            table.add_column("State")
            table.add_column("Started")
            
            for deployment in deployments:
                state_style = "green" if deployment.state == DeploymentState.DEPLOYED else "yellow"
                table.add_row(
                    deployment.deployment_id,
                    deployment.name,
                    deployment.version,
                    deployment.provider,
                    deployment.region,
                    f"[{state_style}]{deployment.state.value}[/{state_style}]",
                    deployment.started_at.strftime("%Y-%m-%d %H:%M:%S")
                )
            
            console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def status(
    deployment_id: str = typer.Argument(..., help="Deployment ID"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch deployment status")
):
    """Show deployment status"""
    try:
        orchestrator = DeploymentOrchestrator()
        
        while True:
            status = asyncio.run(orchestrator.get_status(deployment_id))
            
            console.clear()
            _display_deployment_status(status)
            
            if not watch or status.state in [DeploymentState.DEPLOYED, DeploymentState.FAILED]:
                break
            
            asyncio.run(asyncio.sleep(5))
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def providers():
    """List available providers and their capabilities"""
    try:
        factory = ProviderFactory()
        
        table = Table(title="Available Providers")
        table.add_column("Provider", style="cyan")
        table.add_column("Description", style="magenta")
        table.add_column("Regions")
        table.add_column("Capabilities")
        
        for provider_name in factory.list_providers():
            info = factory.get_provider_info(provider_name)
            if info:
                capabilities = []
                for category, items in info.capabilities.items():
                    capabilities.append(f"{category}: {len(items)}")
                
                table.add_row(
                    info.name,
                    info.description,
                    f"{len(info.supported_regions)} regions",
                    ", ".join(capabilities)
                )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def templates(
    list_all: bool = typer.Option(True, "--list", "-l", help="List available templates"),
    show: Optional[str] = typer.Option(None, "--show", "-s", help="Show template details")
):
    """Manage deployment templates"""
    try:
        engine = TemplateEngine()
        
        if show:
            schema = engine.get_template_schema(show)
            
            console.print(Panel(
                f"[bold]Template: {schema['name']}[/bold]\n"
                f"Version: {schema['version']}\n"
                f"Description: {schema['description']}\n"
                f"Provider: {schema['provider']}",
                title="📄 Template Details"
            ))
            
            if schema['variables']:
                table = Table(title="Variables")
                table.add_column("Name", style="cyan")
                table.add_column("Type")
                table.add_column("Required")
                table.add_column("Default")
                table.add_column("Description")
                
                for var in schema['variables']:
                    table.add_row(
                        var['name'],
                        var['type'],
                        "Yes" if var['required'] else "No",
                        str(var.get('default', '-')),
                        var['description']
                    )
                
                console.print(table)
        
        elif list_all:
            templates = engine.list_templates()
            
            table = Table(title="Available Templates")
            table.add_column("Name", style="cyan")
            table.add_column("Version")
            table.add_column("Provider")
            table.add_column("Description")
            table.add_column("Variables")
            
            for template in templates:
                table.add_row(
                    template['name'],
                    template['version'],
                    template['provider'],
                    template['description'],
                    str(template['variables'])
                )
            
            console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def validate(
    config_file: Path = typer.Argument(..., help="Configuration file to validate"),
    template: Optional[str] = typer.Option(None, "--template", help="Template name if using template"),
    var: List[str] = typer.Option([], "--var", help="Template variables (format: key=value)")
):
    """Validate deployment configuration"""
    try:
        # Parse variables
        variables = {}
        for var_str in var:
            key, value = var_str.split('=', 1)
            variables[key] = value
        
        # Load configuration
        if template:
            engine = TemplateEngine()
            config_data = engine.render_template(template, variables)
        else:
            with open(config_file, 'r') as f:
                if config_file.suffix in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
        
        # Validate
        config = DeploymentConfig(**config_data)
        errors = config.validate()
        
        if errors:
            console.print("[red]❌ Configuration validation failed:[/red]")
            for error in errors:
                console.print(f"  • {error}")
            raise typer.Exit(1)
        else:
            console.print("[green]✓ Configuration is valid![/green]")
            
            # Display configuration summary
            console.print(Panel(
                f"Name: {config.name}\n"
                f"Version: {config.version}\n"
                f"Provider: {config.provider}\n"
                f"Region: {config.region}\n"
                f"Resources: {len(config.resources)}\n"
                f"Strategy: {config.strategy.value}",
                title="📋 Configuration Summary"
            ))
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


def _display_deployment_status(status):
    """Display deployment status in a formatted way"""
    state_color = "green" if status.state == DeploymentState.DEPLOYED else "yellow"
    if status.state == DeploymentState.FAILED:
        state_color = "red"
    
    console.print(Panel(
        f"[bold]Deployment Status[/bold]\n"
        f"ID: {status.deployment_id}\n"
        f"Name: {status.name} v{status.version}\n"
        f"State: [{state_color}]{status.state.value}[/{state_color}]\n"
        f"Provider: {status.provider}\n"
        f"Region: {status.region}\n"
        f"Started: {status.started_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Completed: {status.completed_at.strftime('%Y-%m-%d %H:%M:%S') if status.completed_at else 'In Progress'}",
        title="📊 Status"
    ))
    
    if status.resources:
        table = Table(title="Resources")
        table.add_column("Name", style="cyan")
        table.add_column("Type")
        table.add_column("ID")
        table.add_column("State")
        
        for name, resource in status.resources.items():
            table.add_row(
                name,
                resource.get('type', 'unknown'),
                resource.get('id', '-'),
                resource.get('state', 'unknown')
            )
        
        console.print(table)
    
    if status.errors:
        console.print(Panel(
            "\n".join(status.errors),
            title="❌ Errors",
            border_style="red"
        ))


if __name__ == "__main__":
    app()