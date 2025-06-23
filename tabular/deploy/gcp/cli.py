"""
GCP CLI Commands for Deployment
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..base import (
    ComputeResource,
    DatabaseResource,
    DeploymentConfig,
    NetworkResource,
    SecurityResource,
    StorageResource,
)
from .provider import GCPProvider
from .templates import GCPTemplates

app = typer.Typer(name="gcp", help="GCP deployment commands")
console = Console()


@app.command()
def init(
    project_id: str = typer.Option(..., help="GCP Project ID"),
    region: str = typer.Option("us-central1", help="GCP Region"),
    output_dir: Path = typer.Option(
        "./gcp-deployment", help="Output directory for templates"
    ),
    interactive: bool = typer.Option(False, help="Interactive configuration"),
):
    """Initialize GCP deployment configuration and templates"""

    console.print(f"🚀 Initializing GCP deployment for project: {project_id}")

    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create GCP provider
        provider = GCPProvider(project_id, region)

        # Validate credentials
        if not provider.validate_credentials():
            console.print(
                "❌ GCP credentials not configured. Please run 'gcloud auth login'",
                style="red",
            )
            raise typer.Exit(1)

        console.print("✅ GCP credentials validated")

        # Get configuration interactively if requested
        config = {}
        if interactive:
            config = _get_interactive_config()

        # Generate templates
        console.print("📝 Generating deployment templates...")
        GCPTemplates.save_templates(output_dir)

        # Create deployment configuration
        deployment_config = {
            "provider": "gcp",
            "project_id": project_id,
            "region": region,
            "zone": f"{region}-a",
            "config": config,
        }

        config_file = output_dir / "deployment_config.json"
        with open(config_file, "w") as f:
            json.dump(deployment_config, f, indent=2)

        console.print(f"✅ Templates and configuration saved to: {output_dir}")
        console.print("\n📋 Next steps:")
        console.print("1. Review and customize the generated templates")
        console.print(
            "2. Run 'inferloop-synthetic deploy gcp deploy' to start deployment"
        )
        console.print("3. Build and push your Docker image to GCR")

    except Exception as e:
        console.print(f"❌ Error initializing GCP deployment: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def deploy(
    config_file: Path = typer.Option(
        "./gcp-deployment/deployment_config.json", help="Deployment configuration file"
    ),
    service_type: str = typer.Option(
        "cloud_run", help="Service type (cloud_run, gke, cloud_function)"
    ),
    image: str = typer.Option(
        "gcr.io/inferloop/synthetic-api:latest", help="Docker image"
    ),
    dry_run: bool = typer.Option(
        False, help="Show what would be deployed without actually deploying"
    ),
):
    """Deploy to GCP"""

    console.print("🚀 Starting GCP deployment...")

    try:
        # Load configuration
        if not config_file.exists():
            console.print(f"❌ Configuration file not found: {config_file}", style="red")
            raise typer.Exit(1)

        with open(config_file, "r") as f:
            config = json.load(f)

        # Create provider
        provider = GCPProvider(config["project_id"], config["region"])

        # Validate credentials
        if not provider.validate_credentials():
            console.print("❌ GCP credentials not configured", style="red")
            raise typer.Exit(1)

        if dry_run:
            console.print("🔍 Dry run mode - showing deployment plan:")
            _show_deployment_plan(config, service_type, image)
            return

        # Deploy based on service type
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            if service_type == "cloud_run":
                task = progress.add_task("🔄 Deploying Cloud Run service...", total=None)
                _deploy_cloud_run(provider, config, image)
                progress.update(task, completed=True)

            elif service_type == "gke":
                task = progress.add_task("🔄 Deploying GKE cluster...", total=None)
                _deploy_gke(provider, config, image)
                progress.update(task, completed=True)

            elif service_type == "cloud_function":
                task = progress.add_task("🔄 Deploying Cloud Function...", total=None)
                _deploy_cloud_function(provider, config)
                progress.update(task, completed=True)

            else:
                console.print(
                    f"❌ Unsupported service type: {service_type}", style="red"
                )
                raise typer.Exit(1)

        console.print("✅ Deployment completed successfully!")

    except Exception as e:
        console.print(f"❌ Deployment failed: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def status(
    project_id: str = typer.Option(..., help="GCP Project ID"),
    region: str = typer.Option("us-central1", help="GCP Region"),
):
    """Show deployment status"""

    console.print("📊 Checking GCP deployment status...")

    try:
        provider = GCPProvider(project_id, region)

        # Get resource limits
        limits = provider.get_resource_limits()

        # Display status table
        status_table = Table(title="GCP Resource Status")
        status_table.add_column("Resource", style="cyan")
        status_table.add_column("Limit", style="green")
        status_table.add_column("Available", style="yellow")

        for resource, limit in limits.items():
            status_table.add_row(
                resource.replace("_", " ").title(),
                str(limit),
                "Available",  # Would need to query actual usage
            )

        console.print(status_table)

    except Exception as e:
        console.print(f"❌ Error checking status: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def cost_estimate(
    config_file: Path = typer.Option(
        "./gcp-deployment/deployment_config.json", help="Deployment configuration file"
    ),
    service_type: str = typer.Option("cloud_run", help="Service type"),
    instances: int = typer.Option(3, help="Number of instances"),
    requests_per_month: int = typer.Option(1000000, help="Expected requests per month"),
):
    """Estimate deployment costs"""

    console.print("💰 Calculating cost estimates...")

    try:
        # Load configuration
        with open(config_file, "r") as f:
            config = json.load(f)

        provider = GCPProvider(config["project_id"], config["region"])

        # Create resource specifications for cost estimation
        resources = []

        if service_type == "cloud_run":
            resources.append(
                {
                    "type": "compute",
                    "service_type": "cloud_run",
                    "cpu": 1,
                    "memory": 512,
                    "requests_per_month": requests_per_month,
                }
            )
        elif service_type == "gke":
            resources.append(
                {
                    "type": "compute",
                    "service_type": "gke",
                    "nodes": instances,
                    "machine_type": "e2-standard-4",
                }
            )

        # Add storage
        resources.append(
            {"type": "storage", "size_gb": 100, "storage_class": "standard"}
        )

        # Add database
        resources.append(
            {
                "type": "database",
                "engine": "postgres",
                "tier": "db-f1-micro",
                "storage_gb": 10,
            }
        )

        # Estimate costs
        costs = provider.estimate_costs(resources)

        # Display cost table
        cost_table = Table(title="Monthly Cost Estimate (USD)")
        cost_table.add_column("Category", style="cyan")
        cost_table.add_column("Cost", style="green")

        for category, cost in costs.items():
            if category != "total":
                cost_table.add_row(category.replace("_", " ").title(), f"${cost:.2f}")

        cost_table.add_row("Total", f"${costs['total']:.2f}", style="bold yellow")

        console.print(cost_table)
        console.print(
            "\n💡 Note: These are estimated costs and may vary based on actual usage."
        )

    except Exception as e:
        console.print(f"❌ Error calculating costs: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def destroy(
    project_id: str = typer.Option(..., help="GCP Project ID"),
    region: str = typer.Option("us-central1", help="GCP Region"),
    service_name: str = typer.Option(
        "inferloop-synthetic", help="Service name to destroy"
    ),
    confirm: bool = typer.Option(False, help="Skip confirmation prompt"),
):
    """Destroy GCP resources"""

    if not confirm:
        confirmed = typer.confirm(
            f"⚠️  This will destroy all resources for '{service_name}' in project '{project_id}'. Continue?"
        )
        if not confirmed:
            console.print("❌ Destruction cancelled")
            return

    console.print(f"💥 Destroying GCP resources for {service_name}...")

    try:
        provider = GCPProvider(project_id, region)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            # Try to delete common resources
            resources_to_delete = [
                (
                    f"projects/{project_id}/locations/{region}/services/{service_name}-api",
                    "compute",
                ),
                (
                    f"projects/{project_id}/locations/{region}/clusters/{service_name}-cluster",
                    "compute",
                ),
                (f"gs://{project_id}-{service_name}-data", "storage"),
                (f"gs://{project_id}-{service_name}-models", "storage"),
                (f"projects/{project_id}/instances/{service_name}-db", "database"),
            ]

            for resource_id, resource_type in resources_to_delete:
                task = progress.add_task(
                    f"🗑️  Deleting {resource_type} resource...", total=None
                )
                success = provider.delete_resource(resource_id, resource_type)
                if success:
                    console.print(f"✅ Deleted {resource_id}")
                else:
                    console.print(f"⚠️  Could not delete {resource_id} (may not exist)")
                progress.update(task, completed=True)

        console.print("✅ Resource destruction completed!")

    except Exception as e:
        console.print(f"❌ Error during destruction: {e}", style="red")
        raise typer.Exit(1)


def _get_interactive_config() -> Dict[str, Any]:
    """Get configuration interactively"""

    config = {}

    # Service configuration
    config["api_memory"] = typer.prompt("API Memory (MB)", default=512, type=int)
    config["api_cpu"] = typer.prompt("API CPU", default=1, type=int)
    config["api_max_instances"] = typer.prompt(
        "Max API instances", default=10, type=int
    )

    # Database configuration
    config["database_tier"] = typer.prompt("Database tier", default="db-f1-micro")
    config["database_storage"] = typer.prompt(
        "Database storage (GB)", default=10, type=int
    )

    # Storage configuration
    config["storage_class"] = typer.prompt("Storage class", default="STANDARD")

    return config


def _show_deployment_plan(config: Dict[str, Any], service_type: str, image: str):
    """Show deployment plan for dry run"""

    plan_table = Table(title="Deployment Plan")
    plan_table.add_column("Resource", style="cyan")
    plan_table.add_column("Type", style="green")
    plan_table.add_column("Configuration", style="yellow")

    if service_type == "cloud_run":
        plan_table.add_row(
            "Cloud Run Service", "Compute", f"Image: {image}, Memory: 512Mi, CPU: 1"
        )
    elif service_type == "gke":
        plan_table.add_row("GKE Cluster", "Compute", "Nodes: 3, Machine: e2-standard-4")

    plan_table.add_row("Cloud Storage", "Storage", "Buckets: data, models")

    plan_table.add_row("Cloud SQL", "Database", "Tier: db-f1-micro, Storage: 10GB")

    console.print(plan_table)


def _deploy_cloud_run(provider: GCPProvider, config: Dict[str, Any], image: str):
    """Deploy Cloud Run service"""

    compute_resource = ComputeResource(
        name="inferloop-synthetic-api",
        service_type="cloud_run",
        config={
            "image": image,
            "memory": config.get("config", {}).get("api_memory", 512),
            "cpu": config.get("config", {}).get("api_cpu", 1),
            "max_instances": config.get("config", {}).get("api_max_instances", 10),
            "environment": {
                "PROJECT_ID": config["project_id"],
                "REGION": config["region"],
            },
        },
    )

    resource_id = provider.create_compute_resource(compute_resource)
    console.print(f"✅ Cloud Run service deployed: {resource_id}")


def _deploy_gke(provider: GCPProvider, config: Dict[str, Any], image: str):
    """Deploy GKE cluster"""

    compute_resource = ComputeResource(
        name="inferloop-synthetic",
        service_type="gke",
        config={
            "nodes": 3,
            "machine_type": "e2-standard-4",
            "min_nodes": 1,
            "max_nodes": 10,
        },
    )

    resource_id = provider.create_compute_resource(compute_resource)
    console.print(f"✅ GKE cluster deployed: {resource_id}")


def _deploy_cloud_function(provider: GCPProvider, config: Dict[str, Any]):
    """Deploy Cloud Function"""

    compute_resource = ComputeResource(
        name="inferloop-synthetic-processor",
        service_type="cloud_function",
        config={
            "runtime": "python39",
            "entry_point": "process_data",
            "memory": 256,
            "source": "./functions/",
        },
    )

    resource_id = provider.create_compute_resource(compute_resource)
    console.print(f"✅ Cloud Function deployed: {resource_id}")


if __name__ == "__main__":
    app()
