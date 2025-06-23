"""
Azure CLI Commands for Deployment
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
from .provider import AzureProvider
from .templates import AzureTemplates

app = typer.Typer(name="azure", help="Azure deployment commands")
console = Console()


@app.command()
def init(
    subscription_id: str = typer.Option(..., help="Azure Subscription ID"),
    resource_group: str = typer.Option("inferloop-rg", help="Resource Group Name"),
    location: str = typer.Option("eastus", help="Azure Location"),
    output_dir: Path = typer.Option(
        "./azure-deployment", help="Output directory for templates"
    ),
    interactive: bool = typer.Option(False, help="Interactive configuration"),
):
    """Initialize Azure deployment configuration and templates"""

    console.print(
        f"🚀 Initializing Azure deployment for subscription: {subscription_id}"
    )

    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create Azure provider
        provider = AzureProvider(subscription_id, resource_group, location)

        # Validate credentials
        if not provider.validate_credentials():
            console.print(
                "❌ Azure credentials not configured. Please run 'az login'", style="red"
            )
            raise typer.Exit(1)

        console.print("✅ Azure credentials validated")

        # Get configuration interactively if requested
        config = {}
        if interactive:
            config = _get_interactive_config()

        # Generate templates
        console.print("📝 Generating deployment templates...")
        AzureTemplates.save_templates(output_dir)

        # Create deployment configuration
        deployment_config = {
            "provider": "azure",
            "subscription_id": subscription_id,
            "resource_group": resource_group,
            "location": location,
            "config": config,
        }

        config_file = output_dir / "deployment_config.json"
        with open(config_file, "w") as f:
            json.dump(deployment_config, f, indent=2)

        console.print(f"✅ Templates and configuration saved to: {output_dir}")
        console.print("\n📋 Next steps:")
        console.print("1. Review and customize the generated templates")
        console.print(
            "2. Run 'inferloop-synthetic deploy azure deploy' to start deployment"
        )
        console.print("3. Build and push your Docker image to Azure Container Registry")

    except Exception as e:
        console.print(f"❌ Error initializing Azure deployment: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def deploy(
    config_file: Path = typer.Option(
        "./azure-deployment/deployment_config.json",
        help="Deployment configuration file",
    ),
    service_type: str = typer.Option(
        "container_instance", help="Service type (container_instance, aks, function)"
    ),
    image: str = typer.Option(
        "mcr.microsoft.com/azuredocs/aci-helloworld", help="Docker image"
    ),
    dry_run: bool = typer.Option(
        False, help="Show what would be deployed without actually deploying"
    ),
):
    """Deploy to Azure"""

    console.print("🚀 Starting Azure deployment...")

    try:
        # Load configuration
        if not config_file.exists():
            console.print(f"❌ Configuration file not found: {config_file}", style="red")
            raise typer.Exit(1)

        with open(config_file, "r") as f:
            config = json.load(f)

        # Create provider
        provider = AzureProvider(
            config["subscription_id"], config["resource_group"], config["location"]
        )

        # Validate credentials
        if not provider.validate_credentials():
            console.print("❌ Azure credentials not configured", style="red")
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

            if service_type == "container_instance":
                task = progress.add_task(
                    "🔄 Deploying Container Instance...", total=None
                )
                _deploy_container_instance(provider, config, image)
                progress.update(task, completed=True)

            elif service_type == "aks":
                task = progress.add_task("🔄 Deploying AKS cluster...", total=None)
                _deploy_aks(provider, config, image)
                progress.update(task, completed=True)

            elif service_type == "function":
                task = progress.add_task("🔄 Deploying Azure Function...", total=None)
                _deploy_azure_function(provider, config)
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
    subscription_id: str = typer.Option(..., help="Azure Subscription ID"),
    resource_group: str = typer.Option("inferloop-rg", help="Resource Group Name"),
    location: str = typer.Option("eastus", help="Azure Location"),
):
    """Show deployment status"""

    console.print("📊 Checking Azure deployment status...")

    try:
        provider = AzureProvider(subscription_id, resource_group, location)

        # Get resource limits
        limits = provider.get_resource_limits()

        # Display status table
        status_table = Table(title="Azure Resource Status")
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
        "./azure-deployment/deployment_config.json",
        help="Deployment configuration file",
    ),
    service_type: str = typer.Option("container_instance", help="Service type"),
    instances: int = typer.Option(3, help="Number of instances"),
    hours_per_month: int = typer.Option(730, help="Expected hours per month"),
):
    """Estimate deployment costs"""

    console.print("💰 Calculating cost estimates...")

    try:
        # Load configuration
        with open(config_file, "r") as f:
            config = json.load(f)

        provider = AzureProvider(
            config["subscription_id"], config["resource_group"], config["location"]
        )

        # Create resource specifications for cost estimation
        resources = []

        if service_type == "container_instance":
            resources.append(
                {
                    "type": "compute",
                    "service_type": "container_instance",
                    "cpu": 1,
                    "memory": 1,
                    "hours_per_month": hours_per_month,
                }
            )
        elif service_type == "aks":
            resources.append(
                {
                    "type": "compute",
                    "service_type": "aks",
                    "nodes": instances,
                    "vm_size": "Standard_DS2_v2",
                }
            )
        elif service_type == "function":
            resources.append(
                {
                    "type": "compute",
                    "service_type": "function",
                    "executions_per_month": 1000000,
                    "execution_time_ms": 1000,
                    "memory_mb": 128,
                }
            )

        # Add storage
        resources.append({"type": "storage", "size_gb": 100, "tier": "hot"})

        # Add database
        resources.append({"type": "database", "engine": "sqlserver", "tier": "Basic"})

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
    subscription_id: str = typer.Option(..., help="Azure Subscription ID"),
    resource_group: str = typer.Option("inferloop-rg", help="Resource Group Name"),
    location: str = typer.Option("eastus", help="Azure Location"),
    service_name: str = typer.Option(
        "inferloop-synthetic", help="Service name to destroy"
    ),
    confirm: bool = typer.Option(False, help="Skip confirmation prompt"),
):
    """Destroy Azure resources"""

    if not confirm:
        confirmed = typer.confirm(
            f"⚠️  This will destroy all resources for '{service_name}' in resource group '{resource_group}'. Continue?"
        )
        if not confirmed:
            console.print("❌ Destruction cancelled")
            return

    console.print(f"💥 Destroying Azure resources for {service_name}...")

    try:
        provider = AzureProvider(subscription_id, resource_group, location)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            # Try to delete common resources
            resources_to_delete = [
                (
                    f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.ContainerInstance/containerGroups/{service_name}-api",
                    "compute",
                ),
                (
                    f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.ContainerService/managedClusters/{service_name}-aks",
                    "compute",
                ),
                (
                    f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Storage/storageAccounts/{service_name}storage",
                    "storage",
                ),
                (
                    f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Sql/servers/{service_name}-sql",
                    "database",
                ),
            ]

            for resource_id, resource_type in resources_to_delete:
                task = progress.add_task(
                    f"🗑️  Deleting {resource_type} resource...", total=None
                )
                success = provider.delete_resource(resource_id, resource_type)
                if success:
                    console.print(f"✅ Deleted {resource_id.split('/')[-1]}")
                else:
                    console.print(
                        f"⚠️  Could not delete {resource_id.split('/')[-1]} (may not exist)"
                    )
                progress.update(task, completed=True)

        console.print("✅ Resource destruction completed!")

    except Exception as e:
        console.print(f"❌ Error during destruction: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def logs(
    subscription_id: str = typer.Option(..., help="Azure Subscription ID"),
    resource_group: str = typer.Option("inferloop-rg", help="Resource Group Name"),
    resource_name: str = typer.Option(..., help="Resource name to get logs from"),
    lines: int = typer.Option(100, help="Number of log lines to retrieve"),
):
    """Get logs from Azure resources"""

    console.print(f"📋 Retrieving logs from {resource_name}...")

    try:
        import subprocess

        # Get Container Instance logs
        cmd = [
            "az",
            "container",
            "logs",
            "--resource-group",
            resource_group,
            "--name",
            resource_name,
            "--tail",
            str(lines),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            console.print("\n📄 Logs:")
            console.print(result.stdout)
        else:
            console.print(f"❌ Failed to retrieve logs: {result.stderr}", style="red")

    except Exception as e:
        console.print(f"❌ Error retrieving logs: {e}", style="red")
        raise typer.Exit(1)


def _get_interactive_config() -> Dict[str, Any]:
    """Get configuration interactively"""

    config = {}

    # Service configuration
    config["container_cpu"] = typer.prompt("Container CPU", default=1, type=int)
    config["container_memory"] = typer.prompt(
        "Container Memory (GB)", default=1, type=int
    )
    config["aks_node_count"] = typer.prompt("AKS Node Count", default=3, type=int)
    config["aks_vm_size"] = typer.prompt("AKS VM Size", default="Standard_DS2_v2")

    # Database configuration
    config["sql_tier"] = typer.prompt("SQL Database Tier", default="Basic")
    config["sql_admin_user"] = typer.prompt("SQL Admin User", default="sqladmin")

    # Storage configuration
    config["storage_tier"] = typer.prompt("Storage Tier", default="Hot")
    config["storage_sku"] = typer.prompt("Storage SKU", default="Standard_LRS")

    return config


def _show_deployment_plan(config: Dict[str, Any], service_type: str, image: str):
    """Show deployment plan for dry run"""

    plan_table = Table(title="Deployment Plan")
    plan_table.add_column("Resource", style="cyan")
    plan_table.add_column("Type", style="green")
    plan_table.add_column("Configuration", style="yellow")

    if service_type == "container_instance":
        plan_table.add_row(
            "Container Instance", "Compute", f"Image: {image}, CPU: 1, Memory: 1GB"
        )
    elif service_type == "aks":
        plan_table.add_row("AKS Cluster", "Compute", "Nodes: 3, VM: Standard_DS2_v2")
    elif service_type == "function":
        plan_table.add_row(
            "Function App", "Compute", "Runtime: Python 3.9, Consumption Plan"
        )

    plan_table.add_row("Storage Account", "Storage", "SKU: Standard_LRS, Tier: Hot")

    plan_table.add_row(
        "SQL Database",
        "Database",
        "Tier: Basic, Collation: SQL_Latin1_General_CP1_CI_AS",
    )

    plan_table.add_row("Key Vault", "Security", "SKU: Standard")

    console.print(plan_table)


def _deploy_container_instance(
    provider: AzureProvider, config: Dict[str, Any], image: str
):
    """Deploy Container Instance"""

    compute_resource = ComputeResource(
        name="inferloop-synthetic",
        service_type="container_instance",
        config={
            "image": image,
            "cpu": config.get("config", {}).get("container_cpu", 1),
            "memory": config.get("config", {}).get("container_memory", 1),
            "port": 8000,
            "environment": {
                "AZURE_SUBSCRIPTION_ID": config["subscription_id"],
                "AZURE_RESOURCE_GROUP": config["resource_group"],
                "AZURE_LOCATION": config["location"],
            },
        },
    )

    resource_id = provider.create_compute_resource(compute_resource)
    console.print(f"✅ Container Instance deployed: {resource_id}")


def _deploy_aks(provider: AzureProvider, config: Dict[str, Any], image: str):
    """Deploy AKS cluster"""

    compute_resource = ComputeResource(
        name="inferloop-synthetic",
        service_type="aks",
        config={
            "nodes": config.get("config", {}).get("aks_node_count", 3),
            "vm_size": config.get("config", {}).get("aks_vm_size", "Standard_DS2_v2"),
            "enable_autoscaling": True,
            "min_nodes": 1,
            "max_nodes": 10,
        },
    )

    resource_id = provider.create_compute_resource(compute_resource)
    console.print(f"✅ AKS cluster deployed: {resource_id}")


def _deploy_azure_function(provider: AzureProvider, config: Dict[str, Any]):
    """Deploy Azure Function"""

    compute_resource = ComputeResource(
        name="inferloop-synthetic",
        service_type="function",
        config={"runtime": "python", "runtime_version": "3.9"},
    )

    resource_id = provider.create_compute_resource(compute_resource)
    console.print(f"✅ Azure Function deployed: {resource_id}")


if __name__ == "__main__":
    app()
