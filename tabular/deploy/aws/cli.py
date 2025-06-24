"""AWS deployment CLI commands."""

import typer
from typing import Optional
from rich.console import Console
from rich.table import Table
from pathlib import Path

from .provider import AWSProvider
from ..base import ResourceConfig

console = Console()
aws_cli = typer.Typer(help="AWS deployment commands")


@aws_cli.command()
def deploy(
    project_id: str = typer.Argument(..., help="Project identifier"),
    region: str = typer.Option("us-east-1", help="AWS region"),
    instance_type: str = typer.Option("t3.medium", help="EC2 instance type"),
    disk_size: int = typer.Option(100, help="Disk size in GB"),
    ssh_key: Optional[str] = typer.Option(None, help="SSH key pair name"),
    config_file: Optional[Path] = typer.Option(None, help="Configuration file")
):
    """Deploy infrastructure on AWS."""
    console.print("[bold blue]Deploying AWS infrastructure...[/bold blue]")
    
    # Initialize provider
    provider = AWSProvider(project_id, region)
    
    if not provider.authenticate():
        console.print("[bold red]AWS authentication failed![/bold red]")
        raise typer.Exit(1)
        
    # Create configuration
    config = ResourceConfig(
        name=project_id,
        instance_type=instance_type,
        disk_size_gb=disk_size
    )
    
    if ssh_key:
        config.ssh_key_name = ssh_key
        
    # Deploy infrastructure
    result = provider.deploy_infrastructure(config)
    
    if result.success:
        console.print(f"[bold green]✓[/bold green] {result.message}")
        
        # Display resources
        table = Table(title="Deployed Resources")
        table.add_column("Resource", style="cyan")
        table.add_column("ID/Name", style="green")
        
        for key, value in result.resources.items():
            table.add_row(key, str(value))
            
        console.print(table)
    else:
        console.print(f"[bold red]✗[/bold red] {result.message}")
        raise typer.Exit(1)


@aws_cli.command()
def deploy_container(
    project_id: str = typer.Argument(..., help="Project identifier"),
    image: str = typer.Argument(..., help="Container image"),
    region: str = typer.Option("us-east-1", help="AWS region"),
    cpu: int = typer.Option(1024, help="CPU units (1024 = 1 vCPU)"),
    memory: int = typer.Option(2048, help="Memory in MB"),
    replicas: int = typer.Option(1, help="Number of replicas"),
    use_fargate: bool = typer.Option(True, help="Use Fargate instead of EC2")
):
    """Deploy containerized application on AWS ECS/Fargate."""
    console.print("[bold blue]Deploying container on AWS...[/bold blue]")
    
    provider = AWSProvider(project_id, region)
    
    if not provider.authenticate():
        console.print("[bold red]AWS authentication failed![/bold red]")
        raise typer.Exit(1)
        
    config = ResourceConfig(
        name=project_id,
        container_image=image,
        cpu=cpu,
        memory=memory,
        replicas=replicas
    )
    config.use_fargate = use_fargate
    
    result = provider.deploy_container(config)
    
    if result.success:
        console.print(f"[bold green]✓[/bold green] {result.message}")
        
        table = Table(title="Container Deployment")
        table.add_column("Resource", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in result.resources.items():
            table.add_row(key, str(value))
            
        console.print(table)
    else:
        console.print(f"[bold red]✗[/bold red] {result.message}")
        raise typer.Exit(1)


@aws_cli.command()
def deploy_serverless(
    project_id: str = typer.Argument(..., help="Project identifier"),
    region: str = typer.Option("us-east-1", help="AWS region"),
    memory: int = typer.Option(512, help="Memory in MB"),
    timeout: int = typer.Option(300, help="Timeout in seconds")
):
    """Deploy serverless function on AWS Lambda."""
    console.print("[bold blue]Deploying Lambda function...[/bold blue]")
    
    provider = AWSProvider(project_id, region)
    
    if not provider.authenticate():
        console.print("[bold red]AWS authentication failed![/bold red]")
        raise typer.Exit(1)
        
    config = ResourceConfig(
        name=project_id,
        memory=memory,
        timeout=timeout
    )
    
    result = provider.deploy_serverless(config)
    
    if result.success:
        console.print(f"[bold green]✓[/bold green] {result.message}")
        
        table = Table(title="Lambda Function")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in result.resources.items():
            table.add_row(key, str(value))
            
        console.print(table)
    else:
        console.print(f"[bold red]✗[/bold red] {result.message}")
        raise typer.Exit(1)


@aws_cli.command()
def deploy_batch(
    project_id: str = typer.Argument(..., help="Project identifier"),
    image: str = typer.Argument(..., help="Container image for batch jobs"),
    region: str = typer.Option("us-east-1", help="AWS region"),
    cpu: int = typer.Option(2, help="vCPUs per job"),
    memory: int = typer.Option(2048, help="Memory per job in MB")
):
    """Deploy batch processing environment on AWS Batch."""
    console.print("[bold blue]Creating AWS Batch environment...[/bold blue]")
    
    provider = AWSProvider(project_id, region)
    
    if not provider.authenticate():
        console.print("[bold red]AWS authentication failed![/bold red]")
        raise typer.Exit(1)
        
    config = ResourceConfig(
        name=project_id,
        container_image=image,
        cpu=cpu,
        memory=memory
    )
    
    result = provider.deploy_batch(config)
    
    if result.success:
        console.print(f"[bold green]✓[/bold green] {result.message}")
        
        table = Table(title="Batch Environment")
        table.add_column("Resource", style="cyan")
        table.add_column("Name", style="green")
        
        for key, value in result.resources.items():
            table.add_row(key, str(value))
            
        console.print(table)
    else:
        console.print(f"[bold red]✗[/bold red] {result.message}")
        raise typer.Exit(1)


@aws_cli.command()
def deploy_database(
    project_id: str = typer.Argument(..., help="Project identifier"),
    region: str = typer.Option("us-east-1", help="AWS region"),
    instance_type: str = typer.Option("db.t3.micro", help="RDS instance type"),
    disk_size: int = typer.Option(20, help="Storage size in GB"),
    db_password: str = typer.Option(..., prompt=True, hide_input=True, help="Database password")
):
    """Deploy PostgreSQL database on AWS RDS."""
    console.print("[bold blue]Creating RDS database...[/bold blue]")
    
    provider = AWSProvider(project_id, region)
    
    if not provider.authenticate():
        console.print("[bold red]AWS authentication failed![/bold red]")
        raise typer.Exit(1)
        
    config = ResourceConfig(
        name=project_id,
        instance_type=instance_type,
        disk_size_gb=disk_size,
        db_password=db_password
    )
    
    result = provider.deploy_database(config)
    
    if result.success:
        console.print(f"[bold green]✓[/bold green] {result.message}")
        
        table = Table(title="Database Details")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in result.resources.items():
            table.add_row(key, str(value))
            
        console.print(table)
    else:
        console.print(f"[bold red]✗[/bold red] {result.message}")
        raise typer.Exit(1)


@aws_cli.command()
def status(
    project_id: str = typer.Argument(..., help="Project identifier"),
    region: str = typer.Option("us-east-1", help="AWS region")
):
    """Check status of deployed AWS resources."""
    console.print("[bold blue]Checking AWS resources...[/bold blue]")
    
    provider = AWSProvider(project_id, region)
    
    if not provider.authenticate():
        console.print("[bold red]AWS authentication failed![/bold red]")
        raise typer.Exit(1)
        
    status = provider.get_status()
    
    # Display instances
    if status['resources'].get('instances'):
        table = Table(title="EC2 Instances")
        table.add_column("Instance ID", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("State", style="green")
        table.add_column("Public IP", style="blue")
        
        for instance in status['resources']['instances']:
            table.add_row(
                instance['id'],
                instance['type'],
                instance['state'],
                instance.get('public_ip', 'N/A')
            )
            
        console.print(table)
        
    # Display buckets
    if status['resources'].get('buckets'):
        table = Table(title="S3 Buckets")
        table.add_column("Bucket Name", style="cyan")
        table.add_column("Created", style="yellow")
        
        for bucket in status['resources']['buckets']:
            table.add_row(bucket['name'], bucket['created'])
            
        console.print(table)
        
    # Display databases
    if status['resources'].get('databases'):
        table = Table(title="RDS Databases")
        table.add_column("DB Instance", style="cyan")
        table.add_column("Engine", style="yellow")
        table.add_column("Status", style="green")
        table.add_column("Endpoint", style="blue")
        
        for db in status['resources']['databases']:
            table.add_row(
                db['id'],
                db['engine'],
                db['status'],
                db.get('endpoint', 'N/A')
            )
            
        console.print(table)
        
    if not any(status['resources'].values()):
        console.print("[yellow]No resources found for this project[/yellow]")


@aws_cli.command()
def cost(
    project_id: str = typer.Argument(..., help="Project identifier"),
    region: str = typer.Option("us-east-1", help="AWS region"),
    instance_type: str = typer.Option("t3.medium", help="EC2 instance type"),
    disk_size: int = typer.Option(100, help="Disk size in GB")
):
    """Estimate AWS deployment costs."""
    console.print("[bold blue]Estimating AWS costs...[/bold blue]")
    
    provider = AWSProvider(project_id, region)
    
    config = ResourceConfig(
        name=project_id,
        instance_type=instance_type,
        disk_size_gb=disk_size
    )
    
    costs = provider.estimate_cost(config)
    
    table = Table(title="Estimated Monthly Costs (USD)")
    table.add_column("Component", style="cyan")
    table.add_column("Cost", style="green", justify="right")
    
    table.add_row("Compute", f"${costs['compute']:.2f}")
    table.add_row("Storage", f"${costs['storage']:.2f}")
    table.add_row("Network", f"${costs['network']:.2f}")
    table.add_section()
    table.add_row("Total", f"${costs['total']:.2f}", style="bold")
    
    console.print(table)
    console.print("\n[yellow]Note: These are rough estimates. Actual costs may vary.[/yellow]")


@aws_cli.command()
def deploy_eks(
    project_id: str = typer.Argument(..., help="Project identifier"),
    region: str = typer.Option("us-east-1", help="AWS region"),
    node_count: int = typer.Option(2, help="Number of worker nodes"),
    node_type: str = typer.Option("t3.medium", help="EC2 instance type for nodes")
):
    """Deploy Kubernetes cluster on Amazon EKS."""
    console.print("[bold blue]Creating EKS cluster...[/bold blue]")
    
    provider = AWSProvider(project_id, region)
    
    if not provider.authenticate():
        console.print("[bold red]AWS authentication failed![/bold red]")
        raise typer.Exit(1)
        
    config = ResourceConfig(
        name=project_id,
        node_count=node_count,
        instance_type=node_type
    )
    
    result = provider.deploy_eks(config)
    
    if result.success:
        console.print(f"[bold green]✓[/bold green] {result.message}")
        
        table = Table(title="EKS Cluster Details")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in result.resources.items():
            table.add_row(key, str(value))
            
        console.print(table)
        console.print("\n[yellow]Configure kubectl:[/yellow]")
        console.print(f"aws eks update-kubeconfig --name {result.resources['cluster_name']} --region {region}")
    else:
        console.print(f"[bold red]✗[/bold red] {result.message}")
        raise typer.Exit(1)


@aws_cli.command()
def deploy_dynamodb(
    project_id: str = typer.Argument(..., help="Project identifier"),
    region: str = typer.Option("us-east-1", help="AWS region"),
    enable_autoscaling: bool = typer.Option(False, help="Enable auto-scaling")
):
    """Deploy DynamoDB table for data storage."""
    console.print("[bold blue]Creating DynamoDB table...[/bold blue]")
    
    provider = AWSProvider(project_id, region)
    
    if not provider.authenticate():
        console.print("[bold red]AWS authentication failed![/bold red]")
        raise typer.Exit(1)
        
    config = ResourceConfig(
        name=project_id,
        enable_autoscaling=enable_autoscaling
    )
    
    result = provider.deploy_dynamodb(config)
    
    if result.success:
        console.print(f"[bold green]✓[/bold green] {result.message}")
        
        table = Table(title="DynamoDB Table")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in result.resources.items():
            table.add_row(key, str(value))
            
        console.print(table)
    else:
        console.print(f"[bold red]✗[/bold red] {result.message}")
        raise typer.Exit(1)


@aws_cli.command()
def deploy_api(
    project_id: str = typer.Argument(..., help="Project identifier"),
    region: str = typer.Option("us-east-1", help="AWS region"),
    memory: int = typer.Option(3008, help="Lambda memory in MB")
):
    """Deploy enhanced Lambda with API Gateway."""
    console.print("[bold blue]Deploying Lambda function with API Gateway...[/bold blue]")
    
    provider = AWSProvider(project_id, region)
    
    if not provider.authenticate():
        console.print("[bold red]AWS authentication failed![/bold red]")
        raise typer.Exit(1)
        
    config = ResourceConfig(
        name=project_id,
        memory=memory,
        environment={
            'LOG_LEVEL': 'INFO',
            'ENABLE_XRAY': 'true'
        }
    )
    
    result = provider.deploy_enhanced_lambda(config)
    
    if result.success:
        console.print(f"[bold green]✓[/bold green] {result.message}")
        
        table = Table(title="API Deployment")
        table.add_column("Resource", style="cyan")
        table.add_column("Details", style="green")
        
        for key, value in result.resources.items():
            if isinstance(value, list):
                value = ", ".join(value)
            table.add_row(key, str(value))
            
        console.print(table)
        console.print(f"\n[bold yellow]API Endpoint:[/bold yellow] {result.resources['api_endpoint']}")
        console.print("\n[green]Example request:[/green]")
        console.print(f"curl -X POST {result.resources['api_endpoint']} \\")
        console.print("  -H 'Content-Type: application/json' \\")
        console.print("  -d '{\"generator_type\": \"sdv\", \"num_samples\": 1000}'")
    else:
        console.print(f"[bold red]✗[/bold red] {result.message}")
        raise typer.Exit(1)


@aws_cli.command()
def destroy(
    project_id: str = typer.Argument(..., help="Project identifier"),
    region: str = typer.Option("us-east-1", help="AWS region"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation")
):
    """Destroy AWS resources."""
    if not force:
        confirm = typer.confirm("Are you sure you want to destroy all AWS resources?")
        if not confirm:
            console.print("[yellow]Operation cancelled[/yellow]")
            raise typer.Exit(0)
            
    console.print("[bold red]Destroying AWS resources...[/bold red]")
    
    provider = AWSProvider(project_id, region)
    
    if not provider.authenticate():
        console.print("[bold red]AWS authentication failed![/bold red]")
        raise typer.Exit(1)
        
    # Get current resources
    status = provider.get_status()
    resource_ids = {}
    
    # Collect resource IDs
    for instance in status['resources'].get('instances', []):
        resource_ids['instance_id'] = instance['id']
        
    for bucket in status['resources'].get('buckets', []):
        resource_ids['bucket_name'] = bucket['name']
        
    for db in status['resources'].get('databases', []):
        resource_ids['db_instance_id'] = db['id']
        
    if not resource_ids:
        console.print("[yellow]No resources found to destroy[/yellow]")
        raise typer.Exit(0)
        
    # Destroy resources
    result = provider.destroy(resource_ids)
    
    if result.success:
        console.print(f"[bold green]✓[/bold green] {result.message}")
    else:
        console.print(f"[bold red]✗[/bold red] {result.message}")
        raise typer.Exit(1)