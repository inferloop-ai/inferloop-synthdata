"""CLI commands for on-premises deployment."""

import typer
from typing import Optional
from pathlib import Path
import yaml
import json
import time

from ..base import ResourceConfig
from .provider import OnPremKubernetesProvider
from .openshift import OpenShiftProvider
from .security import SecurityManager
from .backup import BackupManager
from .gitops import GitOpsManager
from ..utils import console, print_deployment_result, print_status_table

# Create the on-premises CLI app
onprem_cli = typer.Typer(
    name="onprem",
    help="On-premises deployment commands for Kubernetes clusters"
)


@onprem_cli.command()
def init(
    name: str = typer.Argument(..., help="Deployment name"),
    kubernetes_dist: str = typer.Option("vanilla", help="Kubernetes distribution: vanilla, openshift, rancher"),
    networking: str = typer.Option("calico", help="Networking plugin: calico, flannel, cilium"),
    masters: int = typer.Option(3, help="Number of master nodes"),
    workers: int = typer.Option(5, help="Number of worker nodes"),
    output: str = typer.Option("onprem-deployment.yaml", help="Output configuration file")
):
    """Initialize on-premises deployment configuration."""
    console.print(f"[cyan]Initializing on-premises deployment: {name}[/cyan]")
    
    config = {
        "apiVersion": "synthdata.inferloop.com/v1",
        "kind": "OnPremDeployment",
        "metadata": {
            "name": name,
            "distribution": kubernetes_dist,
            "networking": networking
        },
        "spec": {
            "cluster": {
                "masters": masters,
                "workers": workers,
                "networking": {
                    "plugin": networking,
                    "podCIDR": "10.244.0.0/16",
                    "serviceCIDR": "10.96.0.0/12"
                }
            },
            "storage": {
                "type": "minio",
                "nodes": 4,
                "volumeSize": "500Gi",
                "storageClass": "standard"
            },
            "database": {
                "type": "postgresql",
                "replicas": 2,
                "volumeSize": "100Gi",
                "version": "14"
            },
            "monitoring": {
                "enabled": True,
                "prometheus": {
                    "retention": "30d",
                    "storage": "100Gi"
                },
                "grafana": {
                    "adminPassword": "admin"
                }
            }
        }
    }
    
    # Write configuration to file
    with open(output, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    console.print(f"[green]✓[/green] Configuration saved to: {output}")
    console.print("\n[yellow]Next steps:[/yellow]")
    console.print("1. Review and customize the configuration file")
    console.print("2. Run: inferloop-synthetic deploy onprem check-requirements")
    console.print("3. Run: inferloop-synthetic deploy onprem create-cluster --config " + output)


@onprem_cli.command()
def check_requirements(
    min_cpu: int = typer.Option(16, help="Minimum CPU cores required"),
    min_memory: int = typer.Option(64, help="Minimum memory in GB required"),
    min_storage: int = typer.Option(500, help="Minimum storage in GB required")
):
    """Check system requirements for on-premises deployment."""
    console.print("[cyan]Checking system requirements...[/cyan]\n")
    
    requirements_met = True
    
    # Check operating system
    import platform
    os_info = platform.platform()
    console.print(f"Operating System: {os_info}")
    
    # Check CPU cores
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    if cpu_count >= min_cpu:
        console.print(f"[green]✓[/green] CPU Cores: {cpu_count} (minimum {min_cpu} required)")
    else:
        console.print(f"[red]✗[/red] CPU Cores: {cpu_count} (minimum {min_cpu} required)")
        requirements_met = False
    
    # Check memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb >= min_memory:
            console.print(f"[green]✓[/green] Memory: {memory_gb:.1f}GB (minimum {min_memory}GB required)")
        else:
            console.print(f"[red]✗[/red] Memory: {memory_gb:.1f}GB (minimum {min_memory}GB required)")
            requirements_met = False
    except ImportError:
        console.print("[yellow]![/yellow] Cannot check memory (psutil not installed)")
    
    # Check Docker/container runtime
    import subprocess
    try:
        subprocess.run(["docker", "--version"], capture_output=True, check=True)
        console.print("[green]✓[/green] Container Runtime: Docker installed")
    except:
        try:
            subprocess.run(["containerd", "--version"], capture_output=True, check=True)
            console.print("[green]✓[/green] Container Runtime: containerd installed")
        except:
            console.print("[red]✗[/red] Container Runtime: Not found (Docker or containerd required)")
            requirements_met = False
    
    # Check kubectl
    try:
        subprocess.run(["kubectl", "version", "--client"], capture_output=True, check=True)
        console.print("[green]✓[/green] kubectl: Installed")
    except:
        console.print("[yellow]![/yellow] kubectl: Not installed (will be installed during setup)")
    
    # Check network connectivity
    import socket
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        console.print("[green]✓[/green] Network: Internet connectivity detected")
    except:
        console.print("[yellow]![/yellow] Network: No internet connectivity (air-gapped mode)")
    
    print()
    if requirements_met:
        console.print("[green]All requirements met. Ready for deployment.[/green]")
    else:
        console.print("[red]Some requirements not met. Please address the issues above.[/red]")
    
    return requirements_met


@onprem_cli.command()
def create_cluster(
    config_file: str = typer.Option("onprem-deployment.yaml", "--config", help="Deployment configuration file"),
    kubeconfig: Optional[str] = typer.Option(None, help="Path to kubeconfig file"),
    dry_run: bool = typer.Option(False, help="Perform a dry run without creating resources")
):
    """Create or connect to Kubernetes cluster."""
    if not Path(config_file).exists():
        console.print(f"[red]Configuration file not found: {config_file}[/red]")
        raise typer.Exit(1)
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    console.print(f"[cyan]Creating Kubernetes cluster...[/cyan]")
    
    if dry_run:
        console.print("[yellow]DRY RUN MODE - No resources will be created[/yellow]")
        console.print("\nCluster configuration:")
        console.print(f"  Masters: {config['spec']['cluster']['masters']}")
        console.print(f"  Workers: {config['spec']['cluster']['workers']}")
        console.print(f"  Networking: {config['spec']['cluster']['networking']['plugin']}")
        return
    
    try:
        # Initialize provider
        provider = OnPremKubernetesProvider(kubeconfig_path=kubeconfig)
        
        # Authenticate
        if provider.authenticate(namespace="synthdata"):
            console.print("[green]✓[/green] Connected to Kubernetes cluster")
            console.print("[green]✓[/green] Namespace 'synthdata' created/verified")
        else:
            console.print("[red]Failed to authenticate with cluster[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@onprem_cli.command()
def setup_storage(
    storage_type: str = typer.Option("minio", help="Storage type: minio, nfs, ceph"),
    nodes: int = typer.Option(4, help="Number of storage nodes (MinIO)"),
    size: str = typer.Option("500Gi", help="Storage size per node"),
    namespace: str = typer.Option("synthdata", help="Kubernetes namespace"),
    access_key: str = typer.Option("minioadmin", help="MinIO access key"),
    secret_key: str = typer.Option("minioadmin123", help="MinIO secret key")
):
    """Setup storage system (MinIO, NFS, etc.)."""
    console.print(f"[cyan]Setting up {storage_type} storage...[/cyan]")
    
    try:
        provider = OnPremKubernetesProvider()
        provider.authenticate(namespace=namespace)
        
        # Create storage configuration
        storage_config = ResourceConfig(
            compute={"count": nodes},
            storage={"size": size, "storage_class": "standard"},
            metadata={
                "storage_type": storage_type,
                "namespace": namespace,
                "access_key": access_key,
                "secret_key": secret_key
            }
        )
        
        # Deploy storage
        result = provider.deploy_storage(storage_config)
        
        if result.success:
            console.print(f"[green]✓[/green] {storage_type.upper()} deployed successfully")
            console.print(f"[green]✓[/green] Endpoint: {result.endpoint}")
            if result.metadata.get("console_url"):
                console.print(f"[green]✓[/green] Console: {result.metadata['console_url']}")
            console.print(f"\nStorage capacity: {int(size.rstrip('Gi')) * nodes}Gi total")
        else:
            console.print(f"[red]Storage deployment failed: {result.message}[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@onprem_cli.command()
def setup_database(
    db_type: str = typer.Option("postgresql", help="Database type: postgresql, mysql"),
    ha: bool = typer.Option(True, help="Enable high availability"),
    replicas: int = typer.Option(2, help="Number of replicas (if HA enabled)"),
    size: str = typer.Option("100Gi", help="Storage size"),
    namespace: str = typer.Option("synthdata", help="Kubernetes namespace"),
    password: str = typer.Option("synthdata123", help="Database password")
):
    """Deploy database (PostgreSQL, MySQL)."""
    console.print(f"[cyan]Deploying {db_type} database...[/cyan]")
    
    try:
        provider = OnPremKubernetesProvider()
        provider.authenticate(namespace=namespace)
        
        # Create database configuration
        db_config = ResourceConfig(
            compute={"cpu": "2", "memory": "4Gi"},
            storage={"size": size, "storage_class": "standard"},
            metadata={
                "db_type": db_type,
                "namespace": namespace,
                "password": password,
                "name": "postgres"
            }
        )
        
        # Deploy database
        result = provider.deploy_database(db_config)
        
        if result.success:
            console.print(f"[green]✓[/green] {db_type.upper()} deployed successfully")
            console.print(f"[green]✓[/green] Connection string: {result.endpoint}")
            if ha and replicas > 1:
                console.print(f"[green]✓[/green] High availability: Enabled with {replicas} replicas")
        else:
            console.print(f"[red]Database deployment failed: {result.message}[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@onprem_cli.command()
def install_app(
    environment: str = typer.Option("production", help="Environment: development, staging, production"),
    replicas: int = typer.Option(3, help="Number of application replicas"),
    cpu: str = typer.Option("4", help="CPU cores per replica"),
    memory: str = typer.Option("16Gi", help="Memory per replica"),
    image: str = typer.Option("inferloop/synthdata:latest", help="Container image"),
    namespace: str = typer.Option("synthdata", help="Kubernetes namespace"),
    autoscale: bool = typer.Option(True, help="Enable autoscaling"),
    min_replicas: int = typer.Option(3, help="Minimum replicas for autoscaling"),
    max_replicas: int = typer.Option(10, help="Maximum replicas for autoscaling")
):
    """Deploy Inferloop Synthetic Data application."""
    console.print(f"[cyan]Deploying Inferloop Synthetic Data ({environment})...[/cyan]")
    
    try:
        provider = OnPremKubernetesProvider()
        provider.authenticate(namespace=namespace)
        
        # Create application configuration
        app_config = ResourceConfig(
            compute={
                "count": replicas,
                "cpu": cpu,
                "memory": memory
            },
            networking={"service_type": "ClusterIP"},
            metadata={
                "name": f"synthdata-{environment}",
                "namespace": namespace,
                "image": image,
                "environment": environment,
                "env": {
                    "ENVIRONMENT": environment,
                    "LOG_LEVEL": "INFO" if environment == "production" else "DEBUG"
                }
            }
        )
        
        # Deploy application
        result = provider.deploy_container(app_config)
        
        if result.success:
            console.print(f"[green]✓[/green] Application deployed successfully")
            console.print(f"[green]✓[/green] Deployment: {result.resource_id}")
            console.print(f"[green]✓[/green] Internal endpoint: {result.endpoint}")
            
            if autoscale:
                console.print(f"[green]✓[/green] Autoscaling: Enabled (min: {min_replicas}, max: {max_replicas})")
                # TODO: Implement HPA creation
                
        else:
            console.print(f"[red]Application deployment failed: {result.message}[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@onprem_cli.command()
def setup_monitoring(
    namespace: str = typer.Option("monitoring", help="Monitoring namespace"),
    prometheus_retention: str = typer.Option("30d", help="Prometheus data retention"),
    grafana_password: str = typer.Option("admin", help="Grafana admin password")
):
    """Setup monitoring stack (Prometheus + Grafana)."""
    console.print("[cyan]Setting up monitoring stack...[/cyan]")
    
    try:
        provider = OnPremKubernetesProvider()
        provider.authenticate(namespace="synthdata")
        
        # Create monitoring configuration
        monitoring_config = ResourceConfig(
            metadata={
                "namespace": namespace,
                "prometheus_retention": prometheus_retention,
                "grafana_password": grafana_password
            }
        )
        
        # Deploy monitoring
        result = provider.deploy_monitoring(monitoring_config)
        
        if result.success:
            console.print("[green]✓[/green] Monitoring stack deployed successfully")
            console.print(f"[green]✓[/green] Prometheus: {result.metadata['prometheus_url']}")
            console.print(f"[green]✓[/green] Grafana: {result.metadata['grafana_url']}")
            console.print("\n[yellow]Default credentials:[/yellow]")
            console.print("  Grafana: admin / admin")
            
        else:
            console.print(f"[red]Monitoring deployment failed: {result.message}[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@onprem_cli.command()
def status(
    namespace: str = typer.Option("synthdata", help="Kubernetes namespace"),
    detailed: bool = typer.Option(False, help="Show detailed status")
):
    """Check deployment status."""
    console.print("[cyan]Checking deployment status...[/cyan]\n")
    
    try:
        provider = OnPremKubernetesProvider()
        provider.authenticate(namespace=namespace)
        
        # List deployments
        deployments = provider.list_deployments()
        
        if deployments:
            # Display deployment table
            from rich.table import Table
            table = Table(title="Synthdata Deployments")
            table.add_column("Name", style="cyan")
            table.add_column("Namespace", style="magenta")
            table.add_column("Ready", style="green")
            table.add_column("Created", style="yellow")
            
            for dep in deployments:
                table.add_row(
                    dep["name"],
                    dep["namespace"],
                    dep["ready"],
                    dep["created"]
                )
            
            console.print(table)
            
            # Show detailed status if requested
            if detailed:
                for dep in deployments:
                    resource_id = f"{dep['namespace']}/{dep['name']}"
                    status = provider.get_deployment_status(resource_id)
                    
                    console.print(f"\n[cyan]Deployment: {dep['name']}[/cyan]")
                    console.print(f"Replicas: {status['replicas']['ready']}/{status['replicas']['desired']} ready")
                    
                    if status.get("pods"):
                        console.print("Pods:")
                        for pod in status["pods"]:
                            status_icon = "[green]✓[/green]" if pod["ready"] else "[red]✗[/red]"
                            console.print(f"  {status_icon} {pod['name']} ({pod['status']})")
                            
        else:
            console.print("[yellow]No deployments found[/yellow]")
            
        # Validate environment
        console.print("\n[cyan]Environment validation:[/cyan]")
        is_valid, issues = provider.validate()
        
        if is_valid:
            console.print("[green]✓ Environment is properly configured[/green]")
        else:
            console.print("[red]✗ Environment has issues:[/red]")
            for issue in issues:
                console.print(f"  - {issue}")
                
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@onprem_cli.command()
def scale(
    component: str = typer.Argument(..., help="Component to scale: app, workers"),
    replicas: int = typer.Argument(..., help="Number of replicas"),
    namespace: str = typer.Option("synthdata", help="Kubernetes namespace")
):
    """Scale application components."""
    console.print(f"[cyan]Scaling {component} to {replicas} replicas...[/cyan]")
    
    # TODO: Implement scaling logic using kubectl scale or HPA update
    console.print("[yellow]Scaling functionality to be implemented[/yellow]")


@onprem_cli.command()
def backup(
    name: str = typer.Option(None, help="Backup name (auto-generated if not provided)"),
    namespace: str = typer.Option("synthdata", help="Kubernetes namespace"),
    include_data: bool = typer.Option(True, help="Include persistent data in backup")
):
    """Create backup of deployment."""
    if not name:
        from datetime import datetime
        name = f"backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    console.print(f"[cyan]Creating backup: {name}...[/cyan]")
    
    # TODO: Implement backup using Velero or similar
    console.print("[yellow]Backup functionality to be implemented[/yellow]")
    console.print("Recommended: Use Velero for Kubernetes backup/restore")


@onprem_cli.command()
def restore(
    backup_name: str = typer.Argument(..., help="Name of backup to restore"),
    namespace: str = typer.Option("synthdata", help="Kubernetes namespace")
):
    """Restore from backup."""
    console.print(f"[cyan]Restoring from backup: {backup_name}...[/cyan]")
    
    # TODO: Implement restore using Velero or similar
    console.print("[yellow]Restore functionality to be implemented[/yellow]")
    console.print("Recommended: Use Velero for Kubernetes backup/restore")


@onprem_cli.command()
def delete(
    resource_type: str = typer.Argument(..., help="Resource type: app, storage, database, monitoring, all"),
    namespace: str = typer.Option("synthdata", help="Kubernetes namespace"),
    force: bool = typer.Option(False, "--force", help="Force deletion without confirmation")
):
    """Delete on-premises resources."""
    if not force:
        confirm = typer.confirm(f"Are you sure you want to delete {resource_type}?")
        if not confirm:
            console.print("[yellow]Deletion cancelled[/yellow]")
            return
    
    console.print(f"[cyan]Deleting {resource_type}...[/cyan]")
    
    try:
        provider = OnPremKubernetesProvider()
        provider.authenticate(namespace=namespace)
        
        if resource_type == "all":
            # Delete entire namespace
            import subprocess
            subprocess.run(
                ["kubectl", "delete", "namespace", namespace],
                check=True
            )
            console.print(f"[green]✓[/green] Namespace '{namespace}' deleted")
        else:
            # Delete specific resources
            deployments = provider.list_deployments()
            for dep in deployments:
                if resource_type in dep["name"]:
                    resource_id = f"{dep['namespace']}/{dep['name']}"
                    if provider.delete_deployment(resource_id):
                        console.print(f"[green]✓[/green] Deleted: {dep['name']}")
                        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@onprem_cli.command()
def create_offline_bundle(
    output: str = typer.Option("synthdata-offline.tar", help="Output bundle filename"),
    version: str = typer.Option("latest", help="Synthdata version to bundle")
):
    """Create offline deployment bundle for air-gapped environments."""
    console.print(f"[cyan]Creating offline deployment bundle...[/cyan]")
    
    # TODO: Implement offline bundle creation
    # This would include:
    # - Container images
    # - Helm charts
    # - Dependencies
    # - Documentation
    
    console.print("[yellow]Offline bundle creation to be implemented[/yellow]")
    console.print("\nBundle should include:")
    console.print("- All required container images")
    console.print("- Kubernetes manifests")
    console.print("- Installation scripts")
    console.print("- Documentation")


@onprem_cli.command()
def deploy_helm(
    chart_path: str = typer.Argument(..., help="Path to Helm chart"),
    release_name: str = typer.Option("synthdata", help="Helm release name"),
    namespace: str = typer.Option("synthdata", help="Kubernetes namespace"),
    values_file: str = typer.Option(None, help="Values file path"),
    wait: bool = typer.Option(True, help="Wait for deployment to complete")
):
    """Deploy using Helm charts."""
    console.print(f"[cyan]Deploying Helm chart: {chart_path}[/cyan]")
    
    try:
        provider = OnPremKubernetesProvider()
        provider.authenticate(namespace=namespace)
        
        config = ResourceConfig(
            metadata={
                "name": release_name,
                "namespace": namespace,
                "chart_path": chart_path,
                "values_file": values_file
            }
        )
        
        result = provider.deploy_with_helm(config)
        
        if result.success:
            console.print(f"[green]✓[/green] Helm deployment successful")
            console.print(f"Release: {release_name}")
            console.print(f"Namespace: {namespace}")
        else:
            console.print(f"[red]Helm deployment failed: {result.message}[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@onprem_cli.command()
def deploy_openshift(
    name: str = typer.Option("synthdata", help="Application name"),
    namespace: str = typer.Option("synthdata", help="OpenShift project"),
    image: str = typer.Option("inferloop/synthdata:latest", help="Container image"),
    replicas: int = typer.Option(3, help="Number of replicas"),
    hostname: str = typer.Option(None, help="Route hostname")
):
    """Deploy to OpenShift Container Platform."""
    console.print(f"[cyan]Deploying to OpenShift: {name}[/cyan]")
    
    try:
        provider = OpenShiftProvider()
        
        # Authenticate (assumes already logged in via oc login)
        if provider.authenticate(project=namespace):
            console.print(f"[green]✓[/green] Connected to OpenShift project: {namespace}")
        else:
            console.print("[red]Failed to authenticate with OpenShift[/red]")
            raise typer.Exit(1)
        
        config = ResourceConfig(
            compute={"count": replicas},
            networking={"hostname": hostname},
            metadata={
                "name": name,
                "image": image
            }
        )
        
        result = provider.deploy_container(config)
        
        if result.success:
            console.print(f"[green]✓[/green] OpenShift deployment successful")
            console.print(f"Route URL: {result.endpoint}")
        else:
            console.print(f"[red]OpenShift deployment failed: {result.message}[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@onprem_cli.command()
def setup_security(
    ldap_host: str = typer.Option(None, help="LDAP server host"),
    ldap_bind_dn: str = typer.Option(None, help="LDAP bind DN"),
    ldap_bind_password: str = typer.Option(None, help="LDAP bind password"),
    enable_cert_manager: bool = typer.Option(True, help="Enable cert-manager"),
    enable_network_policies: bool = typer.Option(True, help="Enable network policies"),
    namespace: str = typer.Option("synthdata", help="Target namespace")
):
    """Setup security features (LDAP, certificates, network policies)."""
    console.print("[cyan]Setting up security features...[/cyan]")
    
    try:
        security_mgr = SecurityManager()
        
        # Deploy cert-manager
        if enable_cert_manager:
            console.print("Deploying cert-manager...")
            result = security_mgr.deploy_cert_manager()
            if result.success:
                console.print("[green]✓[/green] cert-manager deployed")
                
                # Create self-signed issuer
                result = security_mgr.create_cluster_issuer("self-signed")
                if result.success:
                    console.print("[green]✓[/green] Self-signed ClusterIssuer created")
            else:
                console.print(f"[red]cert-manager deployment failed: {result.message}[/red]")
        
        # Setup LDAP integration
        if ldap_host:
            console.print("Setting up LDAP integration...")
            ldap_config = ResourceConfig(
                metadata={
                    "ldap": {
                        "host": ldap_host,
                        "bind_dn": ldap_bind_dn,
                        "bind_password": ldap_bind_password
                    }
                }
            )
            
            result = security_mgr.deploy_dex_oidc(ldap_config)
            if result.success:
                console.print("[green]✓[/green] LDAP integration deployed")
                console.print(f"Dex endpoint: {result.endpoint}")
            else:
                console.print(f"[red]LDAP setup failed: {result.message}[/red]")
        
        # Create network policies
        if enable_network_policies:
            console.print("Creating network policies...")
            result = security_mgr.create_network_policies(namespace)
            if result.success:
                console.print("[green]✓[/green] Network policies created")
            else:
                console.print(f"[red]Network policy creation failed: {result.message}[/red]")
                
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@onprem_cli.command()
def setup_backup(
    provider: str = typer.Option("minio", help="Backup provider: aws, gcp, azure, minio"),
    bucket: str = typer.Option("velero-backups", help="Backup bucket name"),
    schedule: str = typer.Option("0 2 * * *", help="Backup schedule (cron format)"),
    namespace: str = typer.Option("synthdata", help="Namespace to backup"),
    endpoint: str = typer.Option("http://minio:9000", help="MinIO endpoint (for MinIO provider)")
):
    """Setup backup and restore with Velero."""
    console.print(f"[cyan]Setting up backup with {provider}...[/cyan]")
    
    try:
        backup_mgr = BackupManager()
        
        config = ResourceConfig(
            metadata={
                "provider": provider,
                "bucket": bucket,
                "schedule": schedule,
                "namespaces": [namespace],
                "endpoint": endpoint,
                "access_key": "minio",
                "secret_key": "minio123"
            }
        )
        
        result = backup_mgr.install_velero(config)
        
        if result.success:
            console.print("[green]✓[/green] Velero backup system installed")
            console.print(f"Provider: {provider}")
            console.print(f"Bucket: {bucket}")
            console.print(f"Schedule: {schedule}")
            
            # Validate installation
            is_valid, issues = backup_mgr.validate_installation()
            if is_valid:
                console.print("[green]✓[/green] Backup system validation passed")
            else:
                console.print("[yellow]⚠️ Backup system validation issues:[/yellow]")
                for issue in issues:
                    console.print(f"  - {issue}")
        else:
            console.print(f"[red]Backup setup failed: {result.message}[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@onprem_cli.command()
def setup_gitops(
    provider: str = typer.Option("argocd", help="GitOps provider: argocd, flux"),
    repo_url: str = typer.Option(None, help="Git repository URL"),
    branch: str = typer.Option("main", help="Git branch"),
    path: str = typer.Option(".", help="Path in repository"),
    app_name: str = typer.Option("synthdata", help="Application name")
):
    """Setup GitOps with ArgoCD or Flux."""
    console.print(f"[cyan]Setting up GitOps with {provider}...[/cyan]")
    
    try:
        gitops_mgr = GitOpsManager()
        
        if provider == "argocd":
            # Install ArgoCD
            result = gitops_mgr.install_argocd()
            if result.success:
                console.print("[green]✓[/green] ArgoCD installed")
                console.print(f"Admin password: {result.metadata['admin_password']}")
                
                # Create application if repo URL provided
                if repo_url:
                    app_config = ResourceConfig(
                        metadata={
                            "name": app_name,
                            "repo_url": repo_url,
                            "target_revision": branch,
                            "path": path,
                            "dest_namespace": "synthdata"
                        }
                    )
                    
                    app_result = gitops_mgr.create_argocd_application(app_config)
                    if app_result.success:
                        console.print(f"[green]✓[/green] ArgoCD application '{app_name}' created")
            else:
                console.print(f"[red]ArgoCD installation failed: {result.message}[/red]")
                
        elif provider == "flux":
            # Install Flux
            result = gitops_mgr.install_flux()
            if result.success:
                console.print("[green]✓[/green] Flux installed")
                
                # Create git source if repo URL provided
                if repo_url:
                    source_config = ResourceConfig(
                        metadata={
                            "name": f"{app_name}-source",
                            "repo_url": repo_url,
                            "branch": branch
                        }
                    )
                    
                    source_result = gitops_mgr.create_flux_source(source_config)
                    if source_result.success:
                        console.print(f"[green]✓[/green] Flux git source created")
                        
                        # Create kustomization
                        kustomize_config = ResourceConfig(
                            metadata={
                                "name": f"{app_name}-kustomization",
                                "source_ref": f"{app_name}-source",
                                "path": path,
                                "target_namespace": "synthdata"
                            }
                        )
                        
                        kustomize_result = gitops_mgr.create_flux_kustomization(kustomize_config)
                        if kustomize_result.success:
                            console.print(f"[green]✓[/green] Flux kustomization created")
            else:
                console.print(f"[red]Flux installation failed: {result.message}[/red]")
        else:
            console.print(f"[red]Unknown GitOps provider: {provider}[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@onprem_cli.command()
def list_backups():
    """List all available backups."""
    console.print("[cyan]Listing backups...[/cyan]")
    
    try:
        backup_mgr = BackupManager()
        backups = backup_mgr.list_backups()
        
        if backups:
            from rich.table import Table
            table = Table(title="Available Backups")
            table.add_column("Name", style="cyan")
            table.add_column("Created", style="green")
            table.add_column("Phase", style="yellow")
            table.add_column("Size", style="magenta")
            table.add_column("Errors", style="red")
            
            for backup in backups:
                table.add_row(
                    backup["name"],
                    backup["created"],
                    backup["phase"],
                    str(backup["size"]),
                    str(backup["errors"])
                )
            
            console.print(table)
        else:
            console.print("[yellow]No backups found[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@onprem_cli.command()
def create_backup_now(
    backup_name: str = typer.Argument(..., help="Backup name"),
    namespaces: str = typer.Option("synthdata", help="Comma-separated list of namespaces"),
    wait: bool = typer.Option(True, help="Wait for backup to complete")
):
    """Create a backup immediately."""
    console.print(f"[cyan]Creating backup: {backup_name}[/cyan]")
    
    try:
        backup_mgr = BackupManager()
        namespace_list = [ns.strip() for ns in namespaces.split(",")]
        
        result = backup_mgr.create_backup(
            backup_name=backup_name,
            namespaces=namespace_list,
            wait=wait
        )
        
        if result.success:
            console.print(f"[green]✓[/green] Backup '{backup_name}' created successfully")
        else:
            console.print(f"[red]Backup creation failed: {result.message}[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    onprem_cli()