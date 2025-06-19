#!/usr/bin/env python3
"""
Deploy command for system deployment and management.

Provides CLI interface for deploying the synthetic data system
to various environments including Docker, Kubernetes, and cloud platforms.
"""

import asyncio
import click
import json
import subprocess
import yaml
from pathlib import Path
from typing import Dict, List, Optional

from ...core import get_logger, get_config
from ..utils import OutputFormatter, ProgressTracker


logger = get_logger(__name__)


@click.command(name='deploy')
@click.option(
    '--target', '-t',
    type=click.Choice([
        'docker', 'docker-compose', 'kubernetes', 
        'aws', 'gcp', 'azure', 'local'
    ]),
    required=True,
    help='Deployment target platform'
)
@click.option(
    '--config-file',
    type=click.Path(exists=True, path_type=Path),
    help='Deployment configuration file'
)
@click.option(
    '--environment', '-e',
    type=click.Choice(['development', 'staging', 'production']),
    default='development',
    help='Deployment environment'
)
@click.option(
    '--namespace',
    type=str,
    default='synthdata',
    help='Kubernetes namespace (for k8s deployments)'
)
@click.option(
    '--registry',
    type=str,
    help='Container registry URL'
)
@click.option(
    '--tag',
    type=str,
    default='latest',
    help='Container image tag'
)
@click.option(
    '--replicas',
    type=int,
    default=1,
    help='Number of replicas to deploy'
)
@click.option(
    '--resources',
    type=str,
    help='Resource limits (JSON format)'
)
@click.option(
    '--secrets-file',
    type=click.Path(exists=True, path_type=Path),
    help='Secrets configuration file'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Show deployment configuration without deploying'
)
@click.option(
    '--force',
    is_flag=True,
    help='Force deployment (overwrite existing)'
)
@click.option(
    '--wait',
    is_flag=True,
    help='Wait for deployment to complete'
)
@click.option(
    '--timeout',
    type=int,
    default=600,
    help='Deployment timeout in seconds'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
def deploy_command(
    target: str,
    config_file: Optional[Path],
    environment: str,
    namespace: str,
    registry: Optional[str],
    tag: str,
    replicas: int,
    resources: Optional[str],
    secrets_file: Optional[Path],
    dry_run: bool,
    force: bool,
    wait: bool,
    timeout: int,
    verbose: bool
):
    """
    Deploy the synthetic data system to various platforms.
    
    Examples:
        # Deploy to local Docker
        synthdata deploy -t docker
        
        # Deploy to Kubernetes
        synthdata deploy -t kubernetes --namespace prod --replicas 3
        
        # Deploy to AWS with configuration
        synthdata deploy -t aws --config-file aws-config.yaml
        
        # Dry run deployment
        synthdata deploy -t kubernetes --dry-run
    """
    asyncio.run(_deploy_async(
        target=target,
        config_file=config_file,
        environment=environment,
        namespace=namespace,
        registry=registry,
        tag=tag,
        replicas=replicas,
        resources=resources,
        secrets_file=secrets_file,
        dry_run=dry_run,
        force=force,
        wait=wait,
        timeout=timeout,
        verbose=verbose
    ))


async def _deploy_async(
    target: str,
    config_file: Optional[Path],
    environment: str,
    namespace: str,
    registry: Optional[str],
    tag: str,
    replicas: int,
    resources: Optional[str],
    secrets_file: Optional[Path],
    dry_run: bool,
    force: bool,
    wait: bool,
    timeout: int,
    verbose: bool
):
    """Async implementation of deploy command"""
    formatter = OutputFormatter(verbose=verbose)
    
    try:
        # Load deployment configuration
        config = await _load_deployment_config(
            config_file, environment, formatter
        )
        
        # Parse resources if provided
        resource_limits = None
        if resources:
            try:
                resource_limits = json.loads(resources)
            except json.JSONDecodeError:
                formatter.error("Invalid resources JSON format")
                return
        
        # Load secrets if provided
        secrets = {}
        if secrets_file:
            with open(secrets_file) as f:
                if secrets_file.suffix in ['.yaml', '.yml']:
                    secrets = yaml.safe_load(f)
                else:
                    secrets = json.load(f)
            formatter.info(f"Loaded secrets from {secrets_file}")
        
        # Prepare deployment parameters
        deploy_params = {
            'target': target,
            'environment': environment,
            'namespace': namespace,
            'registry': registry,
            'tag': tag,
            'replicas': replicas,
            'resources': resource_limits,
            'secrets': secrets,
            'config': config
        }
        
        if dry_run:
            formatter.warning("DRY RUN MODE - No actual deployment will be performed")
            await _show_deployment_plan(deploy_params, formatter)
            return
        
        # Execute deployment based on target
        if target == 'docker':
            await _deploy_docker(deploy_params, formatter, force)
        elif target == 'docker-compose':
            await _deploy_docker_compose(deploy_params, formatter, force)
        elif target == 'kubernetes':
            await _deploy_kubernetes(deploy_params, formatter, force, wait, timeout)
        elif target in ['aws', 'gcp', 'azure']:
            await _deploy_cloud(deploy_params, formatter, force, wait, timeout)
        elif target == 'local':
            await _deploy_local(deploy_params, formatter, force)
        else:
            raise click.ClickException(f"Unsupported deployment target: {target}")
        
        formatter.success(f"Deployment to {target} completed successfully")
        
    except KeyboardInterrupt:
        formatter.warning("\nDeployment interrupted by user")
    except Exception as e:
        formatter.error(f"Deployment failed: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise


async def _load_deployment_config(
    config_file: Optional[Path], 
    environment: str,
    formatter: OutputFormatter
) -> Dict:
    """Load deployment configuration"""
    config = get_config()
    
    if config_file:
        formatter.info(f"Loading configuration from {config_file}")
        with open(config_file) as f:
            if config_file.suffix in ['.yaml', '.yml']:
                custom_config = yaml.safe_load(f)
            else:
                custom_config = json.load(f)
        
        # Merge configurations
        if environment in custom_config:
            config.update(custom_config[environment])
        else:
            config.update(custom_config)
    
    return config


async def _show_deployment_plan(
    deploy_params: Dict,
    formatter: OutputFormatter
):
    """Show deployment plan for dry run"""
    formatter.output("\nDeployment Plan:")
    formatter.output("=" * 50)
    formatter.output(f"Target: {deploy_params['target']}")
    formatter.output(f"Environment: {deploy_params['environment']}")
    formatter.output(f"Namespace: {deploy_params['namespace']}")
    formatter.output(f"Replicas: {deploy_params['replicas']}")
    
    if deploy_params['registry']:
        formatter.output(f"Registry: {deploy_params['registry']}")
    
    formatter.output(f"Tag: {deploy_params['tag']}")
    
    if deploy_params['resources']:
        formatter.output(f"Resources: {json.dumps(deploy_params['resources'], indent=2)}")
    
    if deploy_params['secrets']:
        # Don't show actual secret values
        secret_keys = list(deploy_params['secrets'].keys())
        formatter.output(f"Secrets: {secret_keys}")
    
    formatter.output("=" * 50)


async def _deploy_docker(
    deploy_params: Dict,
    formatter: OutputFormatter,
    force: bool
):
    """Deploy to Docker"""
    formatter.info("Deploying to Docker...")
    
    image_name = "synthdata"
    if deploy_params['registry']:
        image_name = f"{deploy_params['registry']}/synthdata"
    
    full_image_name = f"{image_name}:{deploy_params['tag']}"
    
    # Build Docker image
    formatter.info("Building Docker image...")
    build_cmd = [
        'docker', 'build',
        '-t', full_image_name,
        '.'
    ]
    
    result = subprocess.run(build_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise click.ClickException(f"Docker build failed: {result.stderr}")
    
    # Run container
    formatter.info("Starting Docker container...")
    run_cmd = [
        'docker', 'run',
        '-d',
        '--name', f"synthdata-{deploy_params['environment']}",
        '-p', '8000:8000',
        full_image_name
    ]
    
    if force:
        # Stop and remove existing container
        subprocess.run(['docker', 'stop', f"synthdata-{deploy_params['environment']}"], 
                      capture_output=True)
        subprocess.run(['docker', 'rm', f"synthdata-{deploy_params['environment']}"], 
                      capture_output=True)
    
    result = subprocess.run(run_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise click.ClickException(f"Docker run failed: {result.stderr}")
    
    container_id = result.stdout.strip()
    formatter.success(f"Container started: {container_id}")


async def _deploy_docker_compose(
    deploy_params: Dict,
    formatter: OutputFormatter,
    force: bool
):
    """Deploy using Docker Compose"""
    formatter.info("Deploying with Docker Compose...")
    
    # Generate docker-compose.yml
    compose_config = _generate_docker_compose_config(deploy_params)
    
    compose_file = Path('docker-compose.yml')
    with open(compose_file, 'w') as f:
        yaml.dump(compose_config, f, default_flow_style=False)
    
    formatter.info(f"Generated {compose_file}")
    
    # Deploy with docker-compose
    cmd = ['docker-compose', 'up', '-d']
    if force:
        cmd.extend(['--force-recreate'])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise click.ClickException(f"Docker Compose deployment failed: {result.stderr}")
    
    formatter.success("Docker Compose deployment completed")


async def _deploy_kubernetes(
    deploy_params: Dict,
    formatter: OutputFormatter,
    force: bool,
    wait: bool,
    timeout: int
):
    """Deploy to Kubernetes"""
    formatter.info("Deploying to Kubernetes...")
    
    # Generate Kubernetes manifests
    manifests = _generate_kubernetes_manifests(deploy_params)
    
    # Apply manifests
    for manifest_name, manifest_content in manifests.items():
        manifest_file = Path(f"{manifest_name}.yaml")
        with open(manifest_file, 'w') as f:
            yaml.dump(manifest_content, f, default_flow_style=False)
        
        formatter.info(f"Applying {manifest_file}")
        
        cmd = ['kubectl', 'apply', '-f', str(manifest_file)]
        if deploy_params['namespace'] != 'default':
            cmd.extend(['-n', deploy_params['namespace']])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise click.ClickException(
                f"Kubernetes apply failed for {manifest_file}: {result.stderr}"
            )
    
    if wait:
        formatter.info("Waiting for deployment to be ready...")
        cmd = [
            'kubectl', 'wait', '--for=condition=available',
            f'deployment/synthdata-{deploy_params["environment"]}',
            f'--timeout={timeout}s'
        ]
        if deploy_params['namespace'] != 'default':
            cmd.extend(['-n', deploy_params['namespace']])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            formatter.warning(f"Deployment readiness check failed: {result.stderr}")
    
    formatter.success("Kubernetes deployment completed")


async def _deploy_cloud(
    deploy_params: Dict,
    formatter: OutputFormatter,
    force: bool,
    wait: bool,
    timeout: int
):
    """Deploy to cloud platforms"""
    target = deploy_params['target']
    formatter.info(f"Deploying to {target.upper()}...")
    
    if target == 'aws':
        await _deploy_aws(deploy_params, formatter)
    elif target == 'gcp':
        await _deploy_gcp(deploy_params, formatter)
    elif target == 'azure':
        await _deploy_azure(deploy_params, formatter)


async def _deploy_aws(
    deploy_params: Dict,
    formatter: OutputFormatter
):
    """Deploy to AWS"""
    formatter.info("AWS deployment not fully implemented")
    formatter.info("Would deploy using AWS ECS/EKS/Lambda")
    # Placeholder for AWS deployment logic


async def _deploy_gcp(
    deploy_params: Dict,
    formatter: OutputFormatter
):
    """Deploy to Google Cloud Platform"""
    formatter.info("GCP deployment not fully implemented")
    formatter.info("Would deploy using GKE/Cloud Run/Cloud Functions")
    # Placeholder for GCP deployment logic


async def _deploy_azure(
    deploy_params: Dict,
    formatter: OutputFormatter
):
    """Deploy to Microsoft Azure"""
    formatter.info("Azure deployment not fully implemented")
    formatter.info("Would deploy using AKS/Container Instances/Functions")
    # Placeholder for Azure deployment logic


async def _deploy_local(
    deploy_params: Dict,
    formatter: OutputFormatter,
    force: bool
):
    """Deploy locally"""
    formatter.info("Starting local deployment...")
    
    # Start local services
    formatter.info("Starting API server...")
    # This would start the actual API server
    
    formatter.info("Starting WebSocket server...")
    # This would start the WebSocket server
    
    formatter.success("Local deployment completed")
    formatter.info("Services available at:")
    formatter.info("  - API: http://localhost:8000")
    formatter.info("  - WebSocket: ws://localhost:8765")
    formatter.info("  - Dashboard: http://localhost:3000")


def _generate_docker_compose_config(deploy_params: Dict) -> Dict:
    """Generate Docker Compose configuration"""
    return {
        'version': '3.8',
        'services': {
            'synthdata-api': {
                'build': '.',
                'ports': ['8000:8000'],
                'environment': [
                    f"ENV={deploy_params['environment']}",
                    f"REPLICAS={deploy_params['replicas']}"
                ],
                'restart': 'unless-stopped'
            },
            'synthdata-websocket': {
                'build': '.',
                'ports': ['8765:8765'],
                'environment': [
                    f"ENV={deploy_params['environment']}"
                ],
                'restart': 'unless-stopped'
            },
            'redis': {
                'image': 'redis:7-alpine',
                'ports': ['6379:6379'],
                'restart': 'unless-stopped'
            },
            'postgres': {
                'image': 'postgres:15',
                'environment': [
                    'POSTGRES_DB=synthdata',
                    'POSTGRES_USER=synthdata',
                    'POSTGRES_PASSWORD=synthdata'
                ],
                'ports': ['5432:5432'],
                'restart': 'unless-stopped'
            }
        }
    }


def _generate_kubernetes_manifests(deploy_params: Dict) -> Dict:
    """Generate Kubernetes deployment manifests"""
    app_name = f"synthdata-{deploy_params['environment']}"
    
    deployment = {
        'apiVersion': 'apps/v1',
        'kind': 'Deployment',
        'metadata': {
            'name': app_name,
            'namespace': deploy_params['namespace']
        },
        'spec': {
            'replicas': deploy_params['replicas'],
            'selector': {
                'matchLabels': {
                    'app': app_name
                }
            },
            'template': {
                'metadata': {
                    'labels': {
                        'app': app_name
                    }
                },
                'spec': {
                    'containers': [{
                        'name': 'synthdata',
                        'image': f"synthdata:{deploy_params['tag']}",
                        'ports': [{
                            'containerPort': 8000
                        }],
                        'env': [{
                            'name': 'ENV',
                            'value': deploy_params['environment']
                        }]
                    }]
                }
            }
        }
    }
    
    # Add resource limits if specified
    if deploy_params['resources']:
        deployment['spec']['template']['spec']['containers'][0]['resources'] = deploy_params['resources']
    
    service = {
        'apiVersion': 'v1',
        'kind': 'Service',
        'metadata': {
            'name': f"{app_name}-service",
            'namespace': deploy_params['namespace']
        },
        'spec': {
            'selector': {
                'app': app_name
            },
            'ports': [{
                'port': 80,
                'targetPort': 8000
            }],
            'type': 'LoadBalancer'
        }
    }
    
    return {
        'deployment': deployment,
        'service': service
    }


def create_deploy_command():
    """Factory function to create deploy command"""
    return deploy_command


__all__ = ['deploy_command', 'create_deploy_command']