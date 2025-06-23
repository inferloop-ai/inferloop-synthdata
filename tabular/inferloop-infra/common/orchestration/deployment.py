"""
Deployment orchestration for multi-cloud infrastructure
"""

import asyncio
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import yaml
import logging

from ..abstractions.base import BaseProvider, ResourceState
from .lifecycle import ResourceLifecycleManager, LifecycleState
from .provider_factory import ProviderFactory


logger = logging.getLogger(__name__)


class DeploymentState(Enum):
    """Deployment states"""
    PENDING = "pending"
    VALIDATING = "validating"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    UPDATING = "updating"
    DELETING = "deleting"
    DELETED = "deleted"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class DeploymentStrategy(Enum):
    """Deployment strategies"""
    ALL_AT_ONCE = "all_at_once"
    ROLLING = "rolling"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"


@dataclass
class DeploymentConfig:
    """Configuration for a deployment"""
    name: str
    version: str
    provider: str
    region: str
    environment: str = "development"
    strategy: DeploymentStrategy = DeploymentStrategy.ALL_AT_ONCE
    resources: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    rollback_on_failure: bool = True
    dry_run: bool = False
    
    @classmethod
    def from_file(cls, file_path: str) -> 'DeploymentConfig':
        """Load deployment config from file"""
        with open(file_path, 'r') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        return cls(**data)
    
    def validate(self) -> List[str]:
        """Validate deployment configuration"""
        errors = []
        if not self.name:
            errors.append("Deployment name is required")
        if not self.version:
            errors.append("Deployment version is required")
        if not self.provider:
            errors.append("Provider is required")
        if not self.region:
            errors.append("Region is required")
        if not self.resources:
            errors.append("At least one resource is required")
        
        # Validate dependencies
        resource_names = set(self.resources.keys())
        for resource, deps in self.dependencies.items():
            if resource not in resource_names:
                errors.append(f"Unknown resource in dependencies: {resource}")
            for dep in deps:
                if dep not in resource_names:
                    errors.append(f"Unknown dependency {dep} for resource {resource}")
        
        return errors


@dataclass
class DeploymentStatus:
    """Status of a deployment"""
    deployment_id: str
    name: str
    version: str
    state: DeploymentState
    provider: str
    region: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    resources: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DeploymentOrchestrator:
    """Orchestrates deployments across multiple cloud providers"""
    
    def __init__(self):
        self.provider_factory = ProviderFactory()
        self.lifecycle_manager = ResourceLifecycleManager()
        self.deployments: Dict[str, DeploymentStatus] = {}
        
    async def deploy(self, config: DeploymentConfig) -> DeploymentStatus:
        """Deploy infrastructure based on configuration"""
        # Validate configuration
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid deployment configuration: {errors}")
        
        # Create deployment status
        deployment_id = f"{config.name}-{config.version}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        status = DeploymentStatus(
            deployment_id=deployment_id,
            name=config.name,
            version=config.version,
            state=DeploymentState.PENDING,
            provider=config.provider,
            region=config.region,
            started_at=datetime.now()
        )
        self.deployments[deployment_id] = status
        
        try:
            # Update state
            status.state = DeploymentState.VALIDATING
            
            # Get provider
            provider = await self._get_provider(config.provider, config.region)
            
            # Validate provider capabilities
            await self._validate_provider_capabilities(provider, config)
            
            # Create deployment plan
            plan = await self._create_deployment_plan(config)
            
            if config.dry_run:
                status.metadata['plan'] = plan
                status.state = DeploymentState.DEPLOYED
                status.completed_at = datetime.now()
                return status
            
            # Execute deployment
            status.state = DeploymentState.DEPLOYING
            await self._execute_deployment(provider, config, plan, status)
            
            status.state = DeploymentState.DEPLOYED
            status.completed_at = datetime.now()
            
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            status.errors.append(str(e))
            status.state = DeploymentState.FAILED
            
            if config.rollback_on_failure:
                await self._rollback_deployment(provider, status)
                status.state = DeploymentState.ROLLED_BACK
            
            raise
        
        return status
    
    async def update(self, deployment_id: str, config: DeploymentConfig) -> DeploymentStatus:
        """Update existing deployment"""
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        current_status = self.deployments[deployment_id]
        if current_status.state != DeploymentState.DEPLOYED:
            raise ValueError(f"Can only update deployed deployments, current state: {current_status.state}")
        
        # Create new deployment status for update
        update_id = f"{deployment_id}-update-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        status = DeploymentStatus(
            deployment_id=update_id,
            name=config.name,
            version=config.version,
            state=DeploymentState.UPDATING,
            provider=config.provider,
            region=config.region,
            started_at=datetime.now()
        )
        self.deployments[update_id] = status
        
        try:
            provider = await self._get_provider(config.provider, config.region)
            
            # Create update plan
            plan = await self._create_update_plan(current_status, config)
            
            # Execute update
            await self._execute_update(provider, config, plan, status)
            
            status.state = DeploymentState.DEPLOYED
            status.completed_at = datetime.now()
            
        except Exception as e:
            logger.error(f"Update failed: {str(e)}")
            status.errors.append(str(e))
            status.state = DeploymentState.FAILED
            raise
        
        return status
    
    async def delete(self, deployment_id: str) -> bool:
        """Delete deployment"""
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        status = self.deployments[deployment_id]
        if status.state == DeploymentState.DELETED:
            return True
        
        status.state = DeploymentState.DELETING
        
        try:
            provider = await self._get_provider(status.provider, status.region)
            
            # Delete resources in reverse dependency order
            resource_order = self._get_deletion_order(status.resources)
            
            for resource_name in resource_order:
                resource_info = status.resources.get(resource_name, {})
                if resource_info.get('id'):
                    await self._delete_resource(provider, resource_info)
            
            status.state = DeploymentState.DELETED
            status.completed_at = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Deletion failed: {str(e)}")
            status.errors.append(str(e))
            status.state = DeploymentState.FAILED
            raise
    
    async def get_status(self, deployment_id: str) -> DeploymentStatus:
        """Get deployment status"""
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        return self.deployments[deployment_id]
    
    async def list_deployments(self, provider: Optional[str] = None,
                             state: Optional[DeploymentState] = None) -> List[DeploymentStatus]:
        """List deployments with optional filters"""
        deployments = list(self.deployments.values())
        
        if provider:
            deployments = [d for d in deployments if d.provider == provider]
        
        if state:
            deployments = [d for d in deployments if d.state == state]
        
        return deployments
    
    async def _get_provider(self, provider_name: str, region: str) -> BaseProvider:
        """Get provider instance"""
        return await self.provider_factory.get_provider(provider_name, {'region': region})
    
    async def _validate_provider_capabilities(self, provider: BaseProvider, 
                                            config: DeploymentConfig) -> None:
        """Validate that provider supports required resources"""
        capabilities = provider.get_capabilities()
        required_types = {res['type'] for res in config.resources.values()}
        
        unsupported = required_types - set(capabilities.get('resources', []))
        if unsupported:
            raise ValueError(f"Provider does not support resource types: {unsupported}")
    
    async def _create_deployment_plan(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Create deployment execution plan"""
        # Topological sort based on dependencies
        sorted_resources = self._topological_sort(config.resources.keys(), config.dependencies)
        
        plan = {
            'order': sorted_resources,
            'resources': config.resources,
            'parameters': config.parameters
        }
        
        return plan
    
    async def _create_update_plan(self, current: DeploymentStatus, 
                                 new_config: DeploymentConfig) -> Dict[str, Any]:
        """Create update plan by comparing current and new configurations"""
        plan = {
            'create': {},
            'update': {},
            'delete': {},
            'order': []
        }
        
        current_resources = set(current.resources.keys())
        new_resources = set(new_config.resources.keys())
        
        # Resources to create
        for name in new_resources - current_resources:
            plan['create'][name] = new_config.resources[name]
        
        # Resources to update
        for name in current_resources & new_resources:
            if current.resources[name] != new_config.resources[name]:
                plan['update'][name] = new_config.resources[name]
        
        # Resources to delete
        for name in current_resources - new_resources:
            plan['delete'][name] = current.resources[name]
        
        # Determine execution order
        plan['order'] = self._topological_sort(new_resources, new_config.dependencies)
        
        return plan
    
    async def _execute_deployment(self, provider: BaseProvider, config: DeploymentConfig,
                                 plan: Dict[str, Any], status: DeploymentStatus) -> None:
        """Execute deployment plan"""
        for resource_name in plan['order']:
            resource_config = plan['resources'][resource_name]
            
            try:
                # Create resource
                resource = await self._create_resource(provider, resource_config)
                
                # Update status
                status.resources[resource_name] = {
                    'id': resource.id,
                    'type': resource_config['type'],
                    'state': resource.state,
                    'config': resource_config
                }
                
                # Wait for resource to be ready
                await self._wait_for_resource(provider, resource)
                
            except Exception as e:
                logger.error(f"Failed to create resource {resource_name}: {str(e)}")
                raise
    
    async def _execute_update(self, provider: BaseProvider, config: DeploymentConfig,
                            plan: Dict[str, Any], status: DeploymentStatus) -> None:
        """Execute update plan"""
        # Delete removed resources
        for name, resource in plan['delete'].items():
            await self._delete_resource(provider, resource)
        
        # Update existing resources
        for name, resource_config in plan['update'].items():
            await self._update_resource(provider, resource_config)
        
        # Create new resources
        for name, resource_config in plan['create'].items():
            resource = await self._create_resource(provider, resource_config)
            status.resources[name] = {
                'id': resource.id,
                'type': resource_config['type'],
                'state': resource.state,
                'config': resource_config
            }
    
    async def _rollback_deployment(self, provider: BaseProvider, 
                                  status: DeploymentStatus) -> None:
        """Rollback failed deployment"""
        logger.info(f"Rolling back deployment {status.deployment_id}")
        
        # Delete created resources in reverse order
        resource_order = list(reversed(list(status.resources.keys())))
        
        for resource_name in resource_order:
            resource_info = status.resources.get(resource_name, {})
            if resource_info.get('id'):
                try:
                    await self._delete_resource(provider, resource_info)
                except Exception as e:
                    logger.error(f"Failed to rollback resource {resource_name}: {str(e)}")
    
    def _topological_sort(self, nodes: Set[str], dependencies: Dict[str, List[str]]) -> List[str]:
        """Perform topological sort on resources based on dependencies"""
        visited = set()
        stack = []
        
        def visit(node: str):
            if node in visited:
                return
            visited.add(node)
            for dep in dependencies.get(node, []):
                visit(dep)
            stack.append(node)
        
        for node in nodes:
            visit(node)
        
        return list(reversed(stack))
    
    def _get_deletion_order(self, resources: Dict[str, Any]) -> List[str]:
        """Get resource deletion order (reverse of creation order)"""
        return list(reversed(list(resources.keys())))
    
    async def _create_resource(self, provider: BaseProvider, 
                             resource_config: Dict[str, Any]) -> Any:
        """Create individual resource"""
        resource_type = resource_config['type']
        config_class = self._get_config_class(resource_type)
        config = config_class(**resource_config['config'])
        
        resource_manager = self._get_resource_manager(provider, resource_type)
        return await resource_manager.create(config)
    
    async def _update_resource(self, provider: BaseProvider,
                             resource_config: Dict[str, Any]) -> Any:
        """Update individual resource"""
        resource_type = resource_config['type']
        resource_id = resource_config['id']
        config_class = self._get_config_class(resource_type)
        config = config_class(**resource_config['config'])
        
        resource_manager = self._get_resource_manager(provider, resource_type)
        return await resource_manager.update(resource_id, config)
    
    async def _delete_resource(self, provider: BaseProvider,
                             resource_info: Dict[str, Any]) -> bool:
        """Delete individual resource"""
        resource_type = resource_info['type']
        resource_id = resource_info['id']
        
        resource_manager = self._get_resource_manager(provider, resource_type)
        return await resource_manager.delete(resource_id)
    
    async def _wait_for_resource(self, provider: BaseProvider, resource: Any) -> None:
        """Wait for resource to be ready"""
        max_wait = 300  # 5 minutes
        resource_manager = self._get_resource_manager(provider, resource.type)
        
        await resource_manager.wait_for_state(
            resource.id, 
            ResourceState.RUNNING,
            timeout=max_wait
        )
    
    def _get_config_class(self, resource_type: str) -> type:
        """Get configuration class for resource type"""
        # Import and return appropriate config class based on resource type
        # This would be implemented based on actual resource types
        pass
    
    def _get_resource_manager(self, provider: BaseProvider, resource_type: str) -> Any:
        """Get resource manager for specific resource type"""
        # Return appropriate resource manager from provider
        # This would be implemented based on actual provider structure
        pass