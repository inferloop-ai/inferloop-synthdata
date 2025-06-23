"""
Tests for deployment orchestration
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.orchestration import (
    DeploymentOrchestrator,
    DeploymentConfig,
    DeploymentState,
    DeploymentStrategy,
    ProviderFactory,
    ResourceLifecycleManager,
    TemplateEngine
)
from common.abstractions.base import BaseProvider, ResourceState


@pytest.fixture
def sample_config():
    """Sample deployment configuration"""
    return DeploymentConfig(
        name="test-deployment",
        version="1.0.0",
        provider="aws",
        region="us-east-1",
        environment="test",
        resources={
            "web_server": {
                "type": "compute",
                "config": {
                    "name": "test-web-server",
                    "cpu": 2,
                    "memory": 4096
                }
            },
            "database": {
                "type": "database",
                "config": {
                    "name": "test-db",
                    "engine": "postgresql",
                    "instance_class": "db.t3.micro"
                }
            }
        },
        dependencies={
            "web_server": ["database"]
        }
    )


@pytest.fixture
def mock_provider():
    """Mock provider for testing"""
    provider = Mock(spec=BaseProvider)
    provider.get_capabilities.return_value = {
        "resources": ["compute", "database", "storage", "network"]
    }
    provider.authenticate = AsyncMock(return_value=True)
    provider.validate_credentials = AsyncMock(return_value=True)
    provider.health_check = AsyncMock(return_value={"healthy": True})
    return provider


class TestDeploymentConfig:
    """Test deployment configuration"""
    
    def test_config_creation(self, sample_config):
        assert sample_config.name == "test-deployment"
        assert sample_config.version == "1.0.0"
        assert sample_config.provider == "aws"
        assert len(sample_config.resources) == 2
    
    def test_config_validation(self, sample_config):
        errors = sample_config.validate()
        assert len(errors) == 0
    
    def test_config_validation_missing_name(self):
        config = DeploymentConfig(
            name="",
            version="1.0.0",
            provider="aws",
            region="us-east-1",
            resources={}
        )
        errors = config.validate()
        assert "Deployment name is required" in errors
    
    def test_config_validation_missing_resources(self):
        config = DeploymentConfig(
            name="test",
            version="1.0.0",
            provider="aws",
            region="us-east-1",
            resources={}
        )
        errors = config.validate()
        assert "At least one resource is required" in errors
    
    def test_config_validation_invalid_dependencies(self):
        config = DeploymentConfig(
            name="test",
            version="1.0.0",
            provider="aws",
            region="us-east-1",
            resources={"web": {"type": "compute"}},
            dependencies={"web": ["non_existent"]}
        )
        errors = config.validate()
        assert any("Unknown dependency" in e for e in errors)


class TestDeploymentOrchestrator:
    """Test deployment orchestrator"""
    
    @pytest.mark.asyncio
    async def test_deploy_success(self, sample_config, mock_provider):
        orchestrator = DeploymentOrchestrator()
        
        # Mock provider factory
        with patch.object(orchestrator.provider_factory, 'get_provider', 
                         return_value=mock_provider):
            # Mock resource creation
            with patch.object(orchestrator, '_create_resource',
                            return_value=Mock(id="resource-123", state="running")):
                
                status = await orchestrator.deploy(sample_config)
                
                assert status.state == DeploymentState.DEPLOYED
                assert status.name == sample_config.name
                assert status.version == sample_config.version
                assert len(status.resources) == 2
    
    @pytest.mark.asyncio
    async def test_deploy_dry_run(self, sample_config, mock_provider):
        orchestrator = DeploymentOrchestrator()
        sample_config.dry_run = True
        
        with patch.object(orchestrator.provider_factory, 'get_provider',
                         return_value=mock_provider):
            status = await orchestrator.deploy(sample_config)
            
            assert status.state == DeploymentState.DEPLOYED
            assert 'plan' in status.metadata
    
    @pytest.mark.asyncio
    async def test_deploy_validation_failure(self):
        orchestrator = DeploymentOrchestrator()
        
        # Invalid config
        config = DeploymentConfig(
            name="",
            version="",
            provider="",
            region="",
            resources={}
        )
        
        with pytest.raises(ValueError, match="Invalid deployment configuration"):
            await orchestrator.deploy(config)
    
    @pytest.mark.asyncio
    async def test_deploy_provider_capability_check(self, sample_config):
        orchestrator = DeploymentOrchestrator()
        
        # Mock provider with limited capabilities
        mock_provider = Mock(spec=BaseProvider)
        mock_provider.get_capabilities.return_value = {"resources": ["compute"]}
        
        with patch.object(orchestrator.provider_factory, 'get_provider',
                         return_value=mock_provider):
            with pytest.raises(ValueError, match="Provider does not support resource types"):
                await orchestrator.deploy(sample_config)
    
    @pytest.mark.asyncio
    async def test_deploy_rollback_on_failure(self, sample_config, mock_provider):
        orchestrator = DeploymentOrchestrator()
        sample_config.rollback_on_failure = True
        
        with patch.object(orchestrator.provider_factory, 'get_provider',
                         return_value=mock_provider):
            # Make resource creation fail
            with patch.object(orchestrator, '_create_resource',
                            side_effect=Exception("Creation failed")):
                with patch.object(orchestrator, '_rollback_deployment',
                                return_value=None) as mock_rollback:
                    
                    with pytest.raises(Exception):
                        await orchestrator.deploy(sample_config)
                    
                    mock_rollback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_deployment(self, sample_config, mock_provider):
        orchestrator = DeploymentOrchestrator()
        deployment_id = "test-deployment-1.0.0-20240101120000"
        
        # Create initial deployment
        initial_status = Mock(
            deployment_id=deployment_id,
            state=DeploymentState.DEPLOYED,
            resources={"web_server": {"id": "web-123", "type": "compute"}}
        )
        orchestrator.deployments[deployment_id] = initial_status
        
        # Update config
        sample_config.resources["cache"] = {
            "type": "cache",
            "config": {"name": "test-cache"}
        }
        
        with patch.object(orchestrator.provider_factory, 'get_provider',
                         return_value=mock_provider):
            with patch.object(orchestrator, '_create_resource',
                            return_value=Mock(id="cache-123", state="running")):
                
                status = await orchestrator.update(deployment_id, sample_config)
                
                assert status.state == DeploymentState.DEPLOYED
    
    @pytest.mark.asyncio
    async def test_delete_deployment(self, mock_provider):
        orchestrator = DeploymentOrchestrator()
        deployment_id = "test-deployment-1.0.0-20240101120000"
        
        # Create deployment
        status = Mock(
            deployment_id=deployment_id,
            provider="aws",
            region="us-east-1",
            state=DeploymentState.DEPLOYED,
            resources={
                "web": {"id": "web-123", "type": "compute"},
                "db": {"id": "db-456", "type": "database"}
            }
        )
        orchestrator.deployments[deployment_id] = status
        
        with patch.object(orchestrator.provider_factory, 'get_provider',
                         return_value=mock_provider):
            with patch.object(orchestrator, '_delete_resource',
                            return_value=True) as mock_delete:
                
                result = await orchestrator.delete(deployment_id)
                
                assert result is True
                assert mock_delete.call_count == 2
                assert status.state == DeploymentState.DELETED
    
    def test_topological_sort(self):
        orchestrator = DeploymentOrchestrator()
        
        nodes = {"A", "B", "C", "D"}
        dependencies = {
            "A": ["B", "C"],
            "B": ["D"],
            "C": ["D"]
        }
        
        sorted_nodes = orchestrator._topological_sort(nodes, dependencies)
        
        # D should come before B and C, which should come before A
        assert sorted_nodes.index("D") < sorted_nodes.index("B")
        assert sorted_nodes.index("D") < sorted_nodes.index("C")
        assert sorted_nodes.index("B") < sorted_nodes.index("A")
        assert sorted_nodes.index("C") < sorted_nodes.index("A")


class TestProviderFactory:
    """Test provider factory"""
    
    @pytest.mark.asyncio
    async def test_get_provider(self, mock_provider):
        factory = ProviderFactory()
        
        with patch.object(factory, '_create_provider', return_value=mock_provider):
            provider = await factory.get_provider("aws", {"region": "us-east-1"})
            
            assert provider == mock_provider
            mock_provider.authenticate.assert_called_once()
            mock_provider.validate_credentials.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_provider_caching(self, mock_provider):
        factory = ProviderFactory()
        
        with patch.object(factory, '_create_provider', return_value=mock_provider) as mock_create:
            # First call
            provider1 = await factory.get_provider("aws", {"region": "us-east-1"})
            # Second call - should use cache
            provider2 = await factory.get_provider("aws", {"region": "us-east-1"})
            
            assert provider1 == provider2
            mock_create.assert_called_once()  # Only called once due to caching
    
    def test_list_providers(self):
        factory = ProviderFactory()
        providers = factory.list_providers()
        
        assert "aws" in providers
        assert "gcp" in providers
        assert "azure" in providers
        assert "onprem" in providers
    
    @pytest.mark.asyncio
    async def test_test_provider_connection(self, mock_provider):
        factory = ProviderFactory()
        
        with patch.object(factory, 'get_provider', return_value=mock_provider):
            result = await factory.test_provider_connection("aws", {"region": "us-east-1"})
            
            assert result['success'] is True
            assert result['provider'] == "aws"
            assert 'health' in result


class TestResourceLifecycleManager:
    """Test resource lifecycle manager"""
    
    def test_register_resource(self):
        manager = ResourceLifecycleManager()
        
        lifecycle = manager.register_resource("res-123", "compute")
        
        assert lifecycle.resource_id == "res-123"
        assert lifecycle.resource_type == "compute"
        assert lifecycle.current_state.value == "initializing"
    
    @pytest.mark.asyncio
    async def test_transition_state(self):
        manager = ResourceLifecycleManager()
        
        manager.register_resource("res-123", "compute")
        
        success = await manager.transition_state(
            "res-123",
            LifecycleState.READY,
            LifecycleEvent.CREATE
        )
        
        assert success is True
        
        lifecycle = manager.get_resource_lifecycle("res-123")
        assert lifecycle.current_state == LifecycleState.READY
        assert len(lifecycle.events) == 1
    
    def test_list_resources_with_filters(self):
        manager = ResourceLifecycleManager()
        
        manager.register_resource("res-1", "compute")
        manager.register_resource("res-2", "database")
        manager.register_resource("res-3", "compute")
        
        # Filter by type
        compute_resources = manager.list_resources(resource_type="compute")
        assert len(compute_resources) == 2
        
        # Filter by state
        init_resources = manager.list_resources(state=LifecycleState.INITIALIZING)
        assert len(init_resources) == 3
    
    @pytest.mark.asyncio
    async def test_health_monitoring(self):
        manager = ResourceLifecycleManager()
        
        mock_resource = Mock()
        mock_resource.get_state = AsyncMock(return_value=ResourceState.RUNNING)
        
        manager.register_resource("res-123", "compute")
        
        # Start health monitoring with short interval
        await manager.start_health_monitoring("res-123", mock_resource, interval=1)
        
        # Wait for health check
        await asyncio.sleep(1.5)
        
        # Stop monitoring
        manager.stop_health_monitoring("res-123")
        
        # Check health was updated
        lifecycle = manager.get_resource_lifecycle("res-123")
        assert 'healthy' in lifecycle.health_status
        assert lifecycle.health_status['healthy'] is True


class TestTemplateEngine:
    """Test template engine"""
    
    def test_render_simple_template(self):
        engine = TemplateEngine()
        
        template_data = {
            "resources": {
                "server": {
                    "name": "{{ environment }}-server",
                    "size": "{{ instance_size }}"
                }
            }
        }
        
        rendered = engine._render_dict(template_data, {
            "environment": "prod",
            "instance_size": "large"
        })
        
        assert rendered["resources"]["server"]["name"] == "prod-server"
        assert rendered["resources"]["server"]["size"] == "large"
    
    def test_render_with_functions(self):
        engine = TemplateEngine()
        
        template_data = {
            "id": "{{ uuid() }}",
            "timestamp": "{{ now() }}",
            "env_var": "{{ env('USER', 'default') }}"
        }
        
        rendered = engine._render_dict(template_data, {})
        
        assert len(rendered["id"]) == 36  # UUID format
        assert rendered["timestamp"]  # Should have timestamp
        assert rendered["env_var"]  # Should have environment variable or default


if __name__ == "__main__":
    pytest.main([__file__])