"""Tests for on-premises Kubernetes provider."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
from pathlib import Path

from deploy.onprem.provider import OnPremKubernetesProvider
from deploy.base import ResourceConfig, DeploymentResult


class TestOnPremKubernetesProvider:
    """Test OnPremKubernetesProvider class."""
    
    @pytest.fixture
    def mock_subprocess(self):
        """Mock subprocess for kubectl commands."""
        with patch('deploy.onprem.provider.subprocess.run') as mock_run:
            yield mock_run
    
    @pytest.fixture
    def mock_helm_deployer(self):
        """Mock Helm deployer."""
        with patch('deploy.onprem.provider.HelmDeployer') as mock_helm:
            mock_instance = MagicMock()
            mock_helm.return_value = mock_instance
            yield mock_instance
    
    @pytest.fixture
    def provider(self, mock_subprocess, mock_helm_deployer):
        """Create provider with mocked dependencies."""
        # Mock successful cluster access validation
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "Kubernetes control plane is running"
        
        return OnPremKubernetesProvider()
    
    def test_provider_initialization(self, provider):
        """Test provider initialization."""
        assert provider.kubeconfig_path.endswith(".kube/config")
        assert "kubectl" in provider.kubectl_base_cmd
        assert provider.helm_deployer is not None
    
    def test_validate_cluster_access_success(self, mock_subprocess):
        """Test successful cluster access validation."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "Kubernetes control plane is running"
        
        # Should not raise exception
        provider = OnPremKubernetesProvider()
        assert provider is not None
    
    def test_validate_cluster_access_failure(self, mock_subprocess):
        """Test failed cluster access validation."""
        mock_subprocess.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception):
            OnPremKubernetesProvider()
    
    def test_authenticate_success(self, provider, mock_subprocess):
        """Test successful authentication."""
        # Mock successful namespace operations
        mock_subprocess.return_value.returncode = 0
        
        result = provider.authenticate(namespace="test-ns")
        
        assert result is True
        assert provider._authenticated is True
        assert provider.default_namespace == "test-ns"
    
    def test_authenticate_with_context(self, provider, mock_subprocess):
        """Test authentication with context switching."""
        mock_subprocess.return_value.returncode = 0
        
        result = provider.authenticate(context="test-context", namespace="test-ns")
        
        assert result is True
        # Should have called kubectl config use-context
        context_call = any("use-context" in str(call) for call in mock_subprocess.call_args_list)
        assert context_call
    
    def test_deploy_container_without_auth(self, provider):
        """Test container deployment without authentication."""
        provider._authenticated = False
        
        config = ResourceConfig(
            metadata={"name": "test-app", "image": "nginx:latest"}
        )
        
        result = provider.deploy_container(config)
        
        assert result.success is False
        assert "not authenticated" in result.message
    
    def test_deploy_container_success(self, provider, mock_subprocess):
        """Test successful container deployment."""
        provider._authenticated = True
        
        # Mock successful kubectl operations
        mock_subprocess.return_value.returncode = 0
        
        # Mock deployment ready check
        deployment_status = {
            "status": {
                "replicas": 3,
                "readyReplicas": 3
            }
        }
        
        # Mock service endpoint
        service_status = {
            "spec": {
                "clusterIP": "10.96.0.1",
                "ports": [{"port": 80}]
            }
        }
        
        mock_subprocess.side_effect = [
            MagicMock(returncode=0),  # apply deployment
            MagicMock(returncode=0),  # apply service
            MagicMock(returncode=0, stdout=json.dumps(deployment_status)),  # get deployment
            MagicMock(returncode=0, stdout=json.dumps(service_status))  # get service
        ]
        
        config = ResourceConfig(
            compute={"count": 3, "cpu": "2", "memory": "4Gi"},
            metadata={"name": "test-app", "image": "nginx:latest"}
        )
        
        result = provider.deploy_container(config)
        
        assert result.success is True
        assert "test-app" in result.resource_id
        assert "10.96.0.1" in result.endpoint
    
    def test_deploy_storage_minio(self, provider, mock_subprocess):
        """Test MinIO storage deployment."""
        provider._authenticated = True
        mock_subprocess.return_value.returncode = 0
        
        # Mock successful StatefulSet ready check
        statefulset_status = {
            "status": {
                "replicas": 4,
                "readyReplicas": 4
            }
        }
        mock_subprocess.side_effect = [
            MagicMock(returncode=0),  # apply secret
            MagicMock(returncode=0),  # apply statefulset
            MagicMock(returncode=0),  # apply service
            MagicMock(returncode=0, stdout=json.dumps(statefulset_status))  # get statefulset
        ]
        
        config = ResourceConfig(
            compute={"count": 4},
            storage={"size": "100Gi", "storage_class": "standard"},
            metadata={
                "storage_type": "minio",
                "access_key": "admin",
                "secret_key": "admin123"
            }
        )
        
        result = provider.deploy_storage(config)
        
        assert result.success is True
        assert "minio" in result.endpoint
    
    def test_deploy_database_postgresql(self, provider, mock_subprocess):
        """Test PostgreSQL database deployment."""
        provider._authenticated = True
        mock_subprocess.return_value.returncode = 0
        
        # Mock successful StatefulSet ready check
        statefulset_status = {
            "status": {
                "replicas": 1,
                "readyReplicas": 1
            }
        }
        mock_subprocess.side_effect = [
            MagicMock(returncode=0),  # apply secret
            MagicMock(returncode=0),  # apply configmap
            MagicMock(returncode=0),  # apply statefulset
            MagicMock(returncode=0),  # apply service
            MagicMock(returncode=0, stdout=json.dumps(statefulset_status))  # get statefulset
        ]
        
        config = ResourceConfig(
            compute={"cpu": "2", "memory": "4Gi"},
            storage={"size": "100Gi", "storage_class": "standard"},
            metadata={
                "db_type": "postgresql",
                "name": "postgres",
                "password": "secret123"
            }
        )
        
        result = provider.deploy_database(config)
        
        assert result.success is True
        assert "postgresql://" in result.endpoint
        assert "postgres" in result.resource_id
    
    def test_deploy_monitoring_success(self, provider, mock_subprocess):
        """Test monitoring stack deployment."""
        provider._authenticated = True
        
        # Mock successful deployments
        deployment_status = {
            "status": {
                "replicas": 1,
                "readyReplicas": 1
            }
        }
        
        mock_subprocess.side_effect = [
            MagicMock(returncode=0),  # create namespace
            MagicMock(returncode=0),  # apply prometheus config
            MagicMock(returncode=0),  # apply prometheus SA
            MagicMock(returncode=0),  # apply prometheus role
            MagicMock(returncode=0),  # apply prometheus binding
            MagicMock(returncode=0),  # apply prometheus deployment
            MagicMock(returncode=0),  # apply prometheus service
            MagicMock(returncode=0, stdout=json.dumps(deployment_status)),  # wait prometheus
            MagicMock(returncode=0),  # apply grafana deployment
            MagicMock(returncode=0),  # apply grafana service
            MagicMock(returncode=0, stdout=json.dumps(deployment_status))  # wait grafana
        ]
        
        config = ResourceConfig(
            metadata={"namespace": "monitoring"}
        )
        
        result = provider.deploy_monitoring(config)
        
        assert result.success is True
        assert "prometheus_url" in result.metadata
        assert "grafana_url" in result.metadata
    
    def test_deploy_with_helm_success(self, provider, mock_helm_deployer):
        """Test Helm deployment."""
        provider._authenticated = True
        
        # Mock successful Helm deployment
        helm_result = DeploymentResult()
        helm_result.success = True
        helm_result.resource_id = "test-ns/test-app"
        mock_helm_deployer.install.return_value = helm_result
        
        config = ResourceConfig(
            metadata={
                "name": "test-app",
                "namespace": "test-ns",
                "version": "1.0.0"
            }
        )
        
        result = provider.deploy_with_helm(config)
        
        assert result.success is True
        mock_helm_deployer.install.assert_called_once()
    
    def test_upgrade_with_helm_success(self, provider, mock_helm_deployer):
        """Test Helm upgrade."""
        provider._authenticated = True
        
        # Mock successful Helm upgrade
        helm_result = DeploymentResult()
        helm_result.success = True
        mock_helm_deployer.upgrade.return_value = helm_result
        
        config = ResourceConfig(
            metadata={
                "name": "test-app",
                "version": "2.0.0"
            }
        )
        
        result = provider.upgrade_with_helm(config)
        
        assert result.success is True
        mock_helm_deployer.upgrade.assert_called_once()
    
    def test_list_deployments_success(self, provider, mock_subprocess):
        """Test listing deployments."""
        provider._authenticated = True
        
        # Mock deployments list
        deployments_data = {
            "items": [
                {
                    "metadata": {
                        "name": "test-app",
                        "namespace": "default",
                        "creationTimestamp": "2024-01-01T00:00:00Z"
                    },
                    "status": {
                        "replicas": 3,
                        "readyReplicas": 3
                    }
                }
            ]
        }
        
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = json.dumps(deployments_data)
        
        deployments = provider.list_deployments()
        
        assert len(deployments) == 1
        assert deployments[0]["name"] == "test-app"
        assert deployments[0]["ready"] == "3/3"
    
    def test_delete_deployment_success(self, provider, mock_subprocess):
        """Test successful deployment deletion."""
        provider._authenticated = True
        mock_subprocess.return_value.returncode = 0
        
        result = provider.delete_deployment("default/test-app")
        
        assert result is True
    
    def test_get_deployment_status(self, provider, mock_subprocess):
        """Test getting deployment status."""
        provider._authenticated = True
        
        # Mock deployment status
        deployment_data = {
            "metadata": {"name": "test-app", "namespace": "default"},
            "status": {
                "replicas": 3,
                "readyReplicas": 2,
                "availableReplicas": 2,
                "conditions": []
            }
        }
        
        pods_data = {
            "items": [
                {
                    "metadata": {"name": "test-app-pod-1"},
                    "status": {
                        "phase": "Running",
                        "conditions": [{"type": "Ready", "status": "True"}]
                    }
                }
            ]
        }
        
        mock_subprocess.side_effect = [
            MagicMock(returncode=0, stdout=json.dumps(deployment_data)),
            MagicMock(returncode=0, stdout=json.dumps(pods_data))
        ]
        
        status = provider.get_deployment_status("default/test-app")
        
        assert status["deployment"] == "test-app"
        assert status["namespace"] == "default"
        assert status["replicas"]["desired"] == 3
        assert status["replicas"]["ready"] == 2
        assert len(status["pods"]) == 1
    
    def test_validate_success(self, provider, mock_subprocess):
        """Test successful validation."""
        mock_subprocess.side_effect = [
            MagicMock(returncode=0, stdout="Client Version: v1.28.0"),  # cluster-info
            MagicMock(returncode=0, stdout='{"serverVersion": {"major": "1", "minor": "28"}}'),  # version
            MagicMock(returncode=0, stdout='{"items": [{"metadata": {"name": "standard"}}]}'),  # storage classes
            MagicMock(returncode=0, stdout='{"items": [{"status": {"capacity": {"cpu": "16", "memory": "64Gi"}}}]}')  # nodes
        ]
        
        is_valid, issues = provider.validate()
        
        assert is_valid is True
        assert len(issues) == 0
    
    def test_validate_old_kubernetes(self, provider, mock_subprocess):
        """Test validation with old Kubernetes version."""
        mock_subprocess.side_effect = [
            MagicMock(returncode=0),  # cluster-info
            MagicMock(returncode=0, stdout='{"serverVersion": {"major": "1", "minor": "18"}}'),  # old version
        ]
        
        is_valid, issues = provider.validate()
        
        assert is_valid is False
        assert any("version" in issue for issue in issues)
    
    def test_validate_insufficient_resources(self, provider, mock_subprocess):
        """Test validation with insufficient resources."""
        mock_subprocess.side_effect = [
            MagicMock(returncode=0),  # cluster-info
            MagicMock(returncode=0, stdout='{"serverVersion": {"major": "1", "minor": "28"}}'),  # version
            MagicMock(returncode=0, stdout='{"items": []}'),  # no storage classes
            MagicMock(returncode=0, stdout='{"items": [{"status": {"capacity": {"cpu": "8", "memory": "32Gi"}}}]}')  # small nodes
        ]
        
        is_valid, issues = provider.validate()
        
        assert is_valid is False
        assert any("storage classes" in issue for issue in issues)
        assert any("CPU" in issue for issue in issues)
        assert any("memory" in issue for issue in issues


class TestHelperMethods:
    """Test helper methods."""
    
    @pytest.fixture
    def provider(self):
        """Create provider without validation."""
        with patch('deploy.onprem.provider.OnPremKubernetesProvider._validate_cluster_access'):
            return OnPremKubernetesProvider()
    
    def test_create_deployment_manifest(self, provider):
        """Test deployment manifest creation."""
        config = ResourceConfig(
            compute={"count": 3, "cpu": "2", "memory": "4Gi", "gpu": 1},
            storage={"size": "100Gi"},
            metadata={
                "name": "test-app",
                "namespace": "test-ns",
                "image": "nginx:latest",
                "env": {"KEY": "value"}
            }
        )
        
        manifest = provider._create_deployment_manifest(config)
        
        assert manifest["kind"] == "Deployment"
        assert manifest["metadata"]["name"] == "test-app"
        assert manifest["metadata"]["namespace"] == "test-ns"
        assert manifest["spec"]["replicas"] == 3
        
        container = manifest["spec"]["template"]["spec"]["containers"][0]
        assert container["image"] == "nginx:latest"
        assert container["resources"]["requests"]["cpu"] == "2"
        assert container["resources"]["requests"]["memory"] == "4Gi"
        assert container["resources"]["requests"]["nvidia.com/gpu"] == "1"
        
        # Check environment variables
        env_vars = container["env"]
        env_names = [env["name"] for env in env_vars]
        assert "KEY" in env_names
    
    def test_create_service_manifest(self, provider):
        """Test service manifest creation."""
        config = ResourceConfig(
            networking={"service_type": "LoadBalancer"},
            metadata={"name": "test-app", "namespace": "test-ns"}
        )
        
        manifest = provider._create_service_manifest(config)
        
        assert manifest["kind"] == "Service"
        assert manifest["metadata"]["name"] == "test-app"
        assert manifest["metadata"]["namespace"] == "test-ns"
        assert manifest["spec"]["type"] == "LoadBalancer"
        assert manifest["spec"]["ports"][0]["port"] == 80
        assert manifest["spec"]["ports"][0]["targetPort"] == 8000
    
    def test_create_env_vars(self, provider):
        """Test environment variables creation."""
        config = ResourceConfig(
            metadata={
                "database_url": "postgres://localhost/db",
                "s3_endpoint": "http://minio:9000",
                "s3_bucket": "test-bucket",
                "env": {"CUSTOM_VAR": "custom_value", "ANOTHER_VAR": "another_value"}
            }
        )
        
        env_vars = provider._create_env_vars(config)
        
        # Check for database URL secret reference
        db_env = next((env for env in env_vars if env["name"] == "DATABASE_URL"), None)
        assert db_env is not None
        assert "valueFrom" in db_env
        assert "secretKeyRef" in db_env["valueFrom"]
        
        # Check for S3 configuration
        s3_endpoint_env = next((env for env in env_vars if env["name"] == "S3_ENDPOINT"), None)
        assert s3_endpoint_env is not None
        assert s3_endpoint_env["value"] == "http://minio:9000"
        
        s3_bucket_env = next((env for env in env_vars if env["name"] == "S3_BUCKET"), None)
        assert s3_bucket_env is not None
        assert s3_bucket_env["value"] == "test-bucket"
        
        # Check for custom environment variables
        custom_env = next((env for env in env_vars if env["name"] == "CUSTOM_VAR"), None)
        assert custom_env is not None
        assert custom_env["value"] == "custom_value"
        
        another_env = next((env for env in env_vars if env["name"] == "ANOTHER_VAR"), None)
        assert another_env is not None
        assert another_env["value"] == "another_value"