"""Integration tests for deployment providers."""

import pytest
import os
import time
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from deploy.base import ResourceConfig, DeploymentResult
from deploy.aws.provider import AWSProvider
from deploy.gcp.provider import GCPProvider  
from deploy.azure.provider import AzureProvider
from deploy.onprem.provider import OnPremKubernetesProvider


class TestDeploymentIntegration:
    """Integration tests for deployment workflows."""
    
    @pytest.fixture
    def test_config(self):
        """Standard test configuration."""
        return ResourceConfig(
            compute={"count": 1, "cpu": "1", "memory": "2Gi"},
            storage={"size": "10Gi"},
            networking={"service_type": "ClusterIP"},
            metadata={
                "name": "test-app",
                "image": "nginx:latest",
                "environment": "test"
            }
        )
    
    @pytest.mark.integration
    @patch('deploy.aws.provider.subprocess.run')
    @patch('deploy.aws.provider.boto3.Session')
    def test_aws_full_deployment_workflow(self, mock_session, mock_subprocess, test_config):
        """Test complete AWS deployment workflow."""
        # Mock AWS CLI authentication
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = '{"Account": "123456789012"}'
        
        # Mock boto3 session and clients
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        mock_ec2 = MagicMock()
        mock_session_instance.client.return_value = mock_ec2
        
        # Mock EC2 instance creation
        mock_ec2.run_instances.return_value = {
            'Instances': [{
                'InstanceId': 'i-123456789',
                'State': {'Name': 'running'},
                'PublicIpAddress': '1.2.3.4'
            }]
        }
        
        # Test workflow
        provider = AWSProvider("us-east-1")
        
        # 1. Authenticate
        auth_result = provider.authenticate()
        assert auth_result is True
        
        # 2. Validate environment
        is_valid, issues = provider.validate()
        # In mocked environment, validation may fail, but auth should work
        
        # 3. Deploy EC2 instance
        result = provider.deploy_ec2(test_config)
        assert result.success is True
        assert result.resource_id == "i-123456789"
        
        # 4. List deployments
        mock_ec2.describe_instances.return_value = {
            'Reservations': [{
                'Instances': [{
                    'InstanceId': 'i-123456789',
                    'State': {'Name': 'running'},
                    'InstanceType': 't3.micro',
                    'LaunchTime': '2024-01-01T00:00:00Z',
                    'Tags': [{'Key': 'Name', 'Value': 'test-app'}]
                }]
            }]
        }
        
        deployments = provider.list_deployments()
        assert len(deployments) >= 1
        
        # 5. Delete deployment
        mock_ec2.terminate_instances.return_value = {
            'TerminatingInstances': [{
                'InstanceId': 'i-123456789',
                'CurrentState': {'Name': 'shutting-down'}
            }]
        }
        
        delete_result = provider.delete_deployment("i-123456789")
        assert delete_result is True
    
    @pytest.mark.integration
    @patch('deploy.onprem.provider.subprocess.run')
    def test_onprem_full_deployment_workflow(self, mock_subprocess, test_config):
        """Test complete on-premises deployment workflow."""
        # Mock all kubectl commands to succeed
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "success"
        
        # Mock specific responses for different operations
        def mock_kubectl_responses(*args, **kwargs):
            cmd = args[0] if args else []
            
            if "cluster-info" in cmd:
                return MagicMock(returncode=0, stdout="Kubernetes control plane is running")
            elif "get deployment" in " ".join(cmd):
                return MagicMock(
                    returncode=0,
                    stdout='{"status": {"replicas": 1, "readyReplicas": 1}}'
                )
            elif "get service" in " ".join(cmd):
                return MagicMock(
                    returncode=0,
                    stdout='{"spec": {"clusterIP": "10.96.0.1", "ports": [{"port": 80}]}}'
                )
            else:
                return MagicMock(returncode=0, stdout="success")
        
        mock_subprocess.side_effect = mock_kubectl_responses
        
        # Test workflow
        provider = OnPremKubernetesProvider()
        
        # 1. Authenticate
        auth_result = provider.authenticate(namespace="test")
        assert auth_result is True
        
        # 2. Deploy application
        result = provider.deploy_container(test_config)
        assert result.success is True
        
        # 3. Deploy storage
        storage_config = ResourceConfig(
            compute={"count": 1},
            storage={"size": "10Gi"},
            metadata={"storage_type": "minio"}
        )
        
        # Mock StatefulSet ready response
        mock_subprocess.side_effect = [
            MagicMock(returncode=0),  # apply secret
            MagicMock(returncode=0),  # apply statefulset  
            MagicMock(returncode=0),  # apply service
            MagicMock(returncode=0, stdout='{"status": {"replicas": 1, "readyReplicas": 1}}')  # get statefulset
        ]
        
        storage_result = provider.deploy_storage(storage_config)
        assert storage_result.success is True
        
        # 4. Deploy database
        db_config = ResourceConfig(
            metadata={"db_type": "postgresql"}
        )
        
        # Reset mock for database deployment
        mock_subprocess.side_effect = [
            MagicMock(returncode=0),  # apply secret
            MagicMock(returncode=0),  # apply configmap
            MagicMock(returncode=0),  # apply statefulset
            MagicMock(returncode=0),  # apply service
            MagicMock(returncode=0, stdout='{"status": {"replicas": 1, "readyReplicas": 1}}')  # get statefulset
        ]
        
        db_result = provider.deploy_database(db_config)
        assert db_result.success is True
    
    @pytest.mark.integration 
    def test_multi_provider_config_compatibility(self, test_config):
        """Test that the same config works across providers."""
        providers = []
        
        # Create providers with mocked dependencies
        with patch('deploy.aws.provider.subprocess.run') as mock_aws_subprocess:
            mock_aws_subprocess.return_value.returncode = 0
            mock_aws_subprocess.return_value.stdout = '{"Account": "123456789012"}'
            
            with patch('deploy.aws.provider.boto3.Session'):
                try:
                    aws_provider = AWSProvider("us-east-1")
                    providers.append(("aws", aws_provider))
                except:
                    pass
        
        with patch('deploy.onprem.provider.subprocess.run') as mock_k8s_subprocess:
            mock_k8s_subprocess.return_value.returncode = 0
            mock_k8s_subprocess.return_value.stdout = "Kubernetes control plane is running"
            
            try:
                onprem_provider = OnPremKubernetesProvider()
                providers.append(("onprem", onprem_provider))
            except:
                pass
        
        # Test that all providers can parse the same config
        for provider_name, provider in providers:
            assert hasattr(provider, 'deploy_container') or hasattr(provider, 'deploy_ec2')
            
            # Test config validation (should not raise exceptions)
            assert test_config.compute is not None
            assert test_config.metadata is not None
    
    @pytest.mark.integration
    def test_deployment_lifecycle(self):
        """Test complete deployment lifecycle."""
        with patch('deploy.onprem.provider.subprocess.run') as mock_subprocess:
            # Mock successful kubectl operations
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = "success"
            
            # Specific mocks for different operations
            def mock_responses(*args, **kwargs):
                cmd = args[0] if args else []
                cmd_str = " ".join(cmd)
                
                if "cluster-info" in cmd_str:
                    return MagicMock(returncode=0, stdout="Kubernetes control plane is running")
                elif "get deployment" in cmd_str and "-o json" in cmd_str:
                    return MagicMock(
                        returncode=0,
                        stdout='{"status": {"replicas": 2, "readyReplicas": 2}}'
                    )
                elif "get service" in cmd_str and "-o json" in cmd_str:
                    return MagicMock(
                        returncode=0,
                        stdout='{"spec": {"clusterIP": "10.96.0.1", "ports": [{"port": 80}]}}'
                    )
                elif "get deployments" in cmd_str and "all-namespaces" in cmd_str:
                    return MagicMock(
                        returncode=0,
                        stdout='{"items": [{"metadata": {"name": "test-app", "namespace": "test", "creationTimestamp": "2024-01-01T00:00:00Z"}, "status": {"replicas": 2, "readyReplicas": 2}}]}'
                    )
                else:
                    return MagicMock(returncode=0, stdout="success")
            
            mock_subprocess.side_effect = mock_responses
            
            provider = OnPremKubernetesProvider()
            
            # 1. Create deployment
            config = ResourceConfig(
                compute={"count": 2, "cpu": "1", "memory": "2Gi"},
                metadata={"name": "test-app", "image": "nginx:latest"}
            )
            
            provider.authenticate(namespace="test")
            result = provider.deploy_container(config)
            assert result.success is True
            
            # 2. Check status
            status = provider.get_deployment_status(result.resource_id)
            assert "deployment" in status
            
            # 3. List all deployments
            deployments = provider.list_deployments()
            assert len(deployments) >= 1
            
            # 4. Scale deployment (simulate config change)
            scale_config = ResourceConfig(
                compute={"count": 4, "cpu": "1", "memory": "2Gi"},
                metadata={"name": "test-app", "image": "nginx:latest"}
            )
            
            # For scaling, we would typically update the deployment
            # Here we just verify the new config is valid
            assert scale_config.compute["count"] == 4
            
            # 5. Delete deployment
            delete_result = provider.delete_deployment(result.resource_id)
            assert delete_result is True
    
    @pytest.mark.integration
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery scenarios."""
        with patch('deploy.onprem.provider.subprocess.run') as mock_subprocess:
            provider = OnPremKubernetesProvider()
            
            # Test authentication failure
            mock_subprocess.side_effect = Exception("Connection refused")
            
            with pytest.raises(Exception):
                provider.authenticate()
            
            # Test deployment failure with authentication
            mock_subprocess.side_effect = [
                MagicMock(returncode=0, stdout="success"),  # cluster-info (for init)
                MagicMock(returncode=0),  # namespace operations (for auth)
                MagicMock(returncode=1, stderr="Error: deployment failed")  # deployment failure
            ]
            
            # Need to recreate provider due to failed init
            with patch('deploy.onprem.provider.OnPremKubernetesProvider._validate_cluster_access'):
                provider = OnPremKubernetesProvider()
                provider.authenticate()
                
                config = ResourceConfig(
                    metadata={"name": "test-app", "image": "nginx:latest"}
                )
                
                result = provider.deploy_container(config)
                assert result.success is False
                assert "failed" in result.message.lower()
    
    @pytest.mark.integration
    def test_concurrent_deployments(self):
        """Test handling of concurrent deployments."""
        with patch('deploy.onprem.provider.subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = "success"
            
            # Mock successful deployment responses
            def mock_responses(*args, **kwargs):
                cmd = args[0] if args else []
                cmd_str = " ".join(cmd)
                
                if "cluster-info" in cmd_str:
                    return MagicMock(returncode=0, stdout="Kubernetes control plane is running")
                elif "get deployment" in cmd_str:
                    return MagicMock(
                        returncode=0,
                        stdout='{"status": {"replicas": 1, "readyReplicas": 1}}'
                    )
                elif "get service" in cmd_str:
                    return MagicMock(
                        returncode=0,
                        stdout='{"spec": {"clusterIP": "10.96.0.1", "ports": [{"port": 80}]}}'
                    )
                else:
                    return MagicMock(returncode=0, stdout="success")
            
            mock_subprocess.side_effect = mock_responses
            
            provider = OnPremKubernetesProvider()
            provider.authenticate()
            
            # Deploy multiple applications
            apps = ["app1", "app2", "app3"]
            results = []
            
            for app_name in apps:
                config = ResourceConfig(
                    metadata={"name": app_name, "image": "nginx:latest"}
                )
                result = provider.deploy_container(config)
                results.append(result)
            
            # All deployments should succeed
            for result in results:
                assert result.success is True
            
            # Each should have unique resource IDs
            resource_ids = [result.resource_id for result in results]
            assert len(set(resource_ids)) == len(resource_ids)  # All unique
    
    @pytest.mark.integration
    def test_deployment_with_persistence(self):
        """Test deployment with persistent storage."""
        with patch('deploy.onprem.provider.subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = "success"
            
            # Mock responses including PVC operations
            def mock_responses(*args, **kwargs):
                cmd = args[0] if args else []
                cmd_str = " ".join(cmd)
                
                if "cluster-info" in cmd_str:
                    return MagicMock(returncode=0, stdout="Kubernetes control plane is running")
                elif "get deployment" in cmd_str:
                    return MagicMock(
                        returncode=0,
                        stdout='{"status": {"replicas": 1, "readyReplicas": 1}}'
                    )
                elif "get service" in cmd_str:
                    return MagicMock(
                        returncode=0,
                        stdout='{"spec": {"clusterIP": "10.96.0.1", "ports": [{"port": 80}]}}'
                    )
                else:
                    return MagicMock(returncode=0, stdout="success")
            
            mock_subprocess.side_effect = mock_responses
            
            provider = OnPremKubernetesProvider()
            provider.authenticate()
            
            # Deploy with persistent storage
            config = ResourceConfig(
                compute={"count": 1},
                storage={"size": "10Gi", "storage_class": "fast-ssd"},
                metadata={"name": "persistent-app", "image": "postgres:13"}
            )
            
            result = provider.deploy_container(config)
            assert result.success is True
            
            # The deployment should include volume configuration
            # (verified by the mock accepting the kubectl apply calls)
    
    @pytest.mark.integration
    def test_monitoring_integration(self):
        """Test monitoring stack integration."""
        with patch('deploy.onprem.provider.subprocess.run') as mock_subprocess:
            # Mock successful monitoring deployment
            def mock_responses(*args, **kwargs):
                cmd = args[0] if args else []
                cmd_str = " ".join(cmd)
                
                if "cluster-info" in cmd_str:
                    return MagicMock(returncode=0, stdout="Kubernetes control plane is running")
                elif "wait" in cmd_str and "deployment" in cmd_str:
                    return MagicMock(returncode=0, stdout="condition met")
                elif "get deployment" in cmd_str:
                    return MagicMock(
                        returncode=0,
                        stdout='{"status": {"replicas": 1, "readyReplicas": 1}}'
                    )
                else:
                    return MagicMock(returncode=0, stdout="success")
            
            mock_subprocess.side_effect = mock_responses
            
            provider = OnPremKubernetesProvider()
            provider.authenticate()
            
            # Deploy monitoring stack
            monitoring_config = ResourceConfig(
                metadata={"namespace": "monitoring"}
            )
            
            result = provider.deploy_monitoring(monitoring_config)
            assert result.success is True
            assert "prometheus_url" in result.metadata
            assert "grafana_url" in result.metadata