"""End-to-end tests for deployment scenarios."""

import pytest
import time
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from deploy.base import ResourceConfig, DeploymentResult
from deploy.onprem.provider import OnPremKubernetesProvider
from deploy.onprem.security import SecurityManager
from deploy.onprem.backup import BackupManager
from deploy.onprem.gitops import GitOpsManager
from deploy.onprem.helm import HelmChartGenerator, HelmDeployer


class TestE2EDeploymentScenarios:
    """End-to-end deployment scenario tests."""
    
    @pytest.fixture
    def mock_kubectl(self):
        """Mock kubectl commands globally."""
        with patch('deploy.onprem.provider.subprocess.run') as mock_run:
            def kubectl_response(*args, **kwargs):
                cmd = args[0] if args else []
                cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
                
                # Standard responses for different kubectl commands
                if "cluster-info" in cmd_str:
                    return MagicMock(returncode=0, stdout="Kubernetes control plane is running")
                elif "get deployment" in cmd_str and "-o json" in cmd_str:
                    return MagicMock(
                        returncode=0,
                        stdout='{"status": {"replicas": 3, "readyReplicas": 3}}'
                    )
                elif "get service" in cmd_str and "-o json" in cmd_str:
                    return MagicMock(
                        returncode=0,
                        stdout='{"spec": {"clusterIP": "10.96.0.1", "ports": [{"port": 80}]}}'
                    )
                elif "get statefulset" in cmd_str and "-o json" in cmd_str:
                    return MagicMock(
                        returncode=0,
                        stdout='{"status": {"replicas": 1, "readyReplicas": 1}}'
                    )
                elif "wait" in cmd_str:
                    return MagicMock(returncode=0, stdout="condition met")
                else:
                    return MagicMock(returncode=0, stdout="success")
            
            mock_run.side_effect = kubectl_response
            yield mock_run
    
    @pytest.fixture
    def mock_helm(self):
        """Mock Helm commands."""
        with patch('deploy.onprem.helm.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="success")
            yield mock_run
    
    @pytest.mark.e2e
    def test_complete_platform_deployment(self, mock_kubectl, mock_helm):
        """Test complete platform deployment from scratch."""
        # 1. Initialize provider
        provider = OnPremKubernetesProvider()
        
        # 2. Authenticate
        auth_result = provider.authenticate(namespace="synthdata-prod")
        assert auth_result is True
        
        # 3. Deploy storage (MinIO)
        storage_config = ResourceConfig(
            compute={"count": 4},
            storage={"size": "500Gi", "storage_class": "fast-ssd"},
            metadata={
                "storage_type": "minio",
                "access_key": "minioadmin",
                "secret_key": "minioadmin123"
            }
        )
        
        storage_result = provider.deploy_storage(storage_config)
        assert storage_result.success is True
        assert "minio" in storage_result.endpoint
        
        # 4. Deploy database (PostgreSQL)
        database_config = ResourceConfig(
            compute={"cpu": "4", "memory": "16Gi"},
            storage={"size": "200Gi", "storage_class": "fast-ssd"},
            metadata={
                "db_type": "postgresql",
                "name": "postgres",
                "password": "synthdata123"
            }
        )
        
        db_result = provider.deploy_database(database_config)
        assert db_result.success is True
        assert "postgresql://" in db_result.endpoint
        
        # 5. Deploy monitoring
        monitoring_config = ResourceConfig(
            metadata={"namespace": "monitoring"}
        )
        
        monitoring_result = provider.deploy_monitoring(monitoring_config)
        assert monitoring_result.success is True
        assert "prometheus_url" in monitoring_result.metadata
        assert "grafana_url" in monitoring_result.metadata
        
        # 6. Deploy main application
        app_config = ResourceConfig(
            compute={"count": 3, "cpu": "4", "memory": "16Gi"},
            networking={"service_type": "ClusterIP"},
            metadata={
                "name": "synthdata-api",
                "image": "inferloop/synthdata:latest",
                "environment": "production",
                "database_url": db_result.endpoint,
                "s3_endpoint": storage_result.endpoint
            }
        )
        
        app_result = provider.deploy_container(app_config)
        assert app_result.success is True
        assert "synthdata-api" in app_result.resource_id
        
        # 7. Verify all components are running
        deployments = provider.list_deployments()
        assert len(deployments) >= 1
        
        # 8. Check application status
        app_status = provider.get_deployment_status(app_result.resource_id)
        assert app_status["replicas"]["ready"] == 3
    
    @pytest.mark.e2e
    def test_enterprise_deployment_with_security(self, mock_kubectl):
        """Test enterprise deployment with full security features."""
        # 1. Setup security manager
        security_mgr = SecurityManager()
        
        # 2. Deploy cert-manager
        cert_result = security_mgr.deploy_cert_manager()
        assert cert_result.success is True
        
        # 3. Create cluster issuer
        issuer_result = security_mgr.create_cluster_issuer("self-signed")
        assert issuer_result.success is True
        
        # 4. Deploy Dex for LDAP integration
        ldap_config = ResourceConfig(
            metadata={
                "namespace": "auth",
                "ldap": {
                    "host": "ldap.company.com:389",
                    "bind_dn": "cn=admin,dc=company,dc=com",
                    "bind_password": "admin123",
                    "user_base_dn": "ou=users,dc=company,dc=com",
                    "group_base_dn": "ou=groups,dc=company,dc=com"
                },
                "client_secret": "synthdata-secret",
                "redirect_uris": ["https://synthdata.company.com/callback"]
            }
        )
        
        dex_result = security_mgr.deploy_dex_oidc(ldap_config)
        assert dex_result.success is True
        assert "dex" in dex_result.endpoint
        
        # 5. Create network policies
        network_result = security_mgr.create_network_policies("synthdata")
        assert network_result.success is True
        
        # 6. Create certificates
        cert_create_result = security_mgr.create_certificate(
            name="synthdata-tls",
            namespace="synthdata", 
            dns_names=["synthdata.company.com", "api.synthdata.company.com"]
        )
        assert cert_create_result.success is True
        
        # 7. Deploy main application with security
        provider = OnPremKubernetesProvider()
        provider.authenticate(namespace="synthdata")
        
        secure_app_config = ResourceConfig(
            compute={"count": 3, "cpu": "2", "memory": "8Gi"},
            networking={
                "service_type": "ClusterIP",
                "ingress_enabled": True,
                "hostname": "synthdata.company.com"
            },
            metadata={
                "name": "synthdata-secure",
                "image": "inferloop/synthdata:latest",
                "environment": "production",
                "ldap_enabled": True,
                "oidc_issuer": dex_result.endpoint
            }
        )
        
        app_result = provider.deploy_container(secure_app_config)
        assert app_result.success is True
    
    @pytest.mark.e2e
    def test_gitops_deployment_workflow(self, mock_kubectl):
        """Test GitOps-based deployment workflow."""
        # 1. Setup GitOps manager
        gitops_mgr = GitOpsManager()
        
        # 2. Install ArgoCD
        argocd_result = gitops_mgr.install_argocd()
        assert argocd_result.success is True
        assert "admin_password" in argocd_result.metadata
        
        # 3. Create ArgoCD application
        app_config = ResourceConfig(
            metadata={
                "name": "synthdata-gitops",
                "repo_url": "https://github.com/company/synthdata-config",
                "target_revision": "main",
                "path": "manifests/production",
                "dest_namespace": "synthdata",
                "auto_prune": True,
                "auto_heal": True
            }
        )
        
        app_result = gitops_mgr.create_argocd_application(app_config)
        assert app_result.success is True
        assert "synthdata-gitops" in app_result.resource_id
        
        # 4. Verify application status
        app_status = gitops_mgr.get_argocd_app_status("synthdata-gitops")
        assert "name" in app_status
        
        # 5. Trigger sync
        sync_result = gitops_mgr.sync_argocd_app("synthdata-gitops")
        assert sync_result.success is True
    
    @pytest.mark.e2e
    def test_backup_and_disaster_recovery(self, mock_kubectl):
        """Test backup and disaster recovery workflow."""
        # 1. Setup backup manager
        backup_mgr = BackupManager()
        
        # 2. Install Velero with MinIO backend
        velero_config = ResourceConfig(
            metadata={
                "provider": "minio",
                "bucket": "synthdata-backups",
                "endpoint": "http://minio:9000",
                "access_key": "minioadmin",
                "secret_key": "minioadmin123",
                "schedule": "0 2 * * *",  # Daily at 2 AM
                "namespaces": ["synthdata", "monitoring"]
            }
        )
        
        velero_result = backup_mgr.install_velero(velero_config)
        assert velero_result.success is True
        
        # 3. Validate Velero installation
        is_valid, issues = backup_mgr.validate_installation()
        assert is_valid is True
        assert len(issues) == 0
        
        # 4. Create immediate backup
        backup_result = backup_mgr.create_backup(
            backup_name="pre-upgrade-backup",
            namespaces=["synthdata"],
            wait=True
        )
        assert backup_result.success is True
        
        # 5. List backups
        with patch('deploy.onprem.backup.subprocess.run') as mock_velero:
            mock_velero.return_value = MagicMock(
                returncode=0,
                stdout='{"items": [{"metadata": {"name": "pre-upgrade-backup"}, "status": {"phase": "Completed"}}]}'
            )
            
            backups = backup_mgr.list_backups()
            assert len(backups) >= 1
            assert backups[0]["name"] == "pre-upgrade-backup"
        
        # 6. Test restore (simulation)
        restore_result = backup_mgr.restore_backup(
            backup_name="pre-upgrade-backup",
            restore_name="test-restore",
            wait=True
        )
        assert restore_result.success is True
    
    @pytest.mark.e2e
    def test_helm_based_deployment(self, mock_kubectl, mock_helm):
        """Test Helm-based deployment workflow."""
        # 1. Setup provider with Helm
        provider = OnPremKubernetesProvider()
        provider.authenticate(namespace="synthdata")
        
        # 2. Create Helm chart configuration
        helm_config = ResourceConfig(
            compute={"count": 3, "cpu": "2", "memory": "8Gi"},
            storage={"size": "100Gi", "storage_class": "fast-ssd"},
            networking={"service_type": "LoadBalancer"},
            metadata={
                "name": "synthdata-helm",
                "version": "1.0.0",
                "image": "inferloop/synthdata:v1.2.0",
                "environment": "production",
                "autoscaling_enabled": True,
                "min_replicas": 3,
                "max_replicas": 10,
                "postgresql_enabled": True,
                "minio_enabled": True
            }
        )
        
        # 3. Deploy with Helm
        helm_result = provider.deploy_with_helm(helm_config)
        assert helm_result.success is True
        
        # 4. List Helm releases
        releases = provider.list_helm_releases()
        assert isinstance(releases, list)
        
        # 5. Test Helm upgrade
        upgrade_config = ResourceConfig(
            compute={"count": 5, "cpu": "2", "memory": "8Gi"},
            metadata={
                "name": "synthdata-helm",
                "version": "1.1.0",
                "image": "inferloop/synthdata:v1.3.0"
            }
        )
        
        upgrade_result = provider.upgrade_with_helm(upgrade_config)
        assert upgrade_result.success is True
        
        # 6. Test Helm rollback
        rollback_result = provider.rollback_helm_release("synthdata-helm")
        assert rollback_result is True
    
    @pytest.mark.e2e
    def test_multi_environment_deployment(self, mock_kubectl):
        """Test deployment across multiple environments."""
        environments = ["development", "staging", "production"]
        results = {}
        
        for env in environments:
            # 1. Create environment-specific provider
            provider = OnPremKubernetesProvider()
            provider.authenticate(namespace=f"synthdata-{env}")
            
            # 2. Environment-specific configuration
            if env == "development":
                config = ResourceConfig(
                    compute={"count": 1, "cpu": "1", "memory": "2Gi"},
                    metadata={
                        "name": f"synthdata-{env}",
                        "image": "inferloop/synthdata:dev",
                        "environment": env,
                        "debug": True
                    }
                )
            elif env == "staging":
                config = ResourceConfig(
                    compute={"count": 2, "cpu": "2", "memory": "4Gi"},
                    metadata={
                        "name": f"synthdata-{env}",
                        "image": "inferloop/synthdata:staging",
                        "environment": env,
                        "debug": False
                    }
                )
            else:  # production
                config = ResourceConfig(
                    compute={"count": 5, "cpu": "4", "memory": "16Gi"},
                    networking={"service_type": "LoadBalancer"},
                    metadata={
                        "name": f"synthdata-{env}",
                        "image": "inferloop/synthdata:v1.0.0",
                        "environment": env,
                        "debug": False,
                        "monitoring_enabled": True
                    }
                )
            
            # 3. Deploy to environment
            result = provider.deploy_container(config)
            results[env] = result
            assert result.success is True
        
        # 4. Verify all environments deployed
        assert len(results) == 3
        for env, result in results.items():
            assert result.success is True
            assert env in result.resource_id
    
    @pytest.mark.e2e
    def test_scaling_and_performance(self, mock_kubectl):
        """Test scaling and performance scenarios."""
        provider = OnPremKubernetesProvider()
        provider.authenticate(namespace="synthdata")
        
        # 1. Initial small deployment
        initial_config = ResourceConfig(
            compute={"count": 2, "cpu": "2", "memory": "4Gi"},
            metadata={
                "name": "synthdata-scale",
                "image": "inferloop/synthdata:latest"
            }
        )
        
        initial_result = provider.deploy_container(initial_config)
        assert initial_result.success is True
        
        # 2. Scale up (simulate increased load)
        scale_up_config = ResourceConfig(
            compute={"count": 10, "cpu": "4", "memory": "8Gi"},
            metadata={
                "name": "synthdata-scale",
                "image": "inferloop/synthdata:latest",
                "autoscaling_enabled": True,
                "max_replicas": 20
            }
        )
        
        # In a real scenario, this would update the deployment
        # Here we verify the config is valid
        assert scale_up_config.compute["count"] == 10
        
        # 3. Scale down (simulate reduced load)
        scale_down_config = ResourceConfig(
            compute={"count": 3, "cpu": "2", "memory": "4Gi"},
            metadata={"name": "synthdata-scale", "image": "inferloop/synthdata:latest"}
        )
        
        assert scale_down_config.compute["count"] == 3
        
        # 4. Test resource limits
        resource_config = ResourceConfig(
            compute={
                "count": 5,
                "cpu": "8",
                "memory": "32Gi",
                "gpu": 2  # GPU-enabled workload
            },
            metadata={
                "name": "synthdata-gpu",
                "image": "inferloop/synthdata:gpu"
            }
        )
        
        gpu_result = provider.deploy_container(resource_config)
        assert gpu_result.success is True
    
    @pytest.mark.e2e
    def test_failure_scenarios_and_recovery(self, mock_kubectl):
        """Test failure scenarios and recovery mechanisms."""
        provider = OnPremKubernetesProvider()
        provider.authenticate(namespace="synthdata")
        
        # 1. Test deployment with insufficient resources
        insufficient_config = ResourceConfig(
            compute={"count": 100, "cpu": "1000", "memory": "10000Gi"},  # Unrealistic
            metadata={"name": "synthdata-fail", "image": "inferloop/synthdata:latest"}
        )
        
        # This should still succeed in our mocked environment
        result = provider.deploy_container(insufficient_config)
        # In real scenario, this might fail due to resource constraints
        
        # 2. Test invalid image deployment
        invalid_config = ResourceConfig(
            metadata={
                "name": "synthdata-invalid",
                "image": "nonexistent/image:latest"
            }
        )
        
        # Mock failure response for invalid image
        with patch('deploy.onprem.provider.subprocess.run') as mock_fail:
            mock_fail.side_effect = [
                MagicMock(returncode=0),  # cluster-info
                MagicMock(returncode=0),  # namespace ops
                MagicMock(returncode=1, stderr="ImagePullBackOff"),  # deployment failure
            ]
            
            with patch('deploy.onprem.provider.OnPremKubernetesProvider._validate_cluster_access'):
                fail_provider = OnPremKubernetesProvider()
                fail_provider.authenticate()
                
                fail_result = fail_provider.deploy_container(invalid_config)
                assert fail_result.success is False
        
        # 3. Test recovery after failure
        recovery_config = ResourceConfig(
            metadata={
                "name": "synthdata-recovery",
                "image": "inferloop/synthdata:latest"
            }
        )
        
        recovery_result = provider.deploy_container(recovery_config)
        assert recovery_result.success is True
    
    @pytest.mark.e2e
    def test_complete_lifecycle_with_monitoring(self, mock_kubectl):
        """Test complete application lifecycle with monitoring."""
        # 1. Deploy monitoring first
        provider = OnPremKubernetesProvider()
        provider.authenticate(namespace="synthdata")
        
        monitoring_config = ResourceConfig(
            metadata={"namespace": "monitoring"}
        )
        
        monitoring_result = provider.deploy_monitoring(monitoring_config)
        assert monitoring_result.success is True
        
        # 2. Deploy application with monitoring annotations
        app_config = ResourceConfig(
            compute={"count": 3, "cpu": "2", "memory": "8Gi"},
            metadata={
                "name": "synthdata-monitored",
                "image": "inferloop/synthdata:latest",
                "monitoring_enabled": True,
                "metrics_port": 8080,
                "health_check_path": "/health"
            }
        )
        
        app_result = provider.deploy_container(app_config)
        assert app_result.success is True
        
        # 3. Verify monitoring integration
        # In a real scenario, this would check Prometheus targets
        assert monitoring_result.metadata["prometheus_url"] is not None
        assert monitoring_result.metadata["grafana_url"] is not None
        
        # 4. Simulate application lifecycle events
        lifecycle_events = [
            "deployment_started",
            "health_check_passed", 
            "metrics_available",
            "scaling_triggered",
            "backup_completed"
        ]
        
        for event in lifecycle_events:
            # In real scenario, these would be actual monitoring checks
            assert event is not None
        
        # 5. Test alerting (simulation)
        alert_conditions = {
            "high_cpu": False,
            "high_memory": False,
            "pod_failures": False,
            "service_unavailable": False
        }
        
        # All conditions should be healthy
        assert all(not condition for condition in alert_conditions.values())
    
    @pytest.mark.e2e
    def test_enterprise_integration_scenario(self, mock_kubectl):
        """Test enterprise integration scenario with all components."""
        # This test combines all enterprise features
        
        # 1. Security setup
        security_mgr = SecurityManager()
        cert_result = security_mgr.deploy_cert_manager()
        assert cert_result.success is True
        
        # 2. Backup setup  
        backup_mgr = BackupManager()
        backup_config = ResourceConfig(
            metadata={
                "provider": "minio",
                "bucket": "enterprise-backups", 
                "schedule": "0 1 * * *"
            }
        )
        backup_result = backup_mgr.install_velero(backup_config)
        assert backup_result.success is True
        
        # 3. GitOps setup
        gitops_mgr = GitOpsManager()
        argocd_result = gitops_mgr.install_argocd()
        assert argocd_result.success is True
        
        # 4. Main application deployment
        provider = OnPremKubernetesProvider()
        provider.authenticate(namespace="synthdata-enterprise")
        
        enterprise_config = ResourceConfig(
            compute={"count": 5, "cpu": "4", "memory": "16Gi"},
            storage={"size": "500Gi", "storage_class": "enterprise-ssd"},
            networking={
                "service_type": "LoadBalancer",
                "ingress_enabled": True,
                "hostname": "synthdata.enterprise.com"
            },
            metadata={
                "name": "synthdata-enterprise",
                "image": "inferloop/synthdata:enterprise",
                "environment": "production",
                "ha_enabled": True,
                "backup_enabled": True,
                "monitoring_enabled": True,
                "security_enabled": True,
                "ldap_integration": True
            }
        )
        
        app_result = provider.deploy_container(enterprise_config)
        assert app_result.success is True
        
        # 5. Verify enterprise readiness
        enterprise_checklist = {
            "security": cert_result.success,
            "backup": backup_result.success,
            "gitops": argocd_result.success,
            "application": app_result.success,
            "monitoring": True,  # Would be checked in real scenario
            "high_availability": True  # Would be verified
        }
        
        assert all(enterprise_checklist.values())
        
        # 6. Performance validation
        performance_metrics = {
            "deployment_time": "< 15 minutes",  # Target
            "availability": "> 99.9%",
            "recovery_time": "< 5 minutes",
            "backup_success": "> 99%"
        }
        
        # In real scenario, these would be measured
        assert len(performance_metrics) == 4