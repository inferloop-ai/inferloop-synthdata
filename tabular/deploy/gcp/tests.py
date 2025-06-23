"""
Tests for GCP Provider
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from ..base import (
    ComputeResource,
    DatabaseResource,
    NetworkResource,
    ResourceStatus,
    SecurityResource,
    StorageResource,
)
from .provider import GCPProvider
from .templates import GCPTemplates


class TestGCPProvider:
    """Test GCP provider functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.provider = GCPProvider("test-project", "us-central1")

    def test_provider_initialization(self):
        """Test provider initialization"""
        assert self.provider.provider_name == "gcp"
        assert self.provider.project_id == "test-project"
        assert self.provider.region == "us-central1"
        assert self.provider.zone == "us-central1-a"

    @patch("subprocess.run")
    def test_validate_credentials_success(self, mock_run):
        """Test successful credential validation"""
        mock_run.return_value.returncode = 0
        assert self.provider.validate_credentials() is True
        assert mock_run.call_count == 2  # gcloud version and auth list

    @patch("subprocess.run")
    def test_validate_credentials_failure(self, mock_run):
        """Test failed credential validation"""
        mock_run.return_value.returncode = 1
        assert self.provider.validate_credentials() is False

    @patch("subprocess.run")
    def test_get_availability_zones_success(self, mock_run):
        """Test getting availability zones"""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "us-central1-a\nus-central1-b\nus-central1-c"

        zones = self.provider.get_availability_zones()
        assert len(zones) == 3
        assert "us-central1-a" in zones

    @patch("subprocess.run")
    def test_get_availability_zones_failure(self, mock_run):
        """Test fallback when getting zones fails"""
        mock_run.return_value.returncode = 1

        zones = self.provider.get_availability_zones()
        assert zones == ["us-central1-a"]  # Fallback to default zone

    def test_estimate_costs_cloud_run(self):
        """Test cost estimation for Cloud Run"""
        resources = [
            {
                "type": "compute",
                "service_type": "cloud_run",
                "cpu": 1,
                "memory": 512,
                "requests_per_month": 1000000,
            }
        ]

        costs = self.provider.estimate_costs(resources)

        assert "compute" in costs
        assert "total" in costs
        assert costs["compute"] > 0
        assert costs["total"] == costs["compute"]

    def test_estimate_costs_gke(self):
        """Test cost estimation for GKE"""
        resources = [
            {
                "type": "compute",
                "service_type": "gke",
                "nodes": 3,
                "machine_type": "e2-standard-4",
            }
        ]

        costs = self.provider.estimate_costs(resources)

        assert "compute" in costs
        assert costs["compute"] > 0

    def test_estimate_costs_storage(self):
        """Test cost estimation for storage"""
        resources = [{"type": "storage", "size_gb": 100, "storage_class": "standard"}]

        costs = self.provider.estimate_costs(resources)

        assert "storage" in costs
        assert costs["storage"] == 100 * 0.020  # 100GB * $0.020

    def test_estimate_costs_database(self):
        """Test cost estimation for database"""
        resources = [
            {
                "type": "database",
                "engine": "postgres",
                "tier": "db-f1-micro",
                "storage_gb": 10,
            }
        ]

        costs = self.provider.estimate_costs(resources)

        assert "database" in costs
        assert costs["database"] > 0

    @patch("subprocess.run")
    def test_get_resource_limits_success(self, mock_run):
        """Test getting resource limits"""
        mock_project_info = {
            "quotas": [
                {"metric": "CPUS", "limit": 1000, "usage": 10},
                {"metric": "MEMORY_GB", "limit": 10000, "usage": 100},
            ]
        }

        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = json.dumps(mock_project_info)

        limits = self.provider.get_resource_limits()

        assert "vcpus" in limits
        assert "memory_gb" in limits
        assert limits["vcpus"] == 1000

    @patch("subprocess.run")
    def test_get_resource_limits_failure(self, mock_run):
        """Test fallback when getting resource limits fails"""
        mock_run.return_value.returncode = 1

        limits = self.provider.get_resource_limits()

        # Should return default limits
        assert limits["vcpus"] == 1000
        assert limits["memory_gb"] == 10000

    @patch("subprocess.run")
    def test_create_cloud_run_service(self, mock_run):
        """Test Cloud Run service creation"""
        mock_run.return_value.returncode = 0

        resource = ComputeResource(
            name="test-service",
            service_type="cloud_run",
            config={
                "image": "gcr.io/test/image",
                "memory": 512,
                "cpu": 1,
                "environment": {"ENV": "test"},
            },
        )

        result = self.provider.create_compute_resource(resource)

        assert "projects/test-project/locations/us-central1/services" in result
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_create_gke_cluster(self, mock_run):
        """Test GKE cluster creation"""
        mock_run.return_value.returncode = 0

        resource = ComputeResource(
            name="test-cluster",
            service_type="gke",
            config={"nodes": 3, "machine_type": "e2-standard-4"},
        )

        result = self.provider.create_compute_resource(resource)

        assert "projects/test-project/locations/us-central1-a/clusters" in result
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_create_cloud_function(self, mock_run):
        """Test Cloud Function creation"""
        mock_run.return_value.returncode = 0

        resource = ComputeResource(
            name="test-function",
            service_type="cloud_function",
            config={"runtime": "python39", "entry_point": "main"},
        )

        result = self.provider.create_compute_resource(resource)

        assert "projects/test-project/locations/us-central1/functions" in result
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_create_storage_resource(self, mock_run):
        """Test storage bucket creation"""
        mock_run.return_value.returncode = 0

        resource = StorageResource(
            name="test-bucket", config={"storage_class": "STANDARD"}
        )

        result = self.provider.create_storage_resource(resource)

        assert result.startswith("gs://")
        assert "test-bucket" in result
        mock_run.assert_called()

    @patch("subprocess.run")
    def test_create_vpc_network(self, mock_run):
        """Test VPC network creation"""
        mock_run.return_value.returncode = 0

        resource = NetworkResource(
            name="test-network",
            service_type="vpc",
            config={"subnets": [{"cidr": "10.0.0.0/24"}]},
        )

        result = self.provider.create_network_resource(resource)

        assert "projects/test-project/global/networks" in result
        assert mock_run.call_count >= 2  # VPC creation + subnet creation

    @patch("subprocess.run")
    def test_create_load_balancer(self, mock_run):
        """Test load balancer creation"""
        mock_run.return_value.returncode = 0

        resource = NetworkResource(
            name="test-lb",
            service_type="load_balancer",
            config={"health_check_port": 8080, "health_check_path": "/health"},
        )

        result = self.provider.create_network_resource(resource)

        assert "forwardingRules" in result
        assert (
            mock_run.call_count >= 4
        )  # Health check, backend, URL map, proxy, forwarding rule

    @patch("subprocess.run")
    def test_create_cloud_sql_postgres(self, mock_run):
        """Test Cloud SQL PostgreSQL creation"""
        mock_run.return_value.returncode = 0

        resource = DatabaseResource(
            name="test-db",
            engine="postgres",
            config={"version": "13", "tier": "db-f1-micro", "storage_gb": 20},
        )

        result = self.provider.create_database_resource(resource)

        assert "projects/test-project/instances" in result
        assert mock_run.call_count >= 2  # Instance creation + database creation

    @patch("subprocess.run")
    def test_create_firestore(self, mock_run):
        """Test Firestore creation"""
        resource = DatabaseResource(
            name="test-firestore", engine="firestore", config={}
        )

        result = self.provider.create_database_resource(resource)

        assert "projects/test-project/databases/(default)" in result

    @patch("subprocess.run")
    def test_create_iam_security(self, mock_run):
        """Test IAM configuration"""
        mock_run.return_value.returncode = 0

        resource = SecurityResource(
            name="test-security",
            service_type="iam",
            config={"service_account": "test-sa", "roles": ["roles/storage.admin"]},
        )

        result = self.provider.create_security_resource(resource)

        assert "@test-project.iam.gserviceaccount.com" in result
        assert mock_run.call_count >= 2  # SA creation + role binding

    @patch("subprocess.run")
    def test_create_secret(self, mock_run):
        """Test secret creation"""
        mock_run.return_value.returncode = 0

        resource = SecurityResource(
            name="test-secret", service_type="secret", config={"value": "secret-value"}
        )

        with patch("tempfile.NamedTemporaryFile"), patch("os.unlink"):
            result = self.provider.create_security_resource(resource)

        assert "projects/test-project/secrets/test-secret" in result
        assert mock_run.call_count >= 2  # Secret creation + version addition

    @patch("subprocess.run")
    def test_delete_cloud_run_service(self, mock_run):
        """Test Cloud Run service deletion"""
        mock_run.return_value.returncode = 0

        success = self.provider.delete_resource(
            "projects/test-project/locations/us-central1/services/test-service",
            "compute",
        )

        assert success is True
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_delete_storage_bucket(self, mock_run):
        """Test storage bucket deletion"""
        mock_run.return_value.returncode = 0

        success = self.provider.delete_resource("gs://test-bucket", "storage")

        assert success is True
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_get_cloud_run_status_running(self, mock_run):
        """Test getting Cloud Run service status"""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "True"

        status = self.provider.get_resource_status(
            "projects/test-project/locations/us-central1/services/test-service",
            "compute",
        )

        assert status == ResourceStatus.RUNNING

    @patch("subprocess.run")
    def test_get_cloud_run_status_pending(self, mock_run):
        """Test getting Cloud Run service status when pending"""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "False"

        status = self.provider.get_resource_status(
            "projects/test-project/locations/us-central1/services/test-service",
            "compute",
        )

        assert status == ResourceStatus.PENDING

    @patch("subprocess.run")
    def test_update_cloud_run_service(self, mock_run):
        """Test Cloud Run service update"""
        mock_run.return_value.returncode = 0

        success = self.provider.update_resource(
            "projects/test-project/locations/us-central1/services/test-service",
            "compute",
            {"memory": 1024, "cpu": 2, "environment": {"NEW_VAR": "value"}},
        )

        assert success is True
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_tag_cloud_run_service(self, mock_run):
        """Test Cloud Run service tagging"""
        mock_run.return_value.returncode = 0

        success = self.provider.tag_resource(
            "projects/test-project/locations/us-central1/services/test-service",
            {"environment": "production", "team": "ml"},
        )

        assert success is True
        mock_run.assert_called_once()


class TestGCPTemplates:
    """Test GCP template generation"""

    def test_cloud_run_template(self):
        """Test Cloud Run template generation"""
        template = GCPTemplates.cloud_run_template(
            "test-service", "gcr.io/test/image", environment={"ENV": "test"}
        )

        assert template["kind"] == "Service"
        assert template["metadata"]["name"] == "test-service"
        assert (
            template["spec"]["template"]["spec"]["containers"][0]["image"]
            == "gcr.io/test/image"
        )
        assert len(template["spec"]["template"]["spec"]["containers"][0]["env"]) == 1

    def test_gke_cluster_template(self):
        """Test GKE cluster template generation"""
        template = GCPTemplates.gke_cluster_template(
            "test-cluster", "us-central1-a", node_count=5
        )

        assert template["name"] == "test-cluster"
        assert template["zone"] == "us-central1-a"
        assert template["initial_node_count"] == 5
        assert template["autoscaling"]["enabled"] is True

    def test_kubernetes_deployment_template(self):
        """Test Kubernetes deployment template generation"""
        template = GCPTemplates.kubernetes_deployment_template(
            "test-app", "gcr.io/test/image", replicas=5
        )

        assert template["kind"] == "Deployment"
        assert template["spec"]["replicas"] == 5
        assert (
            template["spec"]["template"]["spec"]["containers"][0]["name"] == "test-app"
        )
        assert "livenessProbe" in template["spec"]["template"]["spec"]["containers"][0]

    def test_kubernetes_service_template(self):
        """Test Kubernetes service template generation"""
        template = GCPTemplates.kubernetes_service_template("test-app")

        assert template["kind"] == "Service"
        assert template["spec"]["type"] == "LoadBalancer"
        assert template["spec"]["ports"][0]["port"] == 80
        assert template["spec"]["ports"][0]["targetPort"] == 8000

    def test_cloud_function_template(self):
        """Test Cloud Function template generation"""
        template = GCPTemplates.cloud_function_template(
            "test-function", memory=512, timeout=120
        )

        assert template["name"] == "test-function"
        assert template["available_memory_mb"] == 512
        assert template["timeout"] == "120s"
        assert "trigger" in template

    def test_cloud_sql_template(self):
        """Test Cloud SQL template generation"""
        template = GCPTemplates.cloud_sql_template(
            "test-db", tier="db-n1-standard-1", storage_gb=50
        )

        assert template["name"] == "test-db"
        assert template["settings"]["tier"] == "db-n1-standard-1"
        assert template["settings"]["disk_size"] == 50
        assert template["settings"]["backup_configuration"]["enabled"] is True

    def test_terraform_main_template(self):
        """Test Terraform main template generation"""
        template = GCPTemplates.terraform_main_template("test-project", "us-central1")

        assert "terraform {" in template
        assert 'provider "google"' in template
        assert "test-project" in template
        assert "us-central1" in template
        assert "google_compute_network" in template
        assert "google_storage_bucket" in template

    def test_terraform_variables_template(self):
        """Test Terraform variables template generation"""
        template = GCPTemplates.terraform_variables_template()

        assert 'variable "project_id"' in template
        assert 'variable "region"' in template
        assert 'variable "api_image"' in template
        assert 'variable "gke_node_count"' in template

    def test_deployment_script_template(self):
        """Test deployment script template generation"""
        script = GCPTemplates.deployment_script_template()

        assert "#!/bin/bash" in script
        assert "gcloud services enable" in script
        assert "gsutil mb" in script
        assert "gcloud run deploy" in script
        assert "gcloud container clusters create" in script

    def test_docker_template(self):
        """Test Docker template generation"""
        dockerfile = GCPTemplates.docker_template()

        assert "FROM python:3.9-slim" in dockerfile
        assert "WORKDIR /app" in dockerfile
        assert "EXPOSE 8000" in dockerfile
        assert "HEALTHCHECK" in dockerfile
        assert "uvicorn" in dockerfile

    def test_cloudbuild_template(self):
        """Test Cloud Build template generation"""
        cloudbuild = GCPTemplates.cloudbuild_template()

        assert "steps:" in cloudbuild
        assert "gcr.io/cloud-builders/docker" in cloudbuild
        assert "gcloud run deploy" in cloudbuild
        assert "logsBucket:" in cloudbuild

    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("pathlib.Path.chmod")
    def test_save_templates(self, mock_chmod, mock_open, mock_mkdir):
        """Test saving all templates to files"""
        output_dir = Path("/test/output")

        GCPTemplates.save_templates(output_dir)

        # Verify directories were created
        assert mock_mkdir.call_count >= 4  # Main dir + subdirs

        # Verify files were written
        assert mock_open.call_count >= 10  # Multiple template files

        # Verify script was made executable
        mock_chmod.assert_called_with(0o755)


@pytest.mark.integration
class TestGCPIntegration:
    """Integration tests for GCP provider (requires actual GCP access)"""

    @pytest.fixture
    def provider(self):
        """Create provider for integration tests"""
        # Skip if no GCP credentials
        pytest.importorskip("google.cloud")
        return GCPProvider("test-project", "us-central1")

    def test_validate_credentials_integration(self, provider):
        """Integration test for credential validation"""
        # This will only pass if gcloud is configured
        # In CI/CD, this test should be marked as optional
        try:
            result = provider.validate_credentials()
            assert isinstance(result, bool)
        except Exception:
            pytest.skip("GCP credentials not configured")

    def test_cost_estimation_integration(self, provider):
        """Integration test for cost estimation"""
        resources = [
            {
                "type": "compute",
                "service_type": "cloud_run",
                "cpu": 1,
                "memory": 512,
                "requests_per_month": 100000,
            }
        ]

        costs = provider.estimate_costs(resources)

        assert "total" in costs
        assert costs["total"] >= 0
        assert isinstance(costs["compute"], (int, float))
