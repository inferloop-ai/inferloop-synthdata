"""
Tests for Azure Provider
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
from .provider import AzureProvider
from .templates import AzureTemplates


class TestAzureProvider:
    """Test Azure provider functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.provider = AzureProvider("test-subscription", "test-rg", "eastus")

    def test_provider_initialization(self):
        """Test provider initialization"""
        assert self.provider.provider_name == "azure"
        assert self.provider.subscription_id == "test-subscription"
        assert self.provider.resource_group == "test-rg"
        assert self.provider.location == "eastus"

    @patch("subprocess.run")
    def test_validate_credentials_success(self, mock_run):
        """Test successful credential validation"""
        mock_run.return_value.returncode = 0
        assert self.provider.validate_credentials() is True
        assert mock_run.call_count == 3  # az version, account show, account set

    @patch("subprocess.run")
    def test_validate_credentials_failure(self, mock_run):
        """Test failed credential validation"""
        mock_run.return_value.returncode = 1
        assert self.provider.validate_credentials() is False

    @patch("subprocess.run")
    def test_get_availability_zones_success(self, mock_run):
        """Test getting availability zones"""
        mock_zones_data = [["1", "2", "3"]]
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = json.dumps(mock_zones_data)

        zones = self.provider.get_availability_zones()
        assert len(zones) == 3
        assert "1" in zones
        assert "2" in zones
        assert "3" in zones

    @patch("subprocess.run")
    def test_get_availability_zones_failure(self, mock_run):
        """Test fallback when getting zones fails"""
        mock_run.return_value.returncode = 1

        zones = self.provider.get_availability_zones()
        assert zones == ["1"]  # Fallback to default zone

    def test_estimate_costs_container_instance(self):
        """Test cost estimation for Container Instance"""
        resources = [
            {
                "type": "compute",
                "service_type": "container_instance",
                "cpu": 1,
                "memory": 1,
                "hours_per_month": 730,
            }
        ]

        costs = self.provider.estimate_costs(resources)

        assert "compute" in costs
        assert "total" in costs
        assert costs["compute"] > 0
        assert costs["total"] == costs["compute"]

    def test_estimate_costs_aks(self):
        """Test cost estimation for AKS"""
        resources = [
            {
                "type": "compute",
                "service_type": "aks",
                "nodes": 3,
                "vm_size": "Standard_DS2_v2",
            }
        ]

        costs = self.provider.estimate_costs(resources)

        assert "compute" in costs
        assert costs["compute"] > 0

    def test_estimate_costs_azure_function(self):
        """Test cost estimation for Azure Functions"""
        resources = [
            {
                "type": "compute",
                "service_type": "function",
                "executions_per_month": 2000000,
                "execution_time_ms": 1000,
                "memory_mb": 128,
            }
        ]

        costs = self.provider.estimate_costs(resources)

        assert "compute" in costs
        assert costs["compute"] >= 0  # May be 0 due to free tier

    def test_estimate_costs_storage(self):
        """Test cost estimation for storage"""
        resources = [{"type": "storage", "size_gb": 100, "tier": "hot"}]

        costs = self.provider.estimate_costs(resources)

        assert "storage" in costs
        assert costs["storage"] == 100 * 0.0184  # 100GB * $0.0184

    def test_estimate_costs_sql_database(self):
        """Test cost estimation for SQL Database"""
        resources = [{"type": "database", "engine": "sqlserver", "tier": "S1"}]

        costs = self.provider.estimate_costs(resources)

        assert "database" in costs
        assert costs["database"] == 29.48  # S1 tier price

    def test_estimate_costs_cosmos_db(self):
        """Test cost estimation for Cosmos DB"""
        resources = [
            {
                "type": "database",
                "engine": "cosmos",
                "ru_per_second": 400,
                "storage_gb": 10,
            }
        ]

        costs = self.provider.estimate_costs(resources)

        assert "database" in costs
        assert costs["database"] > 0

    @patch("subprocess.run")
    def test_get_resource_limits_success(self, mock_run):
        """Test getting resource limits"""
        mock_quotas = [
            {"name": "cores", "limit": 1000, "currentValue": 10},
            {"name": "virtualMachines", "limit": 500, "currentValue": 5},
        ]

        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = json.dumps(mock_quotas)

        limits = self.provider.get_resource_limits()

        assert "vcpus" in limits
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
    def test_create_container_instance(self, mock_run):
        """Test Container Instance creation"""
        mock_run.return_value.returncode = 0

        resource = ComputeResource(
            name="test-container",
            service_type="container_instance",
            config={
                "image": "nginx:latest",
                "cpu": 1,
                "memory": 1,
                "environment": {"ENV": "test"},
            },
        )

        result = self.provider.create_compute_resource(resource)

        assert "Microsoft.ContainerInstance/containerGroups" in result
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_create_aks_cluster(self, mock_run):
        """Test AKS cluster creation"""
        mock_run.return_value.returncode = 0

        resource = ComputeResource(
            name="test-cluster",
            service_type="aks",
            config={"nodes": 3, "vm_size": "Standard_DS2_v2"},
        )

        result = self.provider.create_compute_resource(resource)

        assert "Microsoft.ContainerService/managedClusters" in result
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_create_azure_function(self, mock_run):
        """Test Azure Function creation"""
        mock_run.return_value.returncode = 0

        resource = ComputeResource(
            name="test-function",
            service_type="function",
            config={"runtime": "python", "runtime_version": "3.9"},
        )

        result = self.provider.create_compute_resource(resource)

        assert "Microsoft.Web/sites" in result
        assert mock_run.call_count >= 2  # Storage account + function app

    @patch("subprocess.run")
    def test_create_storage_resource(self, mock_run):
        """Test storage account creation"""
        mock_run.return_value.returncode = 0

        resource = StorageResource(
            name="test-storage",
            config={"sku": "Standard_LRS", "tier": "Hot", "container_name": "data"},
        )

        result = self.provider.create_storage_resource(resource)

        assert "Microsoft.Storage/storageAccounts" in result
        assert mock_run.call_count >= 2  # Storage account + container

    @patch("subprocess.run")
    def test_create_virtual_network(self, mock_run):
        """Test Virtual Network creation"""
        mock_run.return_value.returncode = 0

        resource = NetworkResource(
            name="test-network",
            service_type="vnet",
            config={
                "address_space": "10.0.0.0/16",
                "subnets": [{"address_prefix": "10.0.1.0/24"}],
            },
        )

        result = self.provider.create_network_resource(resource)

        assert "Microsoft.Network/virtualNetworks" in result
        assert mock_run.call_count >= 2  # VNet creation + subnet creation

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

        assert "Microsoft.Network/loadBalancers" in result
        assert mock_run.call_count >= 3  # Public IP, LB, health probe

    @patch("subprocess.run")
    def test_create_azure_sql_database(self, mock_run):
        """Test Azure SQL Database creation"""
        mock_run.return_value.returncode = 0

        resource = DatabaseResource(
            name="test-db",
            engine="sqlserver",
            config={
                "admin_user": "testadmin",
                "admin_password": "TestPass123!",
                "edition": "Standard",
                "tier": "S1",
            },
        )

        result = self.provider.create_database_resource(resource)

        assert "Microsoft.Sql/servers" in result
        assert mock_run.call_count >= 2  # Server creation + database creation

    @patch("subprocess.run")
    def test_create_cosmos_db(self, mock_run):
        """Test Cosmos DB creation"""
        mock_run.return_value.returncode = 0

        resource = DatabaseResource(
            name="test-cosmos",
            engine="cosmos",
            config={"kind": "GlobalDocumentDB", "consistency": "Session"},
        )

        result = self.provider.create_database_resource(resource)

        assert "Microsoft.DocumentDB/databaseAccounts" in result
        assert mock_run.call_count >= 2  # Account creation + database creation

    @patch("subprocess.run")
    def test_create_key_vault(self, mock_run):
        """Test Key Vault creation"""
        mock_run.return_value.returncode = 0

        resource = SecurityResource(
            name="test-vault",
            service_type="keyvault",
            config={"sku": "standard", "secrets": {"test-secret": "secret-value"}},
        )

        result = self.provider.create_security_resource(resource)

        assert "Microsoft.KeyVault/vaults" in result
        assert mock_run.call_count >= 2  # Vault creation + secret addition

    @patch("subprocess.run")
    def test_create_managed_identity(self, mock_run):
        """Test Managed Identity creation"""
        mock_run.return_value.returncode = 0

        resource = SecurityResource(
            name="test-identity", service_type="identity", config={}
        )

        result = self.provider.create_security_resource(resource)

        assert "Microsoft.ManagedIdentity/userAssignedIdentities" in result
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_delete_container_instance(self, mock_run):
        """Test Container Instance deletion"""
        mock_run.return_value.returncode = 0

        success = self.provider.delete_resource(
            "/subscriptions/test-sub/resourceGroups/test-rg/providers/Microsoft.ContainerInstance/containerGroups/test-container",
            "compute",
        )

        assert success is True
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_delete_storage_account(self, mock_run):
        """Test storage account deletion"""
        mock_run.return_value.returncode = 0

        success = self.provider.delete_resource(
            "/subscriptions/test-sub/resourceGroups/test-rg/providers/Microsoft.Storage/storageAccounts/teststorage",
            "storage",
        )

        assert success is True
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_get_container_instance_status_running(self, mock_run):
        """Test getting Container Instance status"""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Running"

        status = self.provider.get_resource_status(
            "/subscriptions/test-sub/resourceGroups/test-rg/providers/Microsoft.ContainerInstance/containerGroups/test-container",
            "compute",
        )

        assert status == ResourceStatus.RUNNING

    @patch("subprocess.run")
    def test_get_container_instance_status_pending(self, mock_run):
        """Test getting Container Instance status when pending"""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Pending"

        status = self.provider.get_resource_status(
            "/subscriptions/test-sub/resourceGroups/test-rg/providers/Microsoft.ContainerInstance/containerGroups/test-container",
            "compute",
        )

        assert status == ResourceStatus.PENDING

    @patch("subprocess.run")
    def test_tag_resource(self, mock_run):
        """Test resource tagging"""
        mock_run.return_value.returncode = 0

        success = self.provider.tag_resource(
            "/subscriptions/test-sub/resourceGroups/test-rg/providers/Microsoft.ContainerInstance/containerGroups/test-container",
            {"environment": "production", "team": "ml"},
        )

        assert success is True
        mock_run.assert_called_once()


class TestAzureTemplates:
    """Test Azure template generation"""

    def test_container_instance_template(self):
        """Test Container Instance template generation"""
        template = AzureTemplates.container_instance_template(
            "test-container", "nginx:latest", environment={"ENV": "test"}
        )

        assert (
            template["resources"][0]["type"]
            == "Microsoft.ContainerInstance/containerGroups"
        )
        assert (
            template["parameters"]["containerName"]["defaultValue"] == "test-container"
        )
        assert (
            len(
                template["resources"][0]["properties"]["containers"][0]["properties"][
                    "environmentVariables"
                ]
            )
            == 1
        )

    def test_aks_cluster_template(self):
        """Test AKS cluster template generation"""
        template = AzureTemplates.aks_cluster_template("test-cluster", node_count=5)

        assert (
            template["resources"][0]["type"]
            == "Microsoft.ContainerService/managedClusters"
        )
        assert template["parameters"]["clusterName"]["defaultValue"] == "test-cluster"
        assert (
            template["resources"][0]["properties"]["agentPoolProfiles"][0]["count"] == 5
        )

    def test_kubernetes_deployment_template(self):
        """Test Kubernetes deployment template generation"""
        template = AzureTemplates.kubernetes_deployment_template(
            "test-app", "nginx:latest", replicas=5
        )

        assert template["kind"] == "Deployment"
        assert template["spec"]["replicas"] == 5
        assert (
            template["spec"]["template"]["spec"]["containers"][0]["name"] == "test-app"
        )
        assert "livenessProbe" in template["spec"]["template"]["spec"]["containers"][0]

    def test_kubernetes_service_template(self):
        """Test Kubernetes service template generation"""
        template = AzureTemplates.kubernetes_service_template("test-app")

        assert template["kind"] == "Service"
        assert template["spec"]["type"] == "LoadBalancer"
        assert template["spec"]["ports"][0]["port"] == 80
        assert template["spec"]["ports"][0]["targetPort"] == 8000

    def test_function_app_template(self):
        """Test Function App template generation"""
        template = AzureTemplates.function_app_template("test-function", "teststorage")

        assert (
            len(template["resources"]) == 4
        )  # Storage, hosting plan, insights, function app
        function_app = next(
            r for r in template["resources"] if r["type"] == "Microsoft.Web/sites"
        )
        assert function_app["kind"] == "functionapp"

    def test_storage_account_template(self):
        """Test Storage Account template generation"""
        template = AzureTemplates.storage_account_template(
            "teststorage", sku="Premium_LRS"
        )

        assert template["resources"][0]["type"] == "Microsoft.Storage/storageAccounts"
        assert template["resources"][0]["sku"]["name"] == "Premium_LRS"
        assert (
            template["resources"][0]["properties"]["supportsHttpsTrafficOnly"] is True
        )

    def test_virtual_network_template(self):
        """Test Virtual Network template generation"""
        subnets = [
            {"name": "subnet1", "addressPrefix": "10.0.1.0/24"},
            {"name": "subnet2", "addressPrefix": "10.0.2.0/24"},
        ]

        template = AzureTemplates.virtual_network_template("test-vnet", subnets=subnets)

        assert template["resources"][0]["type"] == "Microsoft.Network/virtualNetworks"
        assert len(template["resources"][0]["properties"]["subnets"]) == 2

    def test_azure_sql_template(self):
        """Test Azure SQL template generation"""
        template = AzureTemplates.azure_sql_template(
            "test-server", "test-db", tier="S1"
        )

        # Should have server, database, and firewall rule
        assert len(template["resources"]) == 3

        server = next(
            r for r in template["resources"] if r["type"] == "Microsoft.Sql/servers"
        )
        assert server["properties"]["version"] == "12.0"

        database = next(
            r
            for r in template["resources"]
            if r["type"] == "Microsoft.Sql/servers/databases"
        )
        assert database["sku"]["name"] == "S1"

    def test_bicep_main_template(self):
        """Test Bicep main template generation"""
        template = AzureTemplates.bicep_main_template()

        assert "@description" in template
        assert "param subscriptionId" in template
        assert "resource storageAccount" in template
        assert "resource containerInstance" in template
        assert "resource aksCluster" in template
        assert "resource sqlServer" in template
        assert "resource keyVault" in template

    def test_deployment_script_template(self):
        """Test deployment script template generation"""
        script = AzureTemplates.deployment_script_template()

        assert "#!/bin/bash" in script
        assert "az login" in script
        assert "az group create" in script
        assert "az deployment group create" in script
        assert "az acr create" in script

    def test_dockerfile_template(self):
        """Test Dockerfile template generation"""
        dockerfile = AzureTemplates.dockerfile_template()

        assert "FROM python:3.9-slim" in dockerfile
        assert "WORKDIR /app" in dockerfile
        assert "EXPOSE 8000" in dockerfile
        assert "HEALTHCHECK" in dockerfile
        assert "AZURE_CLIENT_ID" in dockerfile
        assert "uvicorn" in dockerfile

    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("pathlib.Path.chmod")
    def test_save_templates(self, mock_chmod, mock_open, mock_mkdir):
        """Test saving all templates to files"""
        output_dir = Path("/test/output")

        AzureTemplates.save_templates(output_dir)

        # Verify directories were created
        assert mock_mkdir.call_count >= 4  # Main dir + subdirs

        # Verify files were written
        assert mock_open.call_count >= 10  # Multiple template files

        # Verify script was made executable
        mock_chmod.assert_called_with(0o755)


@pytest.mark.integration
class TestAzureIntegration:
    """Integration tests for Azure provider (requires actual Azure access)"""

    @pytest.fixture
    def provider(self):
        """Create provider for integration tests"""
        # Skip if no Azure credentials
        pytest.importorskip("azure.identity")
        return AzureProvider("test-subscription", "test-rg", "eastus")

    def test_validate_credentials_integration(self, provider):
        """Integration test for credential validation"""
        # This will only pass if Azure CLI is configured
        # In CI/CD, this test should be marked as optional
        try:
            result = provider.validate_credentials()
            assert isinstance(result, bool)
        except Exception:
            pytest.skip("Azure credentials not configured")

    def test_cost_estimation_integration(self, provider):
        """Integration test for cost estimation"""
        resources = [
            {
                "type": "compute",
                "service_type": "container_instance",
                "cpu": 1,
                "memory": 1,
                "hours_per_month": 100,
            }
        ]

        costs = provider.estimate_costs(resources)

        assert "total" in costs
        assert costs["total"] >= 0
        assert isinstance(costs["compute"], (int, float))

    def test_get_availability_zones_integration(self, provider):
        """Integration test for getting availability zones"""
        try:
            zones = provider.get_availability_zones()
            assert isinstance(zones, list)
            assert len(zones) >= 1
        except Exception:
            pytest.skip("Azure CLI not configured or network issues")
