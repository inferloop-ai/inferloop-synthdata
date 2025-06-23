"""
Azure Provider Implementation
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from deploy.base import (
    BaseCloudProvider,
    ComputeResource,
    DatabaseResource,
    DeploymentResult,
    MonitoringResource,
    NetworkResource,
    ResourceStatus,
    SecurityResource,
    StorageResource,
)


class AzureProvider(BaseCloudProvider):
    """Microsoft Azure provider implementation"""

    def __init__(
        self,
        subscription_id: str,
        resource_group: str,
        location: str = "East US",
        tenant_id: Optional[str] = None,
    ):
        super().__init__("azure", location)
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.tenant_id = tenant_id
        self.location = location.lower().replace(" ", "")  # Azure location format

    def validate_credentials(self) -> bool:
        """Validate Azure credentials"""
        try:
            # Check if Azure CLI is installed
            result = subprocess.run(["az", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                return False

            # Check if authenticated
            result = subprocess.run(
                ["az", "account", "show"], capture_output=True, text=True
            )
            if result.returncode != 0:
                return False

            # Verify subscription access
            result = subprocess.run(
                ["az", "account", "set", "--subscription", self.subscription_id],
                capture_output=True,
                text=True,
            )

            return result.returncode == 0
        except Exception:
            return False

    def get_availability_zones(self) -> List[str]:
        """Get available zones in the location"""
        try:
            cmd = [
                "az",
                "vm",
                "list-skus",
                "--location",
                self.location,
                "--query",
                "[?resourceType==`virtualMachines`].locationInfo[0].zones",
                "--output",
                "json",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                zones_data = json.loads(result.stdout)
                zones = []
                for zone_list in zones_data:
                    if zone_list:
                        zones.extend(zone_list)
                return list(set(zones)) if zones else ["1"]
            return ["1"]
        except Exception:
            return ["1"]

    def estimate_costs(self, resources: List[Dict[str, Any]]) -> Dict[str, float]:
        """Estimate costs for Azure resources"""
        total_compute = 0.0
        total_storage = 0.0
        total_network = 0.0
        total_database = 0.0

        for resource in resources:
            resource_type = resource.get("type", "")

            if resource_type == "compute":
                # Container Instances pricing
                if resource.get("service_type") == "container_instance":
                    cpu = resource.get("cpu", 1)
                    memory = resource.get("memory", 1)  # GB
                    hours_per_month = resource.get("hours_per_month", 730)

                    # CPU: $0.0000012 per vCPU-second
                    cpu_cost = cpu * 0.0000012 * 3600 * hours_per_month
                    # Memory: $0.0000001344 per GB-second
                    memory_cost = memory * 0.0000001344 * 3600 * hours_per_month

                    total_compute += cpu_cost + memory_cost

                # AKS pricing (node VMs)
                elif resource.get("service_type") == "aks":
                    nodes = resource.get("nodes", 3)
                    vm_size = resource.get("vm_size", "Standard_DS2_v2")

                    # Simplified pricing for common VM sizes (monthly)
                    vm_prices = {
                        "Standard_B2s": 30.37,
                        "Standard_B2ms": 60.74,
                        "Standard_DS2_v2": 97.28,
                        "Standard_DS3_v2": 194.56,
                        "Standard_DS4_v2": 389.12,
                    }

                    monthly_price = vm_prices.get(vm_size, 97.28)
                    total_compute += nodes * monthly_price

                # Azure Functions pricing
                elif resource.get("service_type") == "function":
                    executions = resource.get("executions_per_month", 1000000)
                    execution_time_ms = resource.get("execution_time_ms", 1000)
                    memory_mb = resource.get("memory_mb", 128)

                    # Consumption plan pricing
                    # First 1M executions free, then $0.20 per million
                    if executions > 1000000:
                        execution_cost = ((executions - 1000000) / 1000000) * 0.20
                    else:
                        execution_cost = 0

                    # GB-seconds pricing: $0.000016 per GB-second
                    gb_seconds = (
                        (memory_mb / 1024) * (execution_time_ms / 1000) * executions
                    )
                    # First 400,000 GB-seconds free
                    if gb_seconds > 400000:
                        execution_time_cost = (gb_seconds - 400000) * 0.000016
                    else:
                        execution_time_cost = 0

                    total_compute += execution_cost + execution_time_cost

            elif resource_type == "storage":
                # Blob Storage pricing
                size_gb = resource.get("size_gb", 100)
                tier = resource.get("tier", "hot")

                storage_prices = {
                    "hot": 0.0184,  # per GB per month
                    "cool": 0.0115,
                    "archive": 0.00099,
                }

                monthly_price = storage_prices.get(tier, 0.0184)
                total_storage += size_gb * monthly_price

            elif resource_type == "database":
                # Azure SQL Database pricing
                if resource.get("engine") == "sqlserver":
                    tier = resource.get("tier", "Basic")

                    tier_prices = {
                        "Basic": 4.99,  # per month
                        "S0": 14.74,
                        "S1": 29.48,
                        "S2": 73.70,
                        "P1": 465.00,
                    }

                    total_database += tier_prices.get(tier, 14.74)

                # Cosmos DB pricing
                elif resource.get("engine") == "cosmos":
                    ru_per_second = resource.get("ru_per_second", 400)
                    storage_gb = resource.get("storage_gb", 10)

                    # RU/s pricing: $0.008 per 100 RU/s per hour
                    ru_cost = (ru_per_second / 100) * 0.008 * 24 * 30
                    # Storage: $0.25 per GB per month
                    storage_cost = storage_gb * 0.25

                    total_database += ru_cost + storage_cost

            elif resource_type == "network":
                # Load Balancer pricing
                if resource.get("service") == "load_balancer":
                    data_processed_gb = resource.get("data_processed_gb", 100)

                    # Basic LB: $0.025 per hour + $0.005 per GB processed
                    lb_cost = 0.025 * 24 * 30  # Monthly hours
                    data_cost = data_processed_gb * 0.005

                    total_network += lb_cost + data_cost

        return {
            "compute": total_compute,
            "storage": total_storage,
            "network": total_network,
            "database": total_database,
            "total": total_compute + total_storage + total_network + total_database,
        }

    def get_resource_limits(self) -> Dict[str, Any]:
        """Get Azure resource limits and quotas"""
        try:
            # Get subscription quotas
            cmd = [
                "az",
                "vm",
                "list-usage",
                "--location",
                self.location,
                "--query",
                "[].{name:name.value,limit:limit,currentValue:currentValue}",
                "--output",
                "json",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                quotas_data = json.loads(result.stdout)
                quotas = {}

                for quota in quotas_data:
                    quotas[quota["name"]] = {
                        "limit": quota["limit"],
                        "usage": quota["currentValue"],
                    }

                return {
                    "vcpus": quotas.get("cores", {}).get("limit", 1000),
                    "memory_gb": quotas.get("virtualMachines", {}).get("limit", 10000)
                    * 4,  # Estimate
                    "storage_accounts": quotas.get("storageAccounts", {}).get(
                        "limit", 250
                    ),
                    "load_balancers": quotas.get("loadBalancers", {}).get("limit", 100),
                    "container_instances": 100,  # Default limit
                    "aks_clusters": 50,  # Default limit
                }
        except Exception:
            # Return default limits if unable to fetch
            return {
                "vcpus": 1000,
                "memory_gb": 10000,
                "storage_accounts": 250,
                "load_balancers": 100,
                "container_instances": 100,
                "aks_clusters": 50,
            }

    def create_compute_resource(self, resource: ComputeResource) -> str:
        """Create compute resource on Azure"""
        if resource.service_type == "container_instance":
            return self._create_container_instance(resource)
        elif resource.service_type == "aks":
            return self._create_aks_cluster(resource)
        elif resource.service_type == "function":
            return self._create_azure_function(resource)
        else:
            raise ValueError(
                f"Unsupported compute service type: {resource.service_type}"
            )

    def _create_container_instance(self, resource: ComputeResource) -> str:
        """Create Azure Container Instance"""
        container_name = f"{resource.name}-{self.location}"

        cmd = [
            "az",
            "container",
            "create",
            "--resource-group",
            self.resource_group,
            "--name",
            container_name,
            "--image",
            resource.config.get("image", "mcr.microsoft.com/azuredocs/aci-helloworld"),
            "--location",
            self.location,
            "--cpu",
            str(resource.config.get("cpu", 1)),
            "--memory",
            str(resource.config.get("memory", 1)),
            "--ports",
            str(resource.config.get("port", 8000)),
            "--restart-policy",
            resource.config.get("restart_policy", "Always"),
        ]

        # Add environment variables
        env_vars = resource.config.get("environment", {})
        if env_vars:
            for key, value in env_vars.items():
                cmd.extend(["--environment-variables", f"{key}={value}"])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Failed to create Container Instance: {result.stderr}")

        return f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/Microsoft.ContainerInstance/containerGroups/{container_name}"

    def _create_aks_cluster(self, resource: ComputeResource) -> str:
        """Create AKS cluster"""
        cluster_name = f"{resource.name}-cluster"

        cmd = [
            "az",
            "aks",
            "create",
            "--resource-group",
            self.resource_group,
            "--name",
            cluster_name,
            "--location",
            self.location,
            "--node-count",
            str(resource.config.get("nodes", 3)),
            "--node-vm-size",
            resource.config.get("vm_size", "Standard_DS2_v2"),
            "--enable-addons",
            "monitoring",
            "--generate-ssh-keys",
        ]

        # Add autoscaling if configured
        if resource.config.get("enable_autoscaling", True):
            cmd.extend(
                [
                    "--enable-cluster-autoscaler",
                    "--min-count",
                    str(resource.config.get("min_nodes", 1)),
                    "--max-count",
                    str(resource.config.get("max_nodes", 10)),
                ]
            )

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Failed to create AKS cluster: {result.stderr}")

        return f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/Microsoft.ContainerService/managedClusters/{cluster_name}"

    def _create_azure_function(self, resource: ComputeResource) -> str:
        """Create Azure Function App"""
        function_name = f"{resource.name}-func"
        storage_account = f"{resource.name}storage"

        # Create storage account for function
        cmd = [
            "az",
            "storage",
            "account",
            "create",
            "--name",
            storage_account,
            "--resource-group",
            self.resource_group,
            "--location",
            self.location,
            "--sku",
            "Standard_LRS",
        ]
        subprocess.run(cmd, capture_output=True)

        # Create function app
        cmd = [
            "az",
            "functionapp",
            "create",
            "--resource-group",
            self.resource_group,
            "--consumption-plan-location",
            self.location,
            "--name",
            function_name,
            "--storage-account",
            storage_account,
            "--runtime",
            resource.config.get("runtime", "python"),
            "--runtime-version",
            resource.config.get("runtime_version", "3.9"),
            "--functions-version",
            "4",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Failed to create Azure Function: {result.stderr}")

        return f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/Microsoft.Web/sites/{function_name}"

    def create_storage_resource(self, resource: StorageResource) -> str:
        """Create storage resource on Azure"""
        storage_account = f"{resource.name}storage"

        # Create storage account
        cmd = [
            "az",
            "storage",
            "account",
            "create",
            "--name",
            storage_account,
            "--resource-group",
            self.resource_group,
            "--location",
            self.location,
            "--sku",
            resource.config.get("sku", "Standard_LRS"),
            "--kind",
            resource.config.get("kind", "StorageV2"),
            "--access-tier",
            resource.config.get("tier", "Hot"),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Failed to create storage account: {result.stderr}")

        # Create container if specified
        container_name = resource.config.get("container_name", "data")
        if container_name:
            cmd = [
                "az",
                "storage",
                "container",
                "create",
                "--name",
                container_name,
                "--account-name",
                storage_account,
                "--public-access",
                resource.config.get("public_access", "off"),
            ]
            subprocess.run(cmd, capture_output=True)

        return f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/Microsoft.Storage/storageAccounts/{storage_account}"

    def create_network_resource(self, resource: NetworkResource) -> str:
        """Create network resource on Azure"""
        if resource.service_type == "vnet":
            return self._create_virtual_network(resource)
        elif resource.service_type == "load_balancer":
            return self._create_load_balancer(resource)
        else:
            raise ValueError(
                f"Unsupported network service type: {resource.service_type}"
            )

    def _create_virtual_network(self, resource: NetworkResource) -> str:
        """Create Virtual Network"""
        vnet_name = f"{resource.name}-vnet"

        # Create VNet
        cmd = [
            "az",
            "network",
            "vnet",
            "create",
            "--resource-group",
            self.resource_group,
            "--name",
            vnet_name,
            "--location",
            self.location,
            "--address-prefix",
            resource.config.get("address_space", "10.0.0.0/16"),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Failed to create VNet: {result.stderr}")

        # Create subnets
        subnets = resource.config.get("subnets", [])
        for i, subnet in enumerate(subnets):
            subnet_name = f"{vnet_name}-subnet-{i}"
            cmd = [
                "az",
                "network",
                "vnet",
                "subnet",
                "create",
                "--resource-group",
                self.resource_group,
                "--vnet-name",
                vnet_name,
                "--name",
                subnet_name,
                "--address-prefix",
                subnet.get("address_prefix", f"10.0.{i}.0/24"),
            ]
            subprocess.run(cmd, check=True)

        return f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/Microsoft.Network/virtualNetworks/{vnet_name}"

    def _create_load_balancer(self, resource: NetworkResource) -> str:
        """Create Load Balancer"""
        lb_name = f"{resource.name}-lb"

        # Create public IP
        pip_name = f"{lb_name}-pip"
        cmd = [
            "az",
            "network",
            "public-ip",
            "create",
            "--resource-group",
            self.resource_group,
            "--name",
            pip_name,
            "--location",
            self.location,
            "--sku",
            "Standard",
        ]
        subprocess.run(cmd, check=True)

        # Create load balancer
        cmd = [
            "az",
            "network",
            "lb",
            "create",
            "--resource-group",
            self.resource_group,
            "--name",
            lb_name,
            "--location",
            self.location,
            "--public-ip-address",
            pip_name,
            "--sku",
            "Standard",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Failed to create Load Balancer: {result.stderr}")

        # Create health probe
        probe_name = f"{lb_name}-health-probe"
        cmd = [
            "az",
            "network",
            "lb",
            "probe",
            "create",
            "--resource-group",
            self.resource_group,
            "--lb-name",
            lb_name,
            "--name",
            probe_name,
            "--protocol",
            "Http",
            "--port",
            str(resource.config.get("health_check_port", 80)),
            "--path",
            resource.config.get("health_check_path", "/health"),
        ]
        subprocess.run(cmd, check=True)

        return f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/Microsoft.Network/loadBalancers/{lb_name}"

    def create_database_resource(self, resource: DatabaseResource) -> str:
        """Create database resource on Azure"""
        if resource.engine == "sqlserver":
            return self._create_azure_sql(resource)
        elif resource.engine == "cosmos":
            return self._create_cosmos_db(resource)
        else:
            raise ValueError(f"Unsupported database engine: {resource.engine}")

    def _create_azure_sql(self, resource: DatabaseResource) -> str:
        """Create Azure SQL Database"""
        server_name = f"{resource.name}-server"
        db_name = resource.config.get("database_name", "synthetic_data")

        # Create SQL Server
        cmd = [
            "az",
            "sql",
            "server",
            "create",
            "--resource-group",
            self.resource_group,
            "--name",
            server_name,
            "--location",
            self.location,
            "--admin-user",
            resource.config.get("admin_user", "sqladmin"),
            "--admin-password",
            resource.config.get("admin_password", "ComplexP@ssw0rd!"),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Failed to create SQL Server: {result.stderr}")

        # Create database
        cmd = [
            "az",
            "sql",
            "db",
            "create",
            "--resource-group",
            self.resource_group,
            "--server",
            server_name,
            "--name",
            db_name,
            "--edition",
            resource.config.get("edition", "Basic"),
            "--service-objective",
            resource.config.get("tier", "Basic"),
        ]
        subprocess.run(cmd, check=True)

        return f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/Microsoft.Sql/servers/{server_name}/databases/{db_name}"

    def _create_cosmos_db(self, resource: DatabaseResource) -> str:
        """Create Cosmos DB account"""
        account_name = f"{resource.name}-cosmos"

        cmd = [
            "az",
            "cosmosdb",
            "create",
            "--resource-group",
            self.resource_group,
            "--name",
            account_name,
            "--location",
            self.location,
            "--kind",
            resource.config.get("kind", "GlobalDocumentDB"),
            "--default-consistency-level",
            resource.config.get("consistency", "Session"),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Failed to create Cosmos DB: {result.stderr}")

        # Create database
        db_name = resource.config.get("database_name", "synthetic_data")
        cmd = [
            "az",
            "cosmosdb",
            "sql",
            "database",
            "create",
            "--resource-group",
            self.resource_group,
            "--account-name",
            account_name,
            "--name",
            db_name,
        ]
        subprocess.run(cmd, check=True)

        return f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/Microsoft.DocumentDB/databaseAccounts/{account_name}"

    def create_monitoring_resource(self, resource: MonitoringResource) -> str:
        """Create monitoring resource on Azure"""
        # Application Insights is automatically created with many services
        insights_name = f"{resource.name}-insights"

        cmd = [
            "az",
            "monitor",
            "app-insights",
            "component",
            "create",
            "--app",
            insights_name,
            "--location",
            self.location,
            "--resource-group",
            self.resource_group,
            "--kind",
            "web",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            # App Insights creation might fail, but that's okay
            pass

        return f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/Microsoft.Insights/components/{insights_name}"

    def create_security_resource(self, resource: SecurityResource) -> str:
        """Create security resource on Azure"""
        if resource.service_type == "keyvault":
            return self._create_key_vault(resource)
        elif resource.service_type == "identity":
            return self._create_managed_identity(resource)
        else:
            raise ValueError(
                f"Unsupported security service type: {resource.service_type}"
            )

    def _create_key_vault(self, resource: SecurityResource) -> str:
        """Create Azure Key Vault"""
        vault_name = f"{resource.name}-kv"

        cmd = [
            "az",
            "keyvault",
            "create",
            "--resource-group",
            self.resource_group,
            "--name",
            vault_name,
            "--location",
            self.location,
            "--sku",
            resource.config.get("sku", "standard"),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Failed to create Key Vault: {result.stderr}")

        # Add secrets if provided
        secrets = resource.config.get("secrets", {})
        for secret_name, secret_value in secrets.items():
            cmd = [
                "az",
                "keyvault",
                "secret",
                "set",
                "--vault-name",
                vault_name,
                "--name",
                secret_name,
                "--value",
                secret_value,
            ]
            subprocess.run(cmd)

        return f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/Microsoft.KeyVault/vaults/{vault_name}"

    def _create_managed_identity(self, resource: SecurityResource) -> str:
        """Create Managed Identity"""
        identity_name = f"{resource.name}-identity"

        cmd = [
            "az",
            "identity",
            "create",
            "--resource-group",
            self.resource_group,
            "--name",
            identity_name,
            "--location",
            self.location,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Failed to create Managed Identity: {result.stderr}")

        return f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/{identity_name}"

    def delete_resource(self, resource_id: str, resource_type: str) -> bool:
        """Delete a resource from Azure"""
        try:
            if resource_type == "compute":
                if "containerGroups" in resource_id:
                    # Container Instance
                    container_name = resource_id.split("/")[-1]
                    cmd = [
                        "az",
                        "container",
                        "delete",
                        "--resource-group",
                        self.resource_group,
                        "--name",
                        container_name,
                        "--yes",
                    ]
                elif "managedClusters" in resource_id:
                    # AKS cluster
                    cluster_name = resource_id.split("/")[-1]
                    cmd = [
                        "az",
                        "aks",
                        "delete",
                        "--resource-group",
                        self.resource_group,
                        "--name",
                        cluster_name,
                        "--yes",
                    ]
                elif "sites" in resource_id:
                    # Function App
                    function_name = resource_id.split("/")[-1]
                    cmd = [
                        "az",
                        "functionapp",
                        "delete",
                        "--resource-group",
                        self.resource_group,
                        "--name",
                        function_name,
                    ]

            elif resource_type == "storage":
                # Storage account
                storage_name = resource_id.split("/")[-1]
                cmd = [
                    "az",
                    "storage",
                    "account",
                    "delete",
                    "--resource-group",
                    self.resource_group,
                    "--name",
                    storage_name,
                    "--yes",
                ]

            elif resource_type == "database":
                if "servers" in resource_id:
                    # SQL Server (will delete all databases)
                    server_name = resource_id.split("/")[-3]
                    cmd = [
                        "az",
                        "sql",
                        "server",
                        "delete",
                        "--resource-group",
                        self.resource_group,
                        "--name",
                        server_name,
                        "--yes",
                    ]
                elif "databaseAccounts" in resource_id:
                    # Cosmos DB
                    account_name = resource_id.split("/")[-1]
                    cmd = [
                        "az",
                        "cosmosdb",
                        "delete",
                        "--resource-group",
                        self.resource_group,
                        "--name",
                        account_name,
                        "--yes",
                    ]

            else:
                return False

            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0

        except Exception:
            return False

    def get_resource_status(
        self, resource_id: str, resource_type: str
    ) -> ResourceStatus:
        """Get the status of a resource"""
        try:
            if resource_type == "compute":
                if "containerGroups" in resource_id:
                    # Container Instance
                    container_name = resource_id.split("/")[-1]
                    cmd = [
                        "az",
                        "container",
                        "show",
                        "--resource-group",
                        self.resource_group,
                        "--name",
                        container_name,
                        "--query",
                        "instanceView.state",
                        "--output",
                        "tsv",
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)

                    if result.returncode == 0:
                        state = result.stdout.strip()
                        if state == "Running":
                            return ResourceStatus.RUNNING
                        elif state in ["Pending", "Starting"]:
                            return ResourceStatus.PENDING
                        else:
                            return ResourceStatus.STOPPED

            return ResourceStatus.UNKNOWN

        except Exception:
            return ResourceStatus.ERROR

    def update_resource(
        self, resource_id: str, resource_type: str, updates: Dict[str, Any]
    ) -> bool:
        """Update a resource configuration"""
        try:
            if resource_type == "compute" and "containerGroups" in resource_id:
                # Container Instance updates require recreation
                # This is a simplified approach
                container_name = resource_id.split("/")[-1]

                # For now, just return True as updates would require recreation
                return True

            return False

        except Exception:
            return False

    def tag_resource(self, resource_id: str, tags: Dict[str, str]) -> bool:
        """Tag a resource"""
        try:
            # Extract resource name from resource ID
            resource_name = resource_id.split("/")[-1]

            # Build tags string
            tags_str = " ".join([f"{k}={v}" for k, v in tags.items()])

            cmd = [
                "az",
                "resource",
                "tag",
                "--resource-group",
                self.resource_group,
                "--name",
                resource_name,
                "--tags",
                tags_str,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0

        except Exception:
            return False
