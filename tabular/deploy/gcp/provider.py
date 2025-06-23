"""
Google Cloud Platform (GCP) Provider Implementation
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


class GCPProvider(BaseCloudProvider):
    """Google Cloud Platform provider implementation"""

    def __init__(
        self,
        project_id: str,
        region: str = "us-central1",
        credentials_path: Optional[str] = None,
    ):
        super().__init__("gcp", region)
        self.project_id = project_id
        self.credentials_path = credentials_path or os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )
        self.zone = f"{region}-a"  # Default zone

    def validate_credentials(self) -> bool:
        """Validate GCP credentials"""
        try:
            # Check if gcloud is installed
            result = subprocess.run(
                ["gcloud", "--version"], capture_output=True, text=True
            )
            if result.returncode != 0:
                return False

            # Check if authenticated
            result = subprocess.run(
                ["gcloud", "auth", "list"], capture_output=True, text=True
            )
            return result.returncode == 0
        except Exception:
            return False

    def get_availability_zones(self) -> List[str]:
        """Get available zones in the region"""
        try:
            cmd = [
                "gcloud",
                "compute",
                "zones",
                "list",
                "--filter",
                f"region:{self.region}",
                "--format",
                "value(name)",
                "--project",
                self.project_id,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip().split("\n")
            return [self.zone]
        except Exception:
            return [self.zone]

    def estimate_costs(self, resources: List[Dict[str, Any]]) -> Dict[str, float]:
        """Estimate costs for GCP resources"""
        total_compute = 0.0
        total_storage = 0.0
        total_network = 0.0
        total_database = 0.0

        for resource in resources:
            resource_type = resource.get("type", "")

            if resource_type == "compute":
                # Cloud Run pricing (per million requests + CPU/memory)
                if resource.get("service_type") == "cloud_run":
                    cpu = resource.get("cpu", 1)
                    memory = resource.get("memory", 512)
                    requests_per_month = resource.get("requests_per_month", 1000000)

                    # CPU: $0.00002400 per vCPU-second
                    cpu_cost = cpu * 0.00002400 * 3600 * 24 * 30
                    # Memory: $0.00000250 per GiB-second
                    memory_cost = (memory / 1024) * 0.00000250 * 3600 * 24 * 30
                    # Requests: $0.40 per million
                    request_cost = (requests_per_month / 1000000) * 0.40

                    total_compute += cpu_cost + memory_cost + request_cost

                # GKE pricing
                elif resource.get("service_type") == "gke":
                    nodes = resource.get("nodes", 3)
                    machine_type = resource.get("machine_type", "e2-standard-4")

                    # Simplified pricing for common machine types
                    machine_prices = {
                        "e2-micro": 0.0084,
                        "e2-small": 0.0168,
                        "e2-medium": 0.0334,
                        "e2-standard-2": 0.0668,
                        "e2-standard-4": 0.1336,
                        "e2-standard-8": 0.2672,
                    }

                    hourly_price = machine_prices.get(machine_type, 0.1336)
                    total_compute += nodes * hourly_price * 24 * 30

            elif resource_type == "storage":
                # Cloud Storage pricing
                size_gb = resource.get("size_gb", 100)
                storage_class = resource.get("storage_class", "standard")

                storage_prices = {
                    "standard": 0.020,  # per GB per month
                    "nearline": 0.010,
                    "coldline": 0.004,
                    "archive": 0.0012,
                }

                monthly_price = storage_prices.get(storage_class, 0.020)
                total_storage += size_gb * monthly_price

            elif resource_type == "database":
                # Cloud SQL pricing
                if resource.get("engine") == "postgres":
                    tier = resource.get("tier", "db-f1-micro")
                    storage_gb = resource.get("storage_gb", 10)

                    tier_prices = {
                        "db-f1-micro": 0.0150,
                        "db-g1-small": 0.0500,
                        "db-n1-standard-1": 0.0965,
                        "db-n1-standard-2": 0.1930,
                        "db-n1-standard-4": 0.3860,
                    }

                    hourly_price = tier_prices.get(tier, 0.0965)
                    total_database += hourly_price * 24 * 30
                    total_database += storage_gb * 0.17  # Storage price

            elif resource_type == "network":
                # Load balancer pricing
                if resource.get("service") == "load_balancer":
                    forwarding_rules = resource.get("forwarding_rules", 1)
                    total_network += forwarding_rules * 0.025 * 24 * 30

        return {
            "compute": total_compute,
            "storage": total_storage,
            "network": total_network,
            "database": total_database,
            "total": total_compute + total_storage + total_network + total_database,
        }

    def get_resource_limits(self) -> Dict[str, Any]:
        """Get GCP resource limits and quotas"""
        try:
            # Get project quotas
            cmd = [
                "gcloud",
                "compute",
                "project-info",
                "describe",
                "--project",
                self.project_id,
                "--format",
                "json",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                project_info = json.loads(result.stdout)
                quotas = {}

                for quota in project_info.get("quotas", []):
                    quotas[quota["metric"]] = {
                        "limit": quota["limit"],
                        "usage": quota["usage"],
                    }

                return {
                    "vcpus": quotas.get("CPUS", {}).get("limit", 1000),
                    "memory_gb": quotas.get("MEMORY_GB", {}).get("limit", 10000),
                    "storage_tb": quotas.get("DISKS_TOTAL_GB", {}).get("limit", 100000)
                    / 1024,
                    "load_balancers": quotas.get("FORWARDING_RULES", {}).get(
                        "limit", 100
                    ),
                    "cloud_run_services": 1000,  # Default limit
                    "gke_clusters": quotas.get("GKE_CLUSTERS", {}).get("limit", 50),
                }
        except Exception:
            # Return default limits if unable to fetch
            return {
                "vcpus": 1000,
                "memory_gb": 10000,
                "storage_tb": 100,
                "load_balancers": 100,
                "cloud_run_services": 1000,
                "gke_clusters": 50,
            }

    def create_compute_resource(self, resource: ComputeResource) -> str:
        """Create compute resource on GCP"""
        if resource.service_type == "cloud_run":
            return self._create_cloud_run_service(resource)
        elif resource.service_type == "gke":
            return self._create_gke_cluster(resource)
        elif resource.service_type == "cloud_function":
            return self._create_cloud_function(resource)
        else:
            raise ValueError(
                f"Unsupported compute service type: {resource.service_type}"
            )

    def _create_cloud_run_service(self, resource: ComputeResource) -> str:
        """Create Cloud Run service"""
        service_name = f"{resource.name}-{self.region}"

        # Build the deployment command
        cmd = [
            "gcloud",
            "run",
            "deploy",
            service_name,
            "--image",
            resource.config.get("image", "gcr.io/cloudrun/hello"),
            "--platform",
            "managed",
            "--region",
            self.region,
            "--project",
            self.project_id,
            "--memory",
            f"{resource.config.get('memory', 512)}Mi",
            "--cpu",
            str(resource.config.get("cpu", 1)),
            "--max-instances",
            str(resource.config.get("max_instances", 10)),
            "--allow-unauthenticated",
        ]

        # Add environment variables
        env_vars = resource.config.get("environment", {})
        if env_vars:
            env_str = ",".join([f"{k}={v}" for k, v in env_vars.items()])
            cmd.extend(["--set-env-vars", env_str])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Failed to create Cloud Run service: {result.stderr}")

        return f"projects/{self.project_id}/locations/{self.region}/services/{service_name}"

    def _create_gke_cluster(self, resource: ComputeResource) -> str:
        """Create GKE cluster"""
        cluster_name = f"{resource.name}-cluster"

        cmd = [
            "gcloud",
            "container",
            "clusters",
            "create",
            cluster_name,
            "--zone",
            self.zone,
            "--project",
            self.project_id,
            "--num-nodes",
            str(resource.config.get("nodes", 3)),
            "--machine-type",
            resource.config.get("machine_type", "e2-standard-4"),
            "--enable-autoscaling",
            "--min-nodes",
            str(resource.config.get("min_nodes", 1)),
            "--max-nodes",
            str(resource.config.get("max_nodes", 10)),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Failed to create GKE cluster: {result.stderr}")

        return (
            f"projects/{self.project_id}/locations/{self.zone}/clusters/{cluster_name}"
        )

    def _create_cloud_function(self, resource: ComputeResource) -> str:
        """Create Cloud Function"""
        function_name = f"{resource.name}-function"

        cmd = [
            "gcloud",
            "functions",
            "deploy",
            function_name,
            "--runtime",
            resource.config.get("runtime", "python39"),
            "--trigger-http",
            "--entry-point",
            resource.config.get("entry_point", "main"),
            "--memory",
            f"{resource.config.get('memory', 256)}MB",
            "--region",
            self.region,
            "--project",
            self.project_id,
            "--allow-unauthenticated",
        ]

        # Add source if provided
        source = resource.config.get("source")
        if source:
            cmd.extend(["--source", source])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Failed to create Cloud Function: {result.stderr}")

        return f"projects/{self.project_id}/locations/{self.region}/functions/{function_name}"

    def create_storage_resource(self, resource: StorageResource) -> str:
        """Create storage resource on GCP"""
        bucket_name = f"{resource.name}-{self.project_id}"

        cmd = [
            "gsutil",
            "mb",
            "-p",
            self.project_id,
            "-c",
            resource.config.get("storage_class", "STANDARD"),
            "-l",
            self.region,
            f"gs://{bucket_name}",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Failed to create storage bucket: {result.stderr}")

        # Set lifecycle rules if provided
        lifecycle_rules = resource.config.get("lifecycle_rules", [])
        if lifecycle_rules:
            lifecycle_config = {"lifecycle": {"rule": lifecycle_rules}}

            # Write lifecycle config to temp file
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(lifecycle_config, f)
                lifecycle_file = f.name

            try:
                cmd = [
                    "gsutil",
                    "lifecycle",
                    "set",
                    lifecycle_file,
                    f"gs://{bucket_name}",
                ]
                subprocess.run(cmd, check=True)
            finally:
                os.unlink(lifecycle_file)

        return f"gs://{bucket_name}"

    def create_network_resource(self, resource: NetworkResource) -> str:
        """Create network resource on GCP"""
        if resource.service_type == "vpc":
            return self._create_vpc(resource)
        elif resource.service_type == "load_balancer":
            return self._create_load_balancer(resource)
        else:
            raise ValueError(
                f"Unsupported network service type: {resource.service_type}"
            )

    def _create_vpc(self, resource: NetworkResource) -> str:
        """Create VPC network"""
        network_name = f"{resource.name}-vpc"

        # Create VPC
        cmd = [
            "gcloud",
            "compute",
            "networks",
            "create",
            network_name,
            "--subnet-mode",
            "custom",
            "--project",
            self.project_id,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Failed to create VPC: {result.stderr}")

        # Create subnets
        subnets = resource.config.get("subnets", [])
        for i, subnet in enumerate(subnets):
            subnet_name = f"{network_name}-subnet-{i}"
            cmd = [
                "gcloud",
                "compute",
                "networks",
                "subnets",
                "create",
                subnet_name,
                "--network",
                network_name,
                "--range",
                subnet.get("cidr", "10.0.0.0/24"),
                "--region",
                self.region,
                "--project",
                self.project_id,
            ]
            subprocess.run(cmd, check=True)

        return f"projects/{self.project_id}/global/networks/{network_name}"

    def _create_load_balancer(self, resource: NetworkResource) -> str:
        """Create load balancer"""
        lb_name = f"{resource.name}-lb"

        # Create health check
        health_check_name = f"{lb_name}-health-check"
        cmd = [
            "gcloud",
            "compute",
            "health-checks",
            "create",
            "http",
            health_check_name,
            "--port",
            str(resource.config.get("health_check_port", 80)),
            "--request-path",
            resource.config.get("health_check_path", "/health"),
            "--project",
            self.project_id,
        ]
        subprocess.run(cmd, check=True)

        # Create backend service
        backend_name = f"{lb_name}-backend"
        cmd = [
            "gcloud",
            "compute",
            "backend-services",
            "create",
            backend_name,
            "--protocol",
            "HTTP",
            "--health-checks",
            health_check_name,
            "--global",
            "--project",
            self.project_id,
        ]
        subprocess.run(cmd, check=True)

        # Create URL map
        url_map_name = f"{lb_name}-url-map"
        cmd = [
            "gcloud",
            "compute",
            "url-maps",
            "create",
            url_map_name,
            "--default-service",
            backend_name,
            "--project",
            self.project_id,
        ]
        subprocess.run(cmd, check=True)

        # Create HTTP proxy
        proxy_name = f"{lb_name}-proxy"
        cmd = [
            "gcloud",
            "compute",
            "target-http-proxies",
            "create",
            proxy_name,
            "--url-map",
            url_map_name,
            "--project",
            self.project_id,
        ]
        subprocess.run(cmd, check=True)

        # Create forwarding rule
        cmd = [
            "gcloud",
            "compute",
            "forwarding-rules",
            "create",
            f"{lb_name}-forwarding-rule",
            "--target-http-proxy",
            proxy_name,
            "--ports",
            "80",
            "--global",
            "--project",
            self.project_id,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Failed to create load balancer: {result.stderr}")

        return f"projects/{self.project_id}/global/forwardingRules/{lb_name}-forwarding-rule"

    def create_database_resource(self, resource: DatabaseResource) -> str:
        """Create database resource on GCP"""
        if resource.engine == "postgres":
            return self._create_cloud_sql_postgres(resource)
        elif resource.engine == "firestore":
            return self._create_firestore(resource)
        else:
            raise ValueError(f"Unsupported database engine: {resource.engine}")

    def _create_cloud_sql_postgres(self, resource: DatabaseResource) -> str:
        """Create Cloud SQL PostgreSQL instance"""
        instance_name = f"{resource.name}-{self.region}"

        cmd = [
            "gcloud",
            "sql",
            "instances",
            "create",
            instance_name,
            "--database-version",
            f"POSTGRES_{resource.config.get('version', '13')}",
            "--tier",
            resource.config.get("tier", "db-f1-micro"),
            "--region",
            self.region,
            "--project",
            self.project_id,
            "--storage-size",
            f"{resource.config.get('storage_gb', 10)}GB",
            "--storage-type",
            resource.config.get("storage_type", "SSD"),
            "--backup-start-time",
            "03:00",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Failed to create Cloud SQL instance: {result.stderr}")

        # Create database
        db_name = resource.config.get("database_name", "synthetic_data")
        cmd = [
            "gcloud",
            "sql",
            "databases",
            "create",
            db_name,
            "--instance",
            instance_name,
            "--project",
            self.project_id,
        ]
        subprocess.run(cmd, check=True)

        return f"projects/{self.project_id}/instances/{instance_name}"

    def _create_firestore(self, resource: DatabaseResource) -> str:
        """Create Firestore database"""
        # Firestore is automatically created with the project
        # This would configure collections and indexes

        return f"projects/{self.project_id}/databases/(default)"

    def create_monitoring_resource(self, resource: MonitoringResource) -> str:
        """Create monitoring resource on GCP"""
        # Cloud Monitoring is automatically enabled
        # This would configure alerts and dashboards

        # Create alert policy if specified
        alerts = resource.config.get("alerts", [])
        for alert in alerts:
            self._create_alert_policy(alert)

        return f"projects/{self.project_id}/monitoring"

    def _create_alert_policy(self, alert_config: Dict[str, Any]):
        """Create Cloud Monitoring alert policy"""
        # This would use the monitoring API to create alerts
        pass

    def create_security_resource(self, resource: SecurityResource) -> str:
        """Create security resource on GCP"""
        if resource.service_type == "iam":
            return self._configure_iam(resource)
        elif resource.service_type == "secret":
            return self._create_secret(resource)
        else:
            raise ValueError(
                f"Unsupported security service type: {resource.service_type}"
            )

    def _configure_iam(self, resource: SecurityResource) -> str:
        """Configure IAM roles and policies"""
        service_account = resource.config.get("service_account")
        if service_account:
            # Create service account
            sa_email = f"{service_account}@{self.project_id}.iam.gserviceaccount.com"
            cmd = [
                "gcloud",
                "iam",
                "service-accounts",
                "create",
                service_account,
                "--display-name",
                service_account,
                "--project",
                self.project_id,
            ]
            subprocess.run(cmd)

            # Assign roles
            roles = resource.config.get("roles", [])
            for role in roles:
                cmd = [
                    "gcloud",
                    "projects",
                    "add-iam-policy-binding",
                    self.project_id,
                    "--member",
                    f"serviceAccount:{sa_email}",
                    "--role",
                    role,
                ]
                subprocess.run(cmd)

            return sa_email

        return f"projects/{self.project_id}/iam"

    def _create_secret(self, resource: SecurityResource) -> str:
        """Create secret in Secret Manager"""
        secret_name = resource.name

        cmd = ["gcloud", "secrets", "create", secret_name, "--project", self.project_id]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Failed to create secret: {result.stderr}")

        # Add secret version if value provided
        secret_value = resource.config.get("value")
        if secret_value:
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
                f.write(secret_value)
                secret_file = f.name

            try:
                cmd = [
                    "gcloud",
                    "secrets",
                    "versions",
                    "add",
                    secret_name,
                    "--data-file",
                    secret_file,
                    "--project",
                    self.project_id,
                ]
                subprocess.run(cmd, check=True)
            finally:
                os.unlink(secret_file)

        return f"projects/{self.project_id}/secrets/{secret_name}"

    def delete_resource(self, resource_id: str, resource_type: str) -> bool:
        """Delete a resource from GCP"""
        try:
            if resource_type == "compute":
                if "services" in resource_id:
                    # Cloud Run service
                    service_name = resource_id.split("/")[-1]
                    cmd = [
                        "gcloud",
                        "run",
                        "services",
                        "delete",
                        service_name,
                        "--region",
                        self.region,
                        "--project",
                        self.project_id,
                        "--quiet",
                    ]
                elif "clusters" in resource_id:
                    # GKE cluster
                    cluster_name = resource_id.split("/")[-1]
                    cmd = [
                        "gcloud",
                        "container",
                        "clusters",
                        "delete",
                        cluster_name,
                        "--zone",
                        self.zone,
                        "--project",
                        self.project_id,
                        "--quiet",
                    ]
                elif "functions" in resource_id:
                    # Cloud Function
                    function_name = resource_id.split("/")[-1]
                    cmd = [
                        "gcloud",
                        "functions",
                        "delete",
                        function_name,
                        "--region",
                        self.region,
                        "--project",
                        self.project_id,
                        "--quiet",
                    ]

            elif resource_type == "storage":
                # Storage bucket
                bucket_name = resource_id.replace("gs://", "")
                cmd = ["gsutil", "rm", "-r", f"gs://{bucket_name}"]

            elif resource_type == "database":
                if "instances" in resource_id:
                    # Cloud SQL instance
                    instance_name = resource_id.split("/")[-1]
                    cmd = [
                        "gcloud",
                        "sql",
                        "instances",
                        "delete",
                        instance_name,
                        "--project",
                        self.project_id,
                        "--quiet",
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
                if "services" in resource_id:
                    # Cloud Run service
                    service_name = resource_id.split("/")[-1]
                    cmd = [
                        "gcloud",
                        "run",
                        "services",
                        "describe",
                        service_name,
                        "--region",
                        self.region,
                        "--project",
                        self.project_id,
                        "--format",
                        "value(status.conditions[0].status)",
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)

                    if result.returncode == 0:
                        status = result.stdout.strip()
                        if status == "True":
                            return ResourceStatus.RUNNING
                        else:
                            return ResourceStatus.PENDING

            return ResourceStatus.UNKNOWN

        except Exception:
            return ResourceStatus.ERROR

    def update_resource(
        self, resource_id: str, resource_type: str, updates: Dict[str, Any]
    ) -> bool:
        """Update a resource configuration"""
        try:
            if resource_type == "compute" and "services" in resource_id:
                # Update Cloud Run service
                service_name = resource_id.split("/")[-1]
                cmd = [
                    "gcloud",
                    "run",
                    "services",
                    "update",
                    service_name,
                    "--region",
                    self.region,
                    "--project",
                    self.project_id,
                ]

                # Add update parameters
                if "memory" in updates:
                    cmd.extend(["--memory", f"{updates['memory']}Mi"])
                if "cpu" in updates:
                    cmd.extend(["--cpu", str(updates["cpu"])])
                if "max_instances" in updates:
                    cmd.extend(["--max-instances", str(updates["max_instances"])])
                if "environment" in updates:
                    env_str = ",".join(
                        [f"{k}={v}" for k, v in updates["environment"].items()]
                    )
                    cmd.extend(["--set-env-vars", env_str])

                result = subprocess.run(cmd, capture_output=True, text=True)
                return result.returncode == 0

            return False

        except Exception:
            return False

    def tag_resource(self, resource_id: str, tags: Dict[str, str]) -> bool:
        """Tag a resource with labels"""
        try:
            if "services" in resource_id:
                # Cloud Run service
                service_name = resource_id.split("/")[-1]
                labels = ",".join([f"{k}={v}" for k, v in tags.items()])

                cmd = [
                    "gcloud",
                    "run",
                    "services",
                    "update",
                    service_name,
                    "--region",
                    self.region,
                    "--project",
                    self.project_id,
                    "--update-labels",
                    labels,
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)
                return result.returncode == 0

            return False

        except Exception:
            return False
