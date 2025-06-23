"""AWS infrastructure provider implementation."""

import boto3
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from common.core.base_provider import (
    BaseInfrastructureProvider,
    ResourceInfo,
    ResourceStatus,
    ComputeConfig,
    StorageConfig,
    NetworkConfig,
    ApplicationConfig,
)
from common.core.config import InfrastructureConfig
from common.core.exceptions import (
    InfrastructureError,
    AuthenticationError,
    ResourceCreationError,
)

from .compute import AWSCompute
from .storage import AWSStorage
from .networking import AWSNetworking
from .security import AWSSecurity
from .monitoring import AWSMonitoring


class AWSProvider(BaseInfrastructureProvider):
    """AWS infrastructure provider."""
    
    def __init__(self, config: InfrastructureConfig):
        """Initialize AWS provider."""
        super().__init__(config.to_provider_config())
        self.config = config
        self.session = None
        self.compute = None
        self.storage = None
        self.networking = None
        self.security = None
        self.monitoring = None
        
        # AWS clients
        self.ec2_client = None
        self.s3_client = None
        self.ecs_client = None
        self.cloudwatch_client = None
        self.iam_client = None
        self.pricing_client = None
    
    def initialize(self) -> None:
        """Initialize AWS connection and services."""
        try:
            # Create AWS session
            session_config = {}
            if self.config.region:
                session_config["region_name"] = self.config.region
            
            self.session = boto3.Session(**session_config)
            
            # Initialize clients
            self.ec2_client = self.session.client("ec2")
            self.s3_client = self.session.client("s3")
            self.ecs_client = self.session.client("ecs")
            self.cloudwatch_client = self.session.client("cloudwatch")
            self.iam_client = self.session.client("iam")
            self.pricing_client = self.session.client("pricing", region_name="us-east-1")
            
            # Verify credentials
            self.iam_client.get_user()
            
            # Initialize service modules
            self.compute = AWSCompute(self.session, self.config)
            self.storage = AWSStorage(self.session, self.config)
            self.networking = AWSNetworking(self.session, self.config)
            self.security = AWSSecurity(self.session, self.config)
            self.monitoring = AWSMonitoring(self.session, self.config)
            
        except Exception as e:
            raise AuthenticationError("AWS", str(e))
    
    def create_compute_instance(self, config: ComputeConfig) -> ResourceInfo:
        """Create an EC2 instance."""
        if not self.compute:
            raise InfrastructureError("Provider not initialized")
        
        return self.compute.create_instance(config)
    
    def create_storage_bucket(self, config: StorageConfig) -> ResourceInfo:
        """Create an S3 bucket."""
        if not self.storage:
            raise InfrastructureError("Provider not initialized")
        
        return self.storage.create_bucket(config)
    
    def create_network(self, config: NetworkConfig) -> ResourceInfo:
        """Create a VPC with subnets."""
        if not self.networking:
            raise InfrastructureError("Provider not initialized")
        
        return self.networking.create_vpc(config)
    
    def deploy_application(self, app_config: ApplicationConfig) -> ResourceInfo:
        """Deploy application using ECS Fargate."""
        if not self.compute:
            raise InfrastructureError("Provider not initialized")
        
        return self.compute.deploy_fargate_service(app_config)
    
    def get_resource(self, resource_id: str) -> Optional[ResourceInfo]:
        """Get information about a specific resource."""
        # Check if it's in our tracked resources
        if resource_id in self.resources:
            return self.resources[resource_id]
        
        # Try to find resource in AWS
        try:
            # Check EC2 instances
            if resource_id.startswith("i-"):
                return self.compute.get_instance(resource_id)
            # Check S3 buckets
            elif "/" not in resource_id:  # Simple bucket name
                return self.storage.get_bucket(resource_id)
            # Check VPCs
            elif resource_id.startswith("vpc-"):
                return self.networking.get_vpc(resource_id)
            # Check ECS services
            elif "/" in resource_id and "service" in resource_id:
                return self.compute.get_service(resource_id)
        except Exception:
            pass
        
        return None
    
    def list_resources(self, resource_type: Optional[str] = None) -> List[ResourceInfo]:
        """List all resources."""
        resources = []
        
        if not resource_type or resource_type == "compute":
            resources.extend(self.compute.list_instances())
            resources.extend(self.compute.list_services())
        
        if not resource_type or resource_type == "storage":
            resources.extend(self.storage.list_buckets())
        
        if not resource_type or resource_type == "network":
            resources.extend(self.networking.list_vpcs())
        
        return resources
    
    def delete_resource(self, resource_id: str) -> bool:
        """Delete a specific resource."""
        resource = self.get_resource(resource_id)
        if not resource:
            return False
        
        try:
            if resource.resource_type == "ec2_instance":
                return self.compute.delete_instance(resource_id)
            elif resource.resource_type == "s3_bucket":
                return self.storage.delete_bucket(resource_id)
            elif resource.resource_type == "vpc":
                return self.networking.delete_vpc(resource_id)
            elif resource.resource_type == "ecs_service":
                return self.compute.delete_service(resource_id)
            else:
                return False
        except Exception as e:
            raise InfrastructureError(f"Failed to delete resource: {str(e)}")
    
    def update_resource(self, resource_id: str, updates: Dict[str, Any]) -> ResourceInfo:
        """Update a resource configuration."""
        resource = self.get_resource(resource_id)
        if not resource:
            raise ResourceNotFoundError(resource_id)
        
        # Update based on resource type
        if resource.resource_type == "ec2_instance":
            return self.compute.update_instance(resource_id, updates)
        elif resource.resource_type == "ecs_service":
            return self.compute.update_service(resource_id, updates)
        else:
            raise InfrastructureError(f"Update not supported for {resource.resource_type}")
    
    def get_resource_metrics(
        self,
        resource_id: str,
        metric_names: List[str],
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Any]:
        """Get CloudWatch metrics for a resource."""
        if not self.monitoring:
            raise InfrastructureError("Provider not initialized")
        
        return self.monitoring.get_metrics(resource_id, metric_names, start_time, end_time)
    
    def estimate_cost(self, resources: List[ResourceConfig]) -> Dict[str, float]:
        """Estimate AWS costs for resources."""
        total_cost = 0.0
        breakdown = {}
        
        for resource in resources:
            if isinstance(resource, ComputeConfig):
                # Estimate EC2 costs
                instance_type = resource.metadata.get("instance_type", "t3.medium")
                hours_per_month = 730
                
                try:
                    response = self.pricing_client.get_products(
                        ServiceCode="AmazonEC2",
                        Filters=[
                            {
                                "Type": "TERM_MATCH",
                                "Field": "instanceType",
                                "Value": instance_type,
                            },
                            {
                                "Type": "TERM_MATCH",
                                "Field": "location",
                                "Value": self._get_region_name(),
                            },
                            {
                                "Type": "TERM_MATCH",
                                "Field": "operatingSystem",
                                "Value": "Linux",
                            },
                        ],
                        MaxResults=1,
                    )
                    
                    if response["PriceList"]:
                        price_item = json.loads(response["PriceList"][0])
                        on_demand = price_item["terms"]["OnDemand"]
                        price_dimension = list(list(on_demand.values())[0]["priceDimensions"].values())[0]
                        hourly_price = float(price_dimension["pricePerUnit"]["USD"])
                        
                        monthly_cost = hourly_price * hours_per_month * resource.min_instances
                        breakdown[f"EC2-{resource.name}"] = monthly_cost
                        total_cost += monthly_cost
                except Exception:
                    # Fallback to rough estimates
                    breakdown[f"EC2-{resource.name}"] = 100.0
                    total_cost += 100.0
            
            elif isinstance(resource, StorageConfig):
                # Estimate S3 costs (rough estimate)
                storage_gb = resource.metadata.get("estimated_size_gb", 100)
                storage_cost = storage_gb * 0.023  # S3 Standard pricing
                breakdown[f"S3-{resource.name}"] = storage_cost
                total_cost += storage_cost
        
        return {
            "total_monthly_cost": total_cost,
            "breakdown": breakdown,
            "currency": "USD",
        }
    
    def _get_region_name(self) -> str:
        """Get human-readable region name."""
        region_map = {
            "us-east-1": "US East (N. Virginia)",
            "us-west-2": "US West (Oregon)",
            "eu-west-1": "EU (Ireland)",
            "eu-central-1": "EU (Frankfurt)",
            "ap-southeast-1": "Asia Pacific (Singapore)",
            "ap-northeast-1": "Asia Pacific (Tokyo)",
        }
        return region_map.get(self.config.region, self.config.region)
    
    def create_full_stack(self) -> Dict[str, ResourceInfo]:
        """Create a complete infrastructure stack."""
        stack = {}
        
        try:
            # 1. Create VPC and networking
            network_config = NetworkConfig(
                name=f"{self.config.resource_name}-vpc",
                resource_type="vpc",
                region=self.config.region,
                environment=self.config.environment.value,
                cidr_block=self.config.vpc_cidr,
                enable_dns=True,
                enable_firewall=self.config.enable_firewall,
            )
            stack["vpc"] = self.create_network(network_config)
            
            # 2. Create storage bucket
            storage_config = StorageConfig(
                name=f"{self.config.resource_name}-data",
                resource_type="s3_bucket",
                region=self.config.region,
                environment=self.config.environment.value,
                encryption_enabled=self.config.storage_encryption,
                versioning_enabled=self.config.enable_versioning,
            )
            stack["storage"] = self.create_storage_bucket(storage_config)
            
            # 3. Deploy application
            app_config = ApplicationConfig(
                name=f"{self.config.resource_name}-app",
                version="latest",
                image=self.config.container_image,
                port=8000,
                cpu_limit=self.config.container_cpu,
                memory_limit=self.config.container_memory,
                replicas=self.config.min_instances,
                autoscaling_enabled=True,
                min_replicas=self.config.min_instances,
                max_replicas=self.config.max_instances,
            )
            stack["application"] = self.deploy_application(app_config)
            
            # 4. Set up monitoring
            if self.config.enable_monitoring:
                self.monitoring.create_standard_alerts(
                    stack["application"].resource_id
                )
            
            return stack
            
        except Exception as e:
            # Cleanup on failure
            for resource in stack.values():
                try:
                    self.delete_resource(resource.resource_id)
                except Exception:
                    pass
            raise ResourceCreationError("full_stack", str(e))