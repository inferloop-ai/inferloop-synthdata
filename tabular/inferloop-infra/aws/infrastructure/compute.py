"""AWS compute resources implementation."""

import boto3
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import base64

from common.core.base_provider import (
    ResourceInfo,
    ResourceStatus,
    ComputeConfig,
    ApplicationConfig,
)
from common.core.config import InfrastructureConfig
from common.core.exceptions import (
    ResourceCreationError,
    ResourceNotFoundError,
    DeploymentError,
)


class AWSCompute:
    """AWS compute resources management."""
    
    def __init__(self, session: boto3.Session, config: InfrastructureConfig):
        """Initialize AWS compute manager."""
        self.session = session
        self.config = config
        self.ec2_client = session.client("ec2")
        self.ec2_resource = session.resource("ec2")
        self.ecs_client = session.client("ecs")
        self.ecr_client = session.client("ecr")
        self.autoscaling_client = session.client("autoscaling")
        self.elbv2_client = session.client("elbv2")
        self.batch_client = session.client("batch")
        self.lambda_client = session.client("lambda")
    
    def create_instance(self, config: ComputeConfig) -> ResourceInfo:
        """Create an EC2 instance."""
        try:
            # Get AMI ID (Amazon Linux 2)
            ami_id = self._get_latest_ami()
            
            # Prepare user data script
            user_data = config.user_data or self._get_default_user_data()
            
            # Create instance
            response = self.ec2_client.run_instances(
                ImageId=ami_id,
                InstanceType=config.metadata.get("instance_type", "t3.medium"),
                MinCount=1,
                MaxCount=1,
                KeyName=config.ssh_key_name,
                UserData=user_data,
                BlockDeviceMappings=[
                    {
                        "DeviceName": "/dev/xvda",
                        "Ebs": {
                            "VolumeSize": config.disk_size_gb,
                            "VolumeType": "gp3",
                            "DeleteOnTermination": True,
                            "Encrypted": self.config.storage_encryption,
                        },
                    }
                ],
                NetworkInterfaces=[
                    {
                        "AssociatePublicIpAddress": True,
                        "DeviceIndex": 0,
                        "Groups": [self._get_or_create_security_group()],
                        "SubnetId": self._get_default_subnet(),
                    }
                ],
                TagSpecifications=[
                    {
                        "ResourceType": "instance",
                        "Tags": self._format_tags({**config.tags, "Name": config.name}),
                    }
                ],
            )
            
            instance = response["Instances"][0]
            instance_id = instance["InstanceId"]
            
            # Wait for instance to be running
            waiter = self.ec2_client.get_waiter("instance_running")
            waiter.wait(InstanceIds=[instance_id])
            
            # Get updated instance info
            instance = self.ec2_resource.Instance(instance_id)
            
            return ResourceInfo(
                resource_id=instance_id,
                resource_type="ec2_instance",
                name=config.name,
                status=ResourceStatus.RUNNING,
                region=self.config.region,
                created_at=instance.launch_time,
                endpoint=instance.public_ip_address,
                metadata={
                    "instance_type": instance.instance_type,
                    "private_ip": instance.private_ip_address,
                    "vpc_id": instance.vpc_id,
                    "subnet_id": instance.subnet_id,
                },
            )
            
        except Exception as e:
            raise ResourceCreationError("EC2 instance", str(e))
    
    def deploy_fargate_service(self, app_config: ApplicationConfig) -> ResourceInfo:
        """Deploy application using ECS Fargate."""
        try:
            # Ensure ECS cluster exists
            cluster_name = self._ensure_ecs_cluster()
            
            # Create task definition
            task_def_arn = self._create_task_definition(app_config)
            
            # Create ALB target group
            target_group_arn = self._create_target_group(app_config.name)
            
            # Create ECS service
            service_response = self.ecs_client.create_service(
                cluster=cluster_name,
                serviceName=app_config.name,
                taskDefinition=task_def_arn,
                desiredCount=app_config.replicas,
                launchType="FARGATE",
                networkConfiguration={
                    "awsvpcConfiguration": {
                        "subnets": self._get_private_subnets(),
                        "securityGroups": [self._get_or_create_ecs_security_group()],
                        "assignPublicIp": "ENABLED",
                    }
                },
                loadBalancers=[
                    {
                        "targetGroupArn": target_group_arn,
                        "containerName": "app",
                        "containerPort": app_config.port,
                    }
                ],
                healthCheckGracePeriodSeconds=60,
                deploymentConfiguration={
                    "maximumPercent": 200,
                    "minimumHealthyPercent": 100,
                },
                tags=self._format_tags({"Name": app_config.name}),
            )
            
            service = service_response["service"]
            
            # Set up auto-scaling if enabled
            if app_config.autoscaling_enabled:
                self._setup_service_autoscaling(
                    cluster_name,
                    app_config.name,
                    app_config.min_replicas,
                    app_config.max_replicas,
                    app_config.target_cpu_utilization,
                )
            
            # Get ALB endpoint
            alb_dns = self._get_alb_dns(target_group_arn)
            
            return ResourceInfo(
                resource_id=f"{cluster_name}/service/{app_config.name}",
                resource_type="ecs_service",
                name=app_config.name,
                status=ResourceStatus.RUNNING,
                region=self.config.region,
                created_at=service["createdAt"],
                endpoint=alb_dns,
                metadata={
                    "cluster": cluster_name,
                    "task_definition": task_def_arn,
                    "desired_count": app_config.replicas,
                    "launch_type": "FARGATE",
                },
            )
            
        except Exception as e:
            raise DeploymentError(app_config.name, str(e))
    
    def create_batch_job(self, job_config: Dict[str, Any]) -> ResourceInfo:
        """Create AWS Batch job for large-scale processing."""
        try:
            # Ensure Batch compute environment exists
            compute_env = self._ensure_batch_compute_environment()
            
            # Ensure job queue exists
            job_queue = self._ensure_batch_job_queue(compute_env)
            
            # Register job definition
            job_def_name = f"{self.config.resource_name}-synthdata-job"
            job_def_response = self.batch_client.register_job_definition(
                jobDefinitionName=job_def_name,
                type="container",
                containerProperties={
                    "image": self.config.container_image,
                    "vcpus": int(job_config.get("vcpus", 4)),
                    "memory": int(job_config.get("memory_mb", 8192)),
                    "environment": [
                        {"name": k, "value": v}
                        for k, v in job_config.get("environment", {}).items()
                    ],
                    "jobRoleArn": self._get_batch_job_role(),
                },
                tags=self.config.default_tags,
            )
            
            # Submit job
            job_response = self.batch_client.submit_job(
                jobName=job_config["name"],
                jobQueue=job_queue,
                jobDefinition=job_def_name,
                parameters=job_config.get("parameters", {}),
                tags=self.config.default_tags,
            )
            
            return ResourceInfo(
                resource_id=job_response["jobId"],
                resource_type="batch_job",
                name=job_config["name"],
                status=ResourceStatus.CREATING,
                region=self.config.region,
                created_at=datetime.utcnow(),
                metadata={
                    "job_queue": job_queue,
                    "job_definition": job_def_name,
                    "compute_environment": compute_env,
                },
            )
            
        except Exception as e:
            raise ResourceCreationError("Batch job", str(e))
    
    def create_lambda_function(self, function_config: Dict[str, Any]) -> ResourceInfo:
        """Create Lambda function for lightweight operations."""
        try:
            function_name = f"{self.config.resource_name}-{function_config['name']}"
            
            # Create Lambda function
            response = self.lambda_client.create_function(
                FunctionName=function_name,
                Runtime=function_config.get("runtime", "python3.9"),
                Role=self._get_lambda_role(),
                Handler=function_config.get("handler", "index.handler"),
                Code=function_config["code"],  # Should be {"ZipFile": bytes} or {"S3Bucket": ..., "S3Key": ...}
                Description=function_config.get("description", "Synthetic data processing function"),
                Timeout=function_config.get("timeout", 300),
                MemorySize=function_config.get("memory_size", 512),
                Environment={
                    "Variables": function_config.get("environment", {})
                },
                Tags=self.config.default_tags,
            )
            
            return ResourceInfo(
                resource_id=response["FunctionArn"],
                resource_type="lambda_function",
                name=function_name,
                status=ResourceStatus.RUNNING,
                region=self.config.region,
                created_at=datetime.utcnow(),
                endpoint=response["FunctionArn"],
                metadata={
                    "runtime": response["Runtime"],
                    "memory_size": response["MemorySize"],
                    "timeout": response["Timeout"],
                },
            )
            
        except Exception as e:
            raise ResourceCreationError("Lambda function", str(e))
    
    def get_instance(self, instance_id: str) -> ResourceInfo:
        """Get EC2 instance information."""
        try:
            instance = self.ec2_resource.Instance(instance_id)
            instance.load()
            
            status_map = {
                "pending": ResourceStatus.CREATING,
                "running": ResourceStatus.RUNNING,
                "stopping": ResourceStatus.DELETING,
                "stopped": ResourceStatus.STOPPED,
                "shutting-down": ResourceStatus.DELETING,
                "terminated": ResourceStatus.DELETED,
            }
            
            return ResourceInfo(
                resource_id=instance_id,
                resource_type="ec2_instance",
                name=self._get_tag_value(instance.tags, "Name"),
                status=status_map.get(instance.state["Name"], ResourceStatus.ERROR),
                region=self.config.region,
                created_at=instance.launch_time,
                endpoint=instance.public_ip_address,
                metadata={
                    "instance_type": instance.instance_type,
                    "private_ip": instance.private_ip_address,
                    "vpc_id": instance.vpc_id,
                    "subnet_id": instance.subnet_id,
                },
            )
        except Exception as e:
            raise ResourceNotFoundError(instance_id, "EC2 instance")
    
    def get_service(self, service_id: str) -> ResourceInfo:
        """Get ECS service information."""
        try:
            cluster, service_name = service_id.split("/service/")
            
            response = self.ecs_client.describe_services(
                cluster=cluster,
                services=[service_name],
            )
            
            if not response["services"]:
                raise ResourceNotFoundError(service_id, "ECS service")
            
            service = response["services"][0]
            
            status_map = {
                "ACTIVE": ResourceStatus.RUNNING,
                "DRAINING": ResourceStatus.DELETING,
                "INACTIVE": ResourceStatus.STOPPED,
            }
            
            # Get ALB endpoint
            alb_dns = None
            if service.get("loadBalancers"):
                target_group_arn = service["loadBalancers"][0]["targetGroupArn"]
                alb_dns = self._get_alb_dns(target_group_arn)
            
            return ResourceInfo(
                resource_id=service_id,
                resource_type="ecs_service",
                name=service["serviceName"],
                status=status_map.get(service["status"], ResourceStatus.ERROR),
                region=self.config.region,
                created_at=service["createdAt"],
                endpoint=alb_dns,
                metadata={
                    "cluster": cluster,
                    "task_definition": service["taskDefinition"],
                    "desired_count": service["desiredCount"],
                    "running_count": service["runningCount"],
                    "launch_type": service.get("launchType", "FARGATE"),
                },
            )
        except Exception as e:
            raise ResourceNotFoundError(service_id, "ECS service")
    
    def list_instances(self) -> List[ResourceInfo]:
        """List all EC2 instances."""
        instances = []
        
        try:
            response = self.ec2_client.describe_instances(
                Filters=[
                    {
                        "Name": "tag:Project",
                        "Values": [self.config.project_name],
                    },
                    {
                        "Name": "instance-state-name",
                        "Values": ["pending", "running", "stopping", "stopped"],
                    },
                ]
            )
            
            for reservation in response["Reservations"]:
                for instance in reservation["Instances"]:
                    instances.append(self.get_instance(instance["InstanceId"]))
            
        except Exception:
            pass
        
        return instances
    
    def list_services(self) -> List[ResourceInfo]:
        """List all ECS services."""
        services = []
        
        try:
            # List clusters
            clusters_response = self.ecs_client.list_clusters()
            
            for cluster_arn in clusters_response["clusterArns"]:
                cluster_name = cluster_arn.split("/")[-1]
                
                # Skip if not our cluster
                if self.config.resource_name not in cluster_name:
                    continue
                
                # List services in cluster
                services_response = self.ecs_client.list_services(
                    cluster=cluster_name,
                    maxResults=100,
                )
                
                for service_arn in services_response["serviceArns"]:
                    service_name = service_arn.split("/")[-1]
                    service_id = f"{cluster_name}/service/{service_name}"
                    try:
                        services.append(self.get_service(service_id))
                    except Exception:
                        pass
        
        except Exception:
            pass
        
        return services
    
    def delete_instance(self, instance_id: str) -> bool:
        """Delete an EC2 instance."""
        try:
            self.ec2_client.terminate_instances(InstanceIds=[instance_id])
            return True
        except Exception:
            return False
    
    def delete_service(self, service_id: str) -> bool:
        """Delete an ECS service."""
        try:
            cluster, service_name = service_id.split("/service/")
            
            # Update service to 0 desired count
            self.ecs_client.update_service(
                cluster=cluster,
                service=service_name,
                desiredCount=0,
            )
            
            # Delete service
            self.ecs_client.delete_service(
                cluster=cluster,
                service=service_name,
            )
            
            return True
        except Exception:
            return False
    
    def update_instance(self, instance_id: str, updates: Dict[str, Any]) -> ResourceInfo:
        """Update EC2 instance configuration."""
        try:
            if "instance_type" in updates:
                # Stop instance if running
                instance = self.ec2_resource.Instance(instance_id)
                if instance.state["Name"] == "running":
                    instance.stop()
                    instance.wait_until_stopped()
                
                # Modify instance type
                self.ec2_client.modify_instance_attribute(
                    InstanceId=instance_id,
                    InstanceType={"Value": updates["instance_type"]},
                )
                
                # Start instance
                instance.start()
                instance.wait_until_running()
            
            if "tags" in updates:
                self.ec2_client.create_tags(
                    Resources=[instance_id],
                    Tags=self._format_tags(updates["tags"]),
                )
            
            return self.get_instance(instance_id)
            
        except Exception as e:
            raise ResourceCreationError("EC2 instance update", str(e))
    
    def update_service(self, service_id: str, updates: Dict[str, Any]) -> ResourceInfo:
        """Update ECS service configuration."""
        try:
            cluster, service_name = service_id.split("/service/")
            
            update_params = {
                "cluster": cluster,
                "service": service_name,
            }
            
            if "desired_count" in updates:
                update_params["desiredCount"] = updates["desired_count"]
            
            if "task_definition" in updates:
                update_params["taskDefinition"] = updates["task_definition"]
            
            self.ecs_client.update_service(**update_params)
            
            return self.get_service(service_id)
            
        except Exception as e:
            raise ResourceCreationError("ECS service update", str(e))
    
    def _get_latest_ami(self) -> str:
        """Get latest Amazon Linux 2 AMI."""
        response = self.ec2_client.describe_images(
            Owners=["amazon"],
            Filters=[
                {"Name": "name", "Values": ["amzn2-ami-hvm-*-x86_64-gp2"]},
                {"Name": "state", "Values": ["available"]},
            ],
        )
        
        images = sorted(response["Images"], key=lambda x: x["CreationDate"], reverse=True)
        return images[0]["ImageId"]
    
    def _get_default_user_data(self) -> str:
        """Get default EC2 user data script."""
        script = """#!/bin/bash
# Update system
yum update -y

# Install Docker
amazon-linux-extras install docker -y
service docker start
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install Python 3.9
amazon-linux-extras install python3.9 -y

# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/amazon_linux/amd64/latest/amazon-cloudwatch-agent.rpm
rpm -U ./amazon-cloudwatch-agent.rpm

# Pull synthetic data generator image
docker pull {image}
""".format(image=self.config.container_image)
        
        return script
    
    def _get_or_create_security_group(self) -> str:
        """Get or create EC2 security group."""
        group_name = f"{self.config.resource_name}-ec2-sg"
        
        try:
            response = self.ec2_client.describe_security_groups(
                GroupNames=[group_name]
            )
            return response["SecurityGroups"][0]["GroupId"]
        except:
            # Create security group
            response = self.ec2_client.create_security_group(
                GroupName=group_name,
                Description=f"Security group for {self.config.project_name} EC2 instances",
                VpcId=self._get_default_vpc(),
                TagSpecifications=[
                    {
                        "ResourceType": "security-group",
                        "Tags": self._format_tags({"Name": group_name}),
                    }
                ],
            )
            
            group_id = response["GroupId"]
            
            # Add ingress rules
            self.ec2_client.authorize_security_group_ingress(
                GroupId=group_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 22,
                        "ToPort": 22,
                        "IpRanges": [{"CidrIp": ip} for ip in self.config.allowed_ip_ranges],
                    },
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 80,
                        "ToPort": 80,
                        "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                    },
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 443,
                        "ToPort": 443,
                        "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                    },
                ],
            )
            
            return group_id
    
    def _get_or_create_ecs_security_group(self) -> str:
        """Get or create ECS security group."""
        group_name = f"{self.config.resource_name}-ecs-sg"
        
        try:
            response = self.ec2_client.describe_security_groups(
                GroupNames=[group_name]
            )
            return response["SecurityGroups"][0]["GroupId"]
        except:
            # Create security group
            response = self.ec2_client.create_security_group(
                GroupName=group_name,
                Description=f"Security group for {self.config.project_name} ECS tasks",
                VpcId=self._get_default_vpc(),
                TagSpecifications=[
                    {
                        "ResourceType": "security-group",
                        "Tags": self._format_tags({"Name": group_name}),
                    }
                ],
            )
            
            group_id = response["GroupId"]
            
            # Add ingress rules
            self.ec2_client.authorize_security_group_ingress(
                GroupId=group_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 8000,
                        "ToPort": 8000,
                        "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                    },
                ],
            )
            
            return group_id
    
    def _get_default_vpc(self) -> str:
        """Get default VPC ID."""
        response = self.ec2_client.describe_vpcs(
            Filters=[{"Name": "is-default", "Values": ["true"]}]
        )
        return response["Vpcs"][0]["VpcId"]
    
    def _get_default_subnet(self) -> str:
        """Get default subnet ID."""
        vpc_id = self._get_default_vpc()
        response = self.ec2_client.describe_subnets(
            Filters=[
                {"Name": "vpc-id", "Values": [vpc_id]},
                {"Name": "default-for-az", "Values": ["true"]},
            ]
        )
        return response["Subnets"][0]["SubnetId"]
    
    def _get_private_subnets(self) -> List[str]:
        """Get private subnet IDs."""
        # For now, use default subnets
        # In production, this should return actual private subnets
        vpc_id = self._get_default_vpc()
        response = self.ec2_client.describe_subnets(
            Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
        )
        return [subnet["SubnetId"] for subnet in response["Subnets"][:2]]
    
    def _ensure_ecs_cluster(self) -> str:
        """Ensure ECS cluster exists."""
        cluster_name = f"{self.config.resource_name}-cluster"
        
        try:
            self.ecs_client.describe_clusters(clusters=[cluster_name])
        except:
            self.ecs_client.create_cluster(
                clusterName=cluster_name,
                tags=self._format_tags({"Name": cluster_name}),
                capacityProviders=["FARGATE", "FARGATE_SPOT"],
            )
        
        return cluster_name
    
    def _create_task_definition(self, app_config: ApplicationConfig) -> str:
        """Create ECS task definition."""
        family = f"{self.config.resource_name}-task"
        
        # Parse CPU and memory values
        cpu_value = str(int(float(app_config.cpu_limit) * 1024))
        memory_value = str(int(app_config.memory_limit.rstrip("Gi")) * 1024)
        
        response = self.ecs_client.register_task_definition(
            family=family,
            networkMode="awsvpc",
            requiresCompatibilities=["FARGATE"],
            cpu=cpu_value,
            memory=memory_value,
            taskRoleArn=self._get_task_role(),
            executionRoleArn=self._get_execution_role(),
            containerDefinitions=[
                {
                    "name": "app",
                    "image": app_config.image,
                    "portMappings": [
                        {
                            "containerPort": app_config.port,
                            "protocol": "tcp",
                        }
                    ],
                    "environment": [
                        {"name": k, "value": v}
                        for k, v in app_config.environment_variables.items()
                    ],
                    "secrets": [
                        {
                            "name": secret,
                            "valueFrom": f"arn:aws:secretsmanager:{self.config.region}:{{account_id}}:secret:{secret}",
                        }
                        for secret in app_config.secrets
                    ],
                    "logConfiguration": {
                        "logDriver": "awslogs",
                        "options": {
                            "awslogs-group": f"/ecs/{family}",
                            "awslogs-region": self.config.region,
                            "awslogs-stream-prefix": "ecs",
                        },
                    },
                    "healthCheck": {
                        "command": [
                            "CMD-SHELL",
                            f"curl -f http://localhost:{app_config.port}{app_config.health_check_path} || exit 1",
                        ],
                        "interval": 30,
                        "timeout": 5,
                        "retries": 3,
                        "startPeriod": 60,
                    },
                }
            ],
            tags=self._format_tags({"Name": family}),
        )
        
        return response["taskDefinition"]["taskDefinitionArn"]
    
    def _create_target_group(self, service_name: str) -> str:
        """Create ALB target group."""
        vpc_id = self._get_default_vpc()
        
        response = self.elbv2_client.create_target_group(
            Name=f"{service_name}-tg"[:32],  # Max 32 chars
            Protocol="HTTP",
            Port=8000,
            VpcId=vpc_id,
            TargetType="ip",
            HealthCheckProtocol="HTTP",
            HealthCheckPath="/health",
            HealthCheckIntervalSeconds=30,
            HealthCheckTimeoutSeconds=5,
            HealthyThresholdCount=2,
            UnhealthyThresholdCount=3,
            Tags=self._format_tags({"Name": f"{service_name}-tg"}),
        )
        
        target_group_arn = response["TargetGroups"][0]["TargetGroupArn"]
        
        # Get or create ALB
        alb_arn = self._get_or_create_alb()
        
        # Create listener if needed
        self._ensure_alb_listener(alb_arn, target_group_arn)
        
        return target_group_arn
    
    def _get_or_create_alb(self) -> str:
        """Get or create Application Load Balancer."""
        alb_name = f"{self.config.resource_name}-alb"[:32]
        
        try:
            response = self.elbv2_client.describe_load_balancers(Names=[alb_name])
            return response["LoadBalancers"][0]["LoadBalancerArn"]
        except:
            # Create ALB
            response = self.elbv2_client.create_load_balancer(
                Name=alb_name,
                Subnets=self._get_public_subnets(),
                SecurityGroups=[self._get_or_create_alb_security_group()],
                Scheme="internet-facing",
                Type="application",
                IpAddressType="ipv4",
                Tags=self._format_tags({"Name": alb_name}),
            )
            
            return response["LoadBalancers"][0]["LoadBalancerArn"]
    
    def _get_public_subnets(self) -> List[str]:
        """Get public subnet IDs."""
        vpc_id = self._get_default_vpc()
        response = self.ec2_client.describe_subnets(
            Filters=[
                {"Name": "vpc-id", "Values": [vpc_id]},
                {"Name": "map-public-ip-on-launch", "Values": ["true"]},
            ]
        )
        
        # Return at least 2 subnets in different AZs
        subnets = response["Subnets"]
        if len(subnets) < 2:
            # Use all available subnets
            response = self.ec2_client.describe_subnets(
                Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
            )
            subnets = response["Subnets"]
        
        return [subnet["SubnetId"] for subnet in subnets[:2]]
    
    def _get_or_create_alb_security_group(self) -> str:
        """Get or create ALB security group."""
        group_name = f"{self.config.resource_name}-alb-sg"
        
        try:
            response = self.ec2_client.describe_security_groups(
                GroupNames=[group_name]
            )
            return response["SecurityGroups"][0]["GroupId"]
        except:
            # Create security group
            response = self.ec2_client.create_security_group(
                GroupName=group_name,
                Description=f"Security group for {self.config.project_name} ALB",
                VpcId=self._get_default_vpc(),
                TagSpecifications=[
                    {
                        "ResourceType": "security-group",
                        "Tags": self._format_tags({"Name": group_name}),
                    }
                ],
            )
            
            group_id = response["GroupId"]
            
            # Add ingress rules
            self.ec2_client.authorize_security_group_ingress(
                GroupId=group_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 80,
                        "ToPort": 80,
                        "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                    },
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 443,
                        "ToPort": 443,
                        "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                    },
                ],
            )
            
            return group_id
    
    def _ensure_alb_listener(self, alb_arn: str, target_group_arn: str) -> None:
        """Ensure ALB has a listener configured."""
        try:
            # Check if listener exists
            response = self.elbv2_client.describe_listeners(LoadBalancerArn=alb_arn)
            
            if not response["Listeners"]:
                # Create listener
                self.elbv2_client.create_listener(
                    LoadBalancerArn=alb_arn,
                    Protocol="HTTP",
                    Port=80,
                    DefaultActions=[
                        {
                            "Type": "forward",
                            "TargetGroupArn": target_group_arn,
                        }
                    ],
                )
        except:
            pass
    
    def _get_alb_dns(self, target_group_arn: str) -> Optional[str]:
        """Get ALB DNS name from target group."""
        try:
            # Get load balancer ARNs for target group
            response = self.elbv2_client.describe_target_groups(
                TargetGroupArns=[target_group_arn]
            )
            
            if response["TargetGroups"]:
                lb_arns = response["TargetGroups"][0].get("LoadBalancerArns", [])
                if lb_arns:
                    # Get load balancer details
                    lb_response = self.elbv2_client.describe_load_balancers(
                        LoadBalancerArns=[lb_arns[0]]
                    )
                    if lb_response["LoadBalancers"]:
                        return lb_response["LoadBalancers"][0]["DNSName"]
        except:
            pass
        
        return None
    
    def _setup_service_autoscaling(
        self,
        cluster_name: str,
        service_name: str,
        min_count: int,
        max_count: int,
        target_cpu: int,
    ) -> None:
        """Set up ECS service auto-scaling."""
        resource_id = f"service/{cluster_name}/{service_name}"
        
        # Register scalable target
        self.autoscaling_client.register_scalable_target(
            ServiceNamespace="ecs",
            ResourceId=resource_id,
            ScalableDimension="ecs:service:DesiredCount",
            MinCapacity=min_count,
            MaxCapacity=max_count,
        )
        
        # Create scaling policy
        self.autoscaling_client.put_scaling_policy(
            PolicyName=f"{service_name}-cpu-scaling",
            ServiceNamespace="ecs",
            ResourceId=resource_id,
            ScalableDimension="ecs:service:DesiredCount",
            PolicyType="TargetTrackingScaling",
            TargetTrackingScalingPolicyConfiguration={
                "TargetValue": float(target_cpu),
                "PredefinedMetricSpecification": {
                    "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
                },
                "ScaleOutCooldown": 60,
                "ScaleInCooldown": 300,
            },
        )
    
    def _ensure_batch_compute_environment(self) -> str:
        """Ensure AWS Batch compute environment exists."""
        compute_env_name = f"{self.config.resource_name}-batch-compute"
        
        try:
            response = self.batch_client.describe_compute_environments(
                computeEnvironments=[compute_env_name]
            )
            if response["computeEnvironments"]:
                return compute_env_name
        except:
            pass
        
        # Create compute environment
        self.batch_client.create_compute_environment(
            computeEnvironmentName=compute_env_name,
            type="MANAGED",
            state="ENABLED",
            computeResources={
                "type": "EC2_SPOT" if self.config.enable_spot_instances else "EC2",
                "minvCpus": 0,
                "maxvCpus": 256,
                "desiredvCpus": 4,
                "instanceTypes": ["optimal"],
                "subnets": self._get_private_subnets(),
                "securityGroupIds": [self._get_or_create_security_group()],
                "instanceRole": self._get_batch_instance_profile(),
                "tags": self.config.default_tags,
            },
            serviceRole=self._get_batch_service_role(),
            tags=self.config.default_tags,
        )
        
        return compute_env_name
    
    def _ensure_batch_job_queue(self, compute_env_name: str) -> str:
        """Ensure AWS Batch job queue exists."""
        job_queue_name = f"{self.config.resource_name}-batch-queue"
        
        try:
            response = self.batch_client.describe_job_queues(
                jobQueues=[job_queue_name]
            )
            if response["jobQueues"]:
                return job_queue_name
        except:
            pass
        
        # Create job queue
        self.batch_client.create_job_queue(
            jobQueueName=job_queue_name,
            state="ENABLED",
            priority=1000,
            computeEnvironmentOrder=[
                {
                    "order": 1,
                    "computeEnvironment": compute_env_name,
                }
            ],
            tags=self.config.default_tags,
        )
        
        return job_queue_name
    
    def _get_task_role(self) -> str:
        """Get or create ECS task role ARN."""
        # Placeholder - should create/get actual role
        return f"arn:aws:iam::{{account_id}}:role/{self.config.resource_name}-ecs-task-role"
    
    def _get_execution_role(self) -> str:
        """Get or create ECS execution role ARN."""
        # Placeholder - should create/get actual role
        return f"arn:aws:iam::{{account_id}}:role/{self.config.resource_name}-ecs-execution-role"
    
    def _get_batch_job_role(self) -> str:
        """Get or create Batch job role ARN."""
        # Placeholder - should create/get actual role
        return f"arn:aws:iam::{{account_id}}:role/{self.config.resource_name}-batch-job-role"
    
    def _get_batch_instance_profile(self) -> str:
        """Get or create Batch instance profile ARN."""
        # Placeholder - should create/get actual instance profile
        return f"arn:aws:iam::{{account_id}}:instance-profile/{self.config.resource_name}-batch-instance-profile"
    
    def _get_batch_service_role(self) -> str:
        """Get or create Batch service role ARN."""
        # Placeholder - should create/get actual role
        return f"arn:aws:iam::{{account_id}}:role/{self.config.resource_name}-batch-service-role"
    
    def _get_lambda_role(self) -> str:
        """Get or create Lambda execution role ARN."""
        # Placeholder - should create/get actual role
        return f"arn:aws:iam::{{account_id}}:role/{self.config.resource_name}-lambda-role"
    
    def _format_tags(self, tags: Dict[str, str]) -> List[Dict[str, str]]:
        """Format tags for AWS API."""
        return [{"Key": k, "Value": v} for k, v in tags.items()]
    
    def _get_tag_value(self, tags: List[Dict[str, str]], key: str) -> Optional[str]:
        """Get tag value from AWS tags list."""
        if not tags:
            return None
        
        for tag in tags:
            if tag.get("Key") == key:
                return tag.get("Value")
        
        return None