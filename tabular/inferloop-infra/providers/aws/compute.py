"""
AWS compute resource implementations
"""

import asyncio
import base64
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from botocore.exceptions import ClientError

from ...common.abstractions.compute import (
    BaseCompute, BaseContainer, BaseServerless,
    ComputeConfig, ContainerConfig, ServerlessConfig,
    ComputeResource, ComputeType
)
from ...common.abstractions.base import ResourceState


logger = logging.getLogger(__name__)


class AWSEC2(BaseCompute):
    """AWS EC2 implementation"""
    
    def __init__(self, provider):
        super().__init__("aws")
        self.provider = provider
        self.ec2_client = provider._get_client('ec2')
        self.ec2_resource = provider._get_resource('ec2')
    
    async def create(self, config: ComputeConfig) -> ComputeResource:
        """Create EC2 instance"""
        try:
            # Prepare instance parameters
            params = {
                'ImageId': self._get_latest_ami(),
                'InstanceType': self._get_instance_type(config.cpu, config.memory),
                'MinCount': 1,
                'MaxCount': 1,
                'KeyName': config.ssh_key,
                'SecurityGroupIds': config.security_groups,
                'TagSpecifications': [{
                    'ResourceType': 'instance',
                    'Tags': [
                        {'Key': 'Name', 'Value': config.name},
                        *[{'Key': k, 'Value': v} for k, v in config.tags.items()]
                    ]
                }]
            }
            
            # Add user data if provided
            if config.user_data:
                params['UserData'] = base64.b64encode(config.user_data.encode()).decode()
            
            # Create instance
            response = self.ec2_client.run_instances(**params)
            instance = response['Instances'][0]
            instance_id = instance['InstanceId']
            
            # Wait for instance to be running
            waiter = self.ec2_client.get_waiter('instance_running')
            waiter.wait(InstanceIds=[instance_id])
            
            # Get updated instance info
            instance_data = self.ec2_client.describe_instances(InstanceIds=[instance_id])
            instance = instance_data['Reservations'][0]['Instances'][0]
            
            return ComputeResource(
                id=instance_id,
                name=config.name,
                type=ComputeType.VM,
                state=self._map_instance_state(instance['State']['Name']),
                config=config,
                ip_addresses={
                    'private': instance.get('PrivateIpAddress'),
                    'public': instance.get('PublicIpAddress')
                },
                dns_name=instance.get('PublicDnsName'),
                metadata={
                    'instance_type': instance['InstanceType'],
                    'availability_zone': instance['Placement']['AvailabilityZone'],
                    'launch_time': instance['LaunchTime'].isoformat()
                }
            )
            
        except ClientError as e:
            logger.error(f"Failed to create EC2 instance: {str(e)}")
            raise
    
    async def delete(self, resource_id: str) -> bool:
        """Terminate EC2 instance"""
        try:
            self.ec2_client.terminate_instances(InstanceIds=[resource_id])
            
            # Wait for termination
            waiter = self.ec2_client.get_waiter('instance_terminated')
            waiter.wait(InstanceIds=[resource_id])
            
            return True
            
        except ClientError as e:
            logger.error(f"Failed to terminate instance {resource_id}: {str(e)}")
            return False
    
    async def get(self, resource_id: str) -> Optional[ComputeResource]:
        """Get EC2 instance details"""
        try:
            response = self.ec2_client.describe_instances(InstanceIds=[resource_id])
            
            if not response['Reservations']:
                return None
            
            instance = response['Reservations'][0]['Instances'][0]
            
            # Get name from tags
            name = resource_id
            for tag in instance.get('Tags', []):
                if tag['Key'] == 'Name':
                    name = tag['Value']
                    break
            
            return ComputeResource(
                id=resource_id,
                name=name,
                type=ComputeType.VM,
                state=self._map_instance_state(instance['State']['Name']),
                config=None,  # Would need to reconstruct from instance data
                ip_addresses={
                    'private': instance.get('PrivateIpAddress'),
                    'public': instance.get('PublicIpAddress')
                },
                dns_name=instance.get('PublicDnsName'),
                metadata={
                    'instance_type': instance['InstanceType'],
                    'availability_zone': instance['Placement']['AvailabilityZone'],
                    'launch_time': instance['LaunchTime'].isoformat()
                }
            )
            
        except ClientError as e:
            logger.error(f"Failed to get instance {resource_id}: {str(e)}")
            return None
    
    async def list(self, filters: Optional[Dict[str, Any]] = None) -> List[ComputeResource]:
        """List EC2 instances"""
        try:
            params = {}
            if filters:
                params['Filters'] = [
                    {'Name': k, 'Values': [v] if isinstance(v, str) else v}
                    for k, v in filters.items()
                ]
            
            resources = []
            paginator = self.ec2_client.get_paginator('describe_instances')
            
            for page in paginator.paginate(**params):
                for reservation in page['Reservations']:
                    for instance in reservation['Instances']:
                        resource = await self.get(instance['InstanceId'])
                        if resource:
                            resources.append(resource)
            
            return resources
            
        except ClientError as e:
            logger.error(f"Failed to list instances: {str(e)}")
            return []
    
    async def update(self, resource_id: str, config: ComputeConfig) -> ComputeResource:
        """Update EC2 instance"""
        # EC2 instances can't be updated in place, would need to stop/modify/start
        # or replace the instance
        raise NotImplementedError("EC2 instance updates not implemented")
    
    async def get_state(self, resource_id: str) -> ResourceState:
        """Get instance state"""
        try:
            response = self.ec2_client.describe_instances(InstanceIds=[resource_id])
            instance = response['Reservations'][0]['Instances'][0]
            return self._map_instance_state(instance['State']['Name'])
        except:
            return ResourceState.UNKNOWN
    
    async def wait_for_state(self, resource_id: str, target_state: ResourceState, timeout: int = 300) -> bool:
        """Wait for instance to reach target state"""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < timeout:
            current_state = await self.get_state(resource_id)
            if current_state == target_state:
                return True
            await asyncio.sleep(5)
        
        return False
    
    def estimate_cost(self, config: ComputeConfig) -> Dict[str, float]:
        """Estimate EC2 costs"""
        # Simplified cost estimation
        instance_type = self._get_instance_type(config.cpu, config.memory)
        
        # These are example prices, real implementation would use AWS Pricing API
        hourly_prices = {
            't3.micro': 0.0104,
            't3.small': 0.0208,
            't3.medium': 0.0416,
            't3.large': 0.0832,
            'm5.large': 0.096,
            'm5.xlarge': 0.192
        }
        
        hourly_cost = hourly_prices.get(instance_type, 0.1)
        
        return {
            'hourly': hourly_cost,
            'daily': hourly_cost * 24,
            'monthly': hourly_cost * 24 * 30
        }
    
    async def start(self, resource_id: str) -> bool:
        """Start stopped instance"""
        try:
            self.ec2_client.start_instances(InstanceIds=[resource_id])
            waiter = self.ec2_client.get_waiter('instance_running')
            waiter.wait(InstanceIds=[resource_id])
            return True
        except ClientError as e:
            logger.error(f"Failed to start instance {resource_id}: {str(e)}")
            return False
    
    async def stop(self, resource_id: str) -> bool:
        """Stop running instance"""
        try:
            self.ec2_client.stop_instances(InstanceIds=[resource_id])
            waiter = self.ec2_client.get_waiter('instance_stopped')
            waiter.wait(InstanceIds=[resource_id])
            return True
        except ClientError as e:
            logger.error(f"Failed to stop instance {resource_id}: {str(e)}")
            return False
    
    async def restart(self, resource_id: str) -> bool:
        """Restart instance"""
        try:
            self.ec2_client.reboot_instances(InstanceIds=[resource_id])
            return True
        except ClientError as e:
            logger.error(f"Failed to restart instance {resource_id}: {str(e)}")
            return False
    
    async def resize(self, resource_id: str, cpu: int, memory: int) -> bool:
        """Resize instance"""
        try:
            # Stop instance first
            await self.stop(resource_id)
            
            # Modify instance type
            new_type = self._get_instance_type(cpu, memory)
            self.ec2_client.modify_instance_attribute(
                InstanceId=resource_id,
                InstanceType={'Value': new_type}
            )
            
            # Start instance
            await self.start(resource_id)
            
            return True
            
        except ClientError as e:
            logger.error(f"Failed to resize instance {resource_id}: {str(e)}")
            return False
    
    async def get_console_output(self, resource_id: str) -> str:
        """Get console output"""
        try:
            response = self.ec2_client.get_console_output(InstanceId=resource_id)
            return response.get('Output', '')
        except ClientError as e:
            logger.error(f"Failed to get console output for {resource_id}: {str(e)}")
            return ""
    
    async def execute_command(self, resource_id: str, command: str) -> Dict[str, Any]:
        """Execute command via SSM"""
        try:
            ssm = self.provider._get_client('ssm')
            
            response = ssm.send_command(
                InstanceIds=[resource_id],
                DocumentName='AWS-RunShellScript',
                Parameters={'commands': [command]}
            )
            
            command_id = response['Command']['CommandId']
            
            # Wait for command to complete
            waiter = ssm.get_waiter('command_executed')
            waiter.wait(CommandId=command_id, InstanceId=resource_id)
            
            # Get command output
            output = ssm.get_command_invocation(
                CommandId=command_id,
                InstanceId=resource_id
            )
            
            return {
                'exit_code': output['ResponseCode'],
                'stdout': output['StandardOutputContent'],
                'stderr': output['StandardErrorContent']
            }
            
        except ClientError as e:
            logger.error(f"Failed to execute command on {resource_id}: {str(e)}")
            raise
    
    def _get_latest_ami(self) -> str:
        """Get latest Amazon Linux 2 AMI"""
        try:
            response = self.ec2_client.describe_images(
                Owners=['amazon'],
                Filters=[
                    {'Name': 'name', 'Values': ['amzn2-ami-hvm-*-x86_64-gp2']},
                    {'Name': 'state', 'Values': ['available']}
                ],
                MaxResults=1
            )
            
            if response['Images']:
                return sorted(response['Images'], 
                            key=lambda x: x['CreationDate'], 
                            reverse=True)[0]['ImageId']
            
            # Fallback AMI
            return 'ami-0c02fb55956c7d316'  # us-east-1 Amazon Linux 2
            
        except:
            return 'ami-0c02fb55956c7d316'
    
    def _get_instance_type(self, cpu: int, memory: int) -> str:
        """Map CPU/memory to instance type"""
        # Simplified mapping
        if cpu <= 1 and memory <= 1024:
            return 't3.micro'
        elif cpu <= 2 and memory <= 2048:
            return 't3.small'
        elif cpu <= 2 and memory <= 4096:
            return 't3.medium'
        elif cpu <= 2 and memory <= 8192:
            return 't3.large'
        elif cpu <= 4:
            return 'm5.xlarge'
        else:
            return 'm5.2xlarge'
    
    def _map_instance_state(self, aws_state: str) -> ResourceState:
        """Map AWS instance state to ResourceState"""
        mapping = {
            'pending': ResourceState.CREATING,
            'running': ResourceState.RUNNING,
            'stopping': ResourceState.STOPPING,
            'stopped': ResourceState.STOPPED,
            'shutting-down': ResourceState.DELETING,
            'terminated': ResourceState.DELETED
        }
        return mapping.get(aws_state, ResourceState.UNKNOWN)


class AWSContainer(BaseContainer):
    """AWS ECS/Fargate container implementation"""
    # Implementation would go here
    pass


class AWSLambda(BaseServerless):
    """AWS Lambda serverless implementation"""
    # Implementation would go here
    pass