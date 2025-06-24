"""AWS deployment provider implementation."""

from typing import Dict, Any, Optional, List
import boto3
import json
from pathlib import Path
import yaml

from ..base import BaseCloudProvider, ResourceConfig, DeploymentResult
from .services import AWSAdvancedServices


class AWSProvider(BaseCloudProvider):
    """AWS cloud provider implementation."""
    
    def __init__(self, project_id: str, region: str = "us-east-1", **kwargs):
        """Initialize AWS provider."""
        super().__init__(project_id, region, **kwargs)
        self.session = None
        self.clients = {}
        self.advanced_services = None
        
    def authenticate(self) -> bool:
        """Authenticate with AWS."""
        try:
            self.session = boto3.Session(region_name=self.region)
            # Test authentication
            sts = self.session.client('sts')
            sts.get_caller_identity()
            
            # Initialize advanced services
            self.advanced_services = AWSAdvancedServices(self.session, self.project_id, self.region)
            
            return True
        except Exception as e:
            self.logger.error(f"AWS authentication failed: {e}")
            return False
            
    def _get_client(self, service: str):
        """Get or create AWS client for service."""
        if service not in self.clients:
            self.clients[service] = self.session.client(service)
        return self.clients[service]
        
    def deploy_infrastructure(self, config: ResourceConfig) -> DeploymentResult:
        """Deploy infrastructure resources on AWS."""
        result = DeploymentResult()
        
        try:
            # Deploy VPC and networking
            vpc_result = self._deploy_vpc(config)
            result.resources.update(vpc_result)
            
            # Deploy storage
            storage_result = self._deploy_storage(config)
            result.resources.update(storage_result)
            
            # Deploy compute resources
            if config.instance_type:
                compute_result = self._deploy_compute(config, vpc_result)
                result.resources.update(compute_result)
                
            result.success = True
            result.message = "AWS infrastructure deployed successfully"
            
        except Exception as e:
            result.success = False
            result.message = f"AWS deployment failed: {str(e)}"
            self.logger.error(result.message)
            
        return result
        
    def _deploy_vpc(self, config: ResourceConfig) -> Dict[str, Any]:
        """Deploy VPC and networking resources."""
        ec2 = self._get_client('ec2')
        resources = {}
        
        # Create VPC
        vpc_response = ec2.create_vpc(CidrBlock='10.0.0.0/16')
        vpc_id = vpc_response['Vpc']['VpcId']
        resources['vpc_id'] = vpc_id
        
        # Enable DNS
        ec2.modify_vpc_attribute(VpcId=vpc_id, EnableDnsHostnames={'Value': True})
        
        # Create subnets
        subnet_response = ec2.create_subnet(
            VpcId=vpc_id,
            CidrBlock='10.0.1.0/24',
            AvailabilityZone=f"{self.region}a"
        )
        resources['subnet_id'] = subnet_response['Subnet']['SubnetId']
        
        # Create Internet Gateway
        igw_response = ec2.create_internet_gateway()
        igw_id = igw_response['InternetGateway']['InternetGatewayId']
        ec2.attach_internet_gateway(InternetGatewayId=igw_id, VpcId=vpc_id)
        resources['igw_id'] = igw_id
        
        # Create route table
        route_table = ec2.create_route_table(VpcId=vpc_id)
        route_table_id = route_table['RouteTable']['RouteTableId']
        
        # Add route to Internet Gateway
        ec2.create_route(
            RouteTableId=route_table_id,
            DestinationCidrBlock='0.0.0.0/0',
            GatewayId=igw_id
        )
        
        # Associate subnet with route table
        ec2.associate_route_table(
            RouteTableId=route_table_id,
            SubnetId=resources['subnet_id']
        )
        
        # Create security group
        sg_response = ec2.create_security_group(
            GroupName=f'{self.project_id}-sg',
            Description='Security group for Inferloop Synthetic Data',
            VpcId=vpc_id
        )
        sg_id = sg_response['GroupId']
        
        # Add security group rules
        ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 22,
                    'ToPort': 22,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                },
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 80,
                    'ToPort': 80,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                },
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 443,
                    'ToPort': 443,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                },
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 8000,
                    'ToPort': 8000,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                }
            ]
        )
        resources['security_group_id'] = sg_id
        
        return resources
        
    def _deploy_storage(self, config: ResourceConfig) -> Dict[str, Any]:
        """Deploy S3 bucket for storage."""
        s3 = self._get_client('s3')
        resources = {}
        
        bucket_name = f"{self.project_id}-{self.region}-synthdata"
        
        # Create S3 bucket
        if self.region == 'us-east-1':
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': self.region}
            )
            
        # Enable versioning
        s3.put_bucket_versioning(
            Bucket=bucket_name,
            VersioningConfiguration={'Status': 'Enabled'}
        )
        
        # Enable encryption
        s3.put_bucket_encryption(
            Bucket=bucket_name,
            ServerSideEncryptionConfiguration={
                'Rules': [{
                    'ApplyServerSideEncryptionByDefault': {
                        'SSEAlgorithm': 'AES256'
                    }
                }]
            }
        )
        
        resources['bucket_name'] = bucket_name
        return resources
        
    def _deploy_compute(self, config: ResourceConfig, vpc_resources: Dict) -> Dict[str, Any]:
        """Deploy compute resources."""
        ec2 = self._get_client('ec2')
        resources = {}
        
        # Get latest Amazon Linux 2 AMI
        ami_response = ec2.describe_images(
            Owners=['amazon'],
            Filters=[
                {'Name': 'name', 'Values': ['amzn2-ami-hvm-*-x86_64-gp2']},
                {'Name': 'state', 'Values': ['available']}
            ]
        )
        ami_id = sorted(ami_response['Images'], 
                       key=lambda x: x['CreationDate'], 
                       reverse=True)[0]['ImageId']
        
        # Create EC2 instance
        instance_response = ec2.run_instances(
            ImageId=ami_id,
            InstanceType=config.instance_type or 't3.medium',
            MinCount=1,
            MaxCount=1,
            KeyName=config.ssh_key_name if hasattr(config, 'ssh_key_name') else None,
            SecurityGroupIds=[vpc_resources['security_group_id']],
            SubnetId=vpc_resources['subnet_id'],
            TagSpecifications=[{
                'ResourceType': 'instance',
                'Tags': [
                    {'Key': 'Name', 'Value': f'{self.project_id}-instance'},
                    {'Key': 'Project', 'Value': self.project_id}
                ]
            }],
            UserData=self._get_user_data()
        )
        
        instance_id = instance_response['Instances'][0]['InstanceId']
        resources['instance_id'] = instance_id
        
        # Wait for instance to be running
        waiter = ec2.get_waiter('instance_running')
        waiter.wait(InstanceIds=[instance_id])
        
        # Get instance details
        instance_info = ec2.describe_instances(InstanceIds=[instance_id])
        public_ip = instance_info['Reservations'][0]['Instances'][0].get('PublicIpAddress')
        resources['public_ip'] = public_ip
        
        return resources
        
    def _get_user_data(self) -> str:
        """Get EC2 user data script."""
        return """#!/bin/bash
# Update system
yum update -y

# Install Docker
amazon-linux-extras install docker -y
service docker start
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install Python 3.9
yum install python39 python39-pip -y

# Create app directory
mkdir -p /opt/inferloop-synthdata
"""
        
    def deploy_container(self, config: ResourceConfig) -> DeploymentResult:
        """Deploy containerized application on AWS."""
        result = DeploymentResult()
        
        try:
            if hasattr(config, 'use_fargate') and config.use_fargate:
                result = self._deploy_fargate(config)
            else:
                result = self._deploy_ecs(config)
                
        except Exception as e:
            result.success = False
            result.message = f"Container deployment failed: {str(e)}"
            
        return result
        
    def _deploy_ecs(self, config: ResourceConfig) -> DeploymentResult:
        """Deploy application on ECS."""
        result = DeploymentResult()
        ecs = self._get_client('ecs')
        
        # Create ECS cluster
        cluster_name = f"{self.project_id}-cluster"
        ecs.create_cluster(clusterName=cluster_name)
        
        # Register task definition
        task_def = {
            'family': f"{self.project_id}-task",
            'networkMode': 'awsvpc',
            'requiresCompatibilities': ['EC2'],
            'cpu': str(config.cpu or 1024),
            'memory': str(config.memory or 2048),
            'containerDefinitions': [{
                'name': 'synthdata',
                'image': config.container_image,
                'memory': config.memory or 2048,
                'portMappings': [{
                    'containerPort': 8000,
                    'protocol': 'tcp'
                }],
                'environment': [
                    {'name': k, 'value': v} 
                    for k, v in (config.environment or {}).items()
                ],
                'logConfiguration': {
                    'logDriver': 'awslogs',
                    'options': {
                        'awslogs-group': f'/ecs/{self.project_id}',
                        'awslogs-region': self.region,
                        'awslogs-stream-prefix': 'synthdata'
                    }
                }
            }]
        }
        
        ecs.register_task_definition(**task_def)
        
        result.success = True
        result.message = "ECS deployment completed"
        result.resources = {
            'cluster_name': cluster_name,
            'task_family': task_def['family']
        }
        
        return result
        
    def _deploy_fargate(self, config: ResourceConfig) -> DeploymentResult:
        """Deploy application on Fargate."""
        result = DeploymentResult()
        ecs = self._get_client('ecs')
        ec2 = self._get_client('ec2')
        
        # Create ECS cluster
        cluster_name = f"{self.project_id}-fargate-cluster"
        ecs.create_cluster(
            clusterName=cluster_name,
            capacityProviders=['FARGATE', 'FARGATE_SPOT']
        )
        
        # Get default VPC and subnets
        vpcs = ec2.describe_vpcs(Filters=[{'Name': 'isDefault', 'Values': ['true']}])
        vpc_id = vpcs['Vpcs'][0]['VpcId']
        
        subnets = ec2.describe_subnets(
            Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}]
        )
        subnet_ids = [s['SubnetId'] for s in subnets['Subnets']][:2]
        
        # Create security group
        sg_response = ec2.create_security_group(
            GroupName=f'{self.project_id}-fargate-sg',
            Description='Security group for Fargate',
            VpcId=vpc_id
        )
        sg_id = sg_response['GroupId']
        
        ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[{
                'IpProtocol': 'tcp',
                'FromPort': 8000,
                'ToPort': 8000,
                'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
            }]
        )
        
        # Register task definition
        task_def = {
            'family': f"{self.project_id}-fargate-task",
            'networkMode': 'awsvpc',
            'requiresCompatibilities': ['FARGATE'],
            'cpu': str(config.cpu or 1024),
            'memory': str(config.memory or 2048),
            'executionRoleArn': self._get_or_create_execution_role(),
            'containerDefinitions': [{
                'name': 'synthdata',
                'image': config.container_image,
                'portMappings': [{
                    'containerPort': 8000,
                    'protocol': 'tcp'
                }],
                'environment': [
                    {'name': k, 'value': v} 
                    for k, v in (config.environment or {}).items()
                ],
                'logConfiguration': {
                    'logDriver': 'awslogs',
                    'options': {
                        'awslogs-group': f'/ecs/{self.project_id}',
                        'awslogs-region': self.region,
                        'awslogs-stream-prefix': 'fargate'
                    }
                }
            }]
        }
        
        ecs.register_task_definition(**task_def)
        
        # Create service
        service_name = f"{self.project_id}-service"
        ecs.create_service(
            cluster=cluster_name,
            serviceName=service_name,
            taskDefinition=task_def['family'],
            desiredCount=config.replicas or 1,
            launchType='FARGATE',
            networkConfiguration={
                'awsvpcConfiguration': {
                    'subnets': subnet_ids,
                    'securityGroups': [sg_id],
                    'assignPublicIp': 'ENABLED'
                }
            }
        )
        
        result.success = True
        result.message = "Fargate deployment completed"
        result.resources = {
            'cluster_name': cluster_name,
            'service_name': service_name,
            'security_group_id': sg_id
        }
        
        return result
        
    def _get_or_create_execution_role(self) -> str:
        """Get or create ECS execution role."""
        iam = self._get_client('iam')
        role_name = f"{self.project_id}-ecs-execution-role"
        
        try:
            # Check if role exists
            role = iam.get_role(RoleName=role_name)
            return role['Role']['Arn']
        except iam.exceptions.NoSuchEntityException:
            # Create role
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }
            
            role = iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description='ECS task execution role for Inferloop Synthetic Data'
            )
            
            # Attach managed policy
            iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy'
            )
            
            return role['Role']['Arn']
            
    def deploy_serverless(self, config: ResourceConfig) -> DeploymentResult:
        """Deploy serverless functions on AWS Lambda."""
        result = DeploymentResult()
        lambda_client = self._get_client('lambda')
        
        try:
            # Create Lambda function
            function_name = f"{self.project_id}-synthdata-lambda"
            
            # Create deployment package
            import zipfile
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
                with zipfile.ZipFile(tmp_file.name, 'w') as zf:
                    # Add handler code
                    zf.writestr('lambda_function.py', self._get_lambda_handler())
                    
                # Read zip file
                with open(tmp_file.name, 'rb') as f:
                    zip_content = f.read()
                    
            # Create Lambda function
            response = lambda_client.create_function(
                FunctionName=function_name,
                Runtime='python3.9',
                Role=self._get_or_create_lambda_role(),
                Handler='lambda_function.lambda_handler',
                Code={'ZipFile': zip_content},
                Description='Inferloop Synthetic Data Lambda function',
                Timeout=300,
                MemorySize=config.memory or 512,
                Environment={
                    'Variables': config.environment or {}
                }
            )
            
            result.success = True
            result.message = "Lambda function deployed successfully"
            result.resources = {
                'function_name': function_name,
                'function_arn': response['FunctionArn']
            }
            
        except Exception as e:
            result.success = False
            result.message = f"Lambda deployment failed: {str(e)}"
            
        return result
        
    def _get_lambda_handler(self) -> str:
        """Get Lambda handler code."""
        return '''
import json

def lambda_handler(event, context):
    """Lambda handler for synthetic data generation."""
    
    # Extract parameters from event
    generator_type = event.get('generator_type', 'sdv')
    model_type = event.get('model_type', 'gaussian_copula')
    num_samples = event.get('num_samples', 1000)
    
    # TODO: Integrate with actual synthetic data generation
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Synthetic data generation completed',
            'generator_type': generator_type,
            'model_type': model_type,
            'num_samples': num_samples
        })
    }
'''
        
    def _get_or_create_lambda_role(self) -> str:
        """Get or create Lambda execution role."""
        iam = self._get_client('iam')
        role_name = f"{self.project_id}-lambda-execution-role"
        
        try:
            role = iam.get_role(RoleName=role_name)
            return role['Role']['Arn']
        except iam.exceptions.NoSuchEntityException:
            # Create role
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }
            
            role = iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description='Lambda execution role for Inferloop Synthetic Data'
            )
            
            # Attach managed policy
            iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
            )
            
            return role['Role']['Arn']
            
    def deploy_batch(self, config: ResourceConfig) -> DeploymentResult:
        """Deploy batch processing on AWS Batch."""
        result = DeploymentResult()
        batch = self._get_client('batch')
        
        try:
            # Create compute environment
            ce_name = f"{self.project_id}-compute-env"
            batch.create_compute_environment(
                computeEnvironmentName=ce_name,
                type='MANAGED',
                state='ENABLED',
                computeResources={
                    'type': 'EC2',
                    'minvCpus': 0,
                    'maxvCpus': 256,
                    'desiredvCpus': 4,
                    'instanceTypes': ['optimal'],
                    'subnets': self._get_default_subnets(),
                    'securityGroupIds': [self._get_or_create_batch_sg()],
                    'instanceRole': self._get_or_create_instance_profile()
                },
                serviceRole=self._get_or_create_batch_service_role()
            )
            
            # Create job queue
            queue_name = f"{self.project_id}-job-queue"
            batch.create_job_queue(
                jobQueueName=queue_name,
                state='ENABLED',
                priority=1,
                computeEnvironmentOrder=[{
                    'order': 1,
                    'computeEnvironment': ce_name
                }]
            )
            
            # Register job definition
            job_def_name = f"{self.project_id}-job-def"
            batch.register_job_definition(
                jobDefinitionName=job_def_name,
                type='container',
                containerProperties={
                    'image': config.container_image,
                    'vcpus': config.cpu or 2,
                    'memory': config.memory or 2048,
                    'environment': [
                        {'name': k, 'value': v}
                        for k, v in (config.environment or {}).items()
                    ]
                }
            )
            
            result.success = True
            result.message = "AWS Batch environment created"
            result.resources = {
                'compute_environment': ce_name,
                'job_queue': queue_name,
                'job_definition': job_def_name
            }
            
        except Exception as e:
            result.success = False
            result.message = f"Batch deployment failed: {str(e)}"
            
        return result
        
    def _get_default_subnets(self) -> List[str]:
        """Get default VPC subnets."""
        ec2 = self._get_client('ec2')
        
        vpcs = ec2.describe_vpcs(Filters=[{'Name': 'isDefault', 'Values': ['true']}])
        vpc_id = vpcs['Vpcs'][0]['VpcId']
        
        subnets = ec2.describe_subnets(
            Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}]
        )
        
        return [s['SubnetId'] for s in subnets['Subnets']]
        
    def _get_or_create_batch_sg(self) -> str:
        """Get or create security group for Batch."""
        ec2 = self._get_client('ec2')
        
        sg_name = f"{self.project_id}-batch-sg"
        
        try:
            # Check if exists
            sgs = ec2.describe_security_groups(GroupNames=[sg_name])
            return sgs['SecurityGroups'][0]['GroupId']
        except ec2.exceptions.ClientError:
            # Create new
            vpcs = ec2.describe_vpcs(Filters=[{'Name': 'isDefault', 'Values': ['true']}])
            vpc_id = vpcs['Vpcs'][0]['VpcId']
            
            response = ec2.create_security_group(
                GroupName=sg_name,
                Description='Security group for AWS Batch',
                VpcId=vpc_id
            )
            
            return response['GroupId']
            
    def _get_or_create_instance_profile(self) -> str:
        """Get or create EC2 instance profile for Batch."""
        iam = self._get_client('iam')
        profile_name = f"{self.project_id}-batch-instance-profile"
        
        try:
            profile = iam.get_instance_profile(InstanceProfileName=profile_name)
            return profile['InstanceProfile']['Arn']
        except iam.exceptions.NoSuchEntityException:
            # Create role first
            role_name = f"{self.project_id}-batch-instance-role"
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "ec2.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }
            
            iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy)
            )
            
            # Attach policies
            iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role'
            )
            
            # Create instance profile
            iam.create_instance_profile(InstanceProfileName=profile_name)
            iam.add_role_to_instance_profile(
                InstanceProfileName=profile_name,
                RoleName=role_name
            )
            
            return f"arn:aws:iam::{self._get_account_id()}:instance-profile/{profile_name}"
            
    def _get_or_create_batch_service_role(self) -> str:
        """Get or create Batch service role."""
        iam = self._get_client('iam')
        role_name = f"{self.project_id}-batch-service-role"
        
        try:
            role = iam.get_role(RoleName=role_name)
            return role['Role']['Arn']
        except iam.exceptions.NoSuchEntityException:
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "batch.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }
            
            role = iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy)
            )
            
            iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole'
            )
            
            return role['Role']['Arn']
            
    def _get_account_id(self) -> str:
        """Get AWS account ID."""
        sts = self._get_client('sts')
        return sts.get_caller_identity()['Account']
        
    def deploy_database(self, config: ResourceConfig) -> DeploymentResult:
        """Deploy database on AWS RDS."""
        result = DeploymentResult()
        rds = self._get_client('rds')
        
        try:
            db_instance_id = f"{self.project_id}-synthdata-db"
            
            # Create RDS instance
            response = rds.create_db_instance(
                DBInstanceIdentifier=db_instance_id,
                DBInstanceClass=config.instance_type or 'db.t3.micro',
                Engine='postgres',
                EngineVersion='13.7',
                MasterUsername='synthdata',
                MasterUserPassword=config.db_password,
                AllocatedStorage=config.disk_size_gb or 20,
                StorageType='gp2',
                StorageEncrypted=True,
                BackupRetentionPeriod=7,
                PreferredBackupWindow='03:00-04:00',
                PreferredMaintenanceWindow='sun:04:00-sun:05:00',
                MultiAZ=False,
                PubliclyAccessible=True,
                VpcSecurityGroupIds=[self._get_or_create_db_sg()],
                Tags=[
                    {'Key': 'Name', 'Value': db_instance_id},
                    {'Key': 'Project', 'Value': self.project_id}
                ]
            )
            
            result.success = True
            result.message = "RDS database created"
            result.resources = {
                'db_instance_id': db_instance_id,
                'endpoint': response['DBInstance']['Endpoint']['Address']
            }
            
        except Exception as e:
            result.success = False
            result.message = f"Database deployment failed: {str(e)}"
            
        return result
        
    def _get_or_create_db_sg(self) -> str:
        """Get or create security group for RDS."""
        ec2 = self._get_client('ec2')
        
        sg_name = f"{self.project_id}-rds-sg"
        
        try:
            sgs = ec2.describe_security_groups(GroupNames=[sg_name])
            return sgs['SecurityGroups'][0]['GroupId']
        except ec2.exceptions.ClientError:
            vpcs = ec2.describe_vpcs(Filters=[{'Name': 'isDefault', 'Values': ['true']}])
            vpc_id = vpcs['Vpcs'][0]['VpcId']
            
            response = ec2.create_security_group(
                GroupName=sg_name,
                Description='Security group for RDS',
                VpcId=vpc_id
            )
            
            sg_id = response['GroupId']
            
            # Allow PostgreSQL access
            ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[{
                    'IpProtocol': 'tcp',
                    'FromPort': 5432,
                    'ToPort': 5432,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                }]
            )
            
            return sg_id
            
    def destroy(self, resource_ids: Dict[str, str]) -> DeploymentResult:
        """Destroy AWS resources."""
        result = DeploymentResult()
        
        try:
            # Destroy EC2 instances
            if 'instance_id' in resource_ids:
                ec2 = self._get_client('ec2')
                ec2.terminate_instances(InstanceIds=[resource_ids['instance_id']])
                
            # Delete S3 bucket
            if 'bucket_name' in resource_ids:
                s3 = self._get_client('s3')
                # Empty bucket first
                bucket = self.session.resource('s3').Bucket(resource_ids['bucket_name'])
                bucket.objects.all().delete()
                bucket.delete()
                
            # Delete RDS instance
            if 'db_instance_id' in resource_ids:
                rds = self._get_client('rds')
                rds.delete_db_instance(
                    DBInstanceIdentifier=resource_ids['db_instance_id'],
                    SkipFinalSnapshot=True
                )
                
            result.success = True
            result.message = "Resources destroyed successfully"
            
        except Exception as e:
            result.success = False
            result.message = f"Resource destruction failed: {str(e)}"
            
        return result
        
    def get_status(self) -> Dict[str, Any]:
        """Get status of deployed resources."""
        status = {
            'provider': 'aws',
            'region': self.region,
            'resources': {}
        }
        
        try:
            # Check EC2 instances
            ec2 = self._get_client('ec2')
            instances = ec2.describe_instances(
                Filters=[
                    {'Name': 'tag:Project', 'Values': [self.project_id]},
                    {'Name': 'instance-state-name', 'Values': ['running']}
                ]
            )
            
            status['resources']['instances'] = []
            for reservation in instances['Reservations']:
                for instance in reservation['Instances']:
                    status['resources']['instances'].append({
                        'id': instance['InstanceId'],
                        'type': instance['InstanceType'],
                        'state': instance['State']['Name'],
                        'public_ip': instance.get('PublicIpAddress')
                    })
                    
            # Check S3 buckets
            s3 = self._get_client('s3')
            buckets = s3.list_buckets()
            
            status['resources']['buckets'] = []
            for bucket in buckets['Buckets']:
                if self.project_id in bucket['Name']:
                    status['resources']['buckets'].append({
                        'name': bucket['Name'],
                        'created': bucket['CreationDate'].isoformat()
                    })
                    
            # Check RDS instances
            rds = self._get_client('rds')
            db_instances = rds.describe_db_instances()
            
            status['resources']['databases'] = []
            for db in db_instances['DBInstances']:
                if self.project_id in db['DBInstanceIdentifier']:
                    status['resources']['databases'].append({
                        'id': db['DBInstanceIdentifier'],
                        'engine': db['Engine'],
                        'status': db['DBInstanceStatus'],
                        'endpoint': db.get('Endpoint', {}).get('Address')
                    })
                    
        except Exception as e:
            status['error'] = str(e)
            
        return status
        
    def estimate_cost(self, config: ResourceConfig) -> Dict[str, float]:
        """Estimate AWS costs."""
        costs = {
            'compute': 0.0,
            'storage': 0.0,
            'network': 0.0,
            'total': 0.0
        }
        
        # EC2 costs (rough estimates)
        instance_costs = {
            't3.micro': 0.0104,
            't3.small': 0.0208,
            't3.medium': 0.0416,
            't3.large': 0.0832,
            'm5.large': 0.096,
            'm5.xlarge': 0.192,
            'c5.large': 0.085,
            'c5.xlarge': 0.17
        }
        
        instance_type = config.instance_type or 't3.medium'
        if instance_type in instance_costs:
            costs['compute'] = instance_costs[instance_type] * 24 * 30  # Monthly
            
        # Storage costs
        costs['storage'] = (config.disk_size_gb or 100) * 0.10  # $0.10 per GB
        
        # Network costs (rough estimate)
        costs['network'] = 50.0  # Estimated $50/month for data transfer
        
        costs['total'] = sum(costs.values())
        
        return costs
        
    def deploy_eks(self, config: ResourceConfig) -> DeploymentResult:
        """Deploy EKS cluster."""
        if not self.advanced_services:
            result = DeploymentResult()
            result.success = False
            result.message = "Provider not authenticated"
            return result
            
        return self.advanced_services.deploy_eks_cluster(config)
        
    def deploy_dynamodb(self, config: ResourceConfig) -> DeploymentResult:
        """Deploy DynamoDB table."""
        if not self.advanced_services:
            result = DeploymentResult()
            result.success = False
            result.message = "Provider not authenticated"
            return result
            
        return self.advanced_services.deploy_dynamodb_table(config)
        
    def deploy_enhanced_lambda(self, config: ResourceConfig) -> DeploymentResult:
        """Deploy enhanced Lambda with API Gateway."""
        if not self.advanced_services:
            result = DeploymentResult()
            result.success = False
            result.message = "Provider not authenticated"
            return result
            
        return self.advanced_services.deploy_enhanced_lambda(config)