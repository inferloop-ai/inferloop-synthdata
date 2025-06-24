"""Enhanced AWS services implementation with EKS, DynamoDB, and advanced features."""

import boto3
import json
import base64
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..base import DeploymentResult, ResourceConfig


class AWSAdvancedServices:
    """Advanced AWS services implementation."""
    
    def __init__(self, session: boto3.Session, project_id: str, region: str):
        """Initialize advanced services."""
        self.session = session
        self.project_id = project_id
        self.region = region
        self.clients = {}
        
    def _get_client(self, service: str):
        """Get or create AWS client."""
        if service not in self.clients:
            self.clients[service] = self.session.client(service)
        return self.clients[service]
        
    def deploy_eks_cluster(self, config: ResourceConfig) -> DeploymentResult:
        """Deploy Amazon EKS (Elastic Kubernetes Service) cluster."""
        result = DeploymentResult()
        eks = self._get_client('eks')
        ec2 = self._get_client('ec2')
        iam = self._get_client('iam')
        
        try:
            cluster_name = f"{self.project_id}-eks-cluster"
            
            # Create EKS service role
            role_arn = self._create_eks_service_role()
            
            # Get or create VPC
            vpc_id, subnet_ids = self._get_or_create_eks_vpc()
            
            # Create security group
            sg_id = self._create_eks_security_group(vpc_id)
            
            # Create EKS cluster
            response = eks.create_cluster(
                name=cluster_name,
                version='1.27',
                roleArn=role_arn,
                resourcesVpcConfig={
                    'subnetIds': subnet_ids,
                    'securityGroupIds': [sg_id],
                    'endpointPublicAccess': True,
                    'endpointPrivateAccess': True
                },
                logging={
                    'clusterLogging': [
                        {
                            'types': ['api', 'audit', 'authenticator', 'controllerManager', 'scheduler'],
                            'enabled': True
                        }
                    ]
                },
                tags={
                    'Project': self.project_id,
                    'ManagedBy': 'inferloop-synthdata'
                }
            )
            
            # Wait for cluster to be active
            waiter = eks.get_waiter('cluster_active')
            waiter.wait(name=cluster_name)
            
            # Create node group
            node_group_result = self._create_eks_node_group(cluster_name, subnet_ids)
            
            result.success = True
            result.message = "EKS cluster created successfully"
            result.resources = {
                'cluster_name': cluster_name,
                'cluster_arn': response['cluster']['arn'],
                'endpoint': response['cluster']['endpoint'],
                'node_group': node_group_result['nodegroup']['nodegroupName']
            }
            
        except Exception as e:
            result.success = False
            result.message = f"EKS deployment failed: {str(e)}"
            
        return result
        
    def _create_eks_service_role(self) -> str:
        """Create EKS service role."""
        iam = self._get_client('iam')
        role_name = f"{self.project_id}-eks-service-role"
        
        try:
            role = iam.get_role(RoleName=role_name)
            return role['Role']['Arn']
        except iam.exceptions.NoSuchEntityException:
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "eks.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }
            
            role = iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description='EKS service role for cluster management'
            )
            
            # Attach required policies
            policies = [
                'arn:aws:iam::aws:policy/AmazonEKSClusterPolicy',
                'arn:aws:iam::aws:policy/AmazonEKSVPCResourceController'
            ]
            
            for policy in policies:
                iam.attach_role_policy(RoleName=role_name, PolicyArn=policy)
                
            return role['Role']['Arn']
            
    def _get_or_create_eks_vpc(self) -> tuple:
        """Get or create VPC for EKS."""
        ec2 = self._get_client('ec2')
        
        # For now, use default VPC
        vpcs = ec2.describe_vpcs(Filters=[{'Name': 'isDefault', 'Values': ['true']}])
        vpc_id = vpcs['Vpcs'][0]['VpcId']
        
        # Get all subnets in different AZs
        subnets = ec2.describe_subnets(
            Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}]
        )
        
        # Select subnets from different AZs
        subnet_ids = []
        azs_used = set()
        
        for subnet in subnets['Subnets']:
            az = subnet['AvailabilityZone']
            if az not in azs_used and len(subnet_ids) < 2:
                subnet_ids.append(subnet['SubnetId'])
                azs_used.add(az)
                
        return vpc_id, subnet_ids
        
    def _create_eks_security_group(self, vpc_id: str) -> str:
        """Create security group for EKS."""
        ec2 = self._get_client('ec2')
        sg_name = f"{self.project_id}-eks-sg"
        
        try:
            sgs = ec2.describe_security_groups(GroupNames=[sg_name])
            return sgs['SecurityGroups'][0]['GroupId']
        except ec2.exceptions.ClientError:
            response = ec2.create_security_group(
                GroupName=sg_name,
                Description='Security group for EKS cluster',
                VpcId=vpc_id
            )
            
            sg_id = response['GroupId']
            
            # Add ingress rules
            ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 443,
                        'ToPort': 443,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                    },
                    {
                        'IpProtocol': '-1',
                        'UserIdGroupPairs': [{'GroupId': sg_id}]
                    }
                ]
            )
            
            return sg_id
            
    def _create_eks_node_group(self, cluster_name: str, subnet_ids: List[str]) -> Dict[str, Any]:
        """Create EKS node group."""
        eks = self._get_client('eks')
        iam = self._get_client('iam')
        
        # Create node role
        node_role_name = f"{self.project_id}-eks-node-role"
        
        try:
            role = iam.get_role(RoleName=node_role_name)
            node_role_arn = role['Role']['Arn']
        except iam.exceptions.NoSuchEntityException:
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "ec2.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }
            
            role = iam.create_role(
                RoleName=node_role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy)
            )
            
            node_role_arn = role['Role']['Arn']
            
            # Attach required policies
            policies = [
                'arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy',
                'arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy',
                'arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly'
            ]
            
            for policy in policies:
                iam.attach_role_policy(RoleName=node_role_name, PolicyArn=policy)
                
        # Create node group
        node_group_name = f"{self.project_id}-node-group"
        
        response = eks.create_nodegroup(
            clusterName=cluster_name,
            nodegroupName=node_group_name,
            scalingConfig={
                'minSize': 1,
                'maxSize': 3,
                'desiredSize': 2
            },
            diskSize=100,
            subnets=subnet_ids,
            instanceTypes=['t3.medium'],
            amiType='AL2_x86_64',
            nodeRole=node_role_arn,
            tags={
                'Project': self.project_id,
                'Type': 'eks-node'
            }
        )
        
        return response
        
    def deploy_dynamodb_table(self, config: ResourceConfig) -> DeploymentResult:
        """Deploy DynamoDB table."""
        result = DeploymentResult()
        dynamodb = self._get_client('dynamodb')
        
        try:
            table_name = f"{self.project_id}-synthdata-table"
            
            # Create table
            response = dynamodb.create_table(
                TableName=table_name,
                KeySchema=[
                    {'AttributeName': 'id', 'KeyType': 'HASH'},
                    {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'id', 'AttributeType': 'S'},
                    {'AttributeName': 'timestamp', 'AttributeType': 'N'}
                ],
                BillingMode='PAY_PER_REQUEST',
                StreamSpecification={
                    'StreamEnabled': True,
                    'StreamViewType': 'NEW_AND_OLD_IMAGES'
                },
                SSESpecification={
                    'Enabled': True,
                    'SSEType': 'KMS',
                    'KMSMasterKeyId': 'alias/aws/dynamodb'
                },
                Tags=[
                    {'Key': 'Project', 'Value': self.project_id},
                    {'Key': 'Service', 'Value': 'synthdata'}
                ]
            )
            
            # Wait for table to be active
            waiter = dynamodb.get_waiter('table_exists')
            waiter.wait(TableName=table_name)
            
            # Enable auto-scaling for provisioned mode (if needed)
            if config.enable_autoscaling:
                self._setup_dynamodb_autoscaling(table_name)
                
            result.success = True
            result.message = "DynamoDB table created successfully"
            result.resources = {
                'table_name': table_name,
                'table_arn': response['TableDescription']['TableArn'],
                'stream_arn': response['TableDescription']['LatestStreamArn']
            }
            
        except Exception as e:
            result.success = False
            result.message = f"DynamoDB deployment failed: {str(e)}"
            
        return result
        
    def _setup_dynamodb_autoscaling(self, table_name: str):
        """Setup auto-scaling for DynamoDB table."""
        autoscaling = self._get_client('application-autoscaling')
        
        # Register scalable targets
        for dimension in ['dynamodb:table:ReadCapacityUnits', 'dynamodb:table:WriteCapacityUnits']:
            autoscaling.register_scalable_target(
                ServiceNamespace='dynamodb',
                ResourceId=f'table/{table_name}',
                ScalableDimension=dimension,
                MinCapacity=5,
                MaxCapacity=100,
                RoleARN=self._get_autoscaling_role()
            )
            
        # Create scaling policies
        for metric_type, dimension in [
            ('DynamoDBReadCapacityUtilization', 'dynamodb:table:ReadCapacityUnits'),
            ('DynamoDBWriteCapacityUtilization', 'dynamodb:table:WriteCapacityUnits')
        ]:
            autoscaling.put_scaling_policy(
                PolicyName=f'{table_name}-{metric_type}-policy',
                ServiceNamespace='dynamodb',
                ResourceId=f'table/{table_name}',
                ScalableDimension=dimension,
                PolicyType='TargetTrackingScaling',
                TargetTrackingScalingPolicyConfiguration={
                    'TargetValue': 70.0,
                    'PredefinedMetricSpecification': {
                        'PredefinedMetricType': metric_type
                    },
                    'ScaleInCooldown': 60,
                    'ScaleOutCooldown': 60
                }
            )
            
    def _get_autoscaling_role(self) -> str:
        """Get or create auto-scaling role."""
        iam = self._get_client('iam')
        role_name = f"{self.project_id}-dynamodb-autoscaling-role"
        
        try:
            role = iam.get_role(RoleName=role_name)
            return role['Role']['Arn']
        except iam.exceptions.NoSuchEntityException:
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "application-autoscaling.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }
            
            role = iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy)
            )
            
            # Create and attach policy
            policy_document = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "dynamodb:DescribeTable",
                            "dynamodb:UpdateTable",
                            "cloudwatch:PutMetricAlarm",
                            "cloudwatch:DescribeAlarms",
                            "cloudwatch:GetMetricStatistics",
                            "cloudwatch:SetAlarmState",
                            "cloudwatch:DeleteAlarms"
                        ],
                        "Resource": "*"
                    }
                ]
            }
            
            iam.put_role_policy(
                RoleName=role_name,
                PolicyName='DynamoDBAutoscalingPolicy',
                PolicyDocument=json.dumps(policy_document)
            )
            
            return role['Role']['Arn']
            
    def deploy_enhanced_lambda(self, config: ResourceConfig) -> DeploymentResult:
        """Deploy enhanced Lambda function with API Gateway and EventBridge."""
        result = DeploymentResult()
        lambda_client = self._get_client('lambda')
        apigateway = self._get_client('apigatewayv2')
        events = self._get_client('events')
        
        try:
            function_name = f"{self.project_id}-enhanced-lambda"
            
            # Create Lambda function with enhanced configuration
            function_response = self._create_enhanced_lambda_function(function_name, config)
            function_arn = function_response['FunctionArn']
            
            # Create API Gateway
            api_response = self._create_api_gateway(function_name, function_arn)
            
            # Create EventBridge rules
            event_rules = self._create_event_rules(function_name, function_arn)
            
            # Set up Lambda destinations
            self._configure_lambda_destinations(function_name)
            
            result.success = True
            result.message = "Enhanced Lambda deployment completed"
            result.resources = {
                'function_name': function_name,
                'function_arn': function_arn,
                'api_endpoint': api_response['ApiEndpoint'],
                'event_rules': event_rules
            }
            
        except Exception as e:
            result.success = False
            result.message = f"Enhanced Lambda deployment failed: {str(e)}"
            
        return result
        
    def _create_enhanced_lambda_function(self, function_name: str, config: ResourceConfig) -> Dict[str, Any]:
        """Create enhanced Lambda function."""
        lambda_client = self._get_client('lambda')
        
        # Create deployment package with dependencies
        zip_content = self._create_lambda_package()
        
        response = lambda_client.create_function(
            FunctionName=function_name,
            Runtime='python3.9',
            Role=self._get_or_create_enhanced_lambda_role(),
            Handler='lambda_function.lambda_handler',
            Code={'ZipFile': zip_content},
            Description='Enhanced synthetic data generation Lambda',
            Timeout=900,  # 15 minutes
            MemorySize=config.memory or 3008,  # Up to 3GB
            Environment={
                'Variables': {
                    'PROJECT_ID': self.project_id,
                    'REGION': self.region,
                    **config.environment
                }
            },
            Layers=self._get_lambda_layers(),
            TracingConfig={'Mode': 'Active'},  # X-Ray tracing
            EphemeralStorage={'Size': 10240},  # 10GB ephemeral storage
            Architectures=['x86_64']
        )
        
        # Add Lambda insights
        lambda_client.put_function_configuration(
            FunctionName=function_name,
            Layers=response['Layers'] + [
                f'arn:aws:lambda:{self.region}:580247275435:layer:LambdaInsightsExtension:14'
            ]
        )
        
        return response
        
    def _create_lambda_package(self) -> bytes:
        """Create Lambda deployment package."""
        import zipfile
        import tempfile
        
        code = '''
import json
import boto3
import os
from datetime import datetime

def lambda_handler(event, context):
    """Enhanced Lambda handler for synthetic data generation."""
    
    # Parse event
    if 'body' in event:
        # API Gateway event
        body = json.loads(event['body'])
    else:
        # Direct invocation or EventBridge
        body = event
        
    generator_type = body.get('generator_type', 'sdv')
    model_type = body.get('model_type', 'gaussian_copula')
    num_samples = body.get('num_samples', 1000)
    
    # Initialize services
    s3 = boto3.client('s3')
    dynamodb = boto3.resource('dynamodb')
    
    # Generate unique job ID
    job_id = f"{context.request_id}"
    timestamp = int(datetime.utcnow().timestamp())
    
    # Store job metadata in DynamoDB
    table_name = f"{os.environ['PROJECT_ID']}-synthdata-table"
    table = dynamodb.Table(table_name)
    
    table.put_item(
        Item={
            'id': job_id,
            'timestamp': timestamp,
            'status': 'processing',
            'generator_type': generator_type,
            'model_type': model_type,
            'num_samples': num_samples,
            'created_at': datetime.utcnow().isoformat()
        }
    )
    
    try:
        # TODO: Integrate actual synthetic data generation
        result_data = {
            'job_id': job_id,
            'samples_generated': num_samples,
            'generator_used': generator_type,
            'model_used': model_type
        }
        
        # Store results in S3
        bucket_name = f"{os.environ['PROJECT_ID']}-{os.environ['REGION']}-synthdata"
        s3.put_object(
            Bucket=bucket_name,
            Key=f"results/{job_id}/data.json",
            Body=json.dumps(result_data),
            ContentType='application/json'
        )
        
        # Update job status
        table.update_item(
            Key={'id': job_id, 'timestamp': timestamp},
            UpdateExpression='SET #status = :status, completed_at = :completed',
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={
                ':status': 'completed',
                ':completed': datetime.utcnow().isoformat()
            }
        )
        
        response = {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': True,
                'job_id': job_id,
                'message': 'Synthetic data generation completed',
                'result': result_data
            })
        }
        
    except Exception as e:
        # Update job status to failed
        table.update_item(
            Key={'id': job_id, 'timestamp': timestamp},
            UpdateExpression='SET #status = :status, error = :error',
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={
                ':status': 'failed',
                ':error': str(e)
            }
        )
        
        response = {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': False,
                'job_id': job_id,
                'message': f'Generation failed: {str(e)}'
            })
        }
        
    return response
'''
        
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
            with zipfile.ZipFile(tmp_file.name, 'w') as zf:
                zf.writestr('lambda_function.py', code)
                
            with open(tmp_file.name, 'rb') as f:
                return f.read()
                
    def _get_lambda_layers(self) -> List[str]:
        """Get Lambda layers for data science libraries."""
        # These would be pre-created layers with pandas, numpy, etc.
        return []
        
    def _get_or_create_enhanced_lambda_role(self) -> str:
        """Create enhanced Lambda execution role."""
        iam = self._get_client('iam')
        role_name = f"{self.project_id}-enhanced-lambda-role"
        
        try:
            role = iam.get_role(RoleName=role_name)
            return role['Role']['Arn']
        except iam.exceptions.NoSuchEntityException:
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
                AssumeRolePolicyDocument=json.dumps(trust_policy)
            )
            
            # Attach managed policies
            policies = [
                'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
                'arn:aws:iam::aws:policy/AWSXRayDaemonWriteAccess',
                'arn:aws:iam::aws:policy/CloudWatchLambdaInsightsExecutionRolePolicy'
            ]
            
            for policy in policies:
                iam.attach_role_policy(RoleName=role_name, PolicyArn=policy)
                
            # Create custom policy for S3 and DynamoDB
            custom_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "s3:GetObject",
                            "s3:PutObject",
                            "s3:ListBucket"
                        ],
                        "Resource": [
                            f"arn:aws:s3:::{self.project_id}-*-synthdata",
                            f"arn:aws:s3:::{self.project_id}-*-synthdata/*"
                        ]
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "dynamodb:PutItem",
                            "dynamodb:GetItem",
                            "dynamodb:UpdateItem",
                            "dynamodb:Query",
                            "dynamodb:Scan"
                        ],
                        "Resource": f"arn:aws:dynamodb:{self.region}:*:table/{self.project_id}-*"
                    }
                ]
            }
            
            iam.put_role_policy(
                RoleName=role_name,
                PolicyName='SynthDataAccess',
                PolicyDocument=json.dumps(custom_policy)
            )
            
            return role['Role']['Arn']
            
    def _create_api_gateway(self, function_name: str, function_arn: str) -> Dict[str, Any]:
        """Create API Gateway for Lambda function."""
        apigateway = self._get_client('apigatewayv2')
        lambda_client = self._get_client('lambda')
        
        # Create HTTP API
        api_response = apigateway.create_api(
            Name=f"{self.project_id}-synthdata-api",
            ProtocolType='HTTP',
            Description='API for synthetic data generation',
            CorsConfiguration={
                'AllowOrigins': ['*'],
                'AllowMethods': ['GET', 'POST', 'OPTIONS'],
                'AllowHeaders': ['Content-Type', 'Authorization'],
                'MaxAge': 300
            }
        )
        
        api_id = api_response['ApiId']
        api_endpoint = api_response['ApiEndpoint']
        
        # Create Lambda integration
        integration_response = apigateway.create_integration(
            ApiId=api_id,
            IntegrationType='AWS_PROXY',
            IntegrationUri=function_arn,
            PayloadFormatVersion='2.0'
        )
        
        integration_id = integration_response['IntegrationId']
        
        # Create route
        apigateway.create_route(
            ApiId=api_id,
            RouteKey='POST /generate',
            Target=f'integrations/{integration_id}'
        )
        
        # Create deployment
        apigateway.create_deployment(
            ApiId=api_id,
            Description='Initial deployment'
        )
        
        # Add Lambda permission for API Gateway
        lambda_client.add_permission(
            FunctionName=function_name,
            StatementId=f'{api_id}-invoke',
            Action='lambda:InvokeFunction',
            Principal='apigateway.amazonaws.com',
            SourceArn=f'arn:aws:execute-api:{self.region}:*:{api_id}/*/*'
        )
        
        return {
            'ApiId': api_id,
            'ApiEndpoint': f'{api_endpoint}/generate'
        }
        
    def _create_event_rules(self, function_name: str, function_arn: str) -> List[str]:
        """Create EventBridge rules for Lambda."""
        events = self._get_client('events')
        lambda_client = self._get_client('lambda')
        
        rules = []
        
        # Create scheduled rule for periodic generation
        rule_name = f"{self.project_id}-scheduled-generation"
        
        events.put_rule(
            Name=rule_name,
            Description='Scheduled synthetic data generation',
            ScheduleExpression='rate(1 hour)',
            State='DISABLED'  # Start disabled, user can enable
        )
        
        # Add Lambda as target
        events.put_targets(
            Rule=rule_name,
            Targets=[{
                'Id': '1',
                'Arn': function_arn,
                'Input': json.dumps({
                    'source': 'scheduled',
                    'generator_type': 'sdv',
                    'num_samples': 100
                })
            }]
        )
        
        # Add permission for EventBridge
        lambda_client.add_permission(
            FunctionName=function_name,
            StatementId=f'{rule_name}-invoke',
            Action='lambda:InvokeFunction',
            Principal='events.amazonaws.com',
            SourceArn=f'arn:aws:events:{self.region}:*:rule/{rule_name}'
        )
        
        rules.append(rule_name)
        
        return rules
        
    def _configure_lambda_destinations(self, function_name: str):
        """Configure Lambda destinations for async invocations."""
        lambda_client = self._get_client('lambda')
        sns = self._get_client('sns')
        
        # Create SNS topics for success/failure
        success_topic = sns.create_topic(
            Name=f'{self.project_id}-lambda-success',
            Tags=[{'Key': 'Project', 'Value': self.project_id}]
        )
        
        failure_topic = sns.create_topic(
            Name=f'{self.project_id}-lambda-failure',
            Tags=[{'Key': 'Project', 'Value': self.project_id}]
        )
        
        # Configure destinations
        lambda_client.put_function_event_invoke_config(
            FunctionName=function_name,
            MaximumRetryAttempts=2,
            DestinationConfig={
                'OnSuccess': {'Destination': success_topic['TopicArn']},
                'OnFailure': {'Destination': failure_topic['TopicArn']}
            }
        )