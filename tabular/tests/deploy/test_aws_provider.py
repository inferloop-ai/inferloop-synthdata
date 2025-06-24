"""Tests for AWS provider."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from deploy.aws.provider import AWSProvider
from deploy.aws.services import AWSAdvancedServices
from deploy.base import ResourceConfig, DeploymentResult


class TestAWSProvider:
    """Test AWSProvider class."""
    
    @pytest.fixture
    def mock_boto3_session(self):
        """Mock boto3 session."""
        with patch('deploy.aws.provider.boto3.Session') as mock_session:
            mock_instance = MagicMock()
            mock_session.return_value = mock_instance
            
            # Mock EC2 client
            mock_ec2 = MagicMock()
            mock_instance.client.return_value = mock_ec2
            
            yield mock_session, mock_ec2
    
    @pytest.fixture
    def aws_provider(self, mock_boto3_session):
        """Create AWS provider with mocked dependencies."""
        mock_session, mock_ec2 = mock_boto3_session
        return AWSProvider("us-east-1")
    
    def test_aws_provider_initialization(self, aws_provider):
        """Test AWS provider initialization."""
        assert aws_provider.region == "us-east-1"
        assert aws_provider.provider_name == "aws"
        assert aws_provider.advanced_services is None
    
    @patch('deploy.aws.provider.subprocess.run')
    def test_authenticate_success(self, mock_run, aws_provider):
        """Test successful authentication."""
        # Mock successful AWS CLI command
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = '{"Account": "123456789012"}'
        
        result = aws_provider.authenticate()
        assert result is True
        assert aws_provider.advanced_services is not None
    
    @patch('deploy.aws.provider.subprocess.run')
    def test_authenticate_failure(self, mock_run, aws_provider):
        """Test authentication failure."""
        # Mock failed AWS CLI command
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "Unable to locate credentials"
        
        result = aws_provider.authenticate()
        assert result is False
        assert aws_provider.advanced_services is None
    
    def test_deploy_ec2_without_auth(self, aws_provider):
        """Test EC2 deployment without authentication."""
        config = ResourceConfig(
            compute={"instance_type": "t3.micro", "count": 1},
            metadata={"name": "test-instance"}
        )
        
        result = aws_provider.deploy_ec2(config)
        assert result.success is False
        assert "not authenticated" in result.message.lower()
    
    @patch('deploy.aws.provider.subprocess.run')
    def test_deploy_ec2_success(self, mock_run, aws_provider):
        """Test successful EC2 deployment."""
        # Mock authentication
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = '{"Account": "123456789012"}'
        aws_provider.authenticate()
        
        # Mock EC2 operations
        mock_ec2 = aws_provider.advanced_services.ec2_client
        mock_ec2.run_instances.return_value = {
            'Instances': [{
                'InstanceId': 'i-123456789',
                'State': {'Name': 'running'},
                'PublicIpAddress': '1.2.3.4'
            }]
        }
        
        config = ResourceConfig(
            compute={"instance_type": "t3.micro", "count": 1},
            metadata={"name": "test-instance"}
        )
        
        result = aws_provider.deploy_ec2(config)
        assert result.success is True
        assert result.resource_id == "i-123456789"
        assert "1.2.3.4" in result.endpoint
    
    def test_deploy_ecs_without_auth(self, aws_provider):
        """Test ECS deployment without authentication."""
        config = ResourceConfig(
            metadata={"name": "test-service", "image": "nginx:latest"}
        )
        
        result = aws_provider.deploy_ecs(config)
        assert result.success is False
        assert "not authenticated" in result.message.lower()
    
    @patch('deploy.aws.provider.subprocess.run')
    def test_deploy_ecs_success(self, mock_run, aws_provider):
        """Test successful ECS deployment."""
        # Mock authentication
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = '{"Account": "123456789012"}'
        aws_provider.authenticate()
        
        # Mock advanced services
        aws_provider.advanced_services.deploy_ecs_service = MagicMock()
        aws_provider.advanced_services.deploy_ecs_service.return_value = DeploymentResult()
        aws_provider.advanced_services.deploy_ecs_service.return_value.success = True
        
        config = ResourceConfig(
            metadata={"name": "test-service", "image": "nginx:latest"}
        )
        
        result = aws_provider.deploy_ecs(config)
        assert result.success is True
    
    def test_deploy_eks_without_auth(self, aws_provider):
        """Test EKS deployment without authentication."""
        config = ResourceConfig(
            metadata={"cluster_name": "test-cluster"}
        )
        
        result = aws_provider.deploy_eks(config)
        assert result.success is False
        assert "not authenticated" in result.message.lower()
    
    @patch('deploy.aws.provider.subprocess.run')
    def test_deploy_lambda_success(self, mock_run, aws_provider):
        """Test successful Lambda deployment."""
        # Mock authentication
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = '{"Account": "123456789012"}'
        aws_provider.authenticate()
        
        # Mock advanced services
        aws_provider.advanced_services.deploy_lambda_function = MagicMock()
        aws_provider.advanced_services.deploy_lambda_function.return_value = DeploymentResult()
        aws_provider.advanced_services.deploy_lambda_function.return_value.success = True
        
        config = ResourceConfig(
            metadata={
                "function_name": "test-function",
                "runtime": "python3.9",
                "handler": "index.handler"
            }
        )
        
        result = aws_provider.deploy_lambda(config)
        assert result.success is True
    
    @patch('deploy.aws.provider.subprocess.run')
    def test_deploy_s3_success(self, mock_run, aws_provider):
        """Test successful S3 deployment."""
        # Mock authentication
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = '{"Account": "123456789012"}'
        aws_provider.authenticate()
        
        # Mock S3 operations
        mock_s3 = aws_provider.advanced_services.s3_client
        mock_s3.create_bucket.return_value = {}
        mock_s3.put_bucket_versioning.return_value = {}
        
        config = ResourceConfig(
            storage={"bucket_name": "test-bucket", "versioning": True}
        )
        
        result = aws_provider.deploy_s3(config)
        assert result.success is True
        assert result.resource_id == "test-bucket"
    
    @patch('deploy.aws.provider.subprocess.run')
    def test_deploy_rds_success(self, mock_run, aws_provider):
        """Test successful RDS deployment."""
        # Mock authentication
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = '{"Account": "123456789012"}'
        aws_provider.authenticate()
        
        # Mock advanced services
        aws_provider.advanced_services.deploy_rds_instance = MagicMock()
        aws_provider.advanced_services.deploy_rds_instance.return_value = DeploymentResult()
        aws_provider.advanced_services.deploy_rds_instance.return_value.success = True
        
        config = ResourceConfig(
            metadata={
                "db_name": "test-db",
                "engine": "postgres",
                "instance_class": "db.t3.micro"
            }
        )
        
        result = aws_provider.deploy_rds(config)
        assert result.success is True
    
    @patch('deploy.aws.provider.subprocess.run')
    def test_deploy_dynamodb_success(self, mock_run, aws_provider):
        """Test successful DynamoDB deployment."""
        # Mock authentication
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = '{"Account": "123456789012"}'
        aws_provider.authenticate()
        
        # Mock advanced services
        aws_provider.advanced_services.deploy_dynamodb_table = MagicMock()
        aws_provider.advanced_services.deploy_dynamodb_table.return_value = DeploymentResult()
        aws_provider.advanced_services.deploy_dynamodb_table.return_value.success = True
        
        config = ResourceConfig(
            metadata={
                "table_name": "test-table",
                "hash_key": "id",
                "hash_key_type": "S"
            }
        )
        
        result = aws_provider.deploy_dynamodb(config)
        assert result.success is True
    
    @patch('deploy.aws.provider.subprocess.run')
    def test_estimate_costs(self, mock_run, aws_provider):
        """Test cost estimation."""
        # Mock authentication
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = '{"Account": "123456789012"}'
        aws_provider.authenticate()
        
        resources = [
            {
                "type": "compute",
                "service_type": "ec2",
                "instance_type": "t3.micro",
                "hours_per_month": 730
            },
            {
                "type": "storage", 
                "service_type": "s3",
                "size_gb": 100,
                "storage_class": "standard"
            }
        ]
        
        costs = aws_provider.estimate_costs(resources)
        
        assert isinstance(costs, dict)
        assert "compute_cost" in costs
        assert "storage_cost" in costs
        assert "total_cost" in costs
        assert costs["total_cost"] >= 0
    
    def test_list_deployments_without_auth(self, aws_provider):
        """Test listing deployments without authentication."""
        deployments = aws_provider.list_deployments()
        assert deployments == []
    
    @patch('deploy.aws.provider.subprocess.run')
    def test_list_deployments_success(self, mock_run, aws_provider):
        """Test successful deployment listing."""
        # Mock authentication
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = '{"Account": "123456789012"}'
        aws_provider.authenticate()
        
        # Mock EC2 describe_instances
        mock_ec2 = aws_provider.advanced_services.ec2_client
        mock_ec2.describe_instances.return_value = {
            'Reservations': [{
                'Instances': [{
                    'InstanceId': 'i-123456789',
                    'State': {'Name': 'running'},
                    'InstanceType': 't3.micro',
                    'LaunchTime': '2024-01-01T00:00:00Z',
                    'Tags': [{'Key': 'Name', 'Value': 'test-instance'}]
                }]
            }]
        }
        
        deployments = aws_provider.list_deployments()
        
        assert len(deployments) == 1
        assert deployments[0]["instance_id"] == "i-123456789"
        assert deployments[0]["state"] == "running"
        assert deployments[0]["instance_type"] == "t3.micro"
    
    @patch('deploy.aws.provider.subprocess.run')
    def test_delete_deployment_success(self, mock_run, aws_provider):
        """Test successful deployment deletion."""
        # Mock authentication
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = '{"Account": "123456789012"}'
        aws_provider.authenticate()
        
        # Mock EC2 terminate_instances
        mock_ec2 = aws_provider.advanced_services.ec2_client
        mock_ec2.terminate_instances.return_value = {
            'TerminatingInstances': [{
                'InstanceId': 'i-123456789',
                'CurrentState': {'Name': 'shutting-down'}
            }]
        }
        
        result = aws_provider.delete_deployment("i-123456789")
        assert result is True
    
    def test_delete_deployment_without_auth(self, aws_provider):
        """Test deployment deletion without authentication."""
        result = aws_provider.delete_deployment("i-123456789")
        assert result is False
    
    @patch('deploy.aws.provider.subprocess.run')
    def test_validate_success(self, mock_run, aws_provider):
        """Test successful validation."""
        # Mock AWS CLI commands
        mock_run.side_effect = [
            # aws --version
            MagicMock(returncode=0, stdout="aws-cli/2.0.0"),
            # aws sts get-caller-identity
            MagicMock(returncode=0, stdout='{"Account": "123456789012"}'),
            # aws ec2 describe-regions
            MagicMock(returncode=0, stdout='{"Regions": [{"RegionName": "us-east-1"}]}')
        ]
        
        is_valid, issues = aws_provider.validate()
        
        assert is_valid is True
        assert len(issues) == 0
    
    @patch('deploy.aws.provider.subprocess.run')
    def test_validate_no_aws_cli(self, mock_run, aws_provider):
        """Test validation with missing AWS CLI."""
        # Mock missing AWS CLI
        mock_run.side_effect = FileNotFoundError()
        
        is_valid, issues = aws_provider.validate()
        
        assert is_valid is False
        assert "AWS CLI not installed" in issues
    
    @patch('deploy.aws.provider.subprocess.run')
    def test_validate_no_credentials(self, mock_run, aws_provider):
        """Test validation with missing credentials."""
        # Mock AWS CLI available but no credentials
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="aws-cli/2.0.0"),  # aws --version
            MagicMock(returncode=1, stderr="Unable to locate credentials")  # aws sts get-caller-identity
        ]
        
        is_valid, issues = aws_provider.validate()
        
        assert is_valid is False
        assert "AWS credentials not configured" in issues


class TestAWSAdvancedServices:
    """Test AWSAdvancedServices class."""
    
    @pytest.fixture
    def mock_clients(self):
        """Mock AWS service clients."""
        with patch('deploy.aws.services.boto3.Session') as mock_session:
            mock_instance = MagicMock()
            mock_session.return_value = mock_instance
            
            # Mock all service clients
            mock_clients = {
                'ecs': MagicMock(),
                'eks': MagicMock(),
                'lambda': MagicMock(),
                'rds': MagicMock(),
                'dynamodb': MagicMock(),
                'cloudformation': MagicMock(),
                'application-autoscaling': MagicMock()
            }
            
            mock_instance.client.side_effect = lambda service: mock_clients[service]
            
            yield mock_clients
    
    @pytest.fixture
    def aws_services(self, mock_clients):
        """Create AWSAdvancedServices with mocked clients."""
        with patch('deploy.aws.services.boto3.Session'):
            return AWSAdvancedServices("us-east-1")
    
    def test_deploy_ecs_service_success(self, aws_services, mock_clients):
        """Test successful ECS service deployment."""
        mock_ecs = mock_clients['ecs']
        
        # Mock ECS responses
        mock_ecs.create_cluster.return_value = {
            'cluster': {'clusterArn': 'arn:aws:ecs:us-east-1:123456789012:cluster/test-cluster'}
        }
        mock_ecs.register_task_definition.return_value = {
            'taskDefinition': {'taskDefinitionArn': 'arn:aws:ecs:us-east-1:123456789012:task-definition/test-task:1'}
        }
        mock_ecs.create_service.return_value = {
            'service': {'serviceArn': 'arn:aws:ecs:us-east-1:123456789012:service/test-service'}
        }
        
        config = ResourceConfig(
            metadata={
                "name": "test-service",
                "image": "nginx:latest",
                "cluster_name": "test-cluster"
            },
            compute={"cpu": "256", "memory": "512"}
        )
        
        result = aws_services.deploy_ecs_service(config)
        
        assert result.success is True
        assert "test-service" in result.resource_id
        
        # Verify ECS calls
        mock_ecs.create_cluster.assert_called_once()
        mock_ecs.register_task_definition.assert_called_once()
        mock_ecs.create_service.assert_called_once()
    
    def test_deploy_eks_cluster_success(self, aws_services, mock_clients):
        """Test successful EKS cluster deployment."""
        mock_eks = mock_clients['eks']
        
        # Mock EKS responses
        mock_eks.create_cluster.return_value = {
            'cluster': {
                'name': 'test-cluster',
                'status': 'CREATING',
                'endpoint': 'https://test-cluster.eks.us-east-1.amazonaws.com'
            }
        }
        
        config = ResourceConfig(
            metadata={"cluster_name": "test-cluster"},
            compute={"node_count": 2, "node_type": "t3.medium"}
        )
        
        result = aws_services.deploy_eks_cluster(config)
        
        assert result.success is True
        assert result.resource_id == "test-cluster"
        assert "eks.us-east-1.amazonaws.com" in result.endpoint
        
        # Verify EKS calls
        mock_eks.create_cluster.assert_called_once()
    
    def test_deploy_lambda_function_success(self, aws_services, mock_clients):
        """Test successful Lambda function deployment."""
        mock_lambda = mock_clients['lambda']
        
        # Mock Lambda responses
        mock_lambda.create_function.return_value = {
            'FunctionArn': 'arn:aws:lambda:us-east-1:123456789012:function:test-function',
            'FunctionName': 'test-function'
        }
        
        config = ResourceConfig(
            metadata={
                "function_name": "test-function",
                "runtime": "python3.9",
                "handler": "index.handler",
                "zip_file": b"dummy code"
            }
        )
        
        result = aws_services.deploy_lambda_function(config)
        
        assert result.success is True
        assert result.resource_id == "test-function"
        
        # Verify Lambda calls
        mock_lambda.create_function.assert_called_once()
    
    def test_deploy_rds_instance_success(self, aws_services, mock_clients):
        """Test successful RDS instance deployment."""
        mock_rds = mock_clients['rds']
        
        # Mock RDS responses
        mock_rds.create_db_instance.return_value = {
            'DBInstance': {
                'DBInstanceIdentifier': 'test-db',
                'DBInstanceStatus': 'creating',
                'Endpoint': {
                    'Address': 'test-db.cluster-xyz.us-east-1.rds.amazonaws.com',
                    'Port': 5432
                }
            }
        }
        
        config = ResourceConfig(
            metadata={
                "db_name": "test-db",
                "engine": "postgres",
                "instance_class": "db.t3.micro",
                "username": "admin",
                "password": "password123"
            },
            storage={"size": 20}
        )
        
        result = aws_services.deploy_rds_instance(config)
        
        assert result.success is True
        assert result.resource_id == "test-db"
        assert "rds.amazonaws.com" in result.endpoint
        
        # Verify RDS calls
        mock_rds.create_db_instance.assert_called_once()
    
    def test_deploy_dynamodb_table_success(self, aws_services, mock_clients):
        """Test successful DynamoDB table deployment."""
        mock_dynamodb = mock_clients['dynamodb']
        
        # Mock DynamoDB responses
        mock_dynamodb.create_table.return_value = {
            'TableDescription': {
                'TableName': 'test-table',
                'TableStatus': 'CREATING',
                'TableArn': 'arn:aws:dynamodb:us-east-1:123456789012:table/test-table'
            }
        }
        
        config = ResourceConfig(
            metadata={
                "table_name": "test-table",
                "hash_key": "id",
                "hash_key_type": "S"
            }
        )
        
        result = aws_services.deploy_dynamodb_table(config)
        
        assert result.success is True
        assert result.resource_id == "test-table"
        
        # Verify DynamoDB calls
        mock_dynamodb.create_table.assert_called_once()