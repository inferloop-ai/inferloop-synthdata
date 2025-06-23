"""
AWS Provider implementation
"""

import boto3
from typing import Dict, Any, List, Optional
import logging
from botocore.exceptions import ClientError, NoCredentialsError

from ...common.abstractions.base import BaseProvider


logger = logging.getLogger(__name__)


class AWSProvider(BaseProvider):
    """AWS cloud provider implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.access_key_id = config.get('access_key_id')
        self.secret_access_key = config.get('secret_access_key')
        self.region = config.get('region', 'us-east-1')
        self.session_token = config.get('session_token')
        self.profile = config.get('profile')
        
        self._session = None
        self._clients = {}
    
    def get_provider_name(self) -> str:
        """Get the provider name"""
        return "aws"
    
    async def authenticate(self) -> bool:
        """Authenticate with AWS"""
        try:
            # Create session
            if self.profile:
                self._session = boto3.Session(profile_name=self.profile)
            else:
                self._session = boto3.Session(
                    aws_access_key_id=self.access_key_id,
                    aws_secret_access_key=self.secret_access_key,
                    aws_session_token=self.session_token,
                    region_name=self.region
                )
            
            # Test authentication by calling STS
            sts = self._get_client('sts')
            sts.get_caller_identity()
            
            logger.info("Successfully authenticated with AWS")
            return True
            
        except NoCredentialsError:
            logger.error("No AWS credentials found")
            return False
        except Exception as e:
            logger.error(f"Failed to authenticate with AWS: {str(e)}")
            return False
    
    async def validate_credentials(self) -> bool:
        """Validate AWS credentials"""
        try:
            sts = self._get_client('sts')
            identity = sts.get_caller_identity()
            
            logger.info(f"Validated AWS credentials for account: {identity['Account']}")
            return True
            
        except ClientError as e:
            logger.error(f"Invalid AWS credentials: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Failed to validate credentials: {str(e)}")
            return False
    
    def get_regions(self) -> List[str]:
        """Get available AWS regions"""
        try:
            ec2 = self._get_client('ec2')
            response = ec2.describe_regions()
            return [r['RegionName'] for r in response['Regions']]
        except Exception as e:
            logger.error(f"Failed to get regions: {str(e)}")
            return []
    
    def get_capabilities(self) -> Dict[str, List[str]]:
        """Get AWS provider capabilities"""
        return {
            'compute': ['ec2', 'ecs', 'fargate', 'lambda', 'batch', 'eks'],
            'storage': ['s3', 'ebs', 'efs', 'fsx', 'storage-gateway'],
            'database': ['rds', 'dynamodb', 'elasticache', 'neptune', 'documentdb', 'redshift'],
            'networking': ['vpc', 'elb', 'alb', 'nlb', 'route53', 'cloudfront', 'api-gateway'],
            'security': ['iam', 'kms', 'secrets-manager', 'acm', 'waf', 'shield'],
            'monitoring': ['cloudwatch', 'cloudtrail', 'xray', 'systems-manager'],
            'analytics': ['athena', 'emr', 'kinesis', 'glue', 'quicksight'],
            'ml': ['sagemaker', 'comprehend', 'rekognition', 'polly', 'transcribe'],
            'integration': ['sqs', 'sns', 'eventbridge', 'step-functions', 'mq']
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform AWS health check"""
        health = {
            'healthy': True,
            'provider': 'aws',
            'region': self.region,
            'checks': {}
        }
        
        # Check STS
        try:
            sts = self._get_client('sts')
            identity = sts.get_caller_identity()
            health['checks']['authentication'] = {
                'status': 'healthy',
                'account_id': identity['Account'],
                'user_arn': identity['Arn']
            }
        except Exception as e:
            health['healthy'] = False
            health['checks']['authentication'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        # Check EC2
        try:
            ec2 = self._get_client('ec2')
            ec2.describe_instances(MaxResults=5)
            health['checks']['ec2'] = {'status': 'healthy'}
        except Exception as e:
            health['checks']['ec2'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        # Check S3
        try:
            s3 = self._get_client('s3')
            s3.list_buckets()
            health['checks']['s3'] = {'status': 'healthy'}
        except Exception as e:
            health['checks']['s3'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        return health
    
    def _get_client(self, service: str, region: Optional[str] = None):
        """Get or create boto3 client for service"""
        region = region or self.region
        key = f"{service}:{region}"
        
        if key not in self._clients:
            self._clients[key] = self._session.client(service, region_name=region)
        
        return self._clients[key]
    
    def _get_resource(self, service: str, region: Optional[str] = None):
        """Get or create boto3 resource for service"""
        region = region or self.region
        return self._session.resource(service, region_name=region)
    
    # Resource managers
    def get_compute_manager(self):
        """Get compute resource manager"""
        from .compute import AWSEC2
        return AWSEC2(self)
    
    def get_container_manager(self):
        """Get container resource manager"""
        from .compute import AWSContainer
        return AWSContainer(self)
    
    def get_serverless_manager(self):
        """Get serverless resource manager"""
        from .compute import AWSLambda
        return AWSLambda(self)
    
    def get_storage_manager(self):
        """Get storage resource manager"""
        from .storage import AWSS3
        return AWSS3(self)
    
    def get_network_manager(self):
        """Get network resource manager"""
        from .networking import AWSVPC
        return AWSVPC(self)
    
    def get_loadbalancer_manager(self):
        """Get load balancer resource manager"""
        from .networking import AWSLoadBalancer
        return AWSLoadBalancer(self)
    
    def get_database_manager(self):
        """Get database resource manager"""
        from .database import AWSRDS
        return AWSRDS(self)
    
    def get_cache_manager(self):
        """Get cache resource manager"""
        from .database import AWSElastiCache
        return AWSElastiCache(self)
    
    def get_security_manager(self):
        """Get security resource manager"""
        from .security import AWSIAM
        return AWSIAM(self)
    
    def get_monitoring_manager(self):
        """Get monitoring resource manager"""
        from .monitoring import AWSCloudWatch
        return AWSCloudWatch(self)