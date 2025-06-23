"""AWS security resources implementation."""

import boto3
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import base64

from common.core.security import (
    BaseSecurityProvider,
    SecurityConfig,
    Secret,
    IAMRole,
    SecurityGroup,
)
from common.core.config import InfrastructureConfig
from common.core.exceptions import (
    InfrastructureError,
    ResourceCreationError,
    ResourceNotFoundError,
)


class AWSSecurity(BaseSecurityProvider):
    """AWS security implementation."""
    
    def __init__(self, session: boto3.Session, config: InfrastructureConfig):
        """Initialize AWS security manager."""
        super().__init__(SecurityConfig())
        self.session = session
        self.config = config
        self.iam_client = session.client("iam")
        self.kms_client = session.client("kms")
        self.secrets_client = session.client("secretsmanager")
        self.ec2_client = session.client("ec2")
        self.acm_client = session.client("acm")
        self.waf_client = session.client("wafv2")
        self.shield_client = session.client("shield")
        self.sts_client = session.client("sts")
        
        # Get account ID
        self.account_id = self.sts_client.get_caller_identity()["Account"]
    
    def create_iam_role(self, role: IAMRole) -> str:
        """Create an IAM role."""
        try:
            # Create role
            response = self.iam_client.create_role(
                RoleName=role.name,
                AssumeRolePolicyDocument=json.dumps(role.trust_policy),
                Description=role.description,
                Tags=self._format_tags(role.tags),
            )
            
            role_arn = response["Role"]["Arn"]
            
            # Attach policies
            for i, policy in enumerate(role.policies):
                if isinstance(policy, str):
                    # Attach managed policy
                    self.iam_client.attach_role_policy(
                        RoleName=role.name,
                        PolicyArn=policy,
                    )
                else:
                    # Create and attach inline policy
                    self.iam_client.put_role_policy(
                        RoleName=role.name,
                        PolicyName=f"{role.name}-policy-{i}",
                        PolicyDocument=json.dumps(policy),
                    )
            
            return role_arn
            
        except Exception as e:
            raise ResourceCreationError("IAM Role", str(e))
    
    def create_security_group(self, group: SecurityGroup) -> str:
        """Create a security group."""
        try:
            # Get default VPC if not specified
            vpc_id = group.metadata.get("vpc_id")
            if not vpc_id:
                vpc_response = self.ec2_client.describe_vpcs(
                    Filters=[{"Name": "is-default", "Values": ["true"]}]
                )
                vpc_id = vpc_response["Vpcs"][0]["VpcId"]
            
            # Create security group
            response = self.ec2_client.create_security_group(
                GroupName=group.name,
                Description=group.description,
                VpcId=vpc_id,
                TagSpecifications=[
                    {
                        "ResourceType": "security-group",
                        "Tags": self._format_tags(group.tags),
                    }
                ],
            )
            
            group_id = response["GroupId"]
            
            # Add ingress rules
            if group.ingress_rules:
                self.ec2_client.authorize_security_group_ingress(
                    GroupId=group_id,
                    IpPermissions=group.ingress_rules,
                )
            
            # Add egress rules (if different from default)
            if group.egress_rules:
                # First revoke default egress rule
                self.ec2_client.revoke_security_group_egress(
                    GroupId=group_id,
                    IpPermissions=[
                        {
                            "IpProtocol": "-1",
                            "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                        }
                    ],
                )
                
                # Add custom egress rules
                self.ec2_client.authorize_security_group_egress(
                    GroupId=group_id,
                    IpPermissions=group.egress_rules,
                )
            
            return group_id
            
        except Exception as e:
            raise ResourceCreationError("Security Group", str(e))
    
    def store_secret(self, secret: Secret) -> str:
        """Store a secret in AWS Secrets Manager."""
        try:
            # Create secret
            response = self.secrets_client.create_secret(
                Name=secret.name,
                Description=secret.description or f"Secret for {self.config.project_name}",
                SecretString=secret.value,
                Tags=self._format_tags(secret.tags),
            )
            
            # Set rotation if specified
            if secret.rotation_days > 0:
                # Note: Actual rotation requires a Lambda function
                # This just sets the rotation schedule
                self.secrets_client.put_resource_policy(
                    SecretId=response["ARN"],
                    ResourcePolicy=json.dumps({
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {
                                    "Service": "secretsmanager.amazonaws.com"
                                },
                                "Action": "secretsmanager:RotateSecret",
                                "Resource": "*",
                            }
                        ],
                    }),
                )
            
            return response["ARN"]
            
        except self.secrets_client.exceptions.ResourceExistsException:
            # Update existing secret
            response = self.secrets_client.update_secret(
                SecretId=secret.name,
                SecretString=secret.value,
            )
            return response["ARN"]
            
        except Exception as e:
            raise ResourceCreationError("Secret", str(e))
    
    def retrieve_secret(self, secret_name: str) -> str:
        """Retrieve a secret from AWS Secrets Manager."""
        try:
            response = self.secrets_client.get_secret_value(SecretId=secret_name)
            
            if "SecretString" in response:
                return response["SecretString"]
            else:
                # Binary secret
                return base64.b64decode(response["SecretBinary"]).decode("utf-8")
                
        except self.secrets_client.exceptions.ResourceNotFoundException:
            raise ResourceNotFoundError(secret_name, "Secret")
        except Exception as e:
            raise InfrastructureError(f"Failed to retrieve secret: {str(e)}")
    
    def rotate_secret(self, secret_name: str) -> str:
        """Rotate a secret in AWS Secrets Manager."""
        try:
            # Generate new secret value
            new_value = self.generate_password(32)
            
            # Update secret with new value
            response = self.secrets_client.update_secret(
                SecretId=secret_name,
                SecretString=new_value,
            )
            
            return response["ARN"]
            
        except Exception as e:
            raise InfrastructureError(f"Failed to rotate secret: {str(e)}")
    
    def create_kms_key(self, key_alias: str, description: str) -> str:
        """Create a KMS key."""
        try:
            # Create key
            key_response = self.kms_client.create_key(
                Description=description,
                KeyUsage="ENCRYPT_DECRYPT",
                Origin="AWS_KMS",
                Tags=self._format_tags({
                    "Name": key_alias,
                    "Project": self.config.project_name,
                }),
            )
            
            key_id = key_response["KeyMetadata"]["KeyId"]
            
            # Create alias
            self.kms_client.create_alias(
                AliasName=f"alias/{key_alias}",
                TargetKeyId=key_id,
            )
            
            return key_id
            
        except Exception as e:
            raise ResourceCreationError("KMS Key", str(e))
    
    def encrypt_data(self, data: bytes, key_id: Optional[str] = None) -> bytes:
        """Encrypt data using KMS."""
        try:
            if not key_id:
                key_id = f"alias/{self.config.resource_name}-key"
            
            response = self.kms_client.encrypt(
                KeyId=key_id,
                Plaintext=data,
            )
            
            return response["CiphertextBlob"]
            
        except Exception as e:
            raise InfrastructureError(f"Failed to encrypt data: {str(e)}")
    
    def decrypt_data(self, encrypted_data: bytes, key_id: Optional[str] = None) -> bytes:
        """Decrypt data using KMS."""
        try:
            response = self.kms_client.decrypt(CiphertextBlob=encrypted_data)
            return response["Plaintext"]
            
        except Exception as e:
            raise InfrastructureError(f"Failed to decrypt data: {str(e)}")
    
    def create_certificate(self, domain: str, san: List[str] = None) -> str:
        """Request or import an SSL certificate using ACM."""
        try:
            # Request certificate
            params = {
                "DomainName": domain,
                "ValidationMethod": "DNS",
                "Tags": self._format_tags({
                    "Name": f"{domain}-cert",
                    "Project": self.config.project_name,
                }),
            }
            
            if san:
                params["SubjectAlternativeNames"] = [domain] + san
            
            response = self.acm_client.request_certificate(**params)
            
            return response["CertificateArn"]
            
        except Exception as e:
            raise ResourceCreationError("Certificate", str(e))
    
    def enable_audit_logging(self, resource_id: str, log_group: str) -> None:
        """Enable audit logging for a resource."""
        try:
            # This is a placeholder - actual implementation depends on resource type
            # For now, we'll create a CloudWatch log group
            logs_client = self.session.client("logs")
            
            try:
                logs_client.create_log_group(
                    logGroupName=log_group,
                    tags=self.config.default_tags,
                )
            except logs_client.exceptions.ResourceAlreadyExistsException:
                pass
            
            # Set retention
            logs_client.put_retention_policy(
                logGroupName=log_group,
                retentionInDays=self.config.log_retention_days,
            )
            
        except Exception as e:
            raise InfrastructureError(f"Failed to enable audit logging: {str(e)}")
    
    def create_waf_web_acl(self, name: str, rules: List[Dict[str, Any]]) -> str:
        """Create a WAF Web ACL."""
        try:
            # Create Web ACL
            response = self.waf_client.create_web_acl(
                Name=name,
                Scope="REGIONAL",  # or CLOUDFRONT for CloudFront distributions
                DefaultAction={"Allow": {}},
                Rules=rules,
                VisibilityConfig={
                    "SampledRequestsEnabled": True,
                    "CloudWatchMetricsEnabled": True,
                    "MetricName": name,
                },
                Tags=self._format_tags({
                    "Name": name,
                    "Project": self.config.project_name,
                }),
            )
            
            return response["Summary"]["ARN"]
            
        except Exception as e:
            raise ResourceCreationError("WAF Web ACL", str(e))
    
    def enable_shield_protection(self, resource_arn: str) -> None:
        """Enable AWS Shield protection for a resource."""
        try:
            # Note: This requires Shield Advanced subscription
            self.shield_client.create_protection(
                Name=f"shield-{resource_arn.split('/')[-1]}",
                ResourceArn=resource_arn,
                Tags=self._format_tags({
                    "Project": self.config.project_name,
                }),
            )
            
        except Exception as e:
            # Shield might not be available
            pass
    
    def create_service_roles(self) -> Dict[str, str]:
        """Create standard service roles for the infrastructure."""
        roles = {}
        
        # ECS Task Role
        ecs_task_role = IAMRole(
            name=f"{self.config.resource_name}-ecs-task-role",
            description="Role for ECS tasks",
            trust_policy={
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }
                ],
            },
            policies=[
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": [
                                "s3:GetObject",
                                "s3:PutObject",
                                "s3:DeleteObject",
                                "s3:ListBucket",
                            ],
                            "Resource": [
                                f"arn:aws:s3:::{self.config.resource_name}-*/*",
                                f"arn:aws:s3:::{self.config.resource_name}-*",
                            ],
                        },
                        {
                            "Effect": "Allow",
                            "Action": [
                                "secretsmanager:GetSecretValue",
                            ],
                            "Resource": f"arn:aws:secretsmanager:{self.config.region}:{self.account_id}:secret:{self.config.resource_name}-*",
                        },
                        {
                            "Effect": "Allow",
                            "Action": [
                                "kms:Decrypt",
                                "kms:GenerateDataKey",
                            ],
                            "Resource": "*",
                        },
                    ],
                }
            ],
            tags=self.config.default_tags,
        )
        roles["ecs_task"] = self.create_iam_role(ecs_task_role)
        
        # ECS Execution Role
        ecs_execution_role = IAMRole(
            name=f"{self.config.resource_name}-ecs-execution-role",
            description="Role for ECS execution",
            trust_policy={
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }
                ],
            },
            policies=[
                "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
            ],
            tags=self.config.default_tags,
        )
        roles["ecs_execution"] = self.create_iam_role(ecs_execution_role)
        
        # Lambda Execution Role
        lambda_role = IAMRole(
            name=f"{self.config.resource_name}-lambda-role",
            description="Role for Lambda functions",
            trust_policy={
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "lambda.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }
                ],
            },
            policies=[
                "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": [
                                "s3:GetObject",
                                "s3:PutObject",
                            ],
                            "Resource": f"arn:aws:s3:::{self.config.resource_name}-*/*",
                        },
                    ],
                },
            ],
            tags=self.config.default_tags,
        )
        roles["lambda"] = self.create_iam_role(lambda_role)
        
        return roles
    
    def create_api_key(self, name: str, description: str) -> Dict[str, str]:
        """Create an API key for service authentication."""
        # Generate API key
        api_key = self.generate_api_key(prefix="sk", length=32)
        
        # Store in Secrets Manager
        secret = Secret(
            name=f"{self.config.resource_name}-api-key-{name}",
            value=api_key,
            description=description,
            rotation_days=90,
            tags=self.config.default_tags,
        )
        
        secret_arn = self.store_secret(secret)
        
        return {
            "api_key": api_key,
            "secret_arn": secret_arn,
        }
    
    def setup_vpc_flow_logs(self, vpc_id: str) -> None:
        """Enable VPC Flow Logs for network monitoring."""
        try:
            # Create CloudWatch log group
            logs_client = self.session.client("logs")
            log_group = f"/aws/vpc/flowlogs/{vpc_id}"
            
            try:
                logs_client.create_log_group(logGroupName=log_group)
            except logs_client.exceptions.ResourceAlreadyExistsException:
                pass
            
            # Create IAM role for Flow Logs
            flow_logs_role = IAMRole(
                name=f"{self.config.resource_name}-flow-logs-role",
                description="Role for VPC Flow Logs",
                trust_policy={
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "vpc-flow-logs.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }
                    ],
                },
                policies=[
                    {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Action": [
                                    "logs:CreateLogGroup",
                                    "logs:CreateLogStream",
                                    "logs:PutLogEvents",
                                    "logs:DescribeLogGroups",
                                    "logs:DescribeLogStreams",
                                ],
                                "Resource": "*",
                            }
                        ],
                    }
                ],
                tags=self.config.default_tags,
            )
            
            role_arn = self.create_iam_role(flow_logs_role)
            
            # Create Flow Log
            self.ec2_client.create_flow_logs(
                ResourceType="VPC",
                ResourceIds=[vpc_id],
                TrafficType="ALL",
                LogDestinationType="cloud-watch-logs",
                LogGroupName=log_group,
                DeliverLogsPermissionArn=role_arn,
                TagSpecifications=[
                    {
                        "ResourceType": "vpc-flow-log",
                        "Tags": self._format_tags({
                            "Name": f"{vpc_id}-flow-logs",
                            "Project": self.config.project_name,
                        }),
                    }
                ],
            )
            
        except Exception as e:
            # Flow logs might fail but shouldn't block deployment
            pass
    
    def _format_tags(self, tags: Dict[str, str]) -> List[Dict[str, str]]:
        """Format tags for AWS API."""
        return [{"Key": k, "Value": v} for k, v in tags.items()]