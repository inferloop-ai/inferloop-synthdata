"""Main CDK stack that orchestrates all other stacks."""

from aws_cdk import (
    Stack,
    CfnOutput,
)
from constructs import Construct


class InferloopMainStack(Stack):
    """Main stack that combines all infrastructure components."""
    
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        project_id: str,
        environment_name: str,
        networking_stack,
        security_stack,
        storage_stack,
        compute_stack,
        monitoring_stack,
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        self.project_id = project_id
        self.environment_name = environment_name
        
        # Stack dependencies
        self.add_dependency(security_stack)
        self.add_dependency(storage_stack)
        self.add_dependency(compute_stack)
        self.add_dependency(monitoring_stack)
        
        # Outputs
        CfnOutput(
            self,
            "VpcId",
            value=networking_stack.vpc.vpc_id,
            description="VPC ID",
            export_name=f"{self.stack_name}-VpcId"
        )
        
        CfnOutput(
            self,
            "LoadBalancerDNS",
            value=networking_stack.load_balancer.load_balancer_dns_name,
            description="Load Balancer DNS name",
            export_name=f"{self.stack_name}-LoadBalancerDNS"
        )
        
        CfnOutput(
            self,
            "ApplicationURL",
            value=f"http://{networking_stack.load_balancer.load_balancer_dns_name}",
            description="Application URL",
            export_name=f"{self.stack_name}-ApplicationURL"
        )
        
        CfnOutput(
            self,
            "S3BucketName",
            value=storage_stack.s3_bucket.bucket_name,
            description="S3 bucket name",
            export_name=f"{self.stack_name}-S3BucketName"
        )
        
        CfnOutput(
            self,
            "DynamoDBTableName",
            value=storage_stack.dynamodb_table.table_name,
            description="DynamoDB table name",
            export_name=f"{self.stack_name}-DynamoDBTableName"
        )
        
        CfnOutput(
            self,
            "ECSClusterName",
            value=compute_stack.ecs_cluster.cluster_name,
            description="ECS cluster name",
            export_name=f"{self.stack_name}-ECSClusterName"
        )
        
        CfnOutput(
            self,
            "DashboardURL",
            value=monitoring_stack.dashboard_url,
            description="CloudWatch Dashboard URL",
            export_name=f"{self.stack_name}-DashboardURL"
        )