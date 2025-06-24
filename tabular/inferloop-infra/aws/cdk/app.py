#!/usr/bin/env python3
"""AWS CDK application for Inferloop Synthetic Data infrastructure."""

import os
from aws_cdk import App, Environment, Tags
from stacks.main_stack import InferloopMainStack
from stacks.networking_stack import NetworkingStack
from stacks.security_stack import SecurityStack
from stacks.storage_stack import StorageStack
from stacks.compute_stack import ComputeStack
from stacks.monitoring_stack import MonitoringStack

app = App()

# Get context variables
project_id = app.node.try_get_context("project_id") or "inferloop-synthdata"
environment_name = app.node.try_get_context("environment") or "production"
aws_region = app.node.try_get_context("region") or "us-east-1"
aws_account = app.node.try_get_context("account") or os.environ.get("CDK_DEFAULT_ACCOUNT")

# Environment configuration
env = Environment(
    account=aws_account,
    region=aws_region
)

# Common tags
common_tags = {
    "Project": project_id,
    "Environment": environment_name,
    "ManagedBy": "CDK",
    "Application": "InferloopSyntheticData"
}

# Create stacks
networking_stack = NetworkingStack(
    app,
    f"{project_id}-{environment_name}-networking",
    project_id=project_id,
    environment_name=environment_name,
    env=env
)

security_stack = SecurityStack(
    app,
    f"{project_id}-{environment_name}-security",
    project_id=project_id,
    environment_name=environment_name,
    vpc=networking_stack.vpc,
    env=env
)

storage_stack = StorageStack(
    app,
    f"{project_id}-{environment_name}-storage",
    project_id=project_id,
    environment_name=environment_name,
    vpc=networking_stack.vpc,
    security_group=security_stack.app_security_group,
    env=env
)

compute_stack = ComputeStack(
    app,
    f"{project_id}-{environment_name}-compute",
    project_id=project_id,
    environment_name=environment_name,
    vpc=networking_stack.vpc,
    security_group=security_stack.app_security_group,
    load_balancer=networking_stack.load_balancer,
    target_group=networking_stack.target_group,
    s3_bucket=storage_stack.s3_bucket,
    dynamodb_table=storage_stack.dynamodb_table,
    task_role=security_stack.ecs_task_role,
    execution_role=security_stack.ecs_execution_role,
    env=env
)

monitoring_stack = MonitoringStack(
    app,
    f"{project_id}-{environment_name}-monitoring",
    project_id=project_id,
    environment_name=environment_name,
    auto_scaling_group=compute_stack.auto_scaling_group,
    load_balancer=networking_stack.load_balancer,
    ecs_cluster=compute_stack.ecs_cluster,
    ecs_service=compute_stack.ecs_service,
    env=env
)

# Main stack that combines all resources
main_stack = InferloopMainStack(
    app,
    f"{project_id}-{environment_name}-main",
    project_id=project_id,
    environment_name=environment_name,
    networking_stack=networking_stack,
    security_stack=security_stack,
    storage_stack=storage_stack,
    compute_stack=compute_stack,
    monitoring_stack=monitoring_stack,
    env=env
)

# Apply tags to all stacks
for stack in [networking_stack, security_stack, storage_stack, compute_stack, monitoring_stack, main_stack]:
    for key, value in common_tags.items():
        Tags.of(stack).add(key, value)

app.synth()