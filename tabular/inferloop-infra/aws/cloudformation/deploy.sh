#!/bin/bash
# AWS CloudFormation deployment script for Inferloop Synthetic Data

set -e

# Default values
PROJECT_ID="inferloop-synthdata"
ENVIRONMENT="production"
REGION="us-east-1"
STACK_NAME="${PROJECT_ID}-${ENVIRONMENT}-stack"
SNS_EMAIL="alerts@example.com"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --project-id)
            PROJECT_ID="$2"
            shift 2
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --email)
            SNS_EMAIL="$2"
            shift 2
            ;;
        --stack-name)
            STACK_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --project-id     Project identifier (default: inferloop-synthdata)"
            echo "  --environment    Environment (default: production)"
            echo "  --region         AWS region (default: us-east-1)"
            echo "  --email          Email for alerts (default: alerts@example.com)"
            echo "  --stack-name     CloudFormation stack name"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set AWS region
export AWS_DEFAULT_REGION=$REGION

# Upload templates to S3 (required for nested stacks)
BUCKET_NAME="${PROJECT_ID}-cfn-templates-${REGION}"

echo "Creating S3 bucket for templates..."
aws s3 mb "s3://${BUCKET_NAME}" --region $REGION 2>/dev/null || true

echo "Uploading templates to S3..."
aws s3 sync . "s3://${BUCKET_NAME}/templates/" \
    --exclude "*" \
    --include "*.yaml" \
    --delete

# Get S3 template URLs
TEMPLATE_URL="https://${BUCKET_NAME}.s3.${REGION}.amazonaws.com/templates"

# Deploy the main stack
echo "Deploying CloudFormation stack: ${STACK_NAME}"

aws cloudformation deploy \
    --stack-name "${STACK_NAME}" \
    --template-body file://main-stack.yaml \
    --parameter-overrides \
        ProjectId="${PROJECT_ID}" \
        Environment="${ENVIRONMENT}" \
        InstanceType="t3.medium" \
        MinInstances="1" \
        MaxInstances="5" \
        DesiredCapacity="2" \
        ContainerImage="inferloop/synthdata:latest" \
        DiskSize="100" \
        EnableMonitoring="true" \
    --capabilities CAPABILITY_NAMED_IAM \
    --tags \
        Project="${PROJECT_ID}" \
        Environment="${ENVIRONMENT}" \
        ManagedBy="CloudFormation" \
    --region $REGION

# Wait for stack to complete
echo "Waiting for stack deployment to complete..."
aws cloudformation wait stack-create-complete \
    --stack-name "${STACK_NAME}" \
    --region $REGION

# Get stack outputs
echo "Getting stack outputs..."
aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --query 'Stacks[0].Outputs' \
    --output table \
    --region $REGION

echo "Deployment completed successfully!"

# Display application URL
APP_URL=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --query 'Stacks[0].Outputs[?OutputKey==`ApplicationURL`].OutputValue' \
    --output text \
    --region $REGION)

echo ""
echo "Application URL: ${APP_URL}"
echo "CloudFormation Console: https://console.aws.amazon.com/cloudformation/home?region=${REGION}#/stacks"