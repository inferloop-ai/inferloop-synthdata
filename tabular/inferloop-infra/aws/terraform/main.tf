terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }
}

provider "aws" {
  region = var.region
  
  default_tags {
    tags = local.common_tags
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

locals {
  azs = length(var.availability_zones) > 0 ? var.availability_zones : slice(data.aws_availability_zones.available.names, 0, 3)
}

# VPC Module
module "vpc" {
  source = "./modules/vpc"
  
  name               = local.resource_prefix
  cidr               = var.vpc_cidr
  azs                = local.azs
  enable_nat_gateway = var.enable_nat_gateway
  enable_endpoints   = var.enable_vpc_endpoints
  tags               = local.common_tags
}

# Security Module
module "security" {
  source = "./modules/security"
  
  name               = local.resource_prefix
  vpc_id             = module.vpc.vpc_id
  allowed_ip_ranges  = var.allowed_ip_ranges
  enable_encryption  = var.enable_encryption
  tags               = local.common_tags
}

# Storage Module
module "storage" {
  source = "./modules/storage"
  
  name               = local.resource_prefix
  enable_encryption  = var.enable_encryption
  enable_versioning  = var.enable_backups
  retention_days     = var.backup_retention_days
  kms_key_id         = module.security.kms_key_id
  tags               = local.common_tags
}

# Database Module (optional)
module "database" {
  count  = var.database_engine != "" ? 1 : 0
  source = "./modules/database"
  
  name                = local.resource_prefix
  engine              = var.database_engine
  engine_version      = var.database_version
  instance_class      = var.database_instance_class
  storage_gb          = var.database_storage_gb
  vpc_id              = module.vpc.vpc_id
  subnet_ids          = module.vpc.private_subnet_ids
  security_group_ids  = [module.security.database_security_group_id]
  kms_key_id          = module.security.kms_key_id
  backup_retention    = var.backup_retention_days
  tags                = local.common_tags
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "${local.resource_prefix}-cluster"
  
  setting {
    name  = "containerInsights"
    value = var.enable_monitoring ? "enabled" : "disabled"
  }
  
  configuration {
    execute_command_configuration {
      kms_key_id = module.security.kms_key_id
      logging    = "OVERRIDE"
      
      log_configuration {
        cloud_watch_encryption_enabled = true
        cloud_watch_log_group_name     = aws_cloudwatch_log_group.ecs_exec.name
      }
    }
  }
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${local.resource_prefix}"
  retention_in_days = var.log_retention_days
  kms_key_id        = var.enable_encryption ? module.security.kms_key_arn : null
}

resource "aws_cloudwatch_log_group" "ecs_exec" {
  name              = "/ecs/${local.resource_prefix}/exec"
  retention_in_days = var.log_retention_days
  kms_key_id        = var.enable_encryption ? module.security.kms_key_arn : null
}

# Application Load Balancer
module "alb" {
  source = "./modules/alb"
  
  name               = local.resource_prefix
  vpc_id             = module.vpc.vpc_id
  subnet_ids         = module.vpc.public_subnet_ids
  security_group_ids = [module.security.alb_security_group_id]
  certificate_arn    = var.ssl_certificate_arn
  enable_waf         = var.enable_waf
  tags               = local.common_tags
}

# ECS Service
module "ecs_service" {
  source = "./modules/ecs_service"
  
  name                   = local.resource_prefix
  cluster_id             = aws_ecs_cluster.main.id
  vpc_id                 = module.vpc.vpc_id
  subnet_ids             = module.vpc.private_subnet_ids
  security_group_ids     = [module.security.app_security_group_id]
  target_group_arn       = module.alb.target_group_arn
  container_image        = var.container_image
  container_cpu          = var.container_cpu
  container_memory       = var.container_memory
  desired_count          = var.min_instances
  min_count              = var.min_instances
  max_count              = var.max_instances
  enable_auto_scaling    = var.enable_auto_scaling
  target_cpu_utilization = var.target_cpu_utilization
  task_role_arn          = module.security.ecs_task_role_arn
  execution_role_arn     = module.security.ecs_execution_role_arn
  log_group_name         = aws_cloudwatch_log_group.app.name
  environment_variables = {
    ENVIRONMENT = var.environment
    REGION      = var.region
    S3_BUCKET   = module.storage.bucket_name
  }
  secrets = {
    DATABASE_URL = var.database_engine != "" ? module.database[0].connection_secret_arn : ""
  }
  tags = local.common_tags
}

# Monitoring
module "monitoring" {
  source = "./modules/monitoring"
  
  name               = local.resource_prefix
  cluster_name       = aws_ecs_cluster.main.name
  service_name       = module.ecs_service.service_name
  alb_arn            = module.alb.alb_arn
  target_group_arn   = module.alb.target_group_arn
  log_group_name     = aws_cloudwatch_log_group.app.name
  alert_email        = var.alert_email
  enable_monitoring  = var.enable_monitoring
  tags               = local.common_tags
}

# Cost Management
resource "aws_budgets_budget" "monthly" {
  count = var.monthly_budget_usd > 0 ? 1 : 0
  
  name         = "${local.resource_prefix}-monthly-budget"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"
  
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = var.alert_email != "" ? [var.alert_email] : []
  }
  
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = var.alert_email != "" ? [var.alert_email] : []
  }
}

# CloudFront Distribution (optional)
module "cloudfront" {
  count  = var.domain_name != "" ? 1 : 0
  source = "./modules/cloudfront"
  
  name               = local.resource_prefix
  domain_names       = [var.domain_name]
  origin_domain      = module.alb.dns_name
  certificate_arn    = var.ssl_certificate_arn
  enable_waf         = var.enable_waf
  tags               = local.common_tags
}

# Auto-shutdown for non-production environments
module "auto_shutdown" {
  count  = var.environment != "prod" ? 1 : 0
  source = "./modules/auto_shutdown"
  
  name               = local.resource_prefix
  cluster_name       = aws_ecs_cluster.main.name
  service_name       = module.ecs_service.service_name
  schedule           = "cron(0 20 * * ? *)"  # 8 PM daily
  tags               = local.common_tags
}