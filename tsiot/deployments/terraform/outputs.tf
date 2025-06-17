# VPC Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = module.vpc.private_subnet_ids
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = module.vpc.public_subnet_ids
}

output "database_subnet_ids" {
  description = "IDs of the database subnets"
  value       = module.vpc.database_subnet_ids
}

# EKS Outputs
output "cluster_id" {
  description = "EKS cluster ID"
  value       = module.eks.cluster_id
}

output "cluster_arn" {
  description = "EKS cluster ARN"
  value       = module.eks.cluster_arn
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_version" {
  description = "The Kubernetes version for the EKS cluster"
  value       = module.eks.cluster_version
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "cluster_primary_security_group_id" {
  description = "Primary security group ID for the EKS cluster"
  value       = module.eks.cluster_primary_security_group_id
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
}

output "cluster_oidc_issuer_url" {
  description = "The URL on the EKS cluster OIDC Issuer"
  value       = module.eks.cluster_oidc_issuer_url
}

output "oidc_provider_arn" {
  description = "The ARN of the OIDC Provider if enabled"
  value       = module.eks.oidc_provider_arn
}

# Node Groups Outputs
output "eks_managed_node_groups" {
  description = "Map of EKS managed node group attributes"
  value       = module.eks.eks_managed_node_groups
}

# RDS Outputs
output "rds_instance_id" {
  description = "RDS instance ID"
  value       = module.rds.database_instance_id
}

output "rds_instance_arn" {
  description = "RDS instance ARN"
  value       = module.rds.database_instance_arn
}

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = module.rds.database_endpoint
  sensitive   = true
}

output "rds_port" {
  description = "RDS instance port"
  value       = module.rds.database_port
}

output "rds_database_name" {
  description = "RDS database name"
  value       = module.rds.database_name
}

output "rds_username" {
  description = "RDS database username"
  value       = module.rds.database_username
  sensitive   = true
}

# ElastiCache Outputs
output "redis_cluster_id" {
  description = "Redis cluster ID"
  value       = aws_elasticache_replication_group.redis.replication_group_id
}

output "redis_endpoint" {
  description = "Redis primary endpoint"
  value       = aws_elasticache_replication_group.redis.primary_endpoint_address
  sensitive   = true
}

output "redis_reader_endpoint" {
  description = "Redis reader endpoint"
  value       = aws_elasticache_replication_group.redis.reader_endpoint_address
  sensitive   = true
}

output "redis_port" {
  description = "Redis port"
  value       = aws_elasticache_replication_group.redis.port
}

# OpenSearch Outputs
output "opensearch_domain_id" {
  description = "OpenSearch domain ID"
  value       = aws_opensearch_domain.main.domain_id
}

output "opensearch_domain_arn" {
  description = "OpenSearch domain ARN"
  value       = aws_opensearch_domain.main.arn
}

output "opensearch_endpoint" {
  description = "OpenSearch domain endpoint"
  value       = aws_opensearch_domain.main.endpoint
  sensitive   = true
}

output "opensearch_dashboard_endpoint" {
  description = "OpenSearch dashboard endpoint"
  value       = aws_opensearch_domain.main.dashboard_endpoint
  sensitive   = true
}

# MSK Outputs
output "msk_cluster_arn" {
  description = "MSK cluster ARN"
  value       = aws_msk_cluster.main.arn
}

output "msk_cluster_name" {
  description = "MSK cluster name"
  value       = aws_msk_cluster.main.cluster_name
}

output "msk_bootstrap_brokers" {
  description = "MSK bootstrap brokers"
  value       = aws_msk_cluster.main.bootstrap_brokers_tls
  sensitive   = true
}

output "msk_bootstrap_brokers_sasl_scram" {
  description = "MSK bootstrap brokers for SASL/SCRAM"
  value       = aws_msk_cluster.main.bootstrap_brokers_sasl_scram
  sensitive   = true
}

output "msk_zookeeper_connect_string" {
  description = "MSK Zookeeper connection string"
  value       = aws_msk_cluster.main.zookeeper_connect_string
  sensitive   = true
}

# S3 Outputs
output "s3_data_lake_bucket_id" {
  description = "S3 data lake bucket ID"
  value       = aws_s3_bucket.data_lake.id
}

output "s3_data_lake_bucket_arn" {
  description = "S3 data lake bucket ARN"
  value       = aws_s3_bucket.data_lake.arn
}

output "s3_backup_bucket_id" {
  description = "S3 backup bucket ID"
  value       = aws_s3_bucket.backup.id
}

output "s3_backup_bucket_arn" {
  description = "S3 backup bucket ARN"
  value       = aws_s3_bucket.backup.arn
}

# IAM Outputs
output "tsiot_app_role_arn" {
  description = "TSIoT application role ARN"
  value       = aws_iam_role.tsiot_app_role.arn
}

output "load_balancer_controller_role_arn" {
  description = "Load balancer controller role ARN"
  value       = module.load_balancer_controller_irsa_role.iam_role_arn
}

# Secrets Manager Outputs
output "rds_credentials_secret_arn" {
  description = "RDS credentials secret ARN"
  value       = aws_secretsmanager_secret.rds_credentials.arn
}

output "opensearch_credentials_secret_arn" {
  description = "OpenSearch credentials secret ARN"
  value       = aws_secretsmanager_secret.opensearch_credentials.arn
}

# KMS Outputs
output "kms_key_id" {
  description = "KMS key ID"
  value       = aws_kms_key.tsiot.key_id
}

output "kms_key_arn" {
  description = "KMS key ARN"
  value       = aws_kms_key.tsiot.arn
}

output "kms_alias_arn" {
  description = "KMS alias ARN"
  value       = aws_kms_alias.tsiot.arn
}

# CloudWatch Outputs
output "application_log_group_name" {
  description = "Application CloudWatch log group name"
  value       = aws_cloudwatch_log_group.application.name
}

output "application_log_group_arn" {
  description = "Application CloudWatch log group ARN"
  value       = aws_cloudwatch_log_group.application.arn
}

# Monitoring Outputs (conditional)
output "monitoring_dashboard_url" {
  description = "Monitoring dashboard URL"
  value       = var.enable_monitoring ? module.monitoring[0].dashboard_url : null
}

output "prometheus_endpoint" {
  description = "Prometheus endpoint"
  value       = var.enable_monitoring ? module.monitoring[0].prometheus_endpoint : null
}

output "grafana_endpoint" {
  description = "Grafana endpoint"
  value       = var.enable_monitoring ? module.monitoring[0].grafana_endpoint : null
}

# Connection Information
output "kubectl_config_command" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}

output "connection_info" {
  description = "Connection information for all services"
  value = {
    cluster = {
      name     = module.eks.cluster_name
      endpoint = module.eks.cluster_endpoint
      region   = var.aws_region
    }
    database = {
      endpoint = module.rds.database_endpoint
      port     = module.rds.database_port
      name     = module.rds.database_name
    }
    redis = {
      endpoint = aws_elasticache_replication_group.redis.primary_endpoint_address
      port     = aws_elasticache_replication_group.redis.port
    }
    opensearch = {
      endpoint = aws_opensearch_domain.main.endpoint
    }
    kafka = {
      cluster_name = aws_msk_cluster.main.cluster_name
    }
    storage = {
      data_lake_bucket = aws_s3_bucket.data_lake.id
      backup_bucket    = aws_s3_bucket.backup.id
    }
  }
  sensitive = true
}

# Environment Configuration
output "environment_config" {
  description = "Environment configuration for applications"
  value = {
    environment          = var.environment
    project_name         = var.project_name
    region               = var.aws_region
    vpc_id              = module.vpc.vpc_id
    cluster_name        = module.eks.cluster_name
    rds_secret_arn      = aws_secretsmanager_secret.rds_credentials.arn
    opensearch_secret_arn = aws_secretsmanager_secret.opensearch_credentials.arn
    data_lake_bucket    = aws_s3_bucket.data_lake.id
    backup_bucket       = aws_s3_bucket.backup.id
    kms_key_id          = aws_kms_key.tsiot.key_id
    log_group_name      = aws_cloudwatch_log_group.application.name
  }
}