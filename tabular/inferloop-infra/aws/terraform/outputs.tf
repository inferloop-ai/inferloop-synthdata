output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "public_subnet_ids" {
  description = "IDs of public subnets"
  value       = module.vpc.public_subnet_ids
}

output "private_subnet_ids" {
  description = "IDs of private subnets"
  value       = module.vpc.private_subnet_ids
}

output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = module.alb.dns_name
}

output "alb_zone_id" {
  description = "Zone ID of the Application Load Balancer"
  value       = module.alb.zone_id
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "Name of the ECS service"
  value       = module.ecs_service.service_name
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket for data storage"
  value       = module.storage.bucket_name
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket"
  value       = module.storage.bucket_arn
}

output "cloudfront_distribution_id" {
  description = "ID of the CloudFront distribution"
  value       = var.domain_name != "" ? module.cloudfront[0].distribution_id : ""
}

output "cloudfront_domain_name" {
  description = "Domain name of the CloudFront distribution"
  value       = var.domain_name != "" ? module.cloudfront[0].domain_name : ""
}

output "database_endpoint" {
  description = "Database endpoint"
  value       = var.database_engine != "" ? module.database[0].endpoint : ""
  sensitive   = true
}

output "database_port" {
  description = "Database port"
  value       = var.database_engine != "" ? module.database[0].port : ""
}

output "kms_key_id" {
  description = "ID of the KMS key for encryption"
  value       = module.security.kms_key_id
}

output "kms_key_arn" {
  description = "ARN of the KMS key for encryption"
  value       = module.security.kms_key_arn
}

output "monitoring_dashboard_url" {
  description = "URL of the CloudWatch dashboard"
  value       = module.monitoring.dashboard_url
}

output "api_endpoint" {
  description = "API endpoint URL"
  value       = var.ssl_certificate_arn != "" ? "https://${module.alb.dns_name}" : "http://${module.alb.dns_name}"
}

output "region" {
  description = "AWS region"
  value       = var.region
}

output "environment" {
  description = "Environment name"
  value       = var.environment
}

output "resource_tags" {
  description = "Tags applied to all resources"
  value       = local.common_tags
}