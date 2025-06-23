variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "inferloop-synthdata"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = []
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "enable_nat_gateway" {
  description = "Enable NAT Gateway for private subnets"
  type        = bool
  default     = true
}

variable "enable_vpc_endpoints" {
  description = "Enable VPC endpoints for AWS services"
  type        = bool
  default     = true
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.medium"
}

variable "min_instances" {
  description = "Minimum number of instances"
  type        = number
  default     = 1
}

variable "max_instances" {
  description = "Maximum number of instances"
  type        = number
  default     = 10
}

variable "container_image" {
  description = "Container image for the application"
  type        = string
  default     = "inferloop/synthdata:latest"
}

variable "container_cpu" {
  description = "CPU units for container (1024 = 1 vCPU)"
  type        = string
  default     = "1024"
}

variable "container_memory" {
  description = "Memory for container in MB"
  type        = string
  default     = "2048"
}

variable "enable_monitoring" {
  description = "Enable CloudWatch monitoring"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch logs retention in days"
  type        = number
  default     = 7
}

variable "enable_backups" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Backup retention in days"
  type        = number
  default     = 7
}

variable "enable_encryption" {
  description = "Enable encryption at rest"
  type        = bool
  default     = true
}

variable "allowed_ip_ranges" {
  description = "List of allowed IP ranges for access"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "ssl_certificate_arn" {
  description = "ARN of SSL certificate for HTTPS"
  type        = string
  default     = ""
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = ""
}

variable "enable_waf" {
  description = "Enable Web Application Firewall"
  type        = bool
  default     = false
}

variable "enable_shield" {
  description = "Enable DDoS protection with AWS Shield"
  type        = bool
  default     = false
}

variable "database_engine" {
  description = "Database engine (postgres, mysql)"
  type        = string
  default     = "postgres"
}

variable "database_version" {
  description = "Database engine version"
  type        = string
  default     = "14"
}

variable "database_instance_class" {
  description = "Database instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "database_storage_gb" {
  description = "Database storage in GB"
  type        = number
  default     = 100
}

variable "enable_auto_scaling" {
  description = "Enable auto-scaling for compute resources"
  type        = bool
  default     = true
}

variable "target_cpu_utilization" {
  description = "Target CPU utilization for auto-scaling"
  type        = number
  default     = 70
}

variable "monthly_budget_usd" {
  description = "Monthly budget in USD for cost alerts"
  type        = number
  default     = 1000
}

variable "alert_email" {
  description = "Email address for alerts"
  type        = string
  default     = ""
}

variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}

locals {
  common_tags = merge(
    {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "Terraform"
      CreatedBy   = "inferloop-infra"
    },
    var.tags
  )
  
  resource_prefix = "${var.project_name}-${var.environment}"
}