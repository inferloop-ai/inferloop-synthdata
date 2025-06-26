# AWS Database Infrastructure for TextNLP Platform
# Phase 3: Core Infrastructure Deployment

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

# Local variables
locals {
  cluster_name = "textnlp-postgres-cluster"
  
  tags = {
    Project     = "TextNLP"
    Environment = "production"
    Component   = "database"
    Platform    = "aws"
    Phase       = "3"
  }
}

# Random password for master database
resource "random_password" "master_password" {
  length  = 32
  special = true
}

# KMS key for database encryption
resource "aws_kms_key" "database_key" {
  description = "KMS key for TextNLP database encryption"
  
  tags = merge(local.tags, {
    Name = "textnlp-database-key"
  })
}

resource "aws_kms_alias" "database_key_alias" {
  name          = "alias/textnlp-database-key"
  target_key_id = aws_kms_key.database_key.key_id
}

# Secrets Manager secret for database credentials
resource "aws_secretsmanager_secret" "database_credentials" {
  name                    = "textnlp-database-credentials"
  description            = "Database credentials for TextNLP platform"
  recovery_window_in_days = 7
  kms_key_id             = aws_kms_key.database_key.arn
  
  tags = local.tags
}

resource "aws_secretsmanager_secret_version" "database_credentials" {
  secret_id = aws_secretsmanager_secret.database_credentials.id
  secret_string = jsonencode({
    username = "textnlp_admin"
    password = random_password.master_password.result
  })
}

# Database subnet group
resource "aws_db_subnet_group" "database_subnet_group" {
  name       = "textnlp-db-subnet-group"
  subnet_ids = data.aws_subnets.private.ids
  
  tags = merge(local.tags, {
    Name = "textnlp-db-subnet-group"
  })
}

# Database parameter group
resource "aws_rds_cluster_parameter_group" "database_cluster_params" {
  family = "aurora-postgresql15"
  name   = "textnlp-postgres-cluster-params"
  
  # Performance parameters
  parameter {
    name  = "shared_preload_libraries"
    value = "pg_stat_statements,pg_hint_plan"
  }
  
  parameter {
    name  = "max_connections"
    value = "1000"
  }
  
  parameter {
    name  = "shared_buffers"
    value = "{DBInstanceClassMemory/4}"
  }
  
  parameter {
    name  = "effective_cache_size"
    value = "{DBInstanceClassMemory*3/4}"
  }
  
  parameter {
    name  = "work_mem"
    value = "32768"  # 32MB
  }
  
  parameter {
    name  = "maintenance_work_mem"
    value = "2097152"  # 2GB
  }
  
  parameter {
    name  = "checkpoint_completion_target"
    value = "0.9"
  }
  
  parameter {
    name  = "wal_buffers"
    value = "2048"  # 16MB
  }
  
  parameter {
    name  = "default_statistics_target"
    value = "100"
  }
  
  parameter {
    name  = "random_page_cost"
    value = "1.1"
  }
  
  parameter {
    name  = "effective_io_concurrency"
    value = "200"
  }
  
  # Logging parameters
  parameter {
    name  = "log_statement"
    value = "ddl"
  }
  
  parameter {
    name  = "log_min_duration_statement"
    value = "1000"
  }
  
  parameter {
    name  = "log_checkpoints"
    value = "1"
  }
  
  parameter {
    name  = "log_connections"
    value = "1"
  }
  
  parameter {
    name  = "log_disconnections"
    value = "1"
  }
  
  parameter {
    name  = "log_lock_waits"
    value = "1"
  }
  
  tags = merge(local.tags, {
    Name = "textnlp-postgres-cluster-params"
  })
}

# Database instance parameter group
resource "aws_db_parameter_group" "database_instance_params" {
  family = "aurora-postgresql15"
  name   = "textnlp-postgres-instance-params"
  
  tags = merge(local.tags, {
    Name = "textnlp-postgres-instance-params"
  })
}

# Security group for database
resource "aws_security_group" "database_sg" {
  name_prefix = "textnlp-database-"
  vpc_id      = data.aws_vpc.main.id
  description = "Security group for TextNLP database cluster"
  
  # PostgreSQL access from application subnets
  ingress {
    description = "PostgreSQL from application"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [data.aws_vpc.main.cidr_block]
  }
  
  # PostgreSQL access from VPN
  ingress {
    description = "PostgreSQL from VPN"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
  }
  
  egress {
    description = "All outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(local.tags, {
    Name = "textnlp-database-sg"
  })
}

# Aurora PostgreSQL cluster
resource "aws_rds_cluster" "database_cluster" {
  cluster_identifier             = local.cluster_name
  engine                        = "aurora-postgresql"
  engine_version               = "15.4"
  database_name                = "textnlp"
  master_username              = "textnlp_admin"
  manage_master_user_password  = false
  master_password              = random_password.master_password.result
  
  # Network configuration
  vpc_security_group_ids = [aws_security_group.database_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.database_subnet_group.name
  
  # Parameter groups
  db_cluster_parameter_group_name = aws_rds_cluster_parameter_group.database_cluster_params.name
  db_instance_parameter_group_name = aws_db_parameter_group.database_instance_params.name
  
  # Backup configuration
  backup_retention_period = 30
  preferred_backup_window = "03:00-04:00"
  copy_tags_to_snapshot   = true
  delete_automated_backups = false
  
  # Maintenance configuration
  preferred_maintenance_window = "sun:04:00-sun:05:00"
  
  # Encryption
  storage_encrypted = true
  kms_key_id       = aws_kms_key.database_key.arn
  
  # Enhanced monitoring
  enabled_cloudwatch_logs_exports = ["postgresql"]
  
  # Performance Insights
  enable_performance_insights = true
  performance_insights_kms_key_id = aws_kms_key.database_key.arn
  performance_insights_retention_period = 7
  
  # Deletion protection
  deletion_protection = true
  skip_final_snapshot = false
  final_snapshot_identifier = "${local.cluster_name}-final-snapshot"
  
  tags = merge(local.tags, {
    Name = local.cluster_name
  })
}

# Aurora cluster instances
resource "aws_rds_cluster_instance" "cluster_instances" {
  count              = 3
  identifier         = "${local.cluster_name}-${count.index + 1}"
  cluster_identifier = aws_rds_cluster.database_cluster.id
  instance_class     = count.index == 0 ? "db.r6g.2xlarge" : "db.r6g.xlarge"
  engine             = aws_rds_cluster.database_cluster.engine
  engine_version     = aws_rds_cluster.database_cluster.engine_version
  
  # Parameter group
  db_parameter_group_name = aws_db_parameter_group.database_instance_params.name
  
  # Enhanced monitoring
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_monitoring_role.arn
  
  # Performance Insights
  performance_insights_enabled = true
  performance_insights_kms_key_id = aws_kms_key.database_key.arn
  performance_insights_retention_period = 7
  
  # Auto minor version upgrade
  auto_minor_version_upgrade = true
  
  tags = merge(local.tags, {
    Name = "${local.cluster_name}-${count.index + 1}"
    Role = count.index == 0 ? "writer" : "reader"
  })
}

# IAM role for RDS enhanced monitoring
resource "aws_iam_role" "rds_monitoring_role" {
  name = "textnlp-rds-monitoring-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })
  
  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "rds_monitoring_policy" {
  role       = aws_iam_role.rds_monitoring_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# RDS Proxy for connection pooling
resource "aws_db_proxy" "database_proxy" {
  name                   = "textnlp-postgres-proxy"
  engine_family         = "POSTGRESQL"
  idle_client_timeout   = 1800
  max_connections_percent = 100
  max_idle_connections_percent = 50
  require_tls           = true
  role_arn              = aws_iam_role.proxy_role.arn
  vpc_subnet_ids        = data.aws_subnets.private.ids
  
  auth {
    auth_scheme = "SECRETS"
    secret_arn  = aws_secretsmanager_secret.database_credentials.arn
    iam_auth    = "REQUIRED"
  }
  
  target {
    db_cluster_identifier = aws_rds_cluster.database_cluster.id
  }
  
  tags = merge(local.tags, {
    Name = "textnlp-postgres-proxy"
  })
}

# Security group for RDS Proxy
resource "aws_security_group" "proxy_sg" {
  name_prefix = "textnlp-proxy-"
  vpc_id      = data.aws_vpc.main.id
  description = "Security group for TextNLP RDS Proxy"
  
  # PostgreSQL access from application subnets
  ingress {
    description = "PostgreSQL from application"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [data.aws_vpc.main.cidr_block]
  }
  
  egress {
    description = "All outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(local.tags, {
    Name = "textnlp-proxy-sg"
  })
}

# IAM role for RDS Proxy
resource "aws_iam_role" "proxy_role" {
  name = "textnlp-rds-proxy-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "rds.amazonaws.com"
        }
      }
    ]
  })
  
  tags = local.tags
}

# IAM policy for RDS Proxy
resource "aws_iam_role_policy" "proxy_policy" {
  name = "textnlp-rds-proxy-policy"
  role = aws_iam_role.proxy_role.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = aws_secretsmanager_secret.database_credentials.arn
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt"
        ]
        Resource = aws_kms_key.database_key.arn
        Condition = {
          StringEquals = {
            "kms:ViaService" = "secretsmanager.${data.aws_region.current.name}.amazonaws.com"
          }
        }
      }
    ]
  })
}

# CloudWatch log group for database logs
resource "aws_cloudwatch_log_group" "database_logs" {
  name              = "/aws/rds/cluster/${local.cluster_name}/postgresql"
  retention_in_days = 30
  kms_key_id       = aws_kms_key.database_key.arn
  
  tags = merge(local.tags, {
    Name = "textnlp-database-logs"
  })
}

# CloudWatch alarms for database monitoring
resource "aws_cloudwatch_metric_alarm" "database_cpu" {
  alarm_name          = "textnlp-database-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors database CPU utilization"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    DBClusterIdentifier = aws_rds_cluster.database_cluster.id
  }
  
  tags = local.tags
}

resource "aws_cloudwatch_metric_alarm" "database_connections" {
  alarm_name          = "textnlp-database-high-connections"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "DatabaseConnections"
  namespace           = "AWS/RDS"
  period              = "300"
  statistic           = "Average"
  threshold           = "800"
  alarm_description   = "This metric monitors database connection count"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    DBClusterIdentifier = aws_rds_cluster.database_cluster.id
  }
  
  tags = local.tags
}

# SNS topic for alerts
resource "aws_sns_topic" "alerts" {
  name         = "textnlp-database-alerts"
  display_name = "TextNLP Database Alerts"
  kms_master_key_id = aws_kms_key.database_key.arn
  
  tags = local.tags
}

# Data sources
data "aws_vpc" "main" {
  filter {
    name   = "tag:Name"
    values = ["textnlp-vpc"]
  }
}

data "aws_subnets" "private" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.main.id]
  }
  
  filter {
    name   = "tag:Type"
    values = ["private"]
  }
}

data "aws_region" "current" {}

# Outputs
output "cluster_endpoint" {
  description = "Aurora cluster endpoint"
  value       = aws_rds_cluster.database_cluster.endpoint
}

output "cluster_reader_endpoint" {
  description = "Aurora cluster reader endpoint"
  value       = aws_rds_cluster.database_cluster.reader_endpoint
}

output "proxy_endpoint" {
  description = "RDS Proxy endpoint"
  value       = aws_db_proxy.database_proxy.endpoint
}

output "database_name" {
  description = "Database name"
  value       = aws_rds_cluster.database_cluster.database_name
}

output "secret_arn" {
  description = "Secrets Manager secret ARN for database credentials"
  value       = aws_secretsmanager_secret.database_credentials.arn
  sensitive   = true
}

output "kms_key_id" {
  description = "KMS key ID for database encryption"
  value       = aws_kms_key.database_key.key_id
}