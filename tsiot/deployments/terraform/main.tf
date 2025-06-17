# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# Random password for RDS
resource "random_password" "rds_password" {
  length  = 16
  special = true
}

# KMS key for encryption
resource "aws_kms_key" "tsiot" {
  description             = "KMS key for TSIoT encryption"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = {
    Name = "${var.project_name}-${var.environment}-kms"
  }
}

resource "aws_kms_alias" "tsiot" {
  name          = "alias/${var.project_name}-${var.environment}"
  target_key_id = aws_kms_key.tsiot.key_id
}

# VPC Module
module "vpc" {
  source = "./modules/vpc"

  project_name     = var.project_name
  environment      = var.environment
  vpc_cidr         = var.vpc_cidr
  availability_zones = var.availability_zones
  
  private_subnet_cidrs  = var.private_subnet_cidrs
  public_subnet_cidrs   = var.public_subnet_cidrs
  database_subnet_cidrs = var.database_subnet_cidrs
  
  enable_nat_gateway = true
  enable_vpn_gateway = false
  enable_dns_hostnames = true
  enable_dns_support = true
  
  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

# EKS Module
module "eks" {
  source = "./modules/eks"

  project_name    = var.project_name
  environment     = var.environment
  cluster_name    = var.cluster_name
  cluster_version = var.cluster_version
  
  vpc_id              = module.vpc.vpc_id
  subnet_ids          = module.vpc.private_subnet_ids
  control_plane_subnet_ids = module.vpc.private_subnet_ids
  
  node_groups = var.node_groups
  
  # Security
  cluster_encryption_config = var.enable_encryption ? [{
    provider_key_arn = aws_kms_key.tsiot.arn
    resources        = ["secrets"]
  }] : []
  
  # Logging
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  
  # Add-ons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }
  
  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

# GPU Node Group (conditional)
resource "aws_eks_node_group" "gpu_nodes" {
  count = var.enable_gpu_nodes ? 1 : 0
  
  cluster_name    = module.eks.cluster_name
  node_group_name = "${var.project_name}-${var.environment}-gpu"
  node_role_arn   = module.eks.node_group_role_arn
  subnet_ids      = module.vpc.private_subnet_ids
  
  instance_types = var.gpu_node_config.instance_types
  capacity_type  = "ON_DEMAND"
  disk_size      = 100
  
  scaling_config {
    desired_size = var.gpu_node_config.scaling_config.desired_size
    max_size     = var.gpu_node_config.scaling_config.max_size
    min_size     = var.gpu_node_config.scaling_config.min_size
  }
  
  labels = {
    "node-type" = "gpu"
    "nvidia.com/gpu" = "true"
  }
  
  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }
  
  tags = {
    Environment = var.environment
    Project     = var.project_name
    NodeType    = "gpu"
  }
}

# RDS Module
module "rds" {
  source = "./modules/rds"

  project_name = var.project_name
  environment  = var.environment
  
  # Networking
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.database_subnet_ids
  allowed_security_groups = [module.eks.cluster_primary_security_group_id]
  
  # Instance configuration
  engine_version    = var.rds_engine_version
  instance_class    = var.rds_instance_class
  allocated_storage = var.rds_allocated_storage
  max_allocated_storage = var.rds_max_allocated_storage
  
  # Database configuration
  database_name = "tsiot"
  username      = "tsiot_admin"
  password      = random_password.rds_password.result
  
  # Backup and maintenance
  backup_retention_period = var.rds_backup_retention_period
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  multi_az              = var.rds_multi_az
  
  # Security
  encryption_enabled = var.enable_encryption
  kms_key_id        = var.enable_encryption ? aws_kms_key.tsiot.arn : null
  
  # Performance Insights
  performance_insights_enabled = true
  performance_insights_retention_period = 7
  
  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "redis" {
  name       = "${var.project_name}-${var.environment}-redis"
  subnet_ids = module.vpc.private_subnet_ids
}

resource "aws_security_group" "redis" {
  name_prefix = "${var.project_name}-${var.environment}-redis"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [module.eks.cluster_primary_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-redis"
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = "${var.project_name}-${var.environment}-redis"
  description                = "Redis cluster for TSIoT"
  
  node_type                  = var.redis_node_type
  num_cache_clusters         = var.redis_num_cache_nodes
  port                       = 6379
  parameter_group_name       = "default.redis7"
  engine_version            = var.redis_engine_version
  
  subnet_group_name          = aws_elasticache_subnet_group.redis.name
  security_group_ids         = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = var.enable_encryption
  transit_encryption_enabled = var.enable_encryption
  auth_token_enabled         = var.enable_encryption
  
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  snapshot_retention_limit = 7
  snapshot_window         = "03:00-05:00"
  
  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

# OpenSearch
resource "aws_opensearch_domain" "main" {
  domain_name    = "${var.project_name}-${var.environment}"
  engine_version = "OpenSearch_2.3"

  cluster_config {
    instance_type  = var.opensearch_instance_type
    instance_count = var.opensearch_instance_count
    
    dedicated_master_enabled = var.opensearch_instance_count > 2
    dedicated_master_count   = var.opensearch_instance_count > 2 ? 3 : 0
    dedicated_master_type    = var.opensearch_instance_count > 2 ? "r6g.medium.search" : null
    
    zone_awareness_enabled = var.opensearch_instance_count > 1
    
    dynamic "zone_awareness_config" {
      for_each = var.opensearch_instance_count > 1 ? [1] : []
      content {
        availability_zone_count = min(var.opensearch_instance_count, length(var.availability_zones))
      }
    }
  }

  ebs_options {
    ebs_enabled = true
    volume_type = "gp3"
    volume_size = var.opensearch_ebs_volume_size
  }

  vpc_options {
    subnet_ids         = slice(module.vpc.private_subnet_ids, 0, min(var.opensearch_instance_count, length(module.vpc.private_subnet_ids)))
    security_group_ids = [aws_security_group.opensearch.id]
  }

  encrypt_at_rest {
    enabled    = var.enable_encryption
    kms_key_id = var.enable_encryption ? aws_kms_key.tsiot.key_id : null
  }

  node_to_node_encryption {
    enabled = var.enable_encryption
  }

  domain_endpoint_options {
    enforce_https       = true
    tls_security_policy = "Policy-Min-TLS-1-2-2019-07"
  }

  advanced_security_options {
    enabled                        = true
    anonymous_auth_enabled         = false
    internal_user_database_enabled = true
    
    master_user_options {
      master_user_name     = "tsiot_admin"
      master_user_password = random_password.opensearch_password.result
    }
  }

  log_publishing_options {
    cloudwatch_log_group_arn = aws_cloudwatch_log_group.opensearch.arn
    log_type                 = "INDEX_SLOW_LOGS"
  }

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "random_password" "opensearch_password" {
  length  = 16
  special = true
}

resource "aws_security_group" "opensearch" {
  name_prefix = "${var.project_name}-${var.environment}-opensearch"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    security_groups = [module.eks.cluster_primary_security_group_id]
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-opensearch"
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_cloudwatch_log_group" "opensearch" {
  name              = "/aws/opensearch/${var.project_name}-${var.environment}"
  retention_in_days = 7
  kms_key_id       = var.enable_encryption ? aws_kms_key.tsiot.arn : null
}

# MSK (Managed Streaming for Kafka)
resource "aws_msk_cluster" "main" {
  cluster_name           = "${var.project_name}-${var.environment}-kafka"
  kafka_version          = var.kafka_version
  number_of_broker_nodes = length(module.vpc.private_subnet_ids)

  broker_node_group_info {
    instance_type   = var.kafka_instance_type
    client_subnets  = module.vpc.private_subnet_ids
    security_groups = [aws_security_group.msk.id]
    
    storage_info {
      ebs_storage_info {
        volume_size = var.kafka_ebs_volume_size
      }
    }
  }

  client_authentication {
    tls {
      certificate_authority_arns = []
    }
    sasl {
      scram = true
    }
  }

  encryption_info {
    encryption_at_rest_kms_key_id = aws_kms_key.tsiot.arn
    encryption_in_transit {
      client_broker = "TLS"
      in_cluster    = true
    }
  }

  logging_info {
    broker_logs {
      cloudwatch_logs {
        enabled   = true
        log_group = aws_cloudwatch_log_group.msk.name
      }
    }
  }

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_security_group" "msk" {
  name_prefix = "${var.project_name}-${var.environment}-msk"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 9092
    to_port         = 9092
    protocol        = "tcp"
    security_groups = [module.eks.cluster_primary_security_group_id]
  }

  ingress {
    from_port       = 9094
    to_port         = 9094
    protocol        = "tcp"
    security_groups = [module.eks.cluster_primary_security_group_id]
  }

  ingress {
    from_port       = 2181
    to_port         = 2181
    protocol        = "tcp"
    security_groups = [module.eks.cluster_primary_security_group_id]
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-msk"
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_cloudwatch_log_group" "msk" {
  name              = "/aws/msk/${var.project_name}-${var.environment}"
  retention_in_days = 7
  kms_key_id       = var.enable_encryption ? aws_kms_key.tsiot.arn : null
}

# S3 Buckets
resource "aws_s3_bucket" "data_lake" {
  bucket = "${var.project_name}-${var.environment}-data-lake-${random_id.bucket_suffix.hex}"

  tags = {
    Environment = var.environment
    Project     = var.project_name
    Purpose     = "data-lake"
  }
}

resource "aws_s3_bucket" "backup" {
  bucket = "${var.project_name}-${var.environment}-backup-${random_id.bucket_suffix.hex}"

  tags = {
    Environment = var.environment
    Project     = var.project_name
    Purpose     = "backup"
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# S3 bucket configurations
resource "aws_s3_bucket_versioning" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_versioning" "backup" {
  bucket = aws_s3_bucket.backup.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data_lake" {
  count  = var.enable_encryption ? 1 : 0
  bucket = aws_s3_bucket.data_lake.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.tsiot.arn
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "backup" {
  count  = var.enable_encryption ? 1 : 0
  bucket = aws_s3_bucket.backup.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.tsiot.arn
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id

  rule {
    id     = "transition_to_ia"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    transition {
      days          = 365
      storage_class = "DEEP_ARCHIVE"
    }
  }
}

# IAM Roles and Policies
resource "aws_iam_role" "tsiot_app_role" {
  name = "${var.project_name}-${var.environment}-app-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRoleWithWebIdentity"
        Effect = "Allow"
        Principal = {
          Federated = module.eks.oidc_provider_arn
        }
        Condition = {
          StringEquals = {
            "${replace(module.eks.oidc_provider, "https://", "")}:sub" = "system:serviceaccount:tsiot:tsiot-app"
            "${replace(module.eks.oidc_provider, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_iam_policy" "tsiot_app_policy" {
  name = "${var.project_name}-${var.environment}-app-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.data_lake.arn,
          "${aws_s3_bucket.data_lake.arn}/*",
          aws_s3_bucket.backup.arn,
          "${aws_s3_bucket.backup.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "rds:DescribeDBInstances",
          "rds:DescribeDBClusters"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "kafka:DescribeCluster",
          "kafka:GetBootstrapBrokers"
        ]
        Resource = aws_msk_cluster.main.arn
      },
      {
        Effect = "Allow"
        Action = [
          "es:ESHttpGet",
          "es:ESHttpPost",
          "es:ESHttpPut"
        ]
        Resource = "${aws_opensearch_domain.main.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          aws_secretsmanager_secret.rds_credentials.arn,
          aws_secretsmanager_secret.opensearch_credentials.arn
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "tsiot_app_policy" {
  role       = aws_iam_role.tsiot_app_role.name
  policy_arn = aws_iam_policy.tsiot_app_policy.arn
}

# Secrets Manager
resource "aws_secretsmanager_secret" "rds_credentials" {
  name        = "${var.project_name}/${var.environment}/rds/credentials"
  description = "RDS credentials for TSIoT"
  kms_key_id  = var.enable_encryption ? aws_kms_key.tsiot.arn : null

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_secretsmanager_secret_version" "rds_credentials" {
  secret_id = aws_secretsmanager_secret.rds_credentials.id
  secret_string = jsonencode({
    username = module.rds.database_username
    password = random_password.rds_password.result
    endpoint = module.rds.database_endpoint
    port     = module.rds.database_port
    database = module.rds.database_name
  })
}

resource "aws_secretsmanager_secret" "opensearch_credentials" {
  name        = "${var.project_name}/${var.environment}/opensearch/credentials"
  description = "OpenSearch credentials for TSIoT"
  kms_key_id  = var.enable_encryption ? aws_kms_key.tsiot.arn : null

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_secretsmanager_secret_version" "opensearch_credentials" {
  secret_id = aws_secretsmanager_secret.opensearch_credentials.id
  secret_string = jsonencode({
    username = "tsiot_admin"
    password = random_password.opensearch_password.result
    endpoint = aws_opensearch_domain.main.endpoint
  })
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "application" {
  name              = "/aws/eks/${var.project_name}-${var.environment}/application"
  retention_in_days = 30
  kms_key_id       = var.enable_encryption ? aws_kms_key.tsiot.arn : null

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

# Load Balancer Controller IRSA
module "load_balancer_controller_irsa_role" {
  source = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"

  role_name = "${var.project_name}-${var.environment}-load-balancer-controller"

  attach_load_balancer_controller_policy = true

  oidc_providers = {
    ex = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:aws-load-balancer-controller"]
    }
  }

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

# Monitoring Module (Conditional)
module "monitoring" {
  count  = var.enable_monitoring ? 1 : 0
  source = "./modules/monitoring"

  project_name = var.project_name
  environment  = var.environment
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnet_ids
  
  cluster_name = module.eks.cluster_name
  
  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}