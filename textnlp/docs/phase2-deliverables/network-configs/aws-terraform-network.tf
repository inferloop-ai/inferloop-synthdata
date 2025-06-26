# AWS Network Infrastructure for TextNLP Platform
# Terraform Configuration - Phase 2: Foundation Setup

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket  = "textnlp-terraform-state"
    key     = "network/terraform.tfstate"
    region  = "us-east-1"
    encrypt = true
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "TextNLP"
      Environment = var.environment
      ManagedBy   = "Terraform"
      CostCenter  = "AI-ML"
    }
  }
}

# Variables
variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

# Data sources
data "aws_caller_identity" "current" {}

# VPC
resource "aws_vpc" "textnlp_vpc" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "textnlp-vpc"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "textnlp_igw" {
  vpc_id = aws_vpc.textnlp_vpc.id
  
  tags = {
    Name = "textnlp-igw"
  }
}

# Public Subnets
resource "aws_subnet" "public_subnets" {
  count                   = length(var.availability_zones)
  vpc_id                  = aws_vpc.textnlp_vpc.id
  cidr_block              = "10.0.${count.index + 1}.0/24"
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true
  
  tags = {
    Name = "textnlp-public-${substr(var.availability_zones[count.index], -2, -1)}"
    Type = "public"
  }
}

# Private Application Subnets
resource "aws_subnet" "private_app_subnets" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.textnlp_vpc.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = var.availability_zones[count.index]
  
  tags = {
    Name = "textnlp-private-app-${substr(var.availability_zones[count.index], -2, -1)}"
    Type = "private-app"
  }
}

# GPU Subnets
resource "aws_subnet" "gpu_subnets" {
  count             = 2  # Only need 2 AZs for GPU workloads
  vpc_id            = aws_vpc.textnlp_vpc.id
  cidr_block        = "10.0.${count.index + 20}.0/24"
  availability_zone = var.availability_zones[count.index]
  
  tags = {
    Name = "textnlp-gpu-${substr(var.availability_zones[count.index], -2, -1)}"
    Type = "gpu"
    "kubernetes.io/role/elb" = "1"
  }
}

# Database Subnets
resource "aws_subnet" "database_subnets" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.textnlp_vpc.id
  cidr_block        = "10.0.${count.index + 30}.0/24"
  availability_zone = var.availability_zones[count.index]
  
  tags = {
    Name = "textnlp-db-${substr(var.availability_zones[count.index], -2, -1)}"
    Type = "database"
  }
}

# Elastic IPs for NAT Gateways
resource "aws_eip" "nat_eips" {
  count  = 2
  domain = "vpc"
  
  tags = {
    Name = "textnlp-nat-eip-${count.index + 1}"
  }
  
  depends_on = [aws_internet_gateway.textnlp_igw]
}

# NAT Gateways
resource "aws_nat_gateway" "nat_gateways" {
  count         = 2
  allocation_id = aws_eip.nat_eips[count.index].id
  subnet_id     = aws_subnet.public_subnets[count.index].id
  
  tags = {
    Name = "textnlp-nat-${substr(var.availability_zones[count.index], -2, -1)}"
  }
  
  depends_on = [aws_internet_gateway.textnlp_igw]
}

# Route Tables
resource "aws_route_table" "public_rt" {
  vpc_id = aws_vpc.textnlp_vpc.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.textnlp_igw.id
  }
  
  tags = {
    Name = "textnlp-public-rt"
  }
}

resource "aws_route_table" "private_app_rt" {
  count  = 2
  vpc_id = aws_vpc.textnlp_vpc.id
  
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat_gateways[count.index].id
  }
  
  tags = {
    Name = "textnlp-private-app-rt-${count.index + 1}"
  }
}

resource "aws_route_table" "gpu_rt" {
  vpc_id = aws_vpc.textnlp_vpc.id
  
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat_gateways[0].id
  }
  
  tags = {
    Name = "textnlp-gpu-rt"
  }
}

resource "aws_route_table" "database_rt" {
  vpc_id = aws_vpc.textnlp_vpc.id
  
  tags = {
    Name = "textnlp-db-rt"
  }
}

# Route Table Associations
resource "aws_route_table_association" "public_rta" {
  count          = length(aws_subnet.public_subnets)
  subnet_id      = aws_subnet.public_subnets[count.index].id
  route_table_id = aws_route_table.public_rt.id
}

resource "aws_route_table_association" "private_app_rta" {
  count          = length(aws_subnet.private_app_subnets)
  subnet_id      = aws_subnet.private_app_subnets[count.index].id
  route_table_id = aws_route_table.private_app_rt[count.index < 2 ? count.index : 0].id
}

resource "aws_route_table_association" "gpu_rta" {
  count          = length(aws_subnet.gpu_subnets)
  subnet_id      = aws_subnet.gpu_subnets[count.index].id
  route_table_id = aws_route_table.gpu_rt.id
}

resource "aws_route_table_association" "database_rta" {
  count          = length(aws_subnet.database_subnets)
  subnet_id      = aws_subnet.database_subnets[count.index].id
  route_table_id = aws_route_table.database_rt.id
}

# Security Groups
resource "aws_security_group" "load_balancer_sg" {
  name_prefix = "textnlp-lb-"
  vpc_id      = aws_vpc.textnlp_vpc.id
  description = "Security group for load balancers"
  
  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "textnlp-lb-sg"
  }
  
  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_security_group" "application_sg" {
  name_prefix = "textnlp-app-"
  vpc_id      = aws_vpc.textnlp_vpc.id
  description = "Security group for application servers"
  
  ingress {
    description     = "HTTP from LB"
    from_port       = 80
    to_port         = 80
    protocol        = "tcp"
    security_groups = [aws_security_group.load_balancer_sg.id]
  }
  
  ingress {
    description     = "HTTPS from LB"
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    security_groups = [aws_security_group.load_balancer_sg.id]
  }
  
  ingress {
    description     = "API from LB"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.load_balancer_sg.id]
  }
  
  ingress {
    description     = "SSH from Bastion"
    from_port       = 22
    to_port         = 22
    protocol        = "tcp"
    security_groups = [aws_security_group.bastion_sg.id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "textnlp-app-sg"
  }
  
  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_security_group" "gpu_sg" {
  name_prefix = "textnlp-gpu-"
  vpc_id      = aws_vpc.textnlp_vpc.id
  description = "Security group for GPU worker nodes"
  
  ingress {
    description     = "Model serving from App"
    from_port       = 8080
    to_port         = 8080
    protocol        = "tcp"
    security_groups = [aws_security_group.application_sg.id]
  }
  
  ingress {
    description     = "gRPC from App"
    from_port       = 50051
    to_port         = 50051
    protocol        = "tcp"
    security_groups = [aws_security_group.application_sg.id]
  }
  
  ingress {
    description     = "SSH from Bastion"
    from_port       = 22
    to_port         = 22
    protocol        = "tcp"
    security_groups = [aws_security_group.bastion_sg.id]
  }
  
  # Kubernetes node communication
  ingress {
    description = "Kubernetes communication"
    from_port   = 10250
    to_port     = 10250
    protocol    = "tcp"
    self        = true
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "textnlp-gpu-sg"
  }
  
  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_security_group" "database_sg" {
  name_prefix = "textnlp-db-"
  vpc_id      = aws_vpc.textnlp_vpc.id
  description = "Security group for databases"
  
  ingress {
    description     = "PostgreSQL from App"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.application_sg.id, aws_security_group.gpu_sg.id]
  }
  
  ingress {
    description     = "Redis from App"
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.application_sg.id, aws_security_group.gpu_sg.id]
  }
  
  tags = {
    Name = "textnlp-db-sg"
  }
  
  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_security_group" "bastion_sg" {
  name_prefix = "textnlp-bastion-"
  vpc_id      = aws_vpc.textnlp_vpc.id
  description = "Security group for bastion hosts"
  
  ingress {
    description = "SSH from office"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    # Replace with actual office IP ranges
    cidr_blocks = ["203.0.113.0/24", "198.51.100.0/24"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = [var.vpc_cidr]
  }
  
  tags = {
    Name = "textnlp-bastion-sg"
  }
  
  lifecycle {
    create_before_destroy = true
  }
}

# VPC Endpoints
resource "aws_vpc_endpoint" "s3" {
  vpc_id       = aws_vpc.textnlp_vpc.id
  service_name = "com.amazonaws.${var.aws_region}.s3"
  
  route_table_ids = [
    aws_route_table.private_app_rt[0].id,
    aws_route_table.private_app_rt[1].id,
    aws_route_table.gpu_rt.id,
    aws_route_table.database_rt.id
  ]
  
  tags = {
    Name = "textnlp-s3-endpoint"
  }
}

resource "aws_vpc_endpoint" "ecr_api" {
  vpc_id              = aws_vpc.textnlp_vpc.id
  service_name        = "com.amazonaws.${var.aws_region}.ecr.api"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = [aws_subnet.private_app_subnets[0].id, aws_subnet.private_app_subnets[1].id]
  security_group_ids  = [aws_security_group.application_sg.id]
  private_dns_enabled = true
  
  tags = {
    Name = "textnlp-ecr-api-endpoint"
  }
}

resource "aws_vpc_endpoint" "ecr_dkr" {
  vpc_id              = aws_vpc.textnlp_vpc.id
  service_name        = "com.amazonaws.${var.aws_region}.ecr.dkr"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = [aws_subnet.private_app_subnets[0].id, aws_subnet.private_app_subnets[1].id]
  security_group_ids  = [aws_security_group.application_sg.id]
  private_dns_enabled = true
  
  tags = {
    Name = "textnlp-ecr-dkr-endpoint"
  }
}

# CloudWatch Log Group for VPC Flow Logs
resource "aws_cloudwatch_log_group" "vpc_flow_logs" {
  name              = "/aws/vpc/textnlp/flowlogs"
  retention_in_days = 30
  
  tags = {
    Name = "textnlp-vpc-flow-logs"
  }
}

# IAM Role for VPC Flow Logs
resource "aws_iam_role" "flow_logs_role" {
  name = "textnlp-flow-logs-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "vpc-flow-logs.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "flow_logs_policy" {
  name = "textnlp-flow-logs-policy"
  role = aws_iam_role.flow_logs_role.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams"
        ]
        Resource = "*"
      }
    ]
  })
}

# VPC Flow Logs
resource "aws_flow_log" "vpc_flow_log" {
  iam_role_arn    = aws_iam_role.flow_logs_role.arn
  log_destination = aws_cloudwatch_log_group.vpc_flow_logs.arn
  traffic_type    = "ALL"
  vpc_id          = aws_vpc.textnlp_vpc.id
  
  tags = {
    Name = "textnlp-vpc-flow-log"
  }
}

# Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.textnlp_vpc.id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = aws_vpc.textnlp_vpc.cidr_block
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public_subnets[*].id
}

output "private_app_subnet_ids" {
  description = "IDs of the private application subnets"
  value       = aws_subnet.private_app_subnets[*].id
}

output "gpu_subnet_ids" {
  description = "IDs of the GPU subnets"
  value       = aws_subnet.gpu_subnets[*].id
}

output "database_subnet_ids" {
  description = "IDs of the database subnets"
  value       = aws_subnet.database_subnets[*].id
}

output "security_group_ids" {
  description = "Security group IDs"
  value = {
    load_balancer = aws_security_group.load_balancer_sg.id
    application   = aws_security_group.application_sg.id
    gpu           = aws_security_group.gpu_sg.id
    database      = aws_security_group.database_sg.id
    bastion       = aws_security_group.bastion_sg.id
  }
}

output "nat_gateway_ids" {
  description = "IDs of the NAT gateways"
  value       = aws_nat_gateway.nat_gateways[*].id
}

output "internet_gateway_id" {
  description = "ID of the Internet Gateway"
  value       = aws_internet_gateway.textnlp_igw.id
}