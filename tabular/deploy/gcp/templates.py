"""
GCP Deployment Templates
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class GCPTemplates:
    """Templates for GCP resource deployment"""

    @staticmethod
    def cloud_run_template(
        service_name: str,
        image: str,
        region: str = "us-central1",
        memory: int = 512,
        cpu: int = 1,
        max_instances: int = 10,
        environment: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Generate Cloud Run deployment template"""

        template = {
            "apiVersion": "serving.knative.dev/v1",
            "kind": "Service",
            "metadata": {
                "name": service_name,
                "labels": {"app": "inferloop-synthetic", "component": "api"},
                "annotations": {"run.googleapis.com/launch-stage": "BETA"},
            },
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "autoscaling.knative.dev/maxScale": str(max_instances),
                            "run.googleapis.com/cpu-throttling": "false",
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "image": image,
                                "resources": {
                                    "limits": {"cpu": str(cpu), "memory": f"{memory}Mi"}
                                },
                                "ports": [{"containerPort": 8000}],
                                "env": [
                                    {"name": k, "value": v}
                                    for k, v in (environment or {}).items()
                                ],
                            }
                        ]
                    },
                }
            },
        }

        return template

    @staticmethod
    def gke_cluster_template(
        cluster_name: str,
        zone: str,
        node_count: int = 3,
        machine_type: str = "e2-standard-4",
        min_nodes: int = 1,
        max_nodes: int = 10,
    ) -> Dict[str, Any]:
        """Generate GKE cluster configuration template"""

        template = {
            "name": cluster_name,
            "zone": zone,
            "initial_node_count": node_count,
            "node_config": {
                "machine_type": machine_type,
                "disk_size_gb": 100,
                "oauth_scopes": ["https://www.googleapis.com/auth/cloud-platform"],
                "labels": {"app": "inferloop-synthetic", "component": "compute"},
            },
            "autoscaling": {
                "enabled": True,
                "min_node_count": min_nodes,
                "max_node_count": max_nodes,
            },
            "addons_config": {
                "http_load_balancing": {"disabled": False},
                "horizontal_pod_autoscaling": {"disabled": False},
            },
        }

        return template

    @staticmethod
    def kubernetes_deployment_template(
        app_name: str,
        image: str,
        replicas: int = 3,
        cpu_request: str = "100m",
        memory_request: str = "128Mi",
        cpu_limit: str = "1000m",
        memory_limit: str = "512Mi",
    ) -> Dict[str, Any]:
        """Generate Kubernetes deployment template for GKE"""

        template = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": f"{app_name}-deployment", "labels": {"app": app_name}},
            "spec": {
                "replicas": replicas,
                "selector": {"matchLabels": {"app": app_name}},
                "template": {
                    "metadata": {"labels": {"app": app_name}},
                    "spec": {
                        "containers": [
                            {
                                "name": app_name,
                                "image": image,
                                "ports": [{"containerPort": 8000}],
                                "resources": {
                                    "requests": {
                                        "cpu": cpu_request,
                                        "memory": memory_request,
                                    },
                                    "limits": {
                                        "cpu": cpu_limit,
                                        "memory": memory_limit,
                                    },
                                },
                                "livenessProbe": {
                                    "httpGet": {"path": "/health", "port": 8000},
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10,
                                },
                                "readinessProbe": {
                                    "httpGet": {"path": "/health", "port": 8000},
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 5,
                                },
                            }
                        ]
                    },
                },
            },
        }

        return template

    @staticmethod
    def kubernetes_service_template(app_name: str) -> Dict[str, Any]:
        """Generate Kubernetes service template"""

        template = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": f"{app_name}-service", "labels": {"app": app_name}},
            "spec": {
                "type": "LoadBalancer",
                "ports": [{"port": 80, "targetPort": 8000, "protocol": "TCP"}],
                "selector": {"app": app_name},
            },
        }

        return template

    @staticmethod
    def cloud_function_template(
        function_name: str,
        entry_point: str = "main",
        runtime: str = "python39",
        memory: int = 256,
        timeout: int = 60,
        max_instances: int = 10,
    ) -> Dict[str, Any]:
        """Generate Cloud Function configuration template"""

        template = {
            "name": function_name,
            "source_archive_url": f"gs://inferloop-functions/{function_name}.zip",
            "entry_point": entry_point,
            "runtime": runtime,
            "available_memory_mb": memory,
            "timeout": f"{timeout}s",
            "max_instances": max_instances,
            "trigger": {
                "event_type": "providers/cloud.pubsub/eventTypes/topic.publish",
                "resource": f"projects/{{project_id}}/topics/{function_name}-trigger",
            },
            "environment_variables": {
                "FUNCTION_NAME": function_name,
                "PROJECT_ID": "{project_id}",
            },
        }

        return template

    @staticmethod
    def cloud_sql_template(
        instance_name: str,
        database_version: str = "POSTGRES_13",
        tier: str = "db-f1-micro",
        region: str = "us-central1",
        storage_gb: int = 10,
        backup_enabled: bool = True,
    ) -> Dict[str, Any]:
        """Generate Cloud SQL configuration template"""

        template = {
            "name": instance_name,
            "database_version": database_version,
            "region": region,
            "settings": {
                "tier": tier,
                "disk_size": storage_gb,
                "disk_type": "PD_SSD",
                "backup_configuration": {
                    "enabled": backup_enabled,
                    "start_time": "03:00",
                    "location": region,
                },
                "database_flags": [{"name": "max_connections", "value": "100"}],
                "ip_configuration": {
                    "ipv4_enabled": True,
                    "authorized_networks": [
                        {
                            "name": "inferloop-network",
                            "value": "0.0.0.0/0",  # In production, restrict this
                        }
                    ],
                },
            },
        }

        return template

    @staticmethod
    def terraform_main_template(project_id: str, region: str) -> str:
        """Generate main Terraform configuration"""

        template = f"""
terraform {{
  required_providers {{
    google = {{
      source  = "hashicorp/google"
      version = "~> 4.0"
    }}
  }}
}}

provider "google" {{
  project = "{project_id}"
  region  = "{region}"
}}

# Enable required APIs
resource "google_project_service" "required_apis" {{
  for_each = toset([
    "compute.googleapis.com",
    "container.googleapis.com",
    "run.googleapis.com",
    "cloudfunctions.googleapis.com",
    "cloudbuild.googleapis.com",
    "storage.googleapis.com",
    "sqladmin.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com"
  ])
  
  service = each.key
  disable_on_destroy = false
}}

# VPC Network
resource "google_compute_network" "inferloop_network" {{
  name                    = "inferloop-network"
  auto_create_subnetworks = false
  depends_on             = [google_project_service.required_apis]
}}

# Subnet
resource "google_compute_subnetwork" "inferloop_subnet" {{
  name          = "inferloop-subnet"
  ip_cidr_range = "10.0.0.0/24"
  region        = "{region}"
  network       = google_compute_network.inferloop_network.id
}}

# Cloud Storage bucket for data
resource "google_storage_bucket" "inferloop_data" {{
  name          = "${{var.project_id}}-inferloop-data"
  location      = "{region}"
  force_destroy = true
  
  lifecycle_rule {{
    condition {{
      age = 30
    }}
    action {{
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }}
  }}
}}

# Cloud Storage bucket for models
resource "google_storage_bucket" "inferloop_models" {{
  name          = "${{var.project_id}}-inferloop-models"
  location      = "{region}"
  force_destroy = true
  
  versioning {{
    enabled = true
  }}
}}
"""

        return template

    @staticmethod
    def terraform_variables_template() -> str:
        """Generate Terraform variables configuration"""

        template = """
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "api_image" {
  description = "Docker image for API service"
  type        = string
  default     = "gcr.io/inferloop/synthetic-api:latest"
}

variable "api_cpu" {
  description = "CPU allocation for API service"
  type        = number
  default     = 1
}

variable "api_memory" {
  description = "Memory allocation for API service (MB)"
  type        = number
  default     = 512
}

variable "api_max_instances" {
  description = "Maximum instances for API service"
  type        = number
  default     = 10
}

variable "gke_node_count" {
  description = "Number of nodes in GKE cluster"
  type        = number
  default     = 3
}

variable "gke_machine_type" {
  description = "Machine type for GKE nodes"
  type        = string
  default     = "e2-standard-4"
}

variable "database_tier" {
  description = "Cloud SQL instance tier"
  type        = string
  default     = "db-f1-micro"
}

variable "database_storage_gb" {
  description = "Cloud SQL storage size (GB)"
  type        = number
  default     = 10
}
"""

        return template

    @staticmethod
    def deployment_script_template() -> str:
        """Generate deployment script"""

        script = """#!/bin/bash
# GCP Deployment Script for Inferloop Synthetic Data

set -e

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
NC='\\033[0m' # No Color

# Configuration
PROJECT_ID="${PROJECT_ID}"
REGION="${REGION:-us-central1}"
ZONE="${ZONE:-us-central1-a}"
ENVIRONMENT="${ENVIRONMENT:-dev}"

echo -e "${GREEN}Starting GCP deployment for Inferloop Synthetic Data${NC}"

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check gcloud
    if ! command -v gcloud &> /dev/null; then
        echo -e "${RED}gcloud CLI not found. Please install it first.${NC}"
        exit 1
    fi
    
    # Check terraform (optional)
    if command -v terraform &> /dev/null; then
        echo -e "${GREEN}Terraform found${NC}"
        USE_TERRAFORM=true
    else
        echo -e "${YELLOW}Terraform not found. Using gcloud commands.${NC}"
        USE_TERRAFORM=false
    fi
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        echo -e "${YELLOW}kubectl not found. GKE deployment will be limited.${NC}"
    fi
}

# Enable APIs
enable_apis() {
    echo -e "${YELLOW}Enabling required APIs...${NC}"
    
    gcloud services enable compute.googleapis.com \\
        container.googleapis.com \\
        run.googleapis.com \\
        cloudfunctions.googleapis.com \\
        cloudbuild.googleapis.com \\
        storage.googleapis.com \\
        sqladmin.googleapis.com \\
        monitoring.googleapis.com \\
        logging.googleapis.com \\
        --project="${PROJECT_ID}"
}

# Deploy with Terraform
deploy_with_terraform() {
    echo -e "${YELLOW}Deploying with Terraform...${NC}"
    
    cd terraform/
    terraform init
    terraform plan -var="project_id=${PROJECT_ID}" -var="region=${REGION}"
    terraform apply -var="project_id=${PROJECT_ID}" -var="region=${REGION}" -auto-approve
    cd ..
}

# Deploy with gcloud
deploy_with_gcloud() {
    echo -e "${YELLOW}Deploying with gcloud commands...${NC}"
    
    # Create network
    gcloud compute networks create inferloop-network \\
        --subnet-mode=custom \\
        --project="${PROJECT_ID}" || true
    
    # Create subnet
    gcloud compute networks subnets create inferloop-subnet \\
        --network=inferloop-network \\
        --range=10.0.0.0/24 \\
        --region="${REGION}" \\
        --project="${PROJECT_ID}" || true
    
    # Create storage buckets
    gsutil mb -p "${PROJECT_ID}" -c STANDARD -l "${REGION}" \\
        "gs://${PROJECT_ID}-inferloop-data" || true
    
    gsutil mb -p "${PROJECT_ID}" -c STANDARD -l "${REGION}" \\
        "gs://${PROJECT_ID}-inferloop-models" || true
    
    # Deploy Cloud Run service
    gcloud run deploy inferloop-synthetic-api \\
        --image="gcr.io/inferloop/synthetic-api:latest" \\
        --platform=managed \\
        --region="${REGION}" \\
        --allow-unauthenticated \\
        --memory=512Mi \\
        --cpu=1 \\
        --max-instances=10 \\
        --project="${PROJECT_ID}"
}

# Deploy GKE cluster
deploy_gke() {
    echo -e "${YELLOW}Deploying GKE cluster...${NC}"
    
    gcloud container clusters create inferloop-cluster \\
        --zone="${ZONE}" \\
        --num-nodes=3 \\
        --machine-type=e2-standard-4 \\
        --enable-autoscaling \\
        --min-nodes=1 \\
        --max-nodes=10 \\
        --project="${PROJECT_ID}"
    
    # Get credentials
    gcloud container clusters get-credentials inferloop-cluster \\
        --zone="${ZONE}" \\
        --project="${PROJECT_ID}"
    
    # Deploy to GKE
    if command -v kubectl &> /dev/null; then
        kubectl apply -f kubernetes/
    fi
}

# Main deployment
main() {
    check_prerequisites
    enable_apis
    
    if [ "${USE_TERRAFORM}" = true ]; then
        deploy_with_terraform
    else
        deploy_with_gcloud
    fi
    
    # Optional: Deploy GKE
    read -p "Deploy GKE cluster? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        deploy_gke
    fi
    
    echo -e "${GREEN}Deployment completed successfully!${NC}"
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Build and push your Docker image to gcr.io/${PROJECT_ID}/synthetic-api"
    echo "2. Update the Cloud Run service with your image"
    echo "3. Configure environment variables and secrets"
    echo "4. Set up monitoring and alerts"
}

# Run main function
main
"""

        return script

    @staticmethod
    def docker_template() -> str:
        """Generate Dockerfile for GCP deployment"""

        dockerfile = """# Multi-stage build for optimized image
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml setup.py ./
COPY inferloop_synthetic ./inferloop_synthetic

# Install dependencies
RUN pip install --no-cache-dir -e ".[all]"

# Production stage
FROM python:3.9-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Copy from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /app /app

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Environment variables
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the application
CMD ["uvicorn", "inferloop_synthetic.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
"""

        return dockerfile

    @staticmethod
    def cloudbuild_template() -> str:
        """Generate Cloud Build configuration"""

        cloudbuild = """steps:
  # Build the Docker image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/inferloop-synthetic:$COMMIT_SHA', '.']
    
  # Push the Docker image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/inferloop-synthetic:$COMMIT_SHA']
    
  # Tag latest
  - name: 'gcr.io/cloud-builders/docker'
    args: ['tag', 'gcr.io/$PROJECT_ID/inferloop-synthetic:$COMMIT_SHA', 'gcr.io/$PROJECT_ID/inferloop-synthetic:latest']
    
  # Push latest tag
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/inferloop-synthetic:latest']
    
  # Deploy to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'inferloop-synthetic-api'
      - '--image'
      - 'gcr.io/$PROJECT_ID/inferloop-synthetic:$COMMIT_SHA'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
      - '--allow-unauthenticated'

# Build configuration options
options:
  machineType: 'N1_HIGHCPU_8'
  timeout: '1200s'

# Store build logs in Cloud Storage
logsBucket: 'gs://$PROJECT_ID-cloudbuild-logs'
"""

        return cloudbuild

    @staticmethod
    def save_templates(output_dir: Path):
        """Save all templates to files"""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (output_dir / "terraform").mkdir(exist_ok=True)
        (output_dir / "kubernetes").mkdir(exist_ok=True)
        (output_dir / "scripts").mkdir(exist_ok=True)

        # Save Terraform files
        with open(output_dir / "terraform" / "main.tf", "w") as f:
            f.write(
                GCPTemplates.terraform_main_template(
                    "${var.project_id}", "${var.region}"
                )
            )

        with open(output_dir / "terraform" / "variables.tf", "w") as f:
            f.write(GCPTemplates.terraform_variables_template())

        # Save Kubernetes templates
        k8s_deployment = GCPTemplates.kubernetes_deployment_template(
            "inferloop-synthetic", "gcr.io/${PROJECT_ID}/inferloop-synthetic:latest"
        )
        with open(output_dir / "kubernetes" / "deployment.yaml", "w") as f:
            yaml.dump(k8s_deployment, f, default_flow_style=False)

        k8s_service = GCPTemplates.kubernetes_service_template("inferloop-synthetic")
        with open(output_dir / "kubernetes" / "service.yaml", "w") as f:
            yaml.dump(k8s_service, f, default_flow_style=False)

        # Save deployment script
        with open(output_dir / "scripts" / "deploy.sh", "w") as f:
            f.write(GCPTemplates.deployment_script_template())

        # Make script executable
        (output_dir / "scripts" / "deploy.sh").chmod(0o755)

        # Save Dockerfile
        with open(output_dir / "Dockerfile", "w") as f:
            f.write(GCPTemplates.docker_template())

        # Save Cloud Build config
        with open(output_dir / "cloudbuild.yaml", "w") as f:
            f.write(GCPTemplates.cloudbuild_template())

        # Save example configurations
        examples_dir = output_dir / "examples"
        examples_dir.mkdir(exist_ok=True)

        # Cloud Run example
        cloud_run_example = GCPTemplates.cloud_run_template(
            "inferloop-synthetic-api",
            "gcr.io/my-project/inferloop-synthetic:latest",
            environment={
                "DATABASE_URL": "postgresql://user:pass@host:5432/db",
                "CACHE_ENABLED": "true",
            },
        )
        with open(examples_dir / "cloud_run.yaml", "w") as f:
            yaml.dump(cloud_run_example, f, default_flow_style=False)

        # GKE cluster example
        gke_example = GCPTemplates.gke_cluster_template(
            "inferloop-cluster", "us-central1-a"
        )
        with open(examples_dir / "gke_cluster.json", "w") as f:
            json.dump(gke_example, f, indent=2)

        # Cloud SQL example
        sql_example = GCPTemplates.cloud_sql_template(
            "inferloop-db", tier="db-n1-standard-1", storage_gb=50
        )
        with open(examples_dir / "cloud_sql.json", "w") as f:
            json.dump(sql_example, f, indent=2)
