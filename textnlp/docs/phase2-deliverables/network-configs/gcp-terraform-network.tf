# GCP Network Infrastructure for TextNLP Platform
# Terraform Configuration - Phase 2: Foundation Setup

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 4.0"
    }
  }
  
  backend "gcs" {
    bucket = "textnlp-terraform-state-gcp"
    prefix = "network"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# Variables
variable "project_id" {
  description = "GCP project ID"
  type        = string
  default     = "textnlp-prod-001"
}

variable "region" {
  description = "GCP region for deployment"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

# Local values
locals {
  labels = {
    project     = "textnlp"
    environment = var.environment
    managed-by  = "terraform"
    cost-center = "ai-ml"
  }
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "compute.googleapis.com",
    "container.googleapis.com",
    "servicenetworking.googleapis.com",
    "dns.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "cloudresourcemanager.googleapis.com"
  ])
  
  project = var.project_id
  service = each.value
  
  disable_dependent_services = false
  disable_on_destroy         = false
}

# VPC Network
resource "google_compute_network" "textnlp_vpc" {
  name                    = "textnlp-vpc"
  description             = "TextNLP platform VPC network"
  auto_create_subnetworks = false
  routing_mode           = "REGIONAL"
  mtu                    = 1460
  
  depends_on = [google_project_service.required_apis]
}

# Subnets
resource "google_compute_subnetwork" "app_subnet" {
  name          = "textnlp-app-subnet"
  ip_cidr_range = "10.1.0.0/24"
  region        = var.region
  network       = google_compute_network.textnlp_vpc.id
  description   = "Subnet for TextNLP application workloads"
  
  private_ip_google_access = true
  
  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.1.64.0/18"
  }
  
  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.1.128.0/20"
  }
  
  log_config {
    aggregation_interval = "INTERVAL_5_SEC"
    flow_sampling        = 0.5
    metadata            = "INCLUDE_ALL_METADATA"
  }
}

resource "google_compute_subnetwork" "gpu_subnet" {
  name          = "textnlp-gpu-subnet"
  ip_cidr_range = "10.1.1.0/24"
  region        = var.region
  network       = google_compute_network.textnlp_vpc.id
  description   = "Subnet for GPU-accelerated ML workloads"
  
  private_ip_google_access = true
  
  secondary_ip_range {
    range_name    = "gpu-pods"
    ip_cidr_range = "10.1.192.0/18"
  }
  
  log_config {
    aggregation_interval = "INTERVAL_5_SEC"
    flow_sampling        = 1.0
    metadata            = "INCLUDE_ALL_METADATA"
  }
}

resource "google_compute_subnetwork" "db_subnet" {
  name          = "textnlp-db-subnet"
  ip_cidr_range = "10.1.2.0/24"
  region        = var.region
  network       = google_compute_network.textnlp_vpc.id
  description   = "Subnet for database services"
  
  private_ip_google_access = true
  
  log_config {
    aggregation_interval = "INTERVAL_10_SEC"
    flow_sampling        = 0.1
    metadata            = "INCLUDE_ALL_METADATA"
  }
}

resource "google_compute_subnetwork" "mgmt_subnet" {
  name          = "textnlp-mgmt-subnet"
  ip_cidr_range = "10.1.3.0/24"
  region        = var.region
  network       = google_compute_network.textnlp_vpc.id
  description   = "Subnet for management and monitoring services"
  
  private_ip_google_access = true
}

# Cloud Router for NAT
resource "google_compute_router" "textnlp_router" {
  name    = "textnlp-router"
  region  = var.region
  network = google_compute_network.textnlp_vpc.id
  
  description = "Router for TextNLP platform NAT gateway"
  
  bgp {
    asn            = 64512
    advertise_mode = "DEFAULT"
  }
}

# Cloud NAT
resource "google_compute_router_nat" "textnlp_nat" {
  name   = "textnlp-nat"
  router = google_compute_router.textnlp_router.name
  region = var.region
  
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
  
  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
  
  min_ports_per_vm                 = 64
  max_ports_per_vm                = 65536
  enable_endpoint_independent_mapping = true
}

# Reserved IP ranges for private services
resource "google_compute_global_address" "cloudsql_range" {
  name         = "textnlp-cloudsql-range"
  purpose      = "VPC_PEERING"
  address_type = "INTERNAL"
  prefix_length = 20
  network      = google_compute_network.textnlp_vpc.id
  description  = "Reserved range for Cloud SQL"
}

resource "google_compute_global_address" "memorystore_range" {
  name         = "textnlp-memorystore-range"
  purpose      = "PRIVATE_SERVICE_CONNECT"
  address_type = "INTERNAL"
  prefix_length = 20
  network      = google_compute_network.textnlp_vpc.id
  description  = "Reserved range for Memorystore"
}

# Private service connection for Cloud SQL
resource "google_service_networking_connection" "cloudsql_connection" {
  network                 = google_compute_network.textnlp_vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.cloudsql_range.name]
}

# Firewall Rules
resource "google_compute_firewall" "allow_internal" {
  name    = "textnlp-allow-internal"
  network = google_compute_network.textnlp_vpc.name
  
  description = "Allow internal communication within VPC"
  direction   = "INGRESS"
  priority    = 1000
  
  source_ranges = ["10.1.0.0/16"]
  target_tags   = ["textnlp-internal"]
  
  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }
  
  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }
  
  allow {
    protocol = "icmp"
  }
}

resource "google_compute_firewall" "allow_ssh_iap" {
  name    = "textnlp-allow-ssh-iap"
  network = google_compute_network.textnlp_vpc.name
  
  description = "Allow SSH through Identity-Aware Proxy"
  direction   = "INGRESS"
  priority    = 1000
  
  source_ranges = ["35.235.240.0/20"]
  target_tags   = ["textnlp-ssh"]
  
  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
}

resource "google_compute_firewall" "allow_lb_to_app" {
  name    = "textnlp-allow-lb-to-app"
  network = google_compute_network.textnlp_vpc.name
  
  description = "Allow load balancer traffic to application servers"
  direction   = "INGRESS"
  priority    = 1000
  
  source_tags = ["textnlp-lb"]
  target_tags = ["textnlp-app"]
  
  allow {
    protocol = "tcp"
    ports    = ["80", "443", "8000"]
  }
}

resource "google_compute_firewall" "allow_app_to_gpu" {
  name    = "textnlp-allow-app-to-gpu"
  network = google_compute_network.textnlp_vpc.name
  
  description = "Allow application traffic to GPU workers"
  direction   = "INGRESS"
  priority    = 1000
  
  source_tags = ["textnlp-app"]
  target_tags = ["textnlp-gpu"]
  
  allow {
    protocol = "tcp"
    ports    = ["8080", "50051"]
  }
}

resource "google_compute_firewall" "allow_app_to_db" {
  name    = "textnlp-allow-app-to-db"
  network = google_compute_network.textnlp_vpc.name
  
  description = "Allow application traffic to databases"
  direction   = "INGRESS"
  priority    = 1000
  
  source_tags = ["textnlp-app", "textnlp-gpu"]
  target_tags = ["textnlp-db"]
  
  allow {
    protocol = "tcp"
    ports    = ["5432", "6379"]
  }
}

resource "google_compute_firewall" "allow_health_checks" {
  name    = "textnlp-allow-health-checks"
  network = google_compute_network.textnlp_vpc.name
  
  description = "Allow Google Cloud health checks"
  direction   = "INGRESS"
  priority    = 1000
  
  source_ranges = ["130.211.0.0/22", "35.191.0.0/16"]
  target_tags   = ["textnlp-app", "textnlp-gpu"]
  
  allow {
    protocol = "tcp"
    ports    = ["80", "443", "8000", "8080"]
  }
}

resource "google_compute_firewall" "allow_external_web" {
  name    = "textnlp-allow-external-web"
  network = google_compute_network.textnlp_vpc.name
  
  description = "Allow external HTTP/HTTPS traffic"
  direction   = "INGRESS"
  priority    = 1000
  
  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["textnlp-lb"]
  
  allow {
    protocol = "tcp"
    ports    = ["80", "443"]
  }
}

resource "google_compute_firewall" "deny_all_ingress" {
  name    = "textnlp-deny-all-ingress"
  network = google_compute_network.textnlp_vpc.name
  
  description = "Deny all other ingress traffic"
  direction   = "INGRESS"
  priority    = 65534
  
  source_ranges = ["0.0.0.0/0"]
  
  deny {
    protocol = "all"
  }
}

# Health checks for load balancing
resource "google_compute_health_check" "api_health_check" {
  name        = "textnlp-api-health-check"
  description = "Health check for TextNLP API"
  
  timeout_sec         = 5
  check_interval_sec  = 10
  healthy_threshold   = 2
  unhealthy_threshold = 3
  
  http_health_check {
    port         = 8000
    request_path = "/health"
  }
}

resource "google_compute_health_check" "gpu_health_check" {
  name        = "textnlp-gpu-health-check"
  description = "Health check for TextNLP GPU workers"
  
  timeout_sec         = 10
  check_interval_sec  = 15
  healthy_threshold   = 2
  unhealthy_threshold = 2
  
  http_health_check {
    port         = 8080
    request_path = "/health"
  }
}

# Backend services (will be used by load balancer)
resource "google_compute_backend_service" "api_backend" {
  name        = "textnlp-api-backend"
  description = "Backend service for TextNLP API"
  
  protocol    = "HTTP"
  port_name   = "http"
  timeout_sec = 30
  
  health_checks = [google_compute_health_check.api_health_check.id]
  
  backend {
    balancing_mode  = "UTILIZATION"
    max_utilization = 0.8
    capacity_scaler = 1.0
    # group will be added when instance groups are created
  }
  
  log_config {
    enable      = true
    sample_rate = 1.0
  }
}

resource "google_compute_backend_service" "gpu_backend" {
  name        = "textnlp-gpu-backend"
  description = "Backend service for TextNLP GPU workers"
  
  protocol    = "HTTP"
  port_name   = "gpu-http"
  timeout_sec = 180
  
  health_checks = [google_compute_health_check.gpu_health_check.id]
  
  backend {
    balancing_mode  = "UTILIZATION"
    max_utilization = 0.8
    capacity_scaler = 1.0
    # group will be added when instance groups are created
  }
  
  log_config {
    enable      = true
    sample_rate = 1.0
  }
}

# URL map for load balancer
resource "google_compute_url_map" "textnlp_url_map" {
  name            = "textnlp-url-map"
  description     = "URL map for TextNLP platform"
  default_service = google_compute_backend_service.api_backend.id
  
  path_matcher {
    name            = "gpu-matcher"
    default_service = google_compute_backend_service.gpu_backend.id
    
    path_rule {
      paths   = ["/api/v1/generate", "/api/v1/embed"]
      service = google_compute_backend_service.gpu_backend.id
    }
  }
  
  host_rule {
    hosts        = ["api.textnlp.example.com"]
    path_matcher = "gpu-matcher"
  }
}

# Global forwarding rule (will create load balancer IP)
resource "google_compute_global_address" "lb_ip" {
  name         = "textnlp-lb-ip"
  description  = "Static IP for TextNLP load balancer"
  ip_version   = "IPV4"
  address_type = "EXTERNAL"
}

# SSL certificate (managed)
resource "google_compute_managed_ssl_certificate" "textnlp_ssl" {
  name = "textnlp-ssl-cert"
  
  managed {
    domains = ["api.textnlp.example.com", "textnlp.example.com"]
  }
}

# HTTPS proxy
resource "google_compute_target_https_proxy" "textnlp_https_proxy" {
  name             = "textnlp-https-proxy"
  url_map          = google_compute_url_map.textnlp_url_map.id
  ssl_certificates = [google_compute_managed_ssl_certificate.textnlp_ssl.id]
}

# Global forwarding rule for HTTPS
resource "google_compute_global_forwarding_rule" "textnlp_https_forwarding_rule" {
  name       = "textnlp-https-forwarding-rule"
  target     = google_compute_target_https_proxy.textnlp_https_proxy.id
  port_range = "443"
  ip_address = google_compute_global_address.lb_ip.address
}

# HTTP to HTTPS redirect
resource "google_compute_url_map" "https_redirect" {
  name = "textnlp-https-redirect"
  
  default_url_redirect {
    https_redirect         = true
    redirect_response_code = "MOVED_PERMANENTLY_DEFAULT"
    strip_query            = false
  }
}

resource "google_compute_target_http_proxy" "https_redirect" {
  name    = "textnlp-http-proxy"
  url_map = google_compute_url_map.https_redirect.id
}

resource "google_compute_global_forwarding_rule" "https_redirect" {
  name       = "textnlp-http-forwarding-rule"
  target     = google_compute_target_http_proxy.https_redirect.id
  port_range = "80"
  ip_address = google_compute_global_address.lb_ip.address
}

# Cloud Armor security policy
resource "google_compute_security_policy" "textnlp_security_policy" {
  name        = "textnlp-security-policy"
  description = "Cloud Armor security policy for TextNLP"
  
  # Rate limiting rule
  rule {
    action   = "rate_based_ban"
    priority = 1000
    
    description = "Rate limit per IP"
    
    match {
      versioned_expr = "SRC_IPS_V1"
      
      config {
        src_ip_ranges = ["*"]
      }
    }
    
    rate_limit_options {
      conform_action = "allow"
      exceed_action  = "deny(429)"
      enforce_on_key = "IP"
      
      rate_limit_threshold {
        count        = 100
        interval_sec = 60
      }
    }
  }
  
  # Geographic restrictions
  rule {
    action   = "allow"
    priority = 2000
    
    description = "Allow specific countries"
    
    match {
      expr {
        expression = "origin.region_code == 'US' || origin.region_code == 'CA' || origin.region_code == 'GB'"
      }
    }
  }
  
  # Default rule
  rule {
    action   = "allow"
    priority = 2147483647
    
    description = "Default allow"
    
    match {
      versioned_expr = "SRC_IPS_V1"
      
      config {
        src_ip_ranges = ["*"]
      }
    }
  }
}

# DNS managed zone
resource "google_dns_managed_zone" "textnlp_zone" {
  name        = "textnlp-zone"
  dns_name    = "textnlp.example.com."
  description = "DNS zone for TextNLP platform"
  visibility  = "public"
  
  dnssec_config {
    state = "on"
    
    default_key_specs {
      algorithm  = "rsasha256"
      key_length = 2048
      key_type   = "keySigning"
    }
    
    default_key_specs {
      algorithm  = "rsasha256"
      key_length = 1024
      key_type   = "zoneSigning"
    }
  }
}

# DNS records
resource "google_dns_record_set" "api_a" {
  name = "api.${google_dns_managed_zone.textnlp_zone.dns_name}"
  type = "A"
  ttl  = 300
  
  managed_zone = google_dns_managed_zone.textnlp_zone.name
  
  rrdatas = [google_compute_global_address.lb_ip.address]
}

resource "google_dns_record_set" "root_a" {
  name = google_dns_managed_zone.textnlp_zone.dns_name
  type = "A"
  ttl  = 300
  
  managed_zone = google_dns_managed_zone.textnlp_zone.name
  
  rrdatas = [google_compute_global_address.lb_ip.address]
}

# Monitoring uptime check
resource "google_monitoring_uptime_check_config" "api_uptime" {
  display_name = "TextNLP API Uptime"
  timeout      = "10s"
  period       = "60s"
  
  http_check {
    path           = "/health"
    port           = 443
    use_ssl        = true
    request_method = "GET"
  }
  
  monitored_resource {
    type = "uptime_url"
    
    labels = {
      project_id = var.project_id
      host       = "api.textnlp.example.com"
    }
  }
  
  selected_regions = ["USA", "EUROPE", "ASIA_PACIFIC"]
}

# Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = google_compute_network.textnlp_vpc.id
}

output "vpc_name" {
  description = "Name of the VPC"
  value       = google_compute_network.textnlp_vpc.name
}

output "subnet_ids" {
  description = "IDs of the subnets"
  value = {
    app  = google_compute_subnetwork.app_subnet.id
    gpu  = google_compute_subnetwork.gpu_subnet.id
    db   = google_compute_subnetwork.db_subnet.id
    mgmt = google_compute_subnetwork.mgmt_subnet.id
  }
}

output "load_balancer_ip" {
  description = "IP address of the load balancer"
  value       = google_compute_global_address.lb_ip.address
}

output "dns_name_servers" {
  description = "DNS name servers for the managed zone"
  value       = google_dns_managed_zone.textnlp_zone.name_servers
}

output "cloudsql_connection_name" {
  description = "Cloud SQL private connection name"
  value       = google_service_networking_connection.cloudsql_connection.network
}

output "backend_service_ids" {
  description = "Backend service IDs"
  value = {
    api = google_compute_backend_service.api_backend.id
    gpu = google_compute_backend_service.gpu_backend.id
  }
}