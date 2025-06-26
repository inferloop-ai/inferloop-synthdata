# GCP Database Infrastructure for TextNLP Platform
# Phase 3: Core Infrastructure Deployment

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
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
  project_id = "textnlp-prod-001"
  region     = "us-central1"
  
  labels = {
    project     = "textnlp"
    environment = "production"
    component   = "database"
    platform    = "gcp"
    phase       = "3"
  }
}

# Random password for database
resource "random_password" "postgres_password" {
  length  = 32
  special = true
}

# Service account for Cloud SQL
resource "google_service_account" "cloudsql_sa" {
  account_id   = "textnlp-cloudsql-sa"
  display_name = "TextNLP Cloud SQL Service Account"
  description  = "Service account for TextNLP Cloud SQL operations"
}

# IAM bindings for Cloud SQL service account
resource "google_project_iam_member" "cloudsql_client" {
  project = local.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.cloudsql_sa.email}"
}

# Secret Manager secret for database password
resource "google_secret_manager_secret" "postgres_password" {
  secret_id = "textnlp-postgres-password"
  
  labels = local.labels
  
  replication {
    user_managed {
      replicas {
        location = local.region
      }
      replicas {
        location = "us-east1"
      }
    }
  }
}

resource "google_secret_manager_secret_version" "postgres_password" {
  secret      = google_secret_manager_secret.postgres_password.id
  secret_data = random_password.postgres_password.result
}

# Cloud SQL instance
resource "google_sql_database_instance" "postgres_main" {
  name                = "textnlp-postgres-main"
  database_version    = "POSTGRES_15"
  region              = local.region
  deletion_protection = true
  
  settings {
    tier                        = "db-custom-8-32768"  # 8 vCPU, 32 GB RAM
    availability_type          = "REGIONAL"             # High availability
    disk_type                  = "PD_SSD"
    disk_size                  = 1000
    disk_autoresize           = true
    disk_autoresize_limit     = 5000
    
    # Backup configuration
    backup_configuration {
      enabled                        = true
      start_time                    = "03:00"
      location                      = local.region
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
      
      backup_retention_settings {
        retained_backups = 30
        retention_unit   = "COUNT"
      }
    }
    
    # IP configuration
    ip_configuration {
      ipv4_enabled                                  = false
      private_network                              = data.google_compute_network.vpc.id
      allocated_ip_range                           = "google-managed-services-textnlp-vpc"
      enable_private_path_for_google_cloud_services = true
    }
    
    # Database flags for performance optimization
    database_flags {
      name  = "max_connections"
      value = "1000"
    }
    
    database_flags {
      name  = "shared_buffers"
      value = "8192MB"
    }
    
    database_flags {
      name  = "effective_cache_size"
      value = "24576MB"
    }
    
    database_flags {
      name  = "work_mem"
      value = "32MB"
    }
    
    database_flags {
      name  = "maintenance_work_mem"
      value = "2048MB"
    }
    
    database_flags {
      name  = "checkpoint_completion_target"
      value = "0.9"
    }
    
    database_flags {
      name  = "wal_buffers"
      value = "16MB"
    }
    
    database_flags {
      name  = "default_statistics_target"
      value = "100"
    }
    
    database_flags {
      name  = "random_page_cost"
      value = "1.1"
    }
    
    database_flags {
      name  = "effective_io_concurrency"
      value = "200"
    }
    
    database_flags {
      name  = "log_min_duration_statement"
      value = "1000"
    }
    
    database_flags {
      name  = "log_checkpoints"
      value = "on"
    }
    
    database_flags {
      name  = "log_connections"
      value = "on"
    }
    
    database_flags {
      name  = "log_disconnections"
      value = "on"
    }
    
    database_flags {
      name  = "shared_preload_libraries"
      value = "pg_stat_statements"
    }
    
    # Insights configuration
    insights_config {
      query_insights_enabled  = true
      query_plans_per_minute = 5
      query_string_length    = 1024
      record_application_tags = true
      record_client_address  = true
    }
    
    # Maintenance window
    maintenance_window {
      day          = 7  # Sunday
      hour         = 4
      update_track = "stable"
    }
    
    # User labels
    user_labels = local.labels
  }
  
  # Root password
  root_password = random_password.postgres_password.result
  
  depends_on = [google_service_networking_connection.private_vpc_connection]
}

# Read replica 1 (same region)
resource "google_sql_database_instance" "postgres_replica_1" {
  name                 = "textnlp-postgres-replica-1"
  master_instance_name = google_sql_database_instance.postgres_main.name
  region               = local.region
  database_version     = "POSTGRES_15"
  
  replica_configuration {
    failover_target = false
  }
  
  settings {
    tier              = "db-custom-4-16384"  # 4 vCPU, 16 GB RAM
    availability_type = "ZONAL"
    disk_type         = "PD_SSD"
    disk_size         = 500
    disk_autoresize   = true
    
    # IP configuration
    ip_configuration {
      ipv4_enabled                                  = false
      private_network                              = data.google_compute_network.vpc.id
      enable_private_path_for_google_cloud_services = true
    }
    
    # User labels
    user_labels = merge(local.labels, {
      role = "replica"
    })
  }
}

# Read replica 2 (cross-region)
resource "google_sql_database_instance" "postgres_replica_2" {
  name                 = "textnlp-postgres-replica-2"
  master_instance_name = google_sql_database_instance.postgres_main.name
  region               = "us-east1"
  database_version     = "POSTGRES_15"
  
  replica_configuration {
    failover_target = false
  }
  
  settings {
    tier              = "db-custom-4-16384"  # 4 vCPU, 16 GB RAM
    availability_type = "ZONAL"
    disk_type         = "PD_SSD"
    disk_size         = 500
    disk_autoresize   = true
    
    # IP configuration
    ip_configuration {
      ipv4_enabled                                  = false
      private_network                              = data.google_compute_network.vpc.id
      enable_private_path_for_google_cloud_services = true
    }
    
    # User labels
    user_labels = merge(local.labels, {
      role   = "replica"
      region = "cross-region"
    })
  }
}

# Database creation
resource "google_sql_database" "model_registry" {
  name     = "model_registry"
  instance = google_sql_database_instance.postgres_main.name
}

resource "google_sql_database" "inference_logs" {
  name     = "inference_logs"
  instance = google_sql_database_instance.postgres_main.name
}

resource "google_sql_database" "user_management" {
  name     = "user_management"
  instance = google_sql_database_instance.postgres_main.name
}

resource "google_sql_database" "analytics" {
  name     = "analytics"
  instance = google_sql_database_instance.postgres_main.name
}

resource "google_sql_database" "configuration" {
  name     = "configuration"
  instance = google_sql_database_instance.postgres_main.name
}

# Database users
resource "google_sql_user" "app_user" {
  name     = "textnlp_app"
  instance = google_sql_database_instance.postgres_main.name
  password = random_password.app_user_password.result
}

resource "random_password" "app_user_password" {
  length  = 32
  special = true
}

# Secret for application user
resource "google_secret_manager_secret" "app_user_password" {
  secret_id = "textnlp-app-user-password"
  
  labels = local.labels
  
  replication {
    user_managed {
      replicas {
        location = local.region
      }
    }
  }
}

resource "google_secret_manager_secret_version" "app_user_password" {
  secret      = google_secret_manager_secret.app_user_password.id
  secret_data = random_password.app_user_password.result
}

# Private VPC connection for Cloud SQL
resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = data.google_compute_network.vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]
}

# Private IP allocation for Cloud SQL
resource "google_compute_global_address" "private_ip_address" {
  name          = "textnlp-private-ip-address"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = data.google_compute_network.vpc.id
}

# Cloud SQL Proxy service account
resource "google_service_account" "sql_proxy_sa" {
  account_id   = "textnlp-sql-proxy-sa"
  display_name = "TextNLP SQL Proxy Service Account"
  description  = "Service account for Cloud SQL Proxy connections"
}

resource "google_project_iam_member" "sql_proxy_client" {
  project = local.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.sql_proxy_sa.email}"
}

# Monitoring alert policies
resource "google_monitoring_alert_policy" "database_cpu" {
  display_name = "TextNLP Database High CPU"
  combiner     = "OR"
  
  conditions {
    display_name = "Database CPU utilization is high"
    
    condition_threshold {
      filter         = "resource.type=\"cloudsql_database\" AND resource.labels.database_id=\"${local.project_id}:${google_sql_database_instance.postgres_main.name}\""
      duration       = "300s"
      comparison     = "COMPARISON_GREATER_THAN"
      threshold_value = 0.8
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }
  
  notification_channels = [google_monitoring_notification_channel.email.name]
  
  alert_strategy {
    notification_rate_limit {
      period = "300s"
    }
    auto_close = "1800s"
  }
}

resource "google_monitoring_alert_policy" "database_connections" {
  display_name = "TextNLP Database High Connections"
  combiner     = "OR"
  
  conditions {
    display_name = "Database connection count is high"
    
    condition_threshold {
      filter         = "resource.type=\"cloudsql_database\" AND resource.labels.database_id=\"${local.project_id}:${google_sql_database_instance.postgres_main.name}\""
      duration       = "300s"
      comparison     = "COMPARISON_GREATER_THAN"
      threshold_value = 800
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }
  
  notification_channels = [google_monitoring_notification_channel.email.name]
  
  alert_strategy {
    notification_rate_limit {
      period = "300s"
    }
    auto_close = "1800s"
  }
}

# Notification channel
resource "google_monitoring_notification_channel" "email" {
  display_name = "TextNLP Database Alerts"
  type         = "email"
  
  labels = {
    email_address = "alerts@textnlp.com"
  }
}

# Cloud Functions for automated database maintenance
resource "google_cloudfunctions_function" "db_maintenance" {
  name        = "textnlp-db-maintenance"
  description = "Automated database maintenance tasks"
  runtime     = "python39"
  
  available_memory_mb   = 256
  source_archive_bucket = google_storage_bucket.function_source.name
  source_archive_object = google_storage_bucket_object.db_maintenance_source.name
  trigger {
    event_type = "google.pubsub.topic.publish"
    resource   = google_pubsub_topic.db_maintenance.name
  }
  entry_point = "main"
  
  environment_variables = {
    INSTANCE_CONNECTION_NAME = google_sql_database_instance.postgres_main.connection_name
    DB_NAME                  = "model_registry"
  }
  
  labels = local.labels
}

# Storage bucket for Cloud Function source
resource "google_storage_bucket" "function_source" {
  name     = "${local.project_id}-function-source"
  location = local.region
  
  labels = local.labels
}

# Placeholder for function source
resource "google_storage_bucket_object" "db_maintenance_source" {
  name   = "db-maintenance-source.zip"
  bucket = google_storage_bucket.function_source.name
  source = "/dev/null"  # Replace with actual source zip
}

# Pub/Sub topic for database maintenance scheduling
resource "google_pubsub_topic" "db_maintenance" {
  name = "textnlp-db-maintenance"
  
  labels = local.labels
}

# Cloud Scheduler job for database maintenance
resource "google_cloud_scheduler_job" "db_maintenance" {
  name             = "textnlp-db-maintenance"
  description      = "Trigger database maintenance tasks"
  schedule         = "0 2 * * *"  # Daily at 2 AM
  time_zone        = "UTC"
  attempt_deadline = "300s"
  
  pubsub_target {
    topic_name = google_pubsub_topic.db_maintenance.id
    data       = base64encode("{\"action\": \"maintenance\"}")
  }
}

# Data sources
data "google_compute_network" "vpc" {
  name = "textnlp-vpc"
}

# Outputs
output "instance_connection_name" {
  description = "Cloud SQL instance connection name"
  value       = google_sql_database_instance.postgres_main.connection_name
}

output "instance_ip_address" {
  description = "Cloud SQL instance private IP address"
  value       = google_sql_database_instance.postgres_main.private_ip_address
}

output "replica_1_connection_name" {
  description = "Cloud SQL replica 1 connection name"
  value       = google_sql_database_instance.postgres_replica_1.connection_name
}

output "replica_2_connection_name" {
  description = "Cloud SQL replica 2 connection name"
  value       = google_sql_database_instance.postgres_replica_2.connection_name
}

output "database_names" {
  description = "Created database names"
  value = [
    google_sql_database.model_registry.name,
    google_sql_database.inference_logs.name,
    google_sql_database.user_management.name,
    google_sql_database.analytics.name,
    google_sql_database.configuration.name
  ]
}

output "secret_names" {
  description = "Secret Manager secret names"
  value = {
    postgres_password = google_secret_manager_secret.postgres_password.secret_id
    app_user_password = google_secret_manager_secret.app_user_password.secret_id
  }
  sensitive = true
}

output "service_account_email" {
  description = "Cloud SQL service account email"
  value       = google_service_account.cloudsql_sa.email
}