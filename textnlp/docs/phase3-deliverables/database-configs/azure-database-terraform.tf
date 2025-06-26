# Azure Database Infrastructure for TextNLP Platform
# Phase 3: Core Infrastructure Deployment

terraform {
  required_version = ">= 1.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

# Configure the Azure Provider
provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy    = true
      recover_soft_deleted_key_vaults = true
    }
  }
}

# Local variables
locals {
  resource_group_name = "textnlp-prod-rg"
  location           = "East US"
  
  tags = {
    Project     = "TextNLP"
    Environment = "Production"
    Component   = "Database"
    Platform    = "Azure"
    Phase       = "3"
  }
}

# Random password for PostgreSQL
resource "random_password" "postgres_password" {
  length  = 32
  special = true
}

# Random password for application user
resource "random_password" "app_user_password" {
  length  = 32
  special = true
}

# Data source for resource group
data "azurerm_resource_group" "main" {
  name = local.resource_group_name
}

# Data source for virtual network
data "azurerm_virtual_network" "main" {
  name                = "textnlp-vnet"
  resource_group_name = local.resource_group_name
}

# Data source for database subnet
data "azurerm_subnet" "database" {
  name                 = "database-subnet"
  virtual_network_name = data.azurerm_virtual_network.main.name
  resource_group_name  = local.resource_group_name
}

# Key Vault for storing database secrets
resource "azurerm_key_vault" "database_kv" {
  name                = "textnlp-db-kv"
  location            = data.azurerm_resource_group.main.location
  resource_group_name = data.azurerm_resource_group.main.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "premium"
  
  purge_protection_enabled   = true
  soft_delete_retention_days = 7
  
  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id
    
    secret_permissions = [
      "Get",
      "List",
      "Set",
      "Delete",
      "Purge",
      "Recover"
    ]
    
    key_permissions = [
      "Get",
      "List",
      "Create",
      "Delete",
      "Update",
      "Purge",
      "Recover"
    ]
  }
  
  network_acls {
    default_action = "Deny"
    bypass         = "AzureServices"
    
    virtual_network_subnet_ids = [
      data.azurerm_subnet.database.id
    ]
  }
  
  tags = local.tags
}

# Key Vault secrets
resource "azurerm_key_vault_secret" "postgres_admin_password" {
  name         = "postgres-admin-password"
  value        = random_password.postgres_password.result
  key_vault_id = azurerm_key_vault.database_kv.id
  
  tags = local.tags
}

resource "azurerm_key_vault_secret" "app_user_password" {
  name         = "postgres-app-user-password"
  value        = random_password.app_user_password.result
  key_vault_id = azurerm_key_vault.database_kv.id
  
  tags = local.tags
}

# Private DNS zone for PostgreSQL
resource "azurerm_private_dns_zone" "postgres" {
  name                = "privatelink.postgres.database.azure.com"
  resource_group_name = data.azurerm_resource_group.main.name
  
  tags = local.tags
}

# Link private DNS zone to virtual network
resource "azurerm_private_dns_zone_virtual_network_link" "postgres" {
  name                  = "textnlp-postgres-dns-link"
  resource_group_name   = data.azurerm_resource_group.main.name
  private_dns_zone_name = azurerm_private_dns_zone.postgres.name
  virtual_network_id    = data.azurerm_virtual_network.main.id
  
  tags = local.tags
}

# PostgreSQL Flexible Server
resource "azurerm_postgresql_flexible_server" "main" {
  name                   = "textnlp-postgres-main"
  resource_group_name    = data.azurerm_resource_group.main.name
  location               = data.azurerm_resource_group.main.location
  version                = "15"
  delegated_subnet_id    = data.azurerm_subnet.database.id
  private_dns_zone_id    = azurerm_private_dns_zone.postgres.id
  administrator_login    = "textnlp_admin"
  administrator_password = random_password.postgres_password.result
  zone                   = "1"
  
  storage_mb   = 1048576  # 1 TB
  storage_tier = "P30"
  
  sku_name   = "GP_Standard_D8s_v3"  # 8 vCPU, 32 GB RAM
  
  backup_retention_days        = 30
  geo_redundant_backup_enabled = true
  
  high_availability {
    mode                      = "ZoneRedundant"
    standby_availability_zone = "2"
  }
  
  maintenance_window {
    day_of_week  = 0    # Sunday
    start_hour   = 4
    start_minute = 0
  }
  
  tags = local.tags
  
  depends_on = [azurerm_private_dns_zone_virtual_network_link.postgres]
}

# PostgreSQL server configurations
resource "azurerm_postgresql_flexible_server_configuration" "max_connections" {
  name      = "max_connections"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "1000"
}

resource "azurerm_postgresql_flexible_server_configuration" "shared_buffers" {
  name      = "shared_buffers"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "2097152"  # 8GB in 8KB pages
}

resource "azurerm_postgresql_flexible_server_configuration" "effective_cache_size" {
  name      = "effective_cache_size"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "6291456"  # 24GB in 8KB pages
}

resource "azurerm_postgresql_flexible_server_configuration" "work_mem" {
  name      = "work_mem"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "32768"    # 32MB in KB
}

resource "azurerm_postgresql_flexible_server_configuration" "maintenance_work_mem" {
  name      = "maintenance_work_mem"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "2097152"  # 2GB in KB
}

resource "azurerm_postgresql_flexible_server_configuration" "checkpoint_completion_target" {
  name      = "checkpoint_completion_target"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "0.9"
}

resource "azurerm_postgresql_flexible_server_configuration" "wal_buffers" {
  name      = "wal_buffers"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "2048"     # 16MB in 8KB pages
}

resource "azurerm_postgresql_flexible_server_configuration" "default_statistics_target" {
  name      = "default_statistics_target"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "100"
}

resource "azurerm_postgresql_flexible_server_configuration" "random_page_cost" {
  name      = "random_page_cost"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "1.1"
}

resource "azurerm_postgresql_flexible_server_configuration" "effective_io_concurrency" {
  name      = "effective_io_concurrency"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "200"
}

resource "azurerm_postgresql_flexible_server_configuration" "log_min_duration_statement" {
  name      = "log_min_duration_statement"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "1000"
}

resource "azurerm_postgresql_flexible_server_configuration" "log_checkpoints" {
  name      = "log_checkpoints"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "on"
}

resource "azurerm_postgresql_flexible_server_configuration" "log_connections" {
  name      = "log_connections"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "on"
}

resource "azurerm_postgresql_flexible_server_configuration" "log_disconnections" {
  name      = "log_disconnections"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "on"
}

resource "azurerm_postgresql_flexible_server_configuration" "shared_preload_libraries" {
  name      = "shared_preload_libraries"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "pg_stat_statements"
}

# Read replica 1 (same region)
resource "azurerm_postgresql_flexible_server" "replica_1" {
  name                = "textnlp-postgres-replica-1"
  resource_group_name = data.azurerm_resource_group.main.name
  location            = data.azurerm_resource_group.main.location
  version             = "15"
  
  delegated_subnet_id    = data.azurerm_subnet.database.id
  private_dns_zone_id    = azurerm_private_dns_zone.postgres.id
  source_server_id       = azurerm_postgresql_flexible_server.main.id
  create_mode            = "Replica"
  
  sku_name = "GP_Standard_D4s_v3"  # 4 vCPU, 16 GB RAM
  
  storage_mb   = 524288  # 512 GB
  storage_tier = "P20"
  
  tags = merge(local.tags, {
    Role = "ReadReplica"
  })
}

# Read replica 2 (cross-region)
resource "azurerm_postgresql_flexible_server" "replica_2" {
  name                = "textnlp-postgres-replica-2"
  resource_group_name = data.azurerm_resource_group.main.name
  location            = "West US 2"
  version             = "15"
  
  source_server_id = azurerm_postgresql_flexible_server.main.id
  create_mode      = "Replica"
  
  sku_name = "GP_Standard_D4s_v3"  # 4 vCPU, 16 GB RAM
  
  storage_mb   = 524288  # 512 GB
  storage_tier = "P20"
  
  tags = merge(local.tags, {
    Role     = "ReadReplica"
    Location = "CrossRegion"
  })
}

# PostgreSQL databases
resource "azurerm_postgresql_flexible_server_database" "model_registry" {
  name      = "model_registry"
  server_id = azurerm_postgresql_flexible_server.main.id
  collation = "en_US.utf8"
  charset   = "utf8"
}

resource "azurerm_postgresql_flexible_server_database" "inference_logs" {
  name      = "inference_logs"
  server_id = azurerm_postgresql_flexible_server.main.id
  collation = "en_US.utf8"
  charset   = "utf8"
}

resource "azurerm_postgresql_flexible_server_database" "user_management" {
  name      = "user_management"
  server_id = azurerm_postgresql_flexible_server.main.id
  collation = "en_US.utf8"
  charset   = "utf8"
}

resource "azurerm_postgresql_flexible_server_database" "analytics" {
  name      = "analytics"
  server_id = azurerm_postgresql_flexible_server.main.id
  collation = "en_US.utf8"
  charset   = "utf8"
}

resource "azurerm_postgresql_flexible_server_database" "configuration" {
  name      = "configuration"
  server_id = azurerm_postgresql_flexible_server.main.id
  collation = "en_US.utf8"
  charset   = "utf8"
}

# Log Analytics workspace for monitoring
resource "azurerm_log_analytics_workspace" "database_logs" {
  name                = "textnlp-database-logs"
  location            = data.azurerm_resource_group.main.location
  resource_group_name = data.azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  
  tags = local.tags
}

# Diagnostic settings for PostgreSQL server
resource "azurerm_monitor_diagnostic_setting" "postgres_main" {
  name                       = "textnlp-postgres-diagnostics"
  target_resource_id         = azurerm_postgresql_flexible_server.main.id
  log_analytics_workspace_id = azurerm_log_analytics_workspace.database_logs.id
  
  enabled_log {
    category = "PostgreSQLLogs"
  }
  
  metric {
    category = "AllMetrics"
    enabled  = true
  }
}

# Action group for alerts
resource "azurerm_monitor_action_group" "database_alerts" {
  name                = "textnlp-database-alerts"
  resource_group_name = data.azurerm_resource_group.main.name
  short_name          = "dbAlerts"
  
  email_receiver {
    name                    = "admin"
    email_address          = "admin@textnlp.com"
    use_common_alert_schema = true
  }
  
  tags = local.tags
}

# CPU utilization alert
resource "azurerm_monitor_metric_alert" "database_cpu" {
  name                = "textnlp-database-high-cpu"
  resource_group_name = data.azurerm_resource_group.main.name
  scopes              = [azurerm_postgresql_flexible_server.main.id]
  description         = "Alert when database CPU utilization is high"
  severity            = 2
  frequency           = "PT5M"
  window_size         = "PT5M"
  
  criteria {
    metric_namespace = "Microsoft.DBforPostgreSQL/flexibleServers"
    metric_name      = "cpu_percent"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = 80
  }
  
  action {
    action_group_id = azurerm_monitor_action_group.database_alerts.id
  }
  
  tags = local.tags
}

# Connection count alert
resource "azurerm_monitor_metric_alert" "database_connections" {
  name                = "textnlp-database-high-connections"
  resource_group_name = data.azurerm_resource_group.main.name
  scopes              = [azurerm_postgresql_flexible_server.main.id]
  description         = "Alert when database connection count is high"
  severity            = 2
  frequency           = "PT5M"
  window_size         = "PT5M"
  
  criteria {
    metric_namespace = "Microsoft.DBforPostgreSQL/flexibleServers"
    metric_name      = "connections_used"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = 800
  }
  
  action {
    action_group_id = azurerm_monitor_action_group.database_alerts.id
  }
  
  tags = local.tags
}

# Storage utilization alert
resource "azurerm_monitor_metric_alert" "database_storage" {
  name                = "textnlp-database-high-storage"
  resource_group_name = data.azurerm_resource_group.main.name
  scopes              = [azurerm_postgresql_flexible_server.main.id]
  description         = "Alert when database storage utilization is high"
  severity            = 1
  frequency           = "PT15M"
  window_size         = "PT15M"
  
  criteria {
    metric_namespace = "Microsoft.DBforPostgreSQL/flexibleServers"
    metric_name      = "storage_percent"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = 85
  }
  
  action {
    action_group_id = azurerm_monitor_action_group.database_alerts.id
  }
  
  tags = local.tags
}

# Azure Automation account for database maintenance
resource "azurerm_automation_account" "database_automation" {
  name                = "textnlp-database-automation"
  location            = data.azurerm_resource_group.main.location
  resource_group_name = data.azurerm_resource_group.main.name
  sku_name            = "Basic"
  
  tags = local.tags
}

# PowerShell runbook for database maintenance
resource "azurerm_automation_runbook" "database_maintenance" {
  name                    = "DatabaseMaintenance"
  location                = data.azurerm_resource_group.main.location
  resource_group_name     = data.azurerm_resource_group.main.name
  automation_account_name = azurerm_automation_account.database_automation.name
  log_verbose             = true
  log_progress            = true
  runbook_type            = "PowerShell"
  
  content = <<-EOT
    param(
        [Parameter(Mandatory=$true)]
        [string]$ServerName,
        
        [Parameter(Mandatory=$true)]
        [string]$DatabaseName
    )
    
    Write-Output "Starting database maintenance for $ServerName/$DatabaseName"
    
    # Database maintenance tasks would go here
    # - Update statistics
    # - Rebuild indexes
    # - Cleanup old data
    
    Write-Output "Database maintenance completed"
  EOT
  
  tags = local.tags
}

# Schedule for database maintenance
resource "azurerm_automation_schedule" "database_maintenance" {
  name                    = "DatabaseMaintenanceSchedule"
  resource_group_name     = data.azurerm_resource_group.main.name
  automation_account_name = azurerm_automation_account.database_automation.name
  frequency               = "Day"
  interval                = 1
  timezone                = "UTC"
  start_time              = "2024-01-01T02:00:00Z"
  description             = "Daily database maintenance schedule"
}

# Data sources
data "azurerm_client_config" "current" {}

# Outputs
output "postgres_server_fqdn" {
  description = "PostgreSQL server FQDN"
  value       = azurerm_postgresql_flexible_server.main.fqdn
}

output "postgres_replica_1_fqdn" {
  description = "PostgreSQL replica 1 FQDN"
  value       = azurerm_postgresql_flexible_server.replica_1.fqdn
}

output "postgres_replica_2_fqdn" {
  description = "PostgreSQL replica 2 FQDN"
  value       = azurerm_postgresql_flexible_server.replica_2.fqdn
}

output "database_names" {
  description = "Created database names"
  value = [
    azurerm_postgresql_flexible_server_database.model_registry.name,
    azurerm_postgresql_flexible_server_database.inference_logs.name,
    azurerm_postgresql_flexible_server_database.user_management.name,
    azurerm_postgresql_flexible_server_database.analytics.name,
    azurerm_postgresql_flexible_server_database.configuration.name
  ]
}

output "key_vault_id" {
  description = "Key Vault ID for database secrets"
  value       = azurerm_key_vault.database_kv.id
}

output "log_analytics_workspace_id" {
  description = "Log Analytics workspace ID"
  value       = azurerm_log_analytics_workspace.database_logs.id
}