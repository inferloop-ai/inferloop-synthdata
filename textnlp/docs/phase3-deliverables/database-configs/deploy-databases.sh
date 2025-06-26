#!/bin/bash

# Database Deployment Script for TextNLP Platform
# Phase 3: Core Infrastructure Deployment
# This script deploys database infrastructure across all platforms

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
LOG_FILE="${SCRIPT_DIR}/database-deployment.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOG_FILE}"
}

info() {
    log "INFO" "${BLUE}$*${NC}"
}

warn() {
    log "WARN" "${YELLOW}$*${NC}"
}

error() {
    log "ERROR" "${RED}$*${NC}"
}

success() {
    log "SUCCESS" "${GREEN}$*${NC}"
}

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check for required tools
    if ! command -v terraform &> /dev/null; then
        missing_tools+=("terraform")
    fi
    
    if ! command -v kubectl &> /dev/null; then
        missing_tools+=("kubectl")
    fi
    
    if ! command -v aws &> /dev/null; then
        missing_tools+=("aws")
    fi
    
    if ! command -v gcloud &> /dev/null; then
        missing_tools+=("gcloud")
    fi
    
    if ! command -v az &> /dev/null; then
        missing_tools+=("az")
    fi
    
    if ! command -v psql &> /dev/null; then
        missing_tools+=("postgresql-client")
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        error "Missing required tools: ${missing_tools[*]}"
        error "Please install the missing tools and try again."
        exit 1
    fi
    
    success "All prerequisites satisfied"
}

# Load environment configuration
load_environment() {
    local env_file="${SCRIPT_DIR}/.env"
    
    if [ -f "${env_file}" ]; then
        info "Loading environment configuration from ${env_file}"
        set -a
        source "${env_file}"
        set +a
    else
        warn "Environment file not found: ${env_file}"
        warn "Please create .env file with required variables"
    fi
    
    # Set defaults if not provided
    export DEPLOYMENT_PLATFORM="${DEPLOYMENT_PLATFORM:-all}"
    export ENVIRONMENT="${ENVIRONMENT:-production}"
    export AWS_REGION="${AWS_REGION:-us-east-1}"
    export GCP_PROJECT="${GCP_PROJECT:-textnlp-prod-001}"
    export GCP_REGION="${GCP_REGION:-us-central1}"
    export AZURE_RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-textnlp-prod-rg}"
    export AZURE_LOCATION="${AZURE_LOCATION:-eastus}"
}

# Validate Terraform configuration
validate_terraform() {
    local platform=$1
    local terraform_dir="${SCRIPT_DIR}"
    
    info "Validating Terraform configuration for ${platform}..."
    
    cd "${terraform_dir}"
    
    case $platform in
        aws)
            terraform init
            terraform validate -var-file="${platform}-database-terraform.tf"
            ;;
        gcp)
            terraform init
            terraform validate -var-file="${platform}-database-terraform.tf"
            ;;
        azure)
            terraform init
            terraform validate -var-file="${platform}-database-terraform.tf"
            ;;
        *)
            error "Unknown platform: ${platform}"
            return 1
            ;;
    esac
    
    success "Terraform validation passed for ${platform}"
}

# Deploy AWS database infrastructure
deploy_aws_database() {
    info "Deploying AWS database infrastructure..."
    
    # Check AWS authentication
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS authentication failed. Please configure AWS credentials."
        return 1
    fi
    
    cd "${SCRIPT_DIR}"
    
    # Initialize Terraform
    terraform init
    
    # Plan deployment
    info "Creating Terraform plan for AWS..."
    terraform plan \
        -var-file="aws-variables.tfvars" \
        -out="aws-database.tfplan" \
        aws-database-terraform.tf
    
    # Apply deployment
    info "Applying Terraform plan for AWS..."
    terraform apply "aws-database.tfplan"
    
    # Get outputs
    local cluster_endpoint=$(terraform output -raw cluster_endpoint)
    local proxy_endpoint=$(terraform output -raw proxy_endpoint)
    local secret_arn=$(terraform output -raw secret_arn)
    
    info "AWS database deployment completed:"
    info "  Cluster Endpoint: ${cluster_endpoint}"
    info "  Proxy Endpoint: ${proxy_endpoint}"
    info "  Secret ARN: ${secret_arn}"
    
    # Test connectivity
    test_database_connectivity "aws" "${proxy_endpoint}"
    
    success "AWS database infrastructure deployed successfully"
}

# Deploy GCP database infrastructure
deploy_gcp_database() {
    info "Deploying GCP database infrastructure..."
    
    # Check GCP authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 &> /dev/null; then
        error "GCP authentication failed. Please run 'gcloud auth login'."
        return 1
    fi
    
    # Set project
    gcloud config set project "${GCP_PROJECT}"
    
    cd "${SCRIPT_DIR}"
    
    # Initialize Terraform
    terraform init
    
    # Plan deployment
    info "Creating Terraform plan for GCP..."
    terraform plan \
        -var-file="gcp-variables.tfvars" \
        -out="gcp-database.tfplan" \
        gcp-database-terraform.tf
    
    # Apply deployment
    info "Applying Terraform plan for GCP..."
    terraform apply "gcp-database.tfplan"
    
    # Get outputs
    local connection_name=$(terraform output -raw instance_connection_name)
    local ip_address=$(terraform output -raw instance_ip_address)
    
    info "GCP database deployment completed:"
    info "  Connection Name: ${connection_name}"
    info "  IP Address: ${ip_address}"
    
    # Test connectivity
    test_database_connectivity "gcp" "${connection_name}"
    
    success "GCP database infrastructure deployed successfully"
}

# Deploy Azure database infrastructure
deploy_azure_database() {
    info "Deploying Azure database infrastructure..."
    
    # Check Azure authentication
    if ! az account show &> /dev/null; then
        error "Azure authentication failed. Please run 'az login'."
        return 1
    fi
    
    cd "${SCRIPT_DIR}"
    
    # Initialize Terraform
    terraform init
    
    # Plan deployment
    info "Creating Terraform plan for Azure..."
    terraform plan \
        -var-file="azure-variables.tfvars" \
        -out="azure-database.tfplan" \
        azure-database-terraform.tf
    
    # Apply deployment
    info "Applying Terraform plan for Azure..."
    terraform apply "azure-database.tfplan"
    
    # Get outputs
    local server_fqdn=$(terraform output -raw postgres_server_fqdn)
    local key_vault_id=$(terraform output -raw key_vault_id)
    
    info "Azure database deployment completed:"
    info "  Server FQDN: ${server_fqdn}"
    info "  Key Vault ID: ${key_vault_id}"
    
    # Test connectivity
    test_database_connectivity "azure" "${server_fqdn}"
    
    success "Azure database infrastructure deployed successfully"
}

# Deploy on-premises database infrastructure
deploy_onprem_database() {
    info "Deploying on-premises database infrastructure..."
    
    # Check if we're running on the target environment
    if [ "${ENVIRONMENT}" != "onpremises" ]; then
        warn "On-premises deployment should be run from the target environment"
    fi
    
    # Deploy Patroni configuration
    deploy_patroni_cluster
    
    # Deploy HAProxy configuration
    deploy_haproxy_config
    
    # Deploy PgBouncer configuration
    deploy_pgbouncer_config
    
    # Initialize database schemas
    initialize_database_schemas
    
    # Configure monitoring
    configure_database_monitoring
    
    success "On-premises database infrastructure deployed successfully"
}

# Deploy Patroni cluster
deploy_patroni_cluster() {
    info "Deploying Patroni PostgreSQL cluster..."
    
    # Create Patroni configuration files
    local patroni_config="${SCRIPT_DIR}/patroni-config"
    mkdir -p "${patroni_config}"
    
    # Generate Patroni configuration for each node
    for node in postgres-master-01 postgres-standby-01 postgres-standby-02; do
        info "Generating Patroni configuration for ${node}..."
        
        # Extract node-specific configuration from YAML
        python3 -c "
import yaml
import sys

with open('${SCRIPT_DIR}/onprem-database-setup.yaml', 'r') as f:
    config = yaml.safe_load(f)

node_config = config['patroni_config']['nodes']['${node}']
global_config = config['patroni_config']['global']

patroni_yaml = {
    'scope': global_config['scope'],
    'namespace': global_config['namespace'],
    'name': '${node}',
    'restapi': node_config['restapi'],
    'etcd': global_config['etcd'],
    'bootstrap': global_config['bootstrap'] if '${node}' == 'postgres-master-01' else None,
    'postgresql': node_config['postgresql'],
    'watchdog': node_config.get('watchdog', {})
}

# Remove None values
patroni_yaml = {k: v for k, v in patroni_yaml.items() if v is not None}

with open('${patroni_config}/${node}.yml', 'w') as f:
    yaml.dump(patroni_yaml, f, default_flow_style=False)
"
        
        success "Generated Patroni configuration for ${node}"
    done
    
    # Deploy configurations to nodes
    for node in postgres-master-01 postgres-standby-01 postgres-standby-02; do
        info "Deploying Patroni configuration to ${node}..."
        
        # Copy configuration file
        scp "${patroni_config}/${node}.yml" "${node}:/etc/patroni/patroni.yml"
        
        # Start Patroni service
        ssh "${node}" "
            sudo systemctl enable patroni
            sudo systemctl restart patroni
            sudo systemctl status patroni
        "
        
        success "Patroni deployed on ${node}"
    done
}

# Deploy HAProxy configuration
deploy_haproxy_config() {
    info "Deploying HAProxy load balancer configuration..."
    
    # Generate HAProxy configuration
    python3 -c "
import yaml

with open('${SCRIPT_DIR}/onprem-database-setup.yaml', 'r') as f:
    config = yaml.safe_load(f)

haproxy_config = config['haproxy_config']

# Generate haproxy.cfg
haproxy_cfg = '''
global
    maxconn 4096
    log 127.0.0.1:514 local0
    chroot /var/lib/haproxy
    stats socket /run/haproxy/admin.sock mode 660 level admin
    stats timeout 30s
    user haproxy
    group haproxy
    daemon

defaults
    mode tcp
    log global
    option tcplog
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    timeout check 5000ms
    retries 3

# Statistics interface
stats enable
stats uri /haproxy-stats
stats refresh 30s
stats admin if TRUE

# PostgreSQL write frontend
frontend postgres_write
    bind 10.100.1.70:5432
    default_backend postgres_write_backend
    maxconn 1000
    option tcpka

# PostgreSQL write backend
backend postgres_write_backend
    balance first
    option tcp-check
    tcp-check expect string master
    server postgres-master 10.100.1.71:5432 check port 8008 httpchk GET /master
    server postgres-standby-01 10.100.1.72:5432 check port 8008 httpchk GET /master backup
    server postgres-standby-02 10.100.1.73:5432 check port 8008 httpchk GET /master backup

# PostgreSQL read frontend
frontend postgres_read
    bind 10.100.1.70:5433
    default_backend postgres_read_backend
    maxconn 2000
    option tcpka

# PostgreSQL read backend
backend postgres_read_backend
    balance roundrobin
    option tcp-check
    tcp-check expect string replica
    server postgres-standby-01 10.100.1.72:5432 check port 8008 httpchk GET /replica weight 100
    server postgres-standby-02 10.100.1.73:5432 check port 8008 httpchk GET /replica weight 100
    server postgres-master 10.100.1.71:5432 check port 8008 httpchk GET /replica backup weight 50
'''

with open('${SCRIPT_DIR}/haproxy.cfg', 'w') as f:
    f.write(haproxy_cfg)
"
    
    # Deploy HAProxy configuration
    sudo cp "${SCRIPT_DIR}/haproxy.cfg" /etc/haproxy/haproxy.cfg
    sudo systemctl enable haproxy
    sudo systemctl restart haproxy
    sudo systemctl status haproxy
    
    success "HAProxy configuration deployed"
}

# Deploy PgBouncer configuration
deploy_pgbouncer_config() {
    info "Deploying PgBouncer connection pooling configuration..."
    
    # Generate PgBouncer configuration
    python3 -c "
import yaml

with open('${SCRIPT_DIR}/onprem-database-setup.yaml', 'r') as f:
    config = yaml.safe_load(f)

pgbouncer_config = config['pgbouncer_config']['global']

# Generate pgbouncer.ini
pgbouncer_ini = '''
[databases]
model_registry = host=10.100.1.70 port=5432 dbname=model_registry pool_size=20
inference_logs = host=10.100.1.70 port=5432 dbname=inference_logs pool_size=30
user_management = host=10.100.1.70 port=5432 dbname=user_management pool_size=10
analytics = host=10.100.1.70 port=5432 dbname=analytics pool_size=15
configuration = host=10.100.1.70 port=5432 dbname=configuration pool_size=5
model_registry_ro = host=10.100.1.70 port=5433 dbname=model_registry pool_size=15
analytics_ro = host=10.100.1.70 port=5433 dbname=analytics pool_size=20

[pgbouncer]
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
min_pool_size = 10
reserve_pool_size = 5
reserve_pool_timeout = 5
max_db_connections = 100
max_user_connections = 100
server_reset_query = DISCARD ALL
server_check_query = SELECT 1
server_check_delay = 30
server_connect_timeout = 15
server_login_retry = 15
client_login_timeout = 60
autodb_idle_timeout = 3600
server_idle_timeout = 600
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
admin_users = textnlp_admin
stats_users = textnlp_admin,stats_user
listen_addr = *
listen_port = 6432
unix_socket_dir = /var/run/postgresql
unix_socket_mode = 0777
unix_socket_group = postgres
log_connections = 1
log_disconnections = 1
log_pooler_errors = 1
syslog = 1
syslog_facility = daemon
syslog_ident = pgbouncer
pidfile = /var/run/pgbouncer/pgbouncer.pid
logfile = /var/log/pgbouncer/pgbouncer.log
'''

with open('${SCRIPT_DIR}/pgbouncer.ini', 'w') as f:
    f.write(pgbouncer_ini)
"
    
    # Deploy PgBouncer on both nodes
    for node in pgbouncer-01 pgbouncer-02; do
        info "Deploying PgBouncer on ${node}..."
        
        scp "${SCRIPT_DIR}/pgbouncer.ini" "${node}:/etc/pgbouncer/pgbouncer.ini"
        
        ssh "${node}" "
            sudo systemctl enable pgbouncer
            sudo systemctl restart pgbouncer
            sudo systemctl status pgbouncer
        "
        
        success "PgBouncer deployed on ${node}"
    done
}

# Initialize database schemas
initialize_database_schemas() {
    info "Initializing database schemas..."
    
    # Extract and execute SQL scripts
    python3 -c "
import yaml

with open('${SCRIPT_DIR}/onprem-database-setup.yaml', 'r') as f:
    config = yaml.safe_load(f)

init_scripts = config['schema_initialization']['init_scripts']

for script_name, script_config in init_scripts.items():
    with open('${SCRIPT_DIR}/${script_config[\"file\"]}', 'w') as f:
        f.write(script_config['content'])
"
    
    # Execute SQL scripts in order
    for script in 01_extensions.sql 02_schemas.sql 03_tables.sql 04_roles.sql; do
        info "Executing ${script}..."
        PGPASSWORD="${POSTGRES_PASSWORD}" psql -h 10.100.1.70 -U textnlp_admin -d postgres -f "${SCRIPT_DIR}/${script}"
        success "Executed ${script}"
    done
    
    success "Database schemas initialized"
}

# Configure database monitoring
configure_database_monitoring() {
    info "Configuring database monitoring..."
    
    # Deploy Prometheus exporters
    for node in postgres-master-01 postgres-standby-01 postgres-standby-02; do
        info "Deploying postgres_exporter on ${node}..."
        
        ssh "${node}" "
            # Download and install postgres_exporter
            wget -q https://github.com/prometheus-community/postgres_exporter/releases/download/v0.15.0/postgres_exporter-0.15.0.linux-amd64.tar.gz
            tar xzf postgres_exporter-0.15.0.linux-amd64.tar.gz
            sudo mv postgres_exporter-0.15.0.linux-amd64/postgres_exporter /usr/local/bin/
            
            # Create systemd service
            sudo tee /etc/systemd/system/postgres_exporter.service > /dev/null <<EOF
[Unit]
Description=Postgres Exporter
After=network.target

[Service]
Type=simple
User=postgres
Group=postgres
ExecStart=/usr/local/bin/postgres_exporter
Environment=DATA_SOURCE_NAME=postgresql://postgres_exporter:${EXPORTER_PASSWORD}@localhost:5432/postgres?sslmode=disable
Restart=always

[Install]
WantedBy=multi-user.target
EOF
            
            sudo systemctl daemon-reload
            sudo systemctl enable postgres_exporter
            sudo systemctl start postgres_exporter
        "
        
        success "postgres_exporter deployed on ${node}"
    done
    
    # Deploy HAProxy exporter
    info "Deploying haproxy_exporter..."
    wget -q https://github.com/prometheus/haproxy_exporter/releases/download/v0.15.0/haproxy_exporter-0.15.0.linux-amd64.tar.gz
    tar xzf haproxy_exporter-0.15.0.linux-amd64.tar.gz
    sudo mv haproxy_exporter-0.15.0.linux-amd64/haproxy_exporter /usr/local/bin/
    
    # Create systemd service for HAProxy exporter
    sudo tee /etc/systemd/system/haproxy_exporter.service > /dev/null <<EOF
[Unit]
Description=HAProxy Exporter
After=network.target

[Service]
Type=simple
User=haproxy
Group=haproxy
ExecStart=/usr/local/bin/haproxy_exporter --haproxy.scrape-uri=http://admin:${HAPROXY_STATS_PASSWORD}@localhost:8080/haproxy-stats;csv
Restart=always

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable haproxy_exporter
    sudo systemctl start haproxy_exporter
    
    success "Database monitoring configured"
}

# Test database connectivity
test_database_connectivity() {
    local platform=$1
    local endpoint=$2
    
    info "Testing database connectivity for ${platform}..."
    
    case $platform in
        aws)
            # Test with AWS RDS Proxy
            if command -v psql &> /dev/null; then
                if PGPASSWORD="${DB_PASSWORD}" psql -h "${endpoint}" -U textnlp_admin -d textnlp -c "SELECT version();" &> /dev/null; then
                    success "Database connectivity test passed for ${platform}"
                else
                    warn "Database connectivity test failed for ${platform}"
                fi
            fi
            ;;
        gcp)
            # Test with Cloud SQL Proxy
            if command -v cloud_sql_proxy &> /dev/null; then
                cloud_sql_proxy -instances="${endpoint}"=tcp:5432 &
                local proxy_pid=$!
                sleep 5
                
                if PGPASSWORD="${DB_PASSWORD}" psql -h localhost -p 5432 -U postgres -d postgres -c "SELECT version();" &> /dev/null; then
                    success "Database connectivity test passed for ${platform}"
                else
                    warn "Database connectivity test failed for ${platform}"
                fi
                
                kill ${proxy_pid}
            fi
            ;;
        azure)
            # Test with Azure PostgreSQL
            if command -v psql &> /dev/null; then
                if PGPASSWORD="${DB_PASSWORD}" psql -h "${endpoint}" -U textnlp_admin -d postgres -c "SELECT version();" &> /dev/null; then
                    success "Database connectivity test passed for ${platform}"
                else
                    warn "Database connectivity test failed for ${platform}"
                fi
            fi
            ;;
        onprem)
            # Test with on-premises setup
            if PGPASSWORD="${POSTGRES_PASSWORD}" psql -h 10.100.1.70 -U textnlp_admin -d postgres -c "SELECT version();" &> /dev/null; then
                success "Database connectivity test passed for ${platform}"
            else
                warn "Database connectivity test failed for ${platform}"
            fi
            ;;
    esac
}

# Backup database configuration
backup_configuration() {
    info "Backing up database configuration..."
    
    local backup_dir="${SCRIPT_DIR}/config-backup-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "${backup_dir}"
    
    # Backup all configuration files
    cp -r "${SCRIPT_DIR}"/*.yaml "${backup_dir}/"
    cp -r "${SCRIPT_DIR}"/*.tf "${backup_dir}/"
    cp -r "${SCRIPT_DIR}"/*.sh "${backup_dir}/"
    
    # Create archive
    tar -czf "${backup_dir}.tar.gz" -C "${SCRIPT_DIR}" "$(basename "${backup_dir}")"
    rm -rf "${backup_dir}"
    
    success "Configuration backed up to ${backup_dir}.tar.gz"
}

# Validate deployment
validate_deployment() {
    local platform=$1
    
    info "Validating deployment for ${platform}..."
    
    case $platform in
        aws)
            # Check AWS resources
            aws rds describe-db-clusters --db-cluster-identifier textnlp-postgres-cluster &> /dev/null
            aws rds describe-db-proxy --db-proxy-name textnlp-postgres-proxy &> /dev/null
            ;;
        gcp)
            # Check GCP resources
            gcloud sql instances describe textnlp-postgres-main &> /dev/null
            ;;
        azure)
            # Check Azure resources
            az postgres flexible-server show --name textnlp-postgres-main --resource-group textnlp-prod-rg &> /dev/null
            ;;
        onprem)
            # Check on-premises services
            systemctl is-active patroni &> /dev/null
            systemctl is-active haproxy &> /dev/null
            systemctl is-active pgbouncer &> /dev/null
            ;;
    esac
    
    success "Deployment validation passed for ${platform}"
}

# Print deployment summary
print_summary() {
    info "=== Database Deployment Summary ==="
    info "Platform: ${DEPLOYMENT_PLATFORM}"
    info "Environment: ${ENVIRONMENT}"
    info "Timestamp: $(date)"
    info "Log file: ${LOG_FILE}"
    info ""
    
    case $DEPLOYMENT_PLATFORM in
        aws)
            info "AWS Resources:"
            terraform output -json | jq -r 'to_entries[] | "  \(.key): \(.value.value)"'
            ;;
        gcp)
            info "GCP Resources:"
            terraform output -json | jq -r 'to_entries[] | "  \(.key): \(.value.value)"'
            ;;
        azure)
            info "Azure Resources:"
            terraform output -json | jq -r 'to_entries[] | "  \(.key): \(.value.value)"'
            ;;
        onprem)
            info "On-Premises Services:"
            info "  Patroni Cluster: postgres-master-01, postgres-standby-01, postgres-standby-02"
            info "  HAProxy Load Balancer: 10.100.1.70:5432 (write), 10.100.1.70:5433 (read)"
            info "  PgBouncer Pools: pgbouncer-01, pgbouncer-02"
            ;;
        all)
            info "Multi-platform deployment completed"
            ;;
    esac
    
    info ""
    info "Next steps:"
    info "1. Configure application connection strings"
    info "2. Set up monitoring and alerting"
    info "3. Configure backup and recovery procedures"
    info "4. Run performance testing"
}

# Cleanup function
cleanup() {
    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        error "Deployment failed with exit code ${exit_code}"
        error "Check the log file for details: ${LOG_FILE}"
    fi
    
    # Clean up temporary files
    rm -f "${SCRIPT_DIR}"/*.tfplan
    rm -f "${SCRIPT_DIR}"/*.sql
    rm -f "${SCRIPT_DIR}"/haproxy.cfg
    rm -f "${SCRIPT_DIR}"/pgbouncer.ini
    rm -rf "${SCRIPT_DIR}"/patroni-config
    
    exit $exit_code
}

# Main deployment function
main() {
    info "Starting TextNLP database deployment..."
    info "Platform: ${DEPLOYMENT_PLATFORM}"
    info "Environment: ${ENVIRONMENT}"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Load configuration
    load_environment
    
    # Check prerequisites
    check_prerequisites
    
    # Backup current configuration
    backup_configuration
    
    # Deploy based on platform
    case $DEPLOYMENT_PLATFORM in
        aws)
            validate_terraform "aws"
            deploy_aws_database
            validate_deployment "aws"
            ;;
        gcp)
            validate_terraform "gcp"
            deploy_gcp_database
            validate_deployment "gcp"
            ;;
        azure)
            validate_terraform "azure"
            deploy_azure_database
            validate_deployment "azure"
            ;;
        onprem)
            deploy_onprem_database
            validate_deployment "onprem"
            ;;
        all)
            validate_terraform "aws"
            validate_terraform "gcp"
            validate_terraform "azure"
            
            deploy_aws_database
            deploy_gcp_database
            deploy_azure_database
            deploy_onprem_database
            
            validate_deployment "aws"
            validate_deployment "gcp"
            validate_deployment "azure"
            validate_deployment "onprem"
            ;;
        *)
            error "Unknown platform: ${DEPLOYMENT_PLATFORM}"
            error "Supported platforms: aws, gcp, azure, onprem, all"
            exit 1
            ;;
    esac
    
    # Print summary
    print_summary
    
    success "Database deployment completed successfully!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --platform)
            export DEPLOYMENT_PLATFORM="$2"
            shift 2
            ;;
        --environment)
            export ENVIRONMENT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --platform PLATFORM    Deployment platform (aws|gcp|azure|onprem|all)"
            echo "  --environment ENV       Environment (production|staging|development)"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main "$@"