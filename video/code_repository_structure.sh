#!/bin/bash

# Inferloop Synthetic Data (Video) - Repository Structure Creator
# Creates the complete directory structure and basic files

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_step() { echo -e "${YELLOW}🔨 $1${NC}"; }

REPO_NAME="inferloop-synthdata/video"
PROJECT_ROOT="$(pwd)/$REPO_NAME"

echo "🚀 Creating Inferloop Synthetic Data (Video) Repository Structure..."

# Check if directory exists
if [ -d "$PROJECT_ROOT" ]; then
    read -p "Directory exists. Remove and continue? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$PROJECT_ROOT"
    else
        echo "Aborting to avoid overwriting."
        exit 1
    fi
fi

mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

print_step "Creating directory structure..."

# Core Services
mkdir -p services/{orchestration-service,ingestion-service,metrics-extraction-service,generation-service,validation-service,delivery-service}/{src,tests,config,docs}

# Pipeline Components
mkdir -p pipeline/scrapers/{web-scrapers,api-connectors,file-processors}
mkdir -p pipeline/processors/{video-analysis,metrics-calculation,quality-assessment}
mkdir -p pipeline/generators/{unreal-engine,unity,omniverse,custom-models}
mkdir -p pipeline/validators/{quality-metrics,compliance-checks,performance-tests}
mkdir -p pipeline/distributors/{streaming-apis,batch-delivery,real-time-feeds}

# Verticals
mkdir -p verticals/{autonomous-vehicles,robotics,smart-cities,gaming,healthcare,manufacturing,retail}/{scenarios,validators,metrics,config}

# Integrations
mkdir -p integrations/{mcp-protocol,rest-apis,graphql-apis,grpc-services,webhooks,kafka-streams,websocket-feeds}

# SDKs
mkdir -p sdks/{python-sdk,javascript-sdk,go-sdk,rust-sdk,cli-tools}/{src,tests,examples,docs}

# Infrastructure
mkdir -p infrastructure/{terraform,kubernetes,docker,monitoring,logging,security}/{modules,manifests,configs}

# Configuration
mkdir -p config/{environments,secrets,feature-flags,quality-thresholds,vertical-specific}

# Data Management
mkdir -p data/{schemas,samples,migrations,seeds,raw,processed,generated}

# QA
mkdir -p qa/{test-suites,performance-tests,quality-gates,benchmarks}

# Documentation
mkdir -p docs/{architecture,user-guides,developer-guides,compliance}

# Examples
mkdir -p examples/{use-cases,integrations,benchmarks}

# Scripts
mkdir -p scripts/{setup,deployment,data-management,monitoring,backup-restore}

# CI/CD
mkdir -p .github/workflows .gitlab-ci jenkins/pipelines

# Storage
mkdir -p storage/{object-store-configs,database-schemas,cache-configurations}

# Runtime directories
mkdir -p {logs,tmp}

print_success "Directory structure created!"
print_info "Next: Run individual setup scripts to populate with files"
print_info "Repository location: $PROJECT_ROOT"