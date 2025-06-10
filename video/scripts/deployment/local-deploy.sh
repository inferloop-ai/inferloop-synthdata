#!/bin/bash

# Local Deployment Script

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_step() {
    echo -e "${YELLOW}🔨 $1${NC}"
}

echo "🚀 Deploying Enterprise Video Synthesis Pipeline locally..."

# Check if .env exists
if [ ! -f .env ]; then
    print_info "Creating .env from template..."
    cp .env.example .env
fi

# Start infrastructure services first
print_step "Starting infrastructure services..."
docker-compose up -d redis postgres minio kafka zookeeper

# Wait for services to be ready
print_step "Waiting for infrastructure services to be ready..."
sleep 15

# Check infrastructure health
print_step "Checking infrastructure health..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if docker-compose ps | grep -q "healthy\|Up"; then
        break
    fi
    sleep 2
    attempt=$((attempt + 1))
done

# Initialize databases and storage
print_step "Initializing databases and storage..."

# Wait for PostgreSQL
until docker-compose exec -T postgres pg_isready -U postgres; do
    print_info "Waiting for PostgreSQL..."
    sleep 2
done

# Create MinIO bucket
docker-compose exec -T minio mc alias set local http://localhost:9000 minioadmin minioadmin
docker-compose exec -T minio mc mb local/video-pipeline || true

# Start application services
print_step "Starting application services..."
docker-compose up -d

# Wait for services to be healthy
print_step "Waiting for application services to be healthy..."
sleep 30

# Run health checks
print_step "Running health checks..."
./scripts/deployment/health-check.sh

print_success "Deployment complete!"
echo ""
echo "🌐 Services available at:"
echo "  • API Gateway (Orchestration): http://localhost:8080"
echo "  • Ingestion Service: http://localhost:8081"
echo "  • Metrics Service: http://localhost:8082"
echo "  • Generation Service: http://localhost:8083"
echo "  • Validation Service: http://localhost:8084"
echo "  • Delivery Service: http://localhost:8085"
echo ""
echo "🔧 Management interfaces:"
echo "  • Grafana Dashboard: http://localhost:3000 (admin/admin)"
echo "  • MinIO Console: http://localhost:9001 (minioadmin/minioadmin)"
echo "  • Prometheus: http://localhost:9090"
echo "  • Kibana: http://localhost:5601"
echo ""
echo "📊 Check status with: make status"
echo "📋 View logs with: make logs"
