#!/bin/bash

# Health Check Script

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

echo "🔍 Running health checks..."

services=(
    "orchestration-service:8080:Orchestration Service"
    "ingestion-service:8081:Ingestion Service"
    "metrics-extraction-service:8082:Metrics Extraction Service"
    "generation-service:8083:Generation Service"
    "validation-service:8084:Validation Service"
    "delivery-service:8085:Delivery Service"
)

infrastructure=(
    "redis:6379:Redis"
    "postgres:5432:PostgreSQL"
    "minio:9000:MinIO"
    "kafka:9092:Kafka"
    "prometheus:9090:Prometheus"
    "grafana:3000:Grafana"
)

failed_services=()
failed_infrastructure=()

# Check application services
echo ""
print_info "Checking application services..."
for service in "${services[@]}"; do
    IFS=':' read -r service_name port display_name <<< "$service"
    
    print_info "Checking $display_name..."
    
    if curl -f -s --max-time 10 "http://localhost:$port/health" > /dev/null 2>&1; then
        print_success "$display_name is healthy"
    else
        print_error "$display_name is unhealthy"
        failed_services+=("$display_name")
    fi
done

# Check infrastructure services
echo ""
print_info "Checking infrastructure services..."
for service in "${infrastructure[@]}"; do
    IFS=':' read -r service_name port display_name <<< "$service"
    
    print_info "Checking $display_name..."
    
    case $service_name in
        "redis")
            if docker-compose exec -T redis redis-cli ping | grep -q "PONG"; then
                print_success "$display_name is healthy"
            else
                print_error "$display_name is unhealthy"
                failed_infrastructure+=("$display_name")
            fi
            ;;
        "postgres")
            if docker-compose exec -T postgres pg_isready -U postgres | grep -q "accepting connections"; then
                print_success "$display_name is healthy"
            else
                print_error "$display_name is unhealthy"
                failed_infrastructure+=("$display_name")
            fi
            ;;
        *)
            if curl -f -s --max-time 10 "http://localhost:$port" > /dev/null 2>&1; then
                print_success "$display_name is healthy"
            else
                print_error "$display_name is unhealthy"
                failed_infrastructure+=("$display_name")
            fi
            ;;
    esac
done

# Check Docker containers
echo ""
print_info "Checking Docker container status..."
docker-compose ps

# Summary
echo ""
if [ ${#failed_services[@]} -eq 0 ] && [ ${#failed_infrastructure[@]} -eq 0 ]; then
    print_success "All services are healthy! 🎉"
    echo ""
    print_info "You can now:"
    echo "  • Visit http://localhost:8080 for API documentation"
    echo "  • Check Grafana at http://localhost:3000 (admin/admin)"
    echo "  • Access MinIO at http://localhost:9001 (minioadmin/minioadmin)"
    echo "  • Run example workflows in the examples/ directory"
    exit 0
else
    echo "❌ Health check failed!"
    if [ ${#failed_services[@]} -gt 0 ]; then
        print_error "Failed application services: ${failed_services[*]}"
    fi
    if [ ${#failed_infrastructure[@]} -gt 0 ]; then
        print_error "Failed infrastructure services: ${failed_infrastructure[*]}"
    fi
    echo ""
    print_info "Troubleshooting:"
    echo "  • Check logs with: make logs"
    echo "  • Restart services with: make restart"
    echo "  • View container status with: docker-compose ps"
    exit 1
fi
