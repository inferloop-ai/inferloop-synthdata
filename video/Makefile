.PHONY: help setup build test deploy clean start stop restart logs status

# Default target
help:
	@echo "🎬 Inferloop SynthData Video Pipeline"
	@echo ""
	@echo "Available targets:"
	@echo "  setup     - Setup development environment"
	@echo "  build     - Build all services"
	@echo "  test      - Run all tests"
	@echo "  deploy    - Deploy to local environment"
	@echo "  start     - Start all services"
	@echo "  stop      - Stop all services"
	@echo "  restart   - Restart all services"
	@echo "  logs      - Show logs for all services"
	@echo "  status    - Show status of all services"
	@echo "  clean     - Clean up resources"

setup:
	@echo "🔧 Setting up development environment..."
	./scripts/setup/dev-environment.sh

build:
	@echo "🏗️ Building all services..."
	docker-compose build

test:
	@echo "🧪 Running tests..."
	./scripts/setup/run-tests.sh

deploy: build
	@echo "🚀 Deploying local stack..."
	./scripts/deployment/local-deploy.sh

start:
	@echo "▶️ Starting all services..."
	docker-compose up -d

stop:
	@echo "⏹️ Stopping all services..."
	docker-compose down

restart: stop start

logs:
	@echo "📋 Showing logs..."
	docker-compose logs -f

status:
	@echo "📊 Service status..."
	docker-compose ps
	@echo ""
	@echo "🔍 Running health checks..."
	./scripts/deployment/health-check.sh

clean:
	@echo "🧹 Cleaning up..."
	docker-compose down -v
	docker system prune -f
	@echo "✅ Cleanup complete"

# Development helpers
dev-start:
	@echo "🔧 Starting development environment..."
	docker-compose up -d redis postgres minio kafka zookeeper
	@echo "✅ Infrastructure services started"

dev-stop:
	@echo "🔧 Stopping development environment..."
	docker-compose down

# Production helpers
prod-deploy:
	@echo "🚀 Deploying to production..."
	kubectl apply -f infrastructure/kubernetes/manifests/

prod-status:
	@echo "📊 Production status..."
	kubectl get pods -n video-pipeline