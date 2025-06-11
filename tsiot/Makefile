.PHONY: help build build-server build-cli build-worker test test-unit test-integration lint fmt vet clean run-server run-worker run-cli install deps docker-build docker-run docker-compose-up docker-compose-down

# Default target
help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Build variables
BINARY_NAME_SERVER=tsiot-server
BINARY_NAME_CLI=tsiot-cli
BINARY_NAME_WORKER=tsiot-worker
BUILD_DIR=bin
VERSION?=$(shell git describe --tags --always --dirty)
COMMIT=$(shell git rev-parse HEAD)
DATE=$(shell date -u '+%Y-%m-%d_%H:%M:%S')
LDFLAGS=-ldflags "-X main.Version=$(VERSION) -X main.GitCommit=$(COMMIT) -X main.BuildDate=$(DATE)"

# Go variables
GOOS?=$(shell go env GOOS)
GOARCH?=$(shell go env GOARCH)
CGO_ENABLED?=0

# Build targets
build: build-server build-cli build-worker ## Build all binaries

build-server: ## Build server binary
	@echo "Building server..."
	@mkdir -p $(BUILD_DIR)
	CGO_ENABLED=$(CGO_ENABLED) GOOS=$(GOOS) GOARCH=$(GOARCH) go build $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME_SERVER) ./cmd/server

build-cli: ## Build CLI binary
	@echo "Building CLI..."
	@mkdir -p $(BUILD_DIR)
	CGO_ENABLED=$(CGO_ENABLED) GOOS=$(GOOS) GOARCH=$(GOARCH) go build $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME_CLI) ./cmd/cli

build-worker: ## Build worker binary
	@echo "Building worker..."
	@mkdir -p $(BUILD_DIR)
	CGO_ENABLED=$(CGO_ENABLED) GOOS=$(GOOS) GOARCH=$(GOARCH) go build $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME_WORKER) ./cmd/worker

# Development targets
deps: ## Download dependencies
	@echo "Downloading dependencies..."
	go mod download
	go mod tidy

install: deps ## Install dependencies and tools
	@echo "Installing tools..."
	go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
	go install github.com/securecodewarrior/sast-scan@latest

# Testing targets
test: test-unit ## Run all tests

test-unit: ## Run unit tests
	@echo "Running unit tests..."
	go test -v -race -coverprofile=coverage.out ./...

test-integration: ## Run integration tests
	@echo "Running integration tests..."
	go test -v -tags=integration ./tests/integration/...

test-e2e: ## Run end-to-end tests
	@echo "Running E2E tests..."
	go test -v -tags=e2e ./tests/e2e/...

benchmark: ## Run benchmarks
	@echo "Running benchmarks..."
	go test -v -bench=. -benchmem ./tests/benchmarks/...

# Code quality targets
lint: ## Run linter
	@echo "Running linter..."
	golangci-lint run

fmt: ## Format code
	@echo "Formatting code..."
	go fmt ./...

vet: ## Run go vet
	@echo "Running go vet..."
	go vet ./...

# Runtime targets
run-server: build-server ## Run server
	@echo "Starting server..."
	./$(BUILD_DIR)/$(BINARY_NAME_SERVER) --log-level debug

run-worker: build-worker ## Run worker
	@echo "Starting worker..."
	./$(BUILD_DIR)/$(BINARY_NAME_WORKER) --log-level debug

run-cli: build-cli ## Run CLI help
	@echo "Running CLI..."
	./$(BUILD_DIR)/$(BINARY_NAME_CLI) --help

# Docker targets
docker-build: ## Build Docker images
	@echo "Building Docker images..."
	docker build -f deployments/docker/Dockerfile.server -t tsiot-server:$(VERSION) .
	docker build -f deployments/docker/Dockerfile.worker -t tsiot-worker:$(VERSION) .
	docker build -f deployments/docker/Dockerfile.cli -t tsiot-cli:$(VERSION) .

docker-run-server: ## Run server in Docker
	docker run -p 8080:8080 -p 9090:9090 tsiot-server:$(VERSION)

docker-compose-up: ## Start services with docker-compose
	docker-compose -f deployments/docker/docker-compose.yml up -d

docker-compose-down: ## Stop services with docker-compose
	docker-compose -f deployments/docker/docker-compose.yml down

# Release targets
release-linux: ## Build for Linux
	@echo "Building for Linux..."
	GOOS=linux GOARCH=amd64 $(MAKE) build
	@mkdir -p releases/linux-amd64
	@cp $(BUILD_DIR)/* releases/linux-amd64/

release-darwin: ## Build for macOS
	@echo "Building for macOS..."
	GOOS=darwin GOARCH=amd64 $(MAKE) build
	@mkdir -p releases/darwin-amd64
	@cp $(BUILD_DIR)/* releases/darwin-amd64/

release-windows: ## Build for Windows
	@echo "Building for Windows..."
	GOOS=windows GOARCH=amd64 $(MAKE) build
	@mkdir -p releases/windows-amd64
	@cp $(BUILD_DIR)/* releases/windows-amd64/

release-all: release-linux release-darwin release-windows ## Build for all platforms

# Deployment targets
deploy-k8s: ## Deploy to Kubernetes
	@echo "Deploying to Kubernetes..."
	kubectl apply -f deployments/kubernetes/

deploy-helm: ## Deploy with Helm
	@echo "Deploying with Helm..."
	helm upgrade --install tsiot deployments/helm/ --namespace tsiot --create-namespace

# Database targets
migrate-up: ## Run database migrations up
	@echo "Running migrations up..."
	./$(BUILD_DIR)/$(BINARY_NAME_CLI) migrate up

migrate-down: ## Run database migrations down
	@echo "Running migrations down..."
	./$(BUILD_DIR)/$(BINARY_NAME_CLI) migrate down

# Development helpers
dev-setup: install ## Setup development environment
	@echo "Setting up development environment..."
	@if [ ! -f .env ]; then cp .env.example .env; fi
	@echo "Development setup complete!"

dev-server: ## Run server in development mode with hot reload
	@echo "Starting development server..."
	go run ./cmd/server --log-level debug --config configs/environments/development.yaml

dev-worker: ## Run worker in development mode
	@echo "Starting development worker..."
	go run ./cmd/worker --log-level debug --server-url http://localhost:8080

# Cleanup targets
clean: ## Clean build artifacts
	@echo "Cleaning..."
	rm -rf $(BUILD_DIR)
	rm -rf releases
	rm -f coverage.out
	docker system prune -f

clean-all: clean ## Clean everything including dependencies
	go clean -modcache

# Generate targets
generate: ## Run go generate
	@echo "Running go generate..."
	go generate ./...

proto: ## Generate protobuf code
	@echo "Generating protobuf code..."
	protoc --go_out=. --go-grpc_out=. internal/protocols/grpc/protos/*.proto

# Documentation targets
docs: ## Generate documentation
	@echo "Generating documentation..."
	go doc -all ./... > docs/API.md

docs-serve: ## Serve documentation locally
	@echo "Serving documentation..."
	godoc -http=:6060

# Monitoring targets
metrics: ## Show build metrics
	@echo "Build metrics:"
	@echo "Version: $(VERSION)"
	@echo "Commit: $(COMMIT)"
	@echo "Date: $(DATE)"
	@echo "OS/Arch: $(GOOS)/$(GOARCH)"

# Security targets
security-scan: ## Run security scan
	@echo "Running security scan..."
	gosec ./...

# Default when no target is specified
.DEFAULT_GOAL := help