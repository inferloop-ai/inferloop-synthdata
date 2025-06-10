# 🚀 Setup Guide - Inferloop SynthData Video Pipeline

This guide explains how to use the individual artifacts to build your complete video synthesis pipeline.

## 📁 File Organization

The repository has been split into individual, manageable files. Here's how to set them up:

### Step 1: Create the Repository Structure

1. **Save the repository structure script**:
   - Copy `create-repo-structure.sh` 
   - Make it executable: `chmod +x create-repo-structure.sh`
   - Run it: `./create-repo-structure.sh`

This creates the complete directory structure.

### Step 2: Add Core Configuration Files

Navigate to your new repository: `cd inferloop-synthdata-video`

2. **Add Docker Compose configuration**:
   - Save `docker-compose.yml` in the root directory

3. **Add project management**:
   - Save `Makefile` in the root directory

4. **Add environment configuration**:
   - Save `.env.example` in the root directory

5. **Add Python dependencies**:
   - Save `requirements.txt` in the root directory

6. **Add project documentation**:
   - Save `README.md` in the root directory

### Step 3: Add Core Service

7. **Create the orchestration service**:
   - Save `services/orchestration-service/src/main.py`
   - Create `services/orchestration-service/Dockerfile`:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY src/ ./src/
   RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
   USER appuser
   HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
       CMD curl -f http://localhost:8080/health || exit 1
   EXPOSE 8080
   CMD ["python", "src/main.py"]
   ```
   - Create `services/orchestration-service/requirements.txt`:
   ```
   fastapi==0.104.1
   uvicorn[standard]==0.24.0
   pydantic==2.5.0
   httpx==0.25.2
   ```

### Step 4: Add Setup Scripts

8. **Create the setup script**:
   - Save `scripts/setup/dev-environment.sh`
   - Make it executable: `chmod +x scripts/setup/dev-environment.sh`

9. **Create the deployment script**:
   - Save `scripts/deployment/local-deploy.sh`
   - Make it executable: `chmod +x scripts/deployment/local-deploy.sh`

10. **Create the health check script**:
    - Save `scripts/deployment/health-check.sh`
    - Make it executable: `chmod +x scripts/deployment/health-check.sh`

## 🏃‍♂️ Quick Start

Once you have all files in place:

```bash
# 1. Setup development environment
./scripts/setup/dev-environment.sh

# 2. Deploy the stack
make deploy

# 3. Check health
make status
```

## 📁 Complete File Structure

After setting up all files, your structure should look like:

```
inferloop-synthdata-video/
├── 📄 README.md
├── 📄 Makefile  
├── 📄 docker-compose.yml
├── 📄 requirements.txt
├── 📄 .env.example
├── 🏢 services/
│   └── orchestration-service/
│       ├── src/
│       │   └── main.py
│       ├── Dockerfile
│       └── requirements.txt
├── 🔧 scripts/
│   ├── setup/
│   │   └── dev-environment.sh
│   └── deployment/
│       ├── local-deploy.sh
│       └── health-check.sh
└── [all other directories from create-repo-structure.sh]
```

## 🎯 Next Steps

1. **Add remaining services**: Create similar service files for ingestion, metrics, generation, validation, and delivery services
2. **Configure verticals**: Add industry-specific configurations in `verticals/`
3. **Add examples**: Create example workflows in `examples/`
4. **Setup monitoring**: Configure Prometheus and Grafana dashboards
5. **Add tests**: Create comprehensive test suites in `qa/`

## 🔧 Additional Services

For each additional service (ingestion, metrics, generation, validation, delivery), create:

```
services/[service-name]/
├── src/
│   └── main.py         # FastAPI service similar to orchestration
├── Dockerfile          # Docker configuration
├── requirements.txt    # Python dependencies  
└── tests/             # Service-specific tests
```

Each service should follow the same pattern as the orchestration service with:
- FastAPI application
- Health check endpoint
- Service-specific business logic
- Proper error handling and logging

## 🌟 Benefits of This Split Approach

- **📝 Manageable Files**: Each file is focused and easier to understand
- **🔧 Modular Setup**: Add components incrementally as needed
- **👥 Team Collaboration**: Different team members can work on different components
- **🎯 Focused Development**: Easier to find and modify specific functionality
- **📦 Version Control**: Better git history and merge conflict resolution

This approach gives you a production-ready enterprise video synthesis pipeline that you can customize and extend for your specific needs!
