# Deployment Operations Runbook

This runbook covers all deployment operations for the Inferloop Synthetic Data SDK platform across different environments and cloud providers.

## Table of Contents
1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Production Deployment](#production-deployment)
3. [Staging Deployment](#staging-deployment)
4. [Rollback Procedures](#rollback-procedures)
5. [Multi-Cloud Deployment](#multi-cloud-deployment)
6. [On-Premises Deployment](#on-premises-deployment)
7. [Emergency Procedures](#emergency-procedures)
8. [Post-Deployment Verification](#post-deployment-verification)

## Pre-Deployment Checklist

### Code Quality Verification
- [ ] All tests passing (unit, integration, e2e)
- [ ] Code review completed and approved
- [ ] Security scan completed (no critical vulnerabilities)
- [ ] Performance benchmarks within acceptable limits
- [ ] Documentation updated

### Infrastructure Readiness
- [ ] Infrastructure capacity verified
- [ ] Database migrations tested
- [ ] Configuration files validated
- [ ] SSL certificates valid and up-to-date
- [ ] DNS records configured correctly

### Team Coordination
- [ ] Deployment scheduled and communicated
- [ ] Operations team notified
- [ ] Rollback plan prepared
- [ ] Emergency contacts available
- [ ] Status page updated (maintenance mode if needed)

## Production Deployment

### Standard Deployment Process

#### 1. Preparation Phase
```bash
# Switch to deployment user
sudo su - deployment

# Navigate to deployment directory
cd /opt/inferloop-synthetic

# Verify current version
./scripts/get-version.sh

# Download new release
wget https://releases.inferloop.com/v${VERSION}/inferloop-synthetic-${VERSION}.tar.gz

# Verify checksums
sha256sum inferloop-synthetic-${VERSION}.tar.gz
```

#### 2. Backup Phase
```bash
# Backup current deployment
./scripts/backup-deployment.sh

# Backup database
./scripts/backup-database.sh --environment production

# Verify backups
./scripts/verify-backup.sh --latest
```

#### 3. Deployment Phase
```bash
# Enable maintenance mode
./scripts/maintenance-mode.sh enable

# Extract new version
tar -xzf inferloop-synthetic-${VERSION}.tar.gz

# Update symbolic links
ln -sfn inferloop-synthetic-${VERSION} current

# Install dependencies
cd current && pip install -r requirements.txt

# Run database migrations
./scripts/migrate-database.sh --environment production

# Update configuration
./scripts/update-config.sh --environment production

# Compile static assets
./scripts/build-assets.sh
```

#### 4. Service Restart
```bash
# Restart services in order
sudo systemctl restart inferloop-synthetic-worker
sudo systemctl restart inferloop-synthetic-api
sudo systemctl restart nginx

# Wait for services to be ready
./scripts/wait-for-services.sh

# Disable maintenance mode
./scripts/maintenance-mode.sh disable
```

#### 5. Verification Phase
```bash
# Run health checks
./scripts/health-check.sh --environment production

# Run smoke tests
./scripts/smoke-tests.sh --environment production

# Monitor logs for errors
./scripts/monitor-logs.sh --duration 300
```

### Blue-Green Deployment

#### Setup Blue Environment
```bash
# Deploy to blue environment
kubectl apply -f k8s/blue-deployment.yaml

# Wait for deployment to be ready
kubectl wait --for=condition=available deployment/synthdata-blue

# Run health checks
./scripts/health-check.sh --environment blue
```

#### Switch Traffic
```bash
# Update service to point to blue
kubectl patch service synthdata-service -p '{"spec":{"selector":{"version":"blue"}}}'

# Monitor metrics
./scripts/monitor-switch.sh --duration 600

# Verify switch successful
./scripts/verify-traffic-switch.sh
```

#### Cleanup Green Environment
```bash
# Scale down green deployment
kubectl scale deployment synthdata-green --replicas=0

# Keep for quick rollback (delete after 24h)
# kubectl delete deployment synthdata-green
```

### Canary Deployment

#### Deploy Canary Version
```bash
# Deploy canary with 10% traffic
kubectl apply -f k8s/canary-deployment.yaml

# Configure traffic splitting
istioctl kube-inject -f k8s/virtual-service-canary.yaml | kubectl apply -f -
```

#### Monitor Canary Metrics
```bash
# Monitor error rates
./scripts/monitor-canary.sh --metrics error_rate --threshold 0.1

# Monitor response times
./scripts/monitor-canary.sh --metrics response_time --threshold 500

# Check business metrics
./scripts/monitor-canary.sh --metrics conversion_rate --threshold 0.95
```

#### Promote or Rollback Canary
```bash
# If metrics good, promote canary
./scripts/promote-canary.sh

# If metrics bad, rollback canary
./scripts/rollback-canary.sh
```

## Staging Deployment

### Automated Staging Deployment
```bash
# Triggered by CI/CD pipeline
./scripts/deploy-staging.sh --version ${BUILD_NUMBER}

# Run integration tests
./scripts/run-integration-tests.sh --environment staging

# Generate deployment report
./scripts/generate-deployment-report.sh --environment staging
```

### Manual Staging Deployment
```bash
# Deploy specific branch/commit
./scripts/deploy-staging.sh --branch feature/new-feature --commit abc123

# Verify deployment
./scripts/verify-staging-deployment.sh

# Run smoke tests
./scripts/smoke-tests.sh --environment staging
```

## Rollback Procedures

### Immediate Rollback (Emergency)
```bash
# Enable maintenance mode
./scripts/maintenance-mode.sh enable

# Rollback to previous version
./scripts/rollback.sh --immediate --version previous

# Verify rollback
./scripts/verify-rollback.sh

# Disable maintenance mode
./scripts/maintenance-mode.sh disable
```

### Graceful Rollback
```bash
# Check available versions
./scripts/list-versions.sh

# Rollback to specific version
./scripts/rollback.sh --version v1.2.3 --graceful

# Run verification tests
./scripts/verify-rollback.sh --comprehensive
```

### Database Rollback
```bash
# Rollback database migrations
./scripts/rollback-database.sh --version v1.2.3

# Verify data integrity
./scripts/verify-database.sh --comprehensive

# Restore from backup if needed
./scripts/restore-database.sh --backup latest-good
```

## Multi-Cloud Deployment

### AWS Deployment
```bash
# Set AWS credentials
export AWS_PROFILE=inferloop-production

# Deploy using CDK
cd infrastructure/aws
cdk deploy InferloopSyntheticStack --profile inferloop-production

# Verify deployment
aws ecs describe-services --cluster inferloop-production --services synthdata-api

# Update Route53 records
aws route53 change-resource-record-sets --hosted-zone-id Z123456789 --change-batch file://dns-changes.json
```

### Azure Deployment
```bash
# Set Azure credentials
az login --service-principal -u $AZURE_CLIENT_ID -p $AZURE_CLIENT_SECRET --tenant $AZURE_TENANT_ID

# Deploy using ARM templates
az deployment group create \
  --resource-group inferloop-production \
  --template-file infrastructure/azure/main.json \
  --parameters @infrastructure/azure/parameters.json

# Verify deployment
az container show --resource-group inferloop-production --name synthdata-api
```

### GCP Deployment
```bash
# Set GCP credentials
gcloud auth activate-service-account --key-file=service-account.json
gcloud config set project inferloop-production

# Deploy using Cloud Run
gcloud run deploy synthdata-api \
  --image gcr.io/inferloop-production/synthdata:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

# Verify deployment
gcloud run services describe synthdata-api --region us-central1
```

## On-Premises Deployment

### Kubernetes Deployment
```bash
# Set kubectl context
kubectl config use-context production-cluster

# Apply namespace and RBAC
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/rbac.yaml

# Deploy secrets and configmaps
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/configmaps.yaml

# Deploy applications
kubectl apply -f k8s/deployments.yaml
kubectl apply -f k8s/services.yaml
kubectl apply -f k8s/ingress.yaml

# Verify deployment
kubectl get pods -n synthdata
kubectl get services -n synthdata
```

### OpenShift Deployment
```bash
# Login to OpenShift
oc login https://openshift.company.com:8443

# Create project
oc new-project synthdata-production

# Deploy using templates
oc process -f openshift/template.yaml \
  -p APPLICATION_NAME=synthdata \
  -p IMAGE_TAG=latest | oc apply -f -

# Expose service
oc expose service synthdata-api

# Verify deployment
oc get pods
oc get routes
```

### Docker Swarm Deployment
```bash
# Initialize swarm (if not already done)
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.prod.yml synthdata

# Verify deployment
docker service ls
docker stack ps synthdata
```

## Emergency Procedures

### Service Unavailable
```bash
# Check service status
systemctl status inferloop-synthetic-api

# Check container status
docker ps -a | grep synthdata

# Check logs
journalctl -u inferloop-synthetic-api -f

# Restart services
sudo systemctl restart inferloop-synthetic-api
```

### High Error Rate
```bash
# Enable maintenance mode immediately
./scripts/maintenance-mode.sh enable

# Check error logs
./scripts/check-error-logs.sh --last 1h

# Rollback if necessary
./scripts/emergency-rollback.sh

# Disable maintenance mode after fix
./scripts/maintenance-mode.sh disable
```

### Database Connection Issues
```bash
# Check database status
./scripts/check-database.sh --connection

# Check connection pool
./scripts/check-connection-pool.sh

# Restart database connections
./scripts/restart-db-connections.sh
```

### SSL Certificate Expiry
```bash
# Check certificate status
./scripts/check-ssl-certificates.sh

# Renew certificates
./scripts/renew-certificates.sh --environment production

# Update load balancer
./scripts/update-load-balancer-certs.sh
```

## Post-Deployment Verification

### Automated Verification
```bash
# Run full verification suite
./scripts/post-deployment-verification.sh --environment production

# Check API endpoints
./scripts/verify-api-endpoints.sh

# Verify database connections
./scripts/verify-database-connections.sh

# Check external integrations
./scripts/verify-external-integrations.sh
```

### Manual Verification
```bash
# Check system metrics
./scripts/check-metrics.sh --dashboard production

# Verify user flows
./scripts/verify-user-flows.sh --critical-paths

# Check performance benchmarks
./scripts/performance-check.sh --baseline

# Review security status
./scripts/security-check.sh --comprehensive
```

### Monitoring Setup
```bash
# Set up alerts for new deployment
./scripts/setup-deployment-alerts.sh --version ${VERSION}

# Monitor for 24 hours
./scripts/monitor-deployment.sh --duration 24h

# Generate deployment report
./scripts/generate-deployment-report.sh --environment production --version ${VERSION}
```

## Deployment Metrics

### Key Metrics to Monitor
- **Deployment Success Rate**: Target 99.5%
- **Deployment Duration**: Target < 30 minutes
- **Rollback Rate**: Target < 5%
- **Mean Time to Recovery**: Target < 2 hours
- **Change Failure Rate**: Target < 1%

### Monitoring Commands
```bash
# Check deployment metrics
./scripts/deployment-metrics.sh --period 30d

# Generate deployment report
./scripts/deployment-report.sh --format json --output metrics.json

# Update deployment dashboard
./scripts/update-dashboard.sh --metrics deployment
```

## Troubleshooting

### Common Issues

#### Deployment Stuck
```bash
# Check deployment status
kubectl describe deployment synthdata-api

# Check pod events
kubectl describe pod <pod-name>

# Force restart deployment
kubectl rollout restart deployment synthdata-api
```

#### Configuration Issues
```bash
# Validate configuration
./scripts/validate-config.sh --environment production

# Compare configurations
./scripts/diff-config.sh --source staging --target production

# Update configuration
./scripts/update-config.sh --key database.host --value new-host
```

#### Resource Constraints
```bash
# Check resource usage
kubectl top nodes
kubectl top pods

# Scale deployment
kubectl scale deployment synthdata-api --replicas=5

# Check autoscaling
kubectl describe hpa synthdata-api-hpa
```

## Contact Information

### Deployment Team
- **Deployment Lead**: deployment-lead@inferloop.com
- **DevOps Engineers**: devops@inferloop.com
- **Platform Team**: platform@inferloop.com

### Emergency Contacts
- **On-Call Engineer**: +1-XXX-XXX-XXXX
- **Deployment Manager**: +1-XXX-XXX-XXXX
- **CTO**: +1-XXX-XXX-XXXX

### Escalation Process
1. **Level 1**: DevOps Engineer
2. **Level 2**: Senior DevOps Engineer
3. **Level 3**: Deployment Lead
4. **Level 4**: Engineering Manager