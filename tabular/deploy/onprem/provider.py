"""On-premises deployment provider for Kubernetes clusters."""

import subprocess
import json
import yaml
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import tempfile
import shutil

from ..base import BaseDeploymentProvider, DeploymentResult, ResourceConfig
from ..exceptions import DeploymentError, ValidationError
from .helm import HelmChartGenerator, HelmDeployer


class OnPremKubernetesProvider(BaseDeploymentProvider):
    """Provider for on-premises Kubernetes deployments."""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        """Initialize the on-premises Kubernetes provider.
        
        Args:
            kubeconfig_path: Path to kubeconfig file. If None, uses default ~/.kube/config
        """
        super().__init__()
        self.kubeconfig_path = kubeconfig_path or str(Path.home() / ".kube" / "config")
        self.kubectl_base_cmd = ["kubectl", f"--kubeconfig={self.kubeconfig_path}"]
        self._validate_cluster_access()
        self.helm_deployer = HelmDeployer(kubeconfig_path=self.kubeconfig_path)
        
    def _validate_cluster_access(self):
        """Validate that we can access the Kubernetes cluster."""
        try:
            result = subprocess.run(
                self.kubectl_base_cmd + ["cluster-info"],
                capture_output=True,
                text=True,
                check=True
            )
            if "is running" not in result.stdout:
                raise ValidationError("Cannot connect to Kubernetes cluster")
        except subprocess.CalledProcessError as e:
            raise ValidationError(f"Failed to access Kubernetes cluster: {e.stderr}")
    
    def authenticate(self, **kwargs) -> bool:
        """Authenticate with the Kubernetes cluster.
        
        Args:
            **kwargs: Optional authentication parameters
                - context: Kubernetes context to use
                - namespace: Default namespace
        
        Returns:
            bool: True if authentication successful
        """
        try:
            # Set context if provided
            if "context" in kwargs:
                subprocess.run(
                    self.kubectl_base_cmd + ["config", "use-context", kwargs["context"]],
                    check=True,
                    capture_output=True
                )
            
            # Set default namespace if provided
            if "namespace" in kwargs:
                self.default_namespace = kwargs["namespace"]
            else:
                self.default_namespace = "synthdata"
            
            # Create namespace if it doesn't exist
            self._ensure_namespace(self.default_namespace)
            
            self._authenticated = True
            return True
        except subprocess.CalledProcessError as e:
            self._authenticated = False
            return False
    
    def _ensure_namespace(self, namespace: str):
        """Ensure a namespace exists, create if it doesn't."""
        try:
            subprocess.run(
                self.kubectl_base_cmd + ["get", "namespace", namespace],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError:
            # Namespace doesn't exist, create it
            namespace_yaml = {
                "apiVersion": "v1",
                "kind": "Namespace",
                "metadata": {
                    "name": namespace,
                    "labels": {
                        "app": "inferloop-synthdata",
                        "managed-by": "synthdata-cli"
                    }
                }
            }
            self._apply_manifest(namespace_yaml)
    
    def deploy_container(self, config: ResourceConfig) -> DeploymentResult:
        """Deploy a containerized synthetic data service.
        
        Args:
            config: Container deployment configuration
        
        Returns:
            DeploymentResult: Deployment outcome
        """
        if not self._authenticated:
            result = DeploymentResult()
            result.success = False
            result.message = "Provider not authenticated"
            return result
        
        try:
            # Generate deployment manifest
            deployment = self._create_deployment_manifest(config)
            
            # Generate service manifest
            service = self._create_service_manifest(config)
            
            # Apply manifests
            self._apply_manifest(deployment)
            self._apply_manifest(service)
            
            # Wait for deployment to be ready
            deployment_ready = self._wait_for_deployment(
                config.metadata.get("name", "synthdata-app"),
                config.metadata.get("namespace", self.default_namespace)
            )
            
            if not deployment_ready:
                raise DeploymentError("Deployment failed to become ready")
            
            # Get service endpoint
            endpoint = self._get_service_endpoint(
                config.metadata.get("name", "synthdata-app"),
                config.metadata.get("namespace", self.default_namespace)
            )
            
            result = DeploymentResult()
            result.success = True
            result.resource_id = f"{config.metadata.get('namespace', self.default_namespace)}/{config.metadata.get('name', 'synthdata-app')}"
            result.endpoint = endpoint
            result.message = "Container deployed successfully"
            result.metadata = {
                "deployment_name": config.metadata.get("name", "synthdata-app"),
                "namespace": config.metadata.get("namespace", self.default_namespace),
                "replicas": config.compute.get("count", 3)
            }
            
            return result
            
        except Exception as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"Deployment failed: {str(e)}"
            return result
    
    def _create_deployment_manifest(self, config: ResourceConfig) -> Dict[str, Any]:
        """Create Kubernetes deployment manifest from config."""
        name = config.metadata.get("name", "synthdata-app")
        namespace = config.metadata.get("namespace", self.default_namespace)
        
        # Resource requirements
        resources = {
            "requests": {
                "cpu": config.compute.get("cpu", "2"),
                "memory": config.compute.get("memory", "4Gi")
            },
            "limits": {
                "cpu": str(int(config.compute.get("cpu", "2")) * 2),
                "memory": str(int(config.compute.get("memory", "4Gi").rstrip("Gi")) * 2) + "Gi"
            }
        }
        
        # Add GPU if specified
        if config.compute.get("gpu"):
            resources["requests"]["nvidia.com/gpu"] = str(config.compute["gpu"])
            resources["limits"]["nvidia.com/gpu"] = str(config.compute["gpu"])
        
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app": name,
                    "component": "synthdata",
                    "managed-by": "synthdata-cli"
                }
            },
            "spec": {
                "replicas": config.compute.get("count", 3),
                "selector": {
                    "matchLabels": {
                        "app": name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": name,
                            "component": "synthdata"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "synthdata",
                            "image": config.metadata.get("image", "inferloop/synthdata:latest"),
                            "ports": [{
                                "containerPort": 8000,
                                "name": "http"
                            }],
                            "resources": resources,
                            "env": self._create_env_vars(config),
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }]
                    }
                }
            }
        }
        
        # Add volume mounts if storage specified
        if config.storage:
            deployment["spec"]["template"]["spec"]["volumes"] = [{
                "name": "data",
                "persistentVolumeClaim": {
                    "claimName": f"{name}-data"
                }
            }]
            deployment["spec"]["template"]["spec"]["containers"][0]["volumeMounts"] = [{
                "name": "data",
                "mountPath": "/data"
            }]
        
        return deployment
    
    def _create_service_manifest(self, config: ResourceConfig) -> Dict[str, Any]:
        """Create Kubernetes service manifest."""
        name = config.metadata.get("name", "synthdata-app")
        namespace = config.metadata.get("namespace", self.default_namespace)
        
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app": name,
                    "component": "synthdata"
                }
            },
            "spec": {
                "selector": {
                    "app": name
                },
                "ports": [{
                    "port": 80,
                    "targetPort": 8000,
                    "name": "http"
                }],
                "type": config.networking.get("service_type", "ClusterIP")
            }
        }
        
        return service
    
    def _create_env_vars(self, config: ResourceConfig) -> List[Dict[str, Any]]:
        """Create environment variables for the container."""
        env_vars = []
        
        # Database configuration
        if config.metadata.get("database_url"):
            env_vars.append({
                "name": "DATABASE_URL",
                "valueFrom": {
                    "secretKeyRef": {
                        "name": "synthdata-secrets",
                        "key": "database-url"
                    }
                }
            })
        
        # Storage configuration
        if config.metadata.get("s3_endpoint"):
            env_vars.extend([
                {"name": "S3_ENDPOINT", "value": config.metadata["s3_endpoint"]},
                {"name": "S3_BUCKET", "value": config.metadata.get("s3_bucket", "synthdata")},
                {"name": "S3_ACCESS_KEY", "valueFrom": {"secretKeyRef": {"name": "synthdata-secrets", "key": "s3-access-key"}}},
                {"name": "S3_SECRET_KEY", "valueFrom": {"secretKeyRef": {"name": "synthdata-secrets", "key": "s3-secret-key"}}}
            ])
        
        # Additional environment variables
        for key, value in config.metadata.get("env", {}).items():
            env_vars.append({"name": key, "value": str(value)})
        
        return env_vars
    
    def _apply_manifest(self, manifest: Dict[str, Any]):
        """Apply a Kubernetes manifest."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(manifest, f)
            temp_path = f.name
        
        try:
            subprocess.run(
                self.kubectl_base_cmd + ["apply", "-f", temp_path],
                check=True,
                capture_output=True
            )
        finally:
            Path(temp_path).unlink()
    
    def _wait_for_deployment(self, name: str, namespace: str, timeout: int = 300) -> bool:
        """Wait for a deployment to be ready."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result = subprocess.run(
                    self.kubectl_base_cmd + [
                        "get", "deployment", name,
                        "-n", namespace,
                        "-o", "json"
                    ],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                deployment = json.loads(result.stdout)
                status = deployment.get("status", {})
                
                # Check if deployment is ready
                replicas = status.get("replicas", 0)
                ready_replicas = status.get("readyReplicas", 0)
                
                if replicas > 0 and replicas == ready_replicas:
                    return True
                
            except (subprocess.CalledProcessError, json.JSONDecodeError):
                pass
            
            time.sleep(5)
        
        return False
    
    def _get_service_endpoint(self, name: str, namespace: str) -> str:
        """Get the endpoint for a service."""
        try:
            result = subprocess.run(
                self.kubectl_base_cmd + [
                    "get", "service", name,
                    "-n", namespace,
                    "-o", "json"
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            service = json.loads(result.stdout)
            spec = service.get("spec", {})
            
            # For LoadBalancer type, get external IP
            if spec.get("type") == "LoadBalancer":
                status = service.get("status", {})
                lb = status.get("loadBalancer", {})
                ingress = lb.get("ingress", [])
                if ingress:
                    return f"http://{ingress[0].get('ip', ingress[0].get('hostname'))}:{spec['ports'][0]['port']}"
            
            # For NodePort, get node IP and port
            elif spec.get("type") == "NodePort":
                # Get a node IP
                nodes_result = subprocess.run(
                    self.kubectl_base_cmd + ["get", "nodes", "-o", "json"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                nodes = json.loads(nodes_result.stdout)
                if nodes["items"]:
                    node_ip = None
                    for addr in nodes["items"][0]["status"]["addresses"]:
                        if addr["type"] == "InternalIP":
                            node_ip = addr["address"]
                            break
                    if node_ip:
                        node_port = spec["ports"][0]["nodePort"]
                        return f"http://{node_ip}:{node_port}"
            
            # Default to cluster IP
            cluster_ip = spec.get("clusterIP")
            port = spec["ports"][0]["port"]
            return f"http://{cluster_ip}:{port}"
            
        except Exception:
            return f"http://{name}.{namespace}.svc.cluster.local"
    
    def deploy_storage(self, config: ResourceConfig) -> DeploymentResult:
        """Deploy storage resources (MinIO for S3-compatible storage).
        
        Args:
            config: Storage configuration
        
        Returns:
            DeploymentResult: Deployment outcome
        """
        if not self._authenticated:
            result = DeploymentResult()
            result.success = False
            result.message = "Provider not authenticated"
            return result
        
        try:
            storage_type = config.metadata.get("storage_type", "minio")
            
            if storage_type == "minio":
                return self._deploy_minio(config)
            elif storage_type == "nfs":
                return self._deploy_nfs_provisioner(config)
            else:
                raise ValidationError(f"Unsupported storage type: {storage_type}")
                
        except Exception as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"Storage deployment failed: {str(e)}"
            return result
    
    def _deploy_minio(self, config: ResourceConfig) -> DeploymentResult:
        """Deploy MinIO for S3-compatible storage."""
        namespace = config.metadata.get("namespace", self.default_namespace)
        name = "minio"
        
        # Create MinIO secrets
        secrets = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": "minio-secrets",
                "namespace": namespace
            },
            "stringData": {
                "accesskey": config.metadata.get("access_key", "minioadmin"),
                "secretkey": config.metadata.get("secret_key", "minioadmin123")
            }
        }
        self._apply_manifest(secrets)
        
        # Create MinIO StatefulSet
        statefulset = {
            "apiVersion": "apps/v1",
            "kind": "StatefulSet",
            "metadata": {
                "name": name,
                "namespace": namespace
            },
            "spec": {
                "serviceName": name,
                "replicas": config.compute.get("count", 4),
                "selector": {
                    "matchLabels": {"app": name}
                },
                "template": {
                    "metadata": {
                        "labels": {"app": name}
                    },
                    "spec": {
                        "containers": [{
                            "name": name,
                            "image": "minio/minio:latest",
                            "args": ["server", "/data", "--console-address", ":9001"],
                            "ports": [
                                {"containerPort": 9000, "name": "api"},
                                {"containerPort": 9001, "name": "console"}
                            ],
                            "env": [
                                {"name": "MINIO_ACCESS_KEY", "valueFrom": {"secretKeyRef": {"name": "minio-secrets", "key": "accesskey"}}},
                                {"name": "MINIO_SECRET_KEY", "valueFrom": {"secretKeyRef": {"name": "minio-secrets", "key": "secretkey"}}}
                            ],
                            "volumeMounts": [{
                                "name": "data",
                                "mountPath": "/data"
                            }],
                            "livenessProbe": {
                                "httpGet": {"path": "/minio/health/live", "port": 9000},
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/minio/health/ready", "port": 9000},
                                "initialDelaySeconds": 10,
                                "periodSeconds": 5
                            }
                        }]
                    }
                },
                "volumeClaimTemplates": [{
                    "metadata": {"name": "data"},
                    "spec": {
                        "accessModes": ["ReadWriteOnce"],
                        "storageClassName": config.storage.get("storage_class", "standard"),
                        "resources": {
                            "requests": {
                                "storage": config.storage.get("size", "100Gi")
                            }
                        }
                    }
                }]
            }
        }
        
        self._apply_manifest(statefulset)
        
        # Create MinIO service
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": name,
                "namespace": namespace
            },
            "spec": {
                "type": "ClusterIP",
                "ports": [
                    {"port": 9000, "targetPort": 9000, "name": "api"},
                    {"port": 9001, "targetPort": 9001, "name": "console"}
                ],
                "selector": {"app": name}
            }
        }
        
        self._apply_manifest(service)
        
        # Wait for MinIO to be ready
        if self._wait_for_statefulset(name, namespace):
            result = DeploymentResult()
            result.success = True
            result.resource_id = f"{namespace}/{name}"
            result.endpoint = f"http://{name}.{namespace}.svc.cluster.local:9000"
            result.message = "MinIO deployed successfully"
            result.metadata = {
                "console_url": f"http://{name}.{namespace}.svc.cluster.local:9001",
                "access_key": config.metadata.get("access_key", "minioadmin")
            }
            return result
        else:
            raise DeploymentError("MinIO deployment failed to become ready")
    
    def _wait_for_statefulset(self, name: str, namespace: str, timeout: int = 300) -> bool:
        """Wait for a StatefulSet to be ready."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result = subprocess.run(
                    self.kubectl_base_cmd + [
                        "get", "statefulset", name,
                        "-n", namespace,
                        "-o", "json"
                    ],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                statefulset = json.loads(result.stdout)
                status = statefulset.get("status", {})
                
                # Check if StatefulSet is ready
                replicas = status.get("replicas", 0)
                ready_replicas = status.get("readyReplicas", 0)
                
                if replicas > 0 and replicas == ready_replicas:
                    return True
                
            except (subprocess.CalledProcessError, json.JSONDecodeError):
                pass
            
            time.sleep(5)
        
        return False
    
    def deploy_database(self, config: ResourceConfig) -> DeploymentResult:
        """Deploy database resources (PostgreSQL with operator).
        
        Args:
            config: Database configuration
        
        Returns:
            DeploymentResult: Deployment outcome
        """
        if not self._authenticated:
            result = DeploymentResult()
            result.success = False
            result.message = "Provider not authenticated"
            return result
        
        try:
            db_type = config.metadata.get("db_type", "postgresql")
            
            if db_type == "postgresql":
                return self._deploy_postgresql(config)
            else:
                raise ValidationError(f"Unsupported database type: {db_type}")
                
        except Exception as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"Database deployment failed: {str(e)}"
            return result
    
    def _deploy_postgresql(self, config: ResourceConfig) -> DeploymentResult:
        """Deploy PostgreSQL using StatefulSet (simplified operator approach)."""
        namespace = config.metadata.get("namespace", self.default_namespace)
        name = config.metadata.get("name", "postgres")
        
        # Create PostgreSQL secrets
        secrets = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": f"{name}-secrets",
                "namespace": namespace
            },
            "stringData": {
                "postgres-password": config.metadata.get("password", "synthdata123"),
                "repmgr-password": "repmgr123"
            }
        }
        self._apply_manifest(secrets)
        
        # Create PostgreSQL ConfigMap
        configmap = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"{name}-config",
                "namespace": namespace
            },
            "data": {
                "postgresql.conf": """
                    max_connections = 200
                    shared_buffers = 256MB
                    effective_cache_size = 1GB
                    maintenance_work_mem = 64MB
                    checkpoint_completion_target = 0.9
                    wal_buffers = 16MB
                    default_statistics_target = 100
                    random_page_cost = 1.1
                    effective_io_concurrency = 200
                    work_mem = 4MB
                    min_wal_size = 1GB
                    max_wal_size = 4GB
                """
            }
        }
        self._apply_manifest(configmap)
        
        # Create PostgreSQL StatefulSet
        statefulset = {
            "apiVersion": "apps/v1",
            "kind": "StatefulSet",
            "metadata": {
                "name": name,
                "namespace": namespace
            },
            "spec": {
                "serviceName": name,
                "replicas": 1,  # Start with single instance, can add HA later
                "selector": {
                    "matchLabels": {"app": name}
                },
                "template": {
                    "metadata": {
                        "labels": {"app": name}
                    },
                    "spec": {
                        "containers": [{
                            "name": "postgres",
                            "image": "postgres:14",
                            "ports": [{"containerPort": 5432, "name": "postgres"}],
                            "env": [
                                {"name": "POSTGRES_DB", "value": "synthdata"},
                                {"name": "POSTGRES_USER", "value": "synthdata"},
                                {"name": "POSTGRES_PASSWORD", "valueFrom": {"secretKeyRef": {"name": f"{name}-secrets", "key": "postgres-password"}}},
                                {"name": "PGDATA", "value": "/var/lib/postgresql/data/pgdata"}
                            ],
                            "volumeMounts": [
                                {"name": "data", "mountPath": "/var/lib/postgresql/data"},
                                {"name": "config", "mountPath": "/etc/postgresql", "readOnly": True}
                            ],
                            "livenessProbe": {
                                "exec": {"command": ["pg_isready", "-U", "synthdata"]},
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "exec": {"command": ["pg_isready", "-U", "synthdata"]},
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            },
                            "resources": {
                                "requests": {
                                    "cpu": config.compute.get("cpu", "2"),
                                    "memory": config.compute.get("memory", "4Gi")
                                },
                                "limits": {
                                    "cpu": str(int(config.compute.get("cpu", "2")) * 2),
                                    "memory": str(int(config.compute.get("memory", "4Gi").rstrip("Gi")) * 2) + "Gi"
                                }
                            }
                        }],
                        "volumes": [{
                            "name": "config",
                            "configMap": {"name": f"{name}-config"}
                        }]
                    }
                },
                "volumeClaimTemplates": [{
                    "metadata": {"name": "data"},
                    "spec": {
                        "accessModes": ["ReadWriteOnce"],
                        "storageClassName": config.storage.get("storage_class", "standard"),
                        "resources": {
                            "requests": {
                                "storage": config.storage.get("size", "100Gi")
                            }
                        }
                    }
                }]
            }
        }
        
        self._apply_manifest(statefulset)
        
        # Create PostgreSQL service
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": name,
                "namespace": namespace
            },
            "spec": {
                "type": "ClusterIP",
                "ports": [{"port": 5432, "targetPort": 5432, "name": "postgres"}],
                "selector": {"app": name}
            }
        }
        
        self._apply_manifest(service)
        
        # Wait for PostgreSQL to be ready
        if self._wait_for_statefulset(name, namespace):
            result = DeploymentResult()
            result.success = True
            result.resource_id = f"{namespace}/{name}"
            result.endpoint = f"postgresql://synthdata@{name}.{namespace}.svc.cluster.local:5432/synthdata"
            result.message = "PostgreSQL deployed successfully"
            result.metadata = {
                "database": "synthdata",
                "user": "synthdata",
                "port": 5432
            }
            return result
        else:
            raise DeploymentError("PostgreSQL deployment failed to become ready")
    
    def deploy_monitoring(self, config: ResourceConfig) -> DeploymentResult:
        """Deploy monitoring stack (Prometheus + Grafana).
        
        Args:
            config: Monitoring configuration
        
        Returns:
            DeploymentResult: Deployment outcome
        """
        if not self._authenticated:
            result = DeploymentResult()
            result.success = False
            result.message = "Provider not authenticated"
            return result
        
        try:
            namespace = config.metadata.get("namespace", "monitoring")
            self._ensure_namespace(namespace)
            
            # Deploy Prometheus
            prom_result = self._deploy_prometheus(config, namespace)
            if not prom_result.success:
                return prom_result
            
            # Deploy Grafana
            grafana_result = self._deploy_grafana(config, namespace)
            if not grafana_result.success:
                return grafana_result
            
            result = DeploymentResult()
            result.success = True
            result.message = "Monitoring stack deployed successfully"
            result.metadata = {
                "prometheus_url": prom_result.endpoint,
                "grafana_url": grafana_result.endpoint
            }
            return result
            
        except Exception as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"Monitoring deployment failed: {str(e)}"
            return result
    
    def _deploy_prometheus(self, config: ResourceConfig, namespace: str) -> DeploymentResult:
        """Deploy Prometheus server."""
        # Create Prometheus ConfigMap
        configmap = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "prometheus-config",
                "namespace": namespace
            },
            "data": {
                "prometheus.yml": """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
    - role: pod
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
      action: keep
      regex: true
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
      action: replace
      target_label: __metrics_path__
      regex: (.+)
    - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
      action: replace
      regex: ([^:]+)(?::\d+)?;(\d+)
      replacement: $1:$2
      target_label: __address__
"""
            }
        }
        self._apply_manifest(configmap)
        
        # Create Prometheus Deployment
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "prometheus",
                "namespace": namespace
            },
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": {"app": "prometheus"}},
                "template": {
                    "metadata": {"labels": {"app": "prometheus"}},
                    "spec": {
                        "serviceAccountName": "prometheus",
                        "containers": [{
                            "name": "prometheus",
                            "image": "prom/prometheus:latest",
                            "args": [
                                "--config.file=/etc/prometheus/prometheus.yml",
                                "--storage.tsdb.path=/prometheus",
                                "--web.console.libraries=/usr/share/prometheus/console_libraries",
                                "--web.console.templates=/usr/share/prometheus/consoles"
                            ],
                            "ports": [{"containerPort": 9090}],
                            "volumeMounts": [
                                {"name": "config", "mountPath": "/etc/prometheus"},
                                {"name": "data", "mountPath": "/prometheus"}
                            ]
                        }],
                        "volumes": [
                            {"name": "config", "configMap": {"name": "prometheus-config"}},
                            {"name": "data", "emptyDir": {}}
                        ]
                    }
                }
            }
        }
        
        # Create ServiceAccount and RBAC
        self._create_prometheus_rbac(namespace)
        
        self._apply_manifest(deployment)
        
        # Create Prometheus Service
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "prometheus", "namespace": namespace},
            "spec": {
                "selector": {"app": "prometheus"},
                "ports": [{"port": 9090, "targetPort": 9090}]
            }
        }
        self._apply_manifest(service)
        
        if self._wait_for_deployment("prometheus", namespace):
            result = DeploymentResult()
            result.success = True
            result.endpoint = f"http://prometheus.{namespace}.svc.cluster.local:9090"
            return result
        else:
            raise DeploymentError("Prometheus deployment failed")
    
    def _create_prometheus_rbac(self, namespace: str):
        """Create RBAC resources for Prometheus."""
        # ServiceAccount
        sa = {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {"name": "prometheus", "namespace": namespace}
        }
        self._apply_manifest(sa)
        
        # ClusterRole
        role = {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRole",
            "metadata": {"name": "prometheus"},
            "rules": [
                {
                    "apiGroups": [""],
                    "resources": ["nodes", "services", "endpoints", "pods"],
                    "verbs": ["get", "list", "watch"]
                }
            ]
        }
        self._apply_manifest(role)
        
        # ClusterRoleBinding
        binding = {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRoleBinding",
            "metadata": {"name": "prometheus"},
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "ClusterRole",
                "name": "prometheus"
            },
            "subjects": [{
                "kind": "ServiceAccount",
                "name": "prometheus",
                "namespace": namespace
            }]
        }
        self._apply_manifest(binding)
    
    def _deploy_grafana(self, config: ResourceConfig, namespace: str) -> DeploymentResult:
        """Deploy Grafana."""
        # Create Grafana Deployment
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "grafana", "namespace": namespace},
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": {"app": "grafana"}},
                "template": {
                    "metadata": {"labels": {"app": "grafana"}},
                    "spec": {
                        "containers": [{
                            "name": "grafana",
                            "image": "grafana/grafana:latest",
                            "ports": [{"containerPort": 3000}],
                            "env": [
                                {"name": "GF_SECURITY_ADMIN_PASSWORD", "value": "admin"},
                                {"name": "GF_INSTALL_PLUGINS", "value": "grafana-clock-panel"}
                            ],
                            "volumeMounts": [{"name": "data", "mountPath": "/var/lib/grafana"}]
                        }],
                        "volumes": [{"name": "data", "emptyDir": {}}]
                    }
                }
            }
        }
        self._apply_manifest(deployment)
        
        # Create Grafana Service
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "grafana", "namespace": namespace},
            "spec": {
                "selector": {"app": "grafana"},
                "ports": [{"port": 3000, "targetPort": 3000}]
            }
        }
        self._apply_manifest(service)
        
        if self._wait_for_deployment("grafana", namespace):
            result = DeploymentResult()
            result.success = True
            result.endpoint = f"http://grafana.{namespace}.svc.cluster.local:3000"
            return result
        else:
            raise DeploymentError("Grafana deployment failed")
    
    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all synthdata deployments."""
        try:
            result = subprocess.run(
                self.kubectl_base_cmd + [
                    "get", "deployments",
                    "--all-namespaces",
                    "-l", "managed-by=synthdata-cli",
                    "-o", "json"
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            deployments = json.loads(result.stdout)
            
            deployment_list = []
            for item in deployments.get("items", []):
                metadata = item["metadata"]
                status = item["status"]
                
                deployment_list.append({
                    "name": metadata["name"],
                    "namespace": metadata["namespace"],
                    "ready": f"{status.get('readyReplicas', 0)}/{status.get('replicas', 0)}",
                    "created": metadata["creationTimestamp"]
                })
            
            return deployment_list
            
        except Exception as e:
            return []
    
    def delete_deployment(self, resource_id: str) -> bool:
        """Delete a deployment.
        
        Args:
            resource_id: Resource ID in format namespace/name
        
        Returns:
            bool: True if deletion successful
        """
        try:
            namespace, name = resource_id.split("/")
            
            # Delete deployment
            subprocess.run(
                self.kubectl_base_cmd + [
                    "delete", "deployment", name,
                    "-n", namespace
                ],
                check=True,
                capture_output=True
            )
            
            # Delete service
            subprocess.run(
                self.kubectl_base_cmd + [
                    "delete", "service", name,
                    "-n", namespace
                ],
                capture_output=True
            )
            
            return True
            
        except Exception:
            return False
    
    def get_deployment_status(self, resource_id: str) -> Dict[str, Any]:
        """Get detailed status of a deployment.
        
        Args:
            resource_id: Resource ID in format namespace/name
        
        Returns:
            Dict containing deployment status
        """
        try:
            namespace, name = resource_id.split("/")
            
            # Get deployment status
            result = subprocess.run(
                self.kubectl_base_cmd + [
                    "get", "deployment", name,
                    "-n", namespace,
                    "-o", "json"
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            deployment = json.loads(result.stdout)
            status = deployment["status"]
            
            # Get pods
            pods_result = subprocess.run(
                self.kubectl_base_cmd + [
                    "get", "pods",
                    "-n", namespace,
                    "-l", f"app={name}",
                    "-o", "json"
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            pods = json.loads(pods_result.stdout)
            
            return {
                "deployment": name,
                "namespace": namespace,
                "replicas": {
                    "desired": status.get("replicas", 0),
                    "ready": status.get("readyReplicas", 0),
                    "available": status.get("availableReplicas", 0)
                },
                "conditions": status.get("conditions", []),
                "pods": [
                    {
                        "name": pod["metadata"]["name"],
                        "status": pod["status"]["phase"],
                        "ready": all(c["status"] == "True" for c in pod["status"].get("conditions", []) if c["type"] == "Ready")
                    }
                    for pod in pods.get("items", [])
                ]
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the on-premises environment.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check Kubernetes access
        try:
            subprocess.run(
                self.kubectl_base_cmd + ["cluster-info"],
                capture_output=True,
                check=True
            )
        except subprocess.CalledProcessError:
            issues.append("Cannot access Kubernetes cluster")
        
        # Check Kubernetes version
        try:
            result = subprocess.run(
                self.kubectl_base_cmd + ["version", "--short", "-o", "json"],
                capture_output=True,
                text=True,
                check=True
            )
            version_info = json.loads(result.stdout)
            server_version = version_info.get("serverVersion", {})
            major = int(server_version.get("major", "0"))
            minor = int(server_version.get("minor", "0").rstrip("+"))
            
            if major < 1 or (major == 1 and minor < 19):
                issues.append(f"Kubernetes version {major}.{minor} is too old (minimum 1.19)")
                
        except Exception:
            issues.append("Cannot determine Kubernetes version")
        
        # Check storage classes
        try:
            result = subprocess.run(
                self.kubectl_base_cmd + ["get", "storageclass", "-o", "json"],
                capture_output=True,
                text=True,
                check=True
            )
            storage_classes = json.loads(result.stdout)
            if not storage_classes.get("items"):
                issues.append("No storage classes found")
                
        except Exception:
            issues.append("Cannot check storage classes")
        
        # Check node resources
        try:
            result = subprocess.run(
                self.kubectl_base_cmd + ["get", "nodes", "-o", "json"],
                capture_output=True,
                text=True,
                check=True
            )
            nodes = json.loads(result.stdout)
            
            total_cpu = 0
            total_memory = 0
            
            for node in nodes.get("items", []):
                capacity = node["status"]["capacity"]
                total_cpu += int(capacity.get("cpu", "0"))
                mem_str = capacity.get("memory", "0Ki")
                # Convert memory to GB (rough conversion)
                mem_kb = int(mem_str.rstrip("Ki"))
                total_memory += mem_kb // (1024 * 1024)
            
            if total_cpu < 16:
                issues.append(f"Insufficient CPU resources: {total_cpu} cores (minimum 16)")
            if total_memory < 64:
                issues.append(f"Insufficient memory: {total_memory}GB (minimum 64GB)")
                
        except Exception:
            issues.append("Cannot check node resources")
        
        return len(issues) == 0, issues
    
    def deploy_with_helm(self, config: ResourceConfig) -> DeploymentResult:
        """Deploy using Helm charts.
        
        Args:
            config: Deployment configuration
            
        Returns:
            DeploymentResult
        """
        if not self._authenticated:
            result = DeploymentResult()
            result.success = False
            result.message = "Provider not authenticated"
            return result
        
        try:
            # Generate Helm chart
            chart_generator = HelmChartGenerator(
                chart_name=config.metadata.get("name", "synthdata"),
                version=config.metadata.get("version", "0.1.0")
            )
            
            chart_path = chart_generator.create_chart(config)
            chart_generator.create_notes()
            
            # Deploy using Helm
            result = self.helm_deployer.install(
                release_name=config.metadata.get("name", "synthdata"),
                chart_path=chart_path,
                namespace=config.metadata.get("namespace", self.default_namespace),
                values=config.metadata.get("helm_values", {}),
                wait=True,
                timeout="15m"
            )
            
            # Clean up temporary chart
            if chart_path.startswith("/tmp"):
                shutil.rmtree(Path(chart_path).parent)
            
            return result
            
        except Exception as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"Helm deployment failed: {str(e)}"
            return result
    
    def upgrade_with_helm(self, config: ResourceConfig) -> DeploymentResult:
        """Upgrade a Helm deployment.
        
        Args:
            config: Deployment configuration
            
        Returns:
            DeploymentResult
        """
        if not self._authenticated:
            result = DeploymentResult()
            result.success = False
            result.message = "Provider not authenticated"
            return result
        
        try:
            # Generate updated Helm chart
            chart_generator = HelmChartGenerator(
                chart_name=config.metadata.get("name", "synthdata"),
                version=config.metadata.get("version", "0.2.0")
            )
            
            chart_path = chart_generator.create_chart(config)
            
            # Upgrade using Helm
            result = self.helm_deployer.upgrade(
                release_name=config.metadata.get("name", "synthdata"),
                chart_path=chart_path,
                namespace=config.metadata.get("namespace", self.default_namespace),
                values=config.metadata.get("helm_values", {}),
                wait=True,
                timeout="15m"
            )
            
            # Clean up temporary chart
            if chart_path.startswith("/tmp"):
                shutil.rmtree(Path(chart_path).parent)
            
            return result
            
        except Exception as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"Helm upgrade failed: {str(e)}"
            return result
    
    def list_helm_releases(self, namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        """List Helm releases.
        
        Args:
            namespace: Kubernetes namespace (all if None)
            
        Returns:
            List of Helm releases
        """
        return self.helm_deployer.list_releases(namespace)
    
    def rollback_helm_release(
        self,
        release_name: str,
        revision: Optional[int] = None,
        namespace: str = "default"
    ) -> bool:
        """Rollback a Helm release.
        
        Args:
            release_name: Name of the release
            revision: Revision to rollback to
            namespace: Kubernetes namespace
            
        Returns:
            bool: True if successful
        """
        return self.helm_deployer.rollback(release_name, revision, namespace)