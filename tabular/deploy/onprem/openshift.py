"""OpenShift Container Platform deployment provider."""

import subprocess
import json
import yaml
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from ..base import BaseDeploymentProvider, DeploymentResult, ResourceConfig
from ..exceptions import DeploymentError, ValidationError
from .provider import OnPremKubernetesProvider


class OpenShiftProvider(OnPremKubernetesProvider):
    """Provider for OpenShift Container Platform deployments."""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        """Initialize the OpenShift provider.
        
        Args:
            kubeconfig_path: Path to kubeconfig file
        """
        super().__init__(kubeconfig_path)
        self.oc_base_cmd = ["oc", f"--kubeconfig={self.kubeconfig_path}"]
        self._validate_openshift_access()
    
    def _validate_openshift_access(self):
        """Validate OpenShift CLI access."""
        try:
            result = subprocess.run(
                self.oc_base_cmd + ["version"],
                capture_output=True,
                text=True,
                check=True
            )
            if "Server Version" not in result.stdout:
                raise ValidationError("Cannot connect to OpenShift cluster")
        except subprocess.CalledProcessError:
            raise ValidationError("OpenShift CLI (oc) not available or cluster not accessible")
    
    def authenticate(self, **kwargs) -> bool:
        """Authenticate with OpenShift cluster.
        
        Args:
            **kwargs: Authentication parameters
                - token: OAuth token
                - username: Username for login
                - password: Password for login
                - server: API server URL
                
        Returns:
            bool: True if authentication successful
        """
        try:
            if "token" in kwargs:
                # Token-based authentication
                cmd = self.oc_base_cmd + ["login", "--token", kwargs["token"]]
                if "server" in kwargs:
                    cmd.extend(["--server", kwargs["server"]])
                    
                subprocess.run(cmd, check=True, capture_output=True)
                
            elif "username" in kwargs and "password" in kwargs:
                # Username/password authentication
                cmd = self.oc_base_cmd + ["login"]
                if "server" in kwargs:
                    cmd.extend(["-s", kwargs["server"]])
                cmd.extend(["-u", kwargs["username"], "-p", kwargs["password"]])
                
                subprocess.run(cmd, check=True, capture_output=True)
            
            else:
                # Use existing authentication
                result = subprocess.run(
                    self.oc_base_cmd + ["whoami"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                
            # Create or switch to project
            project = kwargs.get("project", "synthdata")
            self._ensure_project(project)
            self.default_namespace = project
            
            self._authenticated = True
            return True
            
        except subprocess.CalledProcessError:
            self._authenticated = False
            return False
    
    def _ensure_project(self, project: str):
        """Ensure an OpenShift project exists."""
        try:
            # Check if project exists
            subprocess.run(
                self.oc_base_cmd + ["get", "project", project],
                check=True,
                capture_output=True
            )
            # Switch to project
            subprocess.run(
                self.oc_base_cmd + ["project", project],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError:
            # Create new project
            subprocess.run(
                self.oc_base_cmd + ["new-project", project,
                    "--display-name", "Inferloop Synthetic Data",
                    "--description", "Synthetic data generation platform"
                ],
                check=True,
                capture_output=True
            )
    
    def deploy_container(self, config: ResourceConfig) -> DeploymentResult:
        """Deploy using OpenShift-specific features.
        
        Args:
            config: Container deployment configuration
            
        Returns:
            DeploymentResult
        """
        if not self._authenticated:
            result = DeploymentResult()
            result.success = False
            result.message = "Provider not authenticated"
            return result
        
        try:
            name = config.metadata.get("name", "synthdata-app")
            
            # Create DeploymentConfig (OpenShift-specific)
            dc = self._create_deployment_config(config)
            self._apply_openshift_resource(dc)
            
            # Create Service
            service = self._create_service_manifest(config)
            self._apply_openshift_resource(service)
            
            # Create Route (OpenShift-specific)
            route = self._create_route(config)
            self._apply_openshift_resource(route)
            
            # Wait for deployment
            if self._wait_for_deployment_config(name, self.default_namespace):
                # Get route URL
                route_url = self._get_route_url(name, self.default_namespace)
                
                result = DeploymentResult()
                result.success = True
                result.resource_id = f"{self.default_namespace}/{name}"
                result.endpoint = route_url
                result.message = "OpenShift deployment successful"
                result.metadata = {
                    "deployment_config": name,
                    "project": self.default_namespace,
                    "route": route_url
                }
                return result
            else:
                raise DeploymentError("DeploymentConfig failed to become ready")
                
        except Exception as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"OpenShift deployment failed: {str(e)}"
            return result
    
    def _create_deployment_config(self, config: ResourceConfig) -> Dict[str, Any]:
        """Create OpenShift DeploymentConfig."""
        name = config.metadata.get("name", "synthdata-app")
        
        dc = {
            "apiVersion": "apps.openshift.io/v1",
            "kind": "DeploymentConfig",
            "metadata": {
                "name": name,
                "labels": {
                    "app": name,
                    "app.kubernetes.io/name": name,
                    "app.kubernetes.io/component": "synthdata",
                    "app.kubernetes.io/managed-by": "synthdata-cli"
                }
            },
            "spec": {
                "replicas": config.compute.get("count", 3),
                "selector": {
                    "app": name,
                    "deploymentconfig": name
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": name,
                            "deploymentconfig": name
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "synthdata",
                            "image": config.metadata.get("image", "inferloop/synthdata:latest"),
                            "ports": [{
                                "containerPort": 8000,
                                "protocol": "TCP"
                            }],
                            "resources": {
                                "requests": {
                                    "cpu": config.compute.get("cpu", "2"),
                                    "memory": config.compute.get("memory", "4Gi")
                                },
                                "limits": {
                                    "cpu": str(int(config.compute.get("cpu", "2")) * 2),
                                    "memory": str(int(config.compute.get("memory", "4Gi").rstrip("Gi")) * 2) + "Gi"
                                }
                            },
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
                },
                "triggers": [
                    {
                        "type": "ConfigChange"
                    },
                    {
                        "type": "ImageChange",
                        "imageChangeParams": {
                            "automatic": True,
                            "containerNames": ["synthdata"],
                            "from": {
                                "kind": "ImageStreamTag",
                                "name": f"{name}:latest"
                            }
                        }
                    }
                ]
            }
        }
        
        return dc
    
    def _create_route(self, config: ResourceConfig) -> Dict[str, Any]:
        """Create OpenShift Route for external access."""
        name = config.metadata.get("name", "synthdata-app")
        hostname = config.networking.get("hostname")
        
        route = {
            "apiVersion": "route.openshift.io/v1",
            "kind": "Route",
            "metadata": {
                "name": name,
                "labels": {
                    "app": name
                }
            },
            "spec": {
                "to": {
                    "kind": "Service",
                    "name": name
                },
                "port": {
                    "targetPort": "http"
                },
                "tls": {
                    "termination": "edge",
                    "insecureEdgeTerminationPolicy": "Redirect"
                }
            }
        }
        
        if hostname:
            route["spec"]["host"] = hostname
        
        return route
    
    def _apply_openshift_resource(self, resource: Dict[str, Any]):
        """Apply an OpenShift resource."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(resource, f)
            temp_path = f.name
        
        try:
            subprocess.run(
                self.oc_base_cmd + ["apply", "-f", temp_path],
                check=True,
                capture_output=True
            )
        finally:
            Path(temp_path).unlink()
    
    def _wait_for_deployment_config(self, name: str, namespace: str, timeout: int = 300) -> bool:
        """Wait for DeploymentConfig to be ready."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result = subprocess.run(
                    self.oc_base_cmd + [
                        "get", "dc", name,
                        "-n", namespace,
                        "-o", "json"
                    ],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                dc = json.loads(result.stdout)
                status = dc.get("status", {})
                
                # Check if deployment is ready
                replicas = dc["spec"].get("replicas", 0)
                ready_replicas = status.get("readyReplicas", 0)
                
                if replicas > 0 and replicas == ready_replicas:
                    return True
                    
            except (subprocess.CalledProcessError, json.JSONDecodeError):
                pass
            
            time.sleep(5)
        
        return False
    
    def _get_route_url(self, name: str, namespace: str) -> str:
        """Get the URL from an OpenShift Route."""
        try:
            result = subprocess.run(
                self.oc_base_cmd + [
                    "get", "route", name,
                    "-n", namespace,
                    "-o", "json"
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            route = json.loads(result.stdout)
            host = route["spec"]["host"]
            tls = route["spec"].get("tls")
            
            protocol = "https" if tls else "http"
            return f"{protocol}://{host}"
            
        except Exception:
            return f"http://{name}.{namespace}.svc.cluster.local"
    
    def create_build_config(self, config: ResourceConfig) -> DeploymentResult:
        """Create OpenShift BuildConfig for source-to-image builds.
        
        Args:
            config: Build configuration
            
        Returns:
            DeploymentResult
        """
        if not self._authenticated:
            result = DeploymentResult()
            result.success = False
            result.message = "Provider not authenticated"
            return result
        
        try:
            name = config.metadata.get("name", "synthdata-app")
            
            # Create ImageStream
            imagestream = {
                "apiVersion": "image.openshift.io/v1",
                "kind": "ImageStream",
                "metadata": {
                    "name": name,
                    "labels": {"app": name}
                }
            }
            self._apply_openshift_resource(imagestream)
            
            # Create BuildConfig
            bc = {
                "apiVersion": "build.openshift.io/v1",
                "kind": "BuildConfig",
                "metadata": {
                    "name": name,
                    "labels": {"app": name}
                },
                "spec": {
                    "source": {
                        "type": "Git",
                        "git": {
                            "uri": config.metadata.get("git_url", "https://github.com/inferloop/synthdata"),
                            "ref": config.metadata.get("git_ref", "main")
                        },
                        "contextDir": config.metadata.get("context_dir", "/")
                    },
                    "strategy": {
                        "type": "Source",
                        "sourceStrategy": {
                            "from": {
                                "kind": "ImageStreamTag",
                                "namespace": "openshift",
                                "name": config.metadata.get("builder_image", "python:3.9")
                            }
                        }
                    },
                    "output": {
                        "to": {
                            "kind": "ImageStreamTag",
                            "name": f"{name}:latest"
                        }
                    },
                    "triggers": [
                        {"type": "ConfigChange"},
                        {
                            "type": "ImageChange",
                            "imageChange": {}
                        }
                    ]
                }
            }
            
            self._apply_openshift_resource(bc)
            
            # Start build
            subprocess.run(
                self.oc_base_cmd + ["start-build", name, "--follow"],
                check=True
            )
            
            result = DeploymentResult()
            result.success = True
            result.message = "BuildConfig created and build started"
            result.resource_id = f"{self.default_namespace}/{name}"
            result.metadata = {
                "build_config": name,
                "image_stream": name
            }
            
            return result
            
        except Exception as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"BuildConfig creation failed: {str(e)}"
            return result
    
    def create_template(self, config: ResourceConfig) -> str:
        """Create an OpenShift Template for the application.
        
        Args:
            config: Application configuration
            
        Returns:
            Path to the created template file
        """
        name = config.metadata.get("name", "synthdata")
        
        template = {
            "apiVersion": "template.openshift.io/v1",
            "kind": "Template",
            "metadata": {
                "name": f"{name}-template",
                "annotations": {
                    "description": "Inferloop Synthetic Data Platform",
                    "tags": "synthdata,ml,data-generation",
                    "iconClass": "icon-python"
                }
            },
            "parameters": [
                {
                    "name": "APPLICATION_NAME",
                    "description": "Application name",
                    "value": name,
                    "required": True
                },
                {
                    "name": "REPLICAS",
                    "description": "Number of replicas",
                    "value": "3",
                    "required": True
                },
                {
                    "name": "CPU_REQUEST",
                    "description": "CPU request",
                    "value": "2",
                    "required": True
                },
                {
                    "name": "MEMORY_REQUEST",
                    "description": "Memory request",
                    "value": "4Gi",
                    "required": True
                },
                {
                    "name": "IMAGE",
                    "description": "Container image",
                    "value": "inferloop/synthdata:latest",
                    "required": True
                }
            ],
            "objects": [
                {
                    "apiVersion": "apps.openshift.io/v1",
                    "kind": "DeploymentConfig",
                    "metadata": {
                        "name": "${APPLICATION_NAME}",
                        "labels": {
                            "app": "${APPLICATION_NAME}"
                        }
                    },
                    "spec": {
                        "replicas": "${{REPLICAS}}",
                        "selector": {
                            "app": "${APPLICATION_NAME}",
                            "deploymentconfig": "${APPLICATION_NAME}"
                        },
                        "template": {
                            "metadata": {
                                "labels": {
                                    "app": "${APPLICATION_NAME}",
                                    "deploymentconfig": "${APPLICATION_NAME}"
                                }
                            },
                            "spec": {
                                "containers": [{
                                    "name": "synthdata",
                                    "image": "${IMAGE}",
                                    "ports": [{
                                        "containerPort": 8000,
                                        "protocol": "TCP"
                                    }],
                                    "resources": {
                                        "requests": {
                                            "cpu": "${CPU_REQUEST}",
                                            "memory": "${MEMORY_REQUEST}"
                                        }
                                    }
                                }]
                            }
                        }
                    }
                },
                {
                    "apiVersion": "v1",
                    "kind": "Service",
                    "metadata": {
                        "name": "${APPLICATION_NAME}",
                        "labels": {
                            "app": "${APPLICATION_NAME}"
                        }
                    },
                    "spec": {
                        "ports": [{
                            "name": "http",
                            "port": 80,
                            "targetPort": 8000
                        }],
                        "selector": {
                            "app": "${APPLICATION_NAME}",
                            "deploymentconfig": "${APPLICATION_NAME}"
                        }
                    }
                },
                {
                    "apiVersion": "route.openshift.io/v1",
                    "kind": "Route",
                    "metadata": {
                        "name": "${APPLICATION_NAME}",
                        "labels": {
                            "app": "${APPLICATION_NAME}"
                        }
                    },
                    "spec": {
                        "to": {
                            "kind": "Service",
                            "name": "${APPLICATION_NAME}"
                        },
                        "port": {
                            "targetPort": "http"
                        },
                        "tls": {
                            "termination": "edge"
                        }
                    }
                }
            ]
        }
        
        # Save template to file
        template_path = Path(tempfile.mkdtemp()) / f"{name}-template.yaml"
        with open(template_path, "w") as f:
            yaml.dump(template, f, default_flow_style=False)
        
        return str(template_path)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate OpenShift environment.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check OC CLI
        try:
            subprocess.run(
                ["oc", "version"],
                capture_output=True,
                check=True
            )
        except subprocess.CalledProcessError:
            issues.append("OpenShift CLI (oc) not available")
        
        # Check cluster access
        try:
            subprocess.run(
                self.oc_base_cmd + ["whoami"],
                capture_output=True,
                check=True
            )
        except subprocess.CalledProcessError:
            issues.append("Not logged in to OpenShift cluster")
        
        # Check OpenShift version
        try:
            result = subprocess.run(
                self.oc_base_cmd + ["version", "-o", "json"],
                capture_output=True,
                text=True,
                check=True
            )
            version_info = json.loads(result.stdout)
            server_version = version_info.get("serverVersion", {})
            
            # Check for OpenShift 4.x
            if not server_version.get("gitVersion", "").startswith("v4."):
                issues.append("OpenShift 4.x required")
                
        except Exception:
            issues.append("Cannot determine OpenShift version")
        
        # Check project permissions
        try:
            subprocess.run(
                self.oc_base_cmd + ["auth", "can-i", "create", "deploymentconfig"],
                capture_output=True,
                check=True
            )
        except subprocess.CalledProcessError:
            issues.append("Insufficient permissions to create resources")
        
        return len(issues) == 0, issues