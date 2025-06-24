"""GitOps integration with ArgoCD and Flux."""

import subprocess
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile
import time

from ..base import ResourceConfig, DeploymentResult
from ..exceptions import DeploymentError


class GitOpsManager:
    """Manage GitOps deployments using ArgoCD and Flux."""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        self.kubeconfig_path = kubeconfig_path or str(Path.home() / ".kube" / "config")
        self.kubectl_base_cmd = ["kubectl", f"--kubeconfig={self.kubeconfig_path}"]
        self.argocd_cmd = ["argocd"]
    
    def install_argocd(self, namespace: str = "argocd") -> DeploymentResult:
        """Install ArgoCD for GitOps.
        
        Args:
            namespace: Namespace for ArgoCD
            
        Returns:
            DeploymentResult
        """
        try:
            # Create namespace
            subprocess.run(
                self.kubectl_base_cmd + ["create", "namespace", namespace],
                capture_output=True
            )
            
            # Install ArgoCD
            subprocess.run([
                "kubectl", "apply", "-n", namespace, "-f",
                "https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml"
            ], check=True)
            
            # Wait for ArgoCD to be ready
            subprocess.run([
                "kubectl", "wait", "--for=condition=Available",
                "--timeout=600s", "deployment/argocd-server",
                "-n", namespace
            ], check=True)
            
            # Create ArgoCD ingress
            ingress = self._create_argocd_ingress(namespace)
            self._apply_resource(ingress)
            
            # Get initial admin password
            admin_password = self._get_argocd_admin_password(namespace)
            
            result = DeploymentResult()
            result.success = True
            result.message = "ArgoCD installed successfully"
            result.endpoint = f"https://argocd.{namespace}.svc.cluster.local"
            result.metadata = {
                "namespace": namespace,
                "admin_password": admin_password,
                "admin_user": "admin"
            }
            
            return result
            
        except Exception as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"ArgoCD installation failed: {str(e)}"
            return result
    
    def _create_argocd_ingress(self, namespace: str) -> Dict[str, Any]:
        """Create ArgoCD ingress."""
        return {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": "argocd-server-ingress",
                "namespace": namespace,
                "annotations": {
                    "nginx.ingress.kubernetes.io/ssl-redirect": "true",
                    "nginx.ingress.kubernetes.io/backend-protocol": "GRPC",
                    "cert-manager.io/cluster-issuer": "selfsigned-issuer"
                }
            },
            "spec": {
                "ingressClassName": "nginx",
                "tls": [{
                    "hosts": ["argocd.local"],
                    "secretName": "argocd-server-tls"
                }],
                "rules": [{
                    "host": "argocd.local",
                    "http": {
                        "paths": [{
                            "path": "/",
                            "pathType": "Prefix",
                            "backend": {
                                "service": {
                                    "name": "argocd-server",
                                    "port": {"number": 443}
                                }
                            }
                        }]
                    }
                }]
            }
        }
    
    def _get_argocd_admin_password(self, namespace: str) -> str:
        """Get ArgoCD admin password."""
        try:
            result = subprocess.run([
                "kubectl", "get", "secret", "argocd-initial-admin-secret",
                "-n", namespace, "-o", "jsonpath={.data.password}"
            ], capture_output=True, text=True, check=True)
            
            import base64
            return base64.b64decode(result.stdout).decode()
        except:
            return "admin"
    
    def install_flux(self, namespace: str = "flux-system") -> DeploymentResult:
        """Install Flux for GitOps.
        
        Args:
            namespace: Namespace for Flux
            
        Returns:
            DeploymentResult
        """
        try:
            # Check if flux CLI is available
            subprocess.run(["flux", "version"], capture_output=True, check=True)
            
            # Bootstrap Flux
            subprocess.run([
                "flux", "install", "--namespace", namespace
            ], check=True)
            
            # Wait for Flux to be ready
            subprocess.run([
                "kubectl", "wait", "--for=condition=Available",
                "--timeout=300s", "deployment/helm-controller",
                "deployment/kustomize-controller",
                "deployment/notification-controller",
                "deployment/source-controller",
                "-n", namespace
            ], check=True)
            
            result = DeploymentResult()
            result.success = True
            result.message = "Flux installed successfully"
            result.metadata = {"namespace": namespace}
            
            return result
            
        except subprocess.CalledProcessError as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"Flux installation failed: {str(e)}"
            return result
    
    def create_argocd_application(self, config: ResourceConfig) -> DeploymentResult:
        """Create ArgoCD application.
        
        Args:
            config: Application configuration
            
        Returns:
            DeploymentResult
        """
        try:
            app_name = config.metadata.get("name", "synthdata")
            repo_url = config.metadata.get("repo_url", "https://github.com/inferloop/synthdata-config")
            target_revision = config.metadata.get("target_revision", "HEAD")
            path = config.metadata.get("path", ".")
            dest_namespace = config.metadata.get("dest_namespace", "synthdata")
            
            application = {
                "apiVersion": "argoproj.io/v1alpha1",
                "kind": "Application",
                "metadata": {
                    "name": app_name,
                    "namespace": "argocd"
                },
                "spec": {
                    "project": "default",
                    "source": {
                        "repoURL": repo_url,
                        "targetRevision": target_revision,
                        "path": path
                    },
                    "destination": {
                        "server": "https://kubernetes.default.svc",
                        "namespace": dest_namespace
                    },
                    "syncPolicy": {
                        "automated": {
                            "prune": config.metadata.get("auto_prune", True),
                            "selfHeal": config.metadata.get("auto_heal", True)
                        },
                        "syncOptions": [
                            "CreateNamespace=true"
                        ]
                    }
                }
            }
            
            # Add Helm configuration if specified
            if config.metadata.get("helm_chart"):
                application["spec"]["source"]["helm"] = {
                    "valueFiles": config.metadata.get("helm_values_files", ["values.yaml"]),
                    "parameters": config.metadata.get("helm_parameters", [])
                }
            
            # Add Kustomize configuration if specified
            if config.metadata.get("kustomize_path"):
                application["spec"]["source"]["kustomize"] = {
                    "images": config.metadata.get("kustomize_images", [])
                }
            
            self._apply_resource(application)
            
            result = DeploymentResult()
            result.success = True
            result.message = f"ArgoCD application '{app_name}' created"
            result.resource_id = f"argocd/{app_name}"
            result.metadata = {
                "app_name": app_name,
                "repo_url": repo_url,
                "dest_namespace": dest_namespace
            }
            
            return result
            
        except Exception as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"ArgoCD application creation failed: {str(e)}"
            return result
    
    def create_flux_source(self, config: ResourceConfig) -> DeploymentResult:
        """Create Flux git source.
        
        Args:
            config: Source configuration
            
        Returns:
            DeploymentResult
        """
        try:
            source_name = config.metadata.get("name", "synthdata-source")
            repo_url = config.metadata.get("repo_url", "https://github.com/inferloop/synthdata-config")
            branch = config.metadata.get("branch", "main")
            interval = config.metadata.get("interval", "1m")
            namespace = config.metadata.get("namespace", "flux-system")
            
            git_source = {
                "apiVersion": "source.toolkit.fluxcd.io/v1beta2",
                "kind": "GitRepository",
                "metadata": {
                    "name": source_name,
                    "namespace": namespace
                },
                "spec": {
                    "interval": interval,
                    "ref": {"branch": branch},
                    "url": repo_url
                }
            }
            
            # Add authentication if provided
            if config.metadata.get("secret_ref"):
                git_source["spec"]["secretRef"] = {
                    "name": config.metadata["secret_ref"]
                }
            
            self._apply_resource(git_source)
            
            result = DeploymentResult()
            result.success = True
            result.message = f"Flux git source '{source_name}' created"
            result.resource_id = f"{namespace}/{source_name}"
            result.metadata = {
                "source_name": source_name,
                "repo_url": repo_url,
                "branch": branch
            }
            
            return result
            
        except Exception as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"Flux source creation failed: {str(e)}"
            return result
    
    def create_flux_kustomization(self, config: ResourceConfig) -> DeploymentResult:
        """Create Flux kustomization.
        
        Args:
            config: Kustomization configuration
            
        Returns:
            DeploymentResult
        """
        try:
            kustomize_name = config.metadata.get("name", "synthdata-kustomization")
            source_ref = config.metadata.get("source_ref", "synthdata-source")
            path = config.metadata.get("path", "./")
            target_namespace = config.metadata.get("target_namespace", "synthdata")
            interval = config.metadata.get("interval", "10m")
            namespace = config.metadata.get("namespace", "flux-system")
            
            kustomization = {
                "apiVersion": "kustomize.toolkit.fluxcd.io/v1beta2",
                "kind": "Kustomization",
                "metadata": {
                    "name": kustomize_name,
                    "namespace": namespace
                },
                "spec": {
                    "interval": interval,
                    "path": path,
                    "prune": config.metadata.get("prune", True),
                    "sourceRef": {
                        "kind": "GitRepository",
                        "name": source_ref
                    },
                    "targetNamespace": target_namespace
                }
            }
            
            # Add health checks if specified
            if config.metadata.get("health_checks"):
                kustomization["spec"]["healthChecks"] = config.metadata["health_checks"]
            
            # Add post-build substitutions if specified
            if config.metadata.get("substitutions"):
                kustomization["spec"]["postBuild"] = {
                    "substitute": config.metadata["substitutions"]
                }
            
            self._apply_resource(kustomization)
            
            result = DeploymentResult()
            result.success = True
            result.message = f"Flux kustomization '{kustomize_name}' created"
            result.resource_id = f"{namespace}/{kustomize_name}"
            
            return result
            
        except Exception as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"Flux kustomization creation failed: {str(e)}"
            return result
    
    def create_helm_release(self, config: ResourceConfig) -> DeploymentResult:
        """Create Flux Helm release.
        
        Args:
            config: Helm release configuration
            
        Returns:
            DeploymentResult
        """
        try:
            release_name = config.metadata.get("name", "synthdata")
            chart_name = config.metadata.get("chart", "synthdata")
            chart_version = config.metadata.get("version", "*")
            repo_name = config.metadata.get("repo_name", "synthdata-repo")
            target_namespace = config.metadata.get("target_namespace", "synthdata")
            interval = config.metadata.get("interval", "10m")
            namespace = config.metadata.get("namespace", "flux-system")
            
            # Create Helm repository source first
            helm_repo = {
                "apiVersion": "source.toolkit.fluxcd.io/v1beta2",
                "kind": "HelmRepository",
                "metadata": {
                    "name": repo_name,
                    "namespace": namespace
                },
                "spec": {
                    "interval": "1h",
                    "url": config.metadata.get("repo_url", "https://charts.inferloop.com")
                }
            }
            
            self._apply_resource(helm_repo)
            
            # Create Helm release
            helm_release = {
                "apiVersion": "helm.toolkit.fluxcd.io/v2beta1",
                "kind": "HelmRelease",
                "metadata": {
                    "name": release_name,
                    "namespace": namespace
                },
                "spec": {
                    "interval": interval,
                    "chart": {
                        "spec": {
                            "chart": chart_name,
                            "version": chart_version,
                            "sourceRef": {
                                "kind": "HelmRepository",
                                "name": repo_name
                            }
                        }
                    },
                    "targetNamespace": target_namespace,
                    "install": {
                        "createNamespace": True
                    }
                }
            }
            
            # Add values if specified
            if config.metadata.get("values"):
                helm_release["spec"]["values"] = config.metadata["values"]
            
            # Add values from ConfigMap/Secret if specified
            if config.metadata.get("values_from"):
                helm_release["spec"]["valuesFrom"] = config.metadata["values_from"]
            
            self._apply_resource(helm_release)
            
            result = DeploymentResult()
            result.success = True
            result.message = f"Flux Helm release '{release_name}' created"
            result.resource_id = f"{namespace}/{release_name}"
            
            return result
            
        except Exception as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"Flux Helm release creation failed: {str(e)}"
            return result
    
    def sync_argocd_app(self, app_name: str, namespace: str = "argocd") -> DeploymentResult:
        """Trigger ArgoCD application sync.
        
        Args:
            app_name: Application name
            namespace: ArgoCD namespace
            
        Returns:
            DeploymentResult
        """
        try:
            # Patch application to trigger sync
            patch = {
                "operation": {
                    "initiatedBy": {"username": "admin"},
                    "info": [{"name": "Reason", "value": "Manual sync triggered"}]
                }
            }
            
            subprocess.run([
                "kubectl", "patch", "application", app_name,
                "-n", namespace, "--type", "merge",
                "-p", json.dumps({"operation": patch})
            ], check=True)
            
            result = DeploymentResult()
            result.success = True
            result.message = f"ArgoCD application '{app_name}' sync triggered"
            
            return result
            
        except Exception as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"ArgoCD sync failed: {str(e)}"
            return result
    
    def get_argocd_app_status(self, app_name: str, namespace: str = "argocd") -> Dict[str, Any]:
        """Get ArgoCD application status.
        
        Args:
            app_name: Application name
            namespace: ArgoCD namespace
            
        Returns:
            Application status
        """
        try:
            result = subprocess.run([
                "kubectl", "get", "application", app_name,
                "-n", namespace, "-o", "json"
            ], capture_output=True, text=True, check=True)
            
            app_data = json.loads(result.stdout)
            status = app_data.get("status", {})
            
            return {
                "name": app_name,
                "health": status.get("health", {}).get("status"),
                "sync": status.get("sync", {}).get("status"),
                "revision": status.get("sync", {}).get("revision"),
                "resources": len(status.get("resources", [])),
                "conditions": status.get("conditions", [])
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_flux_status(self, namespace: str = "flux-system") -> Dict[str, Any]:
        """Get Flux system status.
        
        Args:
            namespace: Flux namespace
            
        Returns:
            Flux status
        """
        try:
            # Get GitRepository status
            git_repos = subprocess.run([
                "kubectl", "get", "gitrepository",
                "-n", namespace, "-o", "json"
            ], capture_output=True, text=True, check=True)
            
            # Get Kustomization status
            kustomizations = subprocess.run([
                "kubectl", "get", "kustomization",
                "-n", namespace, "-o", "json"
            ], capture_output=True, text=True, check=True)
            
            # Get HelmRelease status
            helm_releases = subprocess.run([
                "kubectl", "get", "helmrelease",
                "-n", namespace, "-o", "json"
            ], capture_output=True, text=True, check=True)
            
            git_data = json.loads(git_repos.stdout)
            kustomize_data = json.loads(kustomizations.stdout)
            helm_data = json.loads(helm_releases.stdout)
            
            return {
                "git_repositories": len(git_data.get("items", [])),
                "kustomizations": len(kustomize_data.get("items", [])),
                "helm_releases": len(helm_data.get("items", [])),
                "sources_ready": sum(1 for item in git_data.get("items", [])
                                   if item.get("status", {}).get("conditions", [{}])[-1].get("status") == "True"),
                "kustomizations_ready": sum(1 for item in kustomize_data.get("items", [])
                                          if item.get("status", {}).get("conditions", [{}])[-1].get("status") == "True")
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _apply_resource(self, resource: Dict[str, Any]):
        """Apply a Kubernetes resource."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(resource, f)
            temp_path = f.name
        
        try:
            subprocess.run(
                self.kubectl_base_cmd + ["apply", "-f", temp_path],
                check=True
            )
        finally:
            Path(temp_path).unlink()
    
    def create_git_credentials_secret(
        self,
        secret_name: str,
        username: str,
        password: str,
        namespace: str = "flux-system"
    ) -> DeploymentResult:
        """Create git credentials secret for Flux.
        
        Args:
            secret_name: Secret name
            username: Git username
            password: Git password/token
            namespace: Namespace
            
        Returns:
            DeploymentResult
        """
        try:
            subprocess.run([
                "kubectl", "create", "secret", "generic", secret_name,
                "--from-literal", f"username={username}",
                "--from-literal", f"password={password}",
                "-n", namespace
            ], check=True)
            
            result = DeploymentResult()
            result.success = True
            result.message = f"Git credentials secret '{secret_name}' created"
            
            return result
            
        except Exception as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"Secret creation failed: {str(e)}"
            return result