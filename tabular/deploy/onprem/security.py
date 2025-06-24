"""Security configuration for on-premises deployments."""

import subprocess
import yaml
import json
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import datetime

from ..base import ResourceConfig, DeploymentResult
from ..exceptions import DeploymentError


class SecurityManager:
    """Manage security configurations for on-premises deployments."""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        self.kubeconfig_path = kubeconfig_path or str(Path.home() / ".kube" / "config")
        self.kubectl_base_cmd = ["kubectl", f"--kubeconfig={self.kubeconfig_path}"]
    
    def deploy_cert_manager(self, namespace: str = "cert-manager") -> DeploymentResult:
        """Deploy cert-manager for automatic certificate management.
        
        Args:
            namespace: Namespace for cert-manager
            
        Returns:
            DeploymentResult
        """
        try:
            # Apply cert-manager CRDs
            subprocess.run([
                "kubectl", "apply", "-f",
                "https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.crds.yaml"
            ], check=True)
            
            # Install cert-manager using kubectl
            subprocess.run([
                "kubectl", "apply", "-f",
                "https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml"
            ], check=True)
            
            # Wait for cert-manager to be ready
            subprocess.run([
                "kubectl", "wait", "--for=condition=Available",
                "--timeout=300s", "deployment/cert-manager",
                "-n", namespace
            ], check=True)
            
            result = DeploymentResult()
            result.success = True
            result.message = "cert-manager deployed successfully"
            result.metadata = {"namespace": namespace}
            return result
            
        except subprocess.CalledProcessError as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"cert-manager deployment failed: {str(e)}"
            return result
    
    def create_cluster_issuer(self, issuer_type: str = "self-signed", **kwargs) -> DeploymentResult:
        """Create a ClusterIssuer for certificate generation.
        
        Args:
            issuer_type: Type of issuer (self-signed, letsencrypt, ca)
            **kwargs: Additional parameters based on issuer type
            
        Returns:
            DeploymentResult
        """
        try:
            if issuer_type == "self-signed":
                issuer = self._create_self_signed_issuer()
            elif issuer_type == "letsencrypt":
                issuer = self._create_letsencrypt_issuer(
                    email=kwargs.get("email", "admin@example.com"),
                    staging=kwargs.get("staging", False)
                )
            elif issuer_type == "ca":
                issuer = self._create_ca_issuer(
                    secret_name=kwargs.get("secret_name", "ca-key-pair")
                )
            else:
                raise ValueError(f"Unknown issuer type: {issuer_type}")
            
            # Apply the issuer
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(issuer, f)
                temp_path = f.name
            
            subprocess.run(
                self.kubectl_base_cmd + ["apply", "-f", temp_path],
                check=True
            )
            
            Path(temp_path).unlink()
            
            result = DeploymentResult()
            result.success = True
            result.message = f"ClusterIssuer '{issuer_type}' created"
            return result
            
        except Exception as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"ClusterIssuer creation failed: {str(e)}"
            return result
    
    def _create_self_signed_issuer(self) -> Dict[str, Any]:
        """Create self-signed ClusterIssuer."""
        return {
            "apiVersion": "cert-manager.io/v1",
            "kind": "ClusterIssuer",
            "metadata": {
                "name": "selfsigned-issuer"
            },
            "spec": {
                "selfSigned": {}
            }
        }
    
    def _create_letsencrypt_issuer(self, email: str, staging: bool = False) -> Dict[str, Any]:
        """Create Let's Encrypt ClusterIssuer."""
        server = (
            "https://acme-staging-v02.api.letsencrypt.org/directory" if staging
            else "https://acme-v02.api.letsencrypt.org/directory"
        )
        
        return {
            "apiVersion": "cert-manager.io/v1",
            "kind": "ClusterIssuer",
            "metadata": {
                "name": "letsencrypt-staging" if staging else "letsencrypt-prod"
            },
            "spec": {
                "acme": {
                    "server": server,
                    "email": email,
                    "privateKeySecretRef": {
                        "name": "letsencrypt-staging" if staging else "letsencrypt-prod"
                    },
                    "solvers": [{
                        "http01": {
                            "ingress": {
                                "class": "nginx"
                            }
                        }
                    }]
                }
            }
        }
    
    def _create_ca_issuer(self, secret_name: str) -> Dict[str, Any]:
        """Create CA ClusterIssuer."""
        return {
            "apiVersion": "cert-manager.io/v1",
            "kind": "ClusterIssuer",
            "metadata": {
                "name": "ca-issuer"
            },
            "spec": {
                "ca": {
                    "secretName": secret_name
                }
            }
        }
    
    def create_certificate(
        self,
        name: str,
        namespace: str,
        dns_names: List[str],
        issuer_name: str = "selfsigned-issuer"
    ) -> DeploymentResult:
        """Create a Certificate resource.
        
        Args:
            name: Certificate name
            namespace: Namespace
            dns_names: List of DNS names
            issuer_name: ClusterIssuer to use
            
        Returns:
            DeploymentResult
        """
        try:
            cert = {
                "apiVersion": "cert-manager.io/v1",
                "kind": "Certificate",
                "metadata": {
                    "name": name,
                    "namespace": namespace
                },
                "spec": {
                    "secretName": f"{name}-tls",
                    "dnsNames": dns_names,
                    "issuerRef": {
                        "name": issuer_name,
                        "kind": "ClusterIssuer"
                    }
                }
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(cert, f)
                temp_path = f.name
            
            subprocess.run(
                self.kubectl_base_cmd + ["apply", "-f", temp_path],
                check=True
            )
            
            Path(temp_path).unlink()
            
            result = DeploymentResult()
            result.success = True
            result.message = f"Certificate '{name}' created"
            result.metadata = {
                "secret_name": f"{name}-tls",
                "dns_names": dns_names
            }
            return result
            
        except Exception as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"Certificate creation failed: {str(e)}"
            return result
    
    def deploy_dex_oidc(self, config: ResourceConfig) -> DeploymentResult:
        """Deploy Dex OIDC provider for LDAP integration.
        
        Args:
            config: Dex configuration
            
        Returns:
            DeploymentResult
        """
        try:
            namespace = config.metadata.get("namespace", "auth")
            
            # Create namespace
            subprocess.run(
                self.kubectl_base_cmd + ["create", "namespace", namespace],
                capture_output=True
            )
            
            # Create Dex configuration
            dex_config = self._create_dex_config(config)
            
            # Create ConfigMap
            configmap = {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {
                    "name": "dex-config",
                    "namespace": namespace
                },
                "data": {
                    "config.yaml": yaml.dump(dex_config)
                }
            }
            
            self._apply_resource(configmap)
            
            # Create Dex deployment
            deployment = self._create_dex_deployment(namespace)
            self._apply_resource(deployment)
            
            # Create service
            service = {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": "dex",
                    "namespace": namespace
                },
                "spec": {
                    "selector": {"app": "dex"},
                    "ports": [
                        {"name": "http", "port": 5556, "targetPort": 5556},
                        {"name": "grpc", "port": 5557, "targetPort": 5557}
                    ]
                }
            }
            
            self._apply_resource(service)
            
            result = DeploymentResult()
            result.success = True
            result.message = "Dex OIDC provider deployed"
            result.endpoint = f"http://dex.{namespace}.svc.cluster.local:5556"
            result.metadata = {
                "namespace": namespace,
                "issuer": f"http://dex.{namespace}.svc.cluster.local:5556/dex"
            }
            return result
            
        except Exception as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"Dex deployment failed: {str(e)}"
            return result
    
    def _create_dex_config(self, config: ResourceConfig) -> Dict[str, Any]:
        """Create Dex configuration."""
        ldap_config = config.metadata.get("ldap", {})
        
        return {
            "issuer": "http://dex:5556/dex",
            "storage": {
                "type": "kubernetes",
                "config": {
                    "inCluster": True
                }
            },
            "web": {
                "http": "0.0.0.0:5556"
            },
            "grpc": {
                "addr": "0.0.0.0:5557"
            },
            "connectors": [{
                "type": "ldap",
                "id": "ldap",
                "name": "LDAP",
                "config": {
                    "host": ldap_config.get("host", "ldap.example.com:389"),
                    "insecureNoSSL": ldap_config.get("insecure", True),
                    "bindDN": ldap_config.get("bind_dn", "cn=admin,dc=example,dc=com"),
                    "bindPW": ldap_config.get("bind_password", "admin"),
                    "userSearch": {
                        "baseDN": ldap_config.get("user_base_dn", "ou=users,dc=example,dc=com"),
                        "filter": "(objectClass=person)",
                        "username": "uid",
                        "idAttr": "uid",
                        "emailAttr": "mail",
                        "nameAttr": "cn"
                    },
                    "groupSearch": {
                        "baseDN": ldap_config.get("group_base_dn", "ou=groups,dc=example,dc=com"),
                        "filter": "(objectClass=groupOfNames)",
                        "userMatchers": [{
                            "userAttr": "DN",
                            "groupAttr": "member"
                        }],
                        "nameAttr": "cn"
                    }
                }
            }],
            "oauth2": {
                "skipApprovalScreen": True
            },
            "staticClients": [{
                "id": "synthdata",
                "secret": config.metadata.get("client_secret", "synthdata-secret"),
                "name": "Synthetic Data Platform",
                "redirectURIs": config.metadata.get("redirect_uris", [
                    "http://localhost:8000/callback",
                    "https://synthdata.example.com/callback"
                ])
            }]
        }
    
    def _create_dex_deployment(self, namespace: str) -> Dict[str, Any]:
        """Create Dex deployment."""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "dex",
                "namespace": namespace
            },
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": {"app": "dex"}},
                "template": {
                    "metadata": {"labels": {"app": "dex"}},
                    "spec": {
                        "serviceAccountName": "dex",
                        "containers": [{
                            "name": "dex",
                            "image": "ghcr.io/dexidp/dex:v2.37.0",
                            "command": ["dex", "serve", "/etc/dex/config.yaml"],
                            "ports": [
                                {"containerPort": 5556},
                                {"containerPort": 5557}
                            ],
                            "volumeMounts": [{
                                "name": "config",
                                "mountPath": "/etc/dex"
                            }],
                            "env": [{
                                "name": "KUBERNETES_POD_NAMESPACE",
                                "valueFrom": {
                                    "fieldRef": {"fieldPath": "metadata.namespace"}
                                }
                            }]
                        }],
                        "volumes": [{
                            "name": "config",
                            "configMap": {"name": "dex-config"}
                        }]
                    }
                }
            }
        }
    
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
    
    def create_network_policies(self, namespace: str) -> DeploymentResult:
        """Create network policies for security.
        
        Args:
            namespace: Target namespace
            
        Returns:
            DeploymentResult
        """
        try:
            # Default deny all ingress
            deny_all = {
                "apiVersion": "networking.k8s.io/v1",
                "kind": "NetworkPolicy",
                "metadata": {
                    "name": "default-deny-ingress",
                    "namespace": namespace
                },
                "spec": {
                    "podSelector": {},
                    "policyTypes": ["Ingress"]
                }
            }
            
            self._apply_resource(deny_all)
            
            # Allow ingress for synthdata app
            allow_app = {
                "apiVersion": "networking.k8s.io/v1",
                "kind": "NetworkPolicy",
                "metadata": {
                    "name": "allow-synthdata-ingress",
                    "namespace": namespace
                },
                "spec": {
                    "podSelector": {
                        "matchLabels": {"app": "synthdata"}
                    },
                    "policyTypes": ["Ingress"],
                    "ingress": [{
                        "from": [
                            {"podSelector": {"matchLabels": {"app": "nginx-ingress"}}},
                            {"namespaceSelector": {"matchLabels": {"name": "monitoring"}}}
                        ],
                        "ports": [
                            {"protocol": "TCP", "port": 8000},
                            {"protocol": "TCP", "port": 8080}
                        ]
                    }]
                }
            }
            
            self._apply_resource(allow_app)
            
            # Allow database access
            allow_db = {
                "apiVersion": "networking.k8s.io/v1",
                "kind": "NetworkPolicy",
                "metadata": {
                    "name": "allow-database-access",
                    "namespace": namespace
                },
                "spec": {
                    "podSelector": {
                        "matchLabels": {"app": "postgres"}
                    },
                    "policyTypes": ["Ingress"],
                    "ingress": [{
                        "from": [
                            {"podSelector": {"matchLabels": {"app": "synthdata"}}}
                        ],
                        "ports": [
                            {"protocol": "TCP", "port": 5432}
                        ]
                    }]
                }
            }
            
            self._apply_resource(allow_db)
            
            result = DeploymentResult()
            result.success = True
            result.message = "Network policies created"
            return result
            
        except Exception as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"Network policy creation failed: {str(e)}"
            return result
    
    def create_pod_security_policy(self) -> DeploymentResult:
        """Create PodSecurityPolicy for enhanced security.
        
        Returns:
            DeploymentResult
        """
        try:
            psp = {
                "apiVersion": "policy/v1beta1",
                "kind": "PodSecurityPolicy",
                "metadata": {
                    "name": "synthdata-restricted"
                },
                "spec": {
                    "privileged": False,
                    "allowPrivilegeEscalation": False,
                    "requiredDropCapabilities": ["ALL"],
                    "volumes": [
                        "configMap",
                        "emptyDir",
                        "projected",
                        "secret",
                        "downwardAPI",
                        "persistentVolumeClaim"
                    ],
                    "hostNetwork": False,
                    "hostIPC": False,
                    "hostPID": False,
                    "runAsUser": {
                        "rule": "MustRunAsNonRoot"
                    },
                    "seLinux": {
                        "rule": "RunAsAny"
                    },
                    "fsGroup": {
                        "rule": "RunAsAny"
                    },
                    "readOnlyRootFilesystem": False
                }
            }
            
            self._apply_resource(psp)
            
            result = DeploymentResult()
            result.success = True
            result.message = "PodSecurityPolicy created"
            return result
            
        except Exception as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"PodSecurityPolicy creation failed: {str(e)}"
            return result
    
    def create_rbac_bindings(self, namespace: str, ldap_groups: Dict[str, str]) -> DeploymentResult:
        """Create RBAC bindings for LDAP groups.
        
        Args:
            namespace: Target namespace
            ldap_groups: Mapping of LDAP groups to K8s roles
            
        Returns:
            DeploymentResult
        """
        try:
            for group_dn, role in ldap_groups.items():
                # Create RoleBinding
                binding = {
                    "apiVersion": "rbac.authorization.k8s.io/v1",
                    "kind": "RoleBinding",
                    "metadata": {
                        "name": f"ldap-{role}-binding",
                        "namespace": namespace
                    },
                    "subjects": [{
                        "kind": "Group",
                        "name": group_dn,
                        "apiGroup": "rbac.authorization.k8s.io"
                    }],
                    "roleRef": {
                        "kind": "ClusterRole",
                        "name": role,
                        "apiGroup": "rbac.authorization.k8s.io"
                    }
                }
                
                self._apply_resource(binding)
            
            result = DeploymentResult()
            result.success = True
            result.message = "RBAC bindings created"
            return result
            
        except Exception as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"RBAC binding creation failed: {str(e)}"
            return result
    
    def generate_ca_certificate(self) -> Tuple[str, str]:
        """Generate a self-signed CA certificate.
        
        Returns:
            Tuple of (certificate_pem, private_key_pem)
        """
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096
        )
        
        # Generate certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Inferloop"),
            x509.NameAttribute(NameOID.COMMON_NAME, "Inferloop CA")
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=3650)
        ).add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True
        ).add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False
            ),
            critical=True
        ).sign(private_key, hashes.SHA256())
        
        # Convert to PEM
        cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode()
        key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ).decode()
        
        return cert_pem, key_pem
    
    def create_ca_secret(self, namespace: str = "cert-manager") -> DeploymentResult:
        """Create CA secret for cert-manager.
        
        Args:
            namespace: Namespace for the secret
            
        Returns:
            DeploymentResult
        """
        try:
            cert_pem, key_pem = self.generate_ca_certificate()
            
            secret = {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {
                    "name": "ca-key-pair",
                    "namespace": namespace
                },
                "type": "Opaque",
                "data": {
                    "tls.crt": base64.b64encode(cert_pem.encode()).decode(),
                    "tls.key": base64.b64encode(key_pem.encode()).decode()
                }
            }
            
            self._apply_resource(secret)
            
            result = DeploymentResult()
            result.success = True
            result.message = "CA secret created"
            return result
            
        except Exception as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"CA secret creation failed: {str(e)}"
            return result