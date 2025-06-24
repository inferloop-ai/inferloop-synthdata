"""Backup and restore functionality using Velero."""

import subprocess
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import tempfile
import time
from datetime import datetime, timedelta

from ..base import ResourceConfig, DeploymentResult
from ..exceptions import DeploymentError


class BackupManager:
    """Manage backup and restore operations using Velero."""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        self.kubeconfig_path = kubeconfig_path or str(Path.home() / ".kube" / "config")
        self.kubectl_base_cmd = ["kubectl", f"--kubeconfig={self.kubeconfig_path}"]
        self.velero_cmd = ["velero", f"--kubeconfig={self.kubeconfig_path}"]
    
    def install_velero(self, config: ResourceConfig) -> DeploymentResult:
        """Install Velero for backup and restore.
        
        Args:
            config: Velero configuration
            
        Returns:
            DeploymentResult
        """
        try:
            provider = config.metadata.get("provider", "aws")
            bucket = config.metadata.get("bucket", "velero-backups")
            region = config.metadata.get("region", "us-east-1")
            
            # Create backup location configuration
            if provider == "aws":
                result = self._install_velero_aws(bucket, region, config)
            elif provider == "gcp":
                result = self._install_velero_gcp(bucket, config)
            elif provider == "azure":
                result = self._install_velero_azure(bucket, config)
            elif provider == "minio":
                result = self._install_velero_minio(bucket, config)
            else:
                raise ValueError(f"Unsupported backup provider: {provider}")
            
            if result.success:
                # Wait for Velero to be ready
                subprocess.run([
                    "kubectl", "wait", "--for=condition=Available",
                    "--timeout=300s", "deployment/velero",
                    "-n", "velero"
                ], check=True)
                
                # Create default backup schedule
                self._create_backup_schedule(config)
            
            return result
            
        except Exception as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"Velero installation failed: {str(e)}"
            return result
    
    def _install_velero_aws(self, bucket: str, region: str, config: ResourceConfig) -> DeploymentResult:
        """Install Velero with AWS S3 backend."""
        cmd = [
            "velero", "install",
            "--provider", "aws",
            "--plugins", "velero/velero-plugin-for-aws:v1.8.0",
            "--bucket", bucket,
            "--backup-location-config", f"region={region}",
            "--snapshot-location-config", f"region={region}",
            "--secret-file", config.metadata.get("credentials_file", "/tmp/credentials-velero")
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            result = DeploymentResult()
            result.success = True
            result.message = "Velero installed with AWS S3 backend"
            result.metadata = {
                "provider": "aws",
                "bucket": bucket,
                "region": region
            }
            return result
            
        except subprocess.CalledProcessError as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"Velero AWS installation failed: {str(e)}"
            return result
    
    def _install_velero_minio(self, bucket: str, config: ResourceConfig) -> DeploymentResult:
        """Install Velero with MinIO backend."""
        # Create MinIO credentials secret
        credentials = f"""[default]
aws_access_key_id = {config.metadata.get('access_key', 'minio')}
aws_secret_access_key = {config.metadata.get('secret_key', 'minio123')}
"""
        
        creds_file = "/tmp/credentials-velero"
        with open(creds_file, "w") as f:
            f.write(credentials)
        
        cmd = [
            "velero", "install",
            "--provider", "aws",
            "--plugins", "velero/velero-plugin-for-aws:v1.8.0",
            "--bucket", bucket,
            "--backup-location-config",
            f"region=minio,s3ForcePathStyle=true,s3Url={config.metadata.get('endpoint', 'http://minio:9000')}",
            "--secret-file", creds_file
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            result = DeploymentResult()
            result.success = True
            result.message = "Velero installed with MinIO backend"
            result.metadata = {
                "provider": "minio",
                "bucket": bucket,
                "endpoint": config.metadata.get('endpoint', 'http://minio:9000')
            }
            return result
            
        except subprocess.CalledProcessError as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"Velero MinIO installation failed: {str(e)}"
            return result
    
    def _install_velero_gcp(self, bucket: str, config: ResourceConfig) -> DeploymentResult:
        """Install Velero with GCP backend."""
        cmd = [
            "velero", "install",
            "--provider", "gcp",
            "--plugins", "velero/velero-plugin-for-gcp:v1.8.0",
            "--bucket", bucket,
            "--secret-file", config.metadata.get("credentials_file", "/tmp/credentials-velero")
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            result = DeploymentResult()
            result.success = True
            result.message = "Velero installed with GCP backend"
            result.metadata = {
                "provider": "gcp",
                "bucket": bucket
            }
            return result
            
        except subprocess.CalledProcessError as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"Velero GCP installation failed: {str(e)}"
            return result
    
    def _install_velero_azure(self, bucket: str, config: ResourceConfig) -> DeploymentResult:
        """Install Velero with Azure backend."""
        cmd = [
            "velero", "install",
            "--provider", "azure",
            "--plugins", "velero/velero-plugin-for-microsoft-azure:v1.8.0",
            "--bucket", bucket,
            "--backup-location-config",
            f"resourceGroup={config.metadata.get('resource_group', 'velero-rg')},"
            f"storageAccount={config.metadata.get('storage_account', 'velerostorage')}",
            "--snapshot-location-config",
            f"apiTimeout={config.metadata.get('api_timeout', '10m')}",
            "--secret-file", config.metadata.get("credentials_file", "/tmp/credentials-velero")
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            result = DeploymentResult()
            result.success = True
            result.message = "Velero installed with Azure backend"
            result.metadata = {
                "provider": "azure",
                "bucket": bucket,
                "resource_group": config.metadata.get('resource_group', 'velero-rg')
            }
            return result
            
        except subprocess.CalledProcessError as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"Velero Azure installation failed: {str(e)}"
            return result
    
    def _create_backup_schedule(self, config: ResourceConfig):
        """Create a default backup schedule."""
        schedule = {
            "apiVersion": "velero.io/v1",
            "kind": "Schedule",
            "metadata": {
                "name": "synthdata-daily-backup",
                "namespace": "velero"
            },
            "spec": {
                "schedule": config.metadata.get("schedule", "0 2 * * *"),  # Daily at 2 AM
                "template": {
                    "includedNamespaces": config.metadata.get("namespaces", ["synthdata"]),
                    "storageLocation": "default",
                    "ttl": config.metadata.get("ttl", "720h0m0s"),  # 30 days
                    "includedResources": [
                        "persistentvolumes",
                        "persistentvolumeclaims",
                        "secrets",
                        "configmaps",
                        "deployments",
                        "services",
                        "statefulsets"
                    ]
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(schedule, f)
            temp_path = f.name
        
        try:
            subprocess.run(
                self.kubectl_base_cmd + ["apply", "-f", temp_path],
                check=True
            )
        finally:
            Path(temp_path).unlink()
    
    def create_backup(
        self,
        backup_name: str,
        namespaces: Optional[List[str]] = None,
        labels: Optional[Dict[str, str]] = None,
        include_cluster_resources: bool = True,
        wait: bool = True
    ) -> DeploymentResult:
        """Create a backup.
        
        Args:
            backup_name: Name of the backup
            namespaces: List of namespaces to backup
            labels: Label selectors for resources
            include_cluster_resources: Include cluster-scoped resources
            wait: Wait for backup to complete
            
        Returns:
            DeploymentResult
        """
        try:
            cmd = self.velero_cmd + ["backup", "create", backup_name]
            
            if namespaces:
                cmd.extend(["--include-namespaces", ",".join(namespaces)])
            
            if labels:
                label_selectors = [f"{k}={v}" for k, v in labels.items()]
                cmd.extend(["--selector", ",".join(label_selectors)])
            
            if not include_cluster_resources:
                cmd.append("--include-cluster-resources=false")
            
            if wait:
                cmd.append("--wait")
            
            subprocess.run(cmd, check=True)
            
            result = DeploymentResult()
            result.success = True
            result.message = f"Backup '{backup_name}' created successfully"
            result.resource_id = backup_name
            result.metadata = {
                "backup_name": backup_name,
                "namespaces": namespaces or [],
                "labels": labels or {}
            }
            
            return result
            
        except subprocess.CalledProcessError as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"Backup creation failed: {str(e)}"
            return result
    
    def restore_backup(
        self,
        backup_name: str,
        restore_name: Optional[str] = None,
        namespaces: Optional[List[str]] = None,
        wait: bool = True
    ) -> DeploymentResult:
        """Restore from a backup.
        
        Args:
            backup_name: Name of the backup to restore
            restore_name: Name for the restore operation
            namespaces: Namespaces to restore
            wait: Wait for restore to complete
            
        Returns:
            DeploymentResult
        """
        try:
            if not restore_name:
                restore_name = f"{backup_name}-restore-{int(time.time())}"
            
            cmd = self.velero_cmd + ["restore", "create", restore_name, "--from-backup", backup_name]
            
            if namespaces:
                cmd.extend(["--include-namespaces", ",".join(namespaces)])
            
            if wait:
                cmd.append("--wait")
            
            subprocess.run(cmd, check=True)
            
            result = DeploymentResult()
            result.success = True
            result.message = f"Restore '{restore_name}' completed successfully"
            result.resource_id = restore_name
            result.metadata = {
                "restore_name": restore_name,
                "backup_name": backup_name,
                "namespaces": namespaces or []
            }
            
            return result
            
        except subprocess.CalledProcessError as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"Restore failed: {str(e)}"
            return result
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all backups.
        
        Returns:
            List of backup information
        """
        try:
            result = subprocess.run(
                self.velero_cmd + ["backup", "get", "-o", "json"],
                capture_output=True,
                text=True,
                check=True
            )
            
            backups_data = json.loads(result.stdout)
            backups = []
            
            for item in backups_data.get("items", []):
                metadata = item["metadata"]
                status = item.get("status", {})
                
                backups.append({
                    "name": metadata["name"],
                    "created": metadata["creationTimestamp"],
                    "phase": status.get("phase", "Unknown"),
                    "size": status.get("progress", {}).get("totalItems", 0),
                    "expiration": status.get("expiration"),
                    "errors": status.get("errors", 0)
                })
            
            return backups
            
        except Exception:
            return []
    
    def get_backup_status(self, backup_name: str) -> Dict[str, Any]:
        """Get detailed backup status.
        
        Args:
            backup_name: Name of the backup
            
        Returns:
            Backup status information
        """
        try:
            result = subprocess.run(
                self.velero_cmd + ["backup", "describe", backup_name, "-o", "json"],
                capture_output=True,
                text=True,
                check=True
            )
            
            backup_data = json.loads(result.stdout)
            status = backup_data.get("status", {})
            
            return {
                "name": backup_data["metadata"]["name"],
                "phase": status.get("phase", "Unknown"),
                "start_time": status.get("startTimestamp"),
                "completion_time": status.get("completionTimestamp"),
                "total_items": status.get("progress", {}).get("totalItems", 0),
                "items_backed_up": status.get("progress", {}).get("itemsBackedUp", 0),
                "size": status.get("progress", {}).get("totalBytes", 0),
                "errors": status.get("errors", 0),
                "warnings": status.get("warnings", 0),
                "included_namespaces": backup_data["spec"].get("includedNamespaces", []),
                "storage_location": backup_data["spec"].get("storageLocation", "default")
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def delete_backup(self, backup_name: str) -> bool:
        """Delete a backup.
        
        Args:
            backup_name: Name of the backup to delete
            
        Returns:
            True if successful
        """
        try:
            subprocess.run(
                self.velero_cmd + ["backup", "delete", backup_name, "--confirm"],
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False
    
    def create_backup_location(
        self,
        name: str,
        provider: str,
        bucket: str,
        config: Dict[str, str]
    ) -> DeploymentResult:
        """Create a backup storage location.
        
        Args:
            name: Name of the backup location
            provider: Storage provider (aws, gcp, azure, etc.)
            bucket: Storage bucket name
            config: Provider-specific configuration
            
        Returns:
            DeploymentResult
        """
        try:
            bsl = {
                "apiVersion": "velero.io/v1",
                "kind": "BackupStorageLocation",
                "metadata": {
                    "name": name,
                    "namespace": "velero"
                },
                "spec": {
                    "provider": provider,
                    "objectStorage": {
                        "bucket": bucket
                    },
                    "config": config
                }
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(bsl, f)
                temp_path = f.name
            
            subprocess.run(
                self.kubectl_base_cmd + ["apply", "-f", temp_path],
                check=True
            )
            
            Path(temp_path).unlink()
            
            result = DeploymentResult()
            result.success = True
            result.message = f"Backup storage location '{name}' created"
            result.metadata = {
                "name": name,
                "provider": provider,
                "bucket": bucket
            }
            
            return result
            
        except Exception as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"Backup storage location creation failed: {str(e)}"
            return result
    
    def create_volume_snapshot_location(
        self,
        name: str,
        provider: str,
        config: Dict[str, str]
    ) -> DeploymentResult:
        """Create a volume snapshot location.
        
        Args:
            name: Name of the snapshot location
            provider: Storage provider
            config: Provider-specific configuration
            
        Returns:
            DeploymentResult
        """
        try:
            vsl = {
                "apiVersion": "velero.io/v1",
                "kind": "VolumeSnapshotLocation",
                "metadata": {
                    "name": name,
                    "namespace": "velero"
                },
                "spec": {
                    "provider": provider,
                    "config": config
                }
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(vsl, f)
                temp_path = f.name
            
            subprocess.run(
                self.kubectl_base_cmd + ["apply", "-f", temp_path],
                check=True
            )
            
            Path(temp_path).unlink()
            
            result = DeploymentResult()
            result.success = True
            result.message = f"Volume snapshot location '{name}' created"
            return result
            
        except Exception as e:
            result = DeploymentResult()
            result.success = False
            result.message = f"Volume snapshot location creation failed: {str(e)}"
            return result
    
    def validate_installation(self) -> Tuple[bool, List[str]]:
        """Validate Velero installation.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if Velero CLI is available
        try:
            subprocess.run(["velero", "version", "--client-only"], capture_output=True, check=True)
        except subprocess.CalledProcessError:
            issues.append("Velero CLI not available")
        
        # Check if Velero is deployed
        try:
            subprocess.run(
                self.kubectl_base_cmd + ["get", "deployment", "velero", "-n", "velero"],
                capture_output=True,
                check=True
            )
        except subprocess.CalledProcessError:
            issues.append("Velero not deployed in cluster")
        
        # Check backup storage locations
        try:
            result = subprocess.run(
                self.kubectl_base_cmd + ["get", "backupstoragelocations", "-n", "velero", "-o", "json"],
                capture_output=True,
                text=True,
                check=True
            )
            
            bsls = json.loads(result.stdout)
            if not bsls.get("items"):
                issues.append("No backup storage locations configured")
            else:
                # Check if any BSL is available
                available = False
                for bsl in bsls["items"]:
                    if bsl.get("status", {}).get("phase") == "Available":
                        available = True
                        break
                
                if not available:
                    issues.append("No backup storage locations are available")
                    
        except Exception:
            issues.append("Cannot check backup storage locations")
        
        return len(issues) == 0, issues
    
    def get_restore_status(self, restore_name: str) -> Dict[str, Any]:
        """Get detailed restore status.
        
        Args:
            restore_name: Name of the restore operation
            
        Returns:
            Restore status information
        """
        try:
            result = subprocess.run(
                self.velero_cmd + ["restore", "describe", restore_name, "-o", "json"],
                capture_output=True,
                text=True,
                check=True
            )
            
            restore_data = json.loads(result.stdout)
            status = restore_data.get("status", {})
            
            return {
                "name": restore_data["metadata"]["name"],
                "phase": status.get("phase", "Unknown"),
                "start_time": status.get("startTimestamp"),
                "completion_time": status.get("completionTimestamp"),
                "total_items": status.get("progress", {}).get("totalItems", 0),
                "items_restored": status.get("progress", {}).get("itemsRestored", 0),
                "errors": status.get("errors", 0),
                "warnings": status.get("warnings", 0),
                "backup_name": restore_data["spec"].get("backupName"),
                "included_namespaces": restore_data["spec"].get("includedNamespaces", [])
            }
            
        except Exception as e:
            return {"error": str(e)}