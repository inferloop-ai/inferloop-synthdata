#!/usr/bin/env python3
"""
Data backup script for system data protection.

Provides automated backup capabilities for datasets, configurations,
generated content, and system state with compression and verification.
"""

import asyncio
import json
import shutil
import tarfile
import gzip
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import sqlite3
import boto3
from botocore.exceptions import ClientError

# Backup configuration
BACKUP_CONFIG = {
    'default_retention_days': 30,
    'compression_level': 6,
    'verification_enabled': True,
    'incremental_enabled': True,
    'encryption_enabled': False,
    'cloud_backup_enabled': False
}

DEFAULT_DATA_DIR = Path.home() / '.structured_docs_synth'
DEFAULT_BACKUP_DIR = DEFAULT_DATA_DIR / 'backups'


class DataBackup:
    """Comprehensive data backup system"""
    
    def __init__(self, data_dir: Optional[Path] = None, 
                 backup_dir: Optional[Path] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.backup_dir = backup_dir or DEFAULT_BACKUP_DIR
        self.config = {**BACKUP_CONFIG, **(config or {})}
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # S3 client for cloud backups
        self.s3_client = None
        if self.config['cloud_backup_enabled']:
            try:
                self.s3_client = boto3.client('s3')
            except Exception as e:
                print(f"   Warning: Could not initialize S3 client: {e}")
    
    async def create_full_backup(self, backup_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create full backup of all data.
        
        Args:
            backup_name: Custom backup name
        
        Returns:
            Backup results
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = backup_name or f"full_backup_{timestamp}"
        
        print(f"=æ Creating full backup: {backup_name}")
        
        try:
            backup_path = self.backup_dir / backup_name
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Backup components
            results = {
                'backup_name': backup_name,
                'backup_path': str(backup_path),
                'timestamp': timestamp,
                'type': 'full',
                'components': {}
            }
            
            # Backup datasets
            datasets_result = await self._backup_datasets(backup_path)
            results['components']['datasets'] = datasets_result
            
            # Backup configurations
            config_result = await self._backup_configurations(backup_path)
            results['components']['configurations'] = config_result
            
            # Backup generated content
            content_result = await self._backup_generated_content(backup_path)
            results['components']['generated_content'] = content_result
            
            # Backup database
            database_result = await self._backup_database(backup_path)
            results['components']['database'] = database_result
            
            # Backup templates
            templates_result = await self._backup_templates(backup_path)
            results['components']['templates'] = templates_result
            
            # Create backup metadata
            await self._create_backup_metadata(backup_path, results)
            
            # Compress backup if enabled
            if self.config.get('compression_enabled', True):
                compressed_path = await self._compress_backup(backup_path)
                results['compressed_path'] = str(compressed_path)
                results['compressed_size_mb'] = compressed_path.stat().st_size / (1024 * 1024)
                
                # Remove uncompressed backup
                shutil.rmtree(backup_path)
                backup_path = compressed_path
            
            # Verify backup if enabled
            if self.config['verification_enabled']:
                verification_result = await self._verify_backup(backup_path)
                results['verification'] = verification_result
            
            # Upload to cloud if enabled
            if self.config['cloud_backup_enabled'] and self.s3_client:
                cloud_result = await self._upload_to_cloud(backup_path)
                results['cloud_upload'] = cloud_result
            
            # Calculate total size
            results['total_size_mb'] = self._get_path_size(backup_path) / (1024 * 1024)
            
            print(f" Full backup completed: {backup_name}")
            print(f"=Á Size: {results['total_size_mb']:.1f} MB")
            
            return {'success': True, **results}
            
        except Exception as e:
            print(f"L Full backup failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'backup_name': backup_name
            }
    
    async def create_incremental_backup(self, last_backup_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Create incremental backup (only changed files).
        
        Args:
            last_backup_path: Path to last backup for comparison
        
        Returns:
            Backup results
        """
        if not self.config['incremental_enabled']:
            return await self.create_full_backup()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"incremental_backup_{timestamp}"
        
        print(f"=æ Creating incremental backup: {backup_name}")
        
        try:
            # Find last backup if not provided
            if not last_backup_path:
                last_backup_path = await self._find_latest_backup()
            
            if not last_backup_path:
                print("= No previous backup found, creating full backup")
                return await self.create_full_backup(backup_name)
            
            print(f"= Comparing with: {last_backup_path.name}")
            
            backup_path = self.backup_dir / backup_name
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Find changed files
            changed_files = await self._find_changed_files(last_backup_path)
            
            if not changed_files:
                print("9 No changes detected since last backup")
                shutil.rmtree(backup_path)
                return {
                    'success': True,
                    'backup_name': backup_name,
                    'type': 'incremental',
                    'changes_found': False,
                    'message': 'No changes detected'
                }
            
            # Copy changed files
            copied_files = 0
            for file_path in changed_files:
                relative_path = file_path.relative_to(self.data_dir)
                dest_path = backup_path / relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                shutil.copy2(file_path, dest_path)
                copied_files += 1
            
            # Create metadata
            metadata = {
                'backup_name': backup_name,
                'type': 'incremental',
                'timestamp': timestamp,
                'last_backup': str(last_backup_path),
                'changed_files': [str(f.relative_to(self.data_dir)) for f in changed_files],
                'copied_files': copied_files
            }
            
            await self._create_backup_metadata(backup_path, metadata)
            
            # Compress if enabled
            if self.config.get('compression_enabled', True):
                compressed_path = await self._compress_backup(backup_path)
                shutil.rmtree(backup_path)
                backup_path = compressed_path
            
            total_size_mb = self._get_path_size(backup_path) / (1024 * 1024)
            
            print(f" Incremental backup completed: {backup_name}")
            print(f"=Á Files: {copied_files}, Size: {total_size_mb:.1f} MB")
            
            return {
                'success': True,
                'backup_name': backup_name,
                'backup_path': str(backup_path),
                'type': 'incremental',
                'changes_found': True,
                'copied_files': copied_files,
                'total_size_mb': total_size_mb
            }
            
        except Exception as e:
            print(f"L Incremental backup failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'backup_name': backup_name
            }
    
    async def restore_backup(self, backup_path: Path, 
                           restore_components: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Restore from backup.
        
        Args:
            backup_path: Path to backup
            restore_components: Components to restore (default: all)
        
        Returns:
            Restoration results
        """
        print(f"= Restoring from backup: {backup_path.name}")
        
        try:
            # Check if backup is compressed
            if backup_path.suffix in ['.tar.gz', '.tgz']:
                # Extract to temporary directory
                temp_dir = self.backup_dir / f"temp_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                backup_path = await self._extract_backup(backup_path, temp_dir)
            
            # Load backup metadata
            metadata_file = backup_path / 'backup_metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {'type': 'unknown'}
            
            print(f"=Ë Backup type: {metadata.get('type', 'unknown')}")
            print(f"=Å Created: {metadata.get('timestamp', 'unknown')}")
            
            # Create data backup before restoration
            print("=¾ Creating safety backup before restoration...")
            safety_backup = await self.create_full_backup(f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            restore_results = {
                'backup_path': str(backup_path),
                'safety_backup': safety_backup,
                'components_restored': {}
            }
            
            # Restore components
            all_components = ['datasets', 'configurations', 'generated_content', 'database', 'templates']
            components_to_restore = restore_components or all_components
            
            for component in components_to_restore:
                component_path = backup_path / component
                if component_path.exists():
                    result = await self._restore_component(component, component_path)
                    restore_results['components_restored'][component] = result
                else:
                    print(f"   Component not found in backup: {component}")
            
            # Clean up temporary directory if created
            if backup_path.parent.name.startswith('temp_restore_'):
                shutil.rmtree(backup_path.parent)
            
            print(f" Restoration completed from {backup_path.name}")
            
            return {'success': True, **restore_results}
            
        except Exception as e:
            print(f"L Restoration failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'backup_path': str(backup_path)
            }
    
    async def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all available backups.
        
        Returns:
            List of backup information
        """
        backups = []
        
        for backup_item in self.backup_dir.iterdir():
            if backup_item.is_dir() or backup_item.suffix in ['.tar.gz', '.tgz']:
                backup_info = await self._get_backup_info(backup_item)
                if backup_info:
                    backups.append(backup_info)
        
        return sorted(backups, key=lambda x: x['timestamp'], reverse=True)
    
    async def cleanup_old_backups(self, retention_days: Optional[int] = None) -> Dict[str, Any]:
        """
        Clean up old backups based on retention policy.
        
        Args:
            retention_days: Number of days to retain (default: from config)
        
        Returns:
            Cleanup results
        """
        retention_days = retention_days or self.config['default_retention_days']
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        print(f">ù Cleaning up backups older than {retention_days} days...")
        
        backups = await self.list_backups()
        old_backups = [
            backup for backup in backups 
            if datetime.fromisoformat(backup['timestamp']) < cutoff_date
        ]
        
        if not old_backups:
            print("9 No old backups found")
            return {
                'cleaned_count': 0,
                'total_backups': len(backups),
                'space_freed_mb': 0
            }
        
        cleaned_count = 0
        space_freed = 0
        
        for backup in old_backups:
            try:
                backup_path = Path(backup['path'])
                size = backup['size_mb']
                
                if backup_path.is_dir():
                    shutil.rmtree(backup_path)
                else:
                    backup_path.unlink()
                
                cleaned_count += 1
                space_freed += size
                
                print(f"=Ñ  Removed: {backup_path.name} ({size:.1f} MB)")
                
            except Exception as e:
                print(f"   Error removing {backup['path']}: {e}")
        
        print(f" Cleaned {cleaned_count} old backups, freed {space_freed:.1f} MB")
        
        return {
            'cleaned_count': cleaned_count,
            'total_backups': len(backups),
            'space_freed_mb': space_freed
        }
    
    # Component backup methods
    
    async def _backup_datasets(self, backup_path: Path) -> Dict[str, Any]:
        """Backup datasets"""
        datasets_dir = self.data_dir / 'datasets'
        backup_datasets_dir = backup_path / 'datasets'
        
        if not datasets_dir.exists():
            return {'status': 'skipped', 'reason': 'no datasets directory'}
        
        shutil.copytree(datasets_dir, backup_datasets_dir)
        
        return {
            'status': 'success',
            'files_copied': len(list(backup_datasets_dir.rglob('*'))),
            'size_mb': self._get_path_size(backup_datasets_dir) / (1024 * 1024)
        }
    
    async def _backup_configurations(self, backup_path: Path) -> Dict[str, Any]:
        """Backup configuration files"""
        config_files = [
            'config.yaml', 'config.json', 'settings.yaml', 'settings.json'
        ]
        
        backup_config_dir = backup_path / 'configurations'
        backup_config_dir.mkdir(exist_ok=True)
        
        copied_files = 0
        
        for config_file in config_files:
            config_path = self.data_dir / config_file
            if config_path.exists():
                shutil.copy2(config_path, backup_config_dir / config_file)
                copied_files += 1
        
        return {
            'status': 'success' if copied_files > 0 else 'skipped',
            'files_copied': copied_files,
            'size_mb': self._get_path_size(backup_config_dir) / (1024 * 1024)
        }
    
    async def _backup_generated_content(self, backup_path: Path) -> Dict[str, Any]:
        """Backup generated content"""
        content_dirs = ['content', 'generated', 'output']
        backup_content_dir = backup_path / 'generated_content'
        backup_content_dir.mkdir(exist_ok=True)
        
        total_files = 0
        
        for content_dir_name in content_dirs:
            content_dir = self.data_dir / content_dir_name
            if content_dir.exists():
                dest_dir = backup_content_dir / content_dir_name
                shutil.copytree(content_dir, dest_dir)
                total_files += len(list(dest_dir.rglob('*')))
        
        return {
            'status': 'success' if total_files > 0 else 'skipped',
            'files_copied': total_files,
            'size_mb': self._get_path_size(backup_content_dir) / (1024 * 1024)
        }
    
    async def _backup_database(self, backup_path: Path) -> Dict[str, Any]:
        """Backup SQLite databases"""
        db_files = list(self.data_dir.glob('*.db')) + list(self.data_dir.glob('*.sqlite'))
        
        if not db_files:
            return {'status': 'skipped', 'reason': 'no database files'}
        
        backup_db_dir = backup_path / 'database'
        backup_db_dir.mkdir(exist_ok=True)
        
        for db_file in db_files:
            # Create database backup with VACUUM
            backup_db_file = backup_db_dir / db_file.name
            
            try:
                # Use SQLite backup API for consistency
                source_conn = sqlite3.connect(db_file)
                backup_conn = sqlite3.connect(backup_db_file)
                
                source_conn.backup(backup_conn)
                
                source_conn.close()
                backup_conn.close()
                
            except Exception as e:
                print(f"   Database backup warning for {db_file.name}: {e}")
                # Fallback to file copy
                shutil.copy2(db_file, backup_db_file)
        
        return {
            'status': 'success',
            'files_copied': len(db_files),
            'size_mb': self._get_path_size(backup_db_dir) / (1024 * 1024)
        }
    
    async def _backup_templates(self, backup_path: Path) -> Dict[str, Any]:
        """Backup templates"""
        templates_dir = self.data_dir / 'templates'
        backup_templates_dir = backup_path / 'templates'
        
        if not templates_dir.exists():
            return {'status': 'skipped', 'reason': 'no templates directory'}
        
        shutil.copytree(templates_dir, backup_templates_dir)
        
        return {
            'status': 'success',
            'files_copied': len(list(backup_templates_dir.rglob('*'))),
            'size_mb': self._get_path_size(backup_templates_dir) / (1024 * 1024)
        }
    
    # Helper methods
    
    async def _create_backup_metadata(self, backup_path: Path, results: Dict[str, Any]):
        """Create backup metadata file"""
        metadata = {
            **results,
            'data_dir': str(self.data_dir),
            'backup_config': self.config,
            'created_by': 'structured_docs_synth_backup_script',
            'version': '1.0.0'
        }
        
        metadata_file = backup_path / 'backup_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    async def _compress_backup(self, backup_path: Path) -> Path:
        """Compress backup directory"""
        compressed_path = backup_path.with_suffix('.tar.gz')
        
        print(f"=æ Compressing backup...")
        
        with tarfile.open(compressed_path, 'w:gz', compresslevel=self.config['compression_level']) as tar:
            tar.add(backup_path, arcname=backup_path.name)
        
        return compressed_path
    
    async def _extract_backup(self, backup_path: Path, extract_dir: Path) -> Path:
        """Extract compressed backup"""
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        with tarfile.open(backup_path, 'r:gz') as tar:
            tar.extractall(extract_dir)
        
        # Return path to extracted backup directory
        extracted_dirs = [d for d in extract_dir.iterdir() if d.is_dir()]
        if extracted_dirs:
            return extracted_dirs[0]
        else:
            return extract_dir
    
    async def _verify_backup(self, backup_path: Path) -> Dict[str, Any]:
        """Verify backup integrity"""
        try:
            if backup_path.suffix in ['.tar.gz', '.tgz']:
                # Verify compressed archive
                with tarfile.open(backup_path, 'r:gz') as tar:
                    # Check if archive can be read
                    members = tar.getmembers()
                    return {
                        'status': 'success',
                        'files_verified': len(members),
                        'checksum': self._calculate_file_checksum(backup_path)
                    }
            else:
                # Verify directory structure
                files_count = len(list(backup_path.rglob('*')))
                return {
                    'status': 'success',
                    'files_verified': files_count,
                    'checksum': 'directory_backup'
                }
        
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def _upload_to_cloud(self, backup_path: Path) -> Dict[str, Any]:
        """Upload backup to cloud storage"""
        if not self.s3_client:
            return {'status': 'skipped', 'reason': 'S3 client not available'}
        
        try:
            bucket_name = self.config.get('s3_bucket_name')
            if not bucket_name:
                return {'status': 'skipped', 'reason': 'S3 bucket not configured'}
            
            s3_key = f"backups/{backup_path.name}"
            
            self.s3_client.upload_file(
                str(backup_path),
                bucket_name,
                s3_key
            )
            
            return {
                'status': 'success',
                'bucket': bucket_name,
                's3_key': s3_key,
                'upload_size_mb': backup_path.stat().st_size / (1024 * 1024)
            }
        
        except ClientError as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def _find_latest_backup(self) -> Optional[Path]:
        """Find the most recent backup"""
        backups = await self.list_backups()
        
        if backups:
            latest_backup = backups[0]  # Already sorted by timestamp descending
            return Path(latest_backup['path'])
        
        return None
    
    async def _find_changed_files(self, last_backup_path: Path) -> List[Path]:
        """Find files changed since last backup"""
        # Load last backup metadata
        if last_backup_path.suffix in ['.tar.gz', '.tgz']:
            # For compressed backups, we'll do a full backup
            # In a more sophisticated implementation, we'd extract and compare
            return list(self.data_dir.rglob('*'))
        
        metadata_file = last_backup_path / 'backup_metadata.json'
        if not metadata_file.exists():
            return list(self.data_dir.rglob('*'))
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        last_backup_time = datetime.fromisoformat(metadata['timestamp'])
        
        changed_files = []
        
        for file_path in self.data_dir.rglob('*'):
            if file_path.is_file() and 'backups' not in file_path.parts:
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime > last_backup_time:
                    changed_files.append(file_path)
        
        return changed_files
    
    async def _restore_component(self, component: str, component_path: Path) -> Dict[str, Any]:
        """Restore individual component"""
        try:
            target_path = self.data_dir / component
            
            # Remove existing component
            if target_path.exists():
                if target_path.is_dir():
                    shutil.rmtree(target_path)
                else:
                    target_path.unlink()
            
            # Copy from backup
            if component_path.is_dir():
                shutil.copytree(component_path, target_path)
            else:
                shutil.copy2(component_path, target_path)
            
            return {
                'status': 'success',
                'restored_to': str(target_path)
            }
        
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def _get_backup_info(self, backup_path: Path) -> Optional[Dict[str, Any]]:
        """Get backup information"""
        try:
            if backup_path.suffix in ['.tar.gz', '.tgz']:
                # Compressed backup
                return {
                    'name': backup_path.stem.replace('.tar', ''),
                    'path': str(backup_path),
                    'type': 'compressed',
                    'size_mb': backup_path.stat().st_size / (1024 * 1024),
                    'timestamp': datetime.fromtimestamp(backup_path.stat().st_mtime).isoformat()
                }
            
            elif backup_path.is_dir():
                # Directory backup
                metadata_file = backup_path / 'backup_metadata.json'
                
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    return {
                        'name': backup_path.name,
                        'path': str(backup_path),
                        'type': metadata.get('type', 'unknown'),
                        'size_mb': self._get_path_size(backup_path) / (1024 * 1024),
                        'timestamp': metadata.get('timestamp', '')
                    }
                else:
                    return {
                        'name': backup_path.name,
                        'path': str(backup_path),
                        'type': 'directory',
                        'size_mb': self._get_path_size(backup_path) / (1024 * 1024),
                        'timestamp': datetime.fromtimestamp(backup_path.stat().st_mtime).isoformat()
                    }
        
        except Exception:
            return None
    
    def _get_path_size(self, path: Path) -> int:
        """Get total size of path in bytes"""
        if path.is_file():
            return path.stat().st_size
        
        total_size = 0
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()


async def main():
    """
    Main backup script function.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Backup and restore structured document synthesis data'
    )
    parser.add_argument(
        'action',
        choices=['backup', 'restore', 'list', 'cleanup'],
        help='Action to perform'
    )
    parser.add_argument(
        '--type',
        choices=['full', 'incremental'],
        default='full',
        help='Backup type'
    )
    parser.add_argument(
        '--name',
        help='Custom backup name'
    )
    parser.add_argument(
        '--backup-path',
        type=Path,
        help='Path to backup for restoration'
    )
    parser.add_argument(
        '--components',
        nargs='+',
        help='Components to restore'
    )
    parser.add_argument(
        '--retention-days',
        type=int,
        help='Retention period for cleanup'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        help='Custom data directory'
    )
    parser.add_argument(
        '--backup-dir',
        type=Path,
        help='Custom backup directory'
    )
    parser.add_argument(
        '--no-compression',
        action='store_true',
        help='Disable compression'
    )
    parser.add_argument(
        '--no-verification',
        action='store_true',
        help='Skip backup verification'
    )
    
    args = parser.parse_args()
    
    # Configure backup system
    config = BACKUP_CONFIG.copy()
    if args.no_compression:
        config['compression_enabled'] = False
    if args.no_verification:
        config['verification_enabled'] = False
    
    backup_system = DataBackup(
        data_dir=args.data_dir,
        backup_dir=args.backup_dir,
        config=config
    )
    
    if args.action == 'backup':
        if args.type == 'full':
            result = await backup_system.create_full_backup(args.name)
        else:
            result = await backup_system.create_incremental_backup()
        
        if result['success']:
            print(f"\n Backup completed successfully")
            if 'total_size_mb' in result:
                print(f"=Á Total size: {result['total_size_mb']:.1f} MB")
        else:
            print(f"\nL Backup failed: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.action == 'restore':
        if not args.backup_path:
            print("L Backup path required for restoration")
            return 1
        
        result = await backup_system.restore_backup(args.backup_path, args.components)
        
        if result['success']:
            print(f"\n Restoration completed successfully")
        else:
            print(f"\nL Restoration failed: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.action == 'list':
        backups = await backup_system.list_backups()
        
        if not backups:
            print("9 No backups found")
            return 0
        
        print(f"=Ë Available Backups ({len(backups)}):")
        print("=" * 80)
        
        for backup in backups:
            print(f"=Æ {backup['name']}")
            print(f"   Type: {backup['type']}")
            print(f"   Size: {backup['size_mb']:.1f} MB")
            print(f"   Created: {backup['timestamp']}")
            print(f"   Path: {backup['path']}")
            print()
    
    elif args.action == 'cleanup':
        result = await backup_system.cleanup_old_backups(args.retention_days)
        
        print(f"\n=Ê Cleanup Summary:")
        print(f"Cleaned: {result['cleaned_count']} backups")
        print(f"Remaining: {result['total_backups'] - result['cleaned_count']} backups")
        print(f"Space freed: {result['space_freed_mb']:.1f} MB")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(asyncio.run(main()))