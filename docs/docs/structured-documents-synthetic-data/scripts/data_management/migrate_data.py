#!/usr/bin/env python3
"""
Data migration script for upgrading data formats and structures.

Provides automated migration capabilities for datasets, configurations,
and generated content when system updates require format changes.
"""

import asyncio
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import yaml
import pickle
from dataclasses import dataclass

# Migration version tracking
CURRENT_VERSION = "2.0.0"
MIN_SUPPORTED_VERSION = "1.0.0"

@dataclass
class MigrationStep:
    """Individual migration step"""
    from_version: str
    to_version: str
    description: str
    migration_func: Callable
    reversible: bool = False
    backup_required: bool = True


class DataMigrator:
    """Comprehensive data migration system"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path.home() / '.structured_docs_synth'
        self.backup_dir = self.data_dir / 'backups'
        self.migration_log = self.data_dir / 'migration.log'
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Migration steps registry
        self.migration_steps: List[MigrationStep] = []
        self._register_migration_steps()
    
    def _register_migration_steps(self):
        """Register all migration steps"""
        # Example migrations from v1.0.0 to v2.0.0
        self.migration_steps.extend([
            MigrationStep(
                from_version="1.0.0",
                to_version="1.1.0",
                description="Migrate config format from JSON to YAML",
                migration_func=self._migrate_config_to_yaml,
                reversible=True,
                backup_required=True
            ),
            MigrationStep(
                from_version="1.1.0",
                to_version="1.2.0",
                description="Update dataset metadata schema",
                migration_func=self._migrate_dataset_metadata,
                reversible=False,
                backup_required=True
            ),
            MigrationStep(
                from_version="1.2.0",
                to_version="1.3.0",
                description="Migrate SQLite database schema",
                migration_func=self._migrate_database_schema,
                reversible=False,
                backup_required=True
            ),
            MigrationStep(
                from_version="1.3.0",
                to_version="2.0.0",
                description="Restructure generated content organization",
                migration_func=self._migrate_content_structure,
                reversible=False,
                backup_required=True
            )
        ])
    
    async def check_migration_needed(self) -> Dict[str, Any]:
        """
        Check if migration is needed.
        
        Returns:
            Migration status information
        """
        current_version = await self._get_current_data_version()
        
        return {
            'current_version': current_version,
            'target_version': CURRENT_VERSION,
            'migration_needed': current_version != CURRENT_VERSION,
            'migration_path': self._get_migration_path(current_version, CURRENT_VERSION),
            'backup_dir': str(self.backup_dir),
            'data_dir': str(self.data_dir)
        }
    
    async def migrate_data(self, target_version: Optional[str] = None, 
                         create_backup: bool = True,
                         dry_run: bool = False) -> Dict[str, Any]:
        """
        Perform data migration.
        
        Args:
            target_version: Target version (default: latest)
            create_backup: Whether to create backup before migration
            dry_run: Perform dry run without making changes
        
        Returns:
            Migration results
        """
        target_version = target_version or CURRENT_VERSION
        current_version = await self._get_current_data_version()
        
        print(f"=€ Starting migration from {current_version} to {target_version}")
        
        if current_version == target_version:
            print(f" Already at target version {target_version}")
            return {'status': 'no_migration_needed', 'version': current_version}
        
        # Get migration path
        migration_path = self._get_migration_path(current_version, target_version)
        
        if not migration_path:
            raise ValueError(f"No migration path from {current_version} to {target_version}")
        
        print(f"=ú Migration path: {' -> '.join([step.to_version for step in migration_path])}")
        
        if dry_run:
            print("= Performing dry run...")
            return await self._dry_run_migration(migration_path)
        
        # Create backup if requested
        backup_path = None
        if create_backup:
            backup_path = await self._create_backup(current_version)
            print(f"=Ë Backup created: {backup_path}")
        
        # Execute migration steps
        migration_results = []
        
        try:
            for step in migration_path:
                print(f"= Executing: {step.description}")
                
                step_result = await self._execute_migration_step(step)
                migration_results.append({
                    'step': step.description,
                    'from_version': step.from_version,
                    'to_version': step.to_version,
                    'success': step_result['success'],
                    'details': step_result.get('details', {})
                })
                
                if not step_result['success']:
                    raise Exception(f"Migration step failed: {step.description}")
                
                # Update version after successful step
                await self._update_data_version(step.to_version)
                print(f" Completed migration to {step.to_version}")
            
            # Log successful migration
            await self._log_migration(current_version, target_version, True, migration_results)
            
            print(f"( Migration completed successfully: {current_version} -> {target_version}")
            
            return {
                'status': 'success',
                'from_version': current_version,
                'to_version': target_version,
                'backup_path': str(backup_path) if backup_path else None,
                'steps_completed': len(migration_results),
                'migration_results': migration_results
            }
            
        except Exception as e:
            # Log failed migration
            await self._log_migration(current_version, target_version, False, migration_results, str(e))
            
            print(f"L Migration failed: {e}")
            
            # Attempt rollback if backup exists
            if backup_path:
                print(f"= Attempting rollback from backup...")
                rollback_success = await self._rollback_from_backup(backup_path)
                if rollback_success:
                    print(f" Rollback successful")
                else:
                    print(f"L Rollback failed - manual intervention required")
            
            return {
                'status': 'failed',
                'error': str(e),
                'from_version': current_version,
                'partial_results': migration_results,
                'backup_path': str(backup_path) if backup_path else None
            }
    
    async def rollback_migration(self, backup_path: Path) -> bool:
        """
        Rollback migration from backup.
        
        Args:
            backup_path: Path to backup directory
        
        Returns:
            True if rollback successful
        """
        print(f"= Rolling back from backup: {backup_path}")
        return await self._rollback_from_backup(backup_path)
    
    async def list_backups(self) -> List[Dict[str, Any]]:
        """
        List available backups.
        
        Returns:
            List of backup information
        """
        backups = []
        
        for backup_dir in self.backup_dir.glob('backup_*'):
            if backup_dir.is_dir():
                metadata_file = backup_dir / 'metadata.json'
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        backups.append({
                            'path': str(backup_dir),
                            'created_at': metadata.get('created_at'),
                            'version': metadata.get('version'),
                            'size_mb': self._get_directory_size(backup_dir) / (1024 * 1024)
                        })
                    except Exception as e:
                        print(f"   Error reading backup metadata: {e}")
        
        return sorted(backups, key=lambda x: x['created_at'], reverse=True)
    
    async def cleanup_old_backups(self, keep_count: int = 5) -> int:
        """
        Clean up old backups, keeping only the most recent ones.
        
        Args:
            keep_count: Number of backups to keep
        
        Returns:
            Number of backups removed
        """
        backups = await self.list_backups()
        
        if len(backups) <= keep_count:
            print(f"=Á Only {len(backups)} backups found, nothing to clean up")
            return 0
        
        backups_to_remove = backups[keep_count:]
        removed_count = 0
        
        for backup in backups_to_remove:
            try:
                backup_path = Path(backup['path'])
                shutil.rmtree(backup_path)
                removed_count += 1
                print(f"=Ñ  Removed backup: {backup_path.name}")
            except Exception as e:
                print(f"   Error removing backup {backup['path']}: {e}")
        
        print(f">ù Cleaned up {removed_count} old backups")
        return removed_count
    
    # Migration step implementations
    
    async def _migrate_config_to_yaml(self, **kwargs) -> Dict[str, Any]:
        """Migrate configuration from JSON to YAML format"""
        config_json = self.data_dir / 'config.json'
        config_yaml = self.data_dir / 'config.yaml'
        
        if not config_json.exists():
            return {'success': True, 'details': {'message': 'No config.json found'}}
        
        try:
            # Load JSON config
            with open(config_json, 'r') as f:
                config_data = json.load(f)
            
            # Save as YAML
            with open(config_yaml, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            # Remove old JSON file
            config_json.unlink()
            
            return {
                'success': True,
                'details': {
                    'migrated_keys': len(config_data),
                    'old_file': str(config_json),
                    'new_file': str(config_yaml)
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _migrate_dataset_metadata(self, **kwargs) -> Dict[str, Any]:
        """Update dataset metadata schema"""
        datasets_dir = self.data_dir / 'datasets'
        
        if not datasets_dir.exists():
            return {'success': True, 'details': {'message': 'No datasets directory found'}}
        
        migrated_count = 0
        error_count = 0
        
        for dataset_dir in datasets_dir.iterdir():
            if dataset_dir.is_dir():
                metadata_file = dataset_dir / 'metadata.json'
                
                if metadata_file.exists():
                    try:
                        # Load existing metadata
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        # Add new fields for v1.2.0
                        metadata.update({
                            'schema_version': '1.2.0',
                            'last_updated': datetime.now().isoformat(),
                            'file_count': len(list(dataset_dir.glob('**/*'))),
                            'tags': metadata.get('tags', []),
                            'license': metadata.get('license', 'unknown')
                        })
                        
                        # Save updated metadata
                        with open(metadata_file, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        
                        migrated_count += 1
                        
                    except Exception as e:
                        print(f"   Error migrating {dataset_dir.name}: {e}")
                        error_count += 1
        
        return {
            'success': error_count == 0,
            'details': {
                'migrated_count': migrated_count,
                'error_count': error_count
            }
        }
    
    async def _migrate_database_schema(self, **kwargs) -> Dict[str, Any]:
        """Migrate SQLite database schema"""
        db_file = self.data_dir / 'app.db'
        
        if not db_file.exists():
            return {'success': True, 'details': {'message': 'No database found'}}
        
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Example schema updates for v1.3.0
            schema_updates = [
                "ALTER TABLE documents ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "ALTER TABLE documents ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at)",
                "CREATE TABLE IF NOT EXISTS migration_history (id INTEGER PRIMARY KEY, version TEXT, applied_at TIMESTAMP)"
            ]
            
            executed_updates = 0
            
            for update in schema_updates:
                try:
                    cursor.execute(update)
                    executed_updates += 1
                except sqlite3.Error as e:
                    if "duplicate column name" not in str(e).lower():
                        print(f"   Schema update warning: {e}")
            
            # Record migration in history
            cursor.execute(
                "INSERT INTO migration_history (version, applied_at) VALUES (?, ?)",
                ('1.3.0', datetime.now().isoformat())
            )
            
            conn.commit()
            conn.close()
            
            return {
                'success': True,
                'details': {
                    'executed_updates': executed_updates,
                    'total_updates': len(schema_updates)
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _migrate_content_structure(self, **kwargs) -> Dict[str, Any]:
        """Restructure generated content organization"""
        old_content_dir = self.data_dir / 'generated'
        new_content_dir = self.data_dir / 'content'
        
        if not old_content_dir.exists():
            return {'success': True, 'details': {'message': 'No generated content found'}}
        
        try:
            # Create new directory structure
            new_content_dir.mkdir(exist_ok=True)
            (new_content_dir / 'documents').mkdir(exist_ok=True)
            (new_content_dir / 'datasets').mkdir(exist_ok=True)
            (new_content_dir / 'templates').mkdir(exist_ok=True)
            
            # Move files to new structure
            moved_files = 0
            
            for file_path in old_content_dir.rglob('*'):
                if file_path.is_file():
                    # Determine new location based on file type
                    if file_path.suffix in ['.pdf', '.docx', '.html']:
                        new_path = new_content_dir / 'documents' / file_path.name
                    elif file_path.suffix in ['.json', '.csv', '.xml']:
                        new_path = new_content_dir / 'datasets' / file_path.name
                    elif file_path.suffix in ['.jinja2', '.template']:
                        new_path = new_content_dir / 'templates' / file_path.name
                    else:
                        new_path = new_content_dir / 'documents' / file_path.name
                    
                    # Ensure unique filename
                    counter = 1
                    original_new_path = new_path
                    while new_path.exists():
                        new_path = original_new_path.with_stem(
                            f"{original_new_path.stem}_{counter}"
                        )
                        counter += 1
                    
                    shutil.move(str(file_path), str(new_path))
                    moved_files += 1
            
            # Remove old directory if empty
            try:
                shutil.rmtree(old_content_dir)
            except OSError:
                pass  # Directory not empty or other issue
            
            return {
                'success': True,
                'details': {
                    'moved_files': moved_files,
                    'old_dir': str(old_content_dir),
                    'new_dir': str(new_content_dir)
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # Helper methods
    
    async def _get_current_data_version(self) -> str:
        """Get current data version"""
        version_file = self.data_dir / 'version.txt'
        
        if version_file.exists():
            try:
                with open(version_file, 'r') as f:
                    return f.read().strip()
            except Exception:
                pass
        
        return MIN_SUPPORTED_VERSION  # Default to minimum version
    
    async def _update_data_version(self, version: str):
        """Update data version file"""
        version_file = self.data_dir / 'version.txt'
        
        with open(version_file, 'w') as f:
            f.write(version)
    
    def _get_migration_path(self, from_version: str, to_version: str) -> List[MigrationStep]:
        """Get migration path between versions"""
        path = []
        current_version = from_version
        
        while current_version != to_version:
            next_step = None
            
            for step in self.migration_steps:
                if step.from_version == current_version:
                    next_step = step
                    break
            
            if not next_step:
                return []  # No path found
            
            path.append(next_step)
            current_version = next_step.to_version
        
        return path
    
    async def _create_backup(self, version: str) -> Path:
        """Create backup of current data"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"backup_{version}_{timestamp}"
        backup_path = self.backup_dir / backup_name
        
        # Copy data directory
        shutil.copytree(
            self.data_dir, 
            backup_path,
            ignore=shutil.ignore_patterns('backups', '*.tmp', '__pycache__')
        )
        
        # Create metadata
        metadata = {
            'version': version,
            'created_at': datetime.now().isoformat(),
            'backup_type': 'pre_migration',
            'data_dir': str(self.data_dir)
        }
        
        with open(backup_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return backup_path
    
    async def _rollback_from_backup(self, backup_path: Path) -> bool:
        """Rollback from backup"""
        try:
            # Remove current data (except backups)
            for item in self.data_dir.iterdir():
                if item.name != 'backups':
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
            
            # Restore from backup
            for item in backup_path.iterdir():
                if item.name not in ['metadata.json', 'backups']:
                    if item.is_dir():
                        shutil.copytree(item, self.data_dir / item.name)
                    else:
                        shutil.copy2(item, self.data_dir / item.name)
            
            return True
            
        except Exception as e:
            print(f"L Rollback failed: {e}")
            return False
    
    async def _execute_migration_step(self, step: MigrationStep) -> Dict[str, Any]:
        """Execute single migration step"""
        try:
            result = await step.migration_func()
            return result
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _dry_run_migration(self, migration_path: List[MigrationStep]) -> Dict[str, Any]:
        """Perform dry run of migration"""
        print("= Dry run results:")
        
        for i, step in enumerate(migration_path, 1):
            print(f"  {i}. {step.description}")
            print(f"     {step.from_version} -> {step.to_version}")
            print(f"     Backup required: {step.backup_required}")
            print(f"     Reversible: {step.reversible}")
        
        return {
            'status': 'dry_run_complete',
            'steps_to_execute': len(migration_path),
            'estimated_duration': len(migration_path) * 30,  # seconds
            'backup_required': any(step.backup_required for step in migration_path)
        }
    
    async def _log_migration(self, from_version: str, to_version: str, 
                           success: bool, results: List[Dict], 
                           error: Optional[str] = None):
        """Log migration attempt"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'from_version': from_version,
            'to_version': to_version,
            'success': success,
            'steps_completed': len(results),
            'error': error,
            'results': results
        }
        
        # Append to log file
        with open(self.migration_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _get_directory_size(self, directory: Path) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size


async def main():
    """
    Main migration script function.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Migrate structured document synthesis data'
    )
    parser.add_argument(
        '--check', 
        action='store_true',
        help='Check if migration is needed'
    )
    parser.add_argument(
        '--migrate', 
        action='store_true',
        help='Perform migration'
    )
    parser.add_argument(
        '--target-version', 
        help='Target version for migration'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Perform dry run without making changes'
    )
    parser.add_argument(
        '--no-backup', 
        action='store_true',
        help='Skip backup creation'
    )
    parser.add_argument(
        '--list-backups', 
        action='store_true',
        help='List available backups'
    )
    parser.add_argument(
        '--cleanup-backups', 
        type=int,
        metavar='KEEP_COUNT',
        help='Clean up old backups, keeping specified number'
    )
    parser.add_argument(
        '--rollback', 
        type=Path,
        metavar='BACKUP_PATH',
        help='Rollback from specified backup'
    )
    parser.add_argument(
        '--data-dir', 
        type=Path,
        help='Custom data directory'
    )
    
    args = parser.parse_args()
    
    migrator = DataMigrator(data_dir=args.data_dir)
    
    if args.check:
        status = await migrator.check_migration_needed()
        print("= Migration Status:")
        print(f"Current version: {status['current_version']}")
        print(f"Target version: {status['target_version']}")
        print(f"Migration needed: {status['migration_needed']}")
        if status['migration_needed']:
            path_versions = [step.to_version for step in status['migration_path']]
            print(f"Migration path: {' -> '.join(path_versions)}")
    
    elif args.list_backups:
        backups = await migrator.list_backups()
        print(f"=Ë Available Backups ({len(backups)}):")
        for backup in backups:
            print(f"  {Path(backup['path']).name}")
            print(f"    Version: {backup['version']}")
            print(f"    Created: {backup['created_at']}")
            print(f"    Size: {backup['size_mb']:.1f} MB")
    
    elif args.cleanup_backups is not None:
        removed = await migrator.cleanup_old_backups(args.cleanup_backups)
        print(f">ù Removed {removed} old backups")
    
    elif args.rollback:
        success = await migrator.rollback_migration(args.rollback)
        if success:
            print(" Rollback completed successfully")
        else:
            print("L Rollback failed")
    
    elif args.migrate:
        result = await migrator.migrate_data(
            target_version=args.target_version,
            create_backup=not args.no_backup,
            dry_run=args.dry_run
        )
        
        if result['status'] == 'success':
            print(" Migration completed successfully")
        elif result['status'] == 'no_migration_needed':
            print("9 No migration needed")
        else:
            print(f"L Migration failed: {result.get('error', 'Unknown error')}")
            return 1
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(asyncio.run(main()))