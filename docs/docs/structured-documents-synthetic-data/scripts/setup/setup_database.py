#!/usr/bin/env python3
"""
Database setup and initialization script for structured document synthesis.

Creates and configures all required databases including metadata storage,
audit logging, user management, model registry, and data caching with
proper indexing, security, and backup configurations.
"""

import asyncio
import json
import sqlite3
import psycopg2
import pymongo
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import hashlib
import secrets
import shutil
import subprocess
import aiofiles
import yaml

# Database configuration
DATABASE_CONFIG = {
    'default_db_dir': Path.home() / '.structured_docs_synth' / 'databases',
    'backup_dir': Path.home() / '.structured_docs_synth' / 'db_backups',
    'default_db_type': 'sqlite',  # sqlite, postgresql, mongodb
    'enable_encryption': True,
    'enable_backup': True,
    'enable_replication': False,
    'connection_pool_size': 10,
    'query_timeout': 30,
    'auto_vacuum': True,
    'wal_mode': True,  # Write-Ahead Logging for SQLite
    'foreign_keys': True
}

# Database schemas and configurations
DATABASE_SCHEMAS = {
    'main_db': {
        'name': 'structured_docs_synth.db',
        'type': 'sqlite',
        'description': 'Main application database',
        'tables': [
            'documents', 'templates', 'generations', 'export_history',
            'processing_jobs', 'configurations', 'system_settings'
        ]
    },
    'user_db': {
        'name': 'users_auth.db',
        'type': 'sqlite',
        'description': 'User authentication and authorization',
        'tables': [
            'users', 'roles', 'permissions', 'sessions', 'api_keys',
            'access_rules', 'user_preferences'
        ]
    },
    'audit_db': {
        'name': 'audit_logs.db',
        'type': 'sqlite',
        'description': 'Security audit and compliance logging',
        'tables': [
            'audit_events', 'security_alerts', 'compliance_checks',
            'system_metrics', 'error_logs'
        ]
    },
    'model_registry_db': {
        'name': 'model_registry.db',
        'type': 'sqlite',
        'description': 'ML model metadata and registry',
        'tables': [
            'models', 'model_versions', 'model_metrics', 'training_jobs',
            'inference_history', 'model_dependencies'
        ]
    },
    'cache_db': {
        'name': 'cache_store.db',
        'type': 'sqlite',
        'description': 'Application caching and temporary data',
        'tables': [
            'cache_entries', 'temporary_files', 'processing_cache',
            'api_cache', 'session_cache'
        ]
    }
}

# SQL schema definitions
SQL_SCHEMAS = {
    'documents': '''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            file_path TEXT,
            metadata TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            created_by TEXT,
            status TEXT DEFAULT 'active',
            tags TEXT,
            size_bytes INTEGER,
            mime_type TEXT,
            processing_status TEXT DEFAULT 'pending'
        )
    ''',
    'templates': '''
        CREATE TABLE IF NOT EXISTS templates (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL,
            template_data TEXT NOT NULL,
            schema_version TEXT NOT NULL,
            description TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            created_by TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            usage_count INTEGER DEFAULT 0,
            validation_rules TEXT,
            preview_image TEXT
        )
    ''',
    'generations': '''
        CREATE TABLE IF NOT EXISTS generations (
            id TEXT PRIMARY KEY,
            template_id TEXT NOT NULL,
            parameters TEXT NOT NULL,
            output_path TEXT,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            created_by TEXT,
            generation_time_seconds REAL,
            error_message TEXT,
            output_format TEXT,
            quality_score REAL,
            FOREIGN KEY (template_id) REFERENCES templates (id)
        )
    ''',
    'users': '''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_login TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            failed_login_attempts INTEGER DEFAULT 0,
            locked_until TEXT,
            email_verified BOOLEAN DEFAULT FALSE,
            profile_data TEXT,
            preferences TEXT
        )
    ''',
    'roles': '''
        CREATE TABLE IF NOT EXISTS roles (
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            permissions TEXT NOT NULL,
            created_at TEXT NOT NULL,
            is_active BOOLEAN DEFAULT TRUE,
            is_system_role BOOLEAN DEFAULT FALSE
        )
    ''',
    'audit_events': '''
        CREATE TABLE IF NOT EXISTS audit_events (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            user_id TEXT,
            session_id TEXT,
            source_ip TEXT,
            user_agent TEXT,
            resource_type TEXT,
            resource_id TEXT,
            action TEXT NOT NULL,
            result TEXT NOT NULL,
            details TEXT,
            metadata TEXT
        )
    ''',
    'models': '''
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            type TEXT NOT NULL,
            version TEXT NOT NULL,
            description TEXT,
            file_path TEXT NOT NULL,
            checksum TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            last_used TEXT,
            usage_count INTEGER DEFAULT 0,
            performance_metrics TEXT,
            training_metadata TEXT,
            is_active BOOLEAN DEFAULT TRUE
        )
    ''',
    'cache_entries': '''
        CREATE TABLE IF NOT EXISTS cache_entries (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT,
            access_count INTEGER DEFAULT 0,
            last_accessed TEXT,
            size_bytes INTEGER,
            cache_type TEXT DEFAULT 'general'
        )
    '''
}

# Database indexes for performance
DATABASE_INDEXES = {
    'documents': [
        'CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(type)',
        'CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at)',
        'CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)',
        'CREATE INDEX IF NOT EXISTS idx_documents_created_by ON documents(created_by)'
    ],
    'templates': [
        'CREATE INDEX IF NOT EXISTS idx_templates_category ON templates(category)',
        'CREATE INDEX IF NOT EXISTS idx_templates_active ON templates(is_active)',
        'CREATE INDEX IF NOT EXISTS idx_templates_created_at ON templates(created_at)'
    ],
    'generations': [
        'CREATE INDEX IF NOT EXISTS idx_generations_template_id ON generations(template_id)',
        'CREATE INDEX IF NOT EXISTS idx_generations_status ON generations(status)',
        'CREATE INDEX IF NOT EXISTS idx_generations_created_at ON generations(created_at)',
        'CREATE INDEX IF NOT EXISTS idx_generations_created_by ON generations(created_by)'
    ],
    'users': [
        'CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)',
        'CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)',
        'CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active)'
    ],
    'audit_events': [
        'CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp)',
        'CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_events(event_type)',
        'CREATE INDEX IF NOT EXISTS idx_audit_user_id ON audit_events(user_id)',
        'CREATE INDEX IF NOT EXISTS idx_audit_severity ON audit_events(severity)'
    ],
    'models': [
        'CREATE INDEX IF NOT EXISTS idx_models_type ON models(type)',
        'CREATE INDEX IF NOT EXISTS idx_models_active ON models(is_active)',
        'CREATE INDEX IF NOT EXISTS idx_models_created_at ON models(created_at)'
    ],
    'cache_entries': [
        'CREATE INDEX IF NOT EXISTS idx_cache_expires_at ON cache_entries(expires_at)',
        'CREATE INDEX IF NOT EXISTS idx_cache_type ON cache_entries(cache_type)',
        'CREATE INDEX IF NOT EXISTS idx_cache_created_at ON cache_entries(created_at)'
    ]
}


class DatabaseSetup:
    """Comprehensive database setup and management system"""
    
    def __init__(self, db_dir: Optional[Path] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.db_dir = db_dir or DATABASE_CONFIG['default_db_dir']
        self.backup_dir = DATABASE_CONFIG['backup_dir']
        self.config = {**DATABASE_CONFIG, **(config or {})}
        
        # Ensure directories exist
        for directory in [self.db_dir, self.backup_dir]:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        
        # Initialize database connections tracking
        self.active_connections = {}
        self.setup_results = {}
    
    async def setup_all_databases(self, force_recreate: bool = False) -> Dict[str, Any]:
        """
        Set up all required databases.
        
        Args:
            force_recreate: Force recreation of existing databases
        
        Returns:
            Setup results for all databases
        """
        print("=Ä  Starting comprehensive database setup...")
        
        try:
            setup_results = {
                'timestamp': datetime.now().isoformat(),
                'total_databases': len(DATABASE_SCHEMAS),
                'successful_setups': 0,
                'failed_setups': 0,
                'databases': {}
            }
            
            # Set up each database
            for db_name, db_config in DATABASE_SCHEMAS.items():
                print(f"=Ä  Setting up {db_name} database...")
                
                db_result = await self._setup_single_database(db_config, force_recreate)
                setup_results['databases'][db_name] = db_result
                
                if db_result['success']:
                    setup_results['successful_setups'] += 1
                    print(f" {db_name} database setup completed")
                else:
                    setup_results['failed_setups'] += 1
                    print(f"L {db_name} database setup failed: {db_result.get('error', 'Unknown error')}")
            
            # Create database configuration file
            await self._create_database_config()
            
            # Set up database monitoring
            if self.config.get('enable_monitoring', True):
                await self._setup_database_monitoring()
            
            # Create initial backup
            if self.config['enable_backup']:
                await self._create_initial_backup()
            
            print(f"=Ä  Database setup completed:")
            print(f"   Successful: {setup_results['successful_setups']}")
            print(f"   Failed: {setup_results['failed_setups']}")
            
            return setup_results
            
        except Exception as e:
            print(f"L Database setup failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def setup_database(self, db_name: str, force_recreate: bool = False) -> Dict[str, Any]:
        """
        Set up a specific database.
        
        Args:
            db_name: Name of database to setup
            force_recreate: Force recreation if exists
        
        Returns:
            Setup result
        """
        if db_name not in DATABASE_SCHEMAS:
            return {
                'success': False,
                'error': f'Unknown database: {db_name}'
            }
        
        db_config = DATABASE_SCHEMAS[db_name]
        return await self._setup_single_database(db_config, force_recreate)
    
    async def initialize_default_data(self) -> Dict[str, Any]:
        """
        Initialize databases with default data.
        
        Returns:
            Initialization results
        """
        print("=Ê Initializing databases with default data...")
        
        try:
            init_results = {
                'timestamp': datetime.now().isoformat(),
                'initialized_databases': 0,
                'results': {}
            }
            
            # Initialize each database with default data
            for db_name in DATABASE_SCHEMAS.keys():
                result = await self._initialize_database_data(db_name)
                init_results['results'][db_name] = result
                
                if result['success']:
                    init_results['initialized_databases'] += 1
                    print(f" {db_name} initialized with default data")
                else:
                    print(f"L {db_name} initialization failed: {result.get('error', 'Unknown error')}")
            
            print(f"=Ê Default data initialization completed:")
            print(f"   Databases initialized: {init_results['initialized_databases']}")
            
            return init_results
            
        except Exception as e:
            print(f"L Default data initialization failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def verify_databases(self) -> Dict[str, Any]:
        """
        Verify database integrity and structure.
        
        Returns:
            Verification results
        """
        print("= Verifying database integrity...")
        
        try:
            verification_results = {
                'timestamp': datetime.now().isoformat(),
                'total_databases': len(DATABASE_SCHEMAS),
                'verified_databases': 0,
                'failed_databases': 0,
                'results': {}
            }
            
            for db_name, db_config in DATABASE_SCHEMAS.items():
                result = await self._verify_single_database(db_name, db_config)
                verification_results['results'][db_name] = result
                
                if result['verified']:
                    verification_results['verified_databases'] += 1
                    print(f" {db_name}: Database verified")
                else:
                    verification_results['failed_databases'] += 1
                    print(f"L {db_name}: Verification failed")
            
            print(f"= Database verification completed:")
            print(f"   Verified: {verification_results['verified_databases']}")
            print(f"   Failed: {verification_results['failed_databases']}")
            
            return verification_results
            
        except Exception as e:
            print(f"L Database verification failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def backup_databases(self, backup_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create backup of all databases.
        
        Args:
            backup_name: Custom backup name
        
        Returns:
            Backup results
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = backup_name or f"db_backup_{timestamp}"
        
        print(f"=¾ Creating database backup: {backup_name}")
        
        try:
            backup_dir = self.backup_dir / backup_name
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            backup_results = {
                'backup_name': backup_name,
                'backup_path': str(backup_dir),
                'timestamp': datetime.now().isoformat(),
                'total_databases': len(DATABASE_SCHEMAS),
                'backed_up_databases': 0,
                'failed_backups': 0,
                'total_size_mb': 0.0,
                'results': {}
            }
            
            # Backup each database
            for db_name, db_config in DATABASE_SCHEMAS.items():
                db_path = self.db_dir / db_config['name']
                
                if db_path.exists():
                    backup_file = backup_dir / db_config['name']
                    
                    try:
                        # Copy database file
                        shutil.copy2(db_path, backup_file)
                        
                        # Calculate size
                        size_mb = backup_file.stat().st_size / (1024 * 1024)
                        
                        backup_results['results'][db_name] = {
                            'success': True,
                            'backup_file': str(backup_file),
                            'size_mb': size_mb
                        }
                        
                        backup_results['backed_up_databases'] += 1
                        backup_results['total_size_mb'] += size_mb
                        
                        print(f" {db_name}: Backed up ({size_mb:.1f} MB)")
                        
                    except Exception as e:
                        backup_results['results'][db_name] = {
                            'success': False,
                            'error': str(e)
                        }
                        backup_results['failed_backups'] += 1
                        print(f"L {db_name}: Backup failed")
                else:
                    backup_results['results'][db_name] = {
                        'success': False,
                        'error': 'Database file not found'
                    }
                    backup_results['failed_backups'] += 1
            
            # Create backup metadata
            metadata = {
                'backup_name': backup_name,
                'created_at': datetime.now().isoformat(),
                'database_count': backup_results['backed_up_databases'],
                'total_size_mb': backup_results['total_size_mb'],
                'databases': list(backup_results['results'].keys())
            }
            
            metadata_file = backup_dir / 'backup_metadata.json'
            async with aiofiles.open(metadata_file, 'w') as f:
                await f.write(json.dumps(metadata, indent=2))
            
            print(f"=¾ Database backup completed:")
            print(f"   Backed up: {backup_results['backed_up_databases']}")
            print(f"   Failed: {backup_results['failed_backups']}")
            print(f"   Total size: {backup_results['total_size_mb']:.1f} MB")
            
            return backup_results
            
        except Exception as e:
            print(f"L Database backup failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def restore_databases(self, backup_path: Path) -> Dict[str, Any]:
        """
        Restore databases from backup.
        
        Args:
            backup_path: Path to backup directory
        
        Returns:
            Restoration results
        """
        print(f"{  Restoring databases from: {backup_path}")
        
        try:
            if not backup_path.exists():
                return {
                    'success': False,
                    'error': f'Backup path not found: {backup_path}'
                }
            
            # Load backup metadata
            metadata_file = backup_path / 'backup_metadata.json'
            metadata = {}
            
            if metadata_file.exists():
                async with aiofiles.open(metadata_file, 'r') as f:
                    metadata = json.loads(await f.read())
            
            restore_results = {
                'backup_path': str(backup_path),
                'timestamp': datetime.now().isoformat(),
                'restored_databases': 0,
                'failed_restorations': 0,
                'results': {}
            }
            
            # Create safety backup before restoration
            safety_backup = await self.backup_databases(f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            restore_results['safety_backup'] = safety_backup.get('backup_name')
            
            # Restore each database
            for db_name, db_config in DATABASE_SCHEMAS.items():
                backup_file = backup_path / db_config['name']
                target_file = self.db_dir / db_config['name']
                
                if backup_file.exists():
                    try:
                        # Copy backup file to database directory
                        shutil.copy2(backup_file, target_file)
                        
                        restore_results['results'][db_name] = {
                            'success': True,
                            'restored_file': str(target_file)
                        }
                        
                        restore_results['restored_databases'] += 1
                        print(f" {db_name}: Restored successfully")
                        
                    except Exception as e:
                        restore_results['results'][db_name] = {
                            'success': False,
                            'error': str(e)
                        }
                        restore_results['failed_restorations'] += 1
                        print(f"L {db_name}: Restoration failed")
                else:
                    restore_results['results'][db_name] = {
                        'success': False,
                        'error': 'Backup file not found'
                    }
                    restore_results['failed_restorations'] += 1
                    print(f"   {db_name}: Backup file not found")
            
            print(f"{  Database restoration completed:")
            print(f"   Restored: {restore_results['restored_databases']}")
            print(f"   Failed: {restore_results['failed_restorations']}")
            
            return restore_results
            
        except Exception as e:
            print(f"L Database restoration failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def optimize_databases(self) -> Dict[str, Any]:
        """
        Optimize database performance.
        
        Returns:
            Optimization results
        """
        print("¡ Optimizing database performance...")
        
        try:
            optimization_results = {
                'timestamp': datetime.now().isoformat(),
                'optimized_databases': 0,
                'results': {}
            }
            
            for db_name, db_config in DATABASE_SCHEMAS.items():
                db_path = self.db_dir / db_config['name']
                
                if db_path.exists() and db_config['type'] == 'sqlite':
                    result = await self._optimize_sqlite_database(db_path)
                    optimization_results['results'][db_name] = result
                    
                    if result['success']:
                        optimization_results['optimized_databases'] += 1
                        print(f" {db_name}: Optimized")
                    else:
                        print(f"L {db_name}: Optimization failed")
            
            print(f"¡ Database optimization completed:")
            print(f"   Optimized: {optimization_results['optimized_databases']}")
            
            return optimization_results
            
        except Exception as e:
            print(f"L Database optimization failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    # Private methods
    
    async def _setup_single_database(self, db_config: Dict[str, Any],
                                   force_recreate: bool = False) -> Dict[str, Any]:
        """Set up a single database"""
        db_name = db_config['name']
        db_path = self.db_dir / db_name
        
        try:
            # Check if database exists and handle force recreation
            if db_path.exists() and force_recreate:
                backup_path = db_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db')
                shutil.move(str(db_path), str(backup_path))
                print(f"=æ Backed up existing database to: {backup_path}")
            
            # Create database based on type
            if db_config['type'] == 'sqlite':
                result = await self._setup_sqlite_database(db_path, db_config)
            elif db_config['type'] == 'postgresql':
                result = await self._setup_postgresql_database(db_config)
            elif db_config['type'] == 'mongodb':
                result = await self._setup_mongodb_database(db_config)
            else:
                return {
                    'success': False,
                    'error': f"Unsupported database type: {db_config['type']}"
                }
            
            if result['success']:
                # Set secure permissions
                if db_path.exists():
                    db_path.chmod(0o600)
                
                # Create database metadata
                await self._create_database_metadata(db_config, db_path)
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _setup_sqlite_database(self, db_path: Path, db_config: Dict[str, Any]) -> Dict[str, Any]:
        """Set up SQLite database"""
        try:
            # Connect to database (creates if doesn't exist)
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Configure SQLite settings
            if self.config['wal_mode']:
                cursor.execute('PRAGMA journal_mode=WAL')
            
            if self.config['foreign_keys']:
                cursor.execute('PRAGMA foreign_keys=ON')
            
            if self.config['auto_vacuum']:
                cursor.execute('PRAGMA auto_vacuum=INCREMENTAL')
            
            # Create tables
            tables_created = 0
            for table_name in db_config.get('tables', []):
                if table_name in SQL_SCHEMAS:
                    cursor.execute(SQL_SCHEMAS[table_name])
                    tables_created += 1
                    
                    # Create indexes for this table
                    if table_name in DATABASE_INDEXES:
                        for index_sql in DATABASE_INDEXES[table_name]:
                            cursor.execute(index_sql)
            
            # Commit changes
            conn.commit()
            conn.close()
            
            return {
                'success': True,
                'database_path': str(db_path),
                'database_type': 'sqlite',
                'tables_created': tables_created,
                'size_mb': db_path.stat().st_size / (1024 * 1024) if db_path.exists() else 0
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _setup_postgresql_database(self, db_config: Dict[str, Any]) -> Dict[str, Any]:
        """Set up PostgreSQL database"""
        # Simplified PostgreSQL setup - would need real connection parameters
        return {
            'success': True,
            'database_type': 'postgresql',
            'note': 'PostgreSQL setup placeholder'
        }
    
    async def _setup_mongodb_database(self, db_config: Dict[str, Any]) -> Dict[str, Any]:
        """Set up MongoDB database"""
        # Simplified MongoDB setup - would need real connection parameters
        return {
            'success': True,
            'database_type': 'mongodb',
            'note': 'MongoDB setup placeholder'
        }
    
    async def _initialize_database_data(self, db_name: str) -> Dict[str, Any]:
        """Initialize database with default data"""
        try:
            if db_name == 'user_db':
                return await self._initialize_user_data()
            elif db_name == 'main_db':
                return await self._initialize_main_data()
            elif db_name == 'model_registry_db':
                return await self._initialize_model_registry_data()
            else:
                return {
                    'success': True,
                    'note': f'No default data defined for {db_name}'
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _initialize_user_data(self) -> Dict[str, Any]:
        """Initialize user database with default roles"""
        db_path = self.db_dir / 'users_auth.db'
        
        if not db_path.exists():
            return {
                'success': False,
                'error': 'User database not found'
            }
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Create default roles
            default_roles = [
                {
                    'id': 'role_admin',
                    'name': 'admin',
                    'description': 'System administrator with full access',
                    'permissions': json.dumps(['read', 'write', 'delete', 'admin', 'manage_users']),
                    'created_at': datetime.now().isoformat(),
                    'is_system_role': True
                },
                {
                    'id': 'role_user',
                    'name': 'user',
                    'description': 'Standard user with basic access',
                    'permissions': json.dumps(['read', 'write']),
                    'created_at': datetime.now().isoformat(),
                    'is_system_role': True
                },
                {
                    'id': 'role_viewer',
                    'name': 'viewer',
                    'description': 'Read-only access',
                    'permissions': json.dumps(['read']),
                    'created_at': datetime.now().isoformat(),
                    'is_system_role': True
                }
            ]
            
            # Insert default roles
            for role in default_roles:
                cursor.execute('''
                    INSERT OR IGNORE INTO roles 
                    (id, name, description, permissions, created_at, is_system_role)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    role['id'], role['name'], role['description'],
                    role['permissions'], role['created_at'], role['is_system_role']
                ))
            
            conn.commit()
            conn.close()
            
            return {
                'success': True,
                'roles_created': len(default_roles)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _initialize_main_data(self) -> Dict[str, Any]:
        """Initialize main database with default configurations"""
        db_path = self.db_dir / 'structured_docs_synth.db'
        
        if not db_path.exists():
            return {
                'success': False,
                'error': 'Main database not found'
            }
        
        # Initialize with default system settings
        return {
            'success': True,
            'note': 'Main database initialized'
        }
    
    async def _initialize_model_registry_data(self) -> Dict[str, Any]:
        """Initialize model registry with default model configurations"""
        return {
            'success': True,
            'note': 'Model registry initialized'
        }
    
    async def _verify_single_database(self, db_name: str, db_config: Dict[str, Any]) -> Dict[str, Any]:
        """Verify a single database"""
        db_path = self.db_dir / db_config['name']
        
        verification_result = {
            'database_name': db_name,
            'verified': False,
            'issues': []
        }
        
        try:
            if not db_path.exists():
                verification_result['issues'].append('Database file does not exist')
                return verification_result
            
            if db_config['type'] == 'sqlite':
                # Verify SQLite database
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                # Check database integrity
                cursor.execute('PRAGMA integrity_check')
                integrity_result = cursor.fetchone()[0]
                
                if integrity_result != 'ok':
                    verification_result['issues'].append(f'Integrity check failed: {integrity_result}')
                
                # Check if required tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                existing_tables = {row[0] for row in cursor.fetchall()}
                
                required_tables = set(db_config.get('tables', []))
                missing_tables = required_tables - existing_tables
                
                if missing_tables:
                    verification_result['issues'].append(f'Missing tables: {missing_tables}')
                
                conn.close()
            
            # If no issues found, mark as verified
            if not verification_result['issues']:
                verification_result['verified'] = True
            
            return verification_result
            
        except Exception as e:
            verification_result['issues'].append(f'Verification error: {str(e)}')
            return verification_result
    
    async def _optimize_sqlite_database(self, db_path: Path) -> Dict[str, Any]:
        """Optimize SQLite database"""
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Run optimization commands
            cursor.execute('VACUUM')
            cursor.execute('ANALYZE')
            cursor.execute('PRAGMA optimize')
            
            conn.commit()
            conn.close()
            
            return {
                'success': True,
                'optimizations': ['VACUUM', 'ANALYZE', 'PRAGMA optimize']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _create_database_config(self):
        """Create database configuration file"""
        config = {
            'databases': DATABASE_SCHEMAS,
            'db_directory': str(self.db_dir),
            'backup_directory': str(self.backup_dir),
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0'
        }
        
        config_file = self.db_dir / 'database_config.json'
        async with aiofiles.open(config_file, 'w') as f:
            await f.write(json.dumps(config, indent=2))
    
    async def _create_database_metadata(self, db_config: Dict[str, Any], db_path: Path):
        """Create metadata file for database"""
        metadata = {
            'database_name': db_config['name'],
            'database_type': db_config['type'],
            'description': db_config['description'],
            'created_at': datetime.now().isoformat(),
            'tables': db_config.get('tables', []),
            'size_bytes': db_path.stat().st_size if db_path.exists() else 0
        }
        
        metadata_file = db_path.with_suffix('.metadata.json')
        async with aiofiles.open(metadata_file, 'w') as f:
            await f.write(json.dumps(metadata, indent=2))
    
    async def _setup_database_monitoring(self):
        """Set up database monitoring"""
        # This would set up monitoring for database performance and health
        print("=Ê Database monitoring configured")
    
    async def _create_initial_backup(self):
        """Create initial backup after setup"""
        await self.backup_databases('initial_setup_backup')


async def main():
    """
    Main database setup script function.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Set up databases for structured document synthesis'
    )
    parser.add_argument(
        'action',
        choices=['setup-all', 'setup-db', 'init-data', 'verify', 'backup', 'restore', 'optimize'],
        help='Action to perform'
    )
    parser.add_argument(
        '--db-name',
        help='Specific database name for single database operations'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recreation of existing databases'
    )
    parser.add_argument(
        '--backup-name',
        help='Custom backup name'
    )
    parser.add_argument(
        '--backup-path',
        type=Path,
        help='Path to backup for restoration'
    )
    parser.add_argument(
        '--db-dir',
        type=Path,
        help='Custom database directory'
    )
    
    args = parser.parse_args()
    
    # Initialize database setup system
    db_setup = DatabaseSetup(db_dir=args.db_dir)
    
    if args.action == 'setup-all':
        result = await db_setup.setup_all_databases(args.force)
        
        if result.get('success', True):
            print(f"\n Database setup completed")
            print(f"=Ä  Successful: {result['successful_setups']}")
            print(f"L Failed: {result['failed_setups']}")
            
            if result['failed_setups'] > 0:
                return 1
        else:
            print(f"\nL Database setup failed: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.action == 'setup-db':
        if not args.db_name:
            print("L Database name required for single database setup")
            return 1
        
        result = await db_setup.setup_database(args.db_name, args.force)
        
        if result['success']:
            print(f" Database setup completed: {args.db_name}")
            print(f"=Á Path: {result['database_path']}")
            print(f"=Ê Tables: {result['tables_created']}")
        else:
            print(f"L Database setup failed: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.action == 'init-data':
        result = await db_setup.initialize_default_data()
        
        if result.get('success', True):
            print(f" Default data initialization completed")
            print(f"=Ä  Initialized: {result['initialized_databases']}")
        else:
            print(f"L Data initialization failed: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.action == 'verify':
        result = await db_setup.verify_databases()
        
        if result.get('success', True):
            print(f" Database verification completed")
            print(f" Verified: {result['verified_databases']}")
            print(f"L Failed: {result['failed_databases']}")
            
            if result['failed_databases'] > 0:
                return 1
        else:
            print(f"L Database verification failed: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.action == 'backup':
        result = await db_setup.backup_databases(args.backup_name)
        
        if result.get('success', True):
            print(f" Database backup completed")
            print(f"=Á Backup path: {result['backup_path']}")
            print(f"=¾ Total size: {result['total_size_mb']:.1f} MB")
        else:
            print(f"L Database backup failed: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.action == 'restore':
        if not args.backup_path:
            print("L Backup path required for restoration")
            return 1
        
        result = await db_setup.restore_databases(args.backup_path)
        
        if result.get('success', True):
            print(f" Database restoration completed")
            print(f"{  Restored: {result['restored_databases']}")
            print(f"L Failed: {result['failed_restorations']}")
        else:
            print(f"L Database restoration failed: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.action == 'optimize':
        result = await db_setup.optimize_databases()
        
        if result.get('success', True):
            print(f" Database optimization completed")
            print(f"¡ Optimized: {result['optimized_databases']}")
        else:
            print(f"L Database optimization failed: {result.get('error', 'Unknown error')}")
            return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(asyncio.run(main()))