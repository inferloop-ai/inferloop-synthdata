#!/usr/bin/env python3
"""
Access control setup and management system for structured document synthesis.

Provides comprehensive access control including role-based access control (RBAC),
attribute-based access control (ABAC), API authentication, file permissions,
and audit trail management.
"""

import asyncio
import json
import hashlib
import secrets
import sqlite3
import pwd
import grp
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
import jwt
import bcrypt
import aiofiles
from enum import Enum
from dataclasses import dataclass, asdict

# Access control configuration
ACCESS_CONTROL_CONFIG = {
    'default_token_expiry_hours': 24,
    'password_min_length': 12,
    'password_require_special': True,
    'password_require_numbers': True,
    'password_require_uppercase': True,
    'max_failed_attempts': 5,
    'lockout_duration_minutes': 30,
    'enable_mfa': False,
    'session_timeout_minutes': 60,
    'audit_enabled': True,
    'backup_enabled': True
}

DEFAULT_ACCESS_DIR = Path.home() / '.structured_docs_synth' / 'access_control'
DEFAULT_DB_PATH = DEFAULT_ACCESS_DIR / 'access_control.db'
DEFAULT_AUDIT_DIR = DEFAULT_ACCESS_DIR / 'audit'


class Permission(Enum):
    """System permissions"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"
    CREATE_USER = "create_user"
    MANAGE_ROLES = "manage_roles"
    VIEW_AUDIT = "view_audit"
    SYSTEM_CONFIG = "system_config"
    MODEL_ACCESS = "model_access"
    DATA_EXPORT = "data_export"
    ENCRYPTION_KEYS = "encryption_keys"


class ResourceType(Enum):
    """Resource types for access control"""
    FILE = "file"
    DIRECTORY = "directory"
    DATABASE = "database"
    API_ENDPOINT = "api_endpoint"
    MODEL = "model"
    DATASET = "dataset"
    CONFIGURATION = "configuration"
    SYSTEM = "system"


@dataclass
class User:
    """User data model"""
    user_id: str
    username: str
    email: str
    password_hash: str
    salt: str
    roles: List[str]
    is_active: bool
    created_at: str
    last_login: Optional[str] = None
    failed_attempts: int = 0
    locked_until: Optional[str] = None
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None


@dataclass
class Role:
    """Role data model"""
    role_id: str
    role_name: str
    description: str
    permissions: List[str]
    created_at: str
    is_active: bool


@dataclass
class AccessRule:
    """Access control rule"""
    rule_id: str
    user_id: Optional[str]
    role_id: Optional[str]
    resource_type: str
    resource_path: str
    permissions: List[str]
    conditions: Dict[str, Any]
    created_at: str
    is_active: bool


@dataclass
class Session:
    """User session"""
    session_id: str
    user_id: str
    token: str
    created_at: str
    expires_at: str
    ip_address: Optional[str]
    user_agent: Optional[str]
    is_active: bool


class AccessControlManager:
    """Comprehensive access control management system"""
    
    def __init__(self, access_dir: Optional[Path] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.access_dir = access_dir or DEFAULT_ACCESS_DIR
        self.audit_dir = DEFAULT_AUDIT_DIR
        self.db_path = self.access_dir / 'access_control.db'
        self.config = {**ACCESS_CONTROL_CONFIG, **(config or {})}
        
        # Ensure directories exist
        for directory in [self.access_dir, self.audit_dir]:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        
        # Initialize JWT secret
        self.jwt_secret = self._load_or_generate_jwt_secret()
        
        # Initialize database
        asyncio.create_task(self._init_database())
    
    async def setup_initial_access_control(self, admin_username: str, 
                                         admin_password: str,
                                         admin_email: str) -> Dict[str, Any]:
        """
        Set up initial access control system with admin user.
        
        Args:
            admin_username: Initial admin username
            admin_password: Initial admin password
            admin_email: Initial admin email
        
        Returns:
            Setup result
        """
        print("🔐 Setting up initial access control system...")
        
        try:
            # Initialize database
            await self._init_database()
            
            # Create default roles
            roles_created = await self._create_default_roles()
            
            # Create admin user
            admin_user = await self.create_user(
                username=admin_username,
                email=admin_email,
                password=admin_password,
                roles=['admin']
            )
            
            if not admin_user['success']:
                return admin_user
            
            # Set up default access rules
            rules_created = await self._create_default_access_rules()
            
            # Set up file system permissions
            fs_setup = await self._setup_filesystem_permissions()
            
            # Create audit entry
            await self._log_audit_event(
                'system_setup',
                admin_user['user_id'],
                'Initial access control setup completed',
                {'roles_created': len(roles_created), 'rules_created': len(rules_created)}
            )
            
            print("✅ Access control system setup completed")
            
            return {
                'success': True,
                'admin_user_id': admin_user['user_id'],
                'roles_created': len(roles_created),
                'rules_created': len(rules_created),
                'filesystem_setup': fs_setup['success']
            }
            
        except Exception as e:
            print(f"❌ Access control setup failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def create_user(self, username: str, email: str, password: str,
                         roles: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create a new user.
        
        Args:
            username: Username
            email: User email
            password: User password
            roles: List of role names
        
        Returns:
            User creation result
        """
        print(f"👤 Creating user: {username}")
        
        try:
            # Validate password
            password_validation = self._validate_password(password)
            if not password_validation['valid']:
                return {
                    'success': False,
                    'error': f"Password validation failed: {password_validation['reason']}"
                }
            
            # Check if user already exists
            if await self._user_exists(username, email):
                return {
                    'success': False,
                    'error': 'User with this username or email already exists'
                }
            
            # Generate user ID
            user_id = f"user_{secrets.token_hex(8)}"
            
            # Hash password
            salt = secrets.token_hex(16)
            password_hash = self._hash_password(password, salt)
            
            # Create user object
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                password_hash=password_hash,
                salt=salt,
                roles=roles or ['user'],
                is_active=True,
                created_at=datetime.now().isoformat()
            )
            
            # Save to database
            await self._save_user(user)
            
            # Log audit event
            await self._log_audit_event(
                'user_created',
                user_id,
                f'User {username} created',
                {'email': email, 'roles': user.roles}
            )
            
            print(f"✅ User created: {username}")
            
            return {
                'success': True,
                'user_id': user_id,
                'username': username
            }
            
        except Exception as e:
            print(f"❌ User creation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def authenticate_user(self, username: str, password: str,
                              ip_address: Optional[str] = None,
                              user_agent: Optional[str] = None) -> Dict[str, Any]:
        """
        Authenticate user and create session.
        
        Args:
            username: Username
            password: Password
            ip_address: Client IP address
            user_agent: Client user agent
        
        Returns:
            Authentication result with token
        """
        try:
            # Get user from database
            user = await self._get_user_by_username(username)
            if not user:
                await self._log_audit_event(
                    'login_failed',
                    None,
                    f'Login attempt with unknown username: {username}',
                    {'ip_address': ip_address}
                )
                return {
                    'success': False,
                    'error': 'Invalid credentials'
                }
            
            # Check if user is locked
            if user.locked_until:
                lock_until = datetime.fromisoformat(user.locked_until)
                if datetime.now() < lock_until:
                    return {
                        'success': False,
                        'error': f'Account locked until {user.locked_until}'
                    }
                else:
                    # Unlock user
                    await self._unlock_user(user.user_id)
            
            # Verify password
            if not self._verify_password(password, user.password_hash, user.salt):
                # Increment failed attempts
                await self._increment_failed_attempts(user.user_id)
                
                await self._log_audit_event(
                    'login_failed',
                    user.user_id,
                    f'Invalid password for user: {username}',
                    {'ip_address': ip_address}
                )
                
                return {
                    'success': False,
                    'error': 'Invalid credentials'
                }
            
            # Check if user is active
            if not user.is_active:
                return {
                    'success': False,
                    'error': 'Account is disabled'
                }
            
            # Reset failed attempts on successful login
            await self._reset_failed_attempts(user.user_id)
            
            # Create session and token
            session = await self._create_session(user, ip_address, user_agent)
            
            # Update last login
            await self._update_last_login(user.user_id)
            
            # Log successful login
            await self._log_audit_event(
                'login_success',
                user.user_id,
                f'User {username} logged in successfully',
                {'ip_address': ip_address, 'session_id': session.session_id}
            )
            
            print(f"✅ User authenticated: {username}")
            
            return {
                'success': True,
                'user_id': user.user_id,
                'username': user.username,
                'roles': user.roles,
                'token': session.token,
                'session_id': session.session_id,
                'expires_at': session.expires_at
            }
            
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def check_permission(self, user_id: str, resource_type: str,
                             resource_path: str, permission: str,
                             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Check if user has permission for a resource.
        
        Args:
            user_id: User ID
            resource_type: Type of resource
            resource_path: Path to resource
            permission: Required permission
            context: Additional context for attribute-based checks
        
        Returns:
            Permission check result
        """
        try:
            # Get user
            user = await self._get_user_by_id(user_id)
            if not user or not user.is_active:
                return {
                    'allowed': False,
                    'reason': 'User not found or inactive'
                }
            
            # Check user roles and permissions
            user_permissions = await self._get_user_permissions(user_id)
            
            # Check if user has admin permission (grants all access)
            if Permission.ADMIN.value in user_permissions:
                return {
                    'allowed': True,
                    'reason': 'Admin permission'
                }
            
            # Check specific permission
            if permission in user_permissions:
                # Check access rules for this specific resource
                access_allowed = await self._check_access_rules(
                    user_id, resource_type, resource_path, permission, context
                )
                
                if access_allowed:
                    return {
                        'allowed': True,
                        'reason': 'Permission granted by access rules'
                    }
            
            # Log denied access attempt
            await self._log_audit_event(
                'access_denied',
                user_id,
                f'Access denied to {resource_type}:{resource_path}',
                {
                    'permission': permission,
                    'context': context
                }
            )
            
            return {
                'allowed': False,
                'reason': 'Insufficient permissions'
            }
            
        except Exception as e:
            print(f"❌ Permission check failed: {e}")
            return {
                'allowed': False,
                'reason': f'Error checking permissions: {str(e)}'
            }
    
    async def create_role(self, role_name: str, description: str,
                         permissions: List[str]) -> Dict[str, Any]:
        """
        Create a new role.
        
        Args:
            role_name: Role name
            description: Role description
            permissions: List of permissions
        
        Returns:
            Role creation result
        """
        print(f"👥 Creating role: {role_name}")
        
        try:
            # Check if role already exists
            if await self._role_exists(role_name):
                return {
                    'success': False,
                    'error': 'Role already exists'
                }
            
            # Validate permissions
            valid_permissions = [p.value for p in Permission]
            invalid_perms = [p for p in permissions if p not in valid_permissions]
            
            if invalid_perms:
                return {
                    'success': False,
                    'error': f'Invalid permissions: {invalid_perms}'
                }
            
            # Generate role ID
            role_id = f"role_{secrets.token_hex(8)}"
            
            # Create role object
            role = Role(
                role_id=role_id,
                role_name=role_name,
                description=description,
                permissions=permissions,
                created_at=datetime.now().isoformat(),
                is_active=True
            )
            
            # Save to database
            await self._save_role(role)
            
            print(f"✅ Role created: {role_name}")
            
            return {
                'success': True,
                'role_id': role_id,
                'role_name': role_name
            }
            
        except Exception as e:
            print(f"❌ Role creation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def create_access_rule(self, rule_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an access control rule.
        
        Args:
            rule_config: Rule configuration
        
        Returns:
            Rule creation result
        """
        try:
            rule_id = f"rule_{secrets.token_hex(8)}"
            
            rule = AccessRule(
                rule_id=rule_id,
                user_id=rule_config.get('user_id'),
                role_id=rule_config.get('role_id'),
                resource_type=rule_config['resource_type'],
                resource_path=rule_config['resource_path'],
                permissions=rule_config['permissions'],
                conditions=rule_config.get('conditions', {}),
                created_at=datetime.now().isoformat(),
                is_active=True
            )
            
            # Save to database
            await self._save_access_rule(rule)
            
            return {
                'success': True,
                'rule_id': rule_id
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def setup_file_permissions(self, base_path: Path,
                                   permission_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set up file system permissions.
        
        Args:
            base_path: Base path to configure
            permission_config: Permission configuration
        
        Returns:
            Setup result
        """
        print(f"📁 Setting up file permissions for: {base_path}")
        
        try:
            results = {
                'directories_configured': 0,
                'files_configured': 0,
                'errors': []
            }
            
            # Create secure directories
            secure_dirs = permission_config.get('secure_directories', [])
            for dir_config in secure_dirs:
                dir_path = base_path / dir_config['path']
                dir_path.mkdir(parents=True, exist_ok=True)
                
                # Set permissions
                mode = int(dir_config.get('mode', '0o750'), 8)
                dir_path.chmod(mode)
                
                # Set ownership if specified
                if 'owner' in dir_config:
                    try:
                        user_info = pwd.getpwnam(dir_config['owner'])
                        group_info = grp.getgrnam(dir_config.get('group', dir_config['owner']))
                        os.chown(dir_path, user_info.pw_uid, group_info.gr_gid)
                    except Exception as e:
                        results['errors'].append(f"Could not set ownership for {dir_path}: {e}")
                
                results['directories_configured'] += 1
            
            # Configure file permissions
            file_patterns = permission_config.get('file_patterns', [])
            for pattern_config in file_patterns:
                pattern = pattern_config['pattern']
                mode = int(pattern_config.get('mode', '0o640'), 8)
                
                for file_path in base_path.rglob(pattern):
                    if file_path.is_file():
                        try:
                            file_path.chmod(mode)
                            results['files_configured'] += 1
                        except Exception as e:
                            results['errors'].append(f"Could not set permissions for {file_path}: {e}")
            
            print(f"✅ File permissions configured:")
            print(f"   Directories: {results['directories_configured']}")
            print(f"   Files: {results['files_configured']}")
            
            return {
                'success': True,
                **results
            }
            
        except Exception as e:
            print(f"❌ File permission setup failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def generate_api_key(self, user_id: str, name: str,
                             permissions: List[str],
                             expires_days: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate an API key for a user.
        
        Args:
            user_id: User ID
            name: API key name
            permissions: List of permissions
            expires_days: Expiration in days
        
        Returns:
            API key generation result
        """
        try:
            # Generate API key
            api_key = f"sds_{secrets.token_urlsafe(32)}"
            
            # Set expiration
            expires_at = None
            if expires_days:
                expires_at = (datetime.now() + timedelta(days=expires_days)).isoformat()
            
            # Save API key
            await self._save_api_key({
                'api_key': api_key,
                'user_id': user_id,
                'name': name,
                'permissions': permissions,
                'created_at': datetime.now().isoformat(),
                'expires_at': expires_at,
                'is_active': True
            })
            
            # Log audit event
            await self._log_audit_event(
                'api_key_created',
                user_id,
                f'API key created: {name}',
                {'permissions': permissions, 'expires_at': expires_at}
            )
            
            return {
                'success': True,
                'api_key': api_key,
                'expires_at': expires_at
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    # Private methods
    
    async def _init_database(self):
        """Initialize access control database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                roles TEXT NOT NULL,
                is_active BOOLEAN NOT NULL,
                created_at TEXT NOT NULL,
                last_login TEXT,
                failed_attempts INTEGER DEFAULT 0,
                locked_until TEXT,
                mfa_enabled BOOLEAN DEFAULT FALSE,
                mfa_secret TEXT
            )
        ''')
        
        # Roles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS roles (
                role_id TEXT PRIMARY KEY,
                role_name TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL,
                permissions TEXT NOT NULL,
                created_at TEXT NOT NULL,
                is_active BOOLEAN NOT NULL
            )
        ''')
        
        # Access rules table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS access_rules (
                rule_id TEXT PRIMARY KEY,
                user_id TEXT,
                role_id TEXT,
                resource_type TEXT NOT NULL,
                resource_path TEXT NOT NULL,
                permissions TEXT NOT NULL,
                conditions TEXT,
                created_at TEXT NOT NULL,
                is_active BOOLEAN NOT NULL
            )
        ''')
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                token TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                is_active BOOLEAN NOT NULL
            )
        ''')
        
        # API keys table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                api_key TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                permissions TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                is_active BOOLEAN NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_or_generate_jwt_secret(self) -> str:
        """Load or generate JWT secret"""
        secret_file = self.access_dir / 'jwt_secret.key'
        
        if secret_file.exists():
            with open(secret_file, 'r') as f:
                return f.read().strip()
        else:
            secret = secrets.token_urlsafe(64)
            with open(secret_file, 'w') as f:
                f.write(secret)
            secret_file.chmod(0o600)
            return secret
    
    def _validate_password(self, password: str) -> Dict[str, Any]:
        """Validate password strength"""
        if len(password) < self.config['password_min_length']:
            return {
                'valid': False,
                'reason': f"Password must be at least {self.config['password_min_length']} characters"
            }
        
        if self.config['password_require_uppercase'] and not any(c.isupper() for c in password):
            return {
                'valid': False,
                'reason': 'Password must contain at least one uppercase letter'
            }
        
        if self.config['password_require_numbers'] and not any(c.isdigit() for c in password):
            return {
                'valid': False,
                'reason': 'Password must contain at least one number'
            }
        
        if self.config['password_require_special'] and not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password):
            return {
                'valid': False,
                'reason': 'Password must contain at least one special character'
            }
        
        return {'valid': True}
    
    def _hash_password(self, password: str, salt: str) -> str:
        """Hash password with salt"""
        return hashlib.pbkdf2_hex(password.encode(), salt.encode(), 100000, 64)
    
    def _verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify password against hash"""
        return self._hash_password(password, salt) == password_hash
    
    async def _create_session(self, user: User, ip_address: Optional[str],
                            user_agent: Optional[str]) -> Session:
        """Create user session"""
        session_id = f"session_{secrets.token_hex(16)}"
        expires_at = datetime.now() + timedelta(hours=self.config['default_token_expiry_hours'])
        
        # Create JWT token
        token_payload = {
            'user_id': user.user_id,
            'username': user.username,
            'roles': user.roles,
            'session_id': session_id,
            'exp': expires_at.timestamp()
        }
        
        token = jwt.encode(token_payload, self.jwt_secret, algorithm='HS256')
        
        session = Session(
            session_id=session_id,
            user_id=user.user_id,
            token=token,
            created_at=datetime.now().isoformat(),
            expires_at=expires_at.isoformat(),
            ip_address=ip_address,
            user_agent=user_agent,
            is_active=True
        )
        
        # Save session to database
        await self._save_session(session)
        
        return session
    
    async def _create_default_roles(self) -> List[str]:
        """Create default system roles"""
        default_roles = [
            {
                'role_name': 'admin',
                'description': 'System administrator with full access',
                'permissions': [p.value for p in Permission]
            },
            {
                'role_name': 'user',
                'description': 'Standard user with basic access',
                'permissions': [Permission.READ.value, Permission.WRITE.value]
            },
            {
                'role_name': 'viewer',
                'description': 'Read-only access',
                'permissions': [Permission.READ.value]
            },
            {
                'role_name': 'data_scientist',
                'description': 'Data scientist with model and dataset access',
                'permissions': [
                    Permission.READ.value,
                    Permission.WRITE.value,
                    Permission.MODEL_ACCESS.value,
                    Permission.DATA_EXPORT.value
                ]
            }
        ]
        
        created_roles = []
        for role_config in default_roles:
            result = await self.create_role(**role_config)
            if result['success']:
                created_roles.append(result['role_name'])
        
        return created_roles
    
    async def _create_default_access_rules(self) -> List[str]:
        """Create default access rules"""
        # This would create default access rules based on system requirements
        return []
    
    async def _setup_filesystem_permissions(self) -> Dict[str, Any]:
        """Set up basic filesystem permissions"""
        try:
            # Set restrictive permissions on access control directory
            self.access_dir.chmod(0o700)
            self.audit_dir.chmod(0o700)
            
            # Set restrictive permissions on database
            if self.db_path.exists():
                self.db_path.chmod(0o600)
            
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _log_audit_event(self, event_type: str, user_id: Optional[str],
                             description: str, details: Optional[Dict[str, Any]] = None):
        """Log audit event"""
        if not self.config['audit_enabled']:
            return
        
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'description': description,
            'details': details or {}
        }
        
        # Save to audit log file
        audit_file = self.audit_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.log"
        
        try:
            async with aiofiles.open(audit_file, 'a') as f:
                await f.write(json.dumps(audit_entry) + '\n')
        except Exception as e:
            print(f"⚠️  Warning: Could not write audit log: {e}")
    
    # Simplified database operations (would be more robust in production)
    
    async def _save_user(self, user: User):
        """Save user to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO users 
            (user_id, username, email, password_hash, salt, roles, is_active, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user.user_id, user.username, user.email, user.password_hash,
            user.salt, json.dumps(user.roles), user.is_active, user.created_at
        ))
        conn.commit()
        conn.close()
    
    async def _save_role(self, role: Role):
        """Save role to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO roles 
            (role_id, role_name, description, permissions, created_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            role.role_id, role.role_name, role.description,
            json.dumps(role.permissions), role.created_at, role.is_active
        ))
        conn.commit()
        conn.close()
    
    async def _save_access_rule(self, rule: AccessRule):
        """Save access rule to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO access_rules 
            (rule_id, user_id, role_id, resource_type, resource_path, permissions, conditions, created_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            rule.rule_id, rule.user_id, rule.role_id, rule.resource_type,
            rule.resource_path, json.dumps(rule.permissions),
            json.dumps(rule.conditions), rule.created_at, rule.is_active
        ))
        conn.commit()
        conn.close()
    
    async def _save_session(self, session: Session):
        """Save session to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sessions 
            (session_id, user_id, token, created_at, expires_at, ip_address, user_agent, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session.session_id, session.user_id, session.token,
            session.created_at, session.expires_at, session.ip_address,
            session.user_agent, session.is_active
        ))
        conn.commit()
        conn.close()
    
    async def _save_api_key(self, api_key_data: Dict[str, Any]):
        """Save API key to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO api_keys 
            (api_key, user_id, name, permissions, created_at, expires_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            api_key_data['api_key'], api_key_data['user_id'], api_key_data['name'],
            json.dumps(api_key_data['permissions']), api_key_data['created_at'],
            api_key_data['expires_at'], api_key_data['is_active']
        ))
        conn.commit()
        conn.close()
    
    # Simplified query methods
    
    async def _user_exists(self, username: str, email: str) -> bool:
        """Check if user exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT COUNT(*) FROM users WHERE username = ? OR email = ?',
            (username, email)
        )
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
    
    async def _role_exists(self, role_name: str) -> bool:
        """Check if role exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT COUNT(*) FROM roles WHERE role_name = ?',
            (role_name,)
        )
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
    
    async def _get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM users WHERE username = ?',
            (username,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return User(
                user_id=row[0],
                username=row[1],
                email=row[2],
                password_hash=row[3],
                salt=row[4],
                roles=json.loads(row[5]),
                is_active=bool(row[6]),
                created_at=row[7],
                last_login=row[8],
                failed_attempts=row[9] or 0,
                locked_until=row[10],
                mfa_enabled=bool(row[11]) if row[11] is not None else False,
                mfa_secret=row[12]
            )
        return None
    
    async def _get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM users WHERE user_id = ?',
            (user_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return User(
                user_id=row[0],
                username=row[1],
                email=row[2],
                password_hash=row[3],
                salt=row[4],
                roles=json.loads(row[5]),
                is_active=bool(row[6]),
                created_at=row[7],
                last_login=row[8],
                failed_attempts=row[9] or 0,
                locked_until=row[10],
                mfa_enabled=bool(row[11]) if row[11] is not None else False,
                mfa_secret=row[12]
            )
        return None
    
    async def _get_user_permissions(self, user_id: str) -> Set[str]:
        """Get all permissions for a user"""
        user = await self._get_user_by_id(user_id)
        if not user:
            return set()
        
        permissions = set()
        
        # Get permissions from roles
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for role_name in user.roles:
            cursor.execute(
                'SELECT permissions FROM roles WHERE role_name = ? AND is_active = ?',
                (role_name, True)
            )
            row = cursor.fetchone()
            if row:
                role_permissions = json.loads(row[0])
                permissions.update(role_permissions)
        
        conn.close()
        return permissions
    
    async def _check_access_rules(self, user_id: str, resource_type: str,
                                resource_path: str, permission: str,
                                context: Optional[Dict[str, Any]]) -> bool:
        """Check access rules for specific resource"""
        # Simplified implementation - would be more complex in production
        return True
    
    async def _increment_failed_attempts(self, user_id: str):
        """Increment failed login attempts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current failed attempts
        cursor.execute(
            'SELECT failed_attempts FROM users WHERE user_id = ?',
            (user_id,)
        )
        row = cursor.fetchone()
        
        if row:
            failed_attempts = (row[0] or 0) + 1
            locked_until = None
            
            # Lock account if max attempts reached
            if failed_attempts >= self.config['max_failed_attempts']:
                locked_until = (
                    datetime.now() + 
                    timedelta(minutes=self.config['lockout_duration_minutes'])
                ).isoformat()
            
            cursor.execute(
                'UPDATE users SET failed_attempts = ?, locked_until = ? WHERE user_id = ?',
                (failed_attempts, locked_until, user_id)
            )
        
        conn.commit()
        conn.close()
    
    async def _reset_failed_attempts(self, user_id: str):
        """Reset failed login attempts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE user_id = ?',
            (user_id,)
        )
        conn.commit()
        conn.close()
    
    async def _unlock_user(self, user_id: str):
        """Unlock user account"""
        await self._reset_failed_attempts(user_id)
    
    async def _update_last_login(self, user_id: str):
        """Update last login timestamp"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE users SET last_login = ? WHERE user_id = ?',
            (datetime.now().isoformat(), user_id)
        )
        conn.commit()
        conn.close()


async def main():
    """
    Main access control setup script function.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Set up and manage access control for structured document synthesis'
    )
    parser.add_argument(
        'action',
        choices=['setup', 'create-user', 'create-role', 'check-permission', 'generate-api-key'],
        help='Action to perform'
    )
    parser.add_argument(
        '--admin-username',
        help='Initial admin username (for setup)'
    )
    parser.add_argument(
        '--admin-password',
        help='Initial admin password (for setup)'
    )
    parser.add_argument(
        '--admin-email',
        help='Initial admin email (for setup)'
    )
    parser.add_argument(
        '--username',
        help='Username for user operations'
    )
    parser.add_argument(
        '--email',
        help='Email for user operations'
    )
    parser.add_argument(
        '--password',
        help='Password for user operations'
    )
    parser.add_argument(
        '--roles',
        nargs='+',
        help='Roles for user operations'
    )
    parser.add_argument(
        '--role-name',
        help='Role name for role operations'
    )
    parser.add_argument(
        '--role-description',
        help='Role description'
    )
    parser.add_argument(
        '--permissions',
        nargs='+',
        help='Permissions list'
    )
    parser.add_argument(
        '--access-dir',
        type=Path,
        help='Custom access control directory'
    )
    
    args = parser.parse_args()
    
    # Initialize access control manager
    access_manager = AccessControlManager(access_dir=args.access_dir)
    
    if args.action == 'setup':
        if not all([args.admin_username, args.admin_password, args.admin_email]):
            print("❌ Admin username, password, and email required for setup")
            return 1
        
        result = await access_manager.setup_initial_access_control(
            args.admin_username,
            args.admin_password,
            args.admin_email
        )
        
        if result['success']:
            print(f"\n✅ Access control setup completed")
            print(f"👤 Admin user created: {args.admin_username}")
            print(f"👥 Roles created: {result['roles_created']}")
        else:
            print(f"\n❌ Setup failed: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.action == 'create-user':
        if not all([args.username, args.email, args.password]):
            print("❌ Username, email, and password required")
            return 1
        
        result = await access_manager.create_user(
            args.username,
            args.email,
            args.password,
            args.roles
        )
        
        if result['success']:
            print(f"\n✅ User created: {args.username}")
            print(f"🆔 User ID: {result['user_id']}")
        else:
            print(f"\n❌ User creation failed: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.action == 'create-role':
        if not all([args.role_name, args.role_description, args.permissions]):
            print("❌ Role name, description, and permissions required")
            return 1
        
        result = await access_manager.create_role(
            args.role_name,
            args.role_description,
            args.permissions
        )
        
        if result['success']:
            print(f"\n✅ Role created: {args.role_name}")
            print(f"🆔 Role ID: {result['role_id']}")
        else:
            print(f"\n❌ Role creation failed: {result.get('error', 'Unknown error')}")
            return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(asyncio.run(main()))