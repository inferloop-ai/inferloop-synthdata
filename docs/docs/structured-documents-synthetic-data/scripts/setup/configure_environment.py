#!/usr/bin/env python3
"""
Environment configuration script for structured document synthesis.

Sets up and configures the complete environment including configuration files,
environment variables, logging setup, service configurations, and system
integration with validation and security checks.
"""

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import yaml
import aiofiles
import platform
import secrets
import hashlib

# Environment configuration
ENVIRONMENT_CONFIG = {
    'config_dir': Path.home() / '.structured_docs_synth',
    'log_dir': Path.home() / '.structured_docs_synth' / 'logs',
    'data_dir': Path.home() / '.structured_docs_synth' / 'data',
    'models_dir': Path.home() / '.structured_docs_synth' / 'models',
    'cache_dir': Path.home() / '.structured_docs_synth' / 'cache',
    'backup_dir': Path.home() / '.structured_docs_synth' / 'backups',
    'temp_dir': Path.home() / '.structured_docs_synth' / 'temp',
    'enable_logging': True,
    'log_level': 'INFO',
    'log_rotation': True,
    'max_log_size_mb': 100,
    'backup_count': 5,
    'create_systemd_service': False,
    'setup_shell_integration': False,
    'validate_config': True,
    'secure_permissions': True
}

# Default configuration templates
CONFIG_TEMPLATES = {
    'main_config': {
        'application': {
            'name': 'Structured Documents Synthetic Data Generation',
            'version': '1.0.0',
            'environment': 'production',
            'debug': False,
            'log_level': 'INFO'
        },
        'directories': {
            'data_dir': '~/.structured_docs_synth/data',
            'models_dir': '~/.structured_docs_synth/models',
            'cache_dir': '~/.structured_docs_synth/cache',
            'log_dir': '~/.structured_docs_synth/logs',
            'temp_dir': '~/.structured_docs_synth/temp'
        },
        'database': {
            'type': 'sqlite',
            'path': '~/.structured_docs_synth/databases/structured_docs_synth.db',
            'backup_enabled': True,
            'connection_pool_size': 10,
            'query_timeout': 30
        },
        'generation': {
            'default_output_format': 'pdf',
            'quality_level': 'high',
            'enable_ocr_noise': True,
            'parallel_processing': True,
            'max_workers': 4,
            'batch_size': 10
        },
        'security': {
            'encryption_enabled': True,
            'audit_logging': True,
            'access_control_enabled': True,
            'api_rate_limiting': True,
            'secure_headers': True
        },
        'performance': {
            'enable_caching': True,
            'cache_ttl_seconds': 3600,
            'memory_limit_gb': 4,
            'gpu_enabled': True,
            'model_optimization': True
        }
    },
    'logging_config': {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
            }
        },
        'handlers': {
            'default': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'level': 'INFO',
                'formatter': 'detailed',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': '~/.structured_docs_synth/logs/application.log',
                'maxBytes': 104857600,  # 100MB
                'backupCount': 5
            },
            'audit': {
                'level': 'INFO',
                'formatter': 'detailed',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': '~/.structured_docs_synth/logs/audit.log',
                'maxBytes': 104857600,
                'backupCount': 10
            }
        },
        'loggers': {
            '': {
                'handlers': ['default', 'file'],
                'level': 'INFO',
                'propagate': False
            },
            'audit': {
                'handlers': ['audit'],
                'level': 'INFO',
                'propagate': False
            }
        }
    },
    'api_config': {
        'server': {
            'host': '0.0.0.0',
            'port': 8000,
            'workers': 4,
            'worker_class': 'uvicorn.workers.UvicornWorker',
            'timeout': 120,
            'keepalive': 2
        },
        'cors': {
            'allow_origins': ['http://localhost:3000'],
            'allow_credentials': True,
            'allow_methods': ['GET', 'POST', 'PUT', 'DELETE'],
            'allow_headers': ['*']
        },
        'rate_limiting': {
            'enabled': True,
            'default_rate': '100/minute',
            'burst_rate': '200/minute',
            'storage': 'memory'
        },
        'authentication': {
            'jwt_secret_key': '',  # Will be generated
            'jwt_algorithm': 'HS256',
            'access_token_expire_minutes': 30,
            'refresh_token_expire_days': 7
        },
        'middleware': {
            'trusted_hosts': ['localhost', '127.0.0.1'],
            'https_redirect': False,
            'gzip_compression': True,
            'request_timeout': 30
        }
    },
    'docker_config': {
        'version': '3.8',
        'services': {
            'structured-docs-synth': {
                'build': {
                    'context': '.',
                    'dockerfile': 'Dockerfile'
                },
                'ports': ['8000:8000'],
                'environment': [
                    'PYTHONPATH=/app',
                    'ENVIRONMENT=production'
                ],
                'volumes': [
                    '~/.structured_docs_synth:/app/data',
                    './logs:/app/logs'
                ],
                'restart': 'unless-stopped',
                'healthcheck': {
                    'test': ['CMD', 'curl', '-f', 'http://localhost:8000/health'],
                    'interval': '30s',
                    'timeout': '10s',
                    'retries': 3
                }
            }
        }
    }
}

# Environment variables to set
ENVIRONMENT_VARIABLES = {
    'STRUCTURED_DOCS_SYNTH_HOME': '~/.structured_docs_synth',
    'STRUCTURED_DOCS_SYNTH_CONFIG': '~/.structured_docs_synth/config.yaml',
    'STRUCTURED_DOCS_SYNTH_LOG_LEVEL': 'INFO',
    'STRUCTURED_DOCS_SYNTH_ENV': 'production',
    'PYTHONPATH': '~/.structured_docs_synth',
    'TRANSFORMERS_CACHE': '~/.structured_docs_synth/cache/transformers',
    'HF_HOME': '~/.structured_docs_synth/cache/huggingface'
}

# Shell integration scripts
SHELL_SCRIPTS = {
    'bash_completion': '''
# Bash completion for structured-docs-synth
_structured_docs_synth_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    opts="generate validate export status config"
    
    if [[ ${cur} == -* ]] ; then
        COMPREPLY=( $(compgen -W "--help --version --config --verbose --output" -- ${cur}) )
        return 0
    fi
    
    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
    return 0
}

complete -F _structured_docs_synth_completion structured-docs-synth
''',
    'zsh_completion': '''
#compdef structured-docs-synth

_structured_docs_synth() {
    local commands
    commands=(
        'generate:Generate synthetic documents'
        'validate:Validate templates and configurations'
        'export:Export generated documents'
        'status:Show system status'
        'config:Manage configuration'
    )
    
    _describe 'commands' commands
}

_structured_docs_synth "$@"
''',
    'environment_setup': '''
#!/bin/bash
# Environment setup for structured-docs-synth

export STRUCTURED_DOCS_SYNTH_HOME="$HOME/.structured_docs_synth"
export STRUCTURED_DOCS_SYNTH_CONFIG="$STRUCTURED_DOCS_SYNTH_HOME/config.yaml"
export PYTHONPATH="$PYTHONPATH:$STRUCTURED_DOCS_SYNTH_HOME"

# Add to PATH if CLI is installed
if [ -d "$STRUCTURED_DOCS_SYNTH_HOME/bin" ]; then
    export PATH="$PATH:$STRUCTURED_DOCS_SYNTH_HOME/bin"
fi

# Load completions
if [ -f "$STRUCTURED_DOCS_SYNTH_HOME/shell/bash_completion" ]; then
    source "$STRUCTURED_DOCS_SYNTH_HOME/shell/bash_completion"
fi
'''
}

# Systemd service template
SYSTEMD_SERVICE = '''
[Unit]
Description=Structured Documents Synthetic Data Generation Service
After=network.target

[Service]
Type=simple
User={user}
Group={group}
WorkingDirectory={working_dir}
Environment=PYTHONPATH={python_path}
Environment=STRUCTURED_DOCS_SYNTH_CONFIG={config_path}
ExecStart={python_path}/bin/python -m structured_docs_synth.api
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
'''


class EnvironmentConfigurator:
    """Comprehensive environment configuration system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**ENVIRONMENT_CONFIG, **(config or {})}
        self.setup_results = {}
        self.generated_secrets = {}
        
        # Expand paths
        for key, value in self.config.items():
            if isinstance(value, (str, Path)) and '~' in str(value):
                self.config[key] = Path(str(value).replace('~', str(Path.home())))
    
    async def setup_complete_environment(self, 
                                       environment_type: str = 'production') -> Dict[str, Any]:
        """
        Set up complete environment configuration.
        
        Args:
            environment_type: Type of environment (development, production, testing)
        
        Returns:
            Setup results
        """
        print("🔧 Starting complete environment setup...")
        
        try:
            start_time = datetime.now()
            
            setup_results = {
                'timestamp': start_time.isoformat(),
                'environment_type': environment_type,
                'platform': platform.system(),
                'total_steps': 8,
                'completed_steps': 0,
                'failed_steps': 0,
                'results': {}
            }
            
            # Step 1: Create directory structure
            print("📁 Creating directory structure...")
            dir_result = await self._create_directory_structure()
            setup_results['results']['directories'] = dir_result
            self._update_step_count(setup_results, dir_result['success'])
            
            # Step 2: Generate configuration files
            print("⚙️  Generating configuration files...")
            config_result = await self._generate_configuration_files(environment_type)
            setup_results['results']['configuration'] = config_result
            self._update_step_count(setup_results, config_result['success'])
            
            # Step 3: Set up logging
            print("📝 Setting up logging configuration...")
            logging_result = await self._setup_logging_configuration()
            setup_results['results']['logging'] = logging_result
            self._update_step_count(setup_results, logging_result['success'])
            
            # Step 4: Configure environment variables
            print("🌍 Configuring environment variables...")
            env_result = await self._setup_environment_variables()
            setup_results['results']['environment_variables'] = env_result
            self._update_step_count(setup_results, env_result['success'])
            
            # Step 5: Set up shell integration
            if self.config['setup_shell_integration']:
                print("🐚 Setting up shell integration...")
                shell_result = await self._setup_shell_integration()
                setup_results['results']['shell_integration'] = shell_result
                self._update_step_count(setup_results, shell_result['success'])
            else:
                setup_results['results']['shell_integration'] = {'skipped': True}
                setup_results['completed_steps'] += 1
            
            # Step 6: Configure services
            if self.config['create_systemd_service']:
                print("🎯 Creating systemd service...")
                service_result = await self._create_systemd_service()
                setup_results['results']['systemd_service'] = service_result
                self._update_step_count(setup_results, service_result['success'])
            else:
                setup_results['results']['systemd_service'] = {'skipped': True}
                setup_results['completed_steps'] += 1
            
            # Step 7: Set secure permissions
            if self.config['secure_permissions']:
                print("🔒 Setting secure permissions...")
                permissions_result = await self._set_secure_permissions()
                setup_results['results']['permissions'] = permissions_result
                self._update_step_count(setup_results, permissions_result['success'])
            else:
                setup_results['results']['permissions'] = {'skipped': True}
                setup_results['completed_steps'] += 1
            
            # Step 8: Validate configuration
            if self.config['validate_config']:
                print("✅ Validating configuration...")
                validation_result = await self._validate_configuration()
                setup_results['results']['validation'] = validation_result
                self._update_step_count(setup_results, validation_result['success'])
            else:
                setup_results['results']['validation'] = {'skipped': True}
                setup_results['completed_steps'] += 1
            
            # Calculate setup time
            end_time = datetime.now()
            setup_results['setup_time_seconds'] = (end_time - start_time).total_seconds()
            
            # Generate setup summary
            await self._generate_setup_summary(setup_results)
            
            print(f"🔧 Environment setup completed:")
            print(f"   Completed steps: {setup_results['completed_steps']}")
            print(f"   Failed steps: {setup_results['failed_steps']}")
            print(f"   Setup time: {setup_results['setup_time_seconds']:.1f}s")
            
            return setup_results
            
        except Exception as e:
            print(f"❌ Environment setup failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def update_configuration(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update existing configuration.
        
        Args:
            config_updates: Configuration updates to apply
        
        Returns:
            Update results
        """
        print("⚙️  Updating configuration...")
        
        try:
            config_file = self.config['config_dir'] / 'config.yaml'
            
            if not config_file.exists():
                return {
                    'success': False,
                    'error': 'Configuration file not found'
                }
            
            # Load existing configuration
            async with aiofiles.open(config_file, 'r') as f:
                existing_config = yaml.safe_load(await f.read())
            
            # Apply updates
            updated_config = self._deep_merge_dict(existing_config, config_updates)
            
            # Backup existing configuration
            backup_file = config_file.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml')
            shutil.copy2(config_file, backup_file)
            
            # Save updated configuration
            async with aiofiles.open(config_file, 'w') as f:
                await f.write(yaml.dump(updated_config, default_flow_style=False))
            
            # Validate updated configuration
            validation_result = await self._validate_configuration()
            
            print(f"✅ Configuration updated successfully")
            
            return {
                'success': True,
                'backup_file': str(backup_file),
                'updates_applied': len(config_updates),
                'validation': validation_result
            }
            
        except Exception as e:
            print(f"❌ Configuration update failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def backup_configuration(self, backup_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create backup of current configuration.
        
        Args:
            backup_name: Custom backup name
        
        Returns:
            Backup results
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = backup_name or f"config_backup_{timestamp}"
        
        print(f"💾 Creating configuration backup: {backup_name}")
        
        try:
            backup_dir = self.config['backup_dir'] / backup_name
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            config_files = [
                'config.yaml',
                'logging_config.yaml',
                'api_config.yaml'
            ]
            
            backup_results = {
                'backup_name': backup_name,
                'backup_path': str(backup_dir),
                'timestamp': datetime.now().isoformat(),
                'files_backed_up': 0,
                'total_size_mb': 0.0
            }
            
            for config_file in config_files:
                source_file = self.config['config_dir'] / config_file
                if source_file.exists():
                    dest_file = backup_dir / config_file
                    shutil.copy2(source_file, dest_file)
                    
                    backup_results['files_backed_up'] += 1
                    backup_results['total_size_mb'] += dest_file.stat().st_size / (1024 * 1024)
            
            # Create backup metadata
            metadata = {
                'backup_name': backup_name,
                'created_at': datetime.now().isoformat(),
                'platform': platform.system(),
                'files_backed_up': backup_results['files_backed_up']
            }
            
            metadata_file = backup_dir / 'backup_metadata.json'
            async with aiofiles.open(metadata_file, 'w') as f:
                await f.write(json.dumps(metadata, indent=2))
            
            print(f"✅ Configuration backup created: {backup_dir}")
            
            return backup_results
            
        except Exception as e:
            print(f"❌ Configuration backup failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def restore_configuration(self, backup_path: Path) -> Dict[str, Any]:
        """
        Restore configuration from backup.
        
        Args:
            backup_path: Path to backup directory
        
        Returns:
            Restoration results
        """
        print(f"♻️  Restoring configuration from: {backup_path}")
        
        try:
            if not backup_path.exists():
                return {
                    'success': False,
                    'error': f'Backup path not found: {backup_path}'
                }
            
            # Create safety backup before restoration
            safety_backup = await self.backup_configuration(f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            restore_results = {
                'backup_path': str(backup_path),
                'timestamp': datetime.now().isoformat(),
                'safety_backup': safety_backup.get('backup_name'),
                'files_restored': 0
            }
            
            # Restore configuration files
            for backup_file in backup_path.glob('*.yaml'):
                if backup_file.name != 'backup_metadata.json':
                    dest_file = self.config['config_dir'] / backup_file.name
                    shutil.copy2(backup_file, dest_file)
                    restore_results['files_restored'] += 1
            
            # Validate restored configuration
            validation_result = await self._validate_configuration()
            restore_results['validation'] = validation_result
            
            print(f"✅ Configuration restored successfully")
            
            return restore_results
            
        except Exception as e:
            print(f"❌ Configuration restoration failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    # Private methods
    
    async def _create_directory_structure(self) -> Dict[str, Any]:
        """Create required directory structure"""
        try:
            directories = [
                self.config['config_dir'],
                self.config['log_dir'],
                self.config['data_dir'],
                self.config['models_dir'],
                self.config['cache_dir'],
                self.config['backup_dir'],
                self.config['temp_dir'],
                self.config['config_dir'] / 'shell',
                self.config['config_dir'] / 'templates',
                self.config['config_dir'] / 'certificates'
            ]
            
            created_dirs = 0
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                if self.config['secure_permissions']:
                    directory.chmod(0o750)
                created_dirs += 1
            
            return {
                'success': True,
                'directories_created': created_dirs,
                'directories': [str(d) for d in directories]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _generate_configuration_files(self, environment_type: str) -> Dict[str, Any]:
        """Generate configuration files"""
        try:
            config_files_created = 0
            
            # Generate main configuration
            main_config = CONFIG_TEMPLATES['main_config'].copy()
            main_config['application']['environment'] = environment_type
            
            # Expand paths in configuration
            for section in main_config.values():
                if isinstance(section, dict):
                    for key, value in section.items():
                        if isinstance(value, str) and '~' in value:
                            section[key] = str(Path(value).expanduser())
            
            config_file = self.config['config_dir'] / 'config.yaml'
            async with aiofiles.open(config_file, 'w') as f:
                await f.write(yaml.dump(main_config, default_flow_style=False))
            config_files_created += 1
            
            # Generate logging configuration
            logging_config = CONFIG_TEMPLATES['logging_config'].copy()
            
            # Expand paths in logging configuration
            for handler in logging_config['handlers'].values():
                if 'filename' in handler and '~' in handler['filename']:
                    handler['filename'] = str(Path(handler['filename']).expanduser())
            
            logging_config_file = self.config['config_dir'] / 'logging_config.yaml'
            async with aiofiles.open(logging_config_file, 'w') as f:
                await f.write(yaml.dump(logging_config, default_flow_style=False))
            config_files_created += 1
            
            # Generate API configuration
            api_config = CONFIG_TEMPLATES['api_config'].copy()
            
            # Generate JWT secret
            jwt_secret = secrets.token_urlsafe(64)
            api_config['authentication']['jwt_secret_key'] = jwt_secret
            self.generated_secrets['jwt_secret'] = jwt_secret
            
            api_config_file = self.config['config_dir'] / 'api_config.yaml'
            async with aiofiles.open(api_config_file, 'w') as f:
                await f.write(yaml.dump(api_config, default_flow_style=False))
            config_files_created += 1
            
            # Generate Docker configuration
            docker_config = CONFIG_TEMPLATES['docker_config'].copy()
            docker_compose_file = self.config['config_dir'] / 'docker-compose.yaml'
            async with aiofiles.open(docker_compose_file, 'w') as f:
                await f.write(yaml.dump(docker_config, default_flow_style=False))
            config_files_created += 1
            
            return {
                'success': True,
                'config_files_created': config_files_created,
                'generated_secrets': list(self.generated_secrets.keys())
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _setup_logging_configuration(self) -> Dict[str, Any]:
        """Set up logging configuration"""
        try:
            # Create log directories
            log_dirs = [
                self.config['log_dir'] / 'application',
                self.config['log_dir'] / 'audit',
                self.config['log_dir'] / 'security',
                self.config['log_dir'] / 'performance'
            ]
            
            for log_dir in log_dirs:
                log_dir.mkdir(parents=True, exist_ok=True)
                if self.config['secure_permissions']:
                    log_dir.chmod(0o750)
            
            # Create log rotation configuration
            logrotate_config = f"""
{self.config['log_dir']}/*.log {{
    daily
    missingok
    rotate {self.config['backup_count']}
    compress
    notifempty
    create 644
    size {self.config['max_log_size_mb']}M
}}
"""
            
            logrotate_file = self.config['config_dir'] / 'logrotate.conf'
            async with aiofiles.open(logrotate_file, 'w') as f:
                await f.write(logrotate_config)
            
            return {
                'success': True,
                'log_directories_created': len(log_dirs),
                'logrotate_configured': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _setup_environment_variables(self) -> Dict[str, Any]:
        """Set up environment variables"""
        try:
            # Expand paths in environment variables
            expanded_env_vars = {}
            for key, value in ENVIRONMENT_VARIABLES.items():
                if '~' in value:
                    expanded_env_vars[key] = str(Path(value).expanduser())
                else:
                    expanded_env_vars[key] = value
            
            # Create environment file
            env_file = self.config['config_dir'] / '.env'
            env_content = '\n'.join([f'{key}="{value}"' for key, value in expanded_env_vars.items()])
            
            async with aiofiles.open(env_file, 'w') as f:
                await f.write(env_content)
            
            # Create shell environment setup script
            shell_env_script = SHELL_SCRIPTS['environment_setup']
            shell_env_file = self.config['config_dir'] / 'shell' / 'env_setup.sh'
            
            async with aiofiles.open(shell_env_file, 'w') as f:
                await f.write(shell_env_script)
            
            shell_env_file.chmod(0o755)
            
            return {
                'success': True,
                'environment_variables_set': len(expanded_env_vars),
                'env_file_created': True,
                'shell_script_created': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _setup_shell_integration(self) -> Dict[str, Any]:
        """Set up shell integration scripts"""
        try:
            shell_files_created = 0
            
            # Create bash completion
            bash_completion_file = self.config['config_dir'] / 'shell' / 'bash_completion'
            async with aiofiles.open(bash_completion_file, 'w') as f:
                await f.write(SHELL_SCRIPTS['bash_completion'])
            shell_files_created += 1
            
            # Create zsh completion
            zsh_completion_file = self.config['config_dir'] / 'shell' / '_structured_docs_synth'
            async with aiofiles.open(zsh_completion_file, 'w') as f:
                await f.write(SHELL_SCRIPTS['zsh_completion'])
            shell_files_created += 1
            
            # Create installation instructions
            install_instructions = f"""
# Shell Integration Installation Instructions

## Bash
Add the following line to your ~/.bashrc:
source {self.config['config_dir']}/shell/bash_completion

## Zsh  
Add the completion file to your fpath:
fpath=({self.config['config_dir']}/shell $fpath)

## Environment Setup
Source the environment setup script:
source {self.config['config_dir']}/shell/env_setup.sh
"""
            
            instructions_file = self.config['config_dir'] / 'shell' / 'INSTALL.md'
            async with aiofiles.open(instructions_file, 'w') as f:
                await f.write(install_instructions)
            
            return {
                'success': True,
                'shell_files_created': shell_files_created,
                'instructions_file': str(instructions_file)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _create_systemd_service(self) -> Dict[str, Any]:
        """Create systemd service file"""
        try:
            import pwd
            import grp
            
            # Get current user information
            user_info = pwd.getpwuid(os.getuid())
            group_info = grp.getgrgid(user_info.pw_gid)
            
            # Generate service file content
            service_content = SYSTEMD_SERVICE.format(
                user=user_info.pw_name,
                group=group_info.gr_name,
                working_dir=self.config['config_dir'],
                python_path=str(Path(sys.executable).parent.parent),
                config_path=self.config['config_dir'] / 'config.yaml'
            )
            
            # Save service file
            service_file = self.config['config_dir'] / 'structured-docs-synth.service'
            async with aiofiles.open(service_file, 'w') as f:
                await f.write(service_content)
            
            # Create installation script
            install_script = f"""#!/bin/bash
# Install systemd service for structured-docs-synth

sudo cp {service_file} /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable structured-docs-synth
sudo systemctl start structured-docs-synth

echo "Service installed and started"
echo "Use 'sudo systemctl status structured-docs-synth' to check status"
"""
            
            install_script_file = self.config['config_dir'] / 'install_service.sh'
            async with aiofiles.open(install_script_file, 'w') as f:
                await f.write(install_script)
            
            install_script_file.chmod(0o755)
            
            return {
                'success': True,
                'service_file': str(service_file),
                'install_script': str(install_script_file),
                'note': 'Run install_service.sh with sudo to install the service'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _set_secure_permissions(self) -> Dict[str, Any]:
        """Set secure file and directory permissions"""
        try:
            # Set directory permissions
            directories_secured = 0
            for directory in [self.config['config_dir'], self.config['log_dir'], 
                            self.config['data_dir'], self.config['backup_dir']]:
                if directory.exists():
                    directory.chmod(0o750)
                    directories_secured += 1
            
            # Set configuration file permissions
            config_files_secured = 0
            for config_file in self.config['config_dir'].glob('*.yaml'):
                config_file.chmod(0o640)
                config_files_secured += 1
            
            # Set secret files permissions
            secret_files = ['.env', 'api_config.yaml']
            for secret_file in secret_files:
                file_path = self.config['config_dir'] / secret_file
                if file_path.exists():
                    file_path.chmod(0o600)
            
            return {
                'success': True,
                'directories_secured': directories_secured,
                'config_files_secured': config_files_secured,
                'secret_files_secured': len(secret_files)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _validate_configuration(self) -> Dict[str, Any]:
        """Validate generated configuration"""
        try:
            validation_results = {
                'config_files_valid': 0,
                'config_files_invalid': 0,
                'issues': []
            }
            
            # Validate main configuration
            config_file = self.config['config_dir'] / 'config.yaml'
            if config_file.exists():
                try:
                    async with aiofiles.open(config_file, 'r') as f:
                        yaml.safe_load(await f.read())
                    validation_results['config_files_valid'] += 1
                except yaml.YAMLError as e:
                    validation_results['config_files_invalid'] += 1
                    validation_results['issues'].append(f'config.yaml: {str(e)}')
            
            # Validate logging configuration
            logging_config_file = self.config['config_dir'] / 'logging_config.yaml'
            if logging_config_file.exists():
                try:
                    async with aiofiles.open(logging_config_file, 'r') as f:
                        yaml.safe_load(await f.read())
                    validation_results['config_files_valid'] += 1
                except yaml.YAMLError as e:
                    validation_results['config_files_invalid'] += 1
                    validation_results['issues'].append(f'logging_config.yaml: {str(e)}')
            
            # Check directory structure
            required_dirs = [self.config['log_dir'], self.config['data_dir'], 
                           self.config['models_dir'], self.config['cache_dir']]
            
            for directory in required_dirs:
                if not directory.exists():
                    validation_results['issues'].append(f'Missing directory: {directory}')
            
            validation_results['valid'] = validation_results['config_files_invalid'] == 0 and len(validation_results['issues']) == 0
            
            return {
                'success': True,
                **validation_results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _update_step_count(self, results: Dict[str, Any], success: bool):
        """Update step counters"""
        if success:
            results['completed_steps'] += 1
        else:
            results['failed_steps'] += 1
    
    def _deep_merge_dict(self, base_dict: Dict, update_dict: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base_dict.copy()
        
        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dict(result[key], value)
            else:
                result[key] = value
        
        return result
    
    async def _generate_setup_summary(self, setup_results: Dict[str, Any]):
        """Generate setup summary report"""
        summary_file = self.config['config_dir'] / f"setup_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        async with aiofiles.open(summary_file, 'w') as f:
            await f.write(json.dumps(setup_results, indent=2, default=str))
        
        print(f"📋 Setup summary saved: {summary_file}")


async def main():
    """
    Main environment configuration script function.
    """
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description='Configure environment for structured document synthesis'
    )
    parser.add_argument(
        'action',
        choices=['setup', 'update', 'backup', 'restore', 'validate'],
        help='Action to perform'
    )
    parser.add_argument(
        '--environment',
        choices=['development', 'production', 'testing'],
        default='production',
        help='Environment type to configure'
    )
    parser.add_argument(
        '--config-updates',
        type=str,
        help='JSON string with configuration updates'
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
        '--shell-integration',
        action='store_true',
        help='Enable shell integration setup'
    )
    parser.add_argument(
        '--systemd-service',
        action='store_true',
        help='Create systemd service file'
    )
    parser.add_argument(
        '--config-dir',
        type=Path,
        help='Custom configuration directory'
    )
    
    args = parser.parse_args()
    
    # Configure environment setup
    config = ENVIRONMENT_CONFIG.copy()
    if args.config_dir:
        config['config_dir'] = args.config_dir
    if args.shell_integration:
        config['setup_shell_integration'] = True
    if args.systemd_service:
        config['create_systemd_service'] = True
    
    configurator = EnvironmentConfigurator(config=config)
    
    if args.action == 'setup':
        result = await configurator.setup_complete_environment(args.environment)
        
        if result.get('success', True):
            print(f"\n✅ Environment setup completed")
            print(f"🔧 Completed steps: {result['completed_steps']}")
            print(f"❌ Failed steps: {result['failed_steps']}")
            
            if result['failed_steps'] > 0:
                return 1
        else:
            print(f"\n❌ Environment setup failed: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.action == 'update':
        if not args.config_updates:
            print("❌ Configuration updates required")
            return 1
        
        try:
            config_updates = json.loads(args.config_updates)
        except json.JSONDecodeError:
            print("❌ Invalid JSON in config updates")
            return 1
        
        result = await configurator.update_configuration(config_updates)
        
        if result['success']:
            print(f"✅ Configuration updated successfully")
            print(f"📝 Updates applied: {result['updates_applied']}")
        else:
            print(f"❌ Configuration update failed: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.action == 'backup':
        result = await configurator.backup_configuration(args.backup_name)
        
        if result.get('success', True):
            print(f"✅ Configuration backup completed")
            print(f"📁 Backup path: {result['backup_path']}")
            print(f"📄 Files backed up: {result['files_backed_up']}")
        else:
            print(f"❌ Configuration backup failed: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.action == 'restore':
        if not args.backup_path:
            print("❌ Backup path required for restoration")
            return 1
        
        result = await configurator.restore_configuration(args.backup_path)
        
        if result.get('success', True):
            print(f"✅ Configuration restored successfully")
            print(f"📄 Files restored: {result['files_restored']}")
        else:
            print(f"❌ Configuration restoration failed: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.action == 'validate':
        result = await configurator._validate_configuration()
        
        if result['success'] and result['valid']:
            print(f"✅ Configuration validation passed")
            print(f"📄 Valid config files: {result['config_files_valid']}")
        else:
            print(f"❌ Configuration validation failed")
            if result.get('issues'):
                for issue in result['issues']:
                    print(f"   - {issue}")
            return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(asyncio.run(main()))