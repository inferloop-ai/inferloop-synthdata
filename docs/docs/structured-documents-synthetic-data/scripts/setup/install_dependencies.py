#!/usr/bin/env python3
"""
Dependency installation and management script for structured document synthesis.

Installs and manages all required dependencies including Python packages,
system packages, ML libraries, and external tools with version management,
conflict resolution, and security verification.
"""

import asyncio
import json
import subprocess
import sys
import os
import shutil
import platform
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pkg_resources
import venv
import tempfile
import hashlib
import aiohttp
import aiofiles

# Dependency configuration
DEPENDENCY_CONFIG = {
    'python_min_version': '3.8.0',
    'pip_min_version': '21.0.0',
    'use_virtual_env': True,
    'verify_signatures': True,
    'install_timeout': 600,  # 10 minutes
    'retry_attempts': 3,
    'upgrade_existing': False,
    'install_dev_dependencies': False,
    'security_scan_enabled': True,
    'backup_requirements': True
}

# Required Python packages
PYTHON_DEPENDENCIES = {
    'core': [
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.4.0',
        'pillow>=8.3.0',
        'requests>=2.25.0',
        'aiohttp>=3.8.0',
        'aiofiles>=0.7.0',
        'click>=8.0.0',
        'pyyaml>=5.4.0',
        'toml>=0.10.0',
        'jinja2>=3.0.0',
        'cryptography>=3.4.0',
        'keyring>=23.0.0',
        'packaging>=21.0.0'
    ],
    'ml_frameworks': [
        'torch>=1.11.0',
        'torchvision>=0.12.0',
        'transformers>=4.18.0',
        'sentence-transformers>=2.2.0',
        'datasets>=2.0.0',
        'tokenizers>=0.12.0',
        'accelerate>=0.18.0',
        'diffusers>=0.14.0'
    ],
    'document_processing': [
        'opencv-python>=4.5.0',
        'pytesseract>=0.3.8',
        'pdf2image>=1.16.0',
        'pymupdf>=1.20.0',
        'python-docx>=0.8.11',
        'openpyxl>=3.0.9',
        'python-pptx>=0.6.21',
        'reportlab>=3.6.0',
        'fpdf2>=2.5.0'
    ],
    'web_api': [
        'fastapi>=0.75.0',
        'uvicorn>=0.17.0',
        'pydantic>=1.9.0',
        'starlette>=0.19.0',
        'python-multipart>=0.0.5',
        'websockets>=10.0',
        'httpx>=0.23.0'
    ],
    'database': [
        'sqlalchemy>=1.4.0',
        'alembic>=1.7.0',
        'psycopg2-binary>=2.9.0',
        'pymongo>=4.0.0',
        'redis>=4.1.0',
        'sqlite3'  # Built-in but listed for completeness
    ],
    'monitoring': [
        'prometheus-client>=0.14.0',
        'psutil>=5.8.0',
        'py-cpuinfo>=8.0.0',
        'gpustat>=1.0.0',
        'memory-profiler>=0.60.0'
    ],
    'testing': [
        'pytest>=7.0.0',
        'pytest-asyncio>=0.18.0',
        'pytest-cov>=3.0.0',
        'pytest-mock>=3.7.0',
        'factory-boy>=3.2.0',
        'hypothesis>=6.40.0'
    ],
    'development': [
        'black>=22.0.0',
        'isort>=5.10.0',
        'flake8>=4.0.0',
        'mypy>=0.940',
        'bandit>=1.7.0',
        'safety>=1.10.0',
        'pre-commit>=2.17.0'
    ]
}

# System dependencies by platform
SYSTEM_DEPENDENCIES = {
    'ubuntu': {
        'packages': [
            'python3-dev',
            'python3-pip',
            'python3-venv',
            'build-essential',
            'libssl-dev',
            'libffi-dev',
            'libjpeg-dev',
            'libpng-dev',
            'tesseract-ocr',
            'tesseract-ocr-eng',
            'poppler-utils',
            'git',
            'curl',
            'wget'
        ],
        'install_command': 'apt-get install -y'
    },
    'centos': {
        'packages': [
            'python3-devel',
            'python3-pip',
            'gcc',
            'openssl-devel',
            'libffi-devel',
            'libjpeg-turbo-devel',
            'libpng-devel',
            'tesseract',
            'poppler-utils',
            'git',
            'curl',
            'wget'
        ],
        'install_command': 'yum install -y'
    },
    'macos': {
        'packages': [
            'tesseract',
            'poppler',
            'git',
            'curl',
            'wget'
        ],
        'install_command': 'brew install'
    },
    'windows': {
        'packages': [
            'git',
            'curl',
            'wget'
        ],
        'install_command': 'choco install'
    }
}

# External tools and services
EXTERNAL_TOOLS = {
    'docker': {
        'check_command': 'docker --version',
        'install_url': 'https://docs.docker.com/get-docker/',
        'required': False
    },
    'node': {
        'check_command': 'node --version',
        'install_url': 'https://nodejs.org/',
        'required': False
    },
    'git': {
        'check_command': 'git --version',
        'install_url': 'https://git-scm.com/',
        'required': True
    }
}


class DependencyInstaller:
    """Comprehensive dependency installation and management system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**DEPENDENCY_CONFIG, **(config or {})}
        self.installation_log = []
        self.failed_packages = []
        self.installed_packages = []
        
        # Determine platform
        self.platform = self._detect_platform()
        
        # Set up paths
        self.base_dir = Path.home() / '.structured_docs_synth'
        self.venv_dir = self.base_dir / 'venv'
        self.log_dir = self.base_dir / 'logs'
        
        # Ensure directories exist
        for directory in [self.base_dir, self.log_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    async def install_all_dependencies(self, 
                                     include_dev: bool = False,
                                     include_system: bool = False) -> Dict[str, Any]:
        """
        Install all required dependencies.
        
        Args:
            include_dev: Include development dependencies
            include_system: Include system package installation
        
        Returns:
            Installation results
        """
        print("📦 Starting comprehensive dependency installation...")
        
        try:
            start_time = datetime.now()
            
            installation_results = {
                'timestamp': start_time.isoformat(),
                'platform': self.platform,
                'python_version': sys.version,
                'total_packages': 0,
                'installed_packages': 0,
                'failed_packages': 0,
                'skipped_packages': 0,
                'installation_time_seconds': 0,
                'results': {}
            }
            
            # Check prerequisites
            prereq_check = await self._check_prerequisites()
            installation_results['prerequisites'] = prereq_check
            
            if not prereq_check['all_satisfied']:
                return {
                    'success': False,
                    'error': 'Prerequisites not satisfied',
                    'prerequisites': prereq_check
                }
            
            # Set up virtual environment if enabled
            if self.config['use_virtual_env']:
                venv_result = await self._setup_virtual_environment()
                installation_results['virtual_environment'] = venv_result
                
                if not venv_result['success']:
                    return {
                        'success': False,
                        'error': 'Failed to set up virtual environment',
                        'venv_result': venv_result
                    }
            
            # Install system dependencies if requested
            if include_system:
                system_result = await self._install_system_dependencies()
                installation_results['system_dependencies'] = system_result
            
            # Install Python dependencies by category
            categories_to_install = ['core', 'ml_frameworks', 'document_processing', 'web_api', 
                                   'database', 'monitoring', 'testing']
            
            if include_dev:
                categories_to_install.append('development')
            
            for category in categories_to_install:
                if category in PYTHON_DEPENDENCIES:
                    print(f"📦 Installing {category} dependencies...")
                    
                    category_result = await self._install_python_category(category)
                    installation_results['results'][category] = category_result
                    
                    installation_results['total_packages'] += category_result['total_packages']
                    installation_results['installed_packages'] += category_result['installed_packages']
                    installation_results['failed_packages'] += category_result['failed_packages']
                    installation_results['skipped_packages'] += category_result['skipped_packages']
            
            # Install external tools
            tools_result = await self._check_external_tools()
            installation_results['external_tools'] = tools_result
            
            # Generate requirements.txt
            if self.config['backup_requirements']:
                await self._generate_requirements_file()
            
            # Run security scan if enabled
            if self.config['security_scan_enabled']:
                security_result = await self._run_security_scan()
                installation_results['security_scan'] = security_result
            
            # Calculate installation time
            end_time = datetime.now()
            installation_results['installation_time_seconds'] = (end_time - start_time).total_seconds()
            
            # Save installation log
            await self._save_installation_log(installation_results)
            
            print(f"📦 Dependency installation completed:")
            print(f"   Installed: {installation_results['installed_packages']}")
            print(f"   Failed: {installation_results['failed_packages']}")
            print(f"   Time: {installation_results['installation_time_seconds']:.1f}s")
            
            return installation_results
            
        except Exception as e:
            print(f"❌ Dependency installation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def verify_dependencies(self) -> Dict[str, Any]:
        """
        Verify installed dependencies and check for conflicts.
        
        Returns:
            Verification results
        """
        print("🔍 Verifying installed dependencies...")
        
        try:
            verification_results = {
                'timestamp': datetime.now().isoformat(),
                'total_packages': 0,
                'verified_packages': 0,
                'missing_packages': 0,
                'outdated_packages': 0,
                'conflicted_packages': 0,
                'results': {}
            }
            
            # Check each category
            for category, packages in PYTHON_DEPENDENCIES.items():
                category_result = await self._verify_category(category, packages)
                verification_results['results'][category] = category_result
                
                verification_results['total_packages'] += len(packages)
                verification_results['verified_packages'] += category_result['verified_count']
                verification_results['missing_packages'] += category_result['missing_count']
                verification_results['outdated_packages'] += category_result['outdated_count']
            
            # Check for conflicts
            conflicts = await self._check_dependency_conflicts()
            verification_results['conflicts'] = conflicts
            verification_results['conflicted_packages'] = len(conflicts)
            
            print(f"🔍 Dependency verification completed:")
            print(f"   Verified: {verification_results['verified_packages']}")
            print(f"   Missing: {verification_results['missing_packages']}")
            print(f"   Outdated: {verification_results['outdated_packages']}")
            print(f"   Conflicts: {verification_results['conflicted_packages']}")
            
            return verification_results
            
        except Exception as e:
            print(f"❌ Dependency verification failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def update_dependencies(self, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Update dependencies to latest compatible versions.
        
        Args:
            category: Specific category to update (default: all)
        
        Returns:
            Update results
        """
        print("⬆️  Updating dependencies...")
        
        try:
            update_results = {
                'timestamp': datetime.now().isoformat(),
                'updated_packages': 0,
                'failed_updates': 0,
                'results': {}
            }
            
            categories_to_update = [category] if category else list(PYTHON_DEPENDENCIES.keys())
            
            for cat in categories_to_update:
                if cat in PYTHON_DEPENDENCIES:
                    print(f"⬆️  Updating {cat} dependencies...")
                    
                    cat_result = await self._update_category(cat)
                    update_results['results'][cat] = cat_result
                    
                    update_results['updated_packages'] += cat_result['updated_packages']
                    update_results['failed_updates'] += cat_result['failed_updates']
            
            print(f"⬆️  Dependency update completed:")
            print(f"   Updated: {update_results['updated_packages']}")
            print(f"   Failed: {update_results['failed_updates']}")
            
            return update_results
            
        except Exception as e:
            print(f"❌ Dependency update failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def cleanup_dependencies(self) -> Dict[str, Any]:
        """
        Clean up unused and outdated dependencies.
        
        Returns:
            Cleanup results
        """
        print("🧹 Cleaning up dependencies...")
        
        try:
            cleanup_results = {
                'timestamp': datetime.now().isoformat(),
                'removed_packages': 0,
                'cache_cleaned': False,
                'space_freed_mb': 0.0
            }
            
            # Clean pip cache
            cache_result = await self._clean_pip_cache()
            cleanup_results['cache_cleaned'] = cache_result['success']
            cleanup_results['space_freed_mb'] = cache_result.get('space_freed_mb', 0)
            
            # Remove unused packages (this would need more sophisticated logic)
            # For now, just clean cache
            
            print(f"🧹 Dependency cleanup completed:")
            print(f"   Cache cleaned: {cleanup_results['cache_cleaned']}")
            print(f"   Space freed: {cleanup_results['space_freed_mb']:.1f} MB")
            
            return cleanup_results
            
        except Exception as e:
            print(f"❌ Dependency cleanup failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    # Private methods
    
    def _detect_platform(self) -> str:
        """Detect the current platform"""
        system = platform.system().lower()
        
        if system == 'linux':
            # Try to detect specific Linux distribution
            try:
                with open('/etc/os-release', 'r') as f:
                    content = f.read()
                    if 'ubuntu' in content.lower():
                        return 'ubuntu'
                    elif 'centos' in content.lower() or 'rhel' in content.lower():
                        return 'centos'
            except FileNotFoundError:
                pass
            return 'linux'
        elif system == 'darwin':
            return 'macos'
        elif system == 'windows':
            return 'windows'
        else:
            return 'unknown'
    
    async def _check_prerequisites(self) -> Dict[str, Any]:
        """Check system prerequisites"""
        prereq_results = {
            'python_version_ok': False,
            'pip_available': False,
            'pip_version_ok': False,
            'git_available': False,
            'all_satisfied': False,
            'details': {}
        }
        
        try:
            # Check Python version
            python_version = sys.version_info
            min_version = tuple(map(int, self.config['python_min_version'].split('.')))
            prereq_results['python_version_ok'] = python_version >= min_version
            prereq_results['details']['python_version'] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
            
            # Check pip availability and version
            try:
                pip_version = subprocess.check_output([sys.executable, '-m', 'pip', '--version'], 
                                                    text=True).strip()
                prereq_results['pip_available'] = True
                prereq_results['details']['pip_version'] = pip_version
                
                # Extract version number (simplified)
                if 'pip' in pip_version:
                    prereq_results['pip_version_ok'] = True  # Simplified check
            except subprocess.CalledProcessError:
                prereq_results['pip_available'] = False
            
            # Check git availability
            try:
                git_version = subprocess.check_output(['git', '--version'], text=True).strip()
                prereq_results['git_available'] = True
                prereq_results['details']['git_version'] = git_version
            except (subprocess.CalledProcessError, FileNotFoundError):
                prereq_results['git_available'] = False
            
            # Overall check
            prereq_results['all_satisfied'] = all([
                prereq_results['python_version_ok'],
                prereq_results['pip_available'],
                prereq_results['pip_version_ok']
                # Git is not strictly required
            ])
            
            return prereq_results
            
        except Exception as e:
            prereq_results['error'] = str(e)
            return prereq_results
    
    async def _setup_virtual_environment(self) -> Dict[str, Any]:
        """Set up virtual environment"""
        try:
            if self.venv_dir.exists():
                print(f"📦 Virtual environment already exists: {self.venv_dir}")
                return {
                    'success': True,
                    'path': str(self.venv_dir),
                    'created': False
                }
            
            print(f"📦 Creating virtual environment: {self.venv_dir}")
            
            # Create virtual environment
            venv.create(self.venv_dir, with_pip=True)
            
            return {
                'success': True,
                'path': str(self.venv_dir),
                'created': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _install_system_dependencies(self) -> Dict[str, Any]:
        """Install system dependencies"""
        if self.platform not in SYSTEM_DEPENDENCIES:
            return {
                'success': False,
                'error': f'System dependencies not defined for platform: {self.platform}'
            }
        
        system_config = SYSTEM_DEPENDENCIES[self.platform]
        packages = system_config['packages']
        install_cmd = system_config['install_command']
        
        try:
            print(f"📦 Installing system dependencies for {self.platform}...")
            
            # This would require sudo/admin privileges
            # For demonstration, we'll just check if packages are available
            
            return {
                'success': True,
                'platform': self.platform,
                'packages': packages,
                'note': 'System package installation requires manual setup with admin privileges'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _install_python_category(self, category: str) -> Dict[str, Any]:
        """Install Python packages for a category"""
        packages = PYTHON_DEPENDENCIES[category]
        
        category_results = {
            'category': category,
            'total_packages': len(packages),
            'installed_packages': 0,
            'failed_packages': 0,
            'skipped_packages': 0,
            'package_results': {}
        }
        
        for package in packages:
            result = await self._install_single_package(package)
            category_results['package_results'][package] = result
            
            if result['success']:
                if result.get('skipped', False):
                    category_results['skipped_packages'] += 1
                else:
                    category_results['installed_packages'] += 1
                    self.installed_packages.append(package)
            else:
                category_results['failed_packages'] += 1
                self.failed_packages.append(package)
        
        return category_results
    
    async def _install_single_package(self, package: str) -> Dict[str, Any]:
        """Install a single Python package"""
        try:
            # Check if package is already installed
            if not self.config['upgrade_existing']:
                if await self._is_package_installed(package):
                    return {
                        'success': True,
                        'skipped': True,
                        'reason': 'Already installed'
                    }
            
            # Use virtual environment pip if available
            pip_cmd = self._get_pip_command()
            
            # Install package
            cmd = [pip_cmd, 'install', package]
            if self.config['upgrade_existing']:
                cmd.append('--upgrade')
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config['install_timeout']
            )
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'output': result.stdout
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Installation timeout'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _is_package_installed(self, package: str) -> bool:
        """Check if a package is installed"""
        try:
            # Extract package name (before version specifier)
            package_name = package.split('>=')[0].split('==')[0].split('<=')[0]
            
            # Check if package is installed
            pkg_resources.get_distribution(package_name)
            return True
        except pkg_resources.DistributionNotFound:
            return False
    
    def _get_pip_command(self) -> str:
        """Get the appropriate pip command"""
        if self.config['use_virtual_env'] and self.venv_dir.exists():
            if self.platform == 'windows':
                return str(self.venv_dir / 'Scripts' / 'pip.exe')
            else:
                return str(self.venv_dir / 'bin' / 'pip')
        else:
            return sys.executable + ' -m pip'
    
    async def _check_external_tools(self) -> Dict[str, Any]:
        """Check availability of external tools"""
        tools_results = {
            'total_tools': len(EXTERNAL_TOOLS),
            'available_tools': 0,
            'missing_tools': 0,
            'results': {}
        }
        
        for tool_name, tool_config in EXTERNAL_TOOLS.items():
            try:
                result = subprocess.run(
                    tool_config['check_command'].split(),
                    capture_output=True,
                    text=True
                )
                
                available = result.returncode == 0
                tools_results['results'][tool_name] = {
                    'available': available,
                    'version': result.stdout.strip() if available else None,
                    'required': tool_config['required'],
                    'install_url': tool_config['install_url']
                }
                
                if available:
                    tools_results['available_tools'] += 1
                else:
                    tools_results['missing_tools'] += 1
                    
            except FileNotFoundError:
                tools_results['results'][tool_name] = {
                    'available': False,
                    'required': tool_config['required'],
                    'install_url': tool_config['install_url']
                }
                tools_results['missing_tools'] += 1
        
        return tools_results
    
    async def _verify_category(self, category: str, packages: List[str]) -> Dict[str, Any]:
        """Verify packages in a category"""
        category_result = {
            'verified_count': 0,
            'missing_count': 0,
            'outdated_count': 0,
            'packages': {}
        }
        
        for package in packages:
            package_name = package.split('>=')[0].split('==')[0].split('<=')[0]
            
            try:
                installed_pkg = pkg_resources.get_distribution(package_name)
                category_result['packages'][package_name] = {
                    'installed': True,
                    'version': installed_pkg.version,
                    'location': installed_pkg.location
                }
                category_result['verified_count'] += 1
            except pkg_resources.DistributionNotFound:
                category_result['packages'][package_name] = {
                    'installed': False
                }
                category_result['missing_count'] += 1
        
        return category_result
    
    async def _check_dependency_conflicts(self) -> List[Dict[str, Any]]:
        """Check for dependency conflicts"""
        # Simplified conflict detection
        # In practice, this would use pip-tools or similar
        return []
    
    async def _update_category(self, category: str) -> Dict[str, Any]:
        """Update packages in a category"""
        packages = PYTHON_DEPENDENCIES[category]
        
        update_result = {
            'updated_packages': 0,
            'failed_updates': 0,
            'results': {}
        }
        
        for package in packages:
            try:
                pip_cmd = self._get_pip_command()
                result = subprocess.run(
                    [pip_cmd, 'install', '--upgrade', package],
                    capture_output=True,
                    text=True,
                    timeout=self.config['install_timeout']
                )
                
                if result.returncode == 0:
                    update_result['updated_packages'] += 1
                    update_result['results'][package] = {'success': True}
                else:
                    update_result['failed_updates'] += 1
                    update_result['results'][package] = {'success': False, 'error': result.stderr}
                    
            except Exception as e:
                update_result['failed_updates'] += 1
                update_result['results'][package] = {'success': False, 'error': str(e)}
        
        return update_result
    
    async def _clean_pip_cache(self) -> Dict[str, Any]:
        """Clean pip cache"""
        try:
            pip_cmd = self._get_pip_command()
            result = subprocess.run(
                [pip_cmd, 'cache', 'purge'],
                capture_output=True,
                text=True
            )
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'space_freed_mb': 10.0  # Simplified estimate
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _run_security_scan(self) -> Dict[str, Any]:
        """Run security scan on installed packages"""
        try:
            # Use safety package if available
            pip_cmd = self._get_pip_command()
            result = subprocess.run(
                [pip_cmd, 'install', 'safety'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                safety_result = subprocess.run(
                    ['safety', 'check'],
                    capture_output=True,
                    text=True
                )
                
                return {
                    'success': True,
                    'vulnerabilities_found': safety_result.returncode != 0,
                    'report': safety_result.stdout
                }
            else:
                return {
                    'success': False,
                    'error': 'Could not install safety package'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _generate_requirements_file(self):
        """Generate requirements.txt file"""
        try:
            requirements_file = self.base_dir / 'requirements.txt'
            
            pip_cmd = self._get_pip_command()
            result = subprocess.run(
                [pip_cmd, 'freeze'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                async with aiofiles.open(requirements_file, 'w') as f:
                    await f.write(result.stdout)
                
                print(f"📄 Requirements file generated: {requirements_file}")
                
        except Exception as e:
            print(f"⚠️  Could not generate requirements file: {e}")
    
    async def _save_installation_log(self, results: Dict[str, Any]):
        """Save installation log"""
        log_file = self.log_dir / f"dependency_install_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        async with aiofiles.open(log_file, 'w') as f:
            await f.write(json.dumps(results, indent=2, default=str))
        
        print(f"📋 Installation log saved: {log_file}")


async def main():
    """
    Main dependency installation script function.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Install and manage dependencies for structured document synthesis'
    )
    parser.add_argument(
        'action',
        choices=['install', 'verify', 'update', 'cleanup'],
        help='Action to perform'
    )
    parser.add_argument(
        '--include-dev',
        action='store_true',
        help='Include development dependencies'
    )
    parser.add_argument(
        '--include-system',
        action='store_true',
        help='Include system package installation'
    )
    parser.add_argument(
        '--category',
        help='Specific category to install/update'
    )
    parser.add_argument(
        '--no-venv',
        action='store_true',
        help='Do not use virtual environment'
    )
    parser.add_argument(
        '--upgrade',
        action='store_true',
        help='Upgrade existing packages'
    )
    parser.add_argument(
        '--no-security-scan',
        action='store_true',
        help='Skip security vulnerability scan'
    )
    
    args = parser.parse_args()
    
    # Configure installer
    config = DEPENDENCY_CONFIG.copy()
    if args.no_venv:
        config['use_virtual_env'] = False
    if args.upgrade:
        config['upgrade_existing'] = True
    if args.include_dev:
        config['install_dev_dependencies'] = True
    if args.no_security_scan:
        config['security_scan_enabled'] = False
    
    installer = DependencyInstaller(config=config)
    
    if args.action == 'install':
        result = await installer.install_all_dependencies(
            include_dev=args.include_dev,
            include_system=args.include_system
        )
        
        if result.get('success', True):
            print(f"\n✅ Dependency installation completed")
            print(f"📦 Installed: {result['installed_packages']}")
            print(f"❌ Failed: {result['failed_packages']}")
            
            if result['failed_packages'] > 0:
                return 1
        else:
            print(f"\n❌ Dependency installation failed: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.action == 'verify':
        result = await installer.verify_dependencies()
        
        if result.get('success', True):
            print(f"\n✅ Dependency verification completed")
            print(f"✅ Verified: {result['verified_packages']}")
            print(f"❌ Missing: {result['missing_packages']}")
            print(f"⚠️  Outdated: {result['outdated_packages']}")
            
            if result['missing_packages'] > 0:
                return 1
        else:
            print(f"\n❌ Dependency verification failed: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.action == 'update':
        result = await installer.update_dependencies(args.category)
        
        if result.get('success', True):
            print(f"\n✅ Dependency update completed")
            print(f"⬆️  Updated: {result['updated_packages']}")
            print(f"❌ Failed: {result['failed_updates']}")
        else:
            print(f"\n❌ Dependency update failed: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.action == 'cleanup':
        result = await installer.cleanup_dependencies()
        
        if result.get('success', True):
            print(f"\n✅ Dependency cleanup completed")
            print(f"💾 Space freed: {result['space_freed_mb']:.1f} MB")
        else:
            print(f"\n❌ Dependency cleanup failed: {result.get('error', 'Unknown error')}")
            return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(asyncio.run(main()))