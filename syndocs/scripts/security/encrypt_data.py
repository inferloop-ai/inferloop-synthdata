#!/usr/bin/env python3
"""
Data encryption and decryption system for structured document synthesis.

Provides comprehensive encryption capabilities for sensitive data including
documents, database records, configuration files, and API keys with support
for multiple encryption algorithms and key management.
"""

import asyncio
import json
import base64
import hashlib
import secrets
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import sqlite3
import aiofiles
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import keyring

# Encryption configuration
ENCRYPTION_CONFIG = {
    'default_algorithm': 'fernet',  # fernet, aes-256-gcm, rsa
    'key_derivation_iterations': 100000,
    'key_size': 32,  # bytes
    'iv_size': 16,   # bytes for AES
    'rsa_key_size': 2048,
    'backup_enabled': True,
    'compression_enabled': True,
    'audit_logging': True
}

DEFAULT_KEY_DIR = Path.home() / '.structured_docs_synth' / 'keys'
DEFAULT_ENCRYPTED_DIR = Path.home() / '.structured_docs_synth' / 'encrypted'
DEFAULT_AUDIT_DIR = Path.home() / '.structured_docs_synth' / 'audit'


class DataEncryption:
    """Comprehensive data encryption and decryption system"""
    
    def __init__(self, key_dir: Optional[Path] = None, 
                 encrypted_dir: Optional[Path] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.key_dir = key_dir or DEFAULT_KEY_DIR
        self.encrypted_dir = encrypted_dir or DEFAULT_ENCRYPTED_DIR
        self.audit_dir = DEFAULT_AUDIT_DIR
        self.config = {**ENCRYPTION_CONFIG, **(config or {})}
        
        # Ensure directories exist
        for directory in [self.key_dir, self.encrypted_dir, self.audit_dir]:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        
        # Initialize audit log
        self.audit_log_file = self.audit_dir / 'encryption_audit.log'
        
        # Key cache for performance
        self._key_cache = {}
    
    async def encrypt_file(self, file_path: Path, output_path: Optional[Path] = None,
                          algorithm: Optional[str] = None, key_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Encrypt a file using specified algorithm.
        
        Args:
            file_path: Path to file to encrypt
            output_path: Output path for encrypted file
            algorithm: Encryption algorithm to use
            key_id: Specific key ID to use
        
        Returns:
            Encryption result
        """
        algorithm = algorithm or self.config['default_algorithm']
        
        print(f"🔐 Encrypting file: {file_path.name} using {algorithm}")
        
        try:
            if not file_path.exists():
                return {
                    'success': False,
                    'error': f'File not found: {file_path}'
                }
            
            # Generate output path if not provided
            if not output_path:
                output_path = self.encrypted_dir / f"{file_path.name}.encrypted"
            
            # Read file content
            async with aiofiles.open(file_path, 'rb') as f:
                file_content = await f.read()
            
            # Compress if enabled
            if self.config['compression_enabled']:
                import gzip
                file_content = gzip.compress(file_content)
            
            # Encrypt content
            if algorithm == 'fernet':
                encryption_result = await self._encrypt_fernet(file_content, key_id)
            elif algorithm == 'aes-256-gcm':
                encryption_result = await self._encrypt_aes_gcm(file_content, key_id)
            elif algorithm == 'rsa':
                encryption_result = await self._encrypt_rsa(file_content, key_id)
            else:
                return {
                    'success': False,
                    'error': f'Unsupported algorithm: {algorithm}'
                }
            
            if not encryption_result['success']:
                return encryption_result
            
            # Create encrypted file metadata
            metadata = {
                'original_file': str(file_path),
                'algorithm': algorithm,
                'key_id': encryption_result['key_id'],
                'encrypted_at': datetime.now().isoformat(),
                'original_size': len(file_content),
                'compressed': self.config['compression_enabled'],
                'checksum': hashlib.sha256(file_content).hexdigest()
            }
            
            # Save encrypted file with metadata
            encrypted_data = {
                'metadata': metadata,
                'data': base64.b64encode(encryption_result['encrypted_data']).decode('utf-8')
            }
            
            async with aiofiles.open(output_path, 'w') as f:
                await f.write(json.dumps(encrypted_data, indent=2))
            
            # Set secure permissions
            output_path.chmod(0o600)
            
            # Log encryption activity
            await self._log_encryption_activity('encrypt_file', {
                'file': str(file_path),
                'algorithm': algorithm,
                'key_id': encryption_result['key_id'],
                'output': str(output_path)
            })
            
            print(f"✅ File encrypted: {output_path}")
            
            return {
                'success': True,
                'input_file': str(file_path),
                'output_file': str(output_path),
                'algorithm': algorithm,
                'key_id': encryption_result['key_id'],
                'original_size': metadata['original_size'],
                'encrypted_size': output_path.stat().st_size
            }
            
        except Exception as e:
            print(f"❌ File encryption failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'file': str(file_path)
            }
    
    async def decrypt_file(self, encrypted_file_path: Path, 
                          output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Decrypt an encrypted file.
        
        Args:
            encrypted_file_path: Path to encrypted file
            output_path: Output path for decrypted file
        
        Returns:
            Decryption result
        """
        print(f"🔓 Decrypting file: {encrypted_file_path.name}")
        
        try:
            if not encrypted_file_path.exists():
                return {
                    'success': False,
                    'error': f'Encrypted file not found: {encrypted_file_path}'
                }
            
            # Load encrypted file
            async with aiofiles.open(encrypted_file_path, 'r') as f:
                encrypted_data = json.loads(await f.read())
            
            metadata = encrypted_data['metadata']
            encrypted_content = base64.b64decode(encrypted_data['data'])
            
            # Decrypt content based on algorithm
            algorithm = metadata['algorithm']
            key_id = metadata['key_id']
            
            if algorithm == 'fernet':
                decryption_result = await self._decrypt_fernet(encrypted_content, key_id)
            elif algorithm == 'aes-256-gcm':
                decryption_result = await self._decrypt_aes_gcm(encrypted_content, key_id)
            elif algorithm == 'rsa':
                decryption_result = await self._decrypt_rsa(encrypted_content, key_id)
            else:
                return {
                    'success': False,
                    'error': f'Unsupported algorithm: {algorithm}'
                }
            
            if not decryption_result['success']:
                return decryption_result
            
            decrypted_content = decryption_result['decrypted_data']
            
            # Decompress if needed
            if metadata.get('compressed', False):
                import gzip
                decrypted_content = gzip.decompress(decrypted_content)
            
            # Verify checksum
            content_checksum = hashlib.sha256(decrypted_content).hexdigest()
            if content_checksum != metadata.get('checksum'):
                return {
                    'success': False,
                    'error': 'Checksum verification failed - data may be corrupted'
                }
            
            # Generate output path if not provided
            if not output_path:
                original_name = Path(metadata['original_file']).name
                output_path = Path.cwd() / f"{original_name}.decrypted"
            
            # Save decrypted file
            async with aiofiles.open(output_path, 'wb') as f:
                await f.write(decrypted_content)
            
            # Log decryption activity
            await self._log_encryption_activity('decrypt_file', {
                'encrypted_file': str(encrypted_file_path),
                'algorithm': algorithm,
                'key_id': key_id,
                'output': str(output_path)
            })
            
            print(f"✅ File decrypted: {output_path}")
            
            return {
                'success': True,
                'encrypted_file': str(encrypted_file_path),
                'output_file': str(output_path),
                'algorithm': algorithm,
                'key_id': key_id,
                'original_size': len(decrypted_content),
                'checksum_verified': True
            }
            
        except Exception as e:
            print(f"❌ File decryption failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'file': str(encrypted_file_path)
            }
    
    async def encrypt_directory(self, directory_path: Path, 
                               output_path: Optional[Path] = None,
                               algorithm: Optional[str] = None,
                               recursive: bool = True) -> Dict[str, Any]:
        """
        Encrypt all files in a directory.
        
        Args:
            directory_path: Directory to encrypt
            output_path: Output directory for encrypted files
            algorithm: Encryption algorithm to use
            recursive: Whether to encrypt subdirectories
        
        Returns:
            Directory encryption results
        """
        algorithm = algorithm or self.config['default_algorithm']
        
        print(f"🔐 Encrypting directory: {directory_path} using {algorithm}")
        
        try:
            if not directory_path.is_dir():
                return {
                    'success': False,
                    'error': f'Directory not found: {directory_path}'
                }
            
            # Generate output directory if not provided
            if not output_path:
                output_path = self.encrypted_dir / f"{directory_path.name}_encrypted"
            
            output_path.mkdir(parents=True, exist_ok=True)
            
            encryption_results = {
                'timestamp': datetime.now().isoformat(),
                'source_directory': str(directory_path),
                'output_directory': str(output_path),
                'algorithm': algorithm,
                'total_files': 0,
                'encrypted_files': 0,
                'failed_files': 0,
                'results': []
            }
            
            # Find files to encrypt
            pattern = '**/*' if recursive else '*'
            for file_path in directory_path.glob(pattern):
                if file_path.is_file():
                    encryption_results['total_files'] += 1
                    
                    # Maintain directory structure
                    relative_path = file_path.relative_to(directory_path)
                    output_file_path = output_path / f"{relative_path}.encrypted"
                    output_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Encrypt file
                    result = await self.encrypt_file(file_path, output_file_path, algorithm)
                    encryption_results['results'].append(result)
                    
                    if result['success']:
                        encryption_results['encrypted_files'] += 1
                        print(f"✅ Encrypted: {relative_path}")
                    else:
                        encryption_results['failed_files'] += 1
                        print(f"❌ Failed to encrypt: {relative_path}")
            
            # Save directory encryption metadata
            metadata_file = output_path / 'directory_encryption_metadata.json'
            async with aiofiles.open(metadata_file, 'w') as f:
                await f.write(json.dumps(encryption_results, indent=2, default=str))
            
            print(f"✅ Directory encryption completed:")
            print(f"   Files encrypted: {encryption_results['encrypted_files']}")
            print(f"   Files failed: {encryption_results['failed_files']}")
            
            return encryption_results
            
        except Exception as e:
            print(f"❌ Directory encryption failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'directory': str(directory_path)
            }
    
    async def encrypt_database(self, db_path: Path, 
                              output_path: Optional[Path] = None,
                              algorithm: Optional[str] = None) -> Dict[str, Any]:
        """
        Encrypt SQLite database with table-level encryption.
        
        Args:
            db_path: Path to database file
            output_path: Output path for encrypted database
            algorithm: Encryption algorithm to use
        
        Returns:
            Database encryption result
        """
        algorithm = algorithm or self.config['default_algorithm']
        
        print(f"🔐 Encrypting database: {db_path.name} using {algorithm}")
        
        try:
            if not db_path.exists():
                return {
                    'success': False,
                    'error': f'Database not found: {db_path}'
                }
            
            # Generate output path if not provided
            if not output_path:
                output_path = self.encrypted_dir / f"{db_path.name}.encrypted"
            
            # Connect to source database
            source_conn = sqlite3.connect(db_path)
            source_cursor = source_conn.cursor()
            
            # Get all tables
            source_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in source_cursor.fetchall()]
            
            encryption_results = {
                'timestamp': datetime.now().isoformat(),
                'source_database': str(db_path),
                'output_file': str(output_path),
                'algorithm': algorithm,
                'total_tables': len(tables),
                'encrypted_tables': 0,
                'table_results': {}
            }
            
            encrypted_db_data = {
                'metadata': {
                    'database_name': db_path.name,
                    'algorithm': algorithm,
                    'encrypted_at': datetime.now().isoformat(),
                    'tables': tables
                },
                'encrypted_tables': {}
            }
            
            # Encrypt each table
            for table_name in tables:
                try:
                    # Export table data
                    source_cursor.execute(f"SELECT * FROM {table_name}")
                    rows = source_cursor.fetchall()
                    
                    # Get column names
                    source_cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = [row[1] for row in source_cursor.fetchall()]
                    
                    table_data = {
                        'columns': columns,
                        'rows': rows
                    }
                    
                    # Serialize and encrypt table data
                    table_json = json.dumps(table_data, default=str)
                    table_bytes = table_json.encode('utf-8')
                    
                    if algorithm == 'fernet':
                        encryption_result = await self._encrypt_fernet(table_bytes)
                    elif algorithm == 'aes-256-gcm':
                        encryption_result = await self._encrypt_aes_gcm(table_bytes)
                    else:
                        raise ValueError(f"Unsupported algorithm for database: {algorithm}")
                    
                    if encryption_result['success']:
                        encrypted_db_data['encrypted_tables'][table_name] = {
                            'key_id': encryption_result['key_id'],
                            'data': base64.b64encode(encryption_result['encrypted_data']).decode('utf-8'),
                            'row_count': len(rows)
                        }
                        encryption_results['encrypted_tables'] += 1
                        encryption_results['table_results'][table_name] = {'success': True}
                        print(f"✅ Encrypted table: {table_name} ({len(rows)} rows)")
                    else:
                        encryption_results['table_results'][table_name] = {
                            'success': False,
                            'error': encryption_result.get('error', 'Unknown error')
                        }
                
                except Exception as e:
                    encryption_results['table_results'][table_name] = {
                        'success': False,
                        'error': str(e)
                    }
                    print(f"❌ Failed to encrypt table: {table_name}")
            
            source_conn.close()
            
            # Save encrypted database
            async with aiofiles.open(output_path, 'w') as f:
                await f.write(json.dumps(encrypted_db_data, indent=2))
            
            # Set secure permissions
            output_path.chmod(0o600)
            
            # Log database encryption
            await self._log_encryption_activity('encrypt_database', {
                'database': str(db_path),
                'algorithm': algorithm,
                'tables': encryption_results['encrypted_tables'],
                'output': str(output_path)
            })
            
            print(f"✅ Database encryption completed:")
            print(f"   Tables encrypted: {encryption_results['encrypted_tables']}/{encryption_results['total_tables']}")
            
            return encryption_results
            
        except Exception as e:
            print(f"❌ Database encryption failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'database': str(db_path)
            }
    
    async def generate_key(self, algorithm: str, key_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate encryption key for specified algorithm.
        
        Args:
            algorithm: Encryption algorithm
            key_id: Custom key ID
        
        Returns:
            Key generation result
        """
        if not key_id:
            key_id = f"{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"🔑 Generating {algorithm} key: {key_id}")
        
        try:
            if algorithm == 'fernet':
                key_data = await self._generate_fernet_key()
            elif algorithm == 'aes-256-gcm':
                key_data = await self._generate_aes_key()
            elif algorithm == 'rsa':
                key_data = await self._generate_rsa_key()
            else:
                return {
                    'success': False,
                    'error': f'Unsupported algorithm: {algorithm}'
                }
            
            # Store key securely
            key_metadata = {
                'key_id': key_id,
                'algorithm': algorithm,
                'generated_at': datetime.now().isoformat(),
                'key_size': len(key_data.get('key', b'')),
                'usage_count': 0
            }
            
            # Save to keyring (secure storage)
            if algorithm == 'rsa':
                keyring.set_password('structured_docs_synth', f'{key_id}_private', key_data['private_key'])
                keyring.set_password('structured_docs_synth', f'{key_id}_public', key_data['public_key'])
            else:
                keyring.set_password('structured_docs_synth', key_id, base64.b64encode(key_data['key']).decode())
            
            # Save metadata
            metadata_file = self.key_dir / f"{key_id}.metadata.json"
            async with aiofiles.open(metadata_file, 'w') as f:
                await f.write(json.dumps(key_metadata, indent=2))
            
            metadata_file.chmod(0o600)
            
            print(f"✅ Key generated and stored: {key_id}")
            
            return {
                'success': True,
                'key_id': key_id,
                'algorithm': algorithm,
                'key_size': key_metadata['key_size']
            }
            
        except Exception as e:
            print(f"❌ Key generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'algorithm': algorithm
            }
    
    async def list_keys(self) -> List[Dict[str, Any]]:
        """
        List all available encryption keys.
        
        Returns:
            List of key information
        """
        keys = []
        
        for metadata_file in self.key_dir.glob('*.metadata.json'):
            try:
                async with aiofiles.open(metadata_file, 'r') as f:
                    metadata = json.loads(await f.read())
                
                # Check if key is still available in keyring
                key_id = metadata['key_id']
                try:
                    keyring.get_password('structured_docs_synth', key_id)
                    key_available = True
                except Exception:
                    key_available = False
                
                keys.append({
                    **metadata,
                    'key_available': key_available
                })
                
            except Exception as e:
                print(f"⚠️  Warning: Could not load key metadata from {metadata_file}: {e}")
        
        return sorted(keys, key=lambda x: x['generated_at'], reverse=True)
    
    async def rotate_keys(self, key_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Rotate encryption keys (generate new keys and re-encrypt data).
        
        Args:
            key_ids: Specific key IDs to rotate (default: all keys)
        
        Returns:
            Key rotation results
        """
        print("🔄 Starting key rotation process...")
        
        try:
            available_keys = await self.list_keys()
            
            if not key_ids:
                key_ids = [key['key_id'] for key in available_keys if key['key_available']]
            
            rotation_results = {
                'timestamp': datetime.now().isoformat(),
                'total_keys': len(key_ids),
                'rotated_keys': 0,
                'failed_keys': 0,
                'results': {}
            }
            
            for key_id in key_ids:
                print(f"🔄 Rotating key: {key_id}")
                
                # Find key metadata
                key_metadata = next((k for k in available_keys if k['key_id'] == key_id), None)
                if not key_metadata:
                    rotation_results['results'][key_id] = {
                        'success': False,
                        'error': 'Key metadata not found'
                    }
                    rotation_results['failed_keys'] += 1
                    continue
                
                # Generate new key with same algorithm
                new_key_result = await self.generate_key(key_metadata['algorithm'])
                
                if new_key_result['success']:
                    # Update key metadata to mark as rotated
                    old_metadata_file = self.key_dir / f"{key_id}.metadata.json"
                    if old_metadata_file.exists():
                        # Rename old key metadata
                        rotated_metadata_file = self.key_dir / f"{key_id}_rotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.metadata.json"
                        old_metadata_file.rename(rotated_metadata_file)
                    
                    rotation_results['results'][key_id] = {
                        'success': True,
                        'new_key_id': new_key_result['key_id'],
                        'algorithm': key_metadata['algorithm']
                    }
                    rotation_results['rotated_keys'] += 1
                    print(f"✅ Rotated key: {key_id} -> {new_key_result['key_id']}")
                else:
                    rotation_results['results'][key_id] = {
                        'success': False,
                        'error': new_key_result.get('error', 'Unknown error')
                    }
                    rotation_results['failed_keys'] += 1
            
            print(f"🔄 Key rotation completed:")
            print(f"   Rotated: {rotation_results['rotated_keys']}")
            print(f"   Failed: {rotation_results['failed_keys']}")
            
            return rotation_results
            
        except Exception as e:
            print(f"❌ Key rotation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # Private encryption methods
    
    async def _encrypt_fernet(self, data: bytes, key_id: Optional[str] = None) -> Dict[str, Any]:
        """Encrypt data using Fernet algorithm"""
        try:
            if not key_id:
                # Generate new key
                key_result = await self.generate_key('fernet')
                if not key_result['success']:
                    return key_result
                key_id = key_result['key_id']
            
            # Get key from keyring
            key_data = keyring.get_password('structured_docs_synth', key_id)
            if not key_data:
                return {
                    'success': False,
                    'error': f'Key not found: {key_id}'
                }
            
            key = base64.b64decode(key_data.encode())
            fernet = Fernet(base64.urlsafe_b64encode(key))
            
            encrypted_data = fernet.encrypt(data)
            
            return {
                'success': True,
                'encrypted_data': encrypted_data,
                'key_id': key_id
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _decrypt_fernet(self, encrypted_data: bytes, key_id: str) -> Dict[str, Any]:
        """Decrypt data using Fernet algorithm"""
        try:
            # Get key from keyring
            key_data = keyring.get_password('structured_docs_synth', key_id)
            if not key_data:
                return {
                    'success': False,
                    'error': f'Key not found: {key_id}'
                }
            
            key = base64.b64decode(key_data.encode())
            fernet = Fernet(base64.urlsafe_b64encode(key))
            
            decrypted_data = fernet.decrypt(encrypted_data)
            
            return {
                'success': True,
                'decrypted_data': decrypted_data
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _encrypt_aes_gcm(self, data: bytes, key_id: Optional[str] = None) -> Dict[str, Any]:
        """Encrypt data using AES-256-GCM algorithm"""
        try:
            if not key_id:
                # Generate new key
                key_result = await self.generate_key('aes-256-gcm')
                if not key_result['success']:
                    return key_result
                key_id = key_result['key_id']
            
            # Get key from keyring
            key_data = keyring.get_password('structured_docs_synth', key_id)
            if not key_data:
                return {
                    'success': False,
                    'error': f'Key not found: {key_id}'
                }
            
            key = base64.b64decode(key_data.encode())
            
            # Generate random IV
            iv = secrets.token_bytes(self.config['iv_size'])
            
            # Create cipher
            cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            
            # Encrypt data
            ciphertext = encryptor.update(data) + encryptor.finalize()
            
            # Combine IV, tag, and ciphertext
            encrypted_data = iv + encryptor.tag + ciphertext
            
            return {
                'success': True,
                'encrypted_data': encrypted_data,
                'key_id': key_id
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _decrypt_aes_gcm(self, encrypted_data: bytes, key_id: str) -> Dict[str, Any]:
        """Decrypt data using AES-256-GCM algorithm"""
        try:
            # Get key from keyring
            key_data = keyring.get_password('structured_docs_synth', key_id)
            if not key_data:
                return {
                    'success': False,
                    'error': f'Key not found: {key_id}'
                }
            
            key = base64.b64decode(key_data.encode())
            
            # Extract IV, tag, and ciphertext
            iv_size = self.config['iv_size']
            tag_size = 16  # GCM tag size
            
            iv = encrypted_data[:iv_size]
            tag = encrypted_data[iv_size:iv_size + tag_size]
            ciphertext = encrypted_data[iv_size + tag_size:]
            
            # Create cipher
            cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend())
            decryptor = cipher.decryptor()
            
            # Decrypt data
            decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()
            
            return {
                'success': True,
                'decrypted_data': decrypted_data
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _encrypt_rsa(self, data: bytes, key_id: Optional[str] = None) -> Dict[str, Any]:
        """Encrypt data using RSA algorithm"""
        try:
            if len(data) > 190:  # RSA can't encrypt large data directly
                return {
                    'success': False,
                    'error': 'Data too large for RSA encryption. Use AES or Fernet for large data.'
                }
            
            if not key_id:
                # Generate new key
                key_result = await self.generate_key('rsa')
                if not key_result['success']:
                    return key_result
                key_id = key_result['key_id']
            
            # Get public key from keyring
            public_key_pem = keyring.get_password('structured_docs_synth', f'{key_id}_public')
            if not public_key_pem:
                return {
                    'success': False,
                    'error': f'Public key not found: {key_id}'
                }
            
            public_key = serialization.load_pem_public_key(public_key_pem.encode(), backend=default_backend())
            
            # Encrypt data
            encrypted_data = public_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return {
                'success': True,
                'encrypted_data': encrypted_data,
                'key_id': key_id
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _decrypt_rsa(self, encrypted_data: bytes, key_id: str) -> Dict[str, Any]:
        """Decrypt data using RSA algorithm"""
        try:
            # Get private key from keyring
            private_key_pem = keyring.get_password('structured_docs_synth', f'{key_id}_private')
            if not private_key_pem:
                return {
                    'success': False,
                    'error': f'Private key not found: {key_id}'
                }
            
            private_key = serialization.load_pem_private_key(
                private_key_pem.encode(), 
                password=None, 
                backend=default_backend()
            )
            
            # Decrypt data
            decrypted_data = private_key.decrypt(
                encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return {
                'success': True,
                'decrypted_data': decrypted_data
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    # Key generation methods
    
    async def _generate_fernet_key(self) -> Dict[str, bytes]:
        """Generate Fernet key"""
        key = Fernet.generate_key()
        return {'key': base64.urlsafe_b64decode(key)}
    
    async def _generate_aes_key(self) -> Dict[str, bytes]:
        """Generate AES-256 key"""
        key = secrets.token_bytes(self.config['key_size'])
        return {'key': key}
    
    async def _generate_rsa_key(self) -> Dict[str, str]:
        """Generate RSA key pair"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.config['rsa_key_size'],
            backend=default_backend()
        )
        
        public_key = private_key.public_key()
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return {
            'private_key': private_pem.decode(),
            'public_key': public_pem.decode()
        }
    
    async def _log_encryption_activity(self, action: str, details: Dict[str, Any]):
        """Log encryption activity for audit purposes"""
        if not self.config['audit_logging']:
            return
        
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'action': action,
                'details': details
            }
            
            async with aiofiles.open(self.audit_log_file, 'a') as f:
                await f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            print(f"⚠️  Warning: Could not log encryption activity: {e}")


async def main():
    """
    Main encryption script function.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Encrypt and decrypt data for structured document synthesis'
    )
    parser.add_argument(
        'action',
        choices=['encrypt-file', 'decrypt-file', 'encrypt-dir', 'encrypt-db', 'generate-key', 'list-keys', 'rotate-keys'],
        help='Action to perform'
    )
    parser.add_argument(
        '--input',
        type=Path,
        help='Input file/directory path'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file/directory path'
    )
    parser.add_argument(
        '--algorithm',
        choices=['fernet', 'aes-256-gcm', 'rsa'],
        default='fernet',
        help='Encryption algorithm'
    )
    parser.add_argument(
        '--key-id',
        help='Specific key ID to use'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Encrypt directories recursively'
    )
    parser.add_argument(
        '--key-dir',
        type=Path,
        help='Custom key directory'
    )
    parser.add_argument(
        '--encrypted-dir',
        type=Path,
        help='Custom encrypted files directory'
    )
    
    args = parser.parse_args()
    
    # Initialize encryption system
    config = ENCRYPTION_CONFIG.copy()
    
    encryption_system = DataEncryption(
        key_dir=args.key_dir,
        encrypted_dir=args.encrypted_dir,
        config=config
    )
    
    if args.action == 'encrypt-file':
        if not args.input:
            print("❌ Input file path required")
            return 1
        
        result = await encryption_system.encrypt_file(args.input, args.output, args.algorithm, args.key_id)
        
        if result['success']:
            print(f"\n✅ File encrypted successfully")
            print(f"📁 Output: {result['output_file']}")
            print(f"🔑 Key ID: {result['key_id']}")
            print(f"📊 Size: {result['original_size']} -> {result['encrypted_size']} bytes")
        else:
            print(f"\n❌ Encryption failed: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.action == 'decrypt-file':
        if not args.input:
            print("❌ Input encrypted file path required")
            return 1
        
        result = await encryption_system.decrypt_file(args.input, args.output)
        
        if result['success']:
            print(f"\n✅ File decrypted successfully")
            print(f"📁 Output: {result['output_file']}")
            print(f"🔑 Key ID: {result['key_id']}")
            print(f"✅ Checksum verified: {result['checksum_verified']}")
        else:
            print(f"\n❌ Decryption failed: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.action == 'encrypt-dir':
        if not args.input:
            print("❌ Input directory path required")
            return 1
        
        result = await encryption_system.encrypt_directory(args.input, args.output, args.algorithm, args.recursive)
        
        if result.get('success', True):
            print(f"\n✅ Directory encrypted successfully")
            print(f"📁 Output: {result['output_directory']}")
            print(f"📊 Files: {result['encrypted_files']}/{result['total_files']} encrypted")
            if result['failed_files'] > 0:
                return 1
        else:
            print(f"\n❌ Directory encryption failed: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.action == 'encrypt-db':
        if not args.input:
            print("❌ Input database path required")
            return 1
        
        result = await encryption_system.encrypt_database(args.input, args.output, args.algorithm)
        
        if result.get('success', True):
            print(f"\n✅ Database encrypted successfully")
            print(f"📁 Output: {result['output_file']}")
            print(f"📊 Tables: {result['encrypted_tables']}/{result['total_tables']} encrypted")
        else:
            print(f"\n❌ Database encryption failed: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.action == 'generate-key':
        result = await encryption_system.generate_key(args.algorithm, args.key_id)
        
        if result['success']:
            print(f"\n✅ Key generated successfully")
            print(f"🔑 Key ID: {result['key_id']}")
            print(f"🔒 Algorithm: {result['algorithm']}")
            print(f"📊 Key size: {result['key_size']} bytes")
        else:
            print(f"\n❌ Key generation failed: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.action == 'list-keys':
        keys = await encryption_system.list_keys()
        
        if not keys:
            print("📋 No encryption keys found")
            return 0
        
        print(f"🔑 Available Encryption Keys ({len(keys)}):")
        print("=" * 80)
        
        for key in keys:
            status = "✅ Available" if key['key_available'] else "❌ Missing"
            print(f"🔑 {key['key_id']}")
            print(f"   Algorithm: {key['algorithm']}")
            print(f"   Generated: {key['generated_at']}")
            print(f"   Key size: {key['key_size']} bytes")
            print(f"   Usage count: {key['usage_count']}")
            print(f"   Status: {status}")
            print()
    
    elif args.action == 'rotate-keys':
        key_ids = [args.key_id] if args.key_id else None
        result = await encryption_system.rotate_keys(key_ids)
        
        if result.get('success', True):
            print(f"\n✅ Key rotation completed")
            print(f"🔄 Keys rotated: {result['rotated_keys']}")
            print(f"❌ Keys failed: {result['failed_keys']}")
        else:
            print(f"\n❌ Key rotation failed: {result.get('error', 'Unknown error')}")
            return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(asyncio.run(main()))