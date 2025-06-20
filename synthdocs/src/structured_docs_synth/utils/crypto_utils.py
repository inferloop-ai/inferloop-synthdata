#!/usr/bin/env python3
"""
Cryptographic utilities for secure data handling.

Provides functions for hashing, encryption, decryption, digital signatures,
and secure token generation for privacy protection and data security.
"""

import base64
import hashlib
import hmac
import os
import secrets
from typing import Any, Dict, Optional, Tuple, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from ..core import get_logger

logger = get_logger(__name__)


def hash_string(data: str, algorithm: str = 'sha256', salt: Optional[bytes] = None) -> str:
    """
    Hash a string using specified algorithm.
    
    Args:
        data: String to hash
        algorithm: Hash algorithm ('sha256', 'sha512', 'md5')
        salt: Optional salt bytes
    
    Returns:
        Hexadecimal hash string
    """
    if salt:
        data_bytes = data.encode('utf-8') + salt
    else:
        data_bytes = data.encode('utf-8')
    
    if algorithm == 'sha256':
        hash_obj = hashlib.sha256(data_bytes)
    elif algorithm == 'sha512':
        hash_obj = hashlib.sha512(data_bytes)
    elif algorithm == 'md5':
        hash_obj = hashlib.md5(data_bytes)
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    return hash_obj.hexdigest()


def generate_salt(length: int = 32) -> bytes:
    """
    Generate cryptographically secure random salt.
    
    Args:
        length: Length of salt in bytes
    
    Returns:
        Random salt bytes
    """
    return os.urandom(length)


def derive_key_from_password(password: str, salt: bytes, iterations: int = 100000) -> bytes:
    """
    Derive encryption key from password using PBKDF2.
    
    Args:
        password: Password string
        salt: Salt bytes
        iterations: Number of iterations
    
    Returns:
        Derived key bytes
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
    )
    return kdf.derive(password.encode('utf-8'))


def encrypt_data(data: Union[str, bytes], key: Optional[bytes] = None, 
                password: Optional[str] = None) -> Dict[str, str]:
    """
    Encrypt data using Fernet symmetric encryption.
    
    Args:
        data: Data to encrypt (string or bytes)
        key: Encryption key (32 bytes)
        password: Password to derive key from
    
    Returns:
        Dictionary with encrypted data and metadata
    """
    try:
        # Convert string to bytes if necessary
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
        
        # Generate or use provided key
        if key is None:
            if password:
                salt = generate_salt()
                key = derive_key_from_password(password, salt)
                # Store salt for later decryption
                key_info = {
                    'method': 'password',
                    'salt': base64.b64encode(salt).decode('ascii')
                }
            else:
                key = Fernet.generate_key()
                key_info = {
                    'method': 'generated',
                    'key': base64.b64encode(key).decode('ascii')
                }
        else:
            key_info = {'method': 'provided'}
        
        # Encrypt data
        f = Fernet(base64.urlsafe_b64encode(key[:32]))
        encrypted_data = f.encrypt(data_bytes)
        
        return {
            'encrypted_data': base64.b64encode(encrypted_data).decode('ascii'),
            'algorithm': 'fernet',
            'key_info': key_info
        }
        
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        raise


def decrypt_data(encrypted_data: Dict[str, str], key: Optional[bytes] = None,
                password: Optional[str] = None) -> bytes:
    """
    Decrypt data encrypted with encrypt_data.
    
    Args:
        encrypted_data: Dictionary from encrypt_data
        key: Decryption key
        password: Password to derive key from
    
    Returns:
        Decrypted data as bytes
    """
    try:
        # Reconstruct key if needed
        if key is None:
            key_info = encrypted_data['key_info']
            if key_info['method'] == 'password':
                if not password:
                    raise ValueError("Password required for decryption")
                salt = base64.b64decode(key_info['salt'].encode('ascii'))
                key = derive_key_from_password(password, salt)
            elif key_info['method'] == 'generated':
                key = base64.b64decode(key_info['key'].encode('ascii'))
            else:
                raise ValueError("Key required for decryption")
        
        # Decrypt data
        encrypted_bytes = base64.b64decode(encrypted_data['encrypted_data'].encode('ascii'))
        f = Fernet(base64.urlsafe_b64encode(key[:32]))
        decrypted_data = f.decrypt(encrypted_bytes)
        
        return decrypted_data
        
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        raise


def create_secure_token(length: int = 32, url_safe: bool = True) -> str:
    """
    Generate cryptographically secure random token.
    
    Args:
        length: Token length in bytes
        url_safe: Whether to use URL-safe encoding
    
    Returns:
        Random token string
    """
    token_bytes = secrets.token_bytes(length)
    
    if url_safe:
        return base64.urlsafe_b64encode(token_bytes).decode('ascii').rstrip('=')
    else:
        return base64.b64encode(token_bytes).decode('ascii')


def generate_rsa_keypair(key_size: int = 2048) -> Tuple[bytes, bytes]:
    """
    Generate RSA public/private key pair.
    
    Args:
        key_size: RSA key size in bits
    
    Returns:
        Tuple of (private_key_pem, public_key_pem)
    """
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size
    )
    
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    return private_pem, public_pem


def sign_data(data: Union[str, bytes], private_key_pem: bytes) -> str:
    """
    Create digital signature for data.
    
    Args:
        data: Data to sign
        private_key_pem: RSA private key in PEM format
    
    Returns:
        Base64-encoded signature
    """
    try:
        # Convert string to bytes if necessary
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
        
        # Load private key
        private_key = serialization.load_pem_private_key(
            private_key_pem,
            password=None
        )
        
        # Create signature
        signature = private_key.sign(
            data_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return base64.b64encode(signature).decode('ascii')
        
    except Exception as e:
        logger.error(f"Signing failed: {e}")
        raise


def verify_signature(data: Union[str, bytes], signature: str, 
                    public_key_pem: bytes) -> bool:
    """
    Verify digital signature.
    
    Args:
        data: Original data
        signature: Base64-encoded signature
        public_key_pem: RSA public key in PEM format
    
    Returns:
        True if signature is valid
    """
    try:
        # Convert string to bytes if necessary
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
        
        # Load public key
        public_key = serialization.load_pem_public_key(public_key_pem)
        
        # Decode signature
        signature_bytes = base64.b64decode(signature.encode('ascii'))
        
        # Verify signature
        public_key.verify(
            signature_bytes,
            data_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return True
        
    except Exception as e:
        logger.debug(f"Signature verification failed: {e}")
        return False


def create_hmac(data: Union[str, bytes], secret: str, algorithm: str = 'sha256') -> str:
    """
    Create HMAC for data integrity verification.
    
    Args:
        data: Data to create HMAC for
        secret: Secret key
        algorithm: Hash algorithm
    
    Returns:
        Hexadecimal HMAC string
    """
    if isinstance(data, str):
        data_bytes = data.encode('utf-8')
    else:
        data_bytes = data
    
    secret_bytes = secret.encode('utf-8')
    
    if algorithm == 'sha256':
        h = hmac.new(secret_bytes, data_bytes, hashlib.sha256)
    elif algorithm == 'sha512':
        h = hmac.new(secret_bytes, data_bytes, hashlib.sha512)
    else:
        raise ValueError(f"Unsupported HMAC algorithm: {algorithm}")
    
    return h.hexdigest()


def verify_hmac(data: Union[str, bytes], hmac_value: str, secret: str,
               algorithm: str = 'sha256') -> bool:
    """
    Verify HMAC for data integrity.
    
    Args:
        data: Original data
        hmac_value: HMAC to verify
        secret: Secret key
        algorithm: Hash algorithm
    
    Returns:
        True if HMAC is valid
    """
    try:
        expected_hmac = create_hmac(data, secret, algorithm)
        return hmac.compare_digest(expected_hmac, hmac_value)
    except Exception:
        return False


def secure_random_string(length: int = 16, alphabet: str = None) -> str:
    """
    Generate secure random string.
    
    Args:
        length: String length
        alphabet: Character set to use
    
    Returns:
        Random string
    """
    if alphabet is None:
        alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def constant_time_compare(a: str, b: str) -> bool:
    """
    Compare two strings in constant time to prevent timing attacks.
    
    Args:
        a: First string
        b: Second string
    
    Returns:
        True if strings are equal
    """
    return hmac.compare_digest(a.encode('utf-8'), b.encode('utf-8'))


def hash_password(password: str, salt: Optional[bytes] = None, 
                 iterations: int = 100000) -> Dict[str, str]:
    """
    Hash password for secure storage.
    
    Args:
        password: Password to hash
        salt: Optional salt (generated if not provided)
        iterations: Number of PBKDF2 iterations
    
    Returns:
        Dictionary with hash and salt
    """
    if salt is None:
        salt = generate_salt()
    
    # Use PBKDF2 for password hashing
    key = derive_key_from_password(password, salt, iterations)
    
    return {
        'hash': base64.b64encode(key).decode('ascii'),
        'salt': base64.b64encode(salt).decode('ascii'),
        'iterations': iterations,
        'algorithm': 'pbkdf2_sha256'
    }


def verify_password(password: str, password_hash: Dict[str, str]) -> bool:
    """
    Verify password against stored hash.
    
    Args:
        password: Password to verify
        password_hash: Dictionary from hash_password
    
    Returns:
        True if password is correct
    """
    try:
        salt = base64.b64decode(password_hash['salt'].encode('ascii'))
        iterations = password_hash['iterations']
        
        # Derive key from provided password
        key = derive_key_from_password(password, salt, iterations)
        expected_hash = base64.b64encode(key).decode('ascii')
        
        # Compare hashes in constant time
        return constant_time_compare(expected_hash, password_hash['hash'])
        
    except Exception:
        return False


__all__ = [
    'hash_string',
    'generate_salt',
    'derive_key_from_password',
    'encrypt_data',
    'decrypt_data',
    'create_secure_token',
    'generate_rsa_keypair',
    'sign_data',
    'verify_signature',
    'create_hmac',
    'verify_hmac',
    'secure_random_string',
    'constant_time_compare',
    'hash_password',
    'verify_password'
]