"""Storage encryption utilities."""

from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
import os
import base64


class StorageEncryption:
    """Client-side encryption for storage."""
    
    def __init__(self, key: Optional[bytes] = None):
        """Initialize encryption with optional key."""
        if key:
            self.key = key
        else:
            self.key = Fernet.generate_key()
        self.fernet = Fernet(self.key)
    
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data using Fernet."""
        return self.fernet.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using Fernet."""
        return self.fernet.decrypt(encrypted_data)
    
    def encrypt_file(self, input_path: str, output_path: str) -> None:
        """Encrypt a file."""
        with open(input_path, "rb") as input_file:
            data = input_file.read()
        
        encrypted_data = self.encrypt_data(data)
        
        with open(output_path, "wb") as output_file:
            output_file.write(encrypted_data)
    
    def decrypt_file(self, input_path: str, output_path: str) -> None:
        """Decrypt a file."""
        with open(input_path, "rb") as input_file:
            encrypted_data = input_file.read()
        
        data = self.decrypt_data(encrypted_data)
        
        with open(output_path, "wb") as output_file:
            output_file.write(data)
    
    @staticmethod
    def generate_aes_key(key_size: int = 256) -> bytes:
        """Generate an AES key."""
        return os.urandom(key_size // 8)
    
    @staticmethod
    def encrypt_aes_256_cbc(data: bytes, key: bytes, iv: Optional[bytes] = None) -> Dict[str, str]:
        """Encrypt data using AES-256-CBC."""
        if iv is None:
            iv = os.urandom(16)
        
        # Pad data to multiple of 16 bytes
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data) + padder.finalize()
        
        # Encrypt
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(padded_data) + encryptor.finalize()
        
        return {
            "encrypted_data": base64.b64encode(encrypted).decode("utf-8"),
            "iv": base64.b64encode(iv).decode("utf-8"),
        }
    
    @staticmethod
    def decrypt_aes_256_cbc(encrypted_data: str, key: bytes, iv: str) -> bytes:
        """Decrypt data using AES-256-CBC."""
        # Decode from base64
        encrypted_bytes = base64.b64decode(encrypted_data)
        iv_bytes = base64.b64decode(iv)
        
        # Decrypt
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv_bytes))
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(encrypted_bytes) + decryptor.finalize()
        
        # Unpad
        unpadder = padding.PKCS7(128).unpadder()
        data = unpadder.update(padded_data) + unpadder.finalize()
        
        return data
    
    def get_key_string(self) -> str:
        """Get the encryption key as a string."""
        return self.key.decode("utf-8")
    
    @classmethod
    def from_key_string(cls, key_string: str) -> "StorageEncryption":
        """Create instance from key string."""
        key = key_string.encode("utf-8")
        return cls(key)