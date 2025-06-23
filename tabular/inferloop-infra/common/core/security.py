"""Security components for infrastructure."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import secrets
import string
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64


class EncryptionType(Enum):
    """Supported encryption types."""
    
    AES_256 = "aes-256"
    RSA_4096 = "rsa-4096"
    KMS = "kms"


class AccessLevel(Enum):
    """Access control levels."""
    
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    OWNER = "owner"


@dataclass
class SecurityConfig:
    """Security configuration."""
    
    enable_encryption_at_rest: bool = True
    enable_encryption_in_transit: bool = True
    encryption_type: EncryptionType = EncryptionType.AES_256
    kms_key_id: Optional[str] = None
    
    enable_mfa: bool = False
    enable_ip_whitelist: bool = True
    allowed_ips: List[str] = field(default_factory=lambda: ["0.0.0.0/0"])
    
    password_min_length: int = 12
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_special: bool = True
    
    session_timeout_minutes: int = 30
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    
    enable_audit_logging: bool = True
    enable_compliance_mode: bool = False
    compliance_standards: List[str] = field(default_factory=list)
    
    certificate_arn: Optional[str] = None
    enable_waf: bool = False
    enable_ddos_protection: bool = False


@dataclass
class Secret:
    """Secret information."""
    
    name: str
    value: str
    description: Optional[str] = None
    rotation_days: int = 90
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class IAMRole:
    """IAM role definition."""
    
    name: str
    description: str
    trust_policy: Dict[str, Any]
    policies: List[Dict[str, Any]]
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class SecurityGroup:
    """Security group/firewall rules."""
    
    name: str
    description: str
    ingress_rules: List[Dict[str, Any]] = field(default_factory=list)
    egress_rules: List[Dict[str, Any]] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)


class BaseSecurityProvider(ABC):
    """Abstract base class for security providers."""
    
    def __init__(self, config: SecurityConfig):
        """Initialize security provider."""
        self.config = config
        self._encryption_key = None
    
    @abstractmethod
    def create_iam_role(self, role: IAMRole) -> str:
        """Create an IAM role."""
        pass
    
    @abstractmethod
    def create_security_group(self, group: SecurityGroup) -> str:
        """Create a security group."""
        pass
    
    @abstractmethod
    def store_secret(self, secret: Secret) -> str:
        """Store a secret securely."""
        pass
    
    @abstractmethod
    def retrieve_secret(self, secret_name: str) -> str:
        """Retrieve a secret."""
        pass
    
    @abstractmethod
    def rotate_secret(self, secret_name: str) -> str:
        """Rotate a secret."""
        pass
    
    @abstractmethod
    def create_kms_key(self, key_alias: str, description: str) -> str:
        """Create a KMS key."""
        pass
    
    @abstractmethod
    def encrypt_data(self, data: bytes, key_id: Optional[str] = None) -> bytes:
        """Encrypt data."""
        pass
    
    @abstractmethod
    def decrypt_data(self, encrypted_data: bytes, key_id: Optional[str] = None) -> bytes:
        """Decrypt data."""
        pass
    
    @abstractmethod
    def create_certificate(self, domain: str, san: List[str] = None) -> str:
        """Create or import an SSL certificate."""
        pass
    
    @abstractmethod
    def enable_audit_logging(self, resource_id: str, log_group: str) -> None:
        """Enable audit logging for a resource."""
        pass
    
    def generate_password(self, length: Optional[int] = None) -> str:
        """Generate a secure password based on config requirements."""
        length = length or self.config.password_min_length
        
        # Character sets
        chars = ""
        required_chars = []
        
        if self.config.password_require_lowercase:
            chars += string.ascii_lowercase
            required_chars.append(secrets.choice(string.ascii_lowercase))
        
        if self.config.password_require_uppercase:
            chars += string.ascii_uppercase
            required_chars.append(secrets.choice(string.ascii_uppercase))
        
        if self.config.password_require_numbers:
            chars += string.digits
            required_chars.append(secrets.choice(string.digits))
        
        if self.config.password_require_special:
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            chars += special_chars
            required_chars.append(secrets.choice(special_chars))
        
        # Generate remaining characters
        remaining_length = length - len(required_chars)
        password_chars = required_chars + [
            secrets.choice(chars) for _ in range(remaining_length)
        ]
        
        # Shuffle to avoid predictable patterns
        secrets.SystemRandom().shuffle(password_chars)
        
        return "".join(password_chars)
    
    def generate_api_key(self, prefix: str = "sk", length: int = 32) -> str:
        """Generate a secure API key."""
        random_bytes = secrets.token_bytes(length)
        key = base64.urlsafe_b64encode(random_bytes).decode("utf-8").rstrip("=")
        return f"{prefix}_{key}"
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> tuple[str, str]:
        """Hash a password using PBKDF2."""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        salt_b64 = base64.urlsafe_b64encode(salt)
        
        return key.decode(), salt_b64.decode()
    
    def verify_password(self, password: str, hashed: str, salt: str) -> bool:
        """Verify a password against its hash."""
        salt_bytes = base64.urlsafe_b64decode(salt.encode())
        new_hash, _ = self.hash_password(password, salt_bytes)
        return new_hash == hashed
    
    def get_encryption_key(self) -> bytes:
        """Get or generate encryption key."""
        if not self._encryption_key:
            self._encryption_key = Fernet.generate_key()
        return self._encryption_key
    
    def encrypt_local(self, data: str) -> str:
        """Encrypt data locally using Fernet."""
        f = Fernet(self.get_encryption_key())
        return f.encrypt(data.encode()).decode()
    
    def decrypt_local(self, encrypted_data: str) -> str:
        """Decrypt data locally using Fernet."""
        f = Fernet(self.get_encryption_key())
        return f.decrypt(encrypted_data.encode()).decode()
    
    def validate_ip_access(self, ip_address: str) -> bool:
        """Validate if an IP address is allowed."""
        if not self.config.enable_ip_whitelist:
            return True
        
        # Simple implementation - in production, use proper IP range checking
        return any(
            ip_address.startswith(allowed.split("/")[0])
            for allowed in self.config.allowed_ips
        )
    
    def create_security_policy(self, resource_type: str, access_level: AccessLevel) -> Dict[str, Any]:
        """Create a security policy for a resource type."""
        base_policy = {
            "Version": "2012-10-17",
            "Statement": []
        }
        
        # Add statements based on access level
        if access_level == AccessLevel.READ:
            base_policy["Statement"].append({
                "Effect": "Allow",
                "Action": [f"{resource_type}:Get*", f"{resource_type}:List*"],
                "Resource": "*"
            })
        elif access_level == AccessLevel.WRITE:
            base_policy["Statement"].extend([
                {
                    "Effect": "Allow",
                    "Action": [f"{resource_type}:Get*", f"{resource_type}:List*"],
                    "Resource": "*"
                },
                {
                    "Effect": "Allow",
                    "Action": [f"{resource_type}:Create*", f"{resource_type}:Update*"],
                    "Resource": "*"
                }
            ])
        elif access_level in [AccessLevel.ADMIN, AccessLevel.OWNER]:
            base_policy["Statement"].append({
                "Effect": "Allow",
                "Action": f"{resource_type}:*",
                "Resource": "*"
            })
        
        return base_policy