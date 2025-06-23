"""
Security resource abstractions for IAM, secrets, and certificates
"""

from abc import abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

from .base import BaseResource, ResourceConfig, ResourceType


class PermissionType(Enum):
    """Types of permissions"""
    ALLOW = "allow"
    DENY = "deny"


class SecretType(Enum):
    """Types of secrets"""
    PASSWORD = "password"
    API_KEY = "api_key"
    CERTIFICATE = "certificate"
    SSH_KEY = "ssh_key"
    DATABASE = "database"
    OAUTH = "oauth"
    OTHER = "other"


class CertificateType(Enum):
    """Types of certificates"""
    SERVER = "server"
    CLIENT = "client"
    CA = "ca"
    WILDCARD = "wildcard"


@dataclass
class IAMConfig(ResourceConfig):
    """Configuration for IAM resources"""
    description: str = ""
    permissions: List[Dict[str, Any]] = field(default_factory=list)
    trust_policy: Optional[Dict[str, Any]] = None
    max_session_duration: int = 3600
    
    def validate(self) -> List[str]:
        errors = super().validate()
        if self.max_session_duration < 900 or self.max_session_duration > 43200:
            errors.append("Max session duration must be between 900 and 43200 seconds")
        return errors


@dataclass
class UserConfig(IAMConfig):
    """Configuration for IAM user"""
    username: str = ""
    groups: List[str] = field(default_factory=list)
    policies: List[str] = field(default_factory=list)
    access_keys: bool = False
    console_access: bool = True
    mfa_enabled: bool = True
    
    def validate(self) -> List[str]:
        errors = super().validate()
        if not self.username:
            errors.append("Username is required")
        return errors


@dataclass
class RoleConfig(IAMConfig):
    """Configuration for IAM role"""
    role_name: str = ""
    assume_role_policy: Dict[str, Any] = field(default_factory=dict)
    policies: List[str] = field(default_factory=list)
    inline_policies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        errors = super().validate()
        if not self.role_name:
            errors.append("Role name is required")
        if not self.assume_role_policy:
            errors.append("Assume role policy is required")
        return errors


@dataclass
class PolicyConfig(IAMConfig):
    """Configuration for IAM policy"""
    policy_name: str = ""
    policy_document: Dict[str, Any] = field(default_factory=dict)
    path: str = "/"
    
    def validate(self) -> List[str]:
        errors = super().validate()
        if not self.policy_name:
            errors.append("Policy name is required")
        if not self.policy_document:
            errors.append("Policy document is required")
        return errors


@dataclass
class SecretConfig(ResourceConfig):
    """Configuration for secrets"""
    secret_type: SecretType = SecretType.OTHER
    secret_value: Optional[str] = None
    secret_binary: Optional[bytes] = None
    description: str = ""
    kms_key_id: Optional[str] = None
    rotation_enabled: bool = False
    rotation_interval_days: int = 30
    version_stages: List[str] = field(default_factory=list)
    
    def validate(self) -> List[str]:
        errors = super().validate()
        if not self.secret_value and not self.secret_binary:
            errors.append("Either secret_value or secret_binary is required")
        if self.rotation_interval_days < 1:
            errors.append("Rotation interval must be at least 1 day")
        return errors


@dataclass
class CertificateConfig(ResourceConfig):
    """Configuration for certificates"""
    certificate_type: CertificateType = CertificateType.SERVER
    domain_name: str = ""
    subject_alternative_names: List[str] = field(default_factory=list)
    validation_method: str = "DNS"
    key_algorithm: str = "RSA_2048"
    certificate_body: Optional[str] = None
    private_key: Optional[str] = None
    certificate_chain: Optional[str] = None
    
    def validate(self) -> List[str]:
        errors = super().validate()
        if not self.domain_name and not self.certificate_body:
            errors.append("Either domain_name or certificate_body is required")
        return errors


@dataclass
class SecurityResource:
    """Representation of a security resource"""
    id: str
    arn: str
    name: str
    type: str
    state: str
    created_date: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseIAM(BaseResource[IAMConfig, SecurityResource]):
    """Base class for IAM resources"""
    
    def get_resource_type(self) -> ResourceType:
        return ResourceType.SECURITY
    
    @abstractmethod
    async def create_user(self, config: UserConfig) -> SecurityResource:
        """Create IAM user"""
        pass
    
    @abstractmethod
    async def create_role(self, config: RoleConfig) -> SecurityResource:
        """Create IAM role"""
        pass
    
    @abstractmethod
    async def create_policy(self, config: PolicyConfig) -> SecurityResource:
        """Create IAM policy"""
        pass
    
    @abstractmethod
    async def attach_policy(self, resource_arn: str, policy_arn: str) -> bool:
        """Attach policy to user/role/group"""
        pass
    
    @abstractmethod
    async def detach_policy(self, resource_arn: str, policy_arn: str) -> bool:
        """Detach policy from user/role/group"""
        pass
    
    @abstractmethod
    async def create_access_key(self, username: str) -> Dict[str, str]:
        """Create access key for user"""
        pass
    
    @abstractmethod
    async def delete_access_key(self, username: str, access_key_id: str) -> bool:
        """Delete access key"""
        pass
    
    @abstractmethod
    async def enable_mfa(self, username: str, mfa_device: str) -> bool:
        """Enable MFA for user"""
        pass
    
    @abstractmethod
    async def assume_role(self, role_arn: str, session_name: str) -> Dict[str, Any]:
        """Assume IAM role"""
        pass
    
    @abstractmethod
    async def get_policy_document(self, policy_arn: str) -> Dict[str, Any]:
        """Get policy document"""
        pass


class BaseSecrets(BaseResource[SecretConfig, SecurityResource]):
    """Base class for secrets management"""
    
    def get_resource_type(self) -> ResourceType:
        return ResourceType.SECURITY
    
    @abstractmethod
    async def get_secret_value(self, secret_id: str, version_id: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve secret value"""
        pass
    
    @abstractmethod
    async def update_secret(self, secret_id: str, secret_value: Optional[str] = None, 
                           secret_binary: Optional[bytes] = None) -> bool:
        """Update secret value"""
        pass
    
    @abstractmethod
    async def rotate_secret(self, secret_id: str) -> bool:
        """Rotate secret"""
        pass
    
    @abstractmethod
    async def create_secret_version(self, secret_id: str, secret_value: str, 
                                   version_stages: List[str]) -> str:
        """Create new version of secret"""
        pass
    
    @abstractmethod
    async def list_secret_versions(self, secret_id: str) -> List[Dict[str, Any]]:
        """List all versions of a secret"""
        pass
    
    @abstractmethod
    async def restore_secret(self, secret_id: str) -> bool:
        """Restore deleted secret"""
        pass
    
    @abstractmethod
    async def replicate_secret(self, secret_id: str, regions: List[str]) -> bool:
        """Replicate secret to other regions"""
        pass
    
    @abstractmethod
    async def set_automatic_rotation(self, secret_id: str, rotation_lambda_arn: str, 
                                    rotation_days: int) -> bool:
        """Set up automatic rotation"""
        pass


class BaseCertificates(BaseResource[CertificateConfig, SecurityResource]):
    """Base class for certificate management"""
    
    def get_resource_type(self) -> ResourceType:
        return ResourceType.SECURITY
    
    @abstractmethod
    async def request_certificate(self, config: CertificateConfig) -> str:
        """Request new certificate"""
        pass
    
    @abstractmethod
    async def import_certificate(self, config: CertificateConfig) -> str:
        """Import existing certificate"""
        pass
    
    @abstractmethod
    async def validate_certificate(self, certificate_id: str) -> bool:
        """Validate certificate"""
        pass
    
    @abstractmethod
    async def get_certificate(self, certificate_id: str) -> Dict[str, Any]:
        """Get certificate details"""
        pass
    
    @abstractmethod
    async def renew_certificate(self, certificate_id: str) -> bool:
        """Renew certificate"""
        pass
    
    @abstractmethod
    async def export_certificate(self, certificate_id: str, passphrase: str) -> Dict[str, str]:
        """Export certificate with private key"""
        pass
    
    @abstractmethod
    async def list_expiring_certificates(self, days: int = 30) -> List[Dict[str, Any]]:
        """List certificates expiring within specified days"""
        pass
    
    @abstractmethod
    async def add_tags_to_certificate(self, certificate_id: str, tags: Dict[str, str]) -> bool:
        """Add tags to certificate"""
        pass