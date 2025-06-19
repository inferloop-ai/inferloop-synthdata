"""
Configuration management for Structured Documents Synthetic Data Generator
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings
import yaml


class GenerationConfig(BaseSettings):
    """Document generation configuration"""
    
    # Basic settings
    output_dir: str = Field(default="./output", description="Output directory for generated documents")
    temp_dir: str = Field(default="./temp", description="Temporary files directory")
    
    # Generation settings
    default_document_type: str = Field(default="legal_contract", description="Default document type")
    max_documents_per_batch: int = Field(default=1000, description="Maximum documents per batch")
    generation_timeout: int = Field(default=300, description="Generation timeout in seconds")
    
    # Quality settings
    min_ocr_accuracy: float = Field(default=0.95, description="Minimum OCR accuracy threshold")
    min_structural_precision: float = Field(default=0.90, description="Minimum structural precision")
    
    # Privacy settings
    enable_pii_detection: bool = Field(default=True, description="Enable PII detection and masking")
    differential_privacy_epsilon: float = Field(default=1.0, description="Differential privacy epsilon")
    
    class Config:
        env_prefix = "SYNTH_DOC_"
        case_sensitive = False


class DatabaseConfig(BaseSettings):
    """Database configuration"""
    
    # SQLite for MVP
    database_url: str = Field(default="sqlite:///./synth_docs.db", description="Database URL")
    echo_sql: bool = Field(default=False, description="Echo SQL queries")
    
    class Config:
        env_prefix = "DB_"


class StorageConfig(BaseSettings):
    """Storage configuration"""
    
    # Local storage for MVP
    storage_type: str = Field(default="local", description="Storage type (local, s3, gcs, azure)")
    local_storage_path: str = Field(default="./data", description="Local storage path")
    
    # Cloud storage (optional)
    aws_bucket: Optional[str] = Field(default=None, description="AWS S3 bucket name")
    aws_region: Optional[str] = Field(default=None, description="AWS region")
    gcs_bucket: Optional[str] = Field(default=None, description="Google Cloud Storage bucket")
    azure_container: Optional[str] = Field(default=None, description="Azure blob container")
    
    class Config:
        env_prefix = "STORAGE_"


class APIConfig(BaseSettings):
    """API configuration"""
    
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Security
    secret_key: str = Field(default="dev-secret-key-change-in-production", description="Secret key")
    access_token_expire_minutes: int = Field(default=60, description="Access token expiration")
    
    # Rate limiting
    rate_limit_per_minute: int = Field(default=100, description="Rate limit per minute")
    
    class Config:
        env_prefix = "API_"


class LoggingConfig(BaseSettings):
    """Logging configuration"""
    
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    file_path: Optional[str] = Field(default=None, description="Log file path")
    max_file_size: int = Field(default=10485760, description="Max log file size (10MB)")
    backup_count: int = Field(default=5, description="Number of backup log files")
    
    class Config:
        env_prefix = "LOG_"


class Config:
    """Main configuration class"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file and environment"""
        
        # Load from YAML file if provided
        file_config = {}
        if self.config_file and os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                file_config = yaml.safe_load(f) or {}
        
        # Initialize configuration sections
        self.generation = GenerationConfig(**file_config.get('generation', {}))
        self.database = DatabaseConfig(**file_config.get('database', {}))
        self.storage = StorageConfig(**file_config.get('storage', {}))
        self.api = APIConfig(**file_config.get('api', {}))
        self.logging = LoggingConfig(**file_config.get('logging', {}))
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.generation.output_dir,
            self.generation.temp_dir,
            self.storage.local_storage_path,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'generation': self.generation.dict(),
            'database': self.database.dict(),
            'storage': self.storage.dict(),
            'api': self.api.dict(),
            'logging': self.logging.dict(),
        }
    
    def save_to_file(self, file_path: str):
        """Save current configuration to YAML file"""
        with open(file_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# Global configuration instance
_config: Optional[Config] = None


def get_config(config_file: Optional[str] = None) -> Config:
    """Get global configuration instance"""
    global _config
    if _config is None or config_file:
        _config = Config(config_file)
    return _config


def reload_config(config_file: Optional[str] = None):
    """Reload configuration"""
    global _config
    _config = Config(config_file)


# Template configuration for different document types
DOCUMENT_TYPES = {
    'legal_contract': {
        'name': 'Legal Contract',
        'description': 'Legal contracts and agreements',
        'template_path': 'legal/contract_template.yaml',
        'supported_formats': ['pdf', 'docx'],
        'required_fields': ['parties', 'effective_date', 'jurisdiction'],
        'optional_fields': ['terms', 'signatures', 'amendments']
    },
    'medical_form': {
        'name': 'Medical Form',
        'description': 'Healthcare and medical forms',
        'template_path': 'healthcare/medical_form_template.yaml',
        'supported_formats': ['pdf', 'docx'],
        'required_fields': ['patient_name', 'date_of_birth', 'medical_record_number'],
        'optional_fields': ['insurance_info', 'emergency_contact', 'medical_history']
    },
    'loan_application': {
        'name': 'Loan Application',
        'description': 'Banking loan application forms',
        'template_path': 'banking/loan_application_template.yaml',
        'supported_formats': ['pdf', 'docx'],
        'required_fields': ['applicant_name', 'loan_amount', 'income'],
        'optional_fields': ['collateral', 'co_signer', 'credit_history']
    },
    'tax_form': {
        'name': 'Tax Form',
        'description': 'Government tax forms',
        'template_path': 'government/tax_form_template.yaml',
        'supported_formats': ['pdf', 'docx'],
        'required_fields': ['taxpayer_name', 'ssn', 'tax_year'],
        'optional_fields': ['dependents', 'deductions', 'filing_status']
    }
}


def get_document_type_config(document_type: str) -> Dict[str, Any]:
    """Get configuration for specific document type"""
    if document_type not in DOCUMENT_TYPES:
        raise ValueError(f"Unknown document type: {document_type}")
    return DOCUMENT_TYPES[document_type]


def list_document_types() -> List[str]:
    """List all available document types"""
    return list(DOCUMENT_TYPES.keys())