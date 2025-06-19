"""
Generation configuration models and settings for the SDK.
Provides comprehensive configuration options for document generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple

from pydantic import BaseModel, Field, validator

from .document_types import DocumentType, DocumentFormat, Language, LayoutComplexity, ContentQuality


class GenerationMode(Enum):
    """Document generation modes"""
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"
    CUSTOM = "custom"


class PrivacyLevel(Enum):
    """Privacy protection levels"""
    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    MAXIMUM = "maximum"


class NoiseType(Enum):
    """OCR noise simulation types"""
    GAUSSIAN = "gaussian"
    SALT_PEPPER = "salt_pepper"
    BLUR = "blur"
    SKEW = "skew"
    COMPRESSION = "compression"
    LIGHTING = "lighting"


class RenderingQuality(Enum):
    """Document rendering quality"""
    DRAFT = "draft"
    STANDARD = "standard"
    HIGH = "high"
    PRINT = "print"


class LayoutConfig(BaseModel):
    """Layout generation configuration"""
    complexity: LayoutComplexity = LayoutComplexity.MODERATE
    columns: int = Field(default=1, ge=1, le=4, description="Number of columns")
    margin_size: float = Field(default=72.0, ge=36.0, le=144.0, description="Margin size in points")
    line_spacing: float = Field(default=1.2, ge=0.8, le=3.0, description="Line spacing multiplier")
    paragraph_spacing: float = Field(default=12.0, ge=0.0, le=36.0, description="Paragraph spacing in points")
    
    # Font settings
    font_family: str = Field(default="Arial", description="Primary font family")
    font_size: float = Field(default=12.0, ge=8.0, le=24.0, description="Base font size")
    heading_font_family: Optional[str] = Field(default=None, description="Heading font family")
    
    # Element placement
    randomize_placement: bool = Field(default=True, description="Randomize element placement")
    placement_variance: float = Field(default=0.1, ge=0.0, le=0.5, description="Placement variance factor")
    
    # Table settings
    table_border_width: float = Field(default=1.0, ge=0.0, le=5.0, description="Table border width")
    table_cell_padding: float = Field(default=6.0, ge=2.0, le=12.0, description="Table cell padding")
    
    class Config:
        use_enum_values = True


class ContentConfig(BaseModel):
    """Content generation configuration"""
    quality: ContentQuality = ContentQuality.STANDARD
    language: Language = Language.ENGLISH
    domain_specific: bool = Field(default=True, description="Use domain-specific content")
    
    # Text generation
    min_words_per_paragraph: int = Field(default=20, ge=5, le=200, description="Minimum words per paragraph")
    max_words_per_paragraph: int = Field(default=150, ge=20, le=500, description="Maximum words per paragraph")
    sentence_variety: bool = Field(default=True, description="Vary sentence structure")
    
    # Entity generation
    include_entities: bool = Field(default=True, description="Include named entities")
    entity_density: float = Field(default=0.1, ge=0.0, le=0.5, description="Entity density in text")
    entity_types: List[str] = Field(
        default=["person", "organization", "location", "date"],
        description="Entity types to include"
    )
    
    # Data generation
    realistic_data: bool = Field(default=True, description="Generate realistic data")
    data_consistency: bool = Field(default=True, description="Maintain data consistency")
    temporal_consistency: bool = Field(default=True, description="Maintain temporal consistency")
    
    class Config:
        use_enum_values = True
    
    @validator('max_words_per_paragraph')
    def validate_word_counts(cls, v, values):
        """Validate word count ranges"""
        if 'min_words_per_paragraph' in values and v <= values['min_words_per_paragraph']:
            raise ValueError("max_words_per_paragraph must be greater than min_words_per_paragraph")
        return v


class PrivacyConfig(BaseModel):
    """Privacy protection configuration"""
    level: PrivacyLevel = PrivacyLevel.STANDARD
    anonymize_names: bool = Field(default=True, description="Anonymize person names")
    anonymize_addresses: bool = Field(default=True, description="Anonymize addresses")
    anonymize_phone_numbers: bool = Field(default=True, description="Anonymize phone numbers")
    anonymize_emails: bool = Field(default=True, description="Anonymize email addresses")
    anonymize_ids: bool = Field(default=True, description="Anonymize ID numbers")
    
    # Compliance
    gdpr_compliance: bool = Field(default=True, description="Ensure GDPR compliance")
    hipaa_compliance: bool = Field(default=False, description="Ensure HIPAA compliance")
    ccpa_compliance: bool = Field(default=False, description="Ensure CCPA compliance")
    
    # Differential privacy
    enable_differential_privacy: bool = Field(default=False, description="Enable differential privacy")
    epsilon: float = Field(default=1.0, ge=0.1, le=10.0, description="Privacy budget epsilon")
    delta: float = Field(default=1e-5, ge=1e-10, le=1e-3, description="Privacy budget delta")
    
    class Config:
        use_enum_values = True


class NoiseConfig(BaseModel):
    """OCR noise simulation configuration"""
    enabled: bool = Field(default=False, description="Enable noise simulation")
    noise_types: List[NoiseType] = Field(default_factory=list, description="Types of noise to apply")
    intensity: float = Field(default=0.1, ge=0.0, le=1.0, description="Noise intensity")
    
    # Specific noise parameters
    gaussian_sigma: float = Field(default=1.0, ge=0.1, le=5.0, description="Gaussian noise sigma")
    blur_radius: float = Field(default=1.0, ge=0.1, le=5.0, description="Blur radius")
    skew_angle: float = Field(default=2.0, ge=0.0, le=15.0, description="Skew angle in degrees")
    compression_quality: int = Field(default=85, ge=10, le=100, description="JPEG compression quality")
    
    class Config:
        use_enum_values = True


class RenderingConfig(BaseModel):
    """Document rendering configuration"""
    quality: RenderingQuality = RenderingQuality.STANDARD
    dpi: int = Field(default=300, ge=72, le=1200, description="Rendering DPI")
    color_mode: str = Field(default="RGB", description="Color mode")
    
    # Image settings
    image_format: str = Field(default="PNG", description="Image format for renders")
    image_compression: bool = Field(default=True, description="Enable image compression")
    
    # PDF settings
    pdf_version: str = Field(default="1.7", description="PDF version")
    embed_fonts: bool = Field(default=True, description="Embed fonts in PDF")
    
    class Config:
        use_enum_values = True


class PerformanceConfig(BaseModel):
    """Performance and resource configuration"""
    max_parallel_jobs: int = Field(default=4, ge=1, le=32, description="Maximum parallel generation jobs")
    memory_limit_mb: int = Field(default=2048, ge=512, le=16384, description="Memory limit in MB")
    timeout_seconds: int = Field(default=300, ge=30, le=3600, description="Generation timeout in seconds")
    
    # Caching
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_ttl_hours: int = Field(default=24, ge=1, le=168, description="Cache TTL in hours")
    
    # Resource management
    auto_cleanup: bool = Field(default=True, description="Auto cleanup temporary files")
    max_temp_files: int = Field(default=100, ge=10, le=1000, description="Maximum temporary files")
    
    class Config:
        use_enum_values = True


class OutputConfig(BaseModel):
    """Output configuration"""
    base_directory: str = Field(default="./output", description="Base output directory")
    create_subdirectories: bool = Field(default=True, description="Create subdirectories by type")
    file_naming_pattern: str = Field(
        default="{document_type}_{timestamp}_{index}",
        description="File naming pattern"
    )
    
    # File options
    include_metadata_file: bool = Field(default=True, description="Include metadata JSON file")
    include_annotations: bool = Field(default=True, description="Include annotation files")
    compress_output: bool = Field(default=False, description="Compress output files")
    
    # Formats
    export_formats: List[DocumentFormat] = Field(
        default=[DocumentFormat.PDF],
        description="Document formats to export"
    )
    
    class Config:
        use_enum_values = True


class ValidationConfig(BaseModel):
    """Document validation configuration"""
    enabled: bool = Field(default=True, description="Enable document validation")
    validation_types: List[str] = Field(
        default=["structural", "completeness"],
        description="Types of validation to perform"
    )
    
    # Validation thresholds
    min_quality_score: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum quality score")
    strict_mode: bool = Field(default=False, description="Enable strict validation mode")
    
    # Auto-correction
    auto_fix_issues: bool = Field(default=False, description="Automatically fix validation issues")
    max_fix_attempts: int = Field(default=3, ge=1, le=10, description="Maximum fix attempts")


class GenerationConfig(BaseModel):
    """Comprehensive generation configuration"""
    mode: GenerationMode = GenerationMode.BALANCED
    
    # Sub-configurations
    layout: LayoutConfig = Field(default_factory=LayoutConfig)
    content: ContentConfig = Field(default_factory=ContentConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    rendering: RenderingConfig = Field(default_factory=RenderingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    
    # Custom settings
    custom_settings: Dict[str, Any] = Field(default_factory=dict, description="Custom configuration settings")
    
    class Config:
        use_enum_values = True
    
    @classmethod
    def fast_mode(cls) -> 'GenerationConfig':
        """Create configuration optimized for speed"""
        config = cls(mode=GenerationMode.FAST)
        config.layout.complexity = LayoutComplexity.SIMPLE
        config.content.quality = ContentQuality.BASIC
        config.rendering.quality = RenderingQuality.DRAFT
        config.performance.max_parallel_jobs = 8
        config.validation.strict_mode = False
        return config
    
    @classmethod
    def quality_mode(cls) -> 'GenerationConfig':
        """Create configuration optimized for quality"""
        config = cls(mode=GenerationMode.QUALITY)
        config.layout.complexity = LayoutComplexity.COMPLEX
        config.content.quality = ContentQuality.PREMIUM
        config.rendering.quality = RenderingQuality.PRINT
        config.rendering.dpi = 600
        config.performance.max_parallel_jobs = 2
        config.validation.strict_mode = True
        config.validation.min_quality_score = 0.9
        return config
    
    @classmethod
    def privacy_focused(cls) -> 'GenerationConfig':
        """Create configuration with maximum privacy protection"""
        config = cls()
        config.privacy.level = PrivacyLevel.MAXIMUM
        config.privacy.enable_differential_privacy = True
        config.privacy.epsilon = 0.5
        config.privacy.gdpr_compliance = True
        config.privacy.hipaa_compliance = True
        config.privacy.ccpa_compliance = True
        return config
    
    def apply_document_type_defaults(self, document_type: DocumentType) -> None:
        """Apply document type specific defaults"""
        if document_type == DocumentType.ACADEMIC_PAPER:
            self.layout.columns = 2
            self.layout.font_size = 11.0
            self.content.quality = ContentQuality.HIGH
            self.content.min_words_per_paragraph = 30
            self.content.max_words_per_paragraph = 200
        
        elif document_type == DocumentType.BUSINESS_FORM:
            self.layout.complexity = LayoutComplexity.MODERATE
            self.content.quality = ContentQuality.STANDARD
            self.content.include_entities = True
            self.privacy.anonymize_names = True
        
        elif document_type == DocumentType.MEDICAL_RECORD:
            self.privacy.level = PrivacyLevel.MAXIMUM
            self.privacy.hipaa_compliance = True
            self.privacy.enable_differential_privacy = True
            self.content.domain_specific = True
        
        elif document_type == DocumentType.FINANCIAL_REPORT:
            self.layout.complexity = LayoutComplexity.COMPLEX
            self.content.quality = ContentQuality.HIGH
            self.validation.strict_mode = True
            self.rendering.quality = RenderingQuality.PRINT


# Predefined configuration presets
CONFIG_PRESETS = {
    "development": GenerationConfig.fast_mode(),
    "production": GenerationConfig.quality_mode(),
    "privacy": GenerationConfig.privacy_focused(),
    "default": GenerationConfig()
}


def get_config_preset(preset_name: str) -> GenerationConfig:
    """Get a predefined configuration preset"""
    if preset_name not in CONFIG_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available presets: {list(CONFIG_PRESETS.keys())}")
    return CONFIG_PRESETS[preset_name].copy()


def create_custom_config(
    base_preset: str = "default",
    **overrides
) -> GenerationConfig:
    """
    Create a custom configuration based on a preset.
    
    Args:
        base_preset: Base configuration preset to start with
        **overrides: Configuration overrides
    
    Returns:
        Custom GenerationConfig
    """
    config = get_config_preset(base_preset)
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            config.custom_settings[key] = value
    
    return config