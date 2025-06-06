# audio_synth/api/models/requests.py
"""
API request models
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum

class GenerationMethod(str, Enum):
    DIFFUSION = "diffusion"
    GAN = "gan"
    VAE = "vae"
    TTS = "tts"
    VOCODER = "vocoder"

class PrivacyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class AudioFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"

class ValidatorType(str, Enum):
    QUALITY = "quality"
    PRIVACY = "privacy"
    FAIRNESS = "fairness"
    PERCEPTUAL = "perceptual"

class AudioConfigRequest(BaseModel):
    """Audio configuration for requests"""
    sample_rate: int = Field(default=22050, ge=8000, le=48000, description="Sample rate in Hz")
    duration: float = Field(default=5.0, ge=0.1, le=300.0, description="Duration in seconds")
    channels: int = Field(default=1, ge=1, le=2, description="Number of channels")
    format: AudioFormat = Field(default=AudioFormat.WAV, description="Audio format")
    bit_depth: int = Field(default=16, ge=8, le=32, description="Bit depth")

class DemographicsRequest(BaseModel):
    """Demographics information for generation"""
    gender: Optional[str] = Field(None, regex="^(male|female|other)$", description="Gender")
    age_group: Optional[str] = Field(None, regex="^(child|adult|elderly)$", description="Age group")
    accent: Optional[str] = Field(None, description="Accent or dialect")
    language: str = Field(default="en", description="Language code")

class VoiceCharacteristicsRequest(BaseModel):
    """Voice characteristics for TTS"""
    pitch_factor: float = Field(default=1.0, ge=0.5, le=2.0, description="Pitch modification factor")
    speed_factor: float = Field(default=1.0, ge=0.5, le=2.0, description="Speed modification factor")
    emotion: Optional[str] = Field(None, description="Emotional tone")
    style: Optional[str] = Field(None, description="Speaking style")

class GenerationRequest(BaseModel):
    """Request for audio generation"""
    method: GenerationMethod = Field(default=GenerationMethod.DIFFUSION, description="Generation method")
    prompt: Optional[str] = Field(None, description="Text prompt for generation")
    num_samples: int = Field(default=1, ge=1, le=100, description="Number of samples to generate")
    audio_config: AudioConfigRequest = Field(default_factory=AudioConfigRequest, description="Audio configuration")
    privacy_level: PrivacyLevel = Field(default=PrivacyLevel.MEDIUM, description="Privacy protection level")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    speaker_id: Optional[str] = Field(None, description="Target speaker ID")
    demographics: Optional[DemographicsRequest] = Field(None, description="Demographic characteristics")
    voice_characteristics: Optional[VoiceCharacteristicsRequest] = Field(None, description="Voice characteristics")
    conditions: Optional[Dict[str, Any]] = Field(None, description="Additional generation conditions")
    
    @validator('prompt')
    def validate_prompt(cls, v, values):
        method = values.get('method')
        if method == GenerationMethod.TTS and not v:
            raise ValueError("Prompt is required for TTS generation")
        return v

class BatchGenerationRequest(BaseModel):
    """Request for batch audio generation"""
    requests: List[GenerationRequest] = Field(..., description="List of generation requests")
    batch_id: Optional[str] = Field(None, description="Optional batch identifier")
    parallel: bool = Field(default=False, description="Generate samples in parallel")
    
    @validator('requests')
    def validate_requests(cls, v):
        if len(v) == 0:
            raise ValueError("At least one generation request required")
        if len(v) > 50:
            raise ValueError("Maximum 50 requests per batch")
        return v

class ValidationRequest(BaseModel):
    """Request for audio validation"""
    validators: List[ValidatorType] = Field(default=[ValidatorType.QUALITY], description="Validators to run")
    thresholds: Optional[Dict[str, float]] = Field(None, description="Custom validation thresholds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @validator('thresholds')
    def validate_thresholds(cls, v):
        if v:
            for key, value in v.items():
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"Threshold {key} must be between 0 and 1")
        return v

class PrivacyEnhancementRequest(BaseModel):
    """Request for privacy enhancement"""
    privacy_level: PrivacyLevel = Field(default=PrivacyLevel.MEDIUM, description="Privacy enhancement level")
    target_speaker: Optional[str] = Field(None, description="Target speaker for voice conversion")
    preserve_quality: bool = Field(default=True, description="Attempt to preserve audio quality")
    techniques: Optional[List[str]] = Field(None, description="Specific privacy techniques to apply")

