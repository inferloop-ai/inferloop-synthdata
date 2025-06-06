# tests/test_validators.py
"""
Test suite for audio validators
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock

from audio_synth.core.validators.quality import QualityValidator
from audio_synth.core.validators.privacy import PrivacyValidator
from audio_synth.core.validators.fairness import FairnessValidator

class TestQualityValidator:
    """Test suite for quality validator"""
    
    @pytest.fixture
    def quality_validator(self):
        config = {"quality_threshold": 0.7}
        return QualityValidator(config)
    
    @pytest.fixture
    def sample_audio(self):
        """Generate sample audio for testing"""
        duration = 2.0
        sample_rate = 22050
        samples = int(duration * sample_rate)
        
        # Generate a simple sine wave
        t = torch.linspace(0, duration, samples)
        frequency = 440  # A4 note
        audio = torch.sin(2 * torch.pi * frequency * t) * 0.5
        
        return audio
    
    def test_snr_calculation(self, quality_validator, sample_audio):
        """Test SNR calculation"""
        metadata = {"type": "clean_speech"}
        
        metrics = quality_validator.validate(sample_audio, metadata)
        
        assert "snr" in metrics
        assert isinstance(metrics["snr"], float)
        assert metrics["snr"] > 0  # Clean sine wave should have good SNR
    
    def test_spectral_metrics(self, quality_validator, sample_audio):
        """Test spectral analysis metrics"""
        metadata = {"type": "tonal"}
        
        metrics = quality_validator.validate(sample_audio, metadata)
        
        expected_metrics = [
            "spectral_centroid",
            "spectral_rolloff",
            "loudness",
            "realism_score"
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
    
    def test_batch_validation(self, quality_validator):
        """Test batch validation"""
        # Create multiple audio samples
        audios = [torch.randn(22050) for _ in range(3)]
        metadata = [{"sample_id": i} for i in range(3)]
        
        results = quality_validator.validate_batch(audios, metadata)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict)
            assert "snr" in result
    
    def test_realism_scoring(self, quality_validator):
        """Test realism scoring"""
        # Create realistic vs unrealistic audio
        realistic_audio = torch.sin(2 * torch.pi * 440 * torch.linspace(0, 1, 22050))
        unrealistic_audio = torch.randn(22050) * 2  # Noisy, clipped audio
        
        realistic_score = quality_validator._calculate_realism_score(realistic_audio)
        unrealistic_score = quality_validator._calculate_realism_score(unrealistic_audio)
        
        assert realistic_score > unrealistic_score

class TestPrivacyValidator:
    """Test suite for privacy validator"""
    
    @pytest.fixture
    def privacy_validator(self):
        config = {"privacy_threshold": 0.8}
        return PrivacyValidator(config)
    
    def test_speaker_anonymity(self, privacy_validator):
        """Test speaker anonymity calculation"""
        # Simulate original and anonymized audio
        original_audio = torch.sin(2 * torch.pi * 200 * torch.linspace(0, 2, 44100))
        anonymized_audio = original_audio + torch.randn_like(original_audio) * 0.1
        
        metadata = {"original_speaker": "speaker_001"}
        
        # Test anonymized audio
        metrics = privacy_validator.validate(anonymized_audio, metadata)
        
        assert "speaker_anonymity" in metrics
        assert 0 <= metrics["speaker_anonymity"] <= 1
    
    def test_privacy_leakage_detection(self, privacy_validator):
        """Test privacy leakage detection"""
        # Create audio with identifiable patterns
        identifiable_audio = torch.cat([
            torch.sin(2 * torch.pi * 440 * torch.linspace(0, 0.5, 11025))
        ] * 4)  # Repeating pattern
        
        random_audio = torch.randn(44100)
        
        metadata = {"type": "test"}
        
        identifiable_metrics = privacy_validator.validate(identifiable_audio, metadata)
        random_metrics = privacy_validator.validate(random_audio, metadata)
        
        # Identifiable audio should have higher privacy leakage
        assert identifiable_metrics["privacy_leakage"] > random_metrics["privacy_leakage"]
    
    def test_voice_conversion_quality(self, privacy_validator):
        """Test voice conversion quality assessment"""
        # Simulate voice-converted audio
        converted_audio = torch.randn(22050) * 0.5
        
        metadata = {"conversion_method": "pitch_shift"}
        metrics = privacy_validator.validate(converted_audio, metadata)
        
        assert "voice_conversion_quality" in metrics
        assert 0 <= metrics["voice_conversion_quality"] <= 1

class TestFairnessValidator:
    """Test suite for fairness validator"""
    
    @pytest.fixture
    def fairness_validator(self):
        config = {
            "fairness_threshold": 0.75,
            "protected_attributes": ["gender", "age", "accent"]
        }
        return FairnessValidator(config)
    
    def test_individual_fairness(self, fairness_validator):
        """Test individual fairness metrics"""
        audio = torch.randn(22050)
        metadata = {
            "demographics": {
                "gender": "female",
                "age_group": "adult",
                "accent": "american"
            }
        }
        
        metrics = fairness_validator.validate(audio, metadata)
        
        assert "representation_quality" in metrics
        assert "bias_score" in metrics
        assert 0 <= metrics["bias_score"] <= 1
    
    def test_group_fairness(self, fairness_validator):
        """Test group fairness across demographics"""
        # Create samples for different groups
        audios = [torch.randn(22050) for _ in range(6)]
        metadata = [
            {"demographics": {"gender": "male", "age_group": "adult"}},
            {"demographics": {"gender": "female", "age_group": "adult"}},
            {"demographics": {"gender": "male", "age_group": "elderly"}},
            {"demographics": {"gender": "female", "age_group": "elderly"}},
            {"demographics": {"gender": "other", "age_group": "adult"}},
            {"demographics": {"gender": "male", "age_group": "child"}}
        ]
        
        results = fairness_validator.validate_batch(audios, metadata)
        
        assert len(results) == 6
        
        # Check that group metrics are included
        first_result = results[0]
        assert "demographic_parity" in first_result
        assert "diversity_score" in first_result
    
    def test_demographic_parity(self, fairness_validator):
        """Test demographic parity calculation"""
        # Create samples with varying quality for different groups
        high_quality_audio = torch.sin(2 * torch.pi * 440 * torch.linspace(0, 1, 22050))
        low_quality_audio = torch.randn(22050) * 0.1
        
        audios = [high_quality_audio, low_quality_audio, high_quality_audio]
        metadata = [
            {"demographics": {"gender": "male"}},
            {"demographics": {"gender": "female"}},
            {"demographics": {"gender": "male"}}
        ]
        
        parity_score = fairness_validator._calculate_demographic_parity(audios, metadata)
        
        assert 0 <= parity_score <= 1
        # Should be lower due to quality differences between groups
        assert parity_score < 1.0

# ============================================================================
