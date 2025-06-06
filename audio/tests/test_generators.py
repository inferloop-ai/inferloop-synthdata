# tests/test_generators.py
"""
Test suite for audio generators
"""

import pytest
import torch
import torchaudio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from audio_synth.core.generators.diffusion import DiffusionAudioGenerator
from audio_synth.core.generators.tts import TTSGenerator
from audio_synth.core.generators.gan import GANAudioGenerator
from audio_synth.core.generators.vae import VAEAudioGenerator
from audio_synth.core.utils.config import AudioConfig, GenerationConfig
from audio_synth.core.utils.metrics import AudioMetrics

class TestDiffusionGenerator:
    """Test suite for diffusion audio generator"""
    
    @pytest.fixture
    def audio_config(self):
        return AudioConfig(
            sample_rate=22050,
            duration=2.0,
            channels=1
        )
    
    @pytest.fixture
    def generation_config(self):
        return GenerationConfig(
            method="diffusion",
            num_samples=1,
            privacy_level="medium"
        )
    
    @pytest.fixture
    def generator(self, generation_config, audio_config):
        return DiffusionAudioGenerator(generation_config, audio_config)
    
    def test_initialization(self, generator):
        """Test generator initialization"""
        assert generator.config.method == "diffusion"
        assert generator.audio_config.sample_rate == 22050
        assert generator.model is None
    
    def test_generate_single_sample(self, generator):
        """Test generating a single audio sample"""
        audio = generator.generate(prompt="Test prompt")
        
        expected_length = int(generator.audio_config.sample_rate * generator.audio_config.duration)
        assert audio.shape == torch.Size([expected_length])
        assert audio.dtype == torch.float32
    
    def test_generate_with_conditions(self, generator):
        """Test generation with specific conditions"""
        conditions = {
            "speaker_id": "test_speaker",
            "demographics": {
                "gender": "female",
                "age_group": "adult"
            }
        }
        
        audio = generator.generate(
            prompt="Test with conditions",
            conditions=conditions
        )
        
        assert audio is not None
        assert len(audio.shape) == 1
    
    def test_generate_batch(self, generator):
        """Test batch generation"""
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        audios = generator.generate_batch(prompts)
        
        assert len(audios) == 3
        for audio in audios:
            assert isinstance(audio, torch.Tensor)
            assert len(audio.shape) == 1
    
    def test_privacy_constraints(self, generator):
        """Test privacy constraint application"""
        generator.config.privacy_level = "high"
        
        audio_high = generator.generate(prompt="Privacy test")
        
        generator.config.privacy_level = "low"
        audio_low = generator.generate(prompt="Privacy test")
        
        # High privacy should produce different results than low privacy
        assert not torch.equal(audio_high, audio_low)
    
    @pytest.mark.slow
    def test_model_loading(self, generator):
        """Test model loading (mocked)"""
        with patch.object(generator, 'load_model') as mock_load:
            generator.load_model("./models/test_model.pt")
            mock_load.assert_called_once_with("./models/test_model.pt")

class TestTTSGenerator:
    """Test suite for TTS generator"""
    
    @pytest.fixture
    def tts_generator(self):
        audio_config = AudioConfig(sample_rate=22050, duration=3.0)
        gen_config = GenerationConfig(method="tts")
        return TTSGenerator(gen_config, audio_config)
    
    def test_text_to_speech_generation(self, tts_generator):
        """Test basic text-to-speech generation"""
        text = "Hello, this is a test of text-to-speech synthesis."
        audio = tts_generator.generate(prompt=text)
        
        assert audio is not None
        assert isinstance(audio, torch.Tensor)
        assert len(audio.shape) == 1
    
    def test_speaker_conditioning(self, tts_generator):
        """Test speaker-conditioned generation"""
        text = "Speaker conditioning test"
        
        # Generate with different speakers
        audio1 = tts_generator.generate(
            prompt=text,
            conditions={"speaker_id": "speaker_1"}
        )
        
        audio2 = tts_generator.generate(
            prompt=text,
            conditions={"speaker_id": "speaker_2"}
        )
        
        # Different speakers should produce different audio
        assert not torch.equal(audio1, audio2)
    
    def test_multilingual_support(self, tts_generator):
        """Test multilingual text-to-speech"""
        test_cases = [
            ("Hello world", "en"),
            ("Bonjour le monde", "fr"),
            ("Hola mundo", "es")
        ]
        
        for text, language in test_cases:
            audio = tts_generator.generate(
                prompt=text,
                conditions={"language": language}
            )
            assert audio is not None

class TestGANGenerator:
    """Test suite for GAN audio generator"""
    
    @pytest.fixture
    def gan_generator(self):
        audio_config = AudioConfig()
        gen_config = GenerationConfig(method="gan")
        return GANAudioGenerator(gen_config, audio_config)
    
    def test_latent_space_generation(self, gan_generator):
        """Test generation from latent space"""
        # Test with random latent vector
        latent_dim = 128
        latent_vector = torch.randn(latent_dim)
        
        audio = gan_generator.generate_from_latent(latent_vector)
        assert audio is not None
        assert isinstance(audio, torch.Tensor)
    
    def test_conditional_generation(self, gan_generator):
        """Test conditional GAN generation"""
        conditions = {
            "class_label": "speech",
            "style": "formal"
        }
        
        audio = gan_generator.generate(conditions=conditions)
        assert audio is not None
    
    def test_interpolation(self, gan_generator):
        """Test latent space interpolation"""
        latent1 = torch.randn(128)
        latent2 = torch.randn(128)
        
        interpolated_audios = gan_generator.interpolate(
            latent1, latent2, steps=5
        )
        
        assert len(interpolated_audios) == 5
        for audio in interpolated_audios:
            assert isinstance(audio, torch.Tensor)

# ============================================================================




