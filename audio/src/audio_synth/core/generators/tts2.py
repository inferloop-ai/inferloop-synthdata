# audio_synth/core/generators/tts.py
"""
Text-to-Speech (TTS) Audio Generator
"""

import torch
import torch.nn as nn
import torchaudio
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging

from .base import AudioGenerator

logger = logging.getLogger(__name__)

class TTSAudioGenerator(AudioGenerator):
    """Text-to-Speech generator"""
    
    def __init__(self, 
                 sample_rate: int = 22050,
                 device: Optional[torch.device] = None,
                 model_path: Optional[str] = None):
        """
        Initialize TTS generator
        
        Args:
            sample_rate: Target sample rate
            device: Torch device
            model_path: Path to TTS model
        """
        super().__init__(sample_rate=sample_rate, device=device)
        
        self.tts_model = None
        self.vocoder = None
        self.processor = None
        self.speaker_embeddings = {}
        
        # Initialize with simple fallback TTS
        self._init_fallback_tts()
        
        # Try to load advanced models if available
        if model_path:
            self._load_advanced_models(model_path)
    
    def _init_fallback_tts(self):
        """Initialize simple fallback TTS"""
        # Create basic speaker embeddings
        self.speaker_embeddings = {
            "default": torch.randn(128),
            "male_adult": torch.randn(128),
            "female_adult": torch.randn(128),
            "child": torch.randn(128)
        }
        logger.info("Initialized fallback TTS generator")
    
    def _load_advanced_models(self, model_path: str):
        """Load advanced TTS models if available"""
        try:
            # Try to load HuggingFace transformers models
            from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
            
            self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self.tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            
            logger.info("Loaded advanced TTS models")
            
        except ImportError:
            logger.warning("Advanced TTS models not available, using fallback")
        except Exception as e:
            logger.warning(f"Failed to load advanced TTS models: {e}")
    
    def generate(self,
                prompt: Optional[str] = None,
                num_samples: int = 1,
                seed: Optional[int] = None,
                speaker_id: Optional[str] = None,
                voice_characteristics: Optional[Dict[str, Any]] = None,
                **kwargs) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Generate speech from text
        
        Args:
            prompt: Text to synthesize
            num_samples: Number of samples to generate
            seed: Random seed
            speaker_id: Speaker identifier
            voice_characteristics: Voice characteristics dict
            **kwargs: Additional parameters
            
        Returns:
            Tuple containing:
                - List of audio tensors
                - Dict with metadata
        """
        if prompt is None:
            prompt = "Hello, this is a test of text-to-speech synthesis."
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        # Get speaker embedding
        speaker_embedding = self._get_speaker_embedding(speaker_id, voice_characteristics)
        
        audios = []
        
        # Generate multiple samples with variation
        for i in range(num_samples):
            if self.tts_model is not None:
                # Use advanced TTS model
                audio = self._generate_with_advanced_tts(prompt, speaker_embedding)
            else:
                # Use fallback TTS
                audio = self._generate_with_fallback_tts(prompt, speaker_embedding, voice_characteristics)
            
            # Add slight variation for multiple samples
            if num_samples > 1:
                audio = self._add_variation(audio, variation_factor=i * 0.1)
            
            audios.append(audio)
        
        metadata = {
            "model_type": "tts",
            "sample_rate": self.sample_rate,
            "prompt": prompt,
            "speaker_id": speaker_id,
            "voice_characteristics": voice_characteristics,
            "seed": seed
        }
        
        return audios, metadata
    
    def _get_speaker_embedding(self, 
                              speaker_id: Optional[str], 
                              voice_characteristics: Optional[Dict[str, Any]]) -> torch.Tensor:
        """Get speaker embedding based on ID and characteristics"""
        
        if speaker_id and speaker_id in self.speaker_embeddings:
            base_embedding = self.speaker_embeddings[speaker_id]
        else:
            # Determine embedding based on voice characteristics
            if voice_characteristics:
                gender = voice_characteristics.get("gender", "default")
                age_group = voice_characteristics.get("age_group", "adult")
                
                if age_group == "child":
                    embedding_key = "child"
                elif gender in ["male", "female"]:
                    embedding_key = f"{gender}_{age_group}"
                else:
                    embedding_key = "default"
            else:
                embedding_key = "default"
            
            base_embedding = self.speaker_embeddings.get(embedding_key, self.speaker_embeddings["default"])
        
        # Add some random variation to the embedding
        embedding = base_embedding.clone()
        if voice_characteristics:
            # Modify embedding based on characteristics
            pitch_factor = voice_characteristics.get("pitch_factor", 1.0)
            speed_factor = voice_characteristics.get("speed_factor", 1.0)
            
            embedding[:32] *= pitch_factor  # Modify pitch-related dimensions
            embedding[32:64] *= speed_factor  # Modify speed-related dimensions
        
        return embedding
    
    def _generate_with_advanced_tts(self, text: str, speaker_embedding: torch.Tensor) -> torch.Tensor:
        """Generate speech using advanced TTS model"""
        try:
            # Process text
            inputs = self.processor(text=text, return_tensors="pt")
            
            # Generate speech
            with torch.no_grad():
                speech = self.tts_model.generate_speech(
                    inputs["input_ids"], 
                    speaker_embedding.unsqueeze(0)
                )
            
            # Apply vocoder
            with torch.no_grad():
                audio = self.vocoder(speech)
            
            # Ensure correct shape and sample rate
            if len(audio.shape) > 1:
                audio = audio.squeeze()
            
            # Resample if necessary
            if hasattr(self.vocoder, 'config') and hasattr(self.vocoder.config, 'sampling_rate'):
                if self.vocoder.config.sampling_rate != self.sample_rate:
                    audio = torchaudio.functional.resample(
                        audio,
                        self.vocoder.config.sampling_rate,
                        self.sample_rate
                    )
            
            return audio
            
        except Exception as e:
            logger.warning(f"Advanced TTS generation failed: {e}, using fallback")
            return self._generate_with_fallback_tts(text, speaker_embedding, {})
    
    def _generate_with_fallback_tts(self, 
                                   text: str, 
                                   speaker_embedding: torch.Tensor,
                                   voice_characteristics: Optional[Dict[str, Any]]) -> torch.Tensor:
        """Generate speech using fallback TTS"""
        
        # Estimate duration based on text length
        duration = max(len(text) * 0.08, 1.0)  # ~0.08s per character, minimum 1s
        samples = int(duration * self.sample_rate)
        
        # Generate time vector
        t = torch.linspace(0, duration, samples)
        
        # Determine voice characteristics from embedding and parameters
        characteristics = voice_characteristics or {}
        
        # Base fundamental frequency from speaker embedding
        f0_base = 100 + torch.mean(speaker_embedding[:16]).item() * 50  # 100-150 Hz range
        
        # Modify based on characteristics
        pitch_factor = characteristics.get("pitch_factor", 1.0)
        speed_factor = characteristics.get("speed_factor", 1.0)
        
        fundamental = f0_base * pitch_factor
        fundamental = np.clip(fundamental, 80, 400)  # Reasonable F0 range
        
        # Generate speech-like audio with formants
        audio = torch.zeros(samples)
        
        # Add harmonics (formants)
        formants = [fundamental, fundamental * 2.2, fundamental * 3.8, fundamental * 5.1]
        amplitudes = [0.4, 0.25, 0.15, 0.1]
        
        for formant, amplitude in zip(formants, amplitudes):
            # Add frequency modulation for more natural sound
            freq_mod = 1.0 + 0.1 * torch.sin(2 * torch.pi * 2 * t)  # 2 Hz vibrato
            modulated_freq = formant * freq_mod
            
            harmonic = amplitude * torch.sin(2 * torch.pi * modulated_freq * t)
            audio += harmonic
        
        # Apply speech envelope
        envelope = self._generate_speech_envelope(samples, text, speed_factor)
        audio = audio * envelope
        
        # Add noise for realism
        noise_level = 0.02
        audio += torch.randn_like(audio) * noise_level
        
        # Apply simple spectral shaping
        audio = self._apply_spectral_shaping(audio)
        
        return audio
    
    def _generate_speech_envelope(self, samples: int, text: str, speed_factor: float = 1.0) -> torch.Tensor:
        """Generate speech amplitude envelope"""
        
        envelope = torch.ones(samples)
        
        # Create pauses for punctuation and spaces
        pause_positions = []
        
        # Find punctuation marks
        for i, char in enumerate(text):
            if char in ".,;:!?":
                pos = int((i / len(text)) * samples)
                pause_positions.append((pos, 0.3, 1500))  # position, depth, duration
            elif char == " ":
                pos = int((i / len(text)) * samples)
                if i > 0 and text[i-1] not in ".,;:!?":  # Don't add space pause after punctuation
                    pause_positions.append((pos, 0.8, 300))  # shorter pause for spaces
        
        # Apply pauses
        for pos, depth, duration in pause_positions:
            duration = int(duration / speed_factor)  # Adjust for speed
            start = max(0, pos - duration // 2)
            end = min(samples, pos + duration // 2)
            
            if end > start:
                # Create fade out and fade in
                fade_samples = min(duration // 4, 200)
                
                # Fade out
                if start + fade_samples < samples:
                    fade_out = torch.linspace(1.0, depth, fade_samples)
                    envelope[start:start + fade_samples] *= fade_out
                
                # Hold at low level
                envelope[start + fade_samples:end - fade_samples] *= depth
                
                # Fade in
                if end - fade_samples > 0:
                    fade_in = torch.linspace(depth, 1.0, fade_samples)
                    envelope[end - fade_samples:end] *= fade_in
        
        # Apply overall amplitude modulation for naturalness
        mod_freq = 5.0 / speed_factor  # Adjust modulation for speed
        amplitude_mod = 1.0 + 0.1 * torch.sin(2 * torch.pi * mod_freq * torch.linspace(0, len(envelope) / self.sample_rate, len(envelope)))
        envelope *= amplitude_mod
        
        # Smooth the envelope
        envelope = self._smooth_envelope(envelope)
        
        return envelope
    
    def _smooth_envelope(self, envelope: torch.Tensor) -> torch.Tensor:
        """Smooth the envelope to avoid artifacts"""
        # Simple moving average smoothing
        kernel_size = 50
        kernel = torch.ones(kernel_size) / kernel_size
        
        # Pad the envelope
        padded = torch.nn.functional.pad(envelope.unsqueeze(0).unsqueeze(0), 
                                        (kernel_size//2, kernel_size//2), 
                                        mode='reflect')
        
        # Apply convolution
        smoothed = torch.nn.functional.conv1d(padded, kernel.unsqueeze(0).unsqueeze(0))
        
        return smoothed.squeeze().squeeze()
    
    def _apply_spectral_shaping(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply spectral shaping to make audio more speech-like"""
        
        # Apply STFT
        stft = torch.stft(audio, n_fft=1024, hop_length=256, return_complex=True)
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Apply speech-like spectral envelope
        freq_bins = magnitude.shape[0]
        frequencies = torch.linspace(0, self.sample_rate / 2, freq_bins)
        
        # Create spectral envelope that emphasizes speech frequencies
        spectral_envelope = torch.ones_like(frequencies)
        
        # Emphasize 300-3000 Hz range (speech formants)
        speech_mask = (frequencies >= 300) & (frequencies <= 3000)
        spectral_envelope[speech_mask] *= 1.5
        
        # Attenuate very high frequencies
        high_freq_mask = frequencies > 8000
        spectral_envelope[high_freq_mask] *= 0.3
        
        # Apply envelope
        shaped_magnitude = magnitude * spectral_envelope.unsqueeze(1)
        
        # Reconstruct audio
        shaped_stft = shaped_magnitude * torch.exp(1j * phase)
        shaped_audio = torch.istft(shaped_stft, n_fft=1024, hop_length=256)
        
        return shaped_audio
    
    def _add_variation(self, audio: torch.Tensor, variation_factor: float = 0.1) -> torch.Tensor:
        """Add slight variation to audio for multiple samples"""
        
        # Add pitch variation
        pitch_shift = 1.0 + (torch.randn(1).item() - 0.5) * variation_factor * 0.2
        
        # Simple pitch shifting by resampling
        if pitch_shift != 1.0:
            # Resample to simulate pitch shift
            new_length = int(len(audio) / pitch_shift)
            if new_length > 0:
                audio_resampled = torch.nn.functional.interpolate(
                    audio.unsqueeze(0).unsqueeze(0),
                    size=new_length,
                    mode='linear'
                ).squeeze()
                
                # Pad or truncate to original length
                if len(audio_resampled) > len(audio):
                    audio = audio_resampled[:len(audio)]
                else:
                    audio = torch.nn.functional.pad(audio_resampled, (0, len(audio) - len(audio_resampled)))
        
        # Add slight timing variation
        timing_shift = int((torch.randn(1).item()) * variation_factor * 1000)
        if timing_shift != 0:
            if timing_shift > 0:
                audio = torch.cat([torch.zeros(abs(timing_shift)), audio[:-abs(timing_shift)]])
            else:
                audio = torch.cat([audio[abs(timing_shift):], torch.zeros(abs(timing_shift))])
        
        # Add subtle noise variation
        noise_level = variation_factor * 0.01
        audio += torch.randn_like(audio) * noise_level
        
        return audio