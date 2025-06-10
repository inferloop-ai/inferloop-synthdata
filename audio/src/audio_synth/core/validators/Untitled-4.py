

# ============================================================================


# audio_synth/core/generators/vocoder.py
"""
Neural Vocoder for high-quality audio synthesis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
import numpy as np

from .base import BaseAudioGenerator
from ..utils.config import GenerationConfig, AudioConfig

class VocoderGenerator(BaseAudioGenerator):
    """Neural vocoder for mel-spectrogram to waveform conversion"""
    
    def __init__(self, config: GenerationConfig, audio_config: AudioConfig):
        super().__init__(config, audio_config)
        self.vocoder = None
        self.mel_channels = 80
        self.hop_length = 256
        self.win_length = 1024
        
    def load_model(self, model_path: str) -> None:
        """Load vocoder model"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Initialize HiFi-GAN vocoder
            self.vocoder = HiFiGANVocoder(
                mel_channels=self.mel_channels,
                sample_rate=self.audio_config.sample_rate
            )
            
            # Load weights
            self.vocoder.load_state_dict(checkpoint['generator'])
            self.vocoder.eval()
            
            print(f"Vocoder model loaded from {model_path}")
            
        except Exception as e:
            print(f"Could not load vocoder model: {e}")
            print("Using simple vocoder")
            self._initialize_simple_vocoder()
    
    def _initialize_simple_vocoder(self):
        """Initialize simple vocoder"""
        self.vocoder = SimpleVocoder(
            mel_channels=self.mel_channels,
            sample_rate=self.audio_config.sample_rate
        )
    
    def generate(self, 
                 prompt: Optional[str] = None,
                 conditions: Optional[Dict[str, Any]] = None,
                 **kwargs) -> torch.Tensor:
        """Generate audio from mel-spectrogram"""
        
        # Generate or load mel-spectrogram
        mel_spec = kwargs.get('mel_spectrogram')
        if mel_spec is None:
            mel_spec = self._generate_mel_spectrogram(prompt, conditions)
        
        # Convert mel to waveform
        audio = self.mel_to_waveform(mel_spec)
        
        return audio
    
    def generate_batch(self, 
                      prompts: List[str],
                      conditions: Optional[List[Dict[str, Any]]] = None,
                      **kwargs) -> List[torch.Tensor]:
        """Generate batch of audio from mel-spectrograms"""
        
        mel_specs = kwargs.get('mel_spectrograms')
        if mel_specs is None:
            mel_specs = [
                self._generate_mel_spectrogram(prompt, cond)
                for prompt, cond in zip(prompts, conditions or [{}] * len(prompts))
            ]
        
        audios = []
        for mel_spec in mel_specs:
            audio = self.mel_to_waveform(mel_spec)
            audios.append(audio)
        
        return audios
    
    def mel_to_waveform(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """Convert mel-spectrogram to waveform"""
        
        if self.vocoder is None:
            self._initialize_simple_vocoder()
        
        if len(mel_spectrogram.shape) == 2:
            mel_spectrogram = mel_spectrogram.unsqueeze(0)
        
        with torch.no_grad():
            audio = self.vocoder(mel_spectrogram)
        
        return audio.squeeze()
    
    def _generate_mel_spectrogram(self, 
                                 prompt: Optional[str], 
                                 conditions: Optional[Dict[str, Any]]) -> torch.Tensor:
        """Generate mel-spectrogram (placeholder - would use TTS model)"""
        
        # Calculate target mel-spectrogram dimensions
        duration = self.audio_config.duration
        frames = int(duration * self.audio_config.sample_rate / self.hop_length)
        
        # Generate realistic mel-spectrogram pattern
        mel_spec = torch.zeros(self.mel_channels, frames)
        
        # Add formant structure for speech-like content
        if prompt or (conditions and "speech" in conditions.get("content_type", "")):
            mel_spec = self._generate_speech_mel(frames, conditions)
        else:
            # Generate music-like mel-spectrogram
            mel_spec = self._generate_music_mel(frames, conditions)
        
        return mel_spec
    
    def _generate_speech_mel(self, frames: int, conditions: Optional[Dict[str, Any]]) -> torch.Tensor:
        """Generate speech-like mel-spectrogram"""
        
        mel_spec = torch.zeros(self.mel_channels, frames)
        
        # Get speaker characteristics
        demographics = conditions.get("demographics", {}) if conditions else {}
        gender = demographics.get("gender", "neutral")
        age_group = demographics.get("age_group", "adult")
        
        # Fundamental frequency based on demographics
        if gender == "male":
            f0_base = 120
        elif gender == "female":
            f0_base = 220
        elif age_group == "child":
            f0_base = 300
        else:
            f0_base = 170
        
        # Generate formant structure
        formants = [f0_base, f0_base * 2.5, f0_base * 4.0, f0_base * 6.0]
        
        for frame in range(frames):
            # Add pitch variation
            pitch_variation = np.sin(2 * np.pi * frame / 50) * 0.1 + 1.0
            
            for formant in formants:
                # Convert frequency to mel bin
                mel_bin = self._freq_to_mel_bin(formant * pitch_variation)
                if 0 <= mel_bin < self.mel_channels:
                    # Add formant energy with some spread
                    for offset in range(-2, 3):
                        bin_idx = mel_bin + offset
                        if 0 <= bin_idx < self.mel_channels:
                            energy = np.exp(-offset**2 / 2) * (0.5 + 0.5 * np.random.random())
                            mel_spec[bin_idx, frame] += energy
        
        # Add noise floor
        mel_spec += torch.randn_like(mel_spec) * 0.1
        
        # Apply temporal smoothing
        mel_spec = self._smooth_mel_spectrogram(mel_spec)
        
        return mel_spec
    
    def _generate_music_mel(self, frames: int, conditions: Optional[Dict[str, Any]]) -> torch.Tensor:
        """Generate music-like mel-spectrogram"""
        
        mel_spec = torch.zeros(self.mel_channels, frames)
        
        # Generate harmonic content
        fundamental = 220  # A3
        harmonics = [fundamental * i for i in range(1, 8)]
        
        for frame in range(frames):
            # Add chord progression
            chord_phase = (frame / frames) * 4  # 4 chord changes
            chord_offset = int(chord_phase) * 3  # Major third intervals
            
            for harmonic in harmonics:
                freq = harmonic * (1 + chord_offset / 12)  # Semitone adjustments
                mel_bin = self._freq_to_mel_bin(freq)
                
                if 0 <= mel_bin < self.mel_channels:
                    # Add harmonic energy
                    amplitude = 1.0 / harmonics.index(harmonic / fundamental) + 1
                    mel_spec[mel_bin, frame] += amplitude * 0.5
        
        # Add rhythmic modulation
        rhythm_pattern = torch.sin(torch.linspace(0, 8 * np.pi, frames))
        rhythm_pattern = (rhythm_pattern + 1) / 2  # Normalize to [0, 1]
        
        mel_spec = mel_spec * rhythm_pattern.unsqueeze(0)
        
        # Add noise
        mel_spec += torch.randn_like(mel_spec) * 0.05
        
        return mel_spec
    
    def _freq_to_mel_bin(self, frequency: float) -> int:
        """Convert frequency to mel-spectrogram bin"""
        # Mel scale conversion
        mel = 2595 * np.log10(1 + frequency / 700)
        
        # Convert to bin index
        mel_min = 0
        mel_max = 2595 * np.log10(1 + (self.audio_config.sample_rate / 2) / 700)
        
        normalized_mel = (mel - mel_min) / (mel_max - mel_min)
        bin_idx = int(normalized_mel * (self.mel_channels - 1))
        
        return bin_idx
    
    def _smooth_mel_spectrogram(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Apply temporal smoothing to mel-spectrogram"""
        
        # Apply Gaussian smoothing in time dimension
        kernel_size = 5
        kernel = torch.ones(1, 1, kernel_size) / kernel_size
        
        # Pad and apply convolution
        padded_mel = F.pad(mel_spec.unsqueeze(0), (kernel_size//2, kernel_size//2), mode='reflect')
        smoothed_mel = F.conv1d(padded_mel, kernel)
        
        return smoothed_mel.squeeze(0)

class HiFiGANVocoder(nn.Module):
    """HiFi-GAN based vocoder"""
    
    def __init__(self, mel_channels: int, sample_rate: int):
        super().__init__()
        
        self.mel_channels = mel_channels
        self.sample_rate = sample_rate
        
        # Initial convolution
        self.input_conv = nn.Conv1d(mel_channels, 512, kernel_size=7, padding=3)
        
        # Upsampling blocks
        self.upsample_blocks = nn.ModuleList([
            UpsampleBlock(512, 256, kernel_size=16, stride=8),
            UpsampleBlock(256, 128, kernel_size=16, stride=8),
            UpsampleBlock(128, 64, kernel_size=4, stride=2),
            UpsampleBlock(64, 32, kernel_size=4, stride=2)
        ])
        
        # Output convolution
        self.output_conv = nn.Conv1d(32, 1, kernel_size=7, padding=3)
        
    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Convert mel-spectrogram to waveform"""
        
        x = self.input_conv(mel_spec)
        
        for upsample_block in self.upsample_blocks:
            x = upsample_block(x)
        
        x = torch.tanh(self.output_conv(x))
        
        return x

class UpsampleBlock(nn.Module):
    """Upsampling block for HiFi-GAN"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        
        self.upsample = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=(kernel_size - stride) // 2
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(out_channels, 3, 1),
            ResidualBlock(out_channels, 3, 3),
            ResidualBlock(out_channels, 3, 5)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.upsample(x), 0.1)
        
        for res_block in self.res_blocks:
            x = x + res_block(x)
        
        return x

class ResidualBlock(nn.Module):
    """Residual block with dilated convolutions"""
    
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        
        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size, dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2
        )
        
        self.conv2 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size, dilation=1,
            padding=(kernel_size - 1) // 2
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x = F.leaky_relu(self.conv1(x), 0.1)
        x = F.leaky_relu(self.conv2(x), 0.1)
        
        return x

class SimpleVocoder(nn.Module):
    """Simple vocoder implementation"""
    
    def __init__(self, mel_channels: int, sample_rate: int):
        super().__init__()
        
        self.mel_channels = mel_channels
        self.sample_rate = sample_rate
        
        # Simple upsampling network
        self.network = nn.Sequential(
            nn.Conv1d(mel_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        return self.network(mel_spec)