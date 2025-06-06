# audio_synth/core/validators/quality.py
"""
Audio quality validation and assessment
"""

import torch
import torchaudio
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from scipy import signal
from scipy.stats import entropy
import math

from .base import BaseValidator

logger = logging.getLogger(__name__)

class QualityValidator(BaseValidator):
    """Validator for audio quality assessment"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.quality_threshold = config.get("quality_threshold", 0.7)
        self.sample_rate = config.get("sample_rate", 22050)
        
    def validate(self, audio: torch.Tensor, metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Validate audio quality
        
        Args:
            audio: Audio tensor [T]
            metadata: Metadata dictionary
            
        Returns:
            Dictionary with quality metrics
        """
        audio_np = audio.detach().cpu().numpy()
        
        metrics = {}
        
        # Signal-to-Noise Ratio
        metrics["snr"] = self._calculate_snr(audio_np)
        
        # Spectral metrics
        spectral_metrics = self._calculate_spectral_metrics(audio_np)
        metrics.update(spectral_metrics)
        
        # Perceptual metrics
        perceptual_metrics = self._calculate_perceptual_metrics(audio_np)
        metrics.update(perceptual_metrics)
        
        # Realism score
        metrics["realism_score"] = self._calculate_realism_score(audio_np)
        
        # Overall quality score
        metrics["overall_quality"] = self._calculate_overall_quality(metrics)
        
        return metrics
    
    def _calculate_snr(self, audio: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio"""
        try:
            # Estimate signal and noise
            # Simple approach: assume signal is the main component
            signal_power = np.mean(audio ** 2)
            
            # Estimate noise from high-frequency content
            # Apply high-pass filter to isolate noise
            nyquist = self.sample_rate // 2
            high_cutoff = 0.8  # 80% of Nyquist frequency
            b, a = signal.butter(4, high_cutoff, 'high')
            noise_estimate = signal.filtfilt(b, a, audio)
            noise_power = np.mean(noise_estimate ** 2)
            
            if noise_power > 0:
                snr_db = 10 * np.log10(signal_power / noise_power)
            else:
                snr_db = 60.0  # Very high SNR
            
            # Normalize to 0-1 range (assume SNR between -10 and 60 dB)
            snr_normalized = np.clip((snr_db + 10) / 70, 0, 1)
            
            return float(snr_normalized)
            
        except Exception as e:
            logger.warning(f"SNR calculation failed: {e}")
            return 0.5
    
    def _calculate_spectral_metrics(self, audio: np.ndarray) -> Dict[str, float]:
        """Calculate spectral quality metrics"""
        metrics = {}
        
        try:
            # Compute power spectral density
            freqs, psd = signal.welch(audio, fs=self.sample_rate, nperseg=1024)
            
            # Spectral centroid
            spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
            metrics["spectral_centroid"] = float(spectral_centroid / (self.sample_rate / 2))
            
            # Spectral rolloff (95% of energy)
            cumulative_energy = np.cumsum(psd)
            total_energy = cumulative_energy[-1]
            rolloff_idx = np.where(cumulative_energy >= 0.95 * total_energy)[0][0]
            spectral_rolloff = freqs[rolloff_idx]
            metrics["spectral_rolloff"] = float(spectral_rolloff / (self.sample_rate / 2))
            
            # Spectral flatness (measure of how noise-like vs tonal)
            # Geometric mean / Arithmetic mean
            geometric_mean = np.exp(np.mean(np.log(psd + 1e-10)))
            arithmetic_mean = np.mean(psd)
            spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
            metrics["spectral_flatness"] = float(spectral_flatness)
            
            # Zero crossing rate
            zero_crossings = np.where(np.diff(np.sign(audio)))[0]
            zcr = len(zero_crossings) / len(audio)
            metrics["zero_crossing_rate"] = float(zcr)
            
        except Exception as e:
            logger.warning(f"Spectral metrics calculation failed: {e}")
            metrics.update({
                "spectral_centroid": 0.5,
                "spectral_rolloff": 0.5,
                "spectral_flatness": 0.5,
                "zero_crossing_rate": 0.5
            })
        
        return metrics
    
    def _calculate_perceptual_metrics(self, audio: np.ndarray) -> Dict[str, float]:
        """Calculate perceptual quality metrics"""
        metrics = {}
        
        try:
            # Loudness (RMS energy)
            rms_energy = np.sqrt(np.mean(audio ** 2))
            metrics["loudness"] = float(np.clip(rms_energy * 10, 0, 1))
            
            # Dynamic range
            max_amplitude = np.max(np.abs(audio))
            min_amplitude = np.min(np.abs(audio[np.abs(audio) > 0.001]))  # Avoid zeros
            if min_amplitude > 0:
                dynamic_range = 20 * np.log10(max_amplitude / min_amplitude)
                metrics["dynamic_range"] = float(np.clip(dynamic_range / 60, 0, 1))
            else:
                metrics["dynamic_range"] = 0.5
            
            # Harmonic-to-Noise Ratio (simplified)
            # Use autocorrelation to find fundamental frequency
            autocorr = np.correlate(audio, audio, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find peaks in autocorrelation
            peaks, _ = signal.find_peaks(autocorr[1:], height=0.1 * np.max(autocorr))
            
            if len(peaks) > 0:
                # Estimate harmonicity based on autocorrelation peaks
                peak_strength = np.mean(autocorr[peaks + 1]) / np.mean(autocorr)
                metrics["harmonicity"] = float(np.clip(peak_strength, 0, 1))
            else:
                metrics["harmonicity"] = 0.3  # Low harmonicity
            
            # Temporal stability (variation in short-time energy)
            frame_size = 1024
            hop_size = 512
            frames = []
            for i in range(0, len(audio) - frame_size, hop_size):
                frame_energy = np.mean(audio[i:i + frame_size] ** 2)
                frames.append(frame_energy)
            
            if len(frames) > 1:
                energy_variation = np.std(frames) / (np.mean(frames) + 1e-10)
                temporal_stability = 1 / (1 + energy_variation)
                metrics["temporal_stability"] = float(temporal_stability)
            else:
                metrics["temporal_stability"] = 0.5
            
        except Exception as e:
            logger.warning(f"Perceptual metrics calculation failed: {e}")
            metrics.update({
                "loudness": 0.5,
                "dynamic_range": 0.5,
                "harmonicity": 0.5,
                "temporal_stability": 0.5
            })
        
        return metrics
    
    def _calculate_realism_score(self, audio: np.ndarray) -> float:
        """Calculate overall realism score"""
        try:
            # Combine multiple indicators of realism
            
            # 1. Amplitude distribution (should be roughly Gaussian for natural audio)
            hist, bins = np.histogram(audio, bins=50, density=True)
            # Compare to Gaussian
            gaussian_hist = signal.gaussian(50, std=np.std(audio))
            gaussian_hist = gaussian_hist / np.sum(gaussian_hist) * np.sum(hist)
            amplitude_score = 1 - np.mean(np.abs(hist - gaussian_hist)) / np.max(hist)
            
            # 2. Frequency content naturalness
            freqs, psd = signal.welch(audio, fs=self.sample_rate, nperseg=1024)
            
            # Natural audio should have 1/f characteristics in some ranges
            # Check if power decreases with frequency (roughly)
            low_freq_power = np.mean(psd[freqs < 1000])
            high_freq_power = np.mean(psd[freqs > 5000])
            
            if high_freq_power > 0:
                freq_score = np.clip(low_freq_power / high_freq_power / 10, 0, 1)
            else:
                freq_score = 0.5
            
            # 3. Clipping detection
            max_val = np.max(np.abs(audio))
            clipping_ratio = np.mean(np.abs(audio) > 0.95 * max_val)
            clipping_score = 1 - clipping_ratio
            
            # 4. Silence detection (too much silence is unrealistic)
            silence_threshold = 0.01 * np.max(np.abs(audio))
            silence_ratio = np.mean(np.abs(audio) < silence_threshold)
            silence_score = 1 - np.clip(silence_ratio - 0.1, 0, 1)  # Allow some silence
            
            # Combine scores
            realism_score = np.mean([
                amplitude_score * 0.25,
                freq_score * 0.35,
                clipping_score * 0.25,
                silence_score * 0.15
            ])
            
            return float(np.clip(realism_score, 0, 1))
            
        except Exception as e:
            logger.warning(f"Realism score calculation failed: {e}")
            return 0.5
    
    def _calculate_overall_quality(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score from individual metrics"""
        # Weighted combination of metrics
        weights = {
            "snr": 0.25,
            "spectral_centroid": 0.1,
            "spectral_flatness": 0.1,
            "loudness": 0.1,
            "dynamic_range": 0.15,
            "harmonicity": 0.1,
            "temporal_stability": 0.1,
            "realism_score": 0.2
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                total_score += metrics[metric] * weight
                total_weight += weight
        
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.5
    
    def validate_batch(self, 
                      audios: List[torch.Tensor], 
                      metadata: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Validate a batch of audio samples"""
        results = []
        
        for audio, meta in zip(audios, metadata):
            try:
                result = self.validate(audio, meta)
                results.append(result)
            except Exception as e:
                logger.error(f"Quality validation failed for sample: {e}")
                results.append(self._get_default_metrics())
        
        return results
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """Get default metrics when validation fails"""
        return {
            "snr": 0.5,
            "spectral_centroid": 0.5,
            "spectral_rolloff": 0.5,
            "spectral_flatness": 0.5,
            "zero_crossing_rate": 0.5,
            "loudness": 0.5,
            "dynamic_range": 0.5,
            "harmonicity": 0.5,
            "temporal_stability": 0.5,
            "realism_score": 0.5,
            "overall_quality": 0.5
        }