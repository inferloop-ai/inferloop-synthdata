# audio_synth/core/validators/privacy.py
"""
Privacy validation for synthetic audio
"""

import torch
import torchaudio
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from scipy import signal
from scipy.spatial.distance import cosine
import hashlib

from .base import BaseValidator

logger = logging.getLogger(__name__)

class PrivacyValidator(BaseValidator):
    """Validator for privacy preservation assessment"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.privacy_threshold = config.get("privacy_threshold", 0.8)
        self.sample_rate = config.get("sample_rate", 22050)
        
        # Speaker verification model (simplified)
        self.speaker_model = None
        self._init_speaker_model()
    
    def _init_speaker_model(self):
        """Initialize speaker verification model (placeholder)"""
        # In a real implementation, load a pre-trained speaker verification model
        # For now, use a simple feature extractor
        self.speaker_model = SimpleSpeakerEncoder()
    
    def validate(self, audio: torch.Tensor, metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Validate privacy preservation
        
        Args:
            audio: Audio tensor [T]
            metadata: Metadata dictionary
            
        Returns:
            Dictionary with privacy metrics
        """
        audio_np = audio.detach().cpu().numpy()
        
        metrics = {}
        
        # Speaker anonymity
        metrics["speaker_anonymity"] = self._calculate_speaker_anonymity(audio_np, metadata)
        
        # Privacy leakage detection
        metrics["privacy_leakage"] = self._calculate_privacy_leakage(audio_np, metadata)
        
        # Voice conversion quality (if applicable)
        metrics["voice_conversion_quality"] = self._calculate_voice_conversion_quality(audio_np, metadata)
        
        # Identifiability risk
        metrics["identifiability_risk"] = self._calculate_identifiability_risk(audio_np, metadata)
        
        # Biometric protection score
        metrics["biometric_protection"] = self._calculate_biometric_protection(audio_np, metadata)
        
        # Overall privacy score
        metrics["overall_privacy"] = self._calculate_overall_privacy(metrics)
        
        return metrics
    
    def _calculate_speaker_anonymity(self, audio: np.ndarray, metadata: Dict[str, Any]) -> float:
        """Calculate speaker anonymity score"""
        try:
            # Extract speaker features
            features = self.speaker_model.extract_features(audio)
            
            # If we have original speaker information, compare
            original_speaker = metadata.get("original_speaker")
            if original_speaker:
                # Simulate comparison with original speaker features
                # In practice, this would use stored speaker embeddings
                original_features = self._get_speaker_features(original_speaker)
                
                if original_features is not None:
                    # Calculate distance between features
                    distance = cosine(features, original_features)
                    # Higher distance means better anonymity
                    anonymity_score = min(distance, 1.0)
                else:
                    anonymity_score = 0.7  # Default when no reference
            else:
                # Estimate anonymity based on feature entropy
                feature_entropy = self._calculate_feature_entropy(features)
                anonymity_score = min(feature_entropy / 5.0, 1.0)  # Normalize
            
            return float(anonymity_score)
            
        except Exception as e:
            logger.warning(f"Speaker anonymity calculation failed: {e}")
            return 0.5
    
    def _calculate_privacy_leakage(self, audio: np.ndarray, metadata: Dict[str, Any]) -> float:
        """Calculate privacy leakage score"""
        try:
            # Look for repeating patterns that could identify speaker
            
            # 1. Autocorrelation analysis for repeating patterns
            autocorr = np.correlate(audio, audio, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find strong periodic patterns
            peaks, properties = signal.find_peaks(
                autocorr[1:], 
                height=0.3 * np.max(autocorr),
                distance=100
            )
            
            if len(peaks) > 0:
                pattern_strength = np.mean(autocorr[peaks + 1]) / np.max(autocorr)
                leakage_patterns = min(pattern_strength * 2, 1.0)
            else:
                leakage_patterns = 0.1
            
            # 2. Spectral uniqueness analysis
            freqs, psd = signal.welch(audio, fs=self.sample_rate, nperseg=1024)
            
            # Look for distinctive spectral peaks
            spectral_peaks, _ = signal.find_peaks(psd, height=np.mean(psd) * 3)
            if len(spectral_peaks) > 5:  # Many distinctive peaks
                spectral_uniqueness = 0.8
            elif len(spectral_peaks) > 2:
                spectral_uniqueness = 0.5
            else:
                spectral_uniqueness = 0.2
            
            # 3. Formant analysis (speech-specific)
            formant_leakage = self._analyze_formant_leakage(audio)
            
            # Combine leakage indicators
            overall_leakage = np.mean([
                leakage_patterns * 0.4,
                spectral_uniqueness * 0.3,
                formant_leakage * 0.3
            ])
            
            return float(overall_leakage)
            
        except Exception as e:
            logger.warning(f"Privacy leakage calculation failed: {e}")
            return 0.3
    
    def _calculate_voice_conversion_quality(self, audio: np.ndarray, metadata: Dict[str, Any]) -> float:
        """Calculate voice conversion quality score"""
        try:
            conversion_method = metadata.get("conversion_method")
            
            if not conversion_method:
                return 0.5  # No conversion applied
            
            # Analyze conversion artifacts
            
            # 1. Spectral consistency
            freqs, psd = signal.welch(audio, fs=self.sample_rate, nperseg=1024)
            
            # Look for unnatural spectral gaps or artifacts
            spectral_smoothness = self._calculate_spectral_smoothness(psd)
            
            # 2. Temporal consistency
            temporal_consistency = self._calculate_temporal_consistency(audio)
            
            # 3. Naturalness preservation
            naturalness = self._calculate_conversion_naturalness(audio)
            
            # Combine quality indicators
            quality_score = np.mean([
                spectral_smoothness * 0.4,
                temporal_consistency * 0.3,
                naturalness * 0.3
            ])
            
            return float(quality_score)
            
        except Exception as e:
            logger.warning(f"Voice conversion quality calculation failed: {e}")
            return 0.5
    
    def _calculate_identifiability_risk(self, audio: np.ndarray, metadata: Dict[str, Any]) -> float:
        """Calculate risk of speaker identification"""
        try:
            # Extract identifying features
            features = self.speaker_model.extract_features(audio)
            
            # 1. Feature uniqueness
            feature_variance = np.var(features)
            uniqueness_score = min(feature_variance / 0.1, 1.0)  # Normalize
            
            # 2. Prosodic patterns
            prosodic_risk = self._analyze_prosodic_patterns(audio)
            
            # 3. Speaking style consistency
            style_consistency = self._analyze_speaking_style(audio)
            
            # Combine risk factors
            identification_risk = np.mean([
                uniqueness_score * 0.4,
                prosodic_risk * 0.3,
                style_consistency * 0.3
            ])
            
            return float(identification_risk)
            
        except Exception as e:
            logger.warning(f"Identifiability risk calculation failed: {e}")
            return 0.5
    
    def _calculate_biometric_protection(self, audio: np.ndarray, metadata: Dict[str, Any]) -> float:
        """Calculate biometric protection score"""
        try:
            # Measure how well biometric information is protected
            
            # 1. Fundamental frequency variation
            f0_variation = self._calculate_f0_variation(audio)
            
            # 2. Formant shifting effectiveness
            formant_protection = self._calculate_formant_protection(audio)
            
            # 3. Temporal pattern disruption
            temporal_protection = self._calculate_temporal_protection(audio)
            
            # Combine protection measures
            protection_score = np.mean([
                f0_variation * 0.4,
                formant_protection * 0.35,
                temporal_protection * 0.25
            ])
            
            return float(protection_score)
            
        except Exception as e:
            logger.warning(f"Biometric protection calculation failed: {e}")
            return 0.5
    
    def _calculate_overall_privacy(self, metrics: Dict[str, float]) -> float:
        """Calculate overall privacy score"""
        weights = {
            "speaker_anonymity": 0.3,
            "privacy_leakage": -0.25,  # Negative because high leakage is bad
            "voice_conversion_quality": 0.2,
            "identifiability_risk": -0.15,  # Negative because high risk is bad
            "biometric_protection": 0.25
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                if weight < 0:
                    # Invert negative metrics
                    total_score += (1 - metrics[metric]) * abs(weight)
                else:
                    total_score += metrics[metric] * weight
                total_weight += abs(weight)
        
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.5
    
    def _get_speaker_features(self, speaker_id: str) -> Optional[np.ndarray]:
        """Get stored features for a speaker (placeholder)"""
        # In practice, this would retrieve stored speaker embeddings
        # For now, generate deterministic features based on speaker ID
        hash_obj = hashlib.md5(speaker_id.encode())
        seed = int(hash_obj.hexdigest()[:8], 16)
        np.random.seed(seed)
        return np.random.randn(128)  # 128-dimensional speaker embedding
    
    def _calculate_feature_entropy(self, features: np.ndarray) -> float:
        """Calculate entropy of speaker features"""
        # Quantize features and calculate entropy
        quantized = np.round(features * 10).astype(int)
        unique, counts = np.unique(quantized, return_counts=True)
        probabilities = counts / len(quantized)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def _analyze_formant_leakage(self, audio: np.ndarray) -> float:
        """Analyze formant information leakage"""
        try:
            # Simple formant analysis using LPC
            from scipy.signal import lfilter
            
            # Pre-emphasis
            emphasized = lfilter([1, -0.97], [1], audio)
            
            # Window and compute LPC
            window_size = 1024
            hop_size = 512
            formant_consistency = []
            
            for i in range(0, len(emphasized) - window_size, hop_size):
                frame = emphasized[i:i + window_size]
                
                # Simple formant estimation (placeholder)
                # In practice, use proper formant tracking
                fft = np.fft.fft(frame * np.hanning(len(frame)))
                magnitude = np.abs(fft[:len(fft)//2])
                
                # Find spectral peaks (simplified formants)
                peaks, _ = signal.find_peaks(magnitude, height=np.mean(magnitude))
                
                if len(peaks) >= 2:
                    formant_consistency.append(1.0)
                else:
                    formant_consistency.append(0.0)
            
            if formant_consistency:
                return np.mean(formant_consistency)
            else:
                return 0.3
                
        except Exception:
            return 0.3
    
    def _calculate_spectral_smoothness(self, psd: np.ndarray) -> float:
        """Calculate spectral smoothness (less artifacts = higher score)"""
        # Calculate second derivative to detect sharp transitions
        second_derivative = np.diff(psd, n=2)
        smoothness = 1 / (1 + np.mean(np.abs(second_derivative)))
        return min(smoothness, 1.0)
    
    def _calculate_temporal_consistency(self, audio: np.ndarray) -> float:
        """Calculate temporal consistency of conversion"""
        # Analyze energy envelope smoothness
        frame_size = 1024
        hop_size = 512
        energy_frames = []
        
        for i in range(0, len(audio) - frame_size, hop_size):
            frame_energy = np.mean(audio[i:i + frame_size] ** 2)
            energy_frames.append(frame_energy)
        
        if len(energy_frames) > 1:
            energy_variation = np.std(energy_frames) / (np.mean(energy_frames) + 1e-10)
            consistency = 1 / (1 + energy_variation)
            return min(consistency, 1.0)
        else:
            return 0.5
    
    def _calculate_conversion_naturalness(self, audio: np.ndarray) -> float:
        """Calculate naturalness preservation after conversion"""
        # Simple naturalness metric based on spectral characteristics
        freqs, psd = signal.welch(audio, fs=self.sample_rate, nperseg=1024)
        
        # Natural speech should have specific spectral characteristics
        low_freq_power = np.mean(psd[freqs < 1000])
        mid_freq_power = np.mean(psd[(freqs >= 1000) & (freqs < 4000)])
        high_freq_power = np.mean(psd[freqs >= 4000])
        
        # Natural ratio: decreasing power with frequency
        if high_freq_power > 0 and mid_freq_power > 0:
            naturalness = min(low_freq_power / mid_freq_power, 2.0) * 0.5
            naturalness += min(mid_freq_power / high_freq_power, 3.0) * 0.5
            return min(naturalness / 2.5, 1.0)
        else:
            return 0.5
    
    def _analyze_prosodic_patterns(self, audio: np.ndarray) -> float:
        """Analyze prosodic patterns for identification risk"""
        # Simplified prosodic analysis
        # In practice, this would analyze pitch contours, rhythm, etc.
        
        # Calculate energy envelope variation (rhythm proxy)
        frame_size = int(0.1 * self.sample_rate)  # 100ms frames
        hop_size = frame_size // 2
        
        energy_envelope = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame_energy = np.mean(audio[i:i + frame_size] ** 2)
            energy_envelope.append(frame_energy)
        
        if len(energy_envelope) > 1:
            # High variation