# audio_synth/core/validators/fairness.py
"""
Fairness validation for synthetic audio across demographics
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from scipy import stats
from collections import defaultdict, Counter

from .base import BaseValidator

logger = logging.getLogger(__name__)

class FairnessValidator(BaseValidator):
    """Validator for fairness across demographic groups"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.fairness_threshold = config.get("fairness_threshold", 0.75)
        self.protected_attributes = config.get("protected_attributes", 
                                              ["gender", "age_group", "accent", "language"])
        self.sample_rate = config.get("sample_rate", 22050)
    
    def validate(self, audio: torch.Tensor, metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Validate fairness for a single audio sample
        
        Args:
            audio: Audio tensor [T]
            metadata: Metadata dictionary with demographics
            
        Returns:
            Dictionary with fairness metrics
        """
        audio_np = audio.detach().cpu().numpy()
        demographics = metadata.get("demographics", {})
        
        metrics = {}
        
        # Individual fairness metrics
        metrics["representation_quality"] = self._calculate_representation_quality(audio_np, demographics)
        metrics["bias_score"] = self._calculate_bias_score(audio_np, demographics)
        metrics["diversity_score"] = self._calculate_diversity_score(audio_np, demographics)
        
        # Demographic-specific quality
        metrics["demographic_quality"] = self._calculate_demographic_quality(audio_np, demographics)
        
        # Group membership prediction resistance
        metrics["group_prediction_resistance"] = self._calculate_group_prediction_resistance(audio_np, demographics)
        
        # Overall fairness score
        metrics["overall_fairness"] = self._calculate_overall_fairness(metrics)
        
        return metrics
    
    def validate_batch(self, 
                      audios: List[torch.Tensor], 
                      metadata: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """
        Validate fairness across a batch of samples
        Includes group-level fairness metrics
        """
        # Individual validations
        individual_results = []
        for audio, meta in zip(audios, metadata):
            try:
                result = self.validate(audio, meta)
                individual_results.append(result)
            except Exception as e:
                logger.error(f"Fairness validation failed for sample: {e}")
                individual_results.append(self._get_default_metrics())
        
        # Add group-level fairness metrics
        group_metrics = self._calculate_group_fairness(audios, metadata)
        
        # Add group metrics to each individual result
        for result in individual_results:
            result.update(group_metrics)
        
        return individual_results
    
    def _calculate_representation_quality(self, audio: np.ndarray, demographics: Dict[str, Any]) -> float:
        """Calculate how well the demographic group is represented"""
        try:
            # Extract audio features that might vary by demographics
            features = self._extract_demographic_features(audio)
            
            # Compare against expected characteristics for this demographic
            expected_features = self._get_expected_features(demographics)
            
            if expected_features is not None:
                # Calculate similarity to expected characteristics
                similarity = self._calculate_feature_similarity(features, expected_features)
                return float(similarity)
            else:
                # If no expectations, assume neutral representation
                return 0.7
                
        except Exception as e:
            logger.warning(f"Representation quality calculation failed: {e}")
            return 0.5
    
    def _calculate_bias_score(self, audio: np.ndarray, demographics: Dict[str, Any]) -> float:
        """Calculate bias score (lower is better, inverted for consistency)"""
        try:
            # Look for biased characteristics in the audio
            
            # 1. Pitch bias detection
            pitch_bias = self._detect_pitch_bias(audio, demographics)
            
            # 2. Quality bias detection
            quality_bias = self._detect_quality_bias(audio, demographics)
            
            # 3. Temporal bias detection
            temporal_bias = self._detect_temporal_bias(audio, demographics)
            
            # Combine bias indicators (lower bias = higher score)
            total_bias = np.mean([pitch_bias, quality_bias, temporal_bias])
            bias_score = 1.0 - total_bias  # Invert so higher is better
            
            return float(max(bias_score, 0.0))
            
        except Exception as e:
            logger.warning(f"Bias score calculation failed: {e}")
            return 0.5
    
    def _calculate_diversity_score(self, audio: np.ndarray, demographics: Dict[str, Any]) -> float:
        """Calculate diversity score for the sample"""
        try:
            # Measure acoustic diversity within the sample
            
            # 1. Spectral diversity
            spectral_diversity = self._calculate_spectral_diversity(audio)
            
            # 2. Temporal diversity
            temporal_diversity = self._calculate_temporal_diversity(audio)
            
            # 3. Prosodic diversity
            prosodic_diversity = self._calculate_prosodic_diversity(audio)
            
            # Combine diversity measures
            diversity_score = np.mean([
                spectral_diversity * 0.4,
                temporal_diversity * 0.3,
                prosodic_diversity * 0.3
            ])
            
            return float(diversity_score)
            
        except Exception as e:
            logger.warning(f"Diversity score calculation failed: {e}")
            return 0.5
    
    def _calculate_demographic_quality(self, audio: np.ndarray, demographics: Dict[str, Any]) -> float:
        """Calculate quality specific to demographic characteristics"""
        try:
            # Assess quality relative to demographic expectations
            
            gender = demographics.get("gender", "unknown")
            age_group = demographics.get("age_group", "unknown")
            
            # Gender-specific quality assessment
            gender_quality = self._assess_gender_quality(audio, gender)
            
            # Age-specific quality assessment
            age_quality = self._assess_age_quality(audio, age_group)
            
            # Accent/language quality
            accent = demographics.get("accent", "unknown")
            accent_quality = self._assess_accent_quality(audio, accent)
            
            # Combine demographic quality scores
            demographic_quality = np.mean([gender_quality, age_quality, accent_quality])
            
            return float(demographic_quality)
            
        except Exception as e:
            logger.warning(f"Demographic quality calculation failed: {e}")
            return 0.5
    
    def _calculate_group_prediction_resistance(self, audio: np.ndarray, demographics: Dict[str, Any]) -> float:
        """Calculate resistance to demographic group prediction"""
        try:
            # Measure how difficult it would be to predict demographics from audio
            
            # Extract features commonly used for demographic classification
            features = self._extract_classification_features(audio)
            
            # Calculate feature entropy (higher entropy = harder to classify)
            feature_entropy = self._calculate_feature_entropy(features)
            
            # Normalize entropy to 0-1 range
            max_entropy = np.log2(len(features))
            normalized_entropy = feature_entropy / max_entropy if max_entropy > 0 else 0.5
            
            return float(normalized_entropy)
            
        except Exception as e:
            logger.warning(f"Group prediction resistance calculation failed: {e}")
            return 0.5
    
    def _calculate_group_fairness(self, audios: List[torch.Tensor], metadata: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate group-level fairness metrics"""
        group_metrics = {}
        
        try:
            # Calculate demographic parity
            group_metrics["demographic_parity"] = self._calculate_demographic_parity(audios, metadata)
            
            # Calculate equal opportunity
            group_metrics["equal_opportunity"] = self._calculate_equal_opportunity(audios, metadata)
            
            # Calculate group quality variance
            group_metrics["group_quality_variance"] = self._calculate_group_quality_variance(audios, metadata)
            
            # Calculate representation balance
            group_metrics["representation_balance"] = self._calculate_representation_balance(metadata)
            
        except Exception as e:
            logger.warning(f"Group fairness calculation failed: {e}")
            group_metrics = {
                "demographic_parity": 0.5,
                "equal_opportunity": 0.5,
                "group_quality_variance": 0.5,
                "representation_balance": 0.5
            }
        
        return group_metrics
    
    def _calculate_demographic_parity(self, audios: List[torch.Tensor], metadata: List[Dict[str, Any]]) -> float:
        """Calculate demographic parity across groups"""
        try:
            # Group samples by protected attributes
            groups = defaultdict(list)
            
            for i, meta in enumerate(metadata):
                demographics = meta.get("demographics", {})
                for attr in self.protected_attributes:
                    if attr in demographics:
                        group_key = f"{attr}_{demographics[attr]}"
                        groups[group_key].append(i)
            
            if len(groups) < 2:
                return 1.0  # Perfect parity if only one group
            
            # Calculate quality scores for each group
            group_scores = {}
            for group, indices in groups.items():
                group_audios = [audios[i] for i in indices]
                avg_quality = self._calculate_average_quality(group_audios)
                group_scores[group] = avg_quality
            
            # Calculate parity as inverse of variance in group scores
            scores = list(group_scores.values())
            if len(scores) > 1:
                score_variance = np.var(scores)
                parity = 1.0 / (1.0 + score_variance * 10)  # Scale variance
                return float(parity)
            else:
                return 1.0
                
        except Exception as e:
            logger.warning(f"Demographic parity calculation failed: {e}")
            return 0.5
    
    def _calculate_equal_opportunity(self, audios: List[torch.Tensor], metadata: List[Dict[str, Any]]) -> float:
        """Calculate equal opportunity across groups"""
        try:
            # Define "positive outcome" as high-quality audio
            quality_threshold = 0.7
            
            # Group samples by protected attributes
            groups = defaultdict(list)
            
            for i, meta in enumerate(metadata):
                demographics = meta.get("demographics", {})
                for attr in self.protected_attributes:
                    if attr in demographics:
                        group_key = f"{attr}_{demographics[attr]}"
                        groups[group_key].append(i)
            
            # Calculate true positive rate for each group
            group_tpr = {}
            for group, indices in groups.items():
                group_audios = [audios[i] for i in indices]
                qualities = [self._calculate_single_quality(audio) for audio in group_audios]
                
                high_quality_count = sum(1 for q in qualities if q >= quality_threshold)
                total_count = len(qualities)
                
                if total_count > 0:
                    tpr = high_quality_count / total_count
                    group_tpr[group] = tpr
            
            # Calculate equal opportunity as inverse of TPR variance
            tpr_values = list(group_tpr.values())
            if len(tpr_values) > 1:
                tpr_variance = np.var(tpr_values)
                equal_opp = 1.0 / (1.0 + tpr_variance * 5)
                return float(equal_opp)
            else:
                return 1.0
                
        except Exception as e:
            logger.warning(f"Equal opportunity calculation failed: {e}")
            return 0.5
    
    def _extract_demographic_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract features relevant to demographics"""
        features = []
        
        # Fundamental frequency statistics
        f0_stats = self._extract_f0_statistics(audio)
        features.extend(f0_stats)
        
        # Formant characteristics
        formant_stats = self._extract_formant_statistics(audio)
        features.extend(formant_stats)
        
        # Spectral characteristics
        spectral_stats = self._extract_spectral_statistics(audio)
        features.extend(spectral_stats)
        
        # Temporal characteristics
        temporal_stats = self._extract_temporal_statistics(audio)
        features.extend(temporal_stats)
        
        return np.array(features)
    
    def _get_expected_features(self, demographics: Dict[str, Any]) -> Optional[np.ndarray]:
        """Get expected features for demographic group"""
        # This would typically be learned from data
        # For now, return None (no strong expectations)
        return None
    
    def _calculate_feature_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate similarity between feature sets"""
        from scipy.spatial.distance import cosine
        try:
            similarity = 1 - cosine(features1, features2)
            return max(similarity, 0.0)
        except:
            return 0.5
    
    def _detect_pitch_bias(self, audio: np.ndarray, demographics: Dict[str, Any]) -> float:
        """Detect pitch-based bias"""
        # Simple pitch bias detection
        # In practice, this would be more sophisticated
        f0_mean = self._estimate_fundamental_frequency(audio)
        
        gender = demographics.get("gender", "unknown")
        
        # Very basic bias detection
        if gender == "male" and f0_mean > 200:
            return 0.3  # Potential bias (male voice too high)
        elif gender == "female" and f0_mean < 150:
            return 0.3  # Potential bias (female voice too low)
        else:
            return 0.1  # Low bias
    
    def _detect_quality_bias(self, audio: np.ndarray, demographics: Dict[str, Any]) -> float:
        """Detect quality-based bias"""
        # Calculate basic quality indicators
        snr = self._estimate_snr(audio)
        
        # Check if quality varies systematically with demographics
        # This is a placeholder - real implementation would compare across groups
        if snr < 0.3:
            return 0.4  # Potential quality bias
        else:
            return 0.1
    
    def _detect_temporal_bias(self, audio: np.ndarray, demographics: Dict[str, Any]) -> float:
        """Detect temporal pattern bias"""
        # Check for biased temporal characteristics
        # Placeholder implementation
        return 0.1
    
    def _calculate_overall_fairness(self, metrics: Dict[str, float]) -> float:
        """Calculate overall fairness score"""
        weights = {
            "representation_quality": 0.25,
            "bias_score": 0.25,
            "diversity_score": 0.2,
            "demographic_quality": 0.2,
            "group_prediction_resistance": 0.1
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
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """Get default metrics when validation fails"""
        return {
            "representation_quality": 0.5,
            "bias_score": 0.5,
            "diversity_score": 0.5,
            "demographic_quality": 0.5,
            "group_prediction_resistance": 0.5,
            "overall_fairness": 0.5
        }
    
    # Helper methods for feature extraction and analysis
    def _extract_f0_statistics(self, audio: np.ndarray) -> List[float]:
        """Extract fundamental frequency statistics"""
        # Simplified F0 extraction
        f0_estimate = self._estimate_fundamental_frequency(audio)
        return [f0_estimate, f0_estimate * 0.1]  # Mean and std placeholder
    
    def _extract_formant_statistics(self, audio: np.ndarray) -> List[float]:
        """Extract formant statistics"""
        # Placeholder formant extraction
        return [0.5, 0.3, 0.2]  # F1, F2, F3 placeholders
    
    def _extract_spectral_statistics(self, audio: np.ndarray) -> List[float]:
        """Extract spectral statistics"""
        from scipy import signal
        freqs, psd = signal.welch(audio, fs=self.sample_rate, nperseg=1024)
        
        # Spectral centroid
        centroid = np.sum(freqs * psd) / np.sum(psd)
        normalized_centroid = centroid / (self.sample_rate / 2)
        
        # Spectral rolloff
        cumulative_energy = np.cumsum(psd)
        total_energy = cumulative_energy[-1]
        rolloff_idx = np.where(cumulative_energy >= 0.95 * total_energy)[0][0]
        rolloff = freqs[rolloff_idx] / (self.sample_rate / 2)
        
        return [normalized_centroid, rolloff]
    
    def _extract_temporal_statistics(self, audio: np.ndarray) -> List[float]:
        """Extract temporal statistics"""
        # RMS energy
        rms = np.sqrt(np.mean(audio ** 2))
        
        # Zero crossing rate
        zero_crossings = len(np.where(np.diff(np.sign(audio)))[0])
        zcr = zero_crossings / len(audio)
        
        return [rms, zcr]
    
    def _estimate_fundamental_frequency(self, audio: np.ndarray) -> float:
        """Estimate fundamental frequency"""
        # Simplified F0 estimation using autocorrelation
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Find first peak after lag 0
        from scipy import signal
        peaks, _ = signal.find_peaks(autocorr[1:], height=0.1 * np.max(autocorr))
        
        if len(peaks) > 0:
            f0 = self.sample_rate / (peaks[0] + 1)
            return min(f0, 500)  # Cap at 500 Hz
        else:
            return 150  # Default F0
    
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate signal-to-noise ratio"""
        signal_power = np.mean(audio ** 2)
        # Simple noise estimation from high frequencies
        from scipy import signal as sig
        b, a = sig.butter(4, 0.8, 'high')
        noise_estimate = sig.filtfilt(b, a, audio)
        noise_power = np.mean(noise_estimate ** 2)
        
        if noise_power > 0:
            snr = signal_power / noise_power
            return min(snr / 100, 1.0)  # Normalize
        else:
            return 1.0