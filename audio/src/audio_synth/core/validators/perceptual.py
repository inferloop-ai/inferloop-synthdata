# audio_synth/core/validators/perceptual.py
"""
Perceptual audio quality validation

Implements perceptual audio quality validation based on models of human auditory perception.
"""

import torch
import torchaudio
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from scipy import signal
import math

from .base import BaseValidator

logger = logging.getLogger(__name__)

class PerceptualValidator(BaseValidator):
    """Validator for perceptual audio quality assessment"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.perceptual_threshold = config.get("perceptual_threshold", 0.75)
        self.sample_rate = config.get("sample_rate", 22050)
        
        # Optional reference audio for comparison
        self.reference_audio = config.get("reference_audio", None)
        
    def validate(self, audio: torch.Tensor, metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Validate perceptual audio quality
        
        Args:
            audio: Audio tensor [T]
            metadata: Metadata dictionary
            
        Returns:
            Dictionary with perceptual quality metrics
        """
        audio_np = audio.detach().cpu().numpy()
        
        metrics = {}
        
        # Basic perceptual metrics
        metrics["clarity"] = self._calculate_clarity(audio_np)
        metrics["warmth"] = self._calculate_warmth(audio_np)
        metrics["brightness"] = self._calculate_brightness(audio_np)
        
        # Speech quality metrics if it's speech audio
        if metadata.get("content_type", "") == "speech":
            speech_metrics = self._calculate_speech_metrics(audio_np)
            metrics.update(speech_metrics)
        
        # Music quality metrics if it's music audio
        if metadata.get("content_type", "") == "music":
            music_metrics = self._calculate_music_metrics(audio_np)
            metrics.update(music_metrics)
        
        # Overall perceptual quality score
        metrics["overall_perceptual_quality"] = self._calculate_overall_quality(metrics)
        
        # If reference audio exists, calculate difference-based metrics
        if self.reference_audio is not None or metadata.get("reference_audio") is not None:
            ref_audio = metadata.get("reference_audio", self.reference_audio)
            if isinstance(ref_audio, torch.Tensor):
                ref_audio_np = ref_audio.detach().cpu().numpy()
                diff_metrics = self._calculate_reference_metrics(audio_np, ref_audio_np)
                metrics.update(diff_metrics)
        
        return metrics