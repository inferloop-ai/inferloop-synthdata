# audio_synth/core/utils/metrics.py
"""
Audio analysis and metrics utilities
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy import signal
import logging

logger = logging.getLogger(__name__)

def calculate_snr(audio: torch.Tensor, noise_floor_db: float = -60.0) -> float:
    """
    Calculate Signal-to-Noise Ratio
    
    Args:
        audio: Audio tensor
        noise_floor_db: Estimated noise floor in dB
        
    Returns:
        SNR in dB
    """
    # Calculate signal power
    signal_power = torch.mean(audio ** 2)
    
    # Estimate noise power from quiet segments
    # Simple approach: use bottom 10th percentile of energy
    frame_size = 1024
    hop_size = 512
    frame_energies = []
    
    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i:i + frame_size]
        energy = torch.mean(frame ** 2)
        frame_energies.append(energy)
    
    if frame_energies:
        frame_energies = torch.tensor(frame_energies)
        noise_power = torch.quantile(frame_energies, 0.1)  # 10th percentile
    else:
        noise_power = signal_power * 0.01  # Fallback
    
    # Avoid division by zero
    noise_power = max(noise_power, 1e-10)
    
    snr_db = 10 * torch.log10(signal_power / noise_power)
    return float(snr_db)

def calculate_spectral_centroid(audio: torch.Tensor, sample_rate: int) -> float:
    """
    Calculate spectral centroid
    
    Args:
        audio: Audio tensor
        sample_rate: Sample rate
        
    Returns:
        Spectral centroid in Hz
    """
    # Convert to numpy for scipy
    audio_np = audio.detach().cpu().numpy()
    
    # Compute power spectral density
    freqs, psd = signal.welch(audio_np, fs=sample_rate, nperseg=1024)
    
    # Calculate spectral centroid
    centroid = np.sum(freqs * psd) / np.sum(psd)
    
    return float(centroid)

def calculate_spectral_rolloff(audio: torch.Tensor, sample_rate: int, rolloff_percent: float = 0.95) -> float:
    """
    Calculate spectral rolloff frequency
    
    Args:
        audio: Audio tensor
        sample_rate: Sample rate
        rolloff_percent: Percentage of energy for rolloff calculation
        
    Returns:
        Rolloff frequency in Hz
    """
    audio_np = audio.detach().cpu().numpy()
    
    # Compute power spectral density
    freqs, psd = signal.welch(audio_np, fs=sample_rate, nperseg=1024)
    
    # Calculate cumulative energy
    cumulative_energy = np.cumsum(psd)
    total_energy = cumulative_energy[-1]
    
    # Find rolloff frequency
    rolloff_energy = rolloff_percent * total_energy
    rolloff_idx = np.where(cumulative_energy >= rolloff_energy)[0][0]
    rolloff_freq = freqs[rolloff_idx]
    
    return float(rolloff_freq)

def calculate_zero_crossing_rate(audio: torch.Tensor) -> float:
    """
    Calculate zero crossing rate
    
    Args:
        audio: Audio tensor
        
    Returns:
        Zero crossing rate (crossings per sample)
    """
    audio_np = audio.detach().cpu().numpy()
    
    # Find zero crossings
    zero_crossings = np.where(np.diff(np.sign(audio_np)))[0]
    zcr = len(zero_crossings) / len(audio_np)
    
    return float(zcr)

def calculate_rms_energy(audio: torch.Tensor) -> float:
    """
    Calculate RMS energy
    
    Args:
        audio: Audio tensor
        
    Returns:
        RMS energy
    """
    rms = torch.sqrt(torch.mean(audio ** 2))
    return float(rms)

def calculate_dynamic_range(audio: torch.Tensor) -> float:
    """
    Calculate dynamic range in dB
    
    Args:
        audio: Audio tensor
        
    Returns:
        Dynamic range in dB
    """
    max_amplitude = torch.max(torch.abs(audio))
    
    # Find noise floor (bottom 5th percentile)
    abs_audio = torch.abs(audio)
    noise_floor = torch.quantile(abs_audio[abs_audio > 0], 0.05)
    
    if noise_floor > 0:
        dynamic_range = 20 * torch.log10(max_amplitude / noise_floor)
    else:
        dynamic_range = torch.tensor(60.0)  # Default high dynamic range
    
    return float(dynamic_range)

def calculate_spectral_flatness(audio: torch.Tensor, sample_rate: int) -> float:
    """
    Calculate spectral flatness (Wiener entropy)
    
    Args:
        audio: Audio tensor
        sample_rate: Sample rate
        
    Returns:
        Spectral flatness (0-1, higher = more noise-like)
    """
    audio_np = audio.detach().cpu().numpy()
    
    # Compute power spectral density
    freqs, psd = signal.welch(audio_np, fs=sample_rate, nperseg=1024)
    
    # Avoid log of zero
    psd = psd + 1e-10
    
    # Calculate geometric and arithmetic means
    geometric_mean = np.exp(np.mean(np.log(psd)))
    arithmetic_mean = np.mean(psd)
    
    # Spectral flatness
    flatness = geometric_mean / arithmetic_mean
    
    return float(flatness)

def analyze_fundamental_frequency(audio: torch.Tensor, sample_rate: int) -> Dict[str, float]:
    """
    Analyze fundamental frequency characteristics
    
    Args:
        audio: Audio tensor
        sample_rate: Sample rate
        
    Returns:
        Dictionary with F0 statistics
    """
    audio_np = audio.detach().cpu().numpy()
    
    # Simple F0 estimation using autocorrelation
    # This is a basic implementation - real systems would use more sophisticated methods
    
    # Frame-based analysis
    frame_size = int(0.025 * sample_rate)  # 25ms frames
    hop_size = int(0.010 * sample_rate)    # 10ms hop
    
    f0_estimates = []
    
    for i in range(0, len(audio_np) - frame_size, hop_size):
        frame = audio_np[i:i + frame_size]
        
        # Autocorrelation
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Find peaks
        peaks, _ = signal.find_peaks(autocorr[1:], height=0.3 * np.max(autocorr))
        
        if len(peaks) > 0:
            # Convert lag to frequency
            f0 = sample_rate / (peaks[0] + 1)
            if 50 <= f0 <= 800:  # Reasonable F0 range
                f0_estimates.append(f0)
    
    if f0_estimates:
        f0_estimates = np.array(f0_estimates)
        
        return {
            "f0_mean": float(np.mean(f0_estimates)),
            "f0_std": float(np.std(f0_estimates)),
            "f0_median": float(np.median(f0_estimates)),
            "f0_min": float(np.min(f0_estimates)),
            "f0_max": float(np.max(f0_estimates)),
            "f0_range": float(np.max(f0_estimates) - np.min(f0_estimates)),
            "voiced_frames": len(f0_estimates)
        }
    else:
        return {
            "f0_mean": 0.0,
            "f0_std": 0.0,
            "f0_median": 0.0,
            "f0_min": 0.0,
            "f0_max": 0.0,
            "f0_range": 0.0,
            "voiced_frames": 0
        }

def calculate_audio_fingerprint(audio: torch.Tensor, sample_rate: int) -> Dict[str, float]:
    """
    Calculate a comprehensive audio fingerprint
    
    Args:
        audio: Audio tensor
        sample_rate: Sample rate
        
    Returns:
        Dictionary with fingerprint features
    """
    fingerprint = {}
    
    # Basic features
    fingerprint["rms_energy"] = calculate_rms_energy(audio)
    fingerprint["zero_crossing_rate"] = calculate_zero_crossing_rate(audio)
    fingerprint["dynamic_range"] = calculate_dynamic_range(audio)
    
    # Spectral features
    fingerprint["spectral_centroid"] = calculate_spectral_centroid(audio, sample_rate)
    fingerprint["spectral_rolloff"] = calculate_spectral_rolloff(audio, sample_rate)
    fingerprint["spectral_flatness"] = calculate_spectral_flatness(audio, sample_rate)
    
    # F0 features
    f0_stats = analyze_fundamental_frequency(audio, sample_rate)
    fingerprint.update(f0_stats)
    
    # Additional features
    fingerprint["duration"] = len(audio) / sample_rate
    fingerprint["max_amplitude"] = float(torch.max(torch.abs(audio)))
    
    return fingerprint
