# audio_synth/core/utils/io.py
"""
Audio I/O utilities
"""

import torch
import torchaudio
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any, List
import logging
import json
import yaml
from datetime import datetime

logger = logging.getLogger(__name__)

def load_audio(filepath: Union[str, Path], 
               target_sample_rate: Optional[int] = None,
               mono: bool = True,
               normalize: bool = True,
               trim_silence: bool = False) -> Tuple[torch.Tensor, int]:
    """
    Load audio file with optional preprocessing
    
    Args:
        filepath: Path to audio file
        target_sample_rate: Target sample rate (None to keep original)
        mono: Convert to mono if True
        normalize: Normalize audio to [-1, 1] range
        trim_silence: Remove leading/trailing silence
        
    Returns:
        Tuple of (audio_tensor, sample_rate)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")
    
    try:
        # Load audio using torchaudio
        audio, sample_rate = torchaudio.load(str(filepath))
        
        # Convert to mono if requested
        if mono and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Resample if target sample rate specified
        if target_sample_rate and target_sample_rate != sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            audio = resampler(audio)
            sample_rate = target_sample_rate
        
        # Trim silence if requested
        if trim_silence:
            audio = trim_silence_from_audio(audio, sample_rate)
        
        # Normalize if requested
        if normalize:
            audio = normalize_audio(audio)
        
        # Ensure single channel output for mono
        if mono:
            audio = audio.squeeze(0)  # Remove channel dimension
        
        logger.info(f"Loaded audio: {filepath}, shape: {audio.shape}, sr: {sample_rate}")
        return audio, sample_rate
        
    except Exception as e:
        logger.error(f"Failed to load audio {filepath}: {e}")
        raise

def save_audio(audio: torch.Tensor,
               filepath: Union[str, Path],
               sample_rate: int,
               format: Optional[str] = None,
               quality: Optional[str] = None,
               normalize: bool = True,
               metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save audio tensor to file
    
    Args:
        audio: Audio tensor to save
        filepath: Output file path
        sample_rate: Sample rate
        format: Audio format (inferred from extension if None)
        quality: Quality setting for lossy formats
        normalize: Normalize audio before saving
        metadata: Optional metadata to embed
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure audio is in correct format
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)  # Add channel dimension
    
    # Normalize if requested
    if normalize:
        audio = normalize_audio(audio)
    
    # Clamp to valid range
    audio = torch.clamp(audio, -1.0, 1.0)
    
    try:
        # Save using torchaudio
        torchaudio.save(str(filepath), audio, sample_rate)
        
        # Add metadata if provided
        if metadata:
            save_audio_metadata(filepath, metadata)
        
        logger.info(f"Saved audio: {filepath}, shape: {audio.shape}, sr: {sample_rate}")
        
    except Exception as e:
        logger.error(f"Failed to save audio {filepath}: {e}")
        raise

def load_audio_batch(filepaths: List[Union[str, Path]],
                    target_sample_rate: Optional[int] = None,
                    mono: bool = True,
                    normalize: bool = True,
                    pad_to_same_length: bool = False) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Load multiple audio files
    
    Args:
        filepaths: List of audio file paths
        target_sample_rate: Target sample rate
        mono: Convert to mono
        normalize: Normalize audio
        pad_to_same_length: Pad all audios to same length
        
    Returns:
        Tuple of (audio_list, sample_rates_list)
    """
    audios = []
    sample_rates = []
    
    for filepath in filepaths:
        try:
            audio, sr = load_audio(filepath, target_sample_rate, mono, normalize)
            audios.append(audio)
            sample_rates.append(sr)
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
            continue
    
    # Pad to same length if requested
    if pad_to_same_length and audios:
        max_length = max(len(audio) for audio in audios)
        audios = [pad_audio(audio, max_length) for audio in audios]
    
    return audios, sample_rates

def save_audio_batch(audios: List[torch.Tensor],
                    filepaths: List[Union[str, Path]],
                    sample_rate: int,
                    metadata_list: Optional[List[Dict[str, Any]]] = None) -> None:
    """
    Save multiple audio tensors
    
    Args:
        audios: List of audio tensors
        filepaths: List of output file paths
        sample_rate: Sample rate
        metadata_list: Optional list of metadata dictionaries
    """
    if len(audios) != len(filepaths):
        raise ValueError("Number of audios must match number of filepaths")
    
    if metadata_list and len(metadata_list) != len(audios):
        raise ValueError("Number of metadata entries must match number of audios")
    
    for i, (audio, filepath) in enumerate(zip(audios, filepaths)):
        metadata = metadata_list[i] if metadata_list else None
        save_audio(audio, filepath, sample_rate, metadata=metadata)

def normalize_audio(audio: torch.Tensor, method: str = "peak") -> torch.Tensor:
    """
    Normalize audio tensor
    
    Args:
        audio: Audio tensor to normalize
        method: Normalization method ("peak", "rms", "lufs")
        
    Returns:
        Normalized audio tensor
    """
    if method == "peak":
        # Peak normalization
        max_val = torch.max(torch.abs(audio))
        if max_val > 0:
            audio = audio / max_val
    
    elif method == "rms":
        # RMS normalization
        rms = torch.sqrt(torch.mean(audio ** 2))
        if rms > 0:
            audio = audio / rms * 0.3  # Target RMS of 0.3
    
    elif method == "lufs":
        # LUFS normalization (simplified)
        # In practice, would use proper LUFS measurement
        rms = torch.sqrt(torch.mean(audio ** 2))
        if rms > 0:
            audio = audio / rms * 0.25  # Approximate LUFS target
    
    return audio

def trim_silence_from_audio(audio: torch.Tensor, 
                           sample_rate: int,
                           threshold_db: float = -40.0,
                           frame_length: int = 2048,
                           hop_length: int = 512) -> torch.Tensor:
    """
    Trim silence from beginning and end of audio
    
    Args:
        audio: Audio tensor
        sample_rate: Sample rate
        threshold_db: Silence threshold in dB
        frame_length: Frame length for analysis
        hop_length: Hop length for analysis
        
    Returns:
        Trimmed audio tensor
    """
    # Convert to numpy for librosa
    audio_np = audio.squeeze().numpy()
    
    # Trim silence using librosa
    trimmed, _ = librosa.effects.trim(
        audio_np,
        top_db=-threshold_db,
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    # Convert back to tensor
    return torch.from_numpy(trimmed).float()

def pad_audio(audio: torch.Tensor, target_length: int, mode: str = "constant") -> torch.Tensor:
    """
    Pad audio to target length
    
    Args:
        audio: Audio tensor to pad
        target_length: Target length in samples
        mode: Padding mode ("constant", "reflect", "replicate")
        
    Returns:
        Padded audio tensor
    """
    current_length = len(audio)
    
    if current_length >= target_length:
        return audio[:target_length]
    
    padding_needed = target_length - current_length
    
    if mode == "constant":
        # Pad with zeros
        audio = torch.nn.functional.pad(audio, (0, padding_needed))
    elif mode == "reflect":
        # Reflect padding
        audio = torch.nn.functional.pad(audio, (0, padding_needed), mode="reflect")
    elif mode == "replicate":
        # Replicate last value
        audio = torch.nn.functional.pad(audio, (0, padding_needed), mode="replicate")
    
    return audio

def chunk_audio(audio: torch.Tensor, 
                chunk_length: int,
                hop_length: Optional[int] = None,
                pad_last: bool = True) -> List[torch.Tensor]:
    """
    Split audio into chunks
    
    Args:
        audio: Audio tensor to chunk
        chunk_length: Length of each chunk in samples
        hop_length: Hop length between chunks (defaults to chunk_length)
        pad_last: Pad last chunk if shorter than chunk_length
        
    Returns:
        List of audio chunks
    """
    if hop_length is None:
        hop_length = chunk_length
    
    chunks = []
    start = 0
    
    while start < len(audio):
        end = start + chunk_length
        chunk = audio[start:end]
        
        # Pad last chunk if necessary
        if len(chunk) < chunk_length and pad_last:
            chunk = pad_audio(chunk, chunk_length)
        
        if len(chunk) > 0:
            chunks.append(chunk)
        
        start += hop_length
    
    return chunks

def concatenate_audio(audios: List[torch.Tensor], 
                     crossfade_samples: int = 0) -> torch.Tensor:
    """
    Concatenate multiple audio tensors
    
    Args:
        audios: List of audio tensors to concatenate
        crossfade_samples: Number of samples to crossfade between segments
        
    Returns:
        Concatenated audio tensor
    """
    if not audios:
        return torch.empty(0)
    
    if len(audios) == 1:
        return audios[0]
    
    if crossfade_samples == 0:
        return torch.cat(audios, dim=0)
    
    # Concatenate with crossfading
    result = audios[0]
    
    for i in range(1, len(audios)):
        current_audio = audios[i]
        
        if len(result) >= crossfade_samples and len(current_audio) >= crossfade_samples:
            # Apply crossfade
            fade_out = torch.linspace(1, 0, crossfade_samples)
            fade_in = torch.linspace(0, 1, crossfade_samples)
            
            # Modify end of result
            result[-crossfade_samples:] *= fade_out
            
            # Modify beginning of current audio and add to result
            current_start = current_audio[:crossfade_samples] * fade_in
            result[-crossfade_samples:] += current_start
            
            # Append rest of current audio
            result = torch.cat([result, current_audio[crossfade_samples:]], dim=0)
        else:
            # No crossfade if segments too short
            result = torch.cat([result, current_audio], dim=0)
    
    return result

def convert_audio_format(input_path: Union[str, Path],
                        output_path: Union[str, Path],
                        target_format: str,
                        sample_rate: Optional[int] = None,
                        quality: Optional[str] = None) -> None:
    """
    Convert audio file to different format
    
    Args:
        input_path: Input audio file path
        output_path: Output audio file path
        target_format: Target format ("wav", "mp3", "flac", etc.)
        sample_rate: Target sample rate (None to keep original)
        quality: Quality setting for lossy formats
    """
    # Load audio
    audio, orig_sr = load_audio(input_path, target_sample_rate=sample_rate)
    
    # Save in target format
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(f".{target_format}")
    
    save_audio(audio, output_path, sample_rate or orig_sr)

def save_audio_metadata(filepath: Union[str, Path], metadata: Dict[str, Any]) -> None:
    """
    Save metadata alongside audio file
    
    Args:
        filepath: Audio file path
        metadata: Metadata dictionary
    """
    filepath = Path(filepath)
    metadata_path = filepath.with_suffix(filepath.suffix + ".meta.json")
    
    # Add timestamp
    metadata["saved_at"] = datetime.utcnow().isoformat()
    metadata["audio_file"] = filepath.name
    
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.debug(f"Saved metadata: {metadata_path}")
        
    except Exception as e:
        logger.warning(f"Failed to save metadata {metadata_path}: {e}")

def load_audio_metadata(filepath: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Load metadata for audio file
    
    Args:
        filepath: Audio file path
        
    Returns:
        Metadata dictionary or None if not found
    """
    filepath = Path(filepath)
    metadata_path = filepath.with_suffix(filepath.suffix + ".meta.json")
    
    if not metadata_path.exists():
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
        
    except Exception as e:
        logger.warning(f"Failed to load metadata {metadata_path}: {e}")
        return None

def analyze_audio_properties(audio: torch.Tensor, sample_rate: int) -> Dict[str, Any]:
    """
    Analyze basic properties of audio
    
    Args:
        audio: Audio tensor
        sample_rate: Sample rate
        
    Returns:
        Dictionary with audio properties
    """
    audio_np = audio.squeeze().numpy()
    
    properties = {
        "duration": len(audio) / sample_rate,
        "samples": len(audio),
        "sample_rate": sample_rate,
        "channels": 1 if len(audio.shape) == 1 else audio.shape[0],
        "max_amplitude": float(torch.max(torch.abs(audio))),
        "rms_energy": float(torch.sqrt(torch.mean(audio ** 2))),
        "dynamic_range_db": float(20 * torch.log10(torch.max(torch.abs(audio)) / (torch.std(audio) + 1e-8))),
        "zero_crossing_rate": len(np.where(np.diff(np.sign(audio_np)))[0]) / len(audio_np)
    }
    
    return properties

def validate_audio_file(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate audio file and return status
    
    Args:
        filepath: Audio file path
        
    Returns:
        Validation results dictionary
    """
    filepath = Path(filepath)
    
    result = {
        "valid": False,
        "error": None,
        "properties": None
    }
    
    try:
        if not filepath.exists():
            result["error"] = "File does not exist"
            return result
        
        # Try to load audio
        audio, sample_rate = load_audio(filepath)
        
        # Analyze properties
        properties = analyze_audio_properties(audio, sample_rate)
        
        # Basic validation
        if properties["duration"] <= 0:
            result["error"] = "Invalid duration"
            return result
        
        if properties["max_amplitude"] == 0:
            result["error"] = "Silent audio"
            return result
        
        result["valid"] = True
        result["properties"] = properties
        
    except Exception as e:
        result["error"] = str(e)
    
    return result