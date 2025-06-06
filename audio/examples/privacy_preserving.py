# examples/privacy_preserving.py
"""
Privacy-preserving audio generation example
"""

import torch
import torchaudio
from pathlib import Path
import json

from audio_synth.sdk.client import AudioSynthSDK

def privacy_preserving_example():
    """Demonstrate privacy-preserving audio generation"""
    
    print("=== Privacy-Preserving Audio Generation ===\n")
    
    sdk = AudioSynthSDK()
    
    # Test different privacy levels
    privacy_levels = ["low", "medium", "high"]
    speaker_id = "target_speaker_001"
    
    output_dir = Path("./output/privacy_preserving")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for privacy_level in privacy_levels:
        print(f"Generating audio with privacy level: {privacy_level}")
        
        result = sdk.generate_and_validate(
            method="diffusion",
            prompt="Hello, this is a test of privacy-preserving speech synthesis",
            num_samples=3,
            validators=["quality", "privacy", "fairness"],
            conditions={
                "speaker_id": speaker_id,
                "privacy_level": privacy_level
            }
        )
        
        # Save generated audio
        for i, audio in enumerate(result["audios"]):
            filename = f"privacy_{privacy_level}_sample_{i+1}.wav"
            torchaudio.save(
                str(output_dir / filename),
                audio.unsqueeze(0),
                22050
            )
        
        # Analyze privacy metrics
        privacy_metrics = result["validation"]["privacy"]
        avg_anonymity = sum(m["speaker_anonymity"] for m in privacy_metrics) / len(privacy_metrics)
        avg_leakage = sum(m["privacy_leakage"] for m in privacy_metrics) / len(privacy_metrics)
        
        results[privacy_level] = {
            "average_speaker_anonymity": avg_anonymity,
            "average_privacy_leakage": avg_leakage,
            "quality_scores": result["validation"]["quality"]
        }
        
        print(f"  Average speaker anonymity: {avg_anonymity:.3f}")
        print(f"  Average privacy leakage: {avg_leakage:.3f}")
    
    # Save privacy analysis
    with open(output_dir / "privacy_analysis.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nPrivacy analysis completed. Results saved to: {output_dir}")
    
    # Print summary
    print("\nPrivacy Level Comparison:")
    print("Level    | Anonymity | Leakage")
    print("---------|-----------|--------")
    for level, metrics in results.items():
        anon = metrics["average_speaker_anonymity"]
        leak = metrics["average_privacy_leakage"]
        print(f"{level:8} | {anon:9.3f} | {leak:7.3f}")

def voice_conversion_example():
    """Demonstrate voice conversion for privacy"""
    
    print("\n=== Voice Conversion Example ===\n")
    
    sdk = AudioSynthSDK()
    
    # Generate base audio
    base_result = sdk.generate(
        method="diffusion",
        prompt="This is the original voice that we want to convert",
        num_samples=1
    )
    
    base_audio = base_result[0]
    
    # Apply different voice conversions
    target_speakers = ["speaker_A", "speaker_B", "speaker_C"]
    
    output_dir = Path("./output/voice_conversion")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original
    torchaudio.save(
        str(output_dir / "original.wav"),
        base_audio.unsqueeze(0),
        22050
    )
    
    conversion_results = {}
    
    for target in target_speakers:
        print(f"Converting to {target}...")
        
        # Generate with target speaker conditioning
        converted_result = sdk.generate_and_validate(
            method="diffusion",
            prompt="This is the original voice that we want to convert",
            num_samples=1,
            validators=["privacy"],
            conditions={
                "speaker_id": target,
                "privacy_level": "high"
            }
        )
        
        converted_audio = converted_result["audios"][0]
        
        # Save converted audio
        filename = f"converted_to_{target}.wav"
        torchaudio.save(
            str(output_dir / filename),
            converted_audio.unsqueeze(0),
            22050
        )
        
        # Analyze conversion quality
        privacy_metrics = converted_result["validation"]["privacy"][0]
        conversion_results[target] = {
            "speaker_anonymity": privacy_metrics["speaker_anonymity"],
            "voice_conversion_quality": privacy_metrics["voice_conversion_quality"],
            "privacy_leakage": privacy_metrics["privacy_leakage"]
        }
        
        print(f"  Conversion quality: {privacy_metrics['voice_conversion_quality']:.3f}")
        print(f"  Speaker anonymity: {privacy_metrics['speaker_anonymity']:.3f}")
    
    # Save conversion analysis
    with open(output_dir / "conversion_analysis.json", 'w') as f:
        json.dump(conversion_results, f, indent=2)
    
    print(f"\nVoice conversion completed. Results saved to: {output_dir}")

if __name__ == "__main__":
    privacy_preserving_example()
    voice_conversion_example()
