# examples/basic_generation.py
"""
Basic audio generation example
"""

import torch
import torchaudio
from pathlib import Path
import yaml

from audio_synth.sdk.client import AudioSynthSDK
from audio_synth.core.utils.config import load_config

def basic_generation_example():
    """Demonstrate basic audio generation"""
    
    print("=== Basic Audio Generation Example ===\n")
    
    # Initialize SDK
    sdk = AudioSynthSDK()
    
    # Generate a single audio sample
    print("1. Generating single audio sample...")
    result = sdk.generate_and_validate(
        method="diffusion",
        prompt="A person speaking clearly in English",
        num_samples=1
    )
    
    audio = result["audios"][0]
    print(f"Generated audio shape: {audio.shape}")
    print(f"Duration: {len(audio) / 22050:.2f} seconds")
    
    # Save the audio
    output_dir = Path("./output/basic_generation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torchaudio.save(
        str(output_dir / "basic_sample.wav"),
        audio.unsqueeze(0),
        22050
    )
    print(f"Saved audio to: {output_dir / 'basic_sample.wav'}")
    
    # Print validation results
    print("\nValidation Results:")
    for validator, metrics_list in result["validation"].items():
        print(f"\n{validator.upper()}:")
        for metric, value in metrics_list[0].items():
            print(f"  {metric}: {value:.3f}")

def batch_generation_example():
    """Demonstrate batch audio generation"""
    
    print("\n=== Batch Audio Generation Example ===\n")
    
    sdk = AudioSynthSDK()
    
    # Generate multiple samples with different prompts
    prompts = [
        "A woman speaking softly",
        "A man with a deep voice",
        "A child reading a story",
        "An elderly person giving advice"
    ]
    
    print(f"2. Generating {len(prompts)} samples with different prompts...")
    
    all_audios = []
    all_metadata = []
    
    for i, prompt in enumerate(prompts):
        print(f"   Generating sample {i+1}: '{prompt}'")
        
        result = sdk.generate_and_validate(
            method="diffusion",
            prompt=prompt,
            num_samples=1,
            validators=["quality", "privacy"]
        )
        
        all_audios.extend(result["audios"])
        
        # Add metadata
        metadata = {
            "prompt": prompt,
            "sample_index": i,
            "validation": result["validation"]
        }
        all_metadata.append(metadata)
    
    # Save all samples
    output_dir = Path("./output/batch_generation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (audio, metadata) in enumerate(zip(all_audios, all_metadata)):
        filename = f"batch_sample_{i+1:02d}.wav"
        torchaudio.save(
            str(output_dir / filename),
            audio.unsqueeze(0),
            22050
        )
        print(f"   Saved: {filename}")
    
    # Save metadata
    import json
    with open(output_dir / "batch_metadata.json", 'w') as f:
        json.dump(all_metadata, f, indent=2, default=str)
    
    print(f"\nBatch generation completed. Files saved to: {output_dir}")

if __name__ == "__main__":
    basic_generation_example()
    batch_generation_example()
