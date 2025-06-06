# Example usage commands

def print_usage_examples():
    """Print usage examples for the framework"""
    
    examples = """
# ============================================================================
# AUDIO SYNTHETIC DATA FRAMEWORK - USAGE EXAMPLES
# ============================================================================

## CLI Usage Examples

# 1. Generate basic audio samples
audio-synth generate \\
    --method diffusion \\
    --prompt "A person speaking clearly" \\
    --num-samples 10 \\
    --output-dir ./output

# 2. Generate with specific demographics
audio-synth generate \\
    --method tts \\
    --prompt "Hello, how are you today?" \\
    --gender female \\
    --age-group adult \\
    --accent british \\
    --privacy-level high

# 3. Validate existing audio files
audio-synth validate \\
    --input-dir ./audio_samples \\
    --validators quality privacy fairness \\
    --threshold-quality 0.8 \\
    --generate-report

# 4. Enhance privacy of existing audio
audio-synth enhance-privacy \\
    --input-file original.wav \\
    --output-file private.wav \\
    --privacy-level high \\
    --pitch-shift 2.0

## SDK Usage Examples

```python
from audio_synth.sdk.client import AudioSynthSDK

# Initialize SDK
sdk = AudioSynthSDK(config_path="./configs/default.yaml")

# Generate and validate in one call
result = sdk.generate_and_validate(
    method="diffusion",
    prompt="Professional business presentation",
    num_samples=5,
    validators=["quality", "privacy", "fairness"]
)

# Access generated audio
audios = result["audios"]
validation_results = result["validation"]

# Generate with specific conditions
result = sdk.generate(
    method="tts",
    prompt="Customer service greeting",
    conditions={
        "speaker_id": "professional_voice_001",
        "demographics": {
            "gender": "female",
            "age_group": "adult",
            "accent": "american"
        }
    }
)
```

## API Usage Examples

```bash
# Start the API server
uvicorn audio_synth.api.server:app --host 0.0.0.0 --port 8000

# Generate audio via API
curl -X POST "http://localhost:8000/api/v1/generate" \\
     -H "Content-Type: application/json" \\
     -d '{
       "method": "diffusion",
       "prompt": "A friendly greeting",
       "num_samples": 3,
       "privacy_level": "medium"
     }'

# Check job status
curl "http://localhost:8000/api/v1/jobs/{job_id}"

# Download generated file
curl "http://localhost:8000/api/v1/jobs/{job_id}/download/generated_001.wav" \\
     --output generated_audio.wav
```

## Docker Usage Examples

```bash
# Build and run with Docker
docker build -t audio-synth -f docker/Dockerfile .
docker run -p 8000:8000 -v $(pwd)/models:/app/models audio-synth

# Use Docker Compose for full stack
docker-compose -f docker/docker-compose.yml up -d

# Scale API instances
docker-compose up --scale audio-synth-api=3
```

## Configuration Examples

```yaml
# Custom configuration (configs/custom.yaml)
audio:
  sample_rate: 44100
  duration: 10.0
  format: "flac"

generation:
  default_method: "diffusion"
  privacy_level: "high"
  
validation:
  quality_threshold: 0.85
  privacy_threshold: 0.9
  fairness_threshold: 0.8
```

## Advanced Examples

```python
# Batch generation with different settings
import asyncio
from audio_synth.sdk.client import AudioSynthSDK

async def batch_generate():
    sdk = AudioSynthSDK()
    
    # Define generation tasks
    tasks = [
        {
            "method": "diffusion",
            "prompt": "News anchor speaking",
            "demographics": {"gender": "male", "accent": "american"}
        },
        {
            "method": "tts", 
            "prompt": "Customer service response",
            "demographics": {"gender": "female", "accent": "british"}
        },
        {
            "method": "vae",
            "prompt": "Casual conversation",
            "demographics": {"gender": "other", "accent": "australian"}
        }
    ]
    
    # Generate all tasks
    results = []
    for task in tasks:
        result = sdk.generate_and_validate(**task)
        results.append(result)
    
    return results

# Privacy-preserving pipeline
def privacy_pipeline():
    sdk = AudioSynthSDK()
    
    # Generate base audio
    base_audio = sdk.generate(
        method="diffusion",
        prompt="Confidential business call",
        num_samples=1
    )[0]
    
    # Apply privacy enhancements
    for privacy_level in ["low", "medium", "high"]:
        enhanced = sdk.enhance_privacy(
            audio=base_audio,
            privacy_level=privacy_level
        )
        
        # Validate privacy preservation
        privacy_metrics = sdk.validate(
            audios=[enhanced],
            validators=["privacy"]
        )
        
        print(f"Privacy level {privacy_level}: {privacy_metrics}")
```

## Monitoring and Logging

```bash
# View API logs
docker logs audio-synth-api

# Monitor metrics with Prometheus
curl http://localhost:9090/metrics

# View Grafana dashboard
open http://localhost:3000
```

## Testing and Validation

```bash
# Run unit tests
pytest tests/unit/

# Run integration tests  
pytest tests/integration/

# Run performance benchmarks
python scripts/benchmark.py --config configs/benchmark.yaml
```

# ============================================================================
"""
    
    print(examples)

if __name__ == "__main__":
    print_usage_examples()