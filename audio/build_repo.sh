#!/bin/bash

# Script to generate the directory hierarchy for audio-synthetic-data project

# Base directory name
BASE_DIR="."

# Create base directory
mkdir -p "$BASE_DIR"

# Create src directory structure
mkdir -p "$BASE_DIR/src/audio_synth/core/generators"
mkdir -p "$BASE_DIR/src/audio_synth/core/validators"
mkdir -p "$BASE_DIR/src/audio_synth/core/processors"
mkdir -p "$BASE_DIR/src/audio_synth/core/utils"
mkdir -p "$BASE_DIR/src/audio_synth/sdk"
mkdir -p "$BASE_DIR/src/audio_synth/cli"
mkdir -p "$BASE_DIR/src/audio_synth/api/routes"
mkdir -p "$BASE_DIR/src/audio_synth/api/models"

# Create examples directory structure
mkdir -p "$BASE_DIR/examples/notebooks"

# Create tests directory
mkdir -p "$BASE_DIR/tests"

# Create configs directory structure
mkdir -p "$BASE_DIR/configs/models"

# Create docs directory
mkdir -p "$BASE_DIR/docs"

# Create docker directory
mkdir -p "$BASE_DIR/docker"

# Create scripts directory
mkdir -p "$BASE_DIR/scripts"

# Create Python files in src
touch "$BASE_DIR/src/audio_synth/__init__.py"
touch "$BASE_DIR/src/audio_synth/core/__init__.py"

# Core generators
touch "$BASE_DIR/src/audio_synth/core/generators/__init__.py"
touch "$BASE_DIR/src/audio_synth/core/generators/base.py"
touch "$BASE_DIR/src/audio_synth/core/generators/diffusion.py"
touch "$BASE_DIR/src/audio_synth/core/generators/gan.py"
touch "$BASE_DIR/src/audio_synth/core/generators/vae.py"
touch "$BASE_DIR/src/audio_synth/core/generators/vocoder.py"
touch "$BASE_DIR/src/audio_synth/core/generators/tts.py"

# Core validators
touch "$BASE_DIR/src/audio_synth/core/validators/__init__.py"
touch "$BASE_DIR/src/audio_synth/core/validators/base.py"
touch "$BASE_DIR/src/audio_synth/core/validators/quality.py"
touch "$BASE_DIR/src/audio_synth/core/validators/privacy.py"
touch "$BASE_DIR/src/audio_synth/core/validators/fairness.py"
touch "$BASE_DIR/src/audio_synth/core/validators/perceptual.py"

# Core processors
touch "$BASE_DIR/src/audio_synth/core/processors/__init__.py"
touch "$BASE_DIR/src/audio_synth/core/processors/audio_processor.py"
touch "$BASE_DIR/src/audio_synth/core/processors/feature_extractor.py"
touch "$BASE_DIR/src/audio_synth/core/processors/augmentation.py"

# Core utils
touch "$BASE_DIR/src/audio_synth/core/utils/__init__.py"
touch "$BASE_DIR/src/audio_synth/core/utils/config.py"
touch "$BASE_DIR/src/audio_synth/core/utils/metrics.py"
touch "$BASE_DIR/src/audio_synth/core/utils/io.py"

# SDK files
touch "$BASE_DIR/src/audio_synth/sdk/__init__.py"
touch "$BASE_DIR/src/audio_synth/sdk/client.py"
touch "$BASE_DIR/src/audio_synth/sdk/pipeline.py"

# CLI files
touch "$BASE_DIR/src/audio_synth/cli/__init__.py"
touch "$BASE_DIR/src/audio_synth/cli/main.py"
touch "$BASE_DIR/src/audio_synth/cli/generate.py"
touch "$BASE_DIR/src/audio_synth/cli/validate.py"
touch "$BASE_DIR/src/audio_synth/cli/utils.py"

# API files
touch "$BASE_DIR/src/audio_synth/api/__init__.py"
touch "$BASE_DIR/src/audio_synth/api/server.py"
touch "$BASE_DIR/src/audio_synth/api/routes/__init__.py"
touch "$BASE_DIR/src/audio_synth/api/routes/generate.py"
touch "$BASE_DIR/src/audio_synth/api/routes/validate.py"
touch "$BASE_DIR/src/audio_synth/api/routes/health.py"
touch "$BASE_DIR/src/audio_synth/api/models/__init__.py"
touch "$BASE_DIR/src/audio_synth/api/models/requests.py"
touch "$BASE_DIR/src/audio_synth/api/models/responses.py"

# Examples files
touch "$BASE_DIR/examples/basic_generation.py"
touch "$BASE_DIR/examples/advanced_validation.py"
touch "$BASE_DIR/examples/speech_synthesis.py"
touch "$BASE_DIR/examples/privacy_preserving.py"
touch "$BASE_DIR/examples/notebooks/getting_started.ipynb"
touch "$BASE_DIR/examples/notebooks/quality_assessment.ipynb"
touch "$BASE_DIR/examples/notebooks/fairness_analysis.ipynb"

# Test files
touch "$BASE_DIR/tests/__init__.py"
touch "$BASE_DIR/tests/test_generators.py"
touch "$BASE_DIR/tests/test_validators.py"
touch "$BASE_DIR/tests/test_api.py"

# Config files
touch "$BASE_DIR/configs/default.yaml"
touch "$BASE_DIR/configs/production.yaml"
touch "$BASE_DIR/configs/models/diffusion_config.yaml"
touch "$BASE_DIR/configs/models/gan_config.yaml"

# Documentation files
touch "$BASE_DIR/docs/README.md"
touch "$BASE_DIR/docs/API.md"
touch "$BASE_DIR/docs/CLI.md"
touch "$BASE_DIR/docs/SDK.md"

# Docker files
touch "$BASE_DIR/docker/Dockerfile"
touch "$BASE_DIR/docker/docker-compose.yml"
touch "$BASE_DIR/docker/requirements.txt"

# Script files
touch "$BASE_DIR/scripts/setup.sh"
touch "$BASE_DIR/scripts/train_models.py"
touch "$BASE_DIR/scripts/benchmark.py"

# Root files
touch "$BASE_DIR/pyproject.toml"
touch "$BASE_DIR/requirements.txt"
touch "$BASE_DIR/README.md"

echo "Directory structure for $BASE_DIR has been created successfully."

# Optionally print the structure
echo "Created directory structure:"
if command -v tree >/dev/null 2>&1; then
    tree "$BASE_DIR"
else
    find "$BASE_DIR" -type d | sort
    echo "\nCreated files:"
    find "$BASE_DIR" -type f | sort
fi