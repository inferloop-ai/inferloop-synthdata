# Gaming Vertical

This directory contains configurations, models, and utilities specific to the gaming vertical for the inferloop-synthdata/video pipeline.

## Overview

The gaming vertical focuses on generating synthetic video data for game development, testing, and training AI models for gaming applications. This includes procedural content generation, asset management, and performance optimization scenarios.

## Subdirectories

- **asset-management**: Tools and configurations for managing game assets in synthetic videos
- **performance-optimization**: Benchmarks and tools for optimizing rendering performance
- **procedural-generation**: Algorithms and models for procedurally generating game environments and scenarios

## Usage

To use the gaming vertical configurations with the video synthesis pipeline:

```bash
# Example command to generate gaming-specific synthetic data
python -m inferloop-synthdata.video.cli generate --vertical gaming --scenario fps_gameplay
```

## Quality Metrics

Gaming-specific quality metrics focus on:
- Frame rate consistency
- Visual fidelity of game assets
- Physics simulation accuracy
- Animation smoothness
- Rendering performance
