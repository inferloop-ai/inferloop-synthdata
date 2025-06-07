# Inferloop Synthetic Image Generation - Profiling Examples

## Overview

Profiling is a critical step in the synthetic image generation pipeline. It allows the system to analyze real-world images and create statistical models that guide the generation process, ensuring that synthetic images match the desired characteristics of the reference dataset.

This document provides practical examples of using the profiling tools in the Inferloop Synthetic Image Generation system.

## Basic Profiling Workflow

1. **Collect reference images** from a relevant source
2. **Create a profile** using the CLI or API
3. **Analyze the profile** to understand the characteristics
4. **Generate synthetic images** based on the profile
5. **Validate** the synthetic images against the profile

## CLI Examples

### Creating a Profile from Unsplash Images

```bash
# Create a profile from 100 Unsplash images with the query "nature"
python -m cli.synth_image_profile create --source unsplash --query nature --count 100 --output-name nature_profile
```

This command:
1. Fetches 100 images from Unsplash with the query "nature"
2. Analyzes their characteristics (color distribution, brightness, contrast, etc.)
3. Creates a profile named "nature_profile"
4. Saves it to `./profiles/stream_nature_profile.json`

### Continuous Profiling

```bash
# Run continuous profiling from webcam, updating every 5 minutes
python -m cli.synth_image_profile create --source webcam --count 30 --output-name office_lighting --continuous --interval 300
```

This is useful for capturing changes over time, such as lighting conditions throughout the day.

### Analyzing a Profile

```bash
# Analyze the nature profile with detailed statistics
python -m cli.synth_image_profile analyze --profile-name nature_profile --detailed
```

Example output:

```
📊 Profile Analysis: nature_profile
==================================================

🔍 GENERAL:
  • Sample Count: 100
  • Created: 2025-06-05 14:30:22
  • Last Updated: 2025-06-05 14:35:10

🔍 COLOR DISTRIBUTION:
  • Red Channel (mean): 0.42
  • Green Channel (mean): 0.56
  • Blue Channel (mean): 0.38
  • Dominant Colors: ["#2C5F1B", "#87A330", "#D6E2C9"]

🔍 BRIGHTNESS AND CONTRAST:
  • Brightness (mean): 0.48
  • Brightness (std): 0.12
  • Contrast (mean): 0.65
  • Contrast (std): 0.08

🔍 COMPOSITION:
  • Rule of Thirds Score (mean): 0.72
  • Symmetry Score (mean): 0.35
  • Complexity Score (mean): 0.68

🔍 RESOLUTION STATISTICS:
  • Width (mean): 3840
  • Height (mean): 2160
  • Aspect Ratio (most common): 16:9

🔍 INSIGHTS:
  • High green channel values suggest nature/outdoor scenes
  • Moderate brightness with good contrast
  • Strong rule of thirds composition
  • Primarily high-resolution images
```

### Exporting Analysis

```bash
# Export the analysis to a JSON file
python -m cli.synth_image_profile analyze --profile-name nature_profile --detailed --export ./analysis/nature_profile_analysis.json
```

## Python API Examples

### Creating a Profile Programmatically

```python
from realtime.profiler.generate_profile_json import ProfileGenerator
from pathlib import Path
import glob

# Initialize the profile generator
profile_generator = ProfileGenerator()

# Get a list of reference images
image_files = glob.glob("./reference_images/*.jpg")

# Process the images and create a profile
profile_path = profile_generator.process_and_save(image_files, "custom_profile")

print(f"Profile created at: {profile_path}")
```

### Using a Profile for Generation

```python
from generation.generate_diffusion import DiffusionImageGenerator
import json

# Load the profile
with open("./profiles/stream_nature_profile.json", "r") as f:
    profile = json.load(f)

# Initialize the generator
generator = DiffusionImageGenerator()

# Generate an image guided by the profile
image = generator.generate_from_profile(
    profile=profile,
    prompt="A serene forest scene with a small stream",
    guidance_scale=7.5
)

# Save the image
image.save("./data/generated/profile_guided_forest.png")
```

## Advanced Profiling Techniques

### Multi-Source Profiling

Combining profiles from different sources can create more diverse and robust generation models.

```python
from realtime.profiler.generate_profile_json import ProfileGenerator
import json

# Load existing profiles
with open("./profiles/stream_nature_profile.json", "r") as f:
    nature_profile = json.load(f)

with open("./profiles/stream_architecture_profile.json", "r") as f:
    architecture_profile = json.load(f)

# Create a new combined profile
profile_generator = ProfileGenerator()
combined_profile = profile_generator.merge_profiles(
    [nature_profile, architecture_profile],
    weights=[0.7, 0.3],  # 70% nature, 30% architecture
    output_name="nature_with_buildings"
)

# Save the combined profile
profile_path = profile_generator.save_profile(combined_profile, "nature_with_buildings")
print(f"Combined profile saved at: {profile_path}")
```

### Time-Series Profiling

For capturing changes over time and generating time-appropriate images:

```python
from realtime.profiler.generate_profile_json import ProfileGenerator
import time
import datetime

profile_generator = ProfileGenerator()

# Define time slots
time_slots = {
    "morning": (6, 11),   # 6 AM to 11 AM
    "midday": (11, 14),  # 11 AM to 2 PM
    "afternoon": (14, 18),  # 2 PM to 6 PM
    "evening": (18, 22),  # 6 PM to 10 PM
    "night": (22, 6)     # 10 PM to 6 AM
}

def get_current_time_slot():
    current_hour = datetime.datetime.now().hour
    for slot_name, (start, end) in time_slots.items():
        if start <= current_hour < end or (start > end and (current_hour >= start or current_hour < end)):
            return slot_name
    return "unknown"

# Run continuous profiling with time slot labeling
try:
    while True:
        time_slot = get_current_time_slot()
        
        # Capture webcam images
        from realtime.ingest_webcam import WebcamIngester
        ingester = WebcamIngester()
        images = ingester.capture_batch(count=10, interval=1.0)
        ingester.release()
        
        # Process images with time slot label
        profile_generator.process_image_batch(images, f"office_lighting_{time_slot}")
        
        # Generate and save time-specific profile
        profile = profile_generator.generate_profile(f"office_lighting_{time_slot}")
        profile_path = profile_generator.save_profile(profile, f"office_lighting_{time_slot}")
        
        print(f"Updated profile for {time_slot} at {datetime.datetime.now()}")
        
        # Wait for 30 minutes
        time.sleep(1800)
        
except KeyboardInterrupt:
    print("Time-series profiling stopped")
```

## Profile Analysis and Visualization

### Visualizing Profile Characteristics

```python
import json
import matplotlib.pyplot as plt
import numpy as np

# Load the profile
with open("./profiles/stream_nature_profile.json", "r") as f:
    profile = json.load(f)

# Extract color distribution
color_means = [
    profile["statistics"]["color_distribution"]["red_mean"],
    profile["statistics"]["color_distribution"]["green_mean"],
    profile["statistics"]["color_distribution"]["blue_mean"]
]

# Create a color distribution bar chart
plt.figure(figsize=(10, 6))
plt.bar(["Red", "Green", "Blue"], color_means, color=["red", "green", "blue"])
plt.title("Color Channel Distribution")
plt.ylabel("Mean Value")
plt.ylim(0, 1)
plt.savefig("./analysis/color_distribution.png")

# Create a sample color swatch based on the profile
dominant_colors = profile["statistics"]["dominant_colors"]
color_blocks = np.zeros((100, 100 * len(dominant_colors), 3))

for i, color_hex in enumerate(dominant_colors):
    # Convert hex to RGB
    r = int(color_hex[1:3], 16) / 255
    g = int(color_hex[3:5], 16) / 255
    b = int(color_hex[5:7], 16) / 255
    
    # Fill the block
    color_blocks[:, i*100:(i+1)*100, 0] = r
    color_blocks[:, i*100:(i+1)*100, 1] = g
    color_blocks[:, i*100:(i+1)*100, 2] = b

plt.figure(figsize=(10, 2))
plt.imshow(color_blocks)
plt.axis('off')
plt.title("Dominant Colors")
plt.savefig("./analysis/dominant_colors.png")
```

## Best Practices for Profiling

1. **Use a diverse set of reference images** to create robust profiles
2. **Create domain-specific profiles** for different generation tasks
3. **Regularly update profiles** to adapt to changing conditions
4. **Validate synthetic images** against the profile to ensure quality
5. **Combine multiple profiles** with appropriate weights for complex scenes
6. **Document profile characteristics** for future reference
7. **Use time-appropriate profiles** for realistic lighting and conditions
8. **Balance profile specificity and generality** based on your use case

## Common Profiling Issues and Solutions

| Issue | Possible Causes | Solution |
|-------|----------------|----------|
| Over-specific profile | Too few reference images | Increase sample size to at least 50-100 images |
| Washed out colors | Poor quality reference images | Filter out low-quality images before profiling |
| Inconsistent generation | Mixed content in reference set | Create separate profiles for different content types |
| Biased generation | Unbalanced reference dataset | Ensure diversity in reference images |
| Profile drift over time | Changing conditions | Use time-series profiling and update regularly |

## Advanced Profile Parameters

Beyond basic color and composition statistics, advanced profiles can include:

- **Semantic content distribution**: Objects, scenes, and their relationships
- **Style characteristics**: Artistic styles, filters, and effects
- **Temporal patterns**: Time-of-day, seasonal variations
- **Camera parameters**: Focal length, aperture, exposure
- **Depth information**: Foreground/background relationships

These advanced parameters can be extracted using specialized models and incorporated into the profile for more nuanced generation control.