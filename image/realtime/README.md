# Real-Time Image Ingestion

This directory contains modules for real-time image ingestion from various sources to feed into the synthetic image generation pipeline. These ingestors can be used for continuous profiling, adaptive generation, and real-time validation.

## Available Ingestors

### 1. Webcam (`ingest_webcam.py`)

Captures images from local webcams connected to the system.

```python
from realtime.ingest_webcam import WebcamIngester

# Initialize
ingester = WebcamIngester(device_id=0, resolution=(1280, 720))

# Capture a single frame
frame = ingester.capture_single_frame()

# Capture a batch of frames
frames = ingester.capture_batch(count=10, interval=0.5)

# Stream frames continuously
for batch in ingester.stream_frames(capture_interval=0.5, batch_size=5, batch_interval=10.0):
    # Process batch['images']
    pass
```

### 2. Drone Feeds (`ingest_drone_feed.py`)

Connects to drone video streams via RTSP, RTMP, HTTP, or HTTPS protocols.

```python
from realtime.ingest_drone_feed import DroneFeedIngester

# Initialize with RTSP stream
ingester = DroneFeedIngester(
    stream_url="rtsp://192.168.1.100:554/live",
    auth_token="optional_auth_token",
    resolution=(1280, 720)
)

# Start continuous ingestion
ingester.start_ingestion()

# Get a batch of frames
frames = ingester.get_batch(count=10, timeout=5.0)

# Process with callback
def process_frame(frame):
    # Do something with each frame
    pass

ingester.start_ingestion(on_frame=process_frame)

# Stop when done
ingester.stop_ingestion()
```

### 3. Edge IoT Cameras (`ingest_edge_camera.py`)

Connects to edge IoT cameras from various manufacturers (generic, Axis, Hikvision, Dahua, Amcrest).

```python
from realtime.ingest_edge_camera import EdgeCameraIngester

# Initialize
ingester = EdgeCameraIngester(
    camera_ip="192.168.1.100",
    port=80,
    username="admin",
    password="password",
    camera_type="hikvision"  # or "axis", "dahua", "amcrest", "generic"
)

# Capture a snapshot
frame = ingester.capture_snapshot()

# Start continuous ingestion
ingester.start_ingestion(interval=1.0)

# Get camera info and settings
info = ingester.get_camera_info()
settings = ingester.get_camera_settings()

# Stop when done
ingester.stop_ingestion()
```

### 4. Unsplash API (`ingest_unsplash.py`)

Fetches images from the Unsplash API for real-time profiling and generation.

```python
from realtime.ingest_unsplash import UnsplashIngester

# Initialize (requires API key)
ingester = UnsplashIngester(api_key="your_unsplash_api_key")

# Fetch random images
images = ingester.fetch_random_images(
    query="nature",
    count=10,
    orientation="landscape"
)

# Fetch from specific collections
collection_ids = ["123456", "789012"]
images = ingester.fetch_by_collections(
    collection_ids=collection_ids,
    per_collection=5
)

# Stream images continuously
for batch in ingester.stream_images(
    queries=["urban", "people", "technology"],
    batch_size=5,
    interval_seconds=60
):
    # Process batch
    pass
```

## Integration with Profiling

The real-time ingestors are designed to work seamlessly with the profiling components:

```python
from realtime.ingest_webcam import WebcamIngester
from realtime.profiler.image_profiler import ImageProfiler
from realtime.profiler.semantic_profiler import SemanticProfiler
from realtime.profiler.distribution_modeler import DistributionModeler
from realtime.profiler.generate_profile_json import ProfileGenerator

# Set up ingestion
webcam = WebcamIngester(device_id=0)

# Set up profilers
image_profiler = ImageProfiler()
semantic_profiler = SemanticProfiler()
dist_modeler = DistributionModeler()
profile_generator = ProfileGenerator()

# Process stream
for batch in webcam.stream_frames(batch_size=5, batch_interval=30.0):
    # Extract images
    images = batch['images']
    
    # Profile images
    image_features = image_profiler.process_batch(images)
    semantic_features = semantic_profiler.process_batch(images)
    
    # Model distributions
    distributions = dist_modeler.fit_distributions(image_features, semantic_features)
    
    # Generate profile
    profile = profile_generator.create_profile(
        image_features=image_features,
        semantic_features=semantic_features,
        distributions=distributions,
        source=batch['source']
    )
    
    # Use profile for generation or validation
    print(f"Generated profile with {len(profile['features'])} features")
```

## Command-Line Usage

Each ingestor includes a command-line interface for testing and standalone usage:

### Webcam

```bash
python -m realtime.ingest_webcam
```

### Drone Feed

```bash
python -m realtime.ingest_drone_feed --url rtsp://192.168.1.100:554/live --token optional_token --frames 5
```

### Edge Camera

```bash
python -m realtime.ingest_edge_camera --ip 192.168.1.100 --port 80 --username admin --password pass --type hikvision --frames 5
```

### Unsplash API

```bash
# Set UNSPLASH_API_KEY environment variable first
export UNSPLASH_API_KEY=your_api_key
python -m realtime.ingest_unsplash --query nature --count 10
```

## Common Patterns

All ingestors follow these common patterns:

1. **Initialization**: Create an instance with source-specific parameters
2. **Single Capture**: Get a single frame/image
3. **Batch Capture**: Get multiple frames/images
4. **Continuous Streaming**: Process frames/images as they arrive
5. **Resource Management**: Properly initialize and release resources
6. **Error Handling**: Retry logic and graceful degradation

## Extending with New Sources

To add a new image source:

1. Create a new file `ingest_your_source.py`
2. Implement a class following the common patterns above
3. Include methods for initialization, capture, and resource management
4. Add command-line interface for testing
5. Update this README with usage examples

## Performance Considerations

- Use appropriate buffer sizes based on memory constraints
- Consider frame rates and network bandwidth for remote sources
- For high-volume sources, use background threads for processing
- Release resources properly when done to avoid memory leaks

## Troubleshooting

### Webcam Issues
- Ensure the device ID is correct
- Check if another application is using the camera
- Try different resolution settings

### Drone Feed Issues
- Verify network connectivity to the drone
- Check that the stream URL format is correct
- Ensure proper authentication credentials

### Edge Camera Issues
- Verify the camera is powered on and connected to the network
- Check IP address, port, and authentication details
- Select the correct camera type for proper API endpoints

### Unsplash API Issues
- Verify your API key is valid and has sufficient quota
- Check network connectivity
- Handle rate limiting with appropriate delays