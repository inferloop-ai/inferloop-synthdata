# Inferloop Synthetic Image Generation - API Reference

## Overview

The Inferloop Synthetic Image Generation API provides programmatic access to the image generation, validation, and delivery capabilities of the system. The API is built using FastAPI and follows RESTful principles.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

API requests require authentication using an API key. Include the API key in the request header:

```
X-API-Key: your_api_key_here
```

## Endpoints

### Image Generation

#### Generate a Single Image

```
POST /generate
```

**Request Body:**

```json
{
  "prompt": "A beautiful sunset over mountains",
  "negative_prompt": "blurry, low quality",
  "model_id": "runwayml/stable-diffusion-v1-5",
  "width": 512,
  "height": 512,
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "seed": 42
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| prompt | string | Yes | Text prompt for image generation |
| negative_prompt | string | No | Text describing what should not appear in the image |
| model_id | string | No | Hugging Face model ID (default: "runwayml/stable-diffusion-v1-5") |
| width | integer | No | Image width in pixels (default: 512) |
| height | integer | No | Image height in pixels (default: 512) |
| num_inference_steps | integer | No | Number of denoising steps (default: 50) |
| guidance_scale | number | No | Guidance scale for classifier-free guidance (default: 7.5) |
| seed | integer | No | Random seed for reproducible generation |

**Response:**

```json
{
  "image_path": "/data/generated/image_1234567890.png",
  "prompt": "A beautiful sunset over mountains",
  "parameters": {
    "negative_prompt": "blurry, low quality",
    "model_id": "runwayml/stable-diffusion-v1-5",
    "width": 512,
    "height": 512,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "seed": 42
  }
}
```

#### Generate Multiple Images (Batch)

```
POST /generate/batch
```

**Request Body:**

```json
{
  "prompts": [
    "A beautiful sunset over mountains",
    "A futuristic city with flying cars"
  ],
  "batch_size": 2,
  "model_id": "runwayml/stable-diffusion-v1-5",
  "width": 512,
  "height": 512,
  "num_inference_steps": 50,
  "guidance_scale": 7.5
}
```

**Response:**

```json
{
  "batch_id": "batch_1234567890",
  "images": [
    {
      "image_path": "/data/generated/image_1234567890_1.png",
      "prompt": "A beautiful sunset over mountains"
    },
    {
      "image_path": "/data/generated/image_1234567890_2.png",
      "prompt": "A futuristic city with flying cars"
    }
  ],
  "parameters": {
    "model_id": "runwayml/stable-diffusion-v1-5",
    "width": 512,
    "height": 512,
    "num_inference_steps": 50,
    "guidance_scale": 7.5
  }
}
```

### Image Validation

#### Validate Image Quality

```
POST /validate/quality
```

**Request Body:**

```json
{
  "image_path": "/data/generated/image_1234567890.png",
  "metrics": ["sharpness", "contrast", "noise"]
}
```

**Response:**

```json
{
  "image_path": "/data/generated/image_1234567890.png",
  "quality_score": 0.85,
  "metrics": {
    "sharpness": 78.5,
    "contrast": 0.72,
    "noise_level": 0.03
  },
  "passed": true
}
```

#### Validate Image Privacy

```
POST /validate/privacy
```

**Request Body:**

```json
{
  "image_path": "/data/generated/image_1234567890.png",
  "check_faces": true,
  "check_text": true
}
```

**Response:**

```json
{
  "image_path": "/data/generated/image_1234567890.png",
  "privacy_score": 95,
  "faces_detected": 0,
  "text_detected": false,
  "pii_detected": false,
  "issues": [],
  "passed": true
}
```

### Profiles

#### List Available Profiles

```
GET /profiles
```

**Response:**

```json
{
  "profiles": [
    {
      "name": "unsplash_nature",
      "created_at": "2025-05-15T14:30:00Z",
      "sample_count": 250,
      "path": "/data/profiles/stream_unsplash_nature.json"
    },
    {
      "name": "webcam_office",
      "created_at": "2025-05-20T09:15:00Z",
      "sample_count": 120,
      "path": "/data/profiles/stream_webcam_office.json"
    }
  ]
}
```

#### Get Profile Details

```
GET /profiles/{profile_name}
```

**Response:**

```json
{
  "name": "unsplash_nature",
  "created_at": "2025-05-15T14:30:00Z",
  "updated_at": "2025-05-16T10:45:00Z",
  "sample_count": 250,
  "path": "/data/profiles/stream_unsplash_nature.json",
  "statistics": {
    "color_distribution": {
      "red_mean": 0.42,
      "green_mean": 0.56,
      "blue_mean": 0.38
    },
    "brightness_mean": 0.48,
    "contrast_mean": 0.65,
    "resolution": {
      "width_mean": 1920,
      "height_mean": 1080
    }
  }
}
```

### Export

#### Export to JSONL

```
POST /export/jsonl
```

**Request Body:**

```json
{
  "dataset_dir": "/data/generated/nature_set",
  "output_file": "/data/exports/nature_set.jsonl",
  "include_metadata": true,
  "include_annotations": true
}
```

**Response:**

```json
{
  "status": "success",
  "exported_count": 120,
  "output_file": "/data/exports/nature_set.jsonl",
  "file_size_bytes": 2450000
}
```

#### Export to S3

```
POST /export/s3
```

**Request Body:**

```json
{
  "dataset_dir": "/data/generated/nature_set",
  "bucket_name": "my-synthetic-data",
  "s3_prefix": "datasets/nature/v1",
  "include_metadata": true,
  "public_read": false
}
```

**Response:**

```json
{
  "status": "success",
  "uploaded_count": 120,
  "failed_count": 0,
  "total_size_bytes": 24500000,
  "bucket_name": "my-synthetic-data",
  "s3_prefix": "datasets/nature/v1",
  "manifest_url": "s3://my-synthetic-data/datasets/nature/v1/manifest.json"
}
```

## Error Handling

All API endpoints return standard HTTP status codes:

- 200: Success
- 400: Bad Request (invalid parameters)
- 401: Unauthorized (invalid API key)
- 404: Not Found (resource not found)
- 500: Internal Server Error

Error responses follow this format:

```json
{
  "error": {
    "code": "INVALID_PARAMETERS",
    "message": "Invalid parameter: width must be between 64 and 1024",
    "details": {
      "parameter": "width",
      "value": 2048,
      "constraint": "64 <= width <= 1024"
    }
  }
}
```

## Rate Limiting

API requests are rate-limited to prevent abuse. The current limits are:

- 10 requests per minute for image generation
- 20 requests per minute for validation
- 5 requests per minute for export operations

Rate limit information is included in the response headers:

```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 5
X-RateLimit-Reset: 1622568000
```

## SDK

A Python SDK is available for easier integration:

```python
from inferloop_synthdata import SyntheticImageClient

# Initialize client
client = SyntheticImageClient(api_key="your_api_key_here")

# Generate an image
response = client.generate_image(
    prompt="A beautiful sunset over mountains",
    width=512,
    height=512
)

print(f"Image generated: {response['image_path']}")
```

## Websocket API

For real-time generation and monitoring, a WebSocket API is available:

```
ws://localhost:8000/api/v1/ws/generation
```

Connect to this endpoint with your API key in the query parameters:

```
ws://localhost:8000/api/v1/ws/generation?api_key=your_api_key_here
```

Send generation requests as JSON messages:

```json
{
  "type": "generate",
  "prompt": "A beautiful sunset over mountains",
  "width": 512,
  "height": 512
}
```

Receive progress updates and results:

```json
{
  "type": "progress",
  "job_id": "job_1234567890",
  "step": 25,
  "total_steps": 50,
  "progress": 0.5
}
```

```json
{
  "type": "result",
  "job_id": "job_1234567890",
  "image_path": "/data/generated/image_1234567890.png",
  "prompt": "A beautiful sunset over mountains"
}
```