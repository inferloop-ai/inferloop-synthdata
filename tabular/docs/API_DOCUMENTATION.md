# Tabular REST API Documentation

## Overview

The Tabular REST API provides programmatic access to synthetic tabular data generation capabilities through HTTP endpoints. This documentation covers authentication, endpoints, request/response formats, and best practices.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Authentication](#authentication)
3. [Base URL](#base-url)
4. [Endpoints](#endpoints)
5. [Request & Response Format](#request--response-format)
6. [Error Handling](#error-handling)
7. [Rate Limiting](#rate-limiting)
8. [Webhooks](#webhooks)
9. [Examples](#examples)

## Getting Started

### Quick Start

```bash
# Start the API server
inferloop-tabular serve --port 8000

# Test the API
curl http://localhost:8000/health

# Generate synthetic data
curl -X POST http://localhost:8000/api/tabular/generate \
  -H "Content-Type: multipart/form-data" \
  -H "X-API-Key: your-api-key" \
  -F "file=@data.csv" \
  -F "algorithm=ctgan" \
  -F "num_rows=1000"
```

### API Client Libraries

```python
# Python client
from inferloop import TabularClient

client = TabularClient(api_key="your-api-key")
result = client.generate("data.csv", algorithm="ctgan", num_rows=1000)
```

```javascript
// JavaScript client
const Tabular = require('@inferloop/tabular');

const client = new Tabular({ apiKey: 'your-api-key' });
const result = await client.generate('data.csv', {
  algorithm: 'ctgan',
  numRows: 1000
});
```

## Authentication

### API Key Authentication

Include your API key in the `X-API-Key` header:

```http
X-API-Key: your-api-key
```

### JWT Authentication

For advanced use cases, use JWT tokens:

```http
Authorization: Bearer your-jwt-token
```

### Getting API Keys

```bash
# Via CLI
inferloop-tabular config set api.key "your-api-key"

# Via environment variable
export INFERLOOP_API_KEY="your-api-key"
```

## Base URL

```
Production: https://api.inferloop.com
Development: http://localhost:8000
```

All endpoints are prefixed with `/api/tabular/v1`

## Endpoints

### Health Check

#### `GET /health`

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Ready Check

#### `GET /ready`

Check if the service is ready to handle requests.

**Response:**
```json
{
  "status": "ready",
  "models_loaded": ["sdv", "ctgan", "ydata"],
  "cache_connected": true,
  "database_connected": true
}
```

### Data Generation

#### `POST /api/tabular/generate`

Generate synthetic tabular data from uploaded file.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`

**Form Data Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | file | Yes | - | Input data file (CSV, Parquet, JSON) |
| `algorithm` | string | No | `sdv` | Algorithm to use |
| `num_rows` | integer | No | Same as input | Number of rows to generate |
| `format` | string | No | `csv` | Output format |
| `seed` | integer | No | None | Random seed |
| `constraints` | file | No | None | Constraints JSON file |
| `config` | file | No | None | Configuration file |

**Algorithm-specific Parameters:**

SDV:
- `sdv_model`: Model type (`gaussian_copula`, `ctgan`, `copulagan`)
- `default_distribution`: Distribution for numerical columns
- `enforce_min_max`: Enforce value bounds

CTGAN:
- `epochs`: Training epochs (default: 300)
- `batch_size`: Batch size (default: 500)
- `embedding_dim`: Embedding dimension (default: 128)
- `discriminator_steps`: Discriminator update steps

YData:
- `noise_dim`: Noise dimension (default: 128)
- `layers_dim`: Layer dimensions (default: 128)
- `privacy_level`: Privacy setting (`low`, `medium`, `high`)

**Response:**
```json
{
  "id": "gen-123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "algorithm": "ctgan",
  "created": 1642329600,
  "data_url": "https://api.inferloop.com/api/tabular/download/gen-123e4567",
  "metadata": {
    "original_rows": 1000,
    "generated_rows": 5000,
    "columns": 15,
    "generation_time": 45.2,
    "quality_score": 0.92
  },
  "download_formats": {
    "csv": "/api/tabular/download/gen-123e4567?format=csv",
    "parquet": "/api/tabular/download/gen-123e4567?format=parquet",
    "json": "/api/tabular/download/gen-123e4567?format=json"
  }
}
```

### Asynchronous Generation

#### `POST /api/tabular/generate/async`

Start asynchronous generation for large datasets.

**Request Body:**
```json
{
  "data_url": "s3://bucket/data.csv",
  "algorithm": "ctgan",
  "num_rows": 1000000,
  "callback_url": "https://your-app.com/webhook",
  "parameters": {
    "epochs": 500,
    "batch_size": 1000
  }
}
```

**Response:**
```json
{
  "job_id": "job-123e4567-e89b-12d3-a456-426614174000",
  "status": "pending",
  "estimated_time": 3600,
  "status_url": "/api/tabular/jobs/job-123e4567"
}
```

#### `GET /api/tabular/jobs/{job_id}`

Check job status.

**Response:**
```json
{
  "job_id": "job-123e4567-e89b-12d3-a456-426614174000",
  "status": "running",
  "progress": 45,
  "started_at": "2024-01-15T10:00:00Z",
  "estimated_completion": "2024-01-15T11:00:00Z",
  "logs": [
    {"timestamp": "2024-01-15T10:00:00Z", "message": "Job started"},
    {"timestamp": "2024-01-15T10:15:00Z", "message": "Data loaded, starting training"}
  ]
}
```

### Data Profiling

#### `POST /api/tabular/profile`

Profile uploaded data to understand its characteristics.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`

**Form Data:**
- `file`: Data file to profile
- `sample_size`: Optional sample size for large files

**Response:**
```json
{
  "profile_id": "prof-123e4567",
  "summary": {
    "rows": 10000,
    "columns": 15,
    "missing_values": 234,
    "duplicate_rows": 12,
    "memory_usage": "2.5 MB"
  },
  "columns": {
    "age": {
      "type": "numeric",
      "dtype": "int64",
      "missing": 0,
      "unique": 73,
      "min": 18,
      "max": 95,
      "mean": 42.3,
      "std": 15.7,
      "distribution": "normal"
    },
    "category": {
      "type": "categorical",
      "dtype": "object",
      "missing": 10,
      "unique": 5,
      "top": "A",
      "freq": 3421,
      "categories": ["A", "B", "C", "D", "E"]
    }
  },
  "correlations": {
    "pearson": {...},
    "spearman": {...}
  },
  "report_url": "/api/tabular/profile/prof-123e4567/report"
}
```

### Data Validation

#### `POST /api/tabular/validate`

Validate synthetic data quality against original data.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`

**Form Data:**
- `original`: Original data file
- `synthetic`: Synthetic data file
- `metrics`: Comma-separated list of metrics

**Response:**
```json
{
  "validation_id": "val-123e4567",
  "overall_score": 0.89,
  "metrics": {
    "statistical_similarity": {
      "score": 0.92,
      "details": {
        "ks_test": {"passed": 12, "failed": 3},
        "chi_squared": {"passed": 5, "failed": 0},
        "correlation_diff": 0.05
      }
    },
    "ml_efficacy": {
      "score": 0.88,
      "classifier_auc": 0.52,
      "feature_importance_correlation": 0.91
    },
    "privacy": {
      "score": 0.95,
      "distance_to_closest_record": 0.42,
      "membership_inference_risk": 0.08,
      "k_anonymity": 5
    }
  },
  "report_url": "/api/tabular/validation/val-123e4567/report"
}
```

### Model Management

#### `POST /api/tabular/models`

Train and save a model for reuse.

**Request Body:**
```json
{
  "name": "customer_model",
  "algorithm": "ctgan",
  "training_data_url": "s3://bucket/customers.csv",
  "parameters": {
    "epochs": 500,
    "batch_size": 500
  },
  "description": "Customer data generator model"
}
```

**Response:**
```json
{
  "model_id": "model-123e4567",
  "name": "customer_model",
  "status": "training",
  "created_at": "2024-01-15T10:00:00Z",
  "training_job_id": "job-789"
}
```

#### `GET /api/tabular/models`

List available models.

**Query Parameters:**
- `page`: Page number (default: 1)
- `per_page`: Items per page (default: 20)
- `algorithm`: Filter by algorithm
- `sort`: Sort field (name, created_at, quality_score)

**Response:**
```json
{
  "models": [
    {
      "model_id": "model-123",
      "name": "customer_model",
      "algorithm": "ctgan",
      "created_at": "2024-01-15T10:00:00Z",
      "last_used": "2024-01-20T15:30:00Z",
      "quality_score": 0.92,
      "size_mb": 125,
      "description": "Customer data generator"
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 45,
    "pages": 3
  }
}
```

#### `POST /api/tabular/models/{model_id}/generate`

Generate data using a saved model.

**Request Body:**
```json
{
  "num_rows": 1000,
  "conditions": {
    "category": "premium",
    "age": ">30"
  },
  "format": "csv"
}
```

**Response:**
```json
{
  "generation_id": "gen-456",
  "model_id": "model-123",
  "status": "completed",
  "rows_generated": 1000,
  "download_url": "/api/tabular/download/gen-456"
}
```

### Batch Processing

#### `POST /api/tabular/batch`

Process multiple files in batch.

**Request Body:**
```json
{
  "files": [
    {"url": "s3://bucket/file1.csv", "name": "dataset1"},
    {"url": "s3://bucket/file2.csv", "name": "dataset2"}
  ],
  "algorithm": "sdv",
  "common_parameters": {
    "num_rows": 5000,
    "seed": 42
  },
  "output_format": "parquet"
}
```

**Response:**
```json
{
  "batch_id": "batch-123",
  "status": "processing",
  "total_files": 2,
  "completed": 0,
  "failed": 0,
  "status_url": "/api/tabular/batch/batch-123"
}
```

### Streaming Generation

#### `GET /api/tabular/stream/{model_id}`

Stream synthetic data in real-time.

**Query Parameters:**
- `rate`: Records per second (default: 100)
- `format`: Output format (json, csv)
- `duration`: Stream duration in seconds

**Response** (Server-Sent Events):
```
event: record
data: {"id": 1, "name": "John Doe", "age": 35, "category": "A"}

event: record
data: {"id": 2, "name": "Jane Smith", "age": 28, "category": "B"}

event: stats
data: {"total_generated": 1000, "rate": 98.5, "elapsed": 10.15}

event: complete
data: {"total": 10000, "duration": 100.0}
```

### Privacy Analysis

#### `POST /api/tabular/privacy/analyze`

Analyze privacy risks in data.

**Request Body:**
```json
{
  "data_url": "s3://bucket/sensitive.csv",
  "quasi_identifiers": ["age", "zipcode", "gender"],
  "sensitive_attributes": ["income", "diagnosis"],
  "analysis_types": ["k_anonymity", "l_diversity", "t_closeness"]
}
```

**Response:**
```json
{
  "analysis_id": "priv-123",
  "risks": {
    "re_identification_risk": 0.15,
    "attribute_disclosure_risk": 0.08,
    "membership_inference_risk": 0.05
  },
  "metrics": {
    "k_anonymity": {
      "k": 3,
      "violating_groups": 12,
      "percentage_protected": 94.5
    },
    "l_diversity": {
      "l": 2,
      "satisfied": false,
      "issues": ["diagnosis column lacks diversity"]
    }
  },
  "recommendations": [
    "Increase k-anonymity to at least 5",
    "Apply generalization to age column",
    "Consider suppressing rare zipcodes"
  ]
}
```

### Constraints Management

#### `POST /api/tabular/constraints`

Create constraint set for generation.

**Request Body:**
```json
{
  "name": "business_rules",
  "constraints": [
    {
      "type": "range",
      "column": "age",
      "min": 0,
      "max": 120
    },
    {
      "type": "relationship",
      "expression": "end_date >= start_date"
    },
    {
      "type": "unique",
      "columns": ["customer_id"]
    },
    {
      "type": "sum",
      "columns": ["q1", "q2", "q3", "q4"],
      "total": "annual_total"
    }
  ]
}
```

**Response:**
```json
{
  "constraint_id": "const-123",
  "name": "business_rules",
  "created_at": "2024-01-15T10:00:00Z",
  "num_constraints": 4,
  "validation_status": "valid"
}
```

### Multi-table Generation

#### `POST /api/tabular/multitable/generate`

Generate related tables maintaining relationships.

**Request Body:**
```json
{
  "tables": {
    "customers": {"url": "s3://bucket/customers.csv"},
    "orders": {"url": "s3://bucket/orders.csv"},
    "products": {"url": "s3://bucket/products.csv"}
  },
  "relationships": [
    {
      "parent": "customers",
      "child": "orders",
      "parent_key": "customer_id",
      "child_key": "customer_id",
      "type": "one_to_many"
    }
  ],
  "algorithm": "sdv",
  "scale_factor": 2.0
}
```

**Response:**
```json
{
  "generation_id": "multi-123",
  "status": "completed",
  "tables_generated": {
    "customers": {
      "rows": 2000,
      "download_url": "/api/tabular/download/multi-123/customers"
    },
    "orders": {
      "rows": 8500,
      "download_url": "/api/tabular/download/multi-123/orders"
    },
    "products": {
      "rows": 500,
      "download_url": "/api/tabular/download/multi-123/products"
    }
  },
  "relationship_integrity": {
    "score": 1.0,
    "valid": true
  }
}
```

### Usage Statistics

#### `GET /api/tabular/usage`

Get usage statistics for the authenticated user.

**Query Parameters:**
- `period`: Time period (`24h`, `7d`, `30d`)
- `timezone`: Timezone for date grouping

**Response:**
```json
{
  "period": "7d",
  "usage": {
    "total_generations": 156,
    "total_rows_generated": 1250000,
    "total_compute_hours": 12.5,
    "by_algorithm": {
      "sdv": {"count": 89, "rows": 500000},
      "ctgan": {"count": 45, "rows": 600000},
      "ydata": {"count": 22, "rows": 150000}
    },
    "by_day": [
      {"date": "2024-01-15", "generations": 23, "rows": 185000},
      {"date": "2024-01-14", "generations": 31, "rows": 220000}
    ]
  },
  "limits": {
    "generations_per_hour": 100,
    "rows_per_hour": 1000000,
    "compute_hours_per_month": 100,
    "remaining": {
      "generations": 44,
      "rows": 250000,
      "compute_hours": 87.5
    }
  }
}
```

### Export and Download

#### `GET /api/tabular/download/{generation_id}`

Download generated data.

**Query Parameters:**
- `format`: Output format (csv, parquet, json)
- `compression`: Compression type (gzip, snappy, none)
- `stream`: Stream download for large files

**Response Headers:**
```
Content-Type: text/csv
Content-Disposition: attachment; filename="synthetic_data.csv"
Content-Length: 1024000
```

## Request & Response Format

### Standard Request Headers

```http
Content-Type: application/json
X-API-Key: your-api-key
X-Request-ID: unique-request-id
Accept: application/json
```

### Standard Response Format

All responses follow this structure:

```json
{
  "data": { ... },           // Response data
  "meta": {                  // Metadata
    "request_id": "req-123",
    "timestamp": "2024-01-15T10:30:00Z",
    "version": "1.0.0"
  },
  "errors": []              // Any errors (empty on success)
}
```

### Pagination

For list endpoints:

```json
{
  "data": [...],
  "meta": {
    "page": 1,
    "per_page": 20,
    "total": 156,
    "total_pages": 8
  },
  "links": {
    "first": "/api/tabular/models?page=1",
    "prev": null,
    "next": "/api/tabular/models?page=2",
    "last": "/api/tabular/models?page=8"
  }
}
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_ALGORITHM",
    "message": "Algorithm 'invalid' is not supported. Valid options are: sdv, ctgan, ydata",
    "field": "algorithm",
    "request_id": "req-123e4567"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNAUTHORIZED` | 401 | Missing or invalid authentication |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `INVALID_REQUEST` | 400 | Invalid request parameters |
| `INVALID_FILE` | 400 | Invalid or corrupted file |
| `UNSUPPORTED_FORMAT` | 400 | Unsupported file format |
| `ALGORITHM_ERROR` | 400 | Algorithm-specific error |
| `CONSTRAINT_VIOLATION` | 400 | Constraint validation failed |
| `RATE_LIMITED` | 429 | Rate limit exceeded |
| `GENERATION_FAILED` | 500 | Generation process failed |
| `TIMEOUT` | 504 | Request timeout |

### Error Examples

```bash
# Invalid API key
curl -X POST https://api.inferloop.com/api/tabular/generate \
  -H "X-API-Key: invalid-key" \
  -F "file=@data.csv"

# Response:
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid API key",
    "request_id": "req-789"
  }
}

# Invalid file format
{
  "error": {
    "code": "INVALID_FILE",
    "message": "File format 'xlsx' is not supported. Supported formats: csv, parquet, json",
    "field": "file",
    "request_id": "req-456"
  }
}
```

## Rate Limiting

### Rate Limit Headers

All responses include rate limit information:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 85
X-RateLimit-Reset: 1642333200
X-RateLimit-Reset-After: 3487
```

### Rate Limits by Tier

| Tier | Requests/Hour | Rows/Hour | Compute Hours/Month | Concurrent Jobs |
|------|---------------|-----------|-------------------|-----------------|
| Starter | 10 | 100,000 | 10 | 1 |
| Professional | 100 | 1,000,000 | 100 | 5 |
| Business | 1,000 | 10,000,000 | 1,000 | 20 |
| Enterprise | Unlimited | Unlimited | Unlimited | Unlimited |

### Handling Rate Limits

```python
import time
import requests
from tenacity import retry, wait_exponential

@retry(wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_with_retry(file_path, **params):
    response = requests.post(
        "https://api.inferloop.com/api/tabular/generate",
        files={"file": open(file_path, "rb")},
        data=params,
        headers={"X-API-Key": api_key}
    )
    
    if response.status_code == 429:
        retry_after = int(response.headers.get("X-RateLimit-Reset-After", 60))
        print(f"Rate limited. Waiting {retry_after} seconds...")
        time.sleep(retry_after)
        raise Exception("Rate limited")
        
    return response.json()
```

## Webhooks

### Webhook Configuration

Configure webhooks for async notifications:

```bash
POST /api/tabular/webhooks
{
  "url": "https://your-app.com/webhook",
  "events": ["generation.completed", "job.failed", "model.trained"],
  "secret": "your-webhook-secret"
}
```

### Webhook Payload

```json
{
  "event": "generation.completed",
  "data": {
    "generation_id": "gen-123",
    "algorithm": "ctgan",
    "rows_generated": 5000,
    "quality_score": 0.91,
    "download_url": "https://api.inferloop.com/api/tabular/download/gen-123"
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "signature": "sha256=abc123..."
}
```

### Webhook Verification

```python
import hmac
import hashlib

def verify_webhook(payload, signature, secret):
    expected = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(
        f"sha256={expected}",
        signature
    )
```

## Examples

### Python Example

```python
import requests
import pandas as pd

class TabularAPI:
    def __init__(self, api_key, base_url="https://api.inferloop.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {"X-API-Key": api_key}
    
    def generate(self, file_path, algorithm="sdv", num_rows=None, **kwargs):
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {
                'algorithm': algorithm,
                'num_rows': num_rows,
                **kwargs
            }
            
            response = requests.post(
                f"{self.base_url}/api/tabular/generate",
                files=files,
                data=data,
                headers=self.headers
            )
            
        response.raise_for_status()
        result = response.json()
        
        # Download generated data
        download_url = result['download_formats']['csv']
        return self._download_data(download_url)
    
    def _download_data(self, url):
        response = requests.get(
            f"{self.base_url}{url}",
            headers=self.headers
        )
        response.raise_for_status()
        
        from io import StringIO
        return pd.read_csv(StringIO(response.text))
    
    def validate(self, original_file, synthetic_file):
        with open(original_file, 'rb') as f1, open(synthetic_file, 'rb') as f2:
            files = {
                'original': f1,
                'synthetic': f2
            }
            
            response = requests.post(
                f"{self.base_url}/api/tabular/validate",
                files=files,
                headers=self.headers
            )
            
        response.raise_for_status()
        return response.json()

# Usage
api = TabularAPI("your-api-key")

# Generate synthetic data
synthetic_df = api.generate(
    "customers.csv",
    algorithm="ctgan",
    num_rows=5000,
    epochs=300
)

# Validate quality
validation = api.validate("customers.csv", "synthetic_customers.csv")
print(f"Quality score: {validation['overall_score']}")
```

### JavaScript Example

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

class TabularAPI {
  constructor(apiKey, baseUrl = 'https://api.inferloop.com') {
    this.apiKey = apiKey;
    this.baseUrl = baseUrl;
    this.headers = { 'X-API-Key': apiKey };
  }

  async generate(filePath, options = {}) {
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));
    
    Object.entries(options).forEach(([key, value]) => {
      form.append(key, value);
    });

    const response = await axios.post(
      `${this.baseUrl}/api/tabular/generate`,
      form,
      {
        headers: {
          ...this.headers,
          ...form.getHeaders()
        }
      }
    );

    // Download generated data
    const downloadUrl = response.data.download_formats.csv;
    const dataResponse = await axios.get(
      `${this.baseUrl}${downloadUrl}`,
      { headers: this.headers }
    );

    return dataResponse.data;
  }

  async streamGenerate(modelId, options = {}) {
    const response = await axios.get(
      `${this.baseUrl}/api/tabular/stream/${modelId}`,
      {
        headers: this.headers,
        params: options,
        responseType: 'stream'
      }
    );

    return response.data;
  }
}

// Usage
const api = new TabularAPI('your-api-key');

// Generate data
const syntheticData = await api.generate('customers.csv', {
  algorithm: 'ctgan',
  num_rows: 1000
});

// Stream data
const stream = await api.streamGenerate('model-123', {
  rate: 100,
  format: 'json'
});

stream.on('data', (chunk) => {
  const records = chunk.toString().split('\n')
    .filter(line => line.startsWith('data: '))
    .map(line => JSON.parse(line.substring(6)));
  
  records.forEach(record => console.log(record));
});
```

### cURL Examples

```bash
# Basic generation
curl -X POST https://api.inferloop.com/api/tabular/generate \
  -H "X-API-Key: $API_KEY" \
  -F "file=@customers.csv" \
  -F "algorithm=sdv" \
  -F "num_rows=1000"

# Advanced generation with constraints
curl -X POST https://api.inferloop.com/api/tabular/generate \
  -H "X-API-Key: $API_KEY" \
  -F "file=@data.csv" \
  -F "algorithm=ctgan" \
  -F "epochs=500" \
  -F "constraints=@constraints.json" \
  -F "config=@config.yaml"

# Profile data
curl -X POST https://api.inferloop.com/api/tabular/profile \
  -H "X-API-Key: $API_KEY" \
  -F "file=@large_dataset.csv" \
  -F "sample_size=10000"

# Validate synthetic data
curl -X POST https://api.inferloop.com/api/tabular/validate \
  -H "X-API-Key: $API_KEY" \
  -F "original=@original.csv" \
  -F "synthetic=@synthetic.csv" \
  -F "metrics=statistical,privacy"

# Stream synthetic data
curl -X GET "https://api.inferloop.com/api/tabular/stream/model-123?rate=50" \
  -H "X-API-Key: $API_KEY" \
  -H "Accept: text/event-stream"
```

## SDK Integration

### Official SDKs

- **Python**: `pip install inferloop-tabular`
- **JavaScript/Node**: `npm install @inferloop/tabular`
- **Java**: `com.inferloop:tabular-sdk:1.0.0`
- **Go**: `go get github.com/inferloop/tabular-go`

### Community SDKs

- **Ruby**: `gem install inferloop-tabular`
- **PHP**: `composer require inferloop/tabular`
- **Rust**: `cargo add inferloop-tabular`

## Best Practices

1. **Use appropriate file formats**: CSV for compatibility, Parquet for performance
2. **Implement retry logic**: Handle transient failures gracefully
3. **Monitor rate limits**: Track usage to avoid hitting limits
4. **Use async generation**: For large datasets (>100k rows)
5. **Cache models**: Reuse trained models for similar data
6. **Validate input**: Check data quality before generation
7. **Handle errors**: Provide meaningful error messages to users
8. **Use webhooks**: For long-running operations
9. **Secure API keys**: Never expose in client-side code
10. **Log requests**: For debugging and analytics

## API Changelog

### v1.2.0 (2024-01-15)
- Added streaming generation endpoints
- Introduced multi-table generation
- Added privacy analysis endpoints
- Improved constraint validation

### v1.1.0 (2023-12-01)
- Added async generation for large datasets
- Introduced model management endpoints
- Added batch processing
- Improved error messages

### v1.0.0 (2023-10-15)
- Initial API release
- Basic generation and validation
- Support for SDV, CTGAN, YData
- JWT and API key authentication

## Support

- **API Status**: [status.inferloop.com](https://status.inferloop.com)
- **Documentation**: [docs.inferloop.com/api](https://docs.inferloop.com/api)
- **Support**: api-support@inferloop.com
- **Discord**: [discord.gg/inferloop](https://discord.gg/inferloop)