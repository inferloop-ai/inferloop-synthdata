# TextNLP REST API Documentation

## Overview

The TextNLP REST API provides programmatic access to synthetic text generation capabilities through HTTP endpoints. This documentation covers authentication, endpoints, request/response formats, and best practices.

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
inferloop-nlp serve --port 8000

# Test the API
curl http://localhost:8000/health

# Generate text
curl -X POST http://localhost:8000/api/textnlp/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"prompt": "Write a story about AI"}'
```

### API Client Libraries

```python
# Python client
from inferloop import TextNLPClient

client = TextNLPClient(api_key="your-api-key")
result = client.generate("Write about machine learning")
```

```javascript
// JavaScript client
const TextNLP = require('@inferloop/textnlp');

const client = new TextNLP({ apiKey: 'your-api-key' });
const result = await client.generate('Write about machine learning');
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
inferloop-nlp config set api.key "your-api-key"

# Via environment variable
export INFERLOOP_API_KEY="your-api-key"
```

## Base URL

```
Production: https://api.inferloop.com
Development: http://localhost:8000
```

All endpoints are prefixed with `/api/textnlp/v1`

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
  "models_loaded": ["gpt2", "gpt2-large"],
  "cache_connected": true,
  "database_connected": true
}
```

### Text Generation

#### `POST /api/textnlp/generate`

Generate synthetic text from a prompt.

**Request Body:**
```json
{
  "prompt": "Write a product review for wireless headphones",
  "model": "gpt2-large",
  "max_tokens": 200,
  "min_tokens": 50,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.2,
  "seed": 42,
  "stream": false
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Input text prompt |
| `model` | string | `"gpt2"` | Model to use |
| `max_tokens` | integer | `100` | Maximum tokens to generate |
| `min_tokens` | integer | `0` | Minimum tokens to generate |
| `temperature` | float | `0.7` | Sampling temperature (0.0-2.0) |
| `top_p` | float | `0.9` | Nucleus sampling threshold |
| `top_k` | integer | `50` | Top-k sampling parameter |
| `repetition_penalty` | float | `1.0` | Penalty for repetition |
| `seed` | integer | `null` | Random seed |
| `stream` | boolean | `false` | Stream response |

**Response:**
```json
{
  "id": "gen-123e4567-e89b-12d3-a456-426614174000",
  "model": "gpt2-large",
  "created": 1642329600,
  "text": "I recently purchased these wireless headphones and I'm extremely impressed. The sound quality is exceptional with crisp highs and deep bass. The noise cancellation feature works wonderfully, blocking out ambient noise during my commute. Battery life easily lasts 30+ hours on a single charge. The comfortable ear cushions make them perfect for extended listening sessions. Build quality feels premium with sturdy construction. The only minor drawback is they're slightly heavy, but the comfort padding compensates well. Overall, these headphones offer excellent value and I highly recommend them to anyone seeking quality wireless audio.",
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 95,
    "total_tokens": 103
  },
  "metadata": {
    "generation_time": 1.234,
    "temperature": 0.7,
    "top_p": 0.9
  }
}
```

### Streaming Generation

#### `POST /api/textnlp/generate` (with streaming)

Stream text generation for real-time output.

**Request:**
```json
{
  "prompt": "Write a long article about space exploration",
  "model": "gpt2-xl",
  "max_tokens": 1000,
  "stream": true
}
```

**Response** (Server-Sent Events):
```
event: token
data: {"token": "Space", "index": 0}

event: token
data: {"token": " exploration", "index": 1}

event: token
data: {"token": " has", "index": 2}

event: done
data: {"total_tokens": 850, "generation_time": 12.5}
```

### Chat Completion

#### `POST /api/textnlp/chat`

Generate chat-style completions.

**Request Body:**
```json
{
  "messages": [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence..."},
    {"role": "user", "content": "Can you give me an example?"}
  ],
  "model": "gpt2-xl",
  "max_tokens": 200,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "id": "chat-123e4567-e89b-12d3-a456-426614174000",
  "model": "gpt2-xl",
  "created": 1642329600,
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Sure! A common example of machine learning is email spam filtering. The system learns from examples of spam and legitimate emails to automatically classify new incoming messages..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 120,
    "total_tokens": 165
  }
}
```

### Text Validation

#### `POST /api/textnlp/validate`

Validate generated text quality.

**Request Body:**
```json
{
  "generated_text": "The generated text to validate",
  "reference_text": "Optional reference text for comparison",
  "metrics": ["bleu", "rouge", "quality", "toxicity"]
}
```

**Response:**
```json
{
  "id": "val-123e4567-e89b-12d3-a456-426614174000",
  "metrics": {
    "bleu": {
      "bleu_1": 0.75,
      "bleu_2": 0.62,
      "bleu_3": 0.48,
      "bleu_4": 0.35,
      "average": 0.55
    },
    "rouge": {
      "rouge_1": {"f1": 0.68, "precision": 0.72, "recall": 0.65},
      "rouge_2": {"f1": 0.45, "precision": 0.48, "recall": 0.42},
      "rouge_l": {"f1": 0.61, "precision": 0.65, "recall": 0.58}
    },
    "quality": {
      "score": 0.82,
      "readability": 8.5,
      "grammar_score": 0.95,
      "coherence": 0.88
    },
    "toxicity": {
      "score": 0.02,
      "safe": true
    }
  },
  "summary": {
    "overall_quality": "high",
    "recommendations": ["Consider varying sentence structure", "Add more specific details"]
  }
}
```

### Batch Generation

#### `POST /api/textnlp/batch`

Generate text for multiple prompts.

**Request Body:**
```json
{
  "prompts": [
    {"id": "1", "text": "Write about AI ethics"},
    {"id": "2", "text": "Describe quantum computing"},
    {"id": "3", "text": "Explain blockchain technology"}
  ],
  "model": "gpt2-large",
  "max_tokens": 150,
  "common_params": {
    "temperature": 0.7,
    "top_p": 0.9
  }
}
```

**Response:**
```json
{
  "batch_id": "batch-123e4567-e89b-12d3-a456-426614174000",
  "status": "processing",
  "total": 3,
  "completed": 0,
  "webhook_url": "https://api.inferloop.com/api/textnlp/batch/batch-123e4567-e89b-12d3-a456-426614174000"
}
```

#### `GET /api/textnlp/batch/{batch_id}`

Get batch generation status.

**Response:**
```json
{
  "batch_id": "batch-123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "total": 3,
  "completed": 3,
  "failed": 0,
  "results": [
    {
      "id": "1",
      "status": "completed",
      "text": "AI ethics is a critical field that examines...",
      "usage": {"total_tokens": 142}
    },
    {
      "id": "2",
      "status": "completed",
      "text": "Quantum computing represents a paradigm shift...",
      "usage": {"total_tokens": 138}
    },
    {
      "id": "3",
      "status": "completed",
      "text": "Blockchain technology is a distributed ledger...",
      "usage": {"total_tokens": 145}
    }
  ],
  "created_at": "2024-01-15T10:00:00Z",
  "completed_at": "2024-01-15T10:02:30Z"
}
```

### Template Generation

#### `POST /api/textnlp/templates/{template_id}/generate`

Generate text using predefined templates.

**Request Body:**
```json
{
  "variables": {
    "product_name": "SmartWatch Pro",
    "features": ["heart rate monitor", "GPS", "water resistant"],
    "price": "$299"
  },
  "model": "gpt2-large",
  "temperature": 0.7
}
```

**Response:**
```json
{
  "id": "gen-template-123e4567",
  "template_id": "product_description",
  "text": "Introducing the SmartWatch Pro - your ultimate fitness companion at just $299. This advanced wearable features a precise heart rate monitor to track your health 24/7, built-in GPS for accurate route mapping during workouts, and water-resistant design that withstands your toughest training sessions...",
  "usage": {
    "total_tokens": 156
  }
}
```

#### `GET /api/textnlp/templates`

List available templates.

**Response:**
```json
{
  "templates": [
    {
      "id": "product_description",
      "name": "Product Description",
      "description": "Generate e-commerce product descriptions",
      "variables": ["product_name", "features", "price"],
      "category": "e-commerce"
    },
    {
      "id": "email_response",
      "name": "Email Response",
      "description": "Generate professional email responses",
      "variables": ["recipient", "subject", "context"],
      "category": "business"
    }
  ],
  "total": 25
}
```

### Model Management

#### `GET /api/textnlp/models`

List available models.

**Response:**
```json
{
  "models": [
    {
      "id": "gpt2",
      "name": "GPT-2",
      "parameters": "124M",
      "description": "Fast, general-purpose text generation",
      "supported_languages": ["en"],
      "max_tokens": 1024,
      "tier_required": "starter"
    },
    {
      "id": "gpt2-large",
      "name": "GPT-2 Large",
      "parameters": "774M",
      "description": "High-quality text generation",
      "supported_languages": ["en"],
      "max_tokens": 2048,
      "tier_required": "professional"
    },
    {
      "id": "gpt-j-6b",
      "name": "GPT-J 6B",
      "parameters": "6B",
      "description": "State-of-the-art text generation",
      "supported_languages": ["en", "es", "fr", "de"],
      "max_tokens": 4096,
      "tier_required": "business",
      "requires_gpu": true
    }
  ]
}
```

### Usage Statistics

#### `GET /api/textnlp/usage`

Get usage statistics for the authenticated user.

**Query Parameters:**
- `period`: Time period (`24h`, `7d`, `30d`)
- `timezone`: Timezone for date grouping

**Response:**
```json
{
  "period": "7d",
  "usage": {
    "total_requests": 1523,
    "total_tokens": 256789,
    "total_characters": 1024567,
    "by_model": {
      "gpt2": {"requests": 523, "tokens": 52300},
      "gpt2-large": {"requests": 1000, "tokens": 204489}
    },
    "by_day": [
      {"date": "2024-01-15", "requests": 234, "tokens": 45678},
      {"date": "2024-01-14", "requests": 189, "tokens": 38901}
    ]
  },
  "limits": {
    "requests_per_hour": 1000,
    "tokens_per_hour": 1000000,
    "requests_remaining": 766,
    "tokens_remaining": 743211
  }
}
```

### WebSocket Streaming

#### `WS /ws/textnlp/stream`

WebSocket endpoint for real-time streaming.

**Connection:**
```javascript
const ws = new WebSocket('wss://api.inferloop.com/ws/textnlp/stream');

ws.on('open', () => {
  ws.send(JSON.stringify({
    type: 'generate',
    payload: {
      prompt: 'Write a story',
      model: 'gpt2-large',
      max_tokens: 500
    }
  }));
});

ws.on('message', (data) => {
  const message = JSON.parse(data);
  if (message.type === 'token') {
    process.stdout.write(message.token);
  } else if (message.type === 'complete') {
    console.log('\nGeneration complete:', message.stats);
  }
});
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
    "first": "/api/textnlp/templates?page=1",
    "prev": null,
    "next": "/api/textnlp/templates?page=2",
    "last": "/api/textnlp/templates?page=8"
  }
}
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Invalid temperature value: must be between 0.0 and 2.0",
    "field": "temperature",
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
| `RATE_LIMITED` | 429 | Rate limit exceeded |
| `MODEL_UNAVAILABLE` | 503 | Requested model is unavailable |
| `GENERATION_FAILED` | 500 | Text generation failed |
| `TIMEOUT` | 504 | Request timeout |

### Error Examples

```bash
# Invalid API key
curl -X POST https://api.inferloop.com/api/textnlp/generate \
  -H "X-API-Key: invalid-key" \
  -d '{"prompt": "Test"}'

# Response:
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid API key",
    "request_id": "req-789"
  }
}

# Rate limit exceeded
{
  "error": {
    "code": "RATE_LIMITED",
    "message": "Rate limit exceeded. Please retry after 60 seconds",
    "retry_after": 60,
    "request_id": "req-456"
  }
}
```

## Rate Limiting

### Rate Limit Headers

All responses include rate limit information:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 985
X-RateLimit-Reset: 1642333200
X-RateLimit-Reset-After: 3487
```

### Rate Limits by Tier

| Tier | Requests/Hour | Tokens/Hour | Concurrent |
|------|---------------|-------------|------------|
| Starter | 100 | 500,000 | 2 |
| Professional | 1,000 | 5,000,000 | 10 |
| Business | 10,000 | 25,000,000 | 50 |
| Enterprise | Unlimited | Unlimited | Unlimited |

### Handling Rate Limits

```python
import time
import requests

def generate_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        response = requests.post(
            "https://api.inferloop.com/api/textnlp/generate",
            json={"prompt": prompt},
            headers={"X-API-Key": api_key}
        )
        
        if response.status_code == 429:
            retry_after = int(response.headers.get("X-RateLimit-Reset-After", 60))
            print(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            continue
            
        return response.json()
    
    raise Exception("Max retries exceeded")
```

## Webhooks

### Webhook Configuration

Configure webhooks for async notifications:

```bash
POST /api/textnlp/webhooks
{
  "url": "https://your-app.com/webhook",
  "events": ["generation.completed", "batch.completed", "generation.failed"],
  "secret": "your-webhook-secret"
}
```

### Webhook Payload

```json
{
  "event": "generation.completed",
  "data": {
    "id": "gen-123",
    "prompt": "Original prompt",
    "text": "Generated text...",
    "model": "gpt2-large",
    "usage": {
      "total_tokens": 156
    }
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
import json

class TextNLPAPI:
    def __init__(self, api_key, base_url="https://api.inferloop.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
    
    def generate(self, prompt, **kwargs):
        data = {"prompt": prompt, **kwargs}
        response = requests.post(
            f"{self.base_url}/api/textnlp/generate",
            json=data,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def validate(self, text, reference=None, metrics=None):
        data = {
            "generated_text": text,
            "reference_text": reference,
            "metrics": metrics or ["quality"]
        }
        response = requests.post(
            f"{self.base_url}/api/textnlp/validate",
            json=data,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

# Usage
api = TextNLPAPI("your-api-key")

# Generate text
result = api.generate(
    "Write a product review",
    model="gpt2-large",
    max_tokens=200,
    temperature=0.8
)
print(result["text"])

# Validate quality
validation = api.validate(result["text"])
print(f"Quality score: {validation['metrics']['quality']['score']}")
```

### JavaScript Example

```javascript
class TextNLPAPI {
  constructor(apiKey, baseUrl = 'https://api.inferloop.com') {
    this.apiKey = apiKey;
    this.baseUrl = baseUrl;
  }

  async generate(prompt, options = {}) {
    const response = await fetch(`${this.baseUrl}/api/textnlp/generate`, {
      method: 'POST',
      headers: {
        'X-API-Key': this.apiKey,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ prompt, ...options })
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }

    return response.json();
  }

  async streamGenerate(prompt, options = {}, onToken) {
    const response = await fetch(`${this.baseUrl}/api/textnlp/generate`, {
      method: 'POST',
      headers: {
        'X-API-Key': this.apiKey,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ prompt, stream: true, ...options })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));
          if (data.token) {
            onToken(data.token);
          }
        }
      }
    }
  }
}

// Usage
const api = new TextNLPAPI('your-api-key');

// Generate text
const result = await api.generate('Write about AI', {
  model: 'gpt2-large',
  maxTokens: 150
});
console.log(result.text);

// Stream generation
await api.streamGenerate(
  'Write a long story',
  { model: 'gpt2-xl', maxTokens: 1000 },
  (token) => process.stdout.write(token)
);
```

### cURL Examples

```bash
# Basic generation
curl -X POST https://api.inferloop.com/api/textnlp/generate \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a haiku about programming",
    "model": "gpt2-large",
    "temperature": 0.9
  }'

# Streaming generation
curl -X POST https://api.inferloop.com/api/textnlp/generate \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "prompt": "Tell me a story",
    "stream": true,
    "max_tokens": 500
  }'

# Batch generation
curl -X POST https://api.inferloop.com/api/textnlp/batch \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": [
      {"id": "1", "text": "Write about space"},
      {"id": "2", "text": "Write about ocean"}
    ],
    "model": "gpt2-large"
  }'

# Check batch status
curl -X GET https://api.inferloop.com/api/textnlp/batch/batch-123 \
  -H "X-API-Key: $API_KEY"
```

## SDK Integration

### Official SDKs

- **Python**: `pip install inferloop-textnlp`
- **JavaScript/Node**: `npm install @inferloop/textnlp`
- **Java**: `com.inferloop:textnlp-sdk:1.0.0`
- **Go**: `go get github.com/inferloop/textnlp-go`

### Community SDKs

- **Ruby**: `gem install inferloop-textnlp`
- **PHP**: `composer require inferloop/textnlp`
- **Rust**: `cargo add inferloop-textnlp`

## Best Practices

1. **Use appropriate models**: Balance quality vs speed
2. **Implement retry logic**: Handle transient failures
3. **Cache responses**: Reduce API calls for repeated prompts
4. **Batch when possible**: More efficient than individual requests
5. **Monitor rate limits**: Track usage to avoid hitting limits
6. **Validate input**: Check prompts before sending
7. **Handle errors gracefully**: Provide fallbacks
8. **Use webhooks**: For long-running operations
9. **Secure API keys**: Never expose in client-side code
10. **Log requests**: For debugging and analytics

## API Changelog

### v1.2.0 (2024-01-15)
- Added streaming support for all generation endpoints
- Introduced batch generation API
- Added GPT-J-6B model support
- Improved rate limiting with tier-based limits

### v1.1.0 (2023-12-01)
- Added template generation endpoints
- Introduced WebSocket streaming
- Added usage statistics endpoint
- Improved validation metrics

### v1.0.0 (2023-10-15)
- Initial API release
- Basic generation and validation endpoints
- Support for GPT-2 model family
- JWT and API key authentication

## Support

- **API Status**: [status.inferloop.com](https://status.inferloop.com)
- **Documentation**: [docs.inferloop.com/api](https://docs.inferloop.com/api)
- **Support**: api-support@inferloop.com
- **Discord**: [discord.gg/inferloop](https://discord.gg/inferloop)