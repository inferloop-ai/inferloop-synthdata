# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in the TextNLP repository.

## Project Overview

TextNLP Synthetic Data SDK is a comprehensive platform for generating synthetic text and NLP data using various language models (GPT-2, GPT-J, LLaMA, etc.). It provides SDK, CLI, and REST API interfaces with advanced validation capabilities including BLEU/ROUGE metrics and human-in-the-loop evaluation.

## Essential Commands

### Development Setup
```bash
# Install for development with all dependencies
pip install -e ".[dev,all]"

# Install pre-commit hooks
pre-commit install
```

### Testing & Quality
```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=textnlp --cov-report=html

# Code formatting (MUST run before committing)
black .
isort .

# Type checking
mypy textnlp

# Linting
flake8
ruff check .
```

### Running the Application
```bash
# CLI interface
inferloop-nlp generate "Write a story" --model gpt2 --output results.jsonl
inferloop-nlp validate refs.txt candidates.txt --output scores.json

# REST API server (development)
uvicorn textnlp.api.app:app --reload --host 0.0.0.0 --port 8000

# REST API server (production)
gunicorn textnlp.api.app:app -w 4 -k uvicorn.workers.UvicornWorker
```

## Architecture Overview

The project follows a multi-interface architecture designed for scalability and extensibility:

```
User Interface Layer:
├── CLI (cli/main.py) - Typer-based command interface
├── REST API (api/app.py) - FastAPI async endpoints with streaming
└── SDK (sdk/base_generator.py) - Direct Python usage

Core Layer:
├── sdk/base_generator.py - Abstract base class for all generators
├── sdk/llm_gpt2.py - GPT-2 family implementation
├── sdk/langchain_template.py - Advanced prompt management
└── sdk/validation/ - Comprehensive validation framework

Data Flow:
User → Interface → Generator Selection → Model Loading → Generation
                                              ↓
                                        Validation → Metrics
                                              ↓
                                        Formatting → Output
```

### Key Design Patterns

1. **Abstract Base Class**: All generators inherit from `BaseGenerator` for consistency
2. **Factory Pattern**: Model selection and instantiation handled dynamically
3. **Async-First**: API designed for high concurrency with async/await
4. **Streaming Support**: Built-in support for token-by-token generation
5. **Validation Pipeline**: Pluggable validation metrics and human evaluation

### Adding New Features

When implementing new text generation models:
1. Create new class inheriting from `BaseGenerator` in `sdk/`
2. Implement required methods: `generate()`, `batch_generate()`, `stream_generate()`
3. Add model configuration in `configs/models.yaml`
4. Update CLI commands in `cli/commands/` if needed
5. Add API endpoints in `api/endpoints/` for new functionality
6. Write comprehensive tests in `tests/`

When adding validation metrics:
1. Create new validator class in `sdk/validation/`
2. Implement `validate()` and `validate_batch()` methods
3. Register validator in the validation pipeline
4. Add CLI support in `cli/commands/validate.py`

## Code Style Requirements

- **Black formatter**: Line length 88 (configured in pyproject.toml)
- **Import sorting**: isort with Black-compatible profile
- **Type hints**: Required for all public functions and methods
- **Docstrings**: Google style for all public classes and functions
- **Variable naming**: snake_case for functions/variables, PascalCase for classes
- **No print statements**: Use logging module instead

## Testing Guidelines

- Tests located in `tests/` mirroring source structure
- Use pytest fixtures for common test data
- Mock external API calls to avoid dependencies
- Aim for >90% code coverage on new features
- Test files must start with `test_`
- Use `pytest.mark.asyncio` for async tests

Example test structure:
```python
import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def mock_model():
    return Mock(spec=GPT2Generator)

@pytest.mark.asyncio
async def test_generate_text(mock_model):
    # Test implementation
    pass
```

## Important Files and Their Purposes

- `sdk/base_generator.py`: Core abstraction for all text generators
- `sdk/llm_gpt2.py`: Reference implementation using Hugging Face transformers
- `sdk/langchain_template.py`: Advanced prompt templating with LangChain
- `sdk/validation/bleu_rouge.py`: Automatic quality metrics
- `api/app.py`: FastAPI application setup and middleware
- `api/routes.py`: API endpoint definitions
- `cli/main.py`: CLI command definitions
- `pyproject.toml`: Project configuration and dependencies
- `docker/Dockerfile`: Multi-stage Docker build configuration

## API Design Principles

1. **RESTful conventions**: Use proper HTTP methods and status codes
2. **Consistent responses**: All endpoints return standardized JSON responses
3. **Error handling**: Comprehensive error messages with proper status codes
4. **Rate limiting**: Built-in rate limiting per user/IP
5. **Authentication**: JWT-based auth with refresh tokens
6. **Versioning**: API versioning through URL path (/v1/, /v2/)

## Performance Considerations

- **Model caching**: Models are cached in memory after first load
- **Batch processing**: Optimize for batch operations over single requests
- **Streaming**: Use server-sent events for real-time generation
- **Connection pooling**: Database and Redis connections are pooled
- **Async operations**: Use async/await for I/O operations
- **GPU optimization**: Automatic batching for GPU inference

## Security Best Practices

- **Input validation**: Sanitize all user inputs
- **Rate limiting**: Prevent abuse through configurable limits
- **Authentication**: Require auth for all generation endpoints
- **Secrets management**: Use environment variables, never commit secrets
- **CORS configuration**: Properly configure allowed origins
- **SQL injection**: Use parameterized queries with SQLAlchemy
- **Prompt injection**: Validate and sanitize prompts

## Deployment Considerations

The application is designed for cloud-native deployment:
- **Containerized**: Full Docker support with multi-stage builds
- **Kubernetes-ready**: Helm charts and manifests provided
- **Cloud agnostic**: Supports AWS, Azure, GCP deployments
- **Horizontally scalable**: Stateless design allows easy scaling
- **Health checks**: Built-in health and readiness endpoints
- **Metrics**: Prometheus-compatible metrics endpoint

## Common Patterns to Follow

### Error Handling
```python
from textnlp.exceptions import GenerationError, ValidationError

try:
    result = await generator.generate(prompt)
except GenerationError as e:
    logger.error(f"Generation failed: {e}")
    raise HTTPException(status_code=500, detail=str(e))
```

### Async Context Managers
```python
async with ModelManager() as manager:
    model = await manager.get_model("gpt2")
    result = await model.generate(prompt)
```

### Configuration Management
```python
from textnlp.config import settings

# Access configuration
max_tokens = settings.generation.max_tokens
model_name = settings.models.default
```

## Development Workflow

1. Create feature branch from `main`
2. Write tests first (TDD approach)
3. Implement feature
4. Ensure all tests pass
5. Run code quality checks
6. Update documentation
7. Submit PR with clear description

## Debugging Tips

- Use `import ipdb; ipdb.set_trace()` for debugging
- Enable debug logging: `LOG_LEVEL=DEBUG`
- Use `pytest -s` to see print outputs during tests
- FastAPI automatic docs at `/docs` for API testing
- Check `logs/` directory for detailed application logs

## Resource Management

- Models are loaded on-demand and cached
- Implement proper cleanup in `__del__` methods
- Use context managers for resource allocation
- Monitor memory usage for large models
- Implement model unloading for memory-constrained environments

## Future Development Areas

Priority areas for enhancement:
1. Multi-GPU support for large model inference
2. Fine-tuning pipeline for custom models
3. Advanced caching strategies (Redis clustering)
4. Real-time collaboration features
5. Enhanced security with OAuth2/OIDC
6. GraphQL API alongside REST
7. WebSocket support for streaming
8. Distributed task queue with Celery