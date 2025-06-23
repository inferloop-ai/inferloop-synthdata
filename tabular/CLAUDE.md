# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Inferloop Synthetic Data SDK is a unified wrapper for synthetic tabular data generation tools (SDV, CTGAN, YData-Synthetic). It provides SDK, CLI, and REST API interfaces with comprehensive validation capabilities.

## Essential Commands

### Development Setup
```bash
# Install for development with all dependencies
pip install -e ".[dev,all]"
```

### Testing & Quality
```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov

# Code formatting (MUST run before committing)
black .
isort .

# Type checking
mypy .

# Linting
flake8
```

### Running the Application
```bash
# CLI interface
inferloop-synthetic generate data.csv output.csv --generator-type sdv --model-type gaussian_copula

# REST API server
uvicorn inferloop_synthetic.api.app:app --host 0.0.0.0 --port 8000
```

## Architecture Overview

The project follows a multi-interface architecture with clear separation of concerns:

```
User Interface Layer:
├── CLI (cli/main.py) - Typer-based command interface
├── REST API (api/app.py) - FastAPI async endpoints
└── SDK (sdk/factory.py) - Direct Python usage

Core Layer:
├── sdk/base.py - Abstract base classes (BaseSyntheticGenerator, SyntheticDataConfig)
├── sdk/*_generator.py - Concrete implementations for each library
└── sdk/validator.py - Comprehensive validation framework

Data Flow:
User → Interface → GeneratorFactory → Specific Generator → Synthetic Data
                                              ↓
                                          Validator → Quality Metrics
```

### Key Design Patterns

1. **Factory Pattern**: `GeneratorFactory.create_generator()` instantiates appropriate generator based on config
2. **Configuration-Driven**: All settings flow through `SyntheticDataConfig` dataclass
3. **Abstract Base Class**: All generators inherit from `BaseSyntheticGenerator`
4. **Result Encapsulation**: `GenerationResult` wraps data + metadata + metrics

### Adding New Features

When implementing new synthetic data generators:
1. Create new class inheriting from `BaseSyntheticGenerator` in `sdk/`
2. Implement required methods: `fit()`, `generate()`, `fit_generate()`
3. Register in `GeneratorFactory._create_*_generator()` methods
4. Add configuration template in `data/sample_templates/`
5. Update CLI and API to support new generator type

When adding validation metrics:
1. Extend `SyntheticDataValidator` class in `sdk/validator.py`
2. Follow existing pattern of returning dict with metric results
3. Ensure graceful handling when optional dependencies missing

## Code Style Requirements

- **Black formatter**: Line length 88 (configured in pyproject.toml)
- **Type hints**: Required for all public functions (mypy strict mode)
- **Imports**: Use isort with Black-compatible profile
- **Docstrings**: Required for all public classes and methods

## Testing Guidelines

- Tests located in `tests/` directory
- Use pytest fixtures for common test data
- Mock external library calls to avoid dependencies in tests
- Test files should mirror source structure

## Important Files

- `sdk/base.py`: Core abstractions and data structures
- `sdk/factory.py`: Main entry point for generator creation
- `cli/main.py`: CLI command definitions
- `api/app.py`: REST API endpoints
- `pyproject.toml`: Project configuration and dependencies