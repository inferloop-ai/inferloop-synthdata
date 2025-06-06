#!/bin/bash
# Script to create the directory and file structure for the CodeAPI DSL project

# Function to create a directory if it doesn't exist
create_directory() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        echo "Created directory: $1"
    else
        echo "Directory already exists: $1"
    fi
}

# Function to create an empty file if it doesn't exist
create_empty_file() {
    if [ ! -f "$1" ]; then
        touch "$1"
        echo "Created empty file: $1"
    else
        echo "File already exists: $1"
    fi
}

# Get the base directory (where this script is located)
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

echo "Creating project structure in: $BASE_DIR"

# Create directories
create_directory "config"
create_directory "src/generators"
create_directory "src/validators"
create_directory "src/delivery"
create_directory "src/api"
create_directory "src/cli"
create_directory "src/sdk"
create_directory "tests"
create_directory "examples/generated_code_samples"
create_directory "examples/api_schemas"
create_directory "examples/dsl_examples"
create_directory "docker"
create_directory "scripts"

# Create root files
create_empty_file "README.md"
create_empty_file "requirements.txt"
create_empty_file "setup.py"

# Create config files
create_empty_file "config/models.yaml"
create_empty_file "config/validation_rules.yaml"
create_empty_file "config/output_templates.yaml"

# Create source files
create_empty_file "src/__init__.py"

# Create generator files
create_empty_file "src/generators/__init__.py"
create_empty_file "src/generators/base_generator.py"
create_empty_file "src/generators/code_llama_generator.py"
create_empty_file "src/generators/starcoder_generator.py"
create_empty_file "src/generators/openapi_generator.py"
create_empty_file "src/generators/dsl_generator.py"

# Create validator files
create_empty_file "src/validators/__init__.py"
create_empty_file "src/validators/syntax_validator.py"
create_empty_file "src/validators/compilation_validator.py"
create_empty_file "src/validators/unit_test_validator.py"

# Create delivery files
create_empty_file "src/delivery/__init__.py"
create_empty_file "src/delivery/formatters.py"
create_empty_file "src/delivery/exporters.py"
create_empty_file "src/delivery/grpc_mocks.py"

# Create API files
create_empty_file "src/api/__init__.py"
create_empty_file "src/api/routes.py"
create_empty_file "src/api/models.py"
create_empty_file "src/api/middleware.py"

# Create CLI files
create_empty_file "src/cli/__init__.py"
create_empty_file "src/cli/commands.py"
create_empty_file "src/cli/utils.py"

# Create SDK files
create_empty_file "src/sdk/__init__.py"
create_empty_file "src/sdk/client.py"
create_empty_file "src/sdk/exceptions.py"

# Create test files
create_empty_file "tests/__init__.py"
create_empty_file "tests/test_generators.py"
create_empty_file "tests/test_validators.py"
create_empty_file "tests/test_api.py"
create_empty_file "tests/test_cli.py"
create_empty_file "tests/test_sdk.py"

# Create example placeholder files
create_empty_file "examples/generated_code_samples/.gitkeep"
create_empty_file "examples/api_schemas/.gitkeep"
create_empty_file "examples/dsl_examples/.gitkeep"

# Create Docker files
create_empty_file "docker/Dockerfile"
create_empty_file "docker/docker-compose.yml"

# Create script files
create_empty_file "scripts/setup.sh"
create_empty_file "scripts/run_validation.sh"

echo "Directory and file structure created successfully!"
