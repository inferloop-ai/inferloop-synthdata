#!/usr/bin/env python3
"""
Script to create the directory and file structure for the CodeAPI DSL project.
This script will create all directories and empty files as specified in the
project structure.
"""

import os
import sys

def create_directory(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

def create_empty_file(path):
    """Create an empty file if it doesn't exist."""
    if not os.path.exists(path):
        with open(path, 'w') as f:
            pass  # Create empty file
        print(f"Created empty file: {path}")
    else:
        print(f"File already exists: {path}")

def main():
    # Get the base directory (where this script is located)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the main project structure
    project_structure = [
        # Root files
        "README.md",
        "requirements.txt",
        "setup.py",
        
        # Config directory
        "config/models.yaml",
        "config/validation_rules.yaml",
        "config/output_templates.yaml",
        
        # Source code
        "src/__init__.py",
        
        # Generators
        "src/generators/__init__.py",
        "src/generators/base_generator.py",
        "src/generators/code_llama_generator.py",
        "src/generators/starcoder_generator.py",
        "src/generators/openapi_generator.py",
        "src/generators/dsl_generator.py",
        
        # Validators
        "src/validators/__init__.py",
        "src/validators/syntax_validator.py",
        "src/validators/compilation_validator.py",
        "src/validators/unit_test_validator.py",
        
        # Delivery
        "src/delivery/__init__.py",
        "src/delivery/formatters.py",
        "src/delivery/exporters.py",
        "src/delivery/grpc_mocks.py",
        
        # API
        "src/api/__init__.py",
        "src/api/routes.py",
        "src/api/models.py",
        "src/api/middleware.py",
        
        # CLI
        "src/cli/__init__.py",
        "src/cli/commands.py",
        "src/cli/utils.py",
        
        # SDK
        "src/sdk/__init__.py",
        "src/sdk/client.py",
        "src/sdk/exceptions.py",
        
        # Tests
        "tests/__init__.py",
        "tests/test_generators.py",
        "tests/test_validators.py",
        "tests/test_api.py",
        "tests/test_cli.py",
        "tests/test_sdk.py",
        
        # Examples directories
        "examples/generated_code_samples/.gitkeep",
        "examples/api_schemas/.gitkeep",
        "examples/dsl_examples/.gitkeep",
        
        # Docker
        "docker/Dockerfile",
        "docker/docker-compose.yml",
        
        # Scripts
        "scripts/setup.sh",
        "scripts/run_validation.sh",
    ]
    
    # Create all directories and files
    for item in project_structure:
        full_path = os.path.join(base_dir, item)
        
        # Create parent directory if it doesn't exist
        parent_dir = os.path.dirname(full_path)
        if parent_dir and not os.path.exists(parent_dir):
            create_directory(parent_dir)
        
        # Create the file
        create_empty_file(full_path)
    
    print("\nDirectory and file structure created successfully!")

if __name__ == "__main__":
    main()
