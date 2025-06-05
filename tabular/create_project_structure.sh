#!/bin/bash

# Create the main project directory
mkdir -p tabular

# Create subdirectories and files
mkdir -p tabular/sdk
mkdir -p tabular/cli
mkdir -p tabular/api
mkdir -p tabular/examples
mkdir -p tabular/data/sample_templates
mkdir -p tabular/tests

# Create empty files
touch tabular/sdk/__init__.py
touch tabular/sdk/base.py
touch tabular/sdk/sdv_generator.py
touch tabular/sdk/ctgan_generator.py
touch tabular/sdk/ydata_generator.py
touch tabular/sdk/validator.py
touch tabular/cli/main.py
touch tabular/api/app.py
touch tabular/api/routes.py
touch tabular/examples/notebook.ipynb
touch tabular/tests/test_sdk.py
touch tabular/pyproject.toml
touch tabular/README.md

echo "Project structure created successfully!"
