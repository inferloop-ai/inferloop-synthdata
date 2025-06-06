#!/bin/bash
# Script to run validation on generated code

set -e

echo "Starting validation process..."

# Define paths
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$BASE_DIR/../src"
CONFIG_DIR="$BASE_DIR/../config"
EXAMPLES_DIR="$BASE_DIR/../examples"

# Run syntax validation
echo "Running syntax validation..."
python "$SRC_DIR/validators/syntax_validator.py"

# Run compilation validation
echo "Running compilation validation..."
python "$SRC_DIR/validators/compilation_validator.py"

# Run unit tests validation
echo "Running unit tests validation..."
python "$SRC_DIR/validators/unit_test_validator.py"

# Generate validation report
echo "Generating validation report..."
python "$SRC_DIR/delivery/formatters.py" --validation-results

echo "Validation process completed successfully!"
