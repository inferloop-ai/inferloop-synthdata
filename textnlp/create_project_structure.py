#!/usr/bin/env python
"""
Script to create the directory structure and empty files for the inferloop-nlp-synthetic project.
"""

import os
import pathlib
from pathlib import Path


def create_directories_and_files():
    """Create all directories and empty files for the project structure."""
    # Get the root directory (where this script is located)
    root_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Define the directory structure
    directories = [
        "sdk",
        "sdk/validation",
        "templates",
        "api",
        "cli",
        "examples",
        "tests",
        "data",
    ]
    
    # Define files to create
    files = [
        "sdk/__init__.py",
        "sdk/base_generator.py",
        "sdk/llm_gpt2.py",
        "sdk/langchain_template.py",
        "sdk/formatter.py",
        "sdk/validation/__init__.py",
        "sdk/validation/bleu_rouge.py",
        "sdk/validation/gpt4_eval.py",
        "sdk/validation/human_interface.py",
        "templates/feedback_summary.json",
        "templates/support_chat_template.json",
        "api/app.py",
        "api/routes.py",
        "cli/main.py",
        "examples/01_generate_text.ipynb",
        "examples/02_validate_bleu.ipynb",
        "examples/03_format_output.ipynb",
        "tests/test_llm_gpt2.py",
        "tests/test_bleu_rouge.py",
        "tests/test_langchain_template.py",
        "data/sample_prompts.csv",
        "data/sample_outputs.jsonl",
        "data/real_vs_synthetic_pairs.json",
        "Dockerfile",
        "pyproject.toml",
        "README.md",
        ".env.example",
    ]
    
    # Create directories
    print("Creating directories...")
    for directory in directories:
        dir_path = root_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # Create empty files
    print("\nCreating empty files...")
    for file in files:
        file_path = root_dir / file
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Create empty file if it doesn't exist
        if not file_path.exists():
            file_path.touch()
            print(f"Created file: {file_path}")
        else:
            print(f"File already exists: {file_path}")
    
    print("\nProject structure created successfully!")


if __name__ == "__main__":
    create_directories_and_files()
