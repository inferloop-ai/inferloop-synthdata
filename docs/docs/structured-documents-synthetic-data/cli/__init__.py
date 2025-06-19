#!/usr/bin/env python3
"""
CLI module for Structured Documents Synthetic Data Generator.

Provides command-line interface for document generation, validation,
export, benchmarking, and deployment operations.
"""

from .commands import (
    generate_command,
    validate_command,
    export_command,
    benchmark_command,
    deploy_command
)

from .utils import (
    OutputFormatter,
    ProgressTracker,
    format_output,
    create_progress_tracker
)

__version__ = "0.1.0"

__all__ = [
    # Commands
    'generate_command',
    'validate_command', 
    'export_command',
    'benchmark_command',
    'deploy_command',
    
    # Utils
    'OutputFormatter',
    'ProgressTracker',
    'format_output',
    'create_progress_tracker'
]