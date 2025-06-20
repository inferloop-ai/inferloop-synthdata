"""
API delivery layer for Structured Documents Synthetic Data Generator
"""

from .rest_api import app, create_app

__all__ = [
    'app',
    'create_app'
]