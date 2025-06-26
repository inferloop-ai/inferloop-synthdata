"""
Tabular Synthetic Data API - Unified Infrastructure Version

This module imports the unified infrastructure version of the API.
"""

from .app_unified import app

# Export the app for uvicorn
__all__ = ["app"]
