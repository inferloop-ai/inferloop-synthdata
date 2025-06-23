"""
Microsoft Azure deployment provider
"""

from .provider import AzureProvider
from .templates import AzureTemplates

__all__ = ["AzureProvider", "AzureTemplates"]
