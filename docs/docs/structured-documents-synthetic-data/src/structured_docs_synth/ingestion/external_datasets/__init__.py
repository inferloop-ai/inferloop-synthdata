#!/usr/bin/env python3
"""
External Dataset Adapters Module

Provides domain-specific adapters for integrating external datasets
into the synthetic document generation pipeline.
"""

# Import domain-specific adapters
from .banking_data_adapter import (
    BankingDataAdapter, BankingDataConfig,
    create_banking_adapter
)

from .document_datasets_adapter import (
    DocumentDatasetsAdapter, DocumentDatasetConfig,
    create_document_adapter
)

from .government_data_adapter import (
    GovernmentDataAdapter, GovernmentDataConfig,
    create_government_adapter
)

from .healthcare_data_adapter import (
    HealthcareDataAdapter, HealthcareDataConfig,
    create_healthcare_adapter
)

from .legal_data_adapter import (
    LegalDataAdapter, LegalDataConfig,
    create_legal_adapter
)

# Factory function for all adapters
def create_all_adapters(**config_kwargs):
    """Create all domain-specific data adapters"""
    return {
        'banking': create_banking_adapter(**config_kwargs.get('banking', {})),
        'documents': create_document_adapter(**config_kwargs.get('documents', {})),
        'government': create_government_adapter(**config_kwargs.get('government', {})),
        'healthcare': create_healthcare_adapter(**config_kwargs.get('healthcare', {})),
        'legal': create_legal_adapter(**config_kwargs.get('legal', {}))
    }

# Export all components
__all__ = [
    # Banking
    'BankingDataAdapter',
    'BankingDataConfig',
    'create_banking_adapter',
    
    # Documents
    'DocumentDatasetsAdapter',
    'DocumentDatasetConfig',
    'create_document_adapter',
    
    # Government
    'GovernmentDataAdapter',
    'GovernmentDataConfig',
    'create_government_adapter',
    
    # Healthcare
    'HealthcareDataAdapter',
    'HealthcareDataConfig',
    'create_healthcare_adapter',
    
    # Legal
    'LegalDataAdapter',
    'LegalDataConfig',
    'create_legal_adapter',
    
    # Factory functions
    'create_all_adapters'
]