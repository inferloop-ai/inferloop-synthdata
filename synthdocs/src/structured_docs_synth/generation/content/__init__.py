#!/usr/bin/env python3
"""
Content Generation Module

Provides content generation capabilities for creating domain-specific
synthetic data and entities for structured document generation.
"""

# Import main components
from .domain_data_generator import (
    DomainDataGenerator, DomainDataConfig, GeneratedData,
    DataDomain, GenerationMode,
    create_domain_generator
)

from .entity_generator import (
    EntityGenerator, EntityGeneratorConfig, GeneratedEntity,
    EntityType, CustomFinancialProvider, CustomMedicalProvider,
    create_entity_generator
)

# Factory functions for content generation
def create_content_generation_pipeline(**config_kwargs):
    """Create a complete content generation pipeline"""
    domain_config = config_kwargs.get('domain_generator', {})
    entity_config = config_kwargs.get('entity_generator', {})
    
    return {
        'domain_generator': create_domain_generator(**domain_config),
        'entity_generator': create_entity_generator(**entity_config)
    }

def create_financial_data_pipeline(**config_kwargs):
    """Create a financial data generation pipeline"""
    from .domain_data_generator import DataDomain
    
    domain_config = config_kwargs.get('domain_config', {})
    domain_config['domain'] = DataDomain.FINANCIAL
    
    entity_config = config_kwargs.get('entity_config', {})
    
    return {
        'domain_generator': create_domain_generator(**domain_config),
        'entity_generator': create_entity_generator(**entity_config)
    }

def create_healthcare_data_pipeline(**config_kwargs):
    """Create a healthcare data generation pipeline"""
    from .domain_data_generator import DataDomain
    
    domain_config = config_kwargs.get('domain_config', {})
    domain_config['domain'] = DataDomain.HEALTHCARE
    
    entity_config = config_kwargs.get('entity_config', {})
    
    return {
        'domain_generator': create_domain_generator(**domain_config),
        'entity_generator': create_entity_generator(**entity_config)
    }

def create_legal_data_pipeline(**config_kwargs):
    """Create a legal data generation pipeline"""
    from .domain_data_generator import DataDomain
    
    domain_config = config_kwargs.get('domain_config', {})
    domain_config['domain'] = DataDomain.LEGAL
    
    entity_config = config_kwargs.get('entity_config', {})
    
    return {
        'domain_generator': create_domain_generator(**domain_config),
        'entity_generator': create_entity_generator(**entity_config)
    }

# Export all components
__all__ = [
    # Domain Data Generator
    'DomainDataGenerator',
    'DomainDataConfig',
    'GeneratedData',
    'DataDomain',
    'GenerationMode',
    'create_domain_generator',
    
    # Entity Generator
    'EntityGenerator',
    'EntityGeneratorConfig',
    'GeneratedEntity',
    'EntityType',
    'CustomFinancialProvider',
    'CustomMedicalProvider',
    'create_entity_generator',
    
    # Factory functions
    'create_content_generation_pipeline',
    'create_financial_data_pipeline',
    'create_healthcare_data_pipeline',
    'create_legal_data_pipeline'
]