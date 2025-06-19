#!/usr/bin/env python3
"""
Content Generation Module

Provides comprehensive document generation capabilities including content creation,
layout design, and rendering for synthetic structured documents.
"""

# Import implemented components
from .content import (
    DomainDataGenerator, DomainDataConfig, GeneratedData, DataDomain, GenerationMode,
    EntityGenerator, EntityGeneratorConfig, GeneratedEntity, EntityType,
    CustomFinancialProvider, CustomMedicalProvider,
    create_domain_generator, create_entity_generator,
    create_content_generation_pipeline, create_financial_data_pipeline,
    create_healthcare_data_pipeline, create_legal_data_pipeline
)

from .engines import (
    TemplateEngine, PDFGenerator, DOCXGenerator,
    get_template_engine
)

# Factory functions for generation pipeline
def create_generation_pipeline(**config_kwargs):
    """Create a complete generation pipeline"""
    content_config = config_kwargs.get('content', {})
    engine_config = config_kwargs.get('engines', {})
    
    return {
        'content_pipeline': create_content_generation_pipeline(**content_config),
        'template_engine': get_template_engine(),
        'pdf_generator': PDFGenerator(),
        'docx_generator': DOCXGenerator(),
        'status': 'production_ready'
    }

def create_document_generation_pipeline(document_type: str = 'financial', **config_kwargs):
    """Create a document generation pipeline for specific document types"""
    
    if document_type == 'financial':
        content_pipeline = create_financial_data_pipeline(**config_kwargs.get('content', {}))
    elif document_type == 'healthcare':
        content_pipeline = create_healthcare_data_pipeline(**config_kwargs.get('content', {}))
    elif document_type == 'legal':
        content_pipeline = create_legal_data_pipeline(**config_kwargs.get('content', {}))
    else:
        content_pipeline = create_content_generation_pipeline(**config_kwargs.get('content', {}))
    
    return {
        'content_pipeline': content_pipeline,
        'template_engine': get_template_engine(),
        'pdf_generator': PDFGenerator(**config_kwargs.get('pdf', {})),
        'docx_generator': DOCXGenerator(**config_kwargs.get('docx', {})),
        'document_type': document_type
    }

# Export implemented components
__all__ = [
    # Factory functions
    'create_generation_pipeline',
    'create_document_generation_pipeline',
    
    # Content Generation
    'DomainDataGenerator',
    'DomainDataConfig',
    'GeneratedData',
    'DataDomain',
    'GenerationMode',
    'EntityGenerator',
    'EntityGeneratorConfig', 
    'GeneratedEntity',
    'EntityType',
    'CustomFinancialProvider',
    'CustomMedicalProvider',
    'create_domain_generator',
    'create_entity_generator',
    'create_content_generation_pipeline',
    'create_financial_data_pipeline',
    'create_healthcare_data_pipeline',
    'create_legal_data_pipeline',
    
    # Generation Engines
    'TemplateEngine',
    'PDFGenerator',
    'DOCXGenerator',
    'get_template_engine'
]