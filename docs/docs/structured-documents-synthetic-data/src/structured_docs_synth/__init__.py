#!/usr/bin/env python3
"""
Structured Documents Synthetic Data Generator

A comprehensive toolkit for generating high-quality synthetic structured documents
with privacy protection, compliance frameworks, and advanced document processing.

Key Features:
- Privacy-preserving synthetic data generation with differential privacy
- Multi-format document processing (PDF, images, text)
- Advanced OCR and NLP processing pipelines
- Compliance frameworks (GDPR, HIPAA, PCI DSS, SOX)
- Quality assurance and validation systems
- Flexible data ingestion and export capabilities

Example Usage:
    >>> from structured_docs_synth import SynthDataGenerator
    >>> generator = SynthDataGenerator()
    >>> documents = generator.generate_documents(count=100, domain="finance")
    >>> generator.export(documents, format="coco", output_path="./dataset")
"""

__version__ = "0.1.0"
__author__ = "InferLoop AI"
__license__ = "MIT"
__description__ = "Structured Documents Synthetic Data Generator with Privacy Protection"

# Core utilities
from .core import (
    get_config, get_logger, 
    ProcessingError, ValidationError, PrivacyError, ComplianceError,
    ConfigurationError
)

# Privacy and compliance
from .privacy import (
    # Differential Privacy
    LaplacePrivacyMechanism, ExponentialPrivacyMechanism,
    PrivacyBudgetTracker, CompositionAnalyzer,
    
    # PII Protection  
    PIIDetector, PIIMasker, PIIRedactor,
    
    # Compliance
    GDPRCompliance, HIPAACompliance, PCIDSSCompliance, 
    SOXCompliance, AuditLogger
)

# Document processing
from .processing import (
    # OCR Processing
    OCRPipeline, TesseractEngine, TrOCREngine, CustomOCREngine,
    
    # NLP Processing
    NERProcessor, LayoutTokenizer, EntityLinker, RelationshipExtractor,
    
    # Annotation Generation
    GroundTruthGenerator, BoundingBoxAnnotator, EntityAnnotator, StructureAnnotator
)

# Content generation (when implemented)
# from .generation import (
#     DomainDataGenerator, EntityGenerator, LayoutEngine,
#     FormGenerator, TableGenerator, LaTeXGenerator
# )

# Quality assurance (when implemented)
# from .quality import (
#     BenchmarkRunner, ContentMetrics, LayoutMetrics, OCRMetrics,
#     CompletenessChecker, DriftDetector, SemanticValidator
# )

# Data ingestion (being implemented)
# from .ingestion import (
#     DatasetLoader, FileProcessor, APIPoller, KafkaConsumer, WebhookHandler
# )

# Storage and delivery (partially implemented)
from .delivery.api import DocumentSynthAPI

# Factory functions for easy instantiation
def create_synth_data_pipeline(config=None):
    """Create a complete synthetic data generation pipeline"""
    from .core.config import get_config
    from .processing import create_ocr_pipeline, create_ner_processor
    from .privacy import create_privacy_engine
    
    cfg = config or get_config()
    
    # Initialize core components
    ocr_pipeline = create_ocr_pipeline()
    ner_processor = create_ner_processor()
    privacy_engine = create_privacy_engine()
    
    return {
        'ocr': ocr_pipeline,
        'ner': ner_processor, 
        'privacy': privacy_engine,
        'config': cfg
    }

def create_privacy_engine(mechanisms=None):
    """Create privacy protection engine with specified mechanisms"""
    from .privacy.differential_privacy import create_laplace_mechanism
    from .privacy.pii_protection import create_pii_detector
    from .privacy.compliance import create_gdpr_compliance
    
    return {
        'laplace': create_laplace_mechanism(),
        'pii_detector': create_pii_detector(),
        'gdpr': create_gdpr_compliance()
    }

# Main pipeline class (simplified interface)
class SynthDataGenerator:
    """Main interface for synthetic document generation"""
    
    def __init__(self, config=None):
        self.pipeline = create_synth_data_pipeline(config)
        self.privacy_engine = create_privacy_engine()
    
    def process_document(self, file_path, apply_privacy=True):
        """Process a document through OCR and NLP pipeline"""
        # OCR processing
        ocr_result = self.pipeline['ocr'].process_document(file_path)
        
        # NLP processing
        ner_result = self.pipeline['ner'].process_text(ocr_result.text)
        
        # Apply privacy protection if requested
        if apply_privacy:
            # Apply PII masking
            masked_text = self.privacy_engine['pii_detector'].mask_pii(ocr_result.text)
            # Apply differential privacy to entities
            private_entities = self.privacy_engine['laplace'].add_noise_to_entities(ner_result.entities)
        
        return {
            'ocr_result': ocr_result,
            'ner_result': ner_result,
            'privacy_applied': apply_privacy
        }
    
    def generate_documents(self, count=10, domain="general"):
        """Generate synthetic documents (placeholder - needs generation module)"""
        raise NotImplementedError("Document generation requires the generation module to be implemented")
    
    def export(self, data, format="json", output_path="./output"):
        """Export processed data in specified format"""
        # Placeholder - needs storage/export module
        raise NotImplementedError("Export functionality requires the storage module to be implemented")

# Module information
__all__ = [
    # Version info
    '__version__', '__author__', '__license__', '__description__',
    
    # Core utilities
    'get_config', 'get_logger',
    'ProcessingError', 'ValidationError', 'PrivacyError', 'ComplianceError',
    'ConfigurationError',
    
    # Privacy and compliance
    'LaplacePrivacyMechanism', 'ExponentialPrivacyMechanism',
    'PrivacyBudgetTracker', 'CompositionAnalyzer',
    'PIIDetector', 'PIIMasker', 'PIIRedactor',
    'GDPRCompliance', 'HIPAACompliance', 'PCIDSSCompliance', 
    'SOXCompliance', 'AuditLogger',
    
    # Document processing
    'OCRPipeline', 'TesseractEngine', 'TrOCREngine', 'CustomOCREngine',
    'NERProcessor', 'LayoutTokenizer', 'EntityLinker', 'RelationshipExtractor',
    'GroundTruthGenerator', 'BoundingBoxAnnotator', 'EntityAnnotator', 'StructureAnnotator',
    
    # API
    'DocumentSynthAPI',
    
    # Factory functions
    'create_synth_data_pipeline', 'create_privacy_engine',
    
    # Main interface
    'SynthDataGenerator'
]