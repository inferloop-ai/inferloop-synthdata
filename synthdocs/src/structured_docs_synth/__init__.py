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
    
    async def generate_documents(self, count=10, domain="general", **kwargs):
        """
        Generate synthetic documents with privacy protection.
        
        Args:
            count: Number of documents to generate
            domain: Domain for document generation (general, finance, legal, healthcare)
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated document dictionaries
        """
        try:
            from .generation import create_layout_engine
            from .orchestration import create_orchestrator
            
            # Create generation engine
            layout_engine = create_layout_engine(self.config.generation)
            orchestrator = create_orchestrator(self.config.orchestration)
            
            # Prepare generation parameters
            generation_params = {
                'domain': domain,
                'count': count,
                'privacy_level': kwargs.get('privacy_level', 'medium'),
                'quality_threshold': kwargs.get('quality_threshold', 0.8),
                'formats': kwargs.get('formats', ['json']),
                **kwargs
            }
            
            # Generate documents through orchestrator
            documents = await orchestrator.generate_batch(generation_params)
            
            # Apply privacy protection if enabled
            if kwargs.get('apply_privacy', True):
                from .privacy import create_privacy_engine
                privacy_engine = create_privacy_engine(self.config.privacy)
                documents = await privacy_engine.protect_documents(documents)
            
            return documents
            
        except ImportError as e:
            raise ProcessingError(f"Required modules not available: {e}")
        except Exception as e:
            raise ProcessingError(f"Document generation failed: {e}")
    
    async def export(self, data, format="json", output_path="./output", **kwargs):
        """
        Export processed data in specified format with privacy protection.
        
        Args:
            data: Data to export (documents, annotations, etc.)
            format: Export format (json, coco, yolo, pascal_voc, csv)
            output_path: Output directory path
            **kwargs: Additional export parameters
            
        Returns:
            Export result dictionary
        """
        try:
            from .delivery.export.format_exporters import create_format_exporter
            from pathlib import Path
            
            # Ensure output directory exists
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create format exporter
            exporter = create_format_exporter(
                format, 
                privacy_protection=kwargs.get('privacy_protection', True)
            )
            
            # Prepare export options
            export_options = {
                'copy_images': kwargs.get('copy_images', True),
                'include_metadata': kwargs.get('include_metadata', True),
                'compression': kwargs.get('compression', False),
                **kwargs
            }
            
            # Perform export
            if isinstance(data, dict):
                data = [data]  # Convert single document to list
            
            export_result = await exporter.export(data, str(output_dir), export_options)
            
            return {
                'success': True,
                'format': format,
                'output_path': str(output_dir),
                'files_generated': export_result.get('exported_count', len(data)),
                'export_details': export_result
            }
            
        except ImportError as e:
            raise ProcessingError(f"Export modules not available: {e}")
        except Exception as e:
            raise ProcessingError(f"Export failed: {e}")

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