"""\nDocument Processing Pipeline Module\n\nThis module provides comprehensive document processing capabilities including:\n- OCR text extraction with multiple engines\n- NLP analysis for entity recognition and layout understanding\n- Annotation generation for training data creation\n"""

# OCR Processing
from .ocr import (
    OCRPipeline, OCRConfig, OCRResult,
    TesseractEngine, TrOCREngine, CustomOCREngine,
    create_ocr_pipeline
)

# NLP Processing
from .nlp import (
    NERProcessor, LayoutTokenizer, EntityLinker, RelationshipExtractor,
    create_ner_processor, create_layout_tokenizer, create_entity_linker,
    create_relationship_extractor
)

# Annotation Generation
from .annotation import (
    GroundTruthGenerator, BoundingBoxAnnotator, EntityAnnotator, StructureAnnotator,
    create_ground_truth_generator, create_bounding_box_annotator,
    create_entity_annotator, create_structure_annotator
)

__all__ = [
    # OCR Processing
    'OCRPipeline',
    'OCRConfig', 
    'OCRResult',
    'TesseractEngine',
    'TrOCREngine',
    'CustomOCREngine',
    'create_ocr_pipeline',
    
    # NLP Processing
    'NERProcessor',
    'LayoutTokenizer',
    'EntityLinker',
    'RelationshipExtractor',
    'create_ner_processor',
    'create_layout_tokenizer',
    'create_entity_linker',
    'create_relationship_extractor',
    
    # Annotation Generation
    'GroundTruthGenerator',
    'BoundingBoxAnnotator',
    'EntityAnnotator',
    'StructureAnnotator',
    'create_ground_truth_generator',
    'create_bounding_box_annotator',
    'create_entity_annotator',
    'create_structure_annotator'
]