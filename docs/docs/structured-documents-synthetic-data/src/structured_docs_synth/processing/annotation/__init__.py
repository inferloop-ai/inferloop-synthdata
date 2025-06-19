"""\nAnnotation module for document annotation generation\n"""

from .ground_truth_generator import (
    GroundTruthGenerator, GroundTruthConfig, GroundTruthDataset,
    GroundTruthDocument, GroundTruthAnnotation, AnnotationType,
    AnnotationFormat, create_ground_truth_generator
)
from .bounding_box_annotator import (
    BoundingBoxAnnotator, BoundingBoxConfig, BoundingBoxResult,
    BoundingBoxAnnotation, SpatialRelation, create_bounding_box_annotator
)
from .entity_annotator import (
    EntityAnnotator, EntityAnnotatorConfig, EntityAnnotationResult,
    EntityAnnotation, AnnotationQuality, create_entity_annotator
)
from .structure_annotator import (
    StructureAnnotator, StructureAnnotatorConfig, StructureAnnotationResult,
    StructureAnnotation, DocumentStructure, StructureType,
    create_structure_annotator
)

__all__ = [
    # Ground Truth Generation
    'GroundTruthGenerator',
    'GroundTruthConfig',
    'GroundTruthDataset',
    'GroundTruthDocument',
    'GroundTruthAnnotation',
    'AnnotationType',
    'AnnotationFormat',
    'create_ground_truth_generator',
    
    # Bounding Box Annotation
    'BoundingBoxAnnotator',
    'BoundingBoxConfig',
    'BoundingBoxResult',
    'BoundingBoxAnnotation',
    'SpatialRelation',
    'create_bounding_box_annotator',
    
    # Entity Annotation
    'EntityAnnotator',
    'EntityAnnotatorConfig',
    'EntityAnnotationResult',
    'EntityAnnotation',
    'AnnotationQuality',
    'create_entity_annotator',
    
    # Structure Annotation
    'StructureAnnotator',
    'StructureAnnotatorConfig',
    'StructureAnnotationResult',
    'StructureAnnotation',
    'DocumentStructure',
    'StructureType',
    'create_structure_annotator'
]