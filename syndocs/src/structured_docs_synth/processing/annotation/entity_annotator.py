#!/usr/bin/env python3
"""
Entity Annotator for creating entity-level annotations
"""

import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field

from ...core.config import get_config
from ...core.logging import get_logger
from ...core.exceptions import ProcessingError, ValidationError
from ..nlp.ner_processor import Entity, EntityType


class AnnotationQuality(Enum):
    """Annotation quality levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERIFIED = "verified"


@dataclass
class EntityAnnotation:
    """Entity annotation with quality metrics"""
    id: str
    entity: Entity
    quality: AnnotationQuality
    verification_score: float = 0.0
    context_window: str = ""
    bbox: Optional[Tuple[int, int, int, int]] = None
    page_num: int = 0
    attributes: Dict[str, Any] = field(default_factory=dict)
    linked_entity: Optional[Any] = None  # From entity linking


@dataclass
class EntityAnnotationResult:
    """Entity annotation result"""
    annotations: List[EntityAnnotation]
    quality_distribution: Dict[str, int]
    entity_statistics: Dict[str, int]
    processing_time: float = 0.0


class EntityAnnotatorConfig(BaseModel):
    """Entity annotator configuration"""
    
    # Quality assessment
    confidence_threshold: float = Field(0.7, description="Minimum confidence for high quality")
    context_window_size: int = Field(50, description="Context window size in characters")
    enable_verification: bool = Field(True, description="Enable annotation verification")
    
    # Spatial mapping
    map_to_bboxes: bool = Field(True, description="Map entities to bounding boxes")
    bbox_search_tolerance: int = Field(10, description="Tolerance for bbox mapping")
    
    # Filtering
    min_entity_length: int = Field(2, description="Minimum entity text length")
    exclude_common_words: bool = Field(True, description="Exclude common words")
    
    # Export settings
    include_context: bool = Field(True, description="Include context in annotations")
    include_quality_scores: bool = Field(True, description="Include quality metrics")


class EntityAnnotator:
    """
    Entity Annotator for creating high-quality entity annotations
    """
    
    def __init__(self, config: Optional[EntityAnnotatorConfig] = None):
        self.logger = get_logger(__name__)
        self.config = config or EntityAnnotatorConfig()
        
        # Common words to exclude (simplified list)
        self.common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'this', 'that', 'these', 'those', 'is', 'are', 
            'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did'
        }
        
        self.logger.info("Entity Annotator initialized")
    
    def annotate_entities(self, text: str, entities: List[Entity], 
                         ocr_result=None) -> EntityAnnotationResult:
        """Create entity annotations with quality assessment"""
        start_time = time.time()
        
        try:
            annotations = []
            
            for i, entity in enumerate(entities):
                # Filter entities
                if not self._should_include_entity(entity):
                    continue
                
                # Create annotation
                annotation = self._create_annotation(entity, text, i)
                
                # Map to bounding box if OCR result available
                if self.config.map_to_bboxes and ocr_result:
                    bbox = self._map_entity_to_bbox(entity, ocr_result)
                    annotation.bbox = bbox
                
                # Assess quality
                annotation.quality = self._assess_quality(annotation, text)
                
                # Verify annotation if enabled
                if self.config.enable_verification:
                    annotation.verification_score = self._verify_annotation(annotation, text)
                
                annotations.append(annotation)
            
            # Calculate statistics
            quality_dist = self._calculate_quality_distribution(annotations)
            entity_stats = self._calculate_entity_statistics(annotations)
            
            processing_time = time.time() - start_time
            
            result = EntityAnnotationResult(
                annotations=annotations,
                quality_distribution=quality_dist,
                entity_statistics=entity_stats,
                processing_time=processing_time
            )
            
            self.logger.debug(f"Entity annotation completed in {processing_time:.2f}s, created {len(annotations)} annotations")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Entity annotation failed: {e}")
            raise ProcessingError(f"Entity annotation error: {e}")
    
    def _should_include_entity(self, entity: Entity) -> bool:
        """Check if entity should be included in annotations"""
        # Check minimum length
        if len(entity.text.strip()) < self.config.min_entity_length:
            return False
        
        # Check common words
        if (self.config.exclude_common_words and 
            entity.text.lower().strip() in self.common_words):
            return False
        
        # Check if text is only punctuation or numbers
        if entity.text.strip().isdigit() or not entity.text.strip().isalnum():
            return False
        
        return True
    
    def _create_annotation(self, entity: Entity, text: str, index: int) -> EntityAnnotation:
        """Create entity annotation"""
        # Extract context window
        context = ""
        if self.config.include_context:
            context = self._extract_context(entity, text)
        
        annotation = EntityAnnotation(
            id=f"entity_{index}",
            entity=entity,
            quality=AnnotationQuality.MEDIUM,  # Default, will be updated
            context_window=context,
            attributes={
                'entity_type': entity.label.value,
                'original_confidence': entity.confidence,
                'source': entity.source,
                'character_span': (entity.start, entity.end)
            }
        )
        
        return annotation
    
    def _extract_context(self, entity: Entity, text: str) -> str:
        """Extract context window around entity"""
        start = max(0, entity.start - self.config.context_window_size)
        end = min(len(text), entity.end + self.config.context_window_size)
        
        context = text[start:end]
        
        # Mark the entity in context
        entity_start_in_context = entity.start - start
        entity_end_in_context = entity.end - start
        
        if 0 <= entity_start_in_context < len(context):
            context = (context[:entity_start_in_context] + 
                      f"**{entity.text}**" + 
                      context[entity_end_in_context:])
        
        return context.strip()
    
    def _map_entity_to_bbox(self, entity: Entity, ocr_result) -> Optional[Tuple[int, int, int, int]]:
        """Map entity to bounding box from OCR result"""
        try:
            # Find words that overlap with entity span
            overlapping_words = []
            
            for page in ocr_result.pages:
                for word in page.words:
                    # Simple position-based mapping (would need more sophisticated matching in practice)
                    if self._text_positions_overlap(entity, word.text, entity.start, entity.end):
                        overlapping_words.append(word)
            
            if overlapping_words:
                # Calculate combined bounding box
                min_x = min(word.bbox.x for word in overlapping_words)
                min_y = min(word.bbox.y for word in overlapping_words)
                max_x = max(word.bbox.x + word.bbox.width for word in overlapping_words)
                max_y = max(word.bbox.y + word.bbox.height for word in overlapping_words)
                
                return (min_x, min_y, max_x, max_y)
        
        except Exception as e:
            self.logger.debug(f"Failed to map entity to bbox: {e}")
        
        return None
    
    def _text_positions_overlap(self, entity: Entity, word_text: str, 
                               start: int, end: int) -> bool:
        """Check if entity and word positions overlap (simplified)"""
        # This is a simplified implementation
        # In practice, you'd need more sophisticated text alignment
        return entity.text.lower() in word_text.lower() or word_text.lower() in entity.text.lower()
    
    def _assess_quality(self, annotation: EntityAnnotation, text: str) -> AnnotationQuality:
        """Assess annotation quality"""
        score = 0
        
        # Confidence score factor
        if annotation.entity.confidence >= 0.9:
            score += 3
        elif annotation.entity.confidence >= self.config.confidence_threshold:
            score += 2
        else:
            score += 1
        
        # Entity type factor
        high_confidence_types = {
            EntityType.EMAIL, EntityType.PHONE, EntityType.URL, 
            EntityType.IP_ADDRESS, EntityType.CREDIT_CARD
        }
        if annotation.entity.label in high_confidence_types:
            score += 2
        
        # Text length factor
        if len(annotation.entity.text) >= 10:
            score += 1
        
        # Context quality factor
        if annotation.context_window and len(annotation.context_window) > 20:
            score += 1
        
        # Bounding box factor
        if annotation.bbox:
            score += 1
        
        # Determine quality level
        if score >= 7:
            return AnnotationQuality.HIGH
        elif score >= 5:
            return AnnotationQuality.MEDIUM
        else:
            return AnnotationQuality.LOW
    
    def _verify_annotation(self, annotation: EntityAnnotation, text: str) -> float:
        """Verify annotation quality"""
        verification_score = 0.0
        checks = 0
        
        # Check 1: Entity text consistency
        entity_text = annotation.entity.text
        if entity_text in text:
            verification_score += 1.0
        checks += 1
        
        # Check 2: Position consistency
        try:
            extracted_text = text[annotation.entity.start:annotation.entity.end]
            if extracted_text.strip() == entity_text.strip():
                verification_score += 1.0
        except IndexError:
            pass
        checks += 1
        
        # Check 3: Entity type plausibility
        if self._is_plausible_entity_type(annotation.entity):
            verification_score += 1.0
        checks += 1
        
        # Check 4: Context relevance
        if annotation.context_window:
            if entity_text.lower() in annotation.context_window.lower():
                verification_score += 1.0
        checks += 1
        
        return verification_score / checks if checks > 0 else 0.0
    
    def _is_plausible_entity_type(self, entity: Entity) -> bool:
        """Check if entity type is plausible for the text"""
        text = entity.text.lower()
        
        # Simple plausibility checks
        if entity.label == EntityType.EMAIL:
            return '@' in text and '.' in text
        elif entity.label == EntityType.PHONE:
            return any(char.isdigit() for char in text)
        elif entity.label == EntityType.URL:
            return 'http' in text or 'www' in text or '.com' in text
        elif entity.label == EntityType.PERSON:
            return text.istitle() and len(text.split()) <= 4
        elif entity.label == EntityType.ORGANIZATION:
            return len(text) > 2 and not text.islower()
        
        return True  # Default to plausible
    
    def _calculate_quality_distribution(self, annotations: List[EntityAnnotation]) -> Dict[str, int]:
        """Calculate quality distribution"""
        distribution = {quality.value: 0 for quality in AnnotationQuality}
        
        for annotation in annotations:
            distribution[annotation.quality.value] += 1
        
        return distribution
    
    def _calculate_entity_statistics(self, annotations: List[EntityAnnotation]) -> Dict[str, int]:
        """Calculate entity type statistics"""
        stats = {}
        
        for annotation in annotations:
            entity_type = annotation.entity.label.value
            if entity_type not in stats:
                stats[entity_type] = 0
            stats[entity_type] += 1
        
        return stats
    
    def filter_by_quality(self, result: EntityAnnotationResult, 
                         min_quality: AnnotationQuality) -> List[EntityAnnotation]:
        """Filter annotations by minimum quality"""
        quality_order = {
            AnnotationQuality.LOW: 0,
            AnnotationQuality.MEDIUM: 1,
            AnnotationQuality.HIGH: 2,
            AnnotationQuality.VERIFIED: 3
        }
        
        min_level = quality_order[min_quality]
        
        return [
            ann for ann in result.annotations 
            if quality_order[ann.quality] >= min_level
        ]
    
    def get_annotations_by_type(self, result: EntityAnnotationResult, 
                               entity_type: EntityType) -> List[EntityAnnotation]:
        """Get annotations by entity type"""
        return [
            ann for ann in result.annotations 
            if ann.entity.label == entity_type
        ]
    
    def export_annotations(self, result: EntityAnnotationResult, 
                          format: str = "json") -> Dict[str, Any]:
        """Export annotations in specified format"""
        export_data = {
            'total_annotations': len(result.annotations),
            'quality_distribution': result.quality_distribution,
            'entity_statistics': result.entity_statistics,
            'processing_time': result.processing_time,
            'annotations': []
        }
        
        for annotation in result.annotations:
            ann_data = {
                'id': annotation.id,
                'text': annotation.entity.text,
                'label': annotation.entity.label.value,
                'start': annotation.entity.start,
                'end': annotation.entity.end,
                'quality': annotation.quality.value
            }
            
            if self.config.include_quality_scores:
                ann_data['confidence'] = annotation.entity.confidence
                ann_data['verification_score'] = annotation.verification_score
            
            if self.config.include_context:
                ann_data['context'] = annotation.context_window
            
            if annotation.bbox:
                ann_data['bbox'] = annotation.bbox
            
            export_data['annotations'].append(ann_data)
        
        return export_data


# Factory function
def create_entity_annotator(**config_kwargs) -> EntityAnnotator:
    """Factory function to create entity annotator"""
    config = EntityAnnotatorConfig(**config_kwargs)
    return EntityAnnotator(config)