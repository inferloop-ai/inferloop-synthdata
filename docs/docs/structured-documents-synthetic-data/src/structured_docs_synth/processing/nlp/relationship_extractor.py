#!/usr/bin/env python3
"""
Relationship Extractor for identifying relationships between entities
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
from .ner_processor import Entity, EntityType


class RelationType(Enum):
    """Types of relationships between entities"""
    LOCATED_IN = "located_in"
    WORKS_FOR = "works_for"
    FOUNDED_BY = "founded_by"
    BORN_IN = "born_in"
    MARRIED_TO = "married_to"
    SIMILAR_TO = "similar_to"
    PART_OF = "part_of"
    OWNS = "owns"
    LEADS = "leads"
    COLLABORATED_WITH = "collaborated_with"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    CAUSAL = "causal"
    CUSTOM = "custom"


@dataclass
class Relationship:
    """Relationship between two entities"""
    subject: Entity
    predicate: RelationType
    object: Entity
    confidence: float = 0.0
    context: str = ""
    source_span: Tuple[int, int] = (0, 0)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationshipExtractionResult:
    """Relationship extraction result"""
    text: str
    entities: List[Entity]
    relationships: List[Relationship]
    processing_time: float = 0.0
    extraction_stats: Dict[str, int] = field(default_factory=dict)


class RelationshipExtractorConfig(BaseModel):
    """Relationship extractor configuration"""
    
    # Extraction settings
    confidence_threshold: float = Field(0.6, description="Minimum confidence threshold")
    max_distance: int = Field(100, description="Maximum word distance between entities")
    use_patterns: bool = Field(True, description="Use pattern-based extraction")
    use_dependency_parsing: bool = Field(True, description="Use dependency parsing")
    
    # Model settings
    spacy_model: str = Field("en_core_web_sm", description="SpaCy model for dependency parsing")
    
    # Pattern settings
    custom_patterns: Dict[str, List[str]] = Field(default_factory=dict, description="Custom relationship patterns")


class RelationshipExtractor:
    """
    Relationship extractor for identifying semantic relationships between entities
    """
    
    def __init__(self, config: Optional[RelationshipExtractorConfig] = None):
        self.logger = get_logger(__name__)
        self.config = config or RelationshipExtractorConfig()
        
        # Initialize NLP model
        self.nlp = None
        self._initialize_nlp()
        
        # Initialize patterns
        self.patterns = {}
        self._initialize_patterns()
        
        self.logger.info("Relationship Extractor initialized")
    
    def _initialize_nlp(self):
        """Initialize SpaCy NLP model"""
        try:
            if self.config.use_dependency_parsing:
                import spacy
                try:
                    self.nlp = spacy.load(self.config.spacy_model)
                except OSError:
                    self.logger.warning(f"SpaCy model {self.config.spacy_model} not found")
                    self.nlp = None
        except ImportError:
            self.logger.warning("SpaCy not available for dependency parsing")
            self.nlp = None
    
    def _initialize_patterns(self):
        """Initialize relationship extraction patterns"""
        # Default patterns for common relationships
        default_patterns = {
            RelationType.WORKS_FOR: [
                r"(\w+)\s+works\s+for\s+(\w+)",
                r"(\w+)\s+is\s+employed\s+by\s+(\w+)",
                r"(\w+),\s+CEO\s+of\s+(\w+)"
            ],
            RelationType.LOCATED_IN: [
                r"(\w+)\s+in\s+(\w+)",
                r"(\w+)\s+located\s+in\s+(\w+)",
                r"(\w+),\s+(\w+)"
            ],
            RelationType.FOUNDED_BY: [
                r"(\w+)\s+founded\s+by\s+(\w+)",
                r"(\w+)\s+was\s+established\s+by\s+(\w+)"
            ]
        }
        
        # Compile patterns
        import re
        for rel_type, patterns in default_patterns.items():
            self.patterns[rel_type] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        
        # Add custom patterns
        for rel_type_str, patterns in self.config.custom_patterns.items():
            try:
                rel_type = RelationType(rel_type_str)
                if rel_type not in self.patterns:
                    self.patterns[rel_type] = []
                self.patterns[rel_type].extend([re.compile(p, re.IGNORECASE) for p in patterns])
            except ValueError:
                self.logger.warning(f"Unknown relationship type: {rel_type_str}")
    
    def extract_relationships(self, text: str, entities: List[Entity]) -> RelationshipExtractionResult:
        """Extract relationships from text and entities"""
        start_time = time.time()
        
        try:
            relationships = []
            extraction_stats = {}
            
            # Pattern-based extraction
            if self.config.use_patterns:
                pattern_rels = self._extract_with_patterns(text, entities)
                relationships.extend(pattern_rels)
                extraction_stats['pattern_based'] = len(pattern_rels)
            
            # Dependency parsing extraction
            if self.config.use_dependency_parsing and self.nlp:
                dep_rels = self._extract_with_dependencies(text, entities)
                relationships.extend(dep_rels)
                extraction_stats['dependency_based'] = len(dep_rels)
            
            # Remove duplicates and filter by confidence
            unique_relationships = self._deduplicate_relationships(relationships)
            filtered_relationships = [
                rel for rel in unique_relationships 
                if rel.confidence >= self.config.confidence_threshold
            ]
            
            processing_time = time.time() - start_time
            
            result = RelationshipExtractionResult(
                text=text,
                entities=entities,
                relationships=filtered_relationships,
                processing_time=processing_time,
                extraction_stats=extraction_stats
            )
            
            self.logger.debug(f"Relationship extraction completed in {processing_time:.2f}s, found {len(filtered_relationships)} relationships")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Relationship extraction failed: {e}")
            raise ProcessingError(f"Relationship extraction error: {e}")
    
    def _extract_with_patterns(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships using pattern matching"""
        relationships = []
        
        try:
            for rel_type, patterns in self.patterns.items():
                for pattern in patterns:
                    for match in pattern.finditer(text):
                        # Find entities that match the pattern groups
                        subject_text = match.group(1)
                        object_text = match.group(2)
                        
                        subject_entity = self._find_entity_by_text(entities, subject_text)
                        object_entity = self._find_entity_by_text(entities, object_text)
                        
                        if subject_entity and object_entity:
                            relationship = Relationship(
                                subject=subject_entity,
                                predicate=rel_type,
                                object=object_entity,
                                confidence=0.8,  # High confidence for pattern matches
                                context=match.group(0),
                                source_span=(match.start(), match.end()),
                                metadata={'extraction_method': 'pattern'}
                            )
                            relationships.append(relationship)
        
        except Exception as e:
            self.logger.warning(f"Pattern-based extraction failed: {e}")
        
        return relationships
    
    def _extract_with_dependencies(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships using dependency parsing"""
        relationships = []
        
        try:
            doc = self.nlp(text)
            
            # Look for subject-verb-object patterns
            for token in doc:
                if token.dep_ == "nsubj":  # Nominal subject
                    subject_entity = self._find_entity_at_position(entities, token.idx)
                    
                    # Find the verb
                    verb = token.head
                    
                    # Find objects
                    for child in verb.children:
                        if child.dep_ in ["dobj", "pobj"]:  # Direct/prepositional object
                            object_entity = self._find_entity_at_position(entities, child.idx)
                            
                            if subject_entity and object_entity:
                                # Determine relationship type based on verb
                                rel_type = self._classify_relationship_from_verb(verb.lemma_)
                                
                                if rel_type:
                                    relationship = Relationship(
                                        subject=subject_entity,
                                        predicate=rel_type,
                                        object=object_entity,
                                        confidence=0.7,
                                        context=f"{token.text} {verb.text} {child.text}",
                                        metadata={'extraction_method': 'dependency'}
                                    )
                                    relationships.append(relationship)
        
        except Exception as e:
            self.logger.warning(f"Dependency-based extraction failed: {e}")
        
        return relationships
    
    def _find_entity_by_text(self, entities: List[Entity], text: str) -> Optional[Entity]:
        """Find entity by matching text"""
        text_lower = text.lower().strip()
        for entity in entities:
            if entity.text.lower().strip() == text_lower:
                return entity
            # Also check if entity text contains the search text
            if text_lower in entity.text.lower() or entity.text.lower() in text_lower:
                return entity
        return None
    
    def _find_entity_at_position(self, entities: List[Entity], position: int) -> Optional[Entity]:
        """Find entity at specific text position"""
        for entity in entities:
            if entity.start <= position <= entity.end:
                return entity
        return None
    
    def _classify_relationship_from_verb(self, verb: str) -> Optional[RelationType]:
        """Classify relationship type from verb"""
        verb_mappings = {
            'work': RelationType.WORKS_FOR,
            'employ': RelationType.WORKS_FOR,
            'found': RelationType.FOUNDED_BY,
            'establish': RelationType.FOUNDED_BY,
            'locate': RelationType.LOCATED_IN,
            'live': RelationType.LOCATED_IN,
            'own': RelationType.OWNS,
            'lead': RelationType.LEADS,
            'manage': RelationType.LEADS
        }
        
        return verb_mappings.get(verb.lower())
    
    def _deduplicate_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        """Remove duplicate relationships"""
        seen = set()
        unique = []
        
        for rel in relationships:
            key = (rel.subject.text, rel.predicate.value, rel.object.text)
            if key not in seen:
                seen.add(key)
                unique.append(rel)
            else:
                # Update confidence if this instance has higher confidence
                for existing in unique:
                    existing_key = (existing.subject.text, existing.predicate.value, existing.object.text)
                    if existing_key == key and rel.confidence > existing.confidence:
                        existing.confidence = rel.confidence
                        break
        
        return unique
    
    def get_relationships_by_type(self, result: RelationshipExtractionResult, 
                                 rel_type: RelationType) -> List[Relationship]:
        """Get relationships of specific type"""
        return [rel for rel in result.relationships if rel.predicate == rel_type]
    
    def get_entity_relationships(self, result: RelationshipExtractionResult, 
                               entity: Entity) -> List[Relationship]:
        """Get all relationships involving a specific entity"""
        return [
            rel for rel in result.relationships 
            if rel.subject == entity or rel.object == entity
        ]


# Factory function for easy extractor creation
def create_relationship_extractor(**config_kwargs) -> RelationshipExtractor:
    """Factory function to create relationship extractor"""
    config = RelationshipExtractorConfig(**config_kwargs)
    return RelationshipExtractor(config)