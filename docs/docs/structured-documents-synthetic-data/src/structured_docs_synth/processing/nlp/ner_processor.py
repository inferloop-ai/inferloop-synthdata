#!/usr/bin/env python3
"""
Named Entity Recognition (NER) Processor for document analysis
"""

import logging
import time
import re
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

import spacy
from spacy import displacy
from spacy.tokens import Doc, Span
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from pydantic import BaseModel, Field

from ...core.config import get_config
from ...core.logging import get_logger
from ...core.exceptions import ProcessingError, ValidationError


class EntityType(Enum):
    """Standard named entity types"""
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "LOC"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    WORK_OF_ART = "WORK_OF_ART"
    LAW = "LAW"
    LANGUAGE = "LANGUAGE"
    
    # Document-specific entities
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    URL = "URL"
    IP_ADDRESS = "IP"
    CREDIT_CARD = "CREDIT_CARD"
    SSN = "SSN"
    
    # Financial entities
    ACCOUNT_NUMBER = "ACCOUNT_NUMBER"
    ROUTING_NUMBER = "ROUTING_NUMBER"
    TICKER_SYMBOL = "TICKER"
    
    # Medical entities
    DRUG = "DRUG"
    DISEASE = "DISEASE"
    MEDICAL_PROCEDURE = "MEDICAL_PROCEDURE"
    
    # Legal entities
    CASE_NUMBER = "CASE_NUMBER"
    COURT = "COURT"
    STATUTE = "STATUTE"


@dataclass
class Entity:
    """Named entity with metadata"""
    text: str
    label: EntityType
    start: int
    end: int
    confidence: float = 0.0
    source: str = "unknown"  # Which model/method detected this entity
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'label': self.label.value,
            'start': self.start,
            'end': self.end,
            'confidence': self.confidence,
            'source': self.source,
            'metadata': self.metadata
        }


@dataclass 
class NERResult:
    """NER processing result"""
    text: str
    entities: List[Entity]
    processing_time: float = 0.0
    model_info: Dict[str, Any] = field(default_factory=dict)
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get entities of specific type"""
        return [e for e in self.entities if e.label == entity_type]
    
    def get_unique_entities(self) -> List[Entity]:
        """Get unique entities (remove duplicates)"""
        seen = set()
        unique = []
        for entity in self.entities:
            key = (entity.text.lower(), entity.label)
            if key not in seen:
                seen.add(key)
                unique.append(entity)
        return unique


class NERConfig(BaseModel):
    """NER processor configuration"""
    
    # Model configuration
    spacy_model: str = Field("en_core_web_sm", description="SpaCy model name")
    transformers_model: str = Field("dbmdz/bert-large-cased-finetuned-conll03-english", description="Transformers NER model")
    use_spacy: bool = Field(True, description="Use SpaCy for NER")
    use_transformers: bool = Field(False, description="Use Transformers for NER")
    use_regex: bool = Field(True, description="Use regex patterns for entity detection")
    
    # Processing options
    enable_custom_entities: bool = Field(True, description="Enable detection of custom entity types")
    merge_overlapping: bool = Field(True, description="Merge overlapping entities")
    confidence_threshold: float = Field(0.5, description="Minimum confidence threshold")
    
    # Device configuration
    device: str = Field("auto", description="Device for transformers model")
    
    # Custom patterns
    custom_patterns: Dict[str, List[str]] = Field(default_factory=dict, description="Custom regex patterns")


class NERProcessor:
    """
    Named Entity Recognition processor with multiple backends
    """
    
    def __init__(self, config: Optional[NERConfig] = None):
        self.logger = get_logger(__name__)
        self.config = config or NERConfig()
        
        # Initialize models
        self.spacy_nlp = None
        self.transformers_pipeline = None
        self.regex_patterns = {}
        
        self._initialize_models()
        self._compile_regex_patterns()
        
        self.logger.info("NER Processor initialized successfully")
    
    def _initialize_models(self):
        """Initialize NER models"""
        try:
            # Initialize SpaCy model
            if self.config.use_spacy:
                try:
                    self.spacy_nlp = spacy.load(self.config.spacy_model)
                    self.logger.info(f"Loaded SpaCy model: {self.config.spacy_model}")
                except OSError:
                    self.logger.warning(f"SpaCy model {self.config.spacy_model} not found, downloading...")
                    spacy.cli.download(self.config.spacy_model)
                    self.spacy_nlp = spacy.load(self.config.spacy_model)
            
            # Initialize Transformers model
            if self.config.use_transformers:
                device = self._get_device()
                self.transformers_pipeline = pipeline(
                    "ner",
                    model=self.config.transformers_model,
                    tokenizer=self.config.transformers_model,
                    aggregation_strategy="simple",
                    device=0 if device == "cuda" else -1
                )
                self.logger.info(f"Loaded Transformers model: {self.config.transformers_model}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize NER models: {e}")
            raise ProcessingError(f"NER model initialization failed: {e}")
    
    def _get_device(self):
        """Get device for transformers model"""
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device
    
    def _compile_regex_patterns(self):
        """Compile regex patterns for entity detection"""
        # Default patterns
        default_patterns = {
            EntityType.EMAIL: [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            EntityType.PHONE: [
                r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
            ],
            EntityType.URL: [
                r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
                r'www\.(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'
            ],
            EntityType.IP_ADDRESS: [
                r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
            ],
            EntityType.CREDIT_CARD: [
                r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'
            ],
            EntityType.SSN: [
                r'\b\d{3}-\d{2}-\d{4}\b',
                r'\b\d{3}\s\d{2}\s\d{4}\b'
            ],
            EntityType.ACCOUNT_NUMBER: [
                r'\b\d{8,17}\b'  # Generic account number pattern
            ],
            EntityType.TICKER_SYMBOL: [
                r'\b[A-Z]{1,5}\b'  # Stock ticker symbols
            ]
        }
        
        # Compile patterns
        for entity_type, patterns in default_patterns.items():
            self.regex_patterns[entity_type] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        
        # Add custom patterns
        for entity_type_str, patterns in self.config.custom_patterns.items():
            try:
                entity_type = EntityType(entity_type_str)
                if entity_type not in self.regex_patterns:
                    self.regex_patterns[entity_type] = []
                self.regex_patterns[entity_type].extend([re.compile(pattern, re.IGNORECASE) for pattern in patterns])
            except ValueError:
                self.logger.warning(f"Unknown entity type in custom patterns: {entity_type_str}")
    
    def process_text(self, text: str) -> NERResult:
        """Process text and extract named entities"""
        start_time = time.time()
        
        try:
            all_entities = []
            model_info = {}
            
            # Process with SpaCy
            if self.config.use_spacy and self.spacy_nlp:
                spacy_entities = self._process_with_spacy(text)
                all_entities.extend(spacy_entities)
                model_info['spacy'] = {
                    'model': self.config.spacy_model,
                    'entities_found': len(spacy_entities)
                }
            
            # Process with Transformers
            if self.config.use_transformers and self.transformers_pipeline:
                transformer_entities = self._process_with_transformers(text)
                all_entities.extend(transformer_entities)
                model_info['transformers'] = {
                    'model': self.config.transformers_model,
                    'entities_found': len(transformer_entities)
                }
            
            # Process with regex patterns
            if self.config.use_regex:
                regex_entities = self._process_with_regex(text)
                all_entities.extend(regex_entities)
                model_info['regex'] = {
                    'patterns_used': len(self.regex_patterns),
                    'entities_found': len(regex_entities)
                }
            
            # Filter by confidence threshold
            filtered_entities = [
                e for e in all_entities 
                if e.confidence >= self.config.confidence_threshold
            ]
            
            # Merge overlapping entities if enabled
            if self.config.merge_overlapping:
                filtered_entities = self._merge_overlapping_entities(filtered_entities)
            
            # Sort entities by start position
            filtered_entities.sort(key=lambda x: x.start)
            
            processing_time = time.time() - start_time
            
            result = NERResult(
                text=text,
                entities=filtered_entities,
                processing_time=processing_time,
                model_info=model_info
            )
            
            self.logger.debug(f"NER processing completed in {processing_time:.2f}s, found {len(filtered_entities)} entities")
            
            return result
            
        except Exception as e:
            self.logger.error(f"NER processing failed: {e}")
            raise ProcessingError(f"NER processing error: {e}")
    
    def _process_with_spacy(self, text: str) -> List[Entity]:
        """Process text with SpaCy NER"""
        entities = []
        
        try:
            doc = self.spacy_nlp(text)
            
            for ent in doc.ents:
                # Map SpaCy labels to our EntityType enum
                entity_type = self._map_spacy_label(ent.label_)
                if entity_type:
                    entity = Entity(
                        text=ent.text,
                        label=entity_type,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=1.0,  # SpaCy doesn't provide confidence scores
                        source="spacy",
                        metadata={
                            'original_label': ent.label_,
                            'lemma': ent.lemma_ if hasattr(ent, 'lemma_') else None
                        }
                    )
                    entities.append(entity)
            
        except Exception as e:
            self.logger.error(f"SpaCy processing failed: {e}")
        
        return entities
    
    def _process_with_transformers(self, text: str) -> List[Entity]:
        """Process text with Transformers NER"""
        entities = []
        
        try:
            results = self.transformers_pipeline(text)
            
            for result in results:
                # Map transformer labels to our EntityType enum
                entity_type = self._map_transformer_label(result['entity_group'])
                if entity_type:
                    entity = Entity(
                        text=result['word'],
                        label=entity_type,
                        start=result['start'],
                        end=result['end'],
                        confidence=result['score'],
                        source="transformers",
                        metadata={
                            'original_label': result['entity_group']
                        }
                    )
                    entities.append(entity)
            
        except Exception as e:
            self.logger.error(f"Transformers processing failed: {e}")
        
        return entities
    
    def _process_with_regex(self, text: str) -> List[Entity]:
        """Process text with regex patterns"""
        entities = []
        
        try:
            for entity_type, patterns in self.regex_patterns.items():
                for pattern in patterns:
                    for match in pattern.finditer(text):
                        entity = Entity(
                            text=match.group(),
                            label=entity_type,
                            start=match.start(),
                            end=match.end(),
                            confidence=0.9,  # High confidence for regex matches
                            source="regex",
                            metadata={
                                'pattern': pattern.pattern
                            }
                        )
                        entities.append(entity)
        
        except Exception as e:
            self.logger.error(f"Regex processing failed: {e}")
        
        return entities
    
    def _map_spacy_label(self, label: str) -> Optional[EntityType]:
        """Map SpaCy entity labels to our EntityType enum"""
        mapping = {
            'PERSON': EntityType.PERSON,
            'ORG': EntityType.ORGANIZATION,
            'GPE': EntityType.LOCATION,
            'LOC': EntityType.LOCATION,
            'DATE': EntityType.DATE,
            'TIME': EntityType.TIME,
            'MONEY': EntityType.MONEY,
            'PERCENT': EntityType.PERCENT,
            'PRODUCT': EntityType.PRODUCT,
            'EVENT': EntityType.EVENT,
            'WORK_OF_ART': EntityType.WORK_OF_ART,
            'LAW': EntityType.LAW,
            'LANGUAGE': EntityType.LANGUAGE
        }
        
        return mapping.get(label)
    
    def _map_transformer_label(self, label: str) -> Optional[EntityType]:
        """Map transformer entity labels to our EntityType enum"""
        # Common transformer labels (BERT-based models)
        mapping = {
            'PER': EntityType.PERSON,
            'PERSON': EntityType.PERSON,
            'ORG': EntityType.ORGANIZATION,
            'LOC': EntityType.LOCATION,
            'MISC': EntityType.PRODUCT  # Miscellaneous often includes products
        }
        
        return mapping.get(label)
    
    def _merge_overlapping_entities(self, entities: List[Entity]) -> List[Entity]:
        """Merge overlapping entities, keeping the one with highest confidence"""
        if not entities:
            return entities
        
        # Sort by start position
        sorted_entities = sorted(entities, key=lambda x: x.start)
        merged = [sorted_entities[0]]
        
        for current in sorted_entities[1:]:
            last = merged[-1]
            
            # Check for overlap
            if current.start < last.end:
                # Overlapping entities - keep the one with higher confidence
                if current.confidence > last.confidence:
                    merged[-1] = current
                # If same confidence, keep the longer entity
                elif current.confidence == last.confidence:
                    if (current.end - current.start) > (last.end - last.start):
                        merged[-1] = current
            else:
                merged.append(current)
        
        return merged
    
    def add_custom_pattern(self, entity_type: EntityType, pattern: str):
        """Add a custom regex pattern for entity detection"""
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            if entity_type not in self.regex_patterns:
                self.regex_patterns[entity_type] = []
            self.regex_patterns[entity_type].append(compiled_pattern)
            self.logger.info(f"Added custom pattern for {entity_type.value}: {pattern}")
        except re.error as e:
            raise ValidationError(f"Invalid regex pattern: {e}")
    
    def get_supported_entity_types(self) -> List[EntityType]:
        """Get list of supported entity types"""
        return list(EntityType)
    
    def visualize_entities(self, ner_result: NERResult, output_path: Optional[str] = None) -> str:
        """Visualize entities using SpaCy's displacy"""
        if not self.spacy_nlp:
            raise ProcessingError("SpaCy model required for visualization")
        
        try:
            # Create SpaCy doc with entities
            doc = self.spacy_nlp(ner_result.text)
            
            # Convert our entities to SpaCy format
            spacy_entities = []
            for entity in ner_result.entities:
                spacy_entities.append((entity.start, entity.end, entity.label.value))
            
            # Create new doc with custom entities
            doc.ents = [Span(doc, start, end, label=label) for start, end, label in spacy_entities]
            
            # Generate visualization
            html = displacy.render(doc, style="ent", jupyter=False)
            
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html)
                self.logger.info(f"Entity visualization saved to {output_path}")
            
            return html
            
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
            raise ProcessingError(f"Entity visualization error: {e}")
    
    def get_entity_statistics(self, ner_result: NERResult) -> Dict[str, Any]:
        """Get statistics about detected entities"""
        stats = {
            'total_entities': len(ner_result.entities),
            'unique_entities': len(ner_result.get_unique_entities()),
            'entity_types': {},
            'confidence_distribution': {
                'high': 0,  # > 0.8
                'medium': 0,  # 0.5 - 0.8
                'low': 0  # < 0.5
            },
            'sources': {}
        }
        
        # Count by entity type
        for entity in ner_result.entities:
            entity_type = entity.label.value
            if entity_type not in stats['entity_types']:
                stats['entity_types'][entity_type] = 0
            stats['entity_types'][entity_type] += 1
            
            # Confidence distribution
            if entity.confidence > 0.8:
                stats['confidence_distribution']['high'] += 1
            elif entity.confidence > 0.5:
                stats['confidence_distribution']['medium'] += 1
            else:
                stats['confidence_distribution']['low'] += 1
            
            # Source distribution
            if entity.source not in stats['sources']:
                stats['sources'][entity.source] = 0
            stats['sources'][entity.source] += 1
        
        return stats
    
    def cleanup(self):
        """Cleanup NER processor resources"""
        try:
            if self.transformers_pipeline:
                del self.transformers_pipeline
                self.transformers_pipeline = None
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.debug("NER Processor cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Cleanup error: {e}")


# Factory function for easy processor creation
def create_ner_processor(
    spacy_model: str = "en_core_web_sm",
    use_transformers: bool = False,
    **config_kwargs
) -> NERProcessor:
    """
    Factory function to create NER processor
    
    Args:
        spacy_model: SpaCy model name
        use_transformers: Whether to use transformer models
        **config_kwargs: Additional configuration parameters
    
    Returns:
        Configured NERProcessor instance
    """
    config = NERConfig(
        spacy_model=spacy_model,
        use_transformers=use_transformers,
        **config_kwargs
    )
    
    return NERProcessor(config)