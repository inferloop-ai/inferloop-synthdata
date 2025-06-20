#!/usr/bin/env python3
"""
Entity Linker for linking detected entities to knowledge bases
"""

import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import requests
import json

from pydantic import BaseModel, Field

from ...core.config import get_config
from ...core.logging import get_logger
from ...core.exceptions import ProcessingError, ValidationError
from .ner_processor import Entity, EntityType


class KnowledgeBase(Enum):
    """Available knowledge bases"""
    WIKIDATA = "wikidata"
    DBPEDIA = "dbpedia"
    FREEBASE = "freebase"
    CUSTOM = "custom"


@dataclass
class LinkedEntity:
    """Entity linked to knowledge base"""
    original_entity: Entity
    kb_id: str
    kb_label: str
    kb_description: Optional[str] = None
    confidence: float = 0.0
    knowledge_base: KnowledgeBase = KnowledgeBase.WIKIDATA
    properties: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)


@dataclass
class EntityLinkingResult:
    """Entity linking result"""
    original_entities: List[Entity]
    linked_entities: List[LinkedEntity]
    unlinked_entities: List[Entity]
    processing_time: float = 0.0
    kb_stats: Dict[str, int] = field(default_factory=dict)


class EntityLinkerConfig(BaseModel):
    """Entity linker configuration"""
    
    # Knowledge base settings
    primary_kb: KnowledgeBase = Field(KnowledgeBase.WIKIDATA, description="Primary knowledge base")
    fallback_kbs: List[KnowledgeBase] = Field(default_factory=list, description="Fallback knowledge bases")
    
    # Linking settings
    confidence_threshold: float = Field(0.7, description="Minimum confidence for linking")
    max_candidates: int = Field(10, description="Maximum candidates to consider")
    enable_fuzzy_matching: bool = Field(True, description="Enable fuzzy string matching")
    
    # API settings
    request_timeout: int = Field(30, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum retry attempts")
    
    # Custom KB settings
    custom_kb_path: Optional[str] = Field(None, description="Path to custom knowledge base")


class EntityLinker:
    """
    Entity linker for connecting detected entities to knowledge bases
    """
    
    def __init__(self, config: Optional[EntityLinkerConfig] = None):
        self.logger = get_logger(__name__)
        self.config = config or EntityLinkerConfig()
        
        # Initialize knowledge base handlers
        self.kb_handlers = {}
        self._initialize_kb_handlers()
        
        self.logger.info(f"Entity Linker initialized with primary KB: {self.config.primary_kb.value}")
    
    def _initialize_kb_handlers(self):
        """Initialize knowledge base handlers"""
        try:
            if self.config.primary_kb == KnowledgeBase.WIKIDATA:
                self.kb_handlers[KnowledgeBase.WIKIDATA] = WikidataHandler(self.config)
            
            if self.config.primary_kb == KnowledgeBase.DBPEDIA:
                self.kb_handlers[KnowledgeBase.DBPEDIA] = DBpediaHandler(self.config)
            
            # Initialize fallback handlers
            for kb in self.config.fallback_kbs:
                if kb not in self.kb_handlers:
                    if kb == KnowledgeBase.WIKIDATA:
                        self.kb_handlers[kb] = WikidataHandler(self.config)
                    elif kb == KnowledgeBase.DBPEDIA:
                        self.kb_handlers[kb] = DBpediaHandler(self.config)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize KB handlers: {e}")
            raise ProcessingError(f"KB handler initialization failed: {e}")
    
    def link_entities(self, entities: List[Entity]) -> EntityLinkingResult:
        """Link entities to knowledge base"""
        start_time = time.time()
        
        try:
            linked_entities = []
            unlinked_entities = []
            kb_stats = {}
            
            for entity in entities:
                # Try to link entity
                linked_entity = self._link_single_entity(entity)
                
                if linked_entity and linked_entity.confidence >= self.config.confidence_threshold:
                    linked_entities.append(linked_entity)
                    
                    # Update KB stats
                    kb_name = linked_entity.knowledge_base.value
                    if kb_name not in kb_stats:
                        kb_stats[kb_name] = 0
                    kb_stats[kb_name] += 1
                else:
                    unlinked_entities.append(entity)
            
            processing_time = time.time() - start_time
            
            result = EntityLinkingResult(
                original_entities=entities,
                linked_entities=linked_entities,
                unlinked_entities=unlinked_entities,
                processing_time=processing_time,
                kb_stats=kb_stats
            )
            
            self.logger.debug(
                f"Entity linking completed in {processing_time:.2f}s: "
                f"{len(linked_entities)} linked, {len(unlinked_entities)} unlinked"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Entity linking failed: {e}")
            raise ProcessingError(f"Entity linking error: {e}")
    
    def _link_single_entity(self, entity: Entity) -> Optional[LinkedEntity]:
        """Link a single entity to knowledge base"""
        try:
            # Try primary knowledge base first
            primary_handler = self.kb_handlers.get(self.config.primary_kb)
            if primary_handler:
                linked_entity = primary_handler.search_entity(entity)
                if linked_entity and linked_entity.confidence >= self.config.confidence_threshold:
                    return linked_entity
            
            # Try fallback knowledge bases
            for kb in self.config.fallback_kbs:
                handler = self.kb_handlers.get(kb)
                if handler:
                    linked_entity = handler.search_entity(entity)
                    if linked_entity and linked_entity.confidence >= self.config.confidence_threshold:
                        return linked_entity
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to link entity '{entity.text}': {e}")
            return None
    
    def get_entity_details(self, linked_entity: LinkedEntity) -> Dict[str, Any]:
        """Get detailed information about linked entity"""
        try:
            handler = self.kb_handlers.get(linked_entity.knowledge_base)
            if handler:
                return handler.get_entity_details(linked_entity.kb_id)
            return {}
        except Exception as e:
            self.logger.error(f"Failed to get entity details: {e}")
            return {}


class BaseKBHandler:
    """Base class for knowledge base handlers"""
    
    def __init__(self, config: EntityLinkerConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    def search_entity(self, entity: Entity) -> Optional[LinkedEntity]:
        """Search for entity in knowledge base"""
        raise NotImplementedError
    
    def get_entity_details(self, entity_id: str) -> Dict[str, Any]:
        """Get detailed entity information"""
        raise NotImplementedError


class WikidataHandler(BaseKBHandler):
    """Wikidata knowledge base handler"""
    
    def __init__(self, config: EntityLinkerConfig):
        super().__init__(config)
        self.base_url = "https://www.wikidata.org/w/api.php"
        self.sparql_url = "https://query.wikidata.org/sparql"
    
    def search_entity(self, entity: Entity) -> Optional[LinkedEntity]:
        """Search entity in Wikidata"""
        try:
            # Search for entity using Wikidata API
            params = {
                'action': 'wbsearchentities',
                'search': entity.text,
                'language': 'en',
                'format': 'json',
                'limit': self.config.max_candidates
            }
            
            response = requests.get(
                self.base_url, 
                params=params, 
                timeout=self.config.request_timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            if 'search' in data and data['search']:
                # Take the best match
                best_match = data['search'][0]
                
                # Calculate confidence based on label similarity
                confidence = self._calculate_confidence(entity.text, best_match.get('label', ''))
                
                linked_entity = LinkedEntity(
                    original_entity=entity,
                    kb_id=best_match['id'],
                    kb_label=best_match.get('label', ''),
                    kb_description=best_match.get('description', ''),
                    confidence=confidence,
                    knowledge_base=KnowledgeBase.WIKIDATA,
                    aliases=best_match.get('aliases', [])
                )
                
                return linked_entity
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Wikidata search failed for '{entity.text}': {e}")
            return None
    
    def get_entity_details(self, entity_id: str) -> Dict[str, Any]:
        """Get detailed Wikidata entity information"""
        try:
            params = {
                'action': 'wbgetentities',
                'ids': entity_id,
                'format': 'json',
                'languages': 'en'
            }
            
            response = requests.get(
                self.base_url,
                params=params,
                timeout=self.config.request_timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            if 'entities' in data and entity_id in data['entities']:
                entity_data = data['entities'][entity_id]
                return {
                    'id': entity_id,
                    'label': entity_data.get('labels', {}).get('en', {}).get('value', ''),
                    'description': entity_data.get('descriptions', {}).get('en', {}).get('value', ''),
                    'aliases': [alias.get('value', '') for alias in entity_data.get('aliases', {}).get('en', [])],
                    'claims': entity_data.get('claims', {}),
                    'sitelinks': entity_data.get('sitelinks', {})
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Failed to get Wikidata details for {entity_id}: {e}")
            return {}
    
    def _calculate_confidence(self, original: str, candidate: str) -> float:
        """Calculate confidence score for entity match"""
        if not original or not candidate:
            return 0.0
        
        # Simple string similarity
        original_lower = original.lower().strip()
        candidate_lower = candidate.lower().strip()
        
        if original_lower == candidate_lower:
            return 1.0
        
        # Jaccard similarity for fuzzy matching
        if self.config.enable_fuzzy_matching:
            original_words = set(original_lower.split())
            candidate_words = set(candidate_lower.split())
            
            intersection = original_words.intersection(candidate_words)
            union = original_words.union(candidate_words)
            
            if union:
                return len(intersection) / len(union)
        
        return 0.0


class DBpediaHandler(BaseKBHandler):
    """DBpedia knowledge base handler"""
    
    def __init__(self, config: EntityLinkerConfig):
        super().__init__(config)
        self.lookup_url = "http://lookup.dbpedia.org/api/search"
        self.sparql_url = "http://dbpedia.org/sparql"
    
    def search_entity(self, entity: Entity) -> Optional[LinkedEntity]:
        """Search entity in DBpedia"""
        try:
            params = {
                'query': entity.text,
                'format': 'json',
                'maxResults': self.config.max_candidates
            }
            
            response = requests.get(
                self.lookup_url,
                params=params,
                timeout=self.config.request_timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            if 'docs' in data and data['docs']:
                # Take the best match
                best_match = data['docs'][0]
                
                # Calculate confidence
                confidence = self._calculate_confidence(entity.text, best_match.get('label', [''])[0])
                
                linked_entity = LinkedEntity(
                    original_entity=entity,
                    kb_id=best_match['resource'][0],
                    kb_label=best_match.get('label', [''])[0],
                    kb_description=best_match.get('comment', [''])[0],
                    confidence=confidence,
                    knowledge_base=KnowledgeBase.DBPEDIA
                )
                
                return linked_entity
            
            return None
            
        except Exception as e:
            self.logger.warning(f"DBpedia search failed for '{entity.text}': {e}")
            return None
    
    def get_entity_details(self, entity_id: str) -> Dict[str, Any]:
        """Get detailed DBpedia entity information"""
        # Simplified implementation - would use SPARQL queries in practice
        return {
            'id': entity_id,
            'resource_uri': entity_id
        }
    
    def _calculate_confidence(self, original: str, candidate: str) -> float:
        """Calculate confidence score for entity match"""
        if not original or not candidate:
            return 0.0
        
        original_lower = original.lower().strip()
        candidate_lower = candidate.lower().strip()
        
        if original_lower == candidate_lower:
            return 1.0
        
        if self.config.enable_fuzzy_matching:
            # Simple containment check
            if original_lower in candidate_lower or candidate_lower in original_lower:
                return 0.8
        
        return 0.0


# Factory function for easy linker creation
def create_entity_linker(
    primary_kb: str = "wikidata",
    **config_kwargs
) -> EntityLinker:
    """
    Factory function to create entity linker
    
    Args:
        primary_kb: Primary knowledge base ('wikidata', 'dbpedia')
        **config_kwargs: Additional configuration parameters
    
    Returns:
        Configured EntityLinker instance
    """
    config = EntityLinkerConfig(
        primary_kb=KnowledgeBase(primary_kb),
        **config_kwargs
    )
    
    return EntityLinker(config)