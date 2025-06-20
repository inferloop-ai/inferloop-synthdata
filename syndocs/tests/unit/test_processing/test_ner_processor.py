"""
Unit tests for Named Entity Recognition (NER) processor.

Tests the NER processor for extracting entities from document text
including persons, organizations, locations, dates, and custom entities.
"""

import pytest
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import spacy
from datetime import datetime

from structured_docs_synth.processing.nlp.ner_processor import (
    NERProcessor,
    NERConfig,
    Entity,
    EntityType,
    NERResult,
    EntityRelation
)
from structured_docs_synth.core.exceptions import ProcessingError


class TestNERConfig:
    """Test NER configuration."""
    
    def test_default_config(self):
        """Test default NER configuration."""
        config = NERConfig()
        
        assert config.model_name == 'en_core_web_sm'
        assert config.confidence_threshold == 0.7
        assert config.enable_custom_entities is True
        assert config.merge_adjacent_entities is True
        assert config.max_entity_length == 100
        assert config.entity_types == set(EntityType)
    
    def test_custom_config(self):
        """Test custom NER configuration."""
        custom_types = {EntityType.PERSON, EntityType.ORGANIZATION}
        config = NERConfig(
            model_name='en_core_web_trf',
            confidence_threshold=0.85,
            entity_types=custom_types,
            enable_gpu=True
        )
        
        assert config.model_name == 'en_core_web_trf'
        assert config.confidence_threshold == 0.85
        assert config.entity_types == custom_types
        assert config.enable_gpu is True


class TestEntity:
    """Test Entity data structure."""
    
    def test_entity_creation(self):
        """Test creating entity."""
        entity = Entity(
            text="John Doe",
            type=EntityType.PERSON,
            start_char=0,
            end_char=8,
            confidence=0.95,
            metadata={'source': 'spacy'}
        )
        
        assert entity.text == "John Doe"
        assert entity.type == EntityType.PERSON
        assert entity.start_char == 0
        assert entity.end_char == 8
        assert entity.confidence == 0.95
        assert entity.metadata['source'] == 'spacy'
    
    def test_entity_overlap(self):
        """Test entity overlap detection."""
        entity1 = Entity("New York", EntityType.LOCATION, 0, 8)
        entity2 = Entity("York City", EntityType.LOCATION, 4, 13)
        entity3 = Entity("Boston", EntityType.LOCATION, 20, 26)
        
        assert entity1.overlaps_with(entity2) is True
        assert entity1.overlaps_with(entity3) is False
    
    def test_entity_contains(self):
        """Test entity containment."""
        entity1 = Entity("United States of America", EntityType.LOCATION, 0, 24)
        entity2 = Entity("States", EntityType.LOCATION, 7, 13)
        
        assert entity1.contains(entity2) is True
        assert entity2.contains(entity1) is False


class TestNERProcessor:
    """Test NER processor functionality."""
    
    @pytest.fixture
    def ner_config(self):
        """Provide NER configuration."""
        return NERConfig(
            model_name='en_core_web_sm',
            confidence_threshold=0.7
        )
    
    @pytest.fixture
    def ner_processor(self, ner_config):
        """Provide NER processor instance."""
        with patch('spacy.load') as mock_load:
            mock_nlp = MagicMock()
            mock_load.return_value = mock_nlp
            processor = NERProcessor(ner_config)
            processor.nlp = mock_nlp
            return processor
    
    @pytest.fixture
    def sample_text(self):
        """Provide sample text for testing."""
        return """
        John Smith, CEO of Acme Corporation, announced today in New York 
        that the company will invest $5 million in AI research. The project 
        will start on January 15, 2024 and involve collaboration with 
        Stanford University. Contact: john.smith@acme.com or call 555-1234.
        """
    
    def test_processor_initialization(self, ner_processor, ner_config):
        """Test processor initialization."""
        assert ner_processor.config == ner_config
        assert ner_processor.nlp is not None
        assert ner_processor.custom_patterns is not None
    
    @pytest.mark.asyncio
    async def test_extract_entities_success(self, ner_processor, sample_text):
        """Test successful entity extraction."""
        # Mock spaCy doc
        mock_doc = Mock()
        mock_ents = [
            Mock(text="John Smith", label_="PERSON", start_char=9, end_char=19),
            Mock(text="Acme Corporation", label_="ORG", start_char=28, end_char=44),
            Mock(text="New York", label_="GPE", start_char=65, end_char=73),
            Mock(text="$5 million", label_="MONEY", start_char=104, end_char=114),
            Mock(text="January 15, 2024", label_="DATE", start_char=159, end_char=175),
            Mock(text="Stanford University", label_="ORG", start_char=205, end_char=224)
        ]
        mock_doc.ents = mock_ents
        ner_processor.nlp.return_value = mock_doc
        
        # Extract entities
        result = await ner_processor.extract_entities(sample_text)
        
        assert len(result.entities) == 6
        assert result.entities[0].text == "John Smith"
        assert result.entities[0].type == EntityType.PERSON
        assert result.entities[1].text == "Acme Corporation"
        assert result.entities[1].type == EntityType.ORGANIZATION
    
    @pytest.mark.asyncio
    async def test_extract_custom_entities(self, ner_processor, sample_text):
        """Test custom entity extraction."""
        # Add custom patterns
        ner_processor.add_custom_pattern(
            pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            entity_type=EntityType.EMAIL
        )
        ner_processor.add_custom_pattern(
            pattern=r'\b\d{3}-\d{4}\b',
            entity_type=EntityType.PHONE
        )
        
        # Mock spaCy (no email/phone entities)
        mock_doc = Mock()
        mock_doc.ents = []
        ner_processor.nlp.return_value = mock_doc
        
        result = await ner_processor.extract_entities(sample_text)
        
        # Should find custom entities
        emails = [e for e in result.entities if e.type == EntityType.EMAIL]
        phones = [e for e in result.entities if e.type == EntityType.PHONE]
        
        assert len(emails) == 1
        assert emails[0].text == "john.smith@acme.com"
        assert len(phones) == 1
        assert phones[0].text == "555-1234"
    
    @pytest.mark.asyncio
    async def test_merge_adjacent_entities(self, ner_processor):
        """Test merging adjacent entities of same type."""
        text = "New York City and Los Angeles are major cities"
        
        # Mock entities that should be merged
        mock_doc = Mock()
        mock_ents = [
            Mock(text="New", label_="GPE", start_char=0, end_char=3),
            Mock(text="York", label_="GPE", start_char=4, end_char=8),
            Mock(text="City", label_="GPE", start_char=9, end_char=13),
            Mock(text="Los Angeles", label_="GPE", start_char=18, end_char=29)
        ]
        mock_doc.ents = mock_ents
        ner_processor.nlp.return_value = mock_doc
        
        ner_processor.config.merge_adjacent_entities = True
        result = await ner_processor.extract_entities(text)
        
        # Should merge "New York City"
        locations = [e for e in result.entities if e.type == EntityType.LOCATION]
        assert len(locations) == 2
        assert any(e.text == "New York City" for e in locations)
    
    @pytest.mark.asyncio
    async def test_confidence_filtering(self, ner_processor):
        """Test entity confidence filtering."""
        # Mock entities with confidence scores
        mock_doc = Mock()
        mock_ents = [
            Mock(text="High Conf", label_="PERSON", start_char=0, end_char=9, 
                 _.confidence=0.9),
            Mock(text="Low Conf", label_="PERSON", start_char=10, end_char=18,
                 _.confidence=0.5)
        ]
        
        # Setup confidence attribute
        for ent in mock_ents:
            ent._.get = lambda attr: 0.9 if "High" in ent.text else 0.5
        
        mock_doc.ents = mock_ents
        ner_processor.nlp.return_value = mock_doc
        
        ner_processor.config.confidence_threshold = 0.7
        result = await ner_processor.extract_entities("High Conf Low Conf")
        
        # Should filter out low confidence
        assert len(result.entities) == 1
        assert result.entities[0].text == "High Conf"
    
    @pytest.mark.asyncio
    async def test_entity_relations(self, ner_processor):
        """Test entity relation extraction."""
        text = "John Smith is the CEO of Acme Corporation"
        
        # Mock entities
        mock_doc = Mock()
        mock_ents = [
            Mock(text="John Smith", label_="PERSON", start_char=0, end_char=10),
            Mock(text="CEO", label_="TITLE", start_char=18, end_char=21),
            Mock(text="Acme Corporation", label_="ORG", start_char=25, end_char=41)
        ]
        mock_doc.ents = mock_ents
        ner_processor.nlp.return_value = mock_doc
        
        # Enable relation extraction
        ner_processor.config.extract_relations = True
        result = await ner_processor.extract_entities(text)
        
        # Should find person-org relation
        assert len(result.relations) > 0
        relation = result.relations[0]
        assert relation.source.text == "John Smith"
        assert relation.target.text == "Acme Corporation"
        assert relation.type == "works_for"
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, ner_processor):
        """Test batch entity extraction."""
        texts = [
            "Apple Inc. is in California",
            "Microsoft is in Washington",
            "Google is in Mountain View"
        ]
        
        # Mock batch processing
        mock_docs = []
        for i, text in enumerate(texts):
            mock_doc = Mock()
            company = text.split()[0]
            location = text.split()[-1]
            mock_doc.ents = [
                Mock(text=company, label_="ORG", start_char=0, end_char=len(company)),
                Mock(text=location, label_="GPE", start_char=text.rfind(location), 
                     end_char=len(text))
            ]
            mock_docs.append(mock_doc)
        
        ner_processor.nlp.pipe = Mock(return_value=mock_docs)
        
        results = await ner_processor.batch_extract_entities(texts)
        
        assert len(results) == 3
        assert all(len(r.entities) >= 2 for r in results)
    
    def test_add_custom_entity_type(self, ner_processor):
        """Test adding custom entity types."""
        # Add custom entity type
        ner_processor.add_custom_entity_type(
            name="PRODUCT_CODE",
            pattern=r'PRD-\d{6}',
            examples=["PRD-123456", "PRD-789012"]
        )
        
        assert "PRODUCT_CODE" in ner_processor.custom_patterns
        assert len(ner_processor.custom_patterns["PRODUCT_CODE"]) > 0
    
    @pytest.mark.asyncio
    async def test_domain_specific_extraction(self, ner_processor):
        """Test domain-specific entity extraction."""
        # Medical domain text
        medical_text = """
        Patient John Doe (MRN: 12345) was prescribed Lisinopril 10mg 
        for hypertension (ICD-10: I10). Follow-up in 3 months.
        """
        
        # Configure for medical domain
        ner_processor.config.domain = "medical"
        
        # Mock medical entities
        mock_doc = Mock()
        mock_ents = [
            Mock(text="John Doe", label_="PERSON", start_char=8, end_char=16),
            Mock(text="12345", label_="ID", start_char=23, end_char=28),
            Mock(text="Lisinopril 10mg", label_="DRUG", start_char=44, end_char=59),
            Mock(text="hypertension", label_="DISEASE", start_char=64, end_char=76),
            Mock(text="I10", label_="ICD", start_char=86, end_char=89)
        ]
        mock_doc.ents = mock_ents
        ner_processor.nlp.return_value = mock_doc
        
        result = await ner_processor.extract_entities(medical_text)
        
        # Should extract medical entities
        assert any(e.type == EntityType.MEDICAL_TERM for e in result.entities)
    
    @pytest.mark.asyncio
    async def test_entity_normalization(self, ner_processor):
        """Test entity normalization."""
        text = "JOHN SMITH and John Smith are the same person"
        
        # Mock entities
        mock_doc = Mock()
        mock_ents = [
            Mock(text="JOHN SMITH", label_="PERSON", start_char=0, end_char=10),
            Mock(text="John Smith", label_="PERSON", start_char=15, end_char=25)
        ]
        mock_doc.ents = mock_ents
        ner_processor.nlp.return_value = mock_doc
        
        ner_processor.config.normalize_entities = True
        result = await ner_processor.extract_entities(text)
        
        # Should normalize to same form
        persons = [e for e in result.entities if e.type == EntityType.PERSON]
        assert len(persons) == 2
        # Check if normalized forms are tracked
        assert result.normalized_entities.get("john smith") == ["JOHN SMITH", "John Smith"]
    
    @pytest.mark.asyncio
    async def test_entity_disambiguation(self, ner_processor):
        """Test entity disambiguation."""
        text = "Apple released a new iPhone. I ate an apple for lunch."
        
        # Mock entities with context
        mock_doc = Mock()
        mock_tokens = []
        
        # First "Apple" - company context
        apple1 = Mock(text="Apple", label_="ORG", start_char=0, end_char=5)
        apple1.sent = Mock(text="Apple released a new iPhone.")
        
        # Second "apple" - fruit context
        apple2 = Mock(text="apple", label_="MISC", start_char=38, end_char=43)
        apple2.sent = Mock(text="I ate an apple for lunch.")
        
        mock_doc.ents = [apple1, apple2]
        ner_processor.nlp.return_value = mock_doc
        
        result = await ner_processor.extract_entities(text)
        
        # Should disambiguate based on context
        assert result.entities[0].type == EntityType.ORGANIZATION
        assert result.entities[0].metadata.get("subtype") == "company"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, ner_processor):
        """Test error handling during extraction."""
        # Mock NLP failure
        ner_processor.nlp.side_effect = Exception("Model loading failed")
        
        with pytest.raises(ProcessingError, match="Entity extraction failed"):
            await ner_processor.extract_entities("Test text")
    
    def test_entity_filtering(self, ner_processor):
        """Test entity type filtering."""
        # Configure to only extract specific types
        ner_processor.config.entity_types = {EntityType.PERSON, EntityType.ORGANIZATION}
        
        entities = [
            Entity("John Doe", EntityType.PERSON, 0, 8),
            Entity("Acme Corp", EntityType.ORGANIZATION, 10, 19),
            Entity("New York", EntityType.LOCATION, 21, 29),  # Should be filtered
            Entity("$1000", EntityType.MONEY, 31, 36)  # Should be filtered
        ]
        
        filtered = ner_processor._filter_entities(entities)
        
        assert len(filtered) == 2
        assert all(e.type in {EntityType.PERSON, EntityType.ORGANIZATION} for e in filtered)
    
    @pytest.mark.asyncio
    async def test_coreference_resolution(self, ner_processor):
        """Test coreference resolution for entities."""
        text = "John Smith is our CEO. He founded the company in 2010."
        
        # Mock entities and coreferences
        mock_doc = Mock()
        mock_ents = [
            Mock(text="John Smith", label_="PERSON", start_char=0, end_char=10),
            Mock(text="CEO", label_="TITLE", start_char=18, end_char=21)
        ]
        
        # Mock coreference clusters
        mock_doc._.coref_clusters = [
            Mock(mentions=[
                Mock(text="John Smith", start=0, end=10),
                Mock(text="He", start=23, end=25)
            ])
        ]
        
        mock_doc.ents = mock_ents
        ner_processor.nlp.return_value = mock_doc
        
        ner_processor.config.resolve_coreferences = True
        result = await ner_processor.extract_entities(text)
        
        # Should link "He" to "John Smith"
        assert any(e.metadata.get("refers_to") == "John Smith" for e in result.entities)