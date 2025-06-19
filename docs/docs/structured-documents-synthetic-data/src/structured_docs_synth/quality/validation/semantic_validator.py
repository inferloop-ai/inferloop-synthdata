"""
Semantic validator for document content consistency and logical validation.
Validates semantic coherence, entity relationships, and content logic.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Union, Any, Tuple

from pydantic import BaseModel, Field, validator

from ...core.config import BaseConfig
from ...core.exceptions import ValidationError
from ...core.logging import get_logger

logger = get_logger(__name__)


class SemanticIssueType(Enum):
    """Types of semantic validation issues"""
    INCONSISTENCY = "inconsistency"
    CONTRADICTION = "contradiction"
    MISSING_CONTEXT = "missing_context"
    INVALID_REFERENCE = "invalid_reference"
    LOGICAL_ERROR = "logical_error"
    ENTITY_MISMATCH = "entity_mismatch"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    FACTUAL_ERROR = "factual_error"


class EntityType(Enum):
    """Named entity types"""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    TIME = "time"
    MONEY = "money"
    PERCENTAGE = "percentage"
    NUMBER = "number"
    EMAIL = "email"
    PHONE = "phone"
    URL = "url"


class SemanticSeverity(Enum):
    """Semantic issue severity levels"""
    MINOR = "minor"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    CRITICAL = "critical"


@dataclass
class Entity:
    """Named entity annotation"""
    entity_id: str
    entity_type: EntityType
    text: str
    start_pos: int
    end_pos: int
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type.value,
            "text": self.text,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": self.confidence,
            "attributes": self.attributes
        }


@dataclass
class SemanticRelation:
    """Semantic relationship between entities"""
    relation_id: str
    relation_type: str
    subject_entity: Entity
    object_entity: Entity
    confidence: float = 1.0
    context: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "relation_id": self.relation_id,
            "relation_type": self.relation_type,
            "subject_entity": self.subject_entity.to_dict(),
            "object_entity": self.object_entity.to_dict(),
            "confidence": self.confidence,
            "context": self.context
        }


@dataclass
class SemanticIssue:
    """Semantic validation issue"""
    issue_id: str
    issue_type: SemanticIssueType
    severity: SemanticSeverity
    message: str
    location: Optional[str] = None
    entities_involved: List[Entity] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "issue_id": self.issue_id,
            "issue_type": self.issue_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "location": self.location,
            "entities_involved": [e.to_dict() for e in self.entities_involved],
            "suggestions": self.suggestions,
            "confidence": self.confidence
        }


@dataclass
class SemanticValidationResult:
    """Semantic validation result"""
    document_id: str
    timestamp: datetime
    semantic_score: float
    issues: List[SemanticIssue]
    entities: List[Entity]
    relations: List[SemanticRelation]
    consistency_metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "document_id": self.document_id,
            "timestamp": self.timestamp.isoformat(),
            "semantic_score": self.semantic_score,
            "issues": [issue.to_dict() for issue in self.issues],
            "entities": [entity.to_dict() for entity in self.entities],
            "relations": [relation.to_dict() for relation in self.relations],
            "consistency_metrics": self.consistency_metrics
        }


@dataclass
class DocumentSemantics:
    """Document semantic content"""
    document_id: str
    text: str
    entities: List[Entity] = field(default_factory=list)
    relations: List[SemanticRelation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    sections: List[Dict[str, Any]] = field(default_factory=list)


class SemanticValidatorConfig(BaseConfig):
    """Semantic validator configuration"""
    entity_consistency_threshold: float = Field(default=0.8, description="Entity consistency threshold")
    temporal_consistency_enabled: bool = Field(default=True, description="Enable temporal consistency checks")
    reference_validation_enabled: bool = Field(default=True, description="Enable reference validation")
    factual_checking_enabled: bool = Field(default=False, description="Enable factual checking")
    min_entity_confidence: float = Field(default=0.5, description="Minimum entity confidence threshold")
    
    # Entity patterns
    person_patterns: List[str] = Field(
        default=[
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # John Smith
            r'\b(?:Mr|Ms|Mrs|Dr|Prof)\.? [A-Z][a-z]+\b'  # Dr. Smith
        ],
        description="Person name patterns"
    )
    
    organization_patterns: List[str] = Field(
        default=[
            r'\b[A-Z][A-Za-z]+ (?:Inc|Corp|LLC|Ltd|Company|Corporation)\b',
            r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b'  # Acronyms
        ],
        description="Organization patterns"
    )
    
    date_patterns: List[str] = Field(
        default=[
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'
        ],
        description="Date patterns"
    )


class SemanticValidator:
    """
    Semantic validator for document content.
    Validates semantic coherence, entity relationships, and content logic.
    """
    
    def __init__(self, config: Optional[SemanticValidatorConfig] = None):
        """Initialize semantic validator"""
        self.config = config or SemanticValidatorConfig()
        self.entity_patterns = self._compile_entity_patterns()
        logger.info("Initialized SemanticValidator")
    
    def validate_document_semantics(
        self,
        document_semantics: DocumentSemantics
    ) -> SemanticValidationResult:
        """
        Validate document semantic content.
        
        Args:
            document_semantics: Document semantic content
        
        Returns:
            Semantic validation result
        """
        # Extract entities if not provided
        if not document_semantics.entities:
            document_semantics.entities = self._extract_entities(document_semantics.text)
        
        # Extract relations if not provided
        if not document_semantics.relations:
            document_semantics.relations = self._extract_relations(
                document_semantics.text, document_semantics.entities
            )
        
        issues = []
        
        # Validate entity consistency
        entity_issues = self._validate_entity_consistency(document_semantics)
        issues.extend(entity_issues)
        
        # Validate temporal consistency
        if self.config.temporal_consistency_enabled:
            temporal_issues = self._validate_temporal_consistency(document_semantics)
            issues.extend(temporal_issues)
        
        # Validate references
        if self.config.reference_validation_enabled:
            reference_issues = self._validate_references(document_semantics)
            issues.extend(reference_issues)
        
        # Validate logical consistency
        logical_issues = self._validate_logical_consistency(document_semantics)
        issues.extend(logical_issues)
        
        # Validate entity relationships
        relation_issues = self._validate_entity_relations(document_semantics)
        issues.extend(relation_issues)
        
        # Calculate consistency metrics
        consistency_metrics = self._calculate_consistency_metrics(document_semantics, issues)
        
        # Calculate semantic score
        semantic_score = self._calculate_semantic_score(issues, consistency_metrics)
        
        return SemanticValidationResult(
            document_id=document_semantics.document_id,
            timestamp=datetime.now(),
            semantic_score=semantic_score,
            issues=issues,
            entities=document_semantics.entities,
            relations=document_semantics.relations,
            consistency_metrics=consistency_metrics
        )
    
    def _compile_entity_patterns(self) -> Dict[EntityType, List[re.Pattern]]:
        """Compile entity recognition patterns"""
        patterns = {}
        
        patterns[EntityType.PERSON] = [re.compile(pattern, re.IGNORECASE) 
                                     for pattern in self.config.person_patterns]
        patterns[EntityType.ORGANIZATION] = [re.compile(pattern, re.IGNORECASE) 
                                           for pattern in self.config.organization_patterns]
        patterns[EntityType.DATE] = [re.compile(pattern, re.IGNORECASE) 
                                   for pattern in self.config.date_patterns]
        
        # Additional patterns
        patterns[EntityType.EMAIL] = [re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')]
        patterns[EntityType.PHONE] = [re.compile(r'\b\+?[\d\s\-\(\)]{10,}\b')]
        patterns[EntityType.URL] = [re.compile(r'https?://[^\s]+')]
        patterns[EntityType.MONEY] = [re.compile(r'\$[\d,]+(?:\.\d{2})?')]
        patterns[EntityType.PERCENTAGE] = [re.compile(r'\b\d+(?:\.\d+)?%\b')]
        patterns[EntityType.NUMBER] = [re.compile(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b')]
        
        return patterns
    
    def _extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text using pattern matching"""
        entities = []
        entity_id_counter = 0
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entity_id_counter += 1
                    entity = Entity(
                        entity_id=f"entity_{entity_id_counter}",
                        entity_type=entity_type,
                        text=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.8  # Pattern-based confidence
                    )
                    entities.append(entity)
        
        # Remove overlapping entities (keep longer ones)
        entities = self._remove_overlapping_entities(entities)
        
        return entities
    
    def _remove_overlapping_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove overlapping entities, keeping the longer ones"""
        # Sort by start position
        sorted_entities = sorted(entities, key=lambda e: e.start_pos)
        
        filtered_entities = []
        for entity in sorted_entities:
            # Check if this entity overlaps with any already added entity
            overlaps = False
            for existing in filtered_entities:
                if (entity.start_pos < existing.end_pos and 
                    entity.end_pos > existing.start_pos):
                    # There's an overlap, keep the longer entity
                    if len(entity.text) > len(existing.text):
                        filtered_entities.remove(existing)
                    else:
                        overlaps = True
                    break
            
            if not overlaps:
                filtered_entities.append(entity)
        
        return filtered_entities
    
    def _extract_relations(self, text: str, entities: List[Entity]) -> List[SemanticRelation]:
        """Extract semantic relations between entities"""
        relations = []
        relation_id_counter = 0
        
        # Simple relation patterns
        relation_patterns = {
            "works_for": [
                r"(?i)({person})\s+(?:works?|worked)\s+(?:for|at)\s+({org})",
                r"(?i)({person})\s+(?:is|was)\s+(?:employed|hired)\s+(?:by|at)\s+({org})"
            ],
            "located_in": [
                r"(?i)({org})\s+(?:is|was)\s+(?:located|based)\s+in\s+({location})",
                r"(?i)({location})\s+(?:houses|hosts)\s+({org})"
            ],
            "founded_on": [
                r"(?i)({org})\s+(?:was\s+)?(?:founded|established)\s+(?:on|in)\s+({date})"
            ]
        }
        
        # Create entity lookup by type
        entities_by_type = defaultdict(list)
        for entity in entities:
            entities_by_type[entity.entity_type].append(entity)
        
        # Find relations
        for relation_type, patterns in relation_patterns.items():
            for pattern_template in patterns:
                # Replace placeholders with entity patterns
                pattern_str = pattern_template
                
                if "{person}" in pattern_str:
                    person_entities = entities_by_type[EntityType.PERSON]
                    person_pattern = "|".join(re.escape(e.text) for e in person_entities)
                    pattern_str = pattern_str.replace("{person}", f"({person_pattern})")
                
                if "{org}" in pattern_str:
                    org_entities = entities_by_type[EntityType.ORGANIZATION]
                    org_pattern = "|".join(re.escape(e.text) for e in org_entities)
                    pattern_str = pattern_str.replace("{org}", f"({org_pattern})")
                
                if "{location}" in pattern_str:
                    loc_entities = entities_by_type[EntityType.LOCATION]
                    if loc_entities:
                        loc_pattern = "|".join(re.escape(e.text) for e in loc_entities)
                        pattern_str = pattern_str.replace("{location}", f"({loc_pattern})")
                
                if "{date}" in pattern_str:
                    date_entities = entities_by_type[EntityType.DATE]
                    date_pattern = "|".join(re.escape(e.text) for e in date_entities)
                    pattern_str = pattern_str.replace("{date}", f"({date_pattern})")
                
                # Skip if pattern couldn't be formed
                if "{" in pattern_str:
                    continue
                
                try:
                    pattern = re.compile(pattern_str)
                    for match in pattern.finditer(text):
                        if len(match.groups()) >= 2:
                            subject_text, object_text = match.groups()[:2]
                            
                            # Find corresponding entities
                            subject_entity = next(
                                (e for e in entities if e.text == subject_text), None
                            )
                            object_entity = next(
                                (e for e in entities if e.text == object_text), None
                            )
                            
                            if subject_entity and object_entity:
                                relation_id_counter += 1
                                relation = SemanticRelation(
                                    relation_id=f"relation_{relation_id_counter}",
                                    relation_type=relation_type,
                                    subject_entity=subject_entity,
                                    object_entity=object_entity,
                                    confidence=0.7,
                                    context=match.group()
                                )
                                relations.append(relation)
                
                except re.error:
                    continue
        
        return relations
    
    def _validate_entity_consistency(self, document_semantics: DocumentSemantics) -> List[SemanticIssue]:
        """Validate entity consistency across document"""
        issues = []
        
        # Group entities by text
        entity_groups = defaultdict(list)
        for entity in document_semantics.entities:
            normalized_text = entity.text.lower().strip()
            entity_groups[normalized_text].append(entity)
        
        # Check for inconsistent entity types
        for text, entities in entity_groups.items():
            if len(entities) > 1:
                entity_types = set(e.entity_type for e in entities)
                if len(entity_types) > 1:
                    issues.append(SemanticIssue(
                        issue_id=f"entity_type_inconsistency_{text.replace(' ', '_')}",
                        issue_type=SemanticIssueType.INCONSISTENCY,
                        severity=SemanticSeverity.MODERATE,
                        message=f"Entity '{text}' has inconsistent types: {[t.value for t in entity_types]}",
                        entities_involved=entities,
                        suggestions=[
                            f"Standardize entity type for '{text}'",
                            "Review entity recognition accuracy"
                        ]
                    ))
        
        # Check for low-confidence entities
        low_confidence_entities = [
            e for e in document_semantics.entities 
            if e.confidence < self.config.min_entity_confidence
        ]
        
        if low_confidence_entities:
            issues.append(SemanticIssue(
                issue_id="low_confidence_entities",
                issue_type=SemanticIssueType.ENTITY_MISMATCH,
                severity=SemanticSeverity.MINOR,
                message=f"Found {len(low_confidence_entities)} entities with low confidence",
                entities_involved=low_confidence_entities,
                suggestions=[
                    "Review and verify low-confidence entity annotations",
                    "Consider manual validation for important entities"
                ]
            ))
        
        return issues
    
    def _validate_temporal_consistency(self, document_semantics: DocumentSemantics) -> List[SemanticIssue]:
        """Validate temporal consistency in the document"""
        issues = []
        
        # Extract date entities
        date_entities = [e for e in document_semantics.entities if e.entity_type == EntityType.DATE]
        
        if not date_entities:
            return issues
        
        # Parse dates and check for logical order
        parsed_dates = []
        for entity in date_entities:
            parsed_date = self._parse_date_entity(entity)
            if parsed_date:
                parsed_dates.append((entity, parsed_date))
        
        # Check for temporal inconsistencies
        for i, (entity1, date1) in enumerate(parsed_dates):
            for entity2, date2 in parsed_dates[i+1:]:
                # Look for context clues about temporal order
                context_text = document_semantics.text[
                    max(0, min(entity1.start_pos, entity2.start_pos) - 100):
                    max(entity1.end_pos, entity2.end_pos) + 100
                ]
                
                # Check for explicit temporal indicators
                if "before" in context_text.lower() and date1 > date2:
                    issues.append(SemanticIssue(
                        issue_id=f"temporal_inconsistency_{entity1.entity_id}_{entity2.entity_id}",
                        issue_type=SemanticIssueType.TEMPORAL_INCONSISTENCY,
                        severity=SemanticSeverity.SIGNIFICANT,
                        message=f"Temporal inconsistency: '{entity1.text}' should be before '{entity2.text}'",
                        entities_involved=[entity1, entity2],
                        suggestions=[
                            "Verify the temporal order of events",
                            "Check date accuracy"
                        ]
                    ))
                
                elif "after" in context_text.lower() and date1 < date2:
                    issues.append(SemanticIssue(
                        issue_id=f"temporal_inconsistency_{entity1.entity_id}_{entity2.entity_id}",
                        issue_type=SemanticIssueType.TEMPORAL_INCONSISTENCY,
                        severity=SemanticSeverity.SIGNIFICANT,
                        message=f"Temporal inconsistency: '{entity1.text}' should be after '{entity2.text}'",
                        entities_involved=[entity1, entity2],
                        suggestions=[
                            "Verify the temporal order of events",
                            "Check date accuracy"
                        ]
                    ))
        
        return issues
    
    def _validate_references(self, document_semantics: DocumentSemantics) -> List[SemanticIssue]:
        """Validate entity references and coreferences"""
        issues = []
        
        # Look for pronouns and references that might be ambiguous
        pronoun_patterns = [
            r'\b(?:he|she|it|they|this|that|these|those)\b',
            r'\b(?:his|her|its|their)\b',
            r'\b(?:him|her|them)\b'
        ]
        
        text = document_semantics.text.lower()
        
        for pattern in pronoun_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            
            for match in matches:
                # Look for nearby entities that could be referenced
                start_search = max(0, match.start() - 200)
                nearby_text = document_semantics.text[start_search:match.start()]
                
                nearby_entities = [
                    e for e in document_semantics.entities
                    if start_search <= e.start_pos < match.start()
                ]
                
                # Check for ambiguous references
                if len(nearby_entities) > 1:
                    # Filter by compatible entity types
                    pronoun = match.group().lower()
                    compatible_entities = []
                    
                    if pronoun in ["he", "his", "him"]:
                        # Likely refers to a person (male)
                        compatible_entities = [e for e in nearby_entities 
                                             if e.entity_type == EntityType.PERSON]
                    elif pronoun in ["she", "her"]:
                        # Likely refers to a person (female)
                        compatible_entities = [e for e in nearby_entities 
                                             if e.entity_type == EntityType.PERSON]
                    elif pronoun in ["it", "its"]:
                        # Likely refers to an organization or thing
                        compatible_entities = [e for e in nearby_entities 
                                             if e.entity_type == EntityType.ORGANIZATION]
                    
                    if len(compatible_entities) > 1:
                        issues.append(SemanticIssue(
                            issue_id=f"ambiguous_reference_{match.start()}",
                            issue_type=SemanticIssueType.INVALID_REFERENCE,
                            severity=SemanticSeverity.MINOR,
                            message=f"Ambiguous reference '{pronoun}' - multiple possible antecedents",
                            location=f"Position {match.start()}",
                            entities_involved=compatible_entities,
                            suggestions=[
                                f"Clarify what '{pronoun}' refers to",
                                "Use specific names instead of pronouns where ambiguous"
                            ]
                        ))
        
        return issues
    
    def _validate_logical_consistency(self, document_semantics: DocumentSemantics) -> List[SemanticIssue]:
        """Validate logical consistency in content"""
        issues = []
        
        # Look for contradictions in statements
        contradiction_patterns = [
            (r'(\w+)\s+is\s+(\w+)', r'\1\s+is\s+not\s+\2'),
            (r'(\w+)\s+has\s+(\w+)', r'\1\s+(?:does\s+not\s+have|lacks)\s+\2'),
            (r'(\w+)\s+can\s+(\w+)', r'\1\s+cannot\s+\2'),
        ]
        
        text = document_semantics.text.lower()
        
        for positive_pattern, negative_pattern in contradiction_patterns:
            positive_matches = list(re.finditer(positive_pattern, text))
            negative_matches = list(re.finditer(negative_pattern, text))
            
            for pos_match in positive_matches:
                for neg_match in negative_matches:
                    # Check if they refer to the same subject
                    if pos_match.group(1) == neg_match.group(1):
                        issues.append(SemanticIssue(
                            issue_id=f"logical_contradiction_{pos_match.start()}_{neg_match.start()}",
                            issue_type=SemanticIssueType.CONTRADICTION,
                            severity=SemanticSeverity.SIGNIFICANT,
                            message=f"Logical contradiction found: '{pos_match.group()}' vs '{neg_match.group()}'",
                            suggestions=[
                                "Review statements for consistency",
                                "Clarify which statement is correct"
                            ]
                        ))
        
        return issues
    
    def _validate_entity_relations(self, document_semantics: DocumentSemantics) -> List[SemanticIssue]:
        """Validate entity relationships"""
        issues = []
        
        # Check for conflicting relations
        relation_groups = defaultdict(list)
        for relation in document_semantics.relations:
            key = (relation.subject_entity.text, relation.object_entity.text)
            relation_groups[key].append(relation)
        
        for (subject, object_), relations in relation_groups.items():
            if len(relations) > 1:
                relation_types = set(r.relation_type for r in relations)
                if len(relation_types) > 1:
                    # Check for conflicting relation types
                    conflicting_types = [
                        ("works_for", "founded"),
                        ("located_in", "owns"),
                    ]
                    
                    for type1, type2 in conflicting_types:
                        if type1 in relation_types and type2 in relation_types:
                            issues.append(SemanticIssue(
                                issue_id=f"conflicting_relations_{subject}_{object_}",
                                issue_type=SemanticIssueType.CONTRADICTION,
                                severity=SemanticSeverity.MODERATE,
                                message=f"Conflicting relations between '{subject}' and '{object_}': {list(relation_types)}",
                                suggestions=[
                                    "Verify the relationship between entities",
                                    "Ensure relation types are not contradictory"
                                ]
                            ))
        
        return issues
    
    def _parse_date_entity(self, entity: Entity) -> Optional[datetime]:
        """Parse date entity to datetime object"""
        date_text = entity.text
        
        # Simple date parsing (would use more sophisticated parser in practice)
        date_patterns = [
            r'(\d{1,2})/(\d{1,2})/(\d{4})',  # MM/DD/YYYY
            r'(\d{4})-(\d{1,2})-(\d{1,2})',  # YYYY-MM-DD
            r'(\w+)\s+(\d{1,2}),?\s+(\d{4})',  # Month DD, YYYY
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, date_text)
            if match:
                try:
                    if len(match.groups()) == 3:
                        if pattern.startswith(r'(\d{1,2})/'):
                            month, day, year = match.groups()
                            return datetime(int(year), int(month), int(day))
                        elif pattern.startswith(r'(\d{4})-'):
                            year, month, day = match.groups()
                            return datetime(int(year), int(month), int(day))
                        # Could add more parsing logic for month names
                except ValueError:
                    continue
        
        return None
    
    def _calculate_consistency_metrics(
        self,
        document_semantics: DocumentSemantics,
        issues: List[SemanticIssue]
    ) -> Dict[str, Any]:
        """Calculate semantic consistency metrics"""
        total_entities = len(document_semantics.entities)
        total_relations = len(document_semantics.relations)
        
        # Count issues by type
        issue_counts = Counter(issue.issue_type for issue in issues)
        severity_counts = Counter(issue.severity for issue in issues)
        
        # Calculate consistency scores
        entity_consistency = 1.0
        if total_entities > 0:
            entity_issues = len([i for i in issues if i.issue_type in [
                SemanticIssueType.ENTITY_MISMATCH, 
                SemanticIssueType.INCONSISTENCY
            ]])
            entity_consistency = max(0.0, 1.0 - (entity_issues / total_entities))
        
        temporal_consistency = 1.0
        temporal_issues = len([i for i in issues if i.issue_type == SemanticIssueType.TEMPORAL_INCONSISTENCY])
        if temporal_issues > 0:
            temporal_consistency = max(0.0, 1.0 - (temporal_issues * 0.2))
        
        logical_consistency = 1.0
        logical_issues = len([i for i in issues if i.issue_type in [
            SemanticIssueType.CONTRADICTION,
            SemanticIssueType.LOGICAL_ERROR
        ]])
        if logical_issues > 0:
            logical_consistency = max(0.0, 1.0 - (logical_issues * 0.3))
        
        return {
            "total_entities": total_entities,
            "total_relations": total_relations,
            "total_issues": len(issues),
            "entity_consistency_score": entity_consistency,
            "temporal_consistency_score": temporal_consistency,
            "logical_consistency_score": logical_consistency,
            "issue_distribution": dict(issue_counts),
            "severity_distribution": dict(severity_counts),
            "average_entity_confidence": statistics.mean([e.confidence for e in document_semantics.entities]) if document_semantics.entities else 0.0,
            "average_relation_confidence": statistics.mean([r.confidence for r in document_semantics.relations]) if document_semantics.relations else 0.0
        }
    
    def _calculate_semantic_score(
        self,
        issues: List[SemanticIssue],
        consistency_metrics: Dict[str, Any]
    ) -> float:
        """Calculate overall semantic score"""
        # Base score from consistency metrics
        entity_score = consistency_metrics["entity_consistency_score"]
        temporal_score = consistency_metrics["temporal_consistency_score"]
        logical_score = consistency_metrics["logical_consistency_score"]
        
        # Weighted average
        base_score = (
            entity_score * 0.4 +
            temporal_score * 0.3 +
            logical_score * 0.3
        )
        
        # Apply penalties for severe issues
        critical_issues = len([i for i in issues if i.severity == SemanticSeverity.CRITICAL])
        significant_issues = len([i for i in issues if i.severity == SemanticSeverity.SIGNIFICANT])
        
        penalty = (critical_issues * 0.2) + (significant_issues * 0.1)
        
        final_score = max(0.0, base_score - penalty)
        return min(1.0, final_score)
    
    def batch_validate_semantics(
        self,
        documents: List[DocumentSemantics]
    ) -> Dict[str, Any]:
        """Validate semantics for multiple documents"""
        results = []
        
        for document in documents:
            try:
                result = self.validate_document_semantics(document)
                results.append({
                    "document_id": document.document_id,
                    "result": result
                })
            except Exception as e:
                logger.error(f"Failed to validate semantics for document {document.document_id}: {str(e)}")
                results.append({
                    "document_id": document.document_id,
                    "error": str(e)
                })
        
        # Calculate aggregate statistics
        successful_results = [r["result"] for r in results if "result" in r]
        
        if successful_results:
            scores = [r.semantic_score for r in successful_results]
            total_entities = sum(len(r.entities) for r in successful_results)
            total_relations = sum(len(r.relations) for r in successful_results)
            total_issues = sum(len(r.issues) for r in successful_results)
            
            aggregate_stats = {
                "total_documents": len(documents),
                "successful_validations": len(successful_results),
                "failed_validations": len(documents) - len(successful_results),
                "mean_semantic_score": statistics.mean(scores),
                "median_semantic_score": statistics.median(scores),
                "min_semantic_score": min(scores),
                "max_semantic_score": max(scores),
                "total_entities_found": total_entities,
                "total_relations_found": total_relations,
                "total_semantic_issues": total_issues,
                "documents_needing_attention": len([r for r in successful_results if r.semantic_score < 0.7])
            }
        else:
            aggregate_stats = {
                "total_documents": len(documents),
                "successful_validations": 0,
                "failed_validations": len(documents),
                "error": "No successful semantic validations"
            }
        
        return {
            "results": results,
            "aggregate_statistics": aggregate_stats
        }


def create_semantic_validator(
    config: Optional[Union[Dict[str, Any], SemanticValidatorConfig]] = None
) -> SemanticValidator:
    """Factory function to create semantic validator"""
    if isinstance(config, dict):
        config = SemanticValidatorConfig(**config)
    return SemanticValidator(config)


def validate_semantics_sample() -> SemanticValidationResult:
    """Generate sample semantic validation for testing"""
    validator = create_semantic_validator()
    
    # Create sample document semantics
    sample_semantics = DocumentSemantics(
        document_id="sample_semantic_doc",
        text="John Smith works for Acme Corporation. The company was founded in 2020. John has been with the company since 2019. Acme Corporation is located in New York.",
        metadata={"title": "Sample Document", "author": "Test Author"}
    )
    
    return validator.validate_document_semantics(sample_semantics)