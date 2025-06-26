"""
Bias Detection Implementation for TextNLP
Advanced bias detection and mitigation for generated text content
"""

import re
import logging
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
from collections import Counter, defaultdict
import spacy
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, Pipeline
)
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class BiasType(Enum):
    """Types of bias that can be detected"""
    GENDER = "gender"
    RACIAL = "racial"
    RELIGIOUS = "religious"
    AGE = "age"
    SOCIOECONOMIC = "socioeconomic"
    DISABILITY = "disability"
    NATIONALITY = "nationality"
    POLITICAL = "political"
    OCCUPATION = "occupation"
    APPEARANCE = "appearance"
    SEXUAL_ORIENTATION = "sexual_orientation"
    LINGUISTIC = "linguistic"


class BiasCategory(Enum):
    """Severity categories for bias"""
    SUBTLE = "subtle"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class BiasIndicator:
    """Individual bias indicator found in text"""
    bias_type: BiasType
    category: BiasCategory
    text_span: str
    start: int
    end: int
    confidence: float
    context: str
    explanation: str
    suggested_replacement: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BiasDetectionResult:
    """Result of bias detection analysis"""
    original_text: str
    bias_indicators: List[BiasIndicator]
    overall_bias_score: float
    bias_types_found: Set[BiasType]
    bias_categories: Dict[BiasCategory, int]
    detection_method: str
    processing_time: float
    debias_suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BiasDetector:
    """Advanced bias detection using multiple strategies"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        
        # Initialize models and resources
        self._initialize_models()
        self._load_bias_lexicons()
        self._load_embeddings()
        
        # Load spaCy model for linguistic analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
    
    def _initialize_models(self):
        """Initialize bias detection models"""
        self.models = {}
        self.classifiers = {}
        
        try:
            # Bias detection using specialized models
            if self.config.get("use_bias_classifier", True):
                try:
                    self.classifiers["bias"] = pipeline(
                        "text-classification",
                        model="unitary/unbiased-toxic-roberta",
                        device=0 if torch.cuda.is_available() else -1
                    )
                    logger.info("Loaded bias classification model")
                except Exception as e:
                    logger.warning(f"Failed to load bias classifier: {e}")
            
            # Gender bias detection
            if self.config.get("use_gender_bias", True):
                try:
                    self.classifiers["gender_bias"] = pipeline(
                        "text-classification",
                        model="d4data/bias-detection-model",
                        device=0 if torch.cuda.is_available() else -1
                    )
                    logger.info("Loaded gender bias detector")
                except Exception as e:
                    logger.warning(f"Failed to load gender bias detector: {e}")
            
            # Sentiment analysis for context
            if self.config.get("use_sentiment", True):
                try:
                    self.classifiers["sentiment"] = pipeline(
                        "sentiment-analysis",
                        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                        device=0 if torch.cuda.is_available() else -1
                    )
                    logger.info("Loaded sentiment analyzer")
                except Exception as e:
                    logger.warning(f"Failed to load sentiment analyzer: {e}")
                    
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def _load_bias_lexicons(self):
        """Load bias-related word lexicons and patterns"""
        self.bias_lexicons = {
            BiasType.GENDER: {
                "gendered_terms": {
                    "male": ["he", "him", "his", "man", "men", "male", "boy", "boys", "guy", "guys", 
                            "gentleman", "gentlemen", "father", "dad", "son", "brother", "husband",
                            "boyfriend", "uncle", "nephew", "grandfather", "grandson"],
                    "female": ["she", "her", "hers", "woman", "women", "female", "girl", "girls", "lady", "ladies",
                              "mother", "mom", "daughter", "sister", "wife", "girlfriend", "aunt", "niece",
                              "grandmother", "granddaughter"],
                    "neutral": ["they", "them", "their", "person", "people", "individual", "someone",
                               "parent", "child", "sibling", "spouse", "partner", "relative"]
                },
                "biased_patterns": [
                    re.compile(r'\b(women|girls?)\s+(are|can\'t|cannot|shouldn\'t|won\'t)\b', re.IGNORECASE),
                    re.compile(r'\b(men|boys?)\s+(are|always|never|should)\b', re.IGNORECASE),
                    re.compile(r'\blike\s+a\s+(girl|woman)\b', re.IGNORECASE),
                    re.compile(r'\bman\s+up\b', re.IGNORECASE),
                    re.compile(r'\b(bossy|emotional|hysterical)\s+(woman|girl)\b', re.IGNORECASE)
                ]
            },
            BiasType.RACIAL: {
                "biased_patterns": [
                    re.compile(r'\ball\s+([a-z]+)\s+people\s+are\b', re.IGNORECASE),
                    re.compile(r'\b(typical|stereotypical)\s+([a-z]+)\b', re.IGNORECASE),
                    re.compile(r'\bacts?\s+like\s+a\s+([a-z]+)\b', re.IGNORECASE)
                ],
                "identity_terms": [
                    "african american", "black", "white", "asian", "hispanic", "latino", "latina",
                    "native american", "indigenous", "middle eastern", "arab", "jewish", "muslim"
                ]
            },
            BiasType.AGE: {
                "biased_patterns": [
                    re.compile(r'\b(too\s+)?(old|young)\s+(for|to)\b', re.IGNORECASE),
                    re.compile(r'\b(millennials?|boomers?|gen\s*z)\s+(are|always|never)\b', re.IGNORECASE),
                    re.compile(r'\b(senior|elderly)\s+(person|people)\s+(can\'t|cannot)\b', re.IGNORECASE)
                ],
                "age_terms": ["young", "old", "senior", "elderly", "millennial", "boomer", "teenager"]
            },
            BiasType.DISABILITY: {
                "biased_patterns": [
                    re.compile(r'\b(suffers?\s+from|victim\s+of|afflicted\s+with)\b', re.IGNORECASE),
                    re.compile(r'\b(disabled|handicapped)\s+person\b', re.IGNORECASE),
                    re.compile(r'\bnormal\s+people\b', re.IGNORECASE)
                ],
                "preferred_terms": {
                    "suffers from": "has",
                    "victim of": "person with",
                    "afflicted with": "person with",
                    "disabled person": "person with a disability",
                    "handicapped": "person with a disability",
                    "normal people": "people without disabilities"
                }
            },
            BiasType.SOCIOECONOMIC: {
                "biased_patterns": [
                    re.compile(r'\b(poor|rich)\s+people\s+(are|always|never)\b', re.IGNORECASE),
                    re.compile(r'\b(ghetto|trailer\s+trash|white\s+trash)\b', re.IGNORECASE),
                    re.compile(r'\b(welfare\s+queen|food\s+stamp)\b', re.IGNORECASE)
                ]
            },
            BiasType.RELIGIOUS: {
                "biased_patterns": [
                    re.compile(r'\ball\s+(muslims?|christians?|jews?|hindus?|buddhists?)\s+are\b', re.IGNORECASE),
                    re.compile(r'\b(typical|stereotypical)\s+(muslim|christian|jewish|hindu|buddhist)\b', re.IGNORECASE)
                ]
            },
            BiasType.SEXUAL_ORIENTATION: {
                "biased_patterns": [
                    re.compile(r'\b(gay|lesbian|bisexual|transgender)\s+(agenda|lifestyle)\b', re.IGNORECASE),
                    re.compile(r'\bchoose\s+to\s+be\s+(gay|lesbian|bisexual)\b', re.IGNORECASE),
                    re.compile(r'\bthat\'s\s+so\s+gay\b', re.IGNORECASE)
                ]
            }
        }
    
    def _load_embeddings(self):
        """Load word embeddings for semantic bias detection"""
        # This would typically load pre-trained embeddings
        # For now, we'll use a simplified approach
        self.embedding_bias_pairs = {
            # Gender occupation bias pairs
            ("programmer", "he"): ("programmer", "she"),
            ("nurse", "she"): ("nurse", "he"),
            ("doctor", "he"): ("doctor", "she"),
            ("teacher", "she"): ("teacher", "he"),
            ("engineer", "he"): ("engineer", "she"),
            ("secretary", "she"): ("secretary", "he"),
            
            # Racial bias pairs
            ("intelligent", "white"): ("intelligent", "black"),
            ("articulate", "white"): ("articulate", "black"),
            ("professional", "white"): ("professional", "black"),
        }
    
    async def detect_bias(self, text: str, context: str = "") -> BiasDetectionResult:
        """Detect bias in text using multiple detection strategies"""
        start_time = asyncio.get_event_loop().time()
        
        bias_indicators = []
        detection_methods = []
        
        # Method 1: Lexicon-based detection
        try:
            lexicon_indicators = await self._detect_with_lexicons(text)
            bias_indicators.extend(lexicon_indicators)
            detection_methods.append("lexicon")
        except Exception as e:
            logger.error(f"Lexicon-based detection failed: {e}")
        
        # Method 2: Pattern-based detection
        try:
            pattern_indicators = await self._detect_with_patterns(text)
            bias_indicators.extend(pattern_indicators)
            detection_methods.append("patterns")
        except Exception as e:
            logger.error(f"Pattern-based detection failed: {e}")
        
        # Method 3: ML model-based detection
        try:
            model_indicators = await self._detect_with_models(text)
            bias_indicators.extend(model_indicators)
            detection_methods.append("models")
        except Exception as e:
            logger.error(f"Model-based detection failed: {e}")
        
        # Method 4: Linguistic analysis
        try:
            linguistic_indicators = await self._detect_with_linguistics(text)
            bias_indicators.extend(linguistic_indicators)
            detection_methods.append("linguistics")
        except Exception as e:
            logger.error(f"Linguistic analysis failed: {e}")
        
        # Method 5: Semantic bias detection
        try:
            semantic_indicators = await self._detect_semantic_bias(text)
            bias_indicators.extend(semantic_indicators)
            detection_methods.append("semantic")
        except Exception as e:
            logger.error(f"Semantic bias detection failed: {e}")
        
        # Remove duplicates and filter by confidence
        bias_indicators = self._deduplicate_indicators(bias_indicators)
        bias_indicators = [i for i in bias_indicators if i.confidence >= self.confidence_threshold]
        
        # Sort by position
        bias_indicators.sort(key=lambda x: x.start)
        
        # Calculate overall bias score
        overall_bias_score = self._calculate_overall_bias_score(bias_indicators)
        
        # Categorize bias types and severity
        bias_types_found = set(indicator.bias_type for indicator in bias_indicators)
        bias_categories = self._categorize_bias_severity(bias_indicators)
        
        # Generate debiasing suggestions
        debias_suggestions = self._generate_debias_suggestions(bias_indicators, text)
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return BiasDetectionResult(
            original_text=text,
            bias_indicators=bias_indicators,
            overall_bias_score=overall_bias_score,
            bias_types_found=bias_types_found,
            bias_categories=bias_categories,
            detection_method="+".join(detection_methods),
            processing_time=processing_time,
            debias_suggestions=debias_suggestions,
            metadata={
                "methods_used": detection_methods,
                "threshold": self.confidence_threshold,
                "total_indicators": len(bias_indicators)
            }
        )
    
    async def _detect_with_lexicons(self, text: str) -> List[BiasIndicator]:
        """Detect bias using lexicon-based approach"""
        indicators = []
        text_lower = text.lower()
        
        for bias_type, lexicon in self.bias_lexicons.items():
            # Check for biased terms
            if "biased_patterns" in lexicon:
                for pattern in lexicon["biased_patterns"]:
                    for match in pattern.finditer(text):
                        confidence = 0.7  # Base confidence for pattern matches
                        
                        # Analyze context for confidence adjustment
                        context_start = max(0, match.start() - 50)
                        context_end = min(len(text), match.end() + 50)
                        context = text[context_start:context_end]
                        
                        indicator = BiasIndicator(
                            bias_type=bias_type,
                            category=self._determine_bias_category(match.group(), bias_type),
                            text_span=match.group(),
                            start=match.start(),
                            end=match.end(),
                            confidence=confidence,
                            context=context,
                            explanation=f"Potentially biased language pattern detected for {bias_type.value}",
                            metadata={"detector": "lexicon", "pattern": pattern.pattern}
                        )
                        indicators.append(indicator)
            
            # Check for identity terms usage
            if "identity_terms" in lexicon:
                for term in lexicon["identity_terms"]:
                    term_pattern = re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE)
                    for match in term_pattern.finditer(text):
                        # Analyze surrounding context for bias
                        context_window = 100
                        context_start = max(0, match.start() - context_window)
                        context_end = min(len(text), match.end() + context_window)
                        context = text[context_start:context_end]
                        
                        # Check for negative associations
                        if self._has_negative_context(context, term):
                            indicator = BiasIndicator(
                                bias_type=bias_type,
                                category=BiasCategory.MODERATE,
                                text_span=match.group(),
                                start=match.start(),
                                end=match.end(),
                                confidence=0.6,
                                context=context,
                                explanation=f"Identity term '{term}' used in potentially biased context",
                                metadata={"detector": "lexicon", "term": term}
                            )
                            indicators.append(indicator)
        
        return indicators
    
    async def _detect_with_patterns(self, text: str) -> List[BiasIndicator]:
        """Detect bias using advanced pattern matching"""
        indicators = []
        
        # Gendered assumption patterns
        gendered_patterns = [
            (r'\b(he|she)\s+must\s+be\s+a\s+(\w+)', "Gender assumption based on role"),
            (r'\b(typical|like\s+a)\s+(man|woman|girl|boy)', "Gender stereotyping"),
            (r'\b(men|women)\s+(are|always|never|should|shouldn\'t)', "Gender generalization"),
            (r'\b(act|behave|think)\s+like\s+a\s+(man|woman)', "Gender role enforcement")
        ]
        
        for pattern_str, explanation in gendered_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            for match in pattern.finditer(text):
                context_start = max(0, match.start() - 50)
                context_end = min(len(text), match.end() + 50)
                context = text[context_start:context_end]
                
                indicator = BiasIndicator(
                    bias_type=BiasType.GENDER,
                    category=BiasCategory.MODERATE,
                    text_span=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.75,
                    context=context,
                    explanation=explanation,
                    metadata={"detector": "patterns", "pattern_type": "gender"}
                )
                indicators.append(indicator)
        
        # Racial bias patterns
        racial_patterns = [
            (r'\ball\s+(\w+)\s+people\s+(are|have|do)', "Racial generalization"),
            (r'\b(acts?|talks?|looks?)\s+like\s+a\s+(\w+)', "Racial stereotyping"),
            (r'\b(you\'re\s+)?(pretty|smart|articulate)\s+for\s+a\s+(\w+)', "Backhanded compliment")
        ]
        
        for pattern_str, explanation in racial_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            for match in pattern.finditer(text):
                context_start = max(0, match.start() - 50)
                context_end = min(len(text), match.end() + 50)
                context = text[context_start:context_end]
                
                indicator = BiasIndicator(
                    bias_type=BiasType.RACIAL,
                    category=BiasCategory.SEVERE,
                    text_span=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8,
                    context=context,
                    explanation=explanation,
                    metadata={"detector": "patterns", "pattern_type": "racial"}
                )
                indicators.append(indicator)
        
        return indicators
    
    async def _detect_with_models(self, text: str) -> List[BiasIndicator]:
        """Detect bias using machine learning models"""
        indicators = []
        
        # General bias classification
        if "bias" in self.classifiers:
            try:
                result = self.classifiers["bias"](text)
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]
                
                label = result.get("label", "").lower()
                score = result.get("score", 0)
                
                if "bias" in label and score > self.confidence_threshold:
                    indicator = BiasIndicator(
                        bias_type=BiasType.GENDER,  # Default, would need more specific classification
                        category=BiasCategory.MODERATE,
                        text_span=text[:100] + "..." if len(text) > 100 else text,
                        start=0,
                        end=len(text),
                        confidence=score,
                        context=text,
                        explanation=f"ML model detected potential bias (label: {label})",
                        metadata={"detector": "ml_model", "model": "unitary/unbiased-toxic-roberta"}
                    )
                    indicators.append(indicator)
            except Exception as e:
                logger.error(f"Bias classification failed: {e}")
        
        # Gender bias detection
        if "gender_bias" in self.classifiers:
            try:
                result = self.classifiers["gender_bias"](text)
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]
                
                label = result.get("label", "").lower()
                score = result.get("score", 0)
                
                if "bias" in label and score > self.confidence_threshold:
                    indicator = BiasIndicator(
                        bias_type=BiasType.GENDER,
                        category=BiasCategory.MODERATE,
                        text_span=text[:100] + "..." if len(text) > 100 else text,
                        start=0,
                        end=len(text),
                        confidence=score,
                        context=text,
                        explanation=f"Gender bias detected by ML model (confidence: {score:.2f})",
                        metadata={"detector": "ml_model", "model": "d4data/bias-detection-model"}
                    )
                    indicators.append(indicator)
            except Exception as e:
                logger.error(f"Gender bias classification failed: {e}")
        
        return indicators
    
    async def _detect_with_linguistics(self, text: str) -> List[BiasIndicator]:
        """Detect bias using linguistic analysis"""
        indicators = []
        
        try:
            doc = self.nlp(text)
            
            # Analyze pronoun usage patterns
            pronouns = {"he": "male", "she": "female", "him": "male", "her": "female", 
                       "his": "male", "hers": "female"}
            
            pronoun_contexts = defaultdict(list)
            
            for token in doc:
                if token.text.lower() in pronouns:
                    # Get surrounding context
                    start_idx = max(0, token.i - 5)
                    end_idx = min(len(doc), token.i + 6)
                    context_tokens = doc[start_idx:end_idx]
                    context = " ".join([t.text for t in context_tokens])
                    
                    gender = pronouns[token.text.lower()]
                    pronoun_contexts[gender].append({
                        "pronoun": token.text,
                        "context": context,
                        "start": token.idx,
                        "end": token.idx + len(token.text)
                    })
            
            # Check for gendered role associations
            roles = ["doctor", "nurse", "teacher", "engineer", "secretary", "manager", 
                    "assistant", "CEO", "programmer", "designer"]
            
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG"]:
                    sent = ent.sent
                    sent_text = sent.text.lower()
                    
                    for role in roles:
                        if role in sent_text:
                            # Check for gendered pronouns in same sentence
                            for token in sent:
                                if token.text.lower() in pronouns:
                                    gender = pronouns[token.text.lower()]
                                    
                                    # Check if this is a stereotypical association
                                    if self._is_stereotypical_association(role, gender):
                                        indicator = BiasIndicator(
                                            bias_type=BiasType.GENDER,
                                            category=BiasCategory.SUBTLE,
                                            text_span=sent.text,
                                            start=sent.start_char,
                                            end=sent.end_char,
                                            confidence=0.6,
                                            context=sent.text,
                                            explanation=f"Potential gender bias in role association: {role} + {gender} pronoun",
                                            metadata={"detector": "linguistics", "role": role, "gender": gender}
                                        )
                                        indicators.append(indicator)
            
            # Analyze adjective usage patterns
            adjectives_by_gender = {"male": [], "female": []}
            
            for sent in doc.sents:
                sent_pronouns = [token for token in sent if token.text.lower() in pronouns]
                sent_adjectives = [token for token in sent if token.pos_ == "ADJ"]
                
                if sent_pronouns and sent_adjectives:
                    for pronoun in sent_pronouns:
                        gender = pronouns[pronoun.text.lower()]
                        for adj in sent_adjectives:
                            adjectives_by_gender[gender].append(adj.text.lower())
            
            # Check for biased adjective distributions
            if len(adjectives_by_gender["male"]) > 0 and len(adjectives_by_gender["female"]) > 0:
                male_adj_counts = Counter(adjectives_by_gender["male"])
                female_adj_counts = Counter(adjectives_by_gender["female"])
                
                # Look for highly skewed adjective usage
                biased_adjectives = self._find_biased_adjectives(male_adj_counts, female_adj_counts)
                
                for adj, bias_info in biased_adjectives.items():
                    # Find instances in text
                    adj_pattern = re.compile(rf'\b{re.escape(adj)}\b', re.IGNORECASE)
                    for match in adj_pattern.finditer(text):
                        indicator = BiasIndicator(
                            bias_type=BiasType.GENDER,
                            category=BiasCategory.SUBTLE,
                            text_span=match.group(),
                            start=match.start(),
                            end=match.end(),
                            confidence=0.5,
                            context=text[max(0, match.start()-50):match.end()+50],
                            explanation=f"Adjective '{adj}' shows gender bias pattern: {bias_info}",
                            metadata={"detector": "linguistics", "adjective": adj, "bias_info": bias_info}
                        )
                        indicators.append(indicator)
        
        except Exception as e:
            logger.error(f"Linguistic analysis failed: {e}")
        
        return indicators
    
    async def _detect_semantic_bias(self, text: str) -> List[BiasIndicator]:
        """Detect bias using semantic analysis"""
        indicators = []
        
        # This is a simplified implementation
        # In practice, you would use word embeddings to detect semantic bias
        
        # Check for implicit associations
        implicit_bias_patterns = [
            ("assertive", "man", "bossy", "woman"),
            ("confident", "man", "arrogant", "woman"),
            ("ambitious", "man", "pushy", "woman"),
            ("direct", "man", "bitchy", "woman")
        ]
        
        text_lower = text.lower()
        for pos_male, male_term, neg_female, female_term in implicit_bias_patterns:
            if (pos_male in text_lower and male_term in text_lower) or \
               (neg_female in text_lower and female_term in text_lower):
                
                # Find the specific occurrence
                if pos_male in text_lower and male_term in text_lower:
                    target_words = [pos_male, male_term]
                    bias_desc = f"Positive trait '{pos_male}' associated with '{male_term}'"
                else:
                    target_words = [neg_female, female_term]
                    bias_desc = f"Negative trait '{neg_female}' associated with '{female_term}'"
                
                # Find position of the bias
                for word in target_words:
                    word_pattern = re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)
                    for match in word_pattern.finditer(text):
                        indicator = BiasIndicator(
                            bias_type=BiasType.GENDER,
                            category=BiasCategory.SUBTLE,
                            text_span=match.group(),
                            start=match.start(),
                            end=match.end(),
                            confidence=0.4,
                            context=text[max(0, match.start()-50):match.end()+50],
                            explanation=f"Semantic bias detected: {bias_desc}",
                            metadata={"detector": "semantic", "pattern": target_words}
                        )
                        indicators.append(indicator)
                        break  # Only add one indicator per pattern
        
        return indicators
    
    def _has_negative_context(self, context: str, term: str) -> bool:
        """Check if identity term is used in negative context"""
        negative_words = [
            "bad", "terrible", "awful", "horrible", "disgusting", "wrong", "evil",
            "dangerous", "criminal", "lazy", "stupid", "ignorant", "primitive",
            "uncivilized", "savage", "aggressive", "violent", "threatening"
        ]
        
        context_lower = context.lower()
        for word in negative_words:
            if word in context_lower:
                return True
        return False
    
    def _determine_bias_category(self, text: str, bias_type: BiasType) -> BiasCategory:
        """Determine the severity category of detected bias"""
        severe_indicators = ["hate", "disgusting", "inferior", "superior", "subhuman", "savage"]
        moderate_indicators = ["typical", "always", "never", "should", "shouldn't", "can't"]
        
        text_lower = text.lower()
        
        for indicator in severe_indicators:
            if indicator in text_lower:
                return BiasCategory.SEVERE
        
        for indicator in moderate_indicators:
            if indicator in text_lower:
                return BiasCategory.MODERATE
        
        return BiasCategory.SUBTLE
    
    def _is_stereotypical_association(self, role: str, gender: str) -> bool:
        """Check if role-gender association follows stereotypes"""
        stereotypical_male = ["engineer", "programmer", "CEO", "manager", "doctor"]
        stereotypical_female = ["nurse", "teacher", "secretary", "assistant"]
        
        if gender == "male" and role in stereotypical_male:
            return True
        if gender == "female" and role in stereotypical_female:
            return True
        
        return False
    
    def _find_biased_adjectives(self, male_adj: Counter, female_adj: Counter) -> Dict[str, str]:
        """Find adjectives that show gender bias"""
        biased_adjectives = {}
        
        # Compare adjective usage between genders
        all_adjectives = set(male_adj.keys()) | set(female_adj.keys())
        
        for adj in all_adjectives:
            male_count = male_adj.get(adj, 0)
            female_count = female_adj.get(adj, 0)
            total_count = male_count + female_count
            
            if total_count >= 2:  # Only consider adjectives used multiple times
                male_ratio = male_count / total_count
                female_ratio = female_count / total_count
                
                # Check for significant skew (>80% usage by one gender)
                if male_ratio > 0.8:
                    biased_adjectives[adj] = f"Used predominantly with male pronouns ({male_ratio:.1%})"
                elif female_ratio > 0.8:
                    biased_adjectives[adj] = f"Used predominantly with female pronouns ({female_ratio:.1%})"
        
        return biased_adjectives
    
    def _deduplicate_indicators(self, indicators: List[BiasIndicator]) -> List[BiasIndicator]:
        """Remove duplicate and overlapping bias indicators"""
        if not indicators:
            return indicators
        
        # Sort by start position
        indicators.sort(key=lambda x: x.start)
        
        deduplicated = []
        for indicator in indicators:
            # Check for overlaps with existing indicators
            overlapping = False
            for existing in deduplicated:
                # Check if ranges overlap
                if (indicator.start < existing.end and indicator.end > existing.start):
                    # Keep the indicator with higher confidence
                    if indicator.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        deduplicated.append(indicator)
                    overlapping = True
                    break
            
            if not overlapping:
                deduplicated.append(indicator)
        
        return deduplicated
    
    def _calculate_overall_bias_score(self, indicators: List[BiasIndicator]) -> float:
        """Calculate overall bias score for the text"""
        if not indicators:
            return 0.0
        
        # Weight by severity and confidence
        total_score = 0.0
        for indicator in indicators:
            severity_weight = {
                BiasCategory.SUBTLE: 0.3,
                BiasCategory.MODERATE: 0.6,
                BiasCategory.SEVERE: 1.0
            }
            
            score = indicator.confidence * severity_weight[indicator.category]
            total_score += score
        
        # Normalize by number of indicators and text length
        normalized_score = min(1.0, total_score / max(1, len(indicators)))
        return normalized_score
    
    def _categorize_bias_severity(self, indicators: List[BiasIndicator]) -> Dict[BiasCategory, int]:
        """Categorize indicators by severity"""
        categories = {BiasCategory.SUBTLE: 0, BiasCategory.MODERATE: 0, BiasCategory.SEVERE: 0}
        
        for indicator in indicators:
            categories[indicator.category] += 1
        
        return categories
    
    def _generate_debias_suggestions(self, indicators: List[BiasIndicator], text: str) -> List[str]:
        """Generate suggestions for reducing bias"""
        suggestions = []
        
        bias_types_found = set(indicator.bias_type for indicator in indicators)
        
        if BiasType.GENDER in bias_types_found:
            suggestions.extend([
                "Use gender-neutral pronouns (they/them) when possible",
                "Avoid gendered assumptions about roles and professions",
                "Use parallel language when describing people of different genders"
            ])
        
        if BiasType.RACIAL in bias_types_found:
            suggestions.extend([
                "Avoid generalizations about racial or ethnic groups",
                "Focus on individual characteristics rather than group stereotypes",
                "Use person-first language"
            ])
        
        if BiasType.AGE in bias_types_found:
            suggestions.extend([
                "Avoid ageist assumptions about capabilities",
                "Use respectful terminology for different age groups"
            ])
        
        if BiasType.DISABILITY in bias_types_found:
            suggestions.extend([
                "Use person-first language (e.g., 'person with a disability')",
                "Avoid medical or victim language",
                "Focus on abilities rather than limitations"
            ])
        
        # Add specific replacement suggestions
        for indicator in indicators:
            if indicator.suggested_replacement:
                suggestions.append(f"Replace '{indicator.text_span}' with '{indicator.suggested_replacement}'")
        
        return list(set(suggestions))  # Remove duplicates
    
    async def batch_detect(self, texts: List[str]) -> List[BiasDetectionResult]:
        """Detect bias in multiple texts concurrently"""
        tasks = [self.detect_bias(text) for text in texts]
        return await asyncio.gather(*tasks)
    
    def create_bias_report(self, results: List[BiasDetectionResult]) -> Dict[str, Any]:
        """Create a comprehensive bias analysis report"""
        total_texts = len(results)
        biased_texts = sum(1 for r in results if r.overall_bias_score > 0.3)
        
        # Bias type distribution
        bias_type_counts = {}
        for result in results:
            for bias_type in result.bias_types_found:
                bias_type_counts[bias_type.value] = bias_type_counts.get(bias_type.value, 0) + 1
        
        # Severity distribution
        severity_distribution = {cat.value: 0 for cat in BiasCategory}
        for result in results:
            for category, count in result.bias_categories.items():
                severity_distribution[category.value] += count
        
        return {
            "summary": {
                "total_texts_analyzed": total_texts,
                "texts_with_bias": biased_texts,
                "bias_detection_rate": biased_texts / total_texts if total_texts > 0 else 0,
                "average_bias_score": sum(r.overall_bias_score for r in results) / total_texts if total_texts > 0 else 0
            },
            "bias_type_distribution": bias_type_counts,
            "severity_distribution": severity_distribution,
            "performance": {
                "average_processing_time": sum(r.processing_time for r in results) / total_texts if total_texts > 0 else 0,
                "total_processing_time": sum(r.processing_time for r in results)
            },
            "timestamp": datetime.utcnow().isoformat()
        }


class BiasDetectionService:
    """Service wrapper for bias detection with configuration and caching"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detector = BiasDetector(config)
        self.cache = {}
        self.cache_enabled = config.get("enable_cache", True)
        self.max_cache_size = config.get("max_cache_size", 1000)
    
    async def detect_and_suggest(self, text: str, cache_key: Optional[str] = None) -> BiasDetectionResult:
        """Detect bias and provide debiasing suggestions"""
        if self.cache_enabled and cache_key:
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        result = await self.detector.detect_bias(text)
        
        if self.cache_enabled and cache_key:
            # Simple LRU cache implementation
            if len(self.cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[cache_key] = result
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "cache_size": len(self.cache),
            "cache_enabled": self.cache_enabled,
            "config": self.config
        }


# Example usage
if __name__ == "__main__":
    async def example():
        # Sample texts with various types of bias
        test_texts = [
            "Women are naturally better at nurturing children.",  # Gender bias
            "He must be a great programmer because he's Asian.",  # Racial bias
            "She's pretty bossy for a woman in management.",  # Gender bias
            "All millennials are lazy and entitled.",  # Age bias
            "This is a neutral statement about technology.",  # Clean
        ]
        
        # Initialize detector
        config = {
            "use_bias_classifier": True,
            "use_gender_bias": True,
            "confidence_threshold": 0.4
        }
        detector = BiasDetector(config)
        
        # Detect bias in texts
        for text in test_texts:
            result = await detector.detect_bias(text)
            print(f"\nText: {text}")
            print(f"Bias score: {result.overall_bias_score:.3f}")
            print(f"Bias types: {[bt.value for bt in result.bias_types_found]}")
            
            for indicator in result.bias_indicators:
                print(f"- {indicator.bias_type.value} ({indicator.category.value}): {indicator.text_span}")
                print(f"  Explanation: {indicator.explanation}")
    
    # Run example
    # asyncio.run(example())