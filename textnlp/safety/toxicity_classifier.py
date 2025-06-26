"""
Toxicity Classification Implementation for TextNLP
Advanced toxicity detection and classification for generated text content
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, Pipeline
)
from detoxify import Detoxify
import openai
from googleapiclient import discovery
import json

logger = logging.getLogger(__name__)


class ToxicityType(Enum):
    """Types of toxicity that can be detected"""
    TOXIC = "toxic"
    SEVERE_TOXIC = "severe_toxic"
    OBSCENE = "obscene"
    THREAT = "threat"
    INSULT = "insult"
    IDENTITY_HATE = "identity_hate"
    HARASSMENT = "harassment"
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    SEXUAL = "sexual"
    PROFANITY = "profanity"
    SPAM = "spam"
    SELF_HARM = "self_harm"
    DISCRIMINATION = "discrimination"


@dataclass
class ToxicityScore:
    """Individual toxicity score for a specific type"""
    toxicity_type: ToxicityType
    score: float
    confidence: float
    threshold: float
    is_toxic: bool
    evidence: List[str] = field(default_factory=list)


@dataclass
class ToxicityResult:
    """Result of toxicity classification"""
    text: str
    overall_toxicity: float
    is_toxic: bool
    toxicity_scores: List[ToxicityScore]
    max_toxicity_type: Optional[ToxicityType]
    detection_method: str
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToxicityClassifier:
    """Advanced toxicity classification using multiple models and APIs"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.threshold = self.config.get("threshold", 0.5)
        self.models = {}
        self.classifiers = {}
        
        # Initialize models based on configuration
        self._initialize_models()
        
        # Toxicity keywords and patterns
        self.toxic_patterns = self._load_toxic_patterns()
        
        # API configurations
        self.openai_client = None
        self.perspective_api = None
        
        if self.config.get("openai_api_key"):
            openai.api_key = self.config["openai_api_key"]
            self.openai_client = openai
        
        if self.config.get("perspective_api_key"):
            self.perspective_api = discovery.build(
                "commentanalyzer",
                "v1alpha1",
                developerKey=self.config["perspective_api_key"],
                discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                static_discovery=False,
            )
    
    def _initialize_models(self):
        """Initialize toxicity detection models"""
        try:
            # Detoxify - fast and accurate
            if self.config.get("use_detoxify", True):
                self.models["detoxify"] = Detoxify("original")
                logger.info("Loaded Detoxify model")
            
            # HuggingFace transformers models
            if self.config.get("use_unitary_toxic", True):
                try:
                    model_name = "unitary/toxic-bert"
                    self.models["toxic_bert_tokenizer"] = AutoTokenizer.from_pretrained(model_name)
                    self.models["toxic_bert"] = AutoModelForSequenceClassification.from_pretrained(model_name)
                    logger.info("Loaded Toxic-BERT model")
                except Exception as e:
                    logger.warning(f"Failed to load Toxic-BERT: {e}")
            
            # Content safety classifier
            if self.config.get("use_content_safety", True):
                try:
                    self.classifiers["content_safety"] = pipeline(
                        "text-classification",
                        model="martin-ha/toxic-comment-model",
                        device=0 if torch.cuda.is_available() else -1
                    )
                    logger.info("Loaded content safety classifier")
                except Exception as e:
                    logger.warning(f"Failed to load content safety classifier: {e}")
            
            # Hate speech detection
            if self.config.get("use_hate_speech", True):
                try:
                    self.classifiers["hate_speech"] = pipeline(
                        "text-classification",
                        model="cardiffnlp/twitter-roberta-base-hate-latest",
                        device=0 if torch.cuda.is_available() else -1
                    )
                    logger.info("Loaded hate speech detector")
                except Exception as e:
                    logger.warning(f"Failed to load hate speech detector: {e}")
                    
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def _load_toxic_patterns(self) -> Dict[ToxicityType, List[re.Pattern]]:
        """Load toxic keyword patterns and regex"""
        patterns = {
            ToxicityType.PROFANITY: [
                re.compile(r'\b(damn|hell|shit|fuck|bitch|ass|crap)\b', re.IGNORECASE),
                re.compile(r'\b(wtf|omfg|stfu|gtfo)\b', re.IGNORECASE),
            ],
            ToxicityType.INSULT: [
                re.compile(r'\b(idiot|stupid|dumb|moron|retard|loser)\b', re.IGNORECASE),
                re.compile(r'\byou\s+(are|r)\s+(so\s+)?(stupid|dumb|ugly)\b', re.IGNORECASE),
            ],
            ToxicityType.THREAT: [
                re.compile(r'\b(kill|murder|die|death|hurt|harm|attack)\b.*\byou\b', re.IGNORECASE),
                re.compile(r'\bi\s+(will|gonna|going\s+to)\s+(kill|hurt|destroy)\b', re.IGNORECASE),
            ],
            ToxicityType.HATE_SPEECH: [
                re.compile(r'\b(nazi|hitler|genocide|ethnic\s+cleansing)\b', re.IGNORECASE),
                # Note: More sensitive patterns would be added in production
            ],
            ToxicityType.SEXUAL: [
                re.compile(r'\b(sex|sexual|porn|nude|naked|orgasm)\b', re.IGNORECASE),
                # Note: This is a simplified example
            ],
            ToxicityType.DISCRIMINATION: [
                re.compile(r'\ball\s+(blacks|whites|jews|muslims|christians)\s+are\b', re.IGNORECASE),
                re.compile(r'\bi\s+hate\s+(black|white|asian|hispanic)\s+people\b', re.IGNORECASE),
            ]
        }
        return patterns
    
    async def classify_toxicity(self, text: str, 
                              methods: Optional[List[str]] = None) -> ToxicityResult:
        """Classify toxicity using multiple detection methods"""
        start_time = asyncio.get_event_loop().time()
        
        if methods is None:
            methods = ["detoxify", "transformers", "patterns", "perspective"]
            if self.openai_client:
                methods.append("openai")
        
        all_scores = []
        detection_methods = []
        
        # Method 1: Detoxify
        if "detoxify" in methods and "detoxify" in self.models:
            try:
                scores = await self._classify_with_detoxify(text)
                all_scores.extend(scores)
                detection_methods.append("detoxify")
            except Exception as e:
                logger.error(f"Detoxify classification failed: {e}")
        
        # Method 2: Transformer models
        if "transformers" in methods:
            try:
                scores = await self._classify_with_transformers(text)
                all_scores.extend(scores)
                detection_methods.append("transformers")
            except Exception as e:
                logger.error(f"Transformer classification failed: {e}")
        
        # Method 3: Pattern matching
        if "patterns" in methods:
            try:
                scores = await self._classify_with_patterns(text)
                all_scores.extend(scores)
                detection_methods.append("patterns")
            except Exception as e:
                logger.error(f"Pattern classification failed: {e}")
        
        # Method 4: Google Perspective API
        if "perspective" in methods and self.perspective_api:
            try:
                scores = await self._classify_with_perspective(text)
                all_scores.extend(scores)
                detection_methods.append("perspective")
            except Exception as e:
                logger.error(f"Perspective API classification failed: {e}")
        
        # Method 5: OpenAI Moderation API
        if "openai" in methods and self.openai_client:
            try:
                scores = await self._classify_with_openai(text)
                all_scores.extend(scores)
                detection_methods.append("openai")
            except Exception as e:
                logger.error(f"OpenAI classification failed: {e}")
        
        # Combine and deduplicate scores
        combined_scores = self._combine_scores(all_scores)
        
        # Calculate overall toxicity
        overall_toxicity = max((score.score for score in combined_scores), default=0.0)
        is_toxic = overall_toxicity > self.threshold
        
        # Find max toxicity type
        max_toxicity_type = None
        if combined_scores:
            max_score_obj = max(combined_scores, key=lambda x: x.score)
            max_toxicity_type = max_score_obj.toxicity_type
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return ToxicityResult(
            text=text,
            overall_toxicity=overall_toxicity,
            is_toxic=is_toxic,
            toxicity_scores=combined_scores,
            max_toxicity_type=max_toxicity_type,
            detection_method="+".join(detection_methods),
            processing_time=processing_time,
            metadata={
                "methods_used": detection_methods,
                "threshold": self.threshold,
                "num_scores": len(all_scores)
            }
        )
    
    async def _classify_with_detoxify(self, text: str) -> List[ToxicityScore]:
        """Classify using Detoxify model"""
        model = self.models["detoxify"]
        results = model.predict(text)
        
        scores = []
        type_mapping = {
            "toxicity": ToxicityType.TOXIC,
            "severe_toxicity": ToxicityType.SEVERE_TOXIC,
            "obscene": ToxicityType.OBSCENE,
            "threat": ToxicityType.THREAT,
            "insult": ToxicityType.INSULT,
            "identity_attack": ToxicityType.IDENTITY_HATE
        }
        
        for key, score in results.items():
            if key in type_mapping:
                toxicity_type = type_mapping[key]
                threshold = self.config.get(f"{key}_threshold", self.threshold)
                
                toxicity_score = ToxicityScore(
                    toxicity_type=toxicity_type,
                    score=float(score),
                    confidence=0.9,  # Detoxify is generally reliable
                    threshold=threshold,
                    is_toxic=score > threshold
                )
                scores.append(toxicity_score)
        
        return scores
    
    async def _classify_with_transformers(self, text: str) -> List[ToxicityScore]:
        """Classify using HuggingFace transformer models"""
        scores = []
        
        # Toxic-BERT
        if "toxic_bert" in self.models:
            try:
                tokenizer = self.models["toxic_bert_tokenizer"]
                model = self.models["toxic_bert"]
                
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = F.softmax(outputs.logits, dim=-1)
                    toxic_prob = probabilities[0][1].item()  # Assuming binary classification
                
                score = ToxicityScore(
                    toxicity_type=ToxicityType.TOXIC,
                    score=toxic_prob,
                    confidence=0.85,
                    threshold=self.threshold,
                    is_toxic=toxic_prob > self.threshold
                )
                scores.append(score)
            except Exception as e:
                logger.error(f"Toxic-BERT classification failed: {e}")
        
        # Content safety classifier
        if "content_safety" in self.classifiers:
            try:
                result = self.classifiers["content_safety"](text)
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]
                
                # Map label to toxicity type
                label = result.get("label", "").lower()
                toxicity_score = result.get("score", 0)
                
                if "toxic" in label:
                    score = ToxicityScore(
                        toxicity_type=ToxicityType.TOXIC,
                        score=toxicity_score,
                        confidence=0.8,
                        threshold=self.threshold,
                        is_toxic=toxicity_score > self.threshold
                    )
                    scores.append(score)
            except Exception as e:
                logger.error(f"Content safety classification failed: {e}")
        
        # Hate speech detector
        if "hate_speech" in self.classifiers:
            try:
                result = self.classifiers["hate_speech"](text)
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]
                
                label = result.get("label", "").lower()
                hate_score = result.get("score", 0)
                
                if "hate" in label:
                    score = ToxicityScore(
                        toxicity_type=ToxicityType.HATE_SPEECH,
                        score=hate_score,
                        confidence=0.8,
                        threshold=self.threshold,
                        is_toxic=hate_score > self.threshold
                    )
                    scores.append(score)
            except Exception as e:
                logger.error(f"Hate speech classification failed: {e}")
        
        return scores
    
    async def _classify_with_patterns(self, text: str) -> List[ToxicityScore]:
        """Classify using regex patterns and keyword matching"""
        scores = []
        
        for toxicity_type, patterns in self.toxic_patterns.items():
            matches = []
            total_score = 0
            
            for pattern in patterns:
                pattern_matches = pattern.findall(text)
                if pattern_matches:
                    matches.extend(pattern_matches)
                    # Score based on number of matches and pattern strength
                    total_score += len(pattern_matches) * 0.3
            
            if matches:
                # Normalize score to 0-1 range
                normalized_score = min(1.0, total_score)
                
                score = ToxicityScore(
                    toxicity_type=toxicity_type,
                    score=normalized_score,
                    confidence=0.6,  # Lower confidence for pattern matching
                    threshold=self.threshold,
                    is_toxic=normalized_score > self.threshold,
                    evidence=matches[:5]  # Keep first 5 matches as evidence
                )
                scores.append(score)
        
        return scores
    
    async def _classify_with_perspective(self, text: str) -> List[ToxicityScore]:
        """Classify using Google Perspective API"""
        if not self.perspective_api:
            return []
        
        try:
            analyze_request = {
                'comment': {'text': text},
                'requestedAttributes': {
                    'TOXICITY': {},
                    'SEVERE_TOXICITY': {},
                    'IDENTITY_ATTACK': {},
                    'INSULT': {},
                    'PROFANITY': {},
                    'THREAT': {},
                    'HARASSMENT': {},
                    'SEXUALLY_EXPLICIT': {}
                }
            }
            
            response = self.perspective_api.comments().analyze(body=analyze_request).execute()
            
            scores = []
            attribute_mapping = {
                'TOXICITY': ToxicityType.TOXIC,
                'SEVERE_TOXICITY': ToxicityType.SEVERE_TOXIC,
                'IDENTITY_ATTACK': ToxicityType.IDENTITY_HATE,
                'INSULT': ToxicityType.INSULT,
                'PROFANITY': ToxicityType.PROFANITY,
                'THREAT': ToxicityType.THREAT,
                'HARASSMENT': ToxicityType.HARASSMENT,
                'SEXUALLY_EXPLICIT': ToxicityType.SEXUAL
            }
            
            for attribute, toxicity_type in attribute_mapping.items():
                if attribute in response['attributeScores']:
                    score_data = response['attributeScores'][attribute]
                    summary_score = score_data['summaryScore']['value']
                    
                    score = ToxicityScore(
                        toxicity_type=toxicity_type,
                        score=summary_score,
                        confidence=0.95,  # Perspective API is highly reliable
                        threshold=self.threshold,
                        is_toxic=summary_score > self.threshold
                    )
                    scores.append(score)
            
            return scores
            
        except Exception as e:
            logger.error(f"Perspective API classification failed: {e}")
            return []
    
    async def _classify_with_openai(self, text: str) -> List[ToxicityScore]:
        """Classify using OpenAI Moderation API"""
        if not self.openai_client:
            return []
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openai_client.Moderation.create(input=text)
            )
            
            scores = []
            results = response["results"][0]
            
            category_mapping = {
                "hate": ToxicityType.HATE_SPEECH,
                "hate/threatening": ToxicityType.THREAT,
                "harassment": ToxicityType.HARASSMENT,
                "harassment/threatening": ToxicityType.THREAT,
                "self-harm": ToxicityType.SELF_HARM,
                "sexual": ToxicityType.SEXUAL,
                "sexual/minors": ToxicityType.SEXUAL,
                "violence": ToxicityType.VIOLENCE,
                "violence/graphic": ToxicityType.VIOLENCE
            }
            
            for category, toxicity_type in category_mapping.items():
                if category in results["category_scores"]:
                    category_score = results["category_scores"][category]
                    
                    score = ToxicityScore(
                        toxicity_type=toxicity_type,
                        score=category_score,
                        confidence=0.9,
                        threshold=self.threshold,
                        is_toxic=category_score > self.threshold
                    )
                    scores.append(score)
            
            return scores
            
        except Exception as e:
            logger.error(f"OpenAI moderation failed: {e}")
            return []
    
    def _combine_scores(self, all_scores: List[ToxicityScore]) -> List[ToxicityScore]:
        """Combine and deduplicate scores from different methods"""
        # Group scores by toxicity type
        grouped_scores = {}
        for score in all_scores:
            if score.toxicity_type not in grouped_scores:
                grouped_scores[score.toxicity_type] = []
            grouped_scores[score.toxicity_type].append(score)
        
        # Combine scores for each type using weighted average
        combined_scores = []
        for toxicity_type, scores in grouped_scores.items():
            if scores:
                # Weight scores by confidence
                total_weight = sum(score.confidence for score in scores)
                if total_weight > 0:
                    weighted_score = sum(
                        score.score * score.confidence for score in scores
                    ) / total_weight
                    
                    avg_confidence = sum(score.confidence for score in scores) / len(scores)
                    max_threshold = max(score.threshold for score in scores)
                    
                    # Combine evidence
                    all_evidence = []
                    for score in scores:
                        all_evidence.extend(score.evidence)
                    
                    combined_score = ToxicityScore(
                        toxicity_type=toxicity_type,
                        score=weighted_score,
                        confidence=avg_confidence,
                        threshold=max_threshold,
                        is_toxic=weighted_score > max_threshold,
                        evidence=list(set(all_evidence))  # Remove duplicates
                    )
                    combined_scores.append(combined_score)
        
        return combined_scores
    
    async def batch_classify(self, texts: List[str], 
                           methods: Optional[List[str]] = None) -> List[ToxicityResult]:
        """Classify toxicity for multiple texts concurrently"""
        tasks = [self.classify_toxicity(text, methods) for text in texts]
        return await asyncio.gather(*tasks)
    
    def create_toxicity_report(self, results: List[ToxicityResult]) -> Dict[str, Any]:
        """Create a comprehensive toxicity analysis report"""
        total_texts = len(results)
        toxic_texts = sum(1 for r in results if r.is_toxic)
        
        # Toxicity distribution
        toxicity_levels = {"low": 0, "medium": 0, "high": 0}
        for result in results:
            if result.overall_toxicity < 0.3:
                toxicity_levels["low"] += 1
            elif result.overall_toxicity < 0.7:
                toxicity_levels["medium"] += 1
            else:
                toxicity_levels["high"] += 1
        
        # Type distribution
        type_counts = {}
        for result in results:
            for score in result.toxicity_scores:
                if score.is_toxic:
                    type_name = score.toxicity_type.value
                    type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        return {
            "summary": {
                "total_texts_analyzed": total_texts,
                "toxic_texts": toxic_texts,
                "toxicity_rate": toxic_texts / total_texts if total_texts > 0 else 0,
                "average_toxicity_score": sum(r.overall_toxicity for r in results) / total_texts if total_texts > 0 else 0
            },
            "toxicity_distribution": toxicity_levels,
            "toxicity_type_distribution": type_counts,
            "performance": {
                "average_processing_time": sum(r.processing_time for r in results) / total_texts if total_texts > 0 else 0,
                "total_processing_time": sum(r.processing_time for r in results)
            },
            "timestamp": datetime.utcnow().isoformat()
        }


class ToxicityModerationService:
    """Service for real-time toxicity moderation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.classifier = ToxicityClassifier(config)
        self.action_thresholds = config.get("action_thresholds", {
            "warn": 0.3,
            "flag": 0.5,
            "block": 0.8
        })
        self.cache = {}
        self.cache_enabled = config.get("enable_cache", True)
    
    async def moderate_content(self, text: str, 
                             user_id: Optional[str] = None) -> Dict[str, Any]:
        """Moderate content and return action recommendation"""
        # Check cache first
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if self.cache_enabled and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Classify toxicity
        result = await self.classifier.classify_toxicity(text)
        
        # Determine action
        action = self._determine_action(result)
        
        moderation_result = {
            "text": text,
            "toxicity_score": result.overall_toxicity,
            "is_toxic": result.is_toxic,
            "action": action,
            "toxicity_types": [score.toxicity_type.value for score in result.toxicity_scores if score.is_toxic],
            "confidence": max((score.confidence for score in result.toxicity_scores), default=0),
            "processing_time": result.processing_time,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Cache result
        if self.cache_enabled:
            self.cache[cache_key] = moderation_result
        
        return moderation_result
    
    def _determine_action(self, result: ToxicityResult) -> str:
        """Determine moderation action based on toxicity score"""
        score = result.overall_toxicity
        
        if score >= self.action_thresholds["block"]:
            return "block"
        elif score >= self.action_thresholds["flag"]:
            return "flag"
        elif score >= self.action_thresholds["warn"]:
            return "warn"
        else:
            return "allow"


# Example usage
if __name__ == "__main__":
    async def example():
        # Sample toxic and non-toxic texts
        test_texts = [
            "Hello, how are you today?",  # Clean
            "You are such an idiot!",  # Insult
            "I'm going to kill you!",  # Threat
            "This is a normal message about cats.",  # Clean
        ]
        
        # Initialize classifier
        config = {
            "use_detoxify": True,
            "use_unitary_toxic": True,
            "threshold": 0.5
        }
        classifier = ToxicityClassifier(config)
        
        # Classify texts
        for text in test_texts:
            result = await classifier.classify_toxicity(text)
            print(f"\nText: {text}")
            print(f"Toxic: {result.is_toxic}")
            print(f"Overall score: {result.overall_toxicity:.3f}")
            print(f"Max type: {result.max_toxicity_type}")
    
    # Run example
    # asyncio.run(example())