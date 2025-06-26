"""
Quality Metrics Implementation for TextNLP
Advanced quality evaluation including BLEU, ROUGE, METEOR, BERTScore, and custom metrics
"""

import re
import math
import logging
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter, defaultdict
import statistics
import nltk
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
from concurrent.futures import ThreadPoolExecutor
import spacy

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logger = logging.getLogger(__name__)


class QualityMetricType(Enum):
    """Types of quality metrics"""
    BLEU = "bleu"
    ROUGE_1 = "rouge_1"
    ROUGE_2 = "rouge_2"
    ROUGE_L = "rouge_l"
    ROUGE_LSUM = "rouge_lsum"
    METEOR = "meteor"
    BERTSCORE = "bertscore"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    FLUENCY = "fluency"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    DIVERSITY = "diversity"
    NOVELTY = "novelty"
    FACTUALITY = "factuality"
    COVERAGE = "coverage"
    CONCISENESS = "conciseness"


@dataclass
class QualityScore:
    """Individual quality score"""
    metric_type: QualityMetricType
    score: float
    confidence: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""


@dataclass
class QualityEvaluation:
    """Complete quality evaluation result"""
    reference_text: str
    candidate_text: str
    scores: List[QualityScore]
    overall_quality: float
    evaluation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_score(self, metric_type: QualityMetricType) -> Optional[float]:
        """Get score for specific metric type"""
        for score in self.scores:
            if score.metric_type == metric_type:
                return score.score
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "reference_text": self.reference_text,
            "candidate_text": self.candidate_text,
            "scores": {
                score.metric_type.value: {
                    "score": score.score,
                    "confidence": score.confidence,
                    "details": score.details,
                    "explanation": score.explanation
                }
                for score in self.scores
            },
            "overall_quality": self.overall_quality,
            "evaluation_time": self.evaluation_time,
            "metadata": self.metadata
        }


class QualityMetricsCalculator:
    """Advanced quality metrics calculator"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.enabled_metrics = set(self.config.get("enabled_metrics", [
            QualityMetricType.BLEU, QualityMetricType.ROUGE_1, QualityMetricType.ROUGE_2,
            QualityMetricType.ROUGE_L, QualityMetricType.SEMANTIC_SIMILARITY
        ]))
        
        # Initialize components
        self._initialize_rouge_scorer()
        self._initialize_bert_scorer()
        self._initialize_semantic_models()
        self._initialize_nlp_models()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.config.get("max_workers", 4))
    
    def _initialize_rouge_scorer(self):
        """Initialize ROUGE scorer"""
        try:
            rouge_types = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
            self.rouge_scorer = rouge_scorer.RougeScorer(
                rouge_types, 
                use_stemmer=True
            )
            logger.info("Initialized ROUGE scorer")
        except Exception as e:
            logger.warning(f"Failed to initialize ROUGE scorer: {e}")
            self.rouge_scorer = None
    
    def _initialize_bert_scorer(self):
        """Initialize BERTScore components"""
        try:
            model_name = self.config.get("bertscore_model", "bert-base-uncased")
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            self.bert_model.eval()
            logger.info(f"Initialized BERTScore with model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize BERTScore: {e}")
            self.bert_tokenizer = None
            self.bert_model = None
    
    def _initialize_semantic_models(self):
        """Initialize semantic similarity models"""
        try:
            # Sentence transformer for semantic similarity
            from sentence_transformers import SentenceTransformer
            model_name = self.config.get("semantic_model", "all-MiniLM-L6-v2")
            self.semantic_model = SentenceTransformer(model_name)
            logger.info(f"Initialized semantic model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize semantic model: {e}")
            self.semantic_model = None
    
    def _initialize_nlp_models(self):
        """Initialize NLP models for linguistic analysis"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Initialized spaCy model")
        except Exception as e:
            logger.warning(f"Failed to initialize spaCy: {e}")
            self.nlp = None
    
    async def evaluate_quality(self, reference: str, candidate: str, 
                             context: Optional[str] = None) -> QualityEvaluation:
        """Evaluate quality of generated text against reference"""
        start_time = asyncio.get_event_loop().time()
        
        scores = []
        
        # Calculate enabled metrics
        metric_tasks = []
        
        if QualityMetricType.BLEU in self.enabled_metrics:
            metric_tasks.append(self._calculate_bleu_async(reference, candidate))
        
        if any(mt.value.startswith('rouge') for mt in self.enabled_metrics):
            metric_tasks.append(self._calculate_rouge_async(reference, candidate))
        
        if QualityMetricType.METEOR in self.enabled_metrics:
            metric_tasks.append(self._calculate_meteor_async(reference, candidate))
        
        if QualityMetricType.BERTSCORE in self.enabled_metrics:
            metric_tasks.append(self._calculate_bertscore_async(reference, candidate))
        
        if QualityMetricType.SEMANTIC_SIMILARITY in self.enabled_metrics:
            metric_tasks.append(self._calculate_semantic_similarity_async(reference, candidate))
        
        if QualityMetricType.FLUENCY in self.enabled_metrics:
            metric_tasks.append(self._calculate_fluency_async(candidate))
        
        if QualityMetricType.COHERENCE in self.enabled_metrics:
            metric_tasks.append(self._calculate_coherence_async(candidate, context))
        
        if QualityMetricType.DIVERSITY in self.enabled_metrics:
            metric_tasks.append(self._calculate_diversity_async(candidate))
        
        if QualityMetricType.RELEVANCE in self.enabled_metrics:
            metric_tasks.append(self._calculate_relevance_async(reference, candidate, context))
        
        if QualityMetricType.FACTUALITY in self.enabled_metrics:
            metric_tasks.append(self._calculate_factuality_async(reference, candidate))
        
        # Execute all metric calculations
        metric_results = await asyncio.gather(*metric_tasks, return_exceptions=True)
        
        # Collect results
        for result in metric_results:
            if isinstance(result, Exception):
                logger.error(f"Metric calculation failed: {result}")
            elif isinstance(result, list):
                scores.extend(result)
            elif isinstance(result, QualityScore):
                scores.append(result)
        
        # Calculate overall quality score
        overall_quality = self._calculate_overall_quality(scores)
        
        evaluation_time = asyncio.get_event_loop().time() - start_time
        
        return QualityEvaluation(
            reference_text=reference,
            candidate_text=candidate,
            scores=scores,
            overall_quality=overall_quality,
            evaluation_time=evaluation_time,
            metadata={
                "context": context,
                "metrics_calculated": len(scores),
                "enabled_metrics": [m.value for m in self.enabled_metrics]
            }
        )
    
    async def _calculate_bleu_async(self, reference: str, candidate: str) -> QualityScore:
        """Calculate BLEU score asynchronously"""
        def calculate_bleu():
            # Tokenize texts
            ref_tokens = nltk.word_tokenize(reference.lower())
            cand_tokens = nltk.word_tokenize(candidate.lower())
            
            # Calculate BLEU with smoothing
            smoothing = SmoothingFunction()
            bleu_score = sentence_bleu(
                [ref_tokens], 
                cand_tokens,
                smoothing_function=smoothing.method1
            )
            
            # Calculate individual n-gram scores
            bleu_1 = sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0))
            bleu_2 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5, 0, 0))
            bleu_3 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.33, 0.33, 0.33, 0))
            bleu_4 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25))
            
            return QualityScore(
                metric_type=QualityMetricType.BLEU,
                score=bleu_score,
                details={
                    "bleu_1": bleu_1,
                    "bleu_2": bleu_2,
                    "bleu_3": bleu_3,
                    "bleu_4": bleu_4,
                    "reference_length": len(ref_tokens),
                    "candidate_length": len(cand_tokens)
                },
                explanation=f"BLEU score measures n-gram overlap. Score: {bleu_score:.3f}"
            )
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, calculate_bleu)
    
    async def _calculate_rouge_async(self, reference: str, candidate: str) -> List[QualityScore]:
        """Calculate ROUGE scores asynchronously"""
        def calculate_rouge():
            if not self.rouge_scorer:
                return []
            
            scores = self.rouge_scorer.score(reference, candidate)
            results = []
            
            for rouge_type, score in scores.items():
                metric_type_map = {
                    'rouge1': QualityMetricType.ROUGE_1,
                    'rouge2': QualityMetricType.ROUGE_2,
                    'rougeL': QualityMetricType.ROUGE_L,
                    'rougeLsum': QualityMetricType.ROUGE_LSUM
                }
                
                if rouge_type in metric_type_map:
                    metric_type = metric_type_map[rouge_type]
                    if metric_type in self.enabled_metrics:
                        results.append(QualityScore(
                            metric_type=metric_type,
                            score=score.fmeasure,
                            details={
                                "precision": score.precision,
                                "recall": score.recall,
                                "fmeasure": score.fmeasure
                            },
                            explanation=f"{rouge_type.upper()} F1 score: {score.fmeasure:.3f}"
                        ))
            
            return results
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, calculate_rouge)
    
    async def _calculate_meteor_async(self, reference: str, candidate: str) -> QualityScore:
        """Calculate METEOR score asynchronously"""
        def calculate_meteor():
            try:
                ref_tokens = nltk.word_tokenize(reference.lower())
                cand_tokens = nltk.word_tokenize(candidate.lower())
                
                score = meteor_score([ref_tokens], cand_tokens)
                
                return QualityScore(
                    metric_type=QualityMetricType.METEOR,
                    score=score,
                    explanation=f"METEOR score considers synonyms and word order. Score: {score:.3f}"
                )
            except Exception as e:
                logger.warning(f"METEOR calculation failed: {e}")
                return QualityScore(
                    metric_type=QualityMetricType.METEOR,
                    score=0.0,
                    confidence=0.0,
                    explanation="METEOR calculation failed"
                )
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, calculate_meteor)
    
    async def _calculate_bertscore_async(self, reference: str, candidate: str) -> QualityScore:
        """Calculate BERTScore asynchronously"""
        def calculate_bertscore():
            if not self.bert_model or not self.bert_tokenizer:
                return QualityScore(
                    metric_type=QualityMetricType.BERTSCORE,
                    score=0.0,
                    confidence=0.0,
                    explanation="BERTScore model not available"
                )
            
            try:
                # Tokenize both texts
                ref_inputs = self.bert_tokenizer(
                    reference, return_tensors="pt", 
                    truncation=True, padding=True, max_length=512
                )
                cand_inputs = self.bert_tokenizer(
                    candidate, return_tensors="pt", 
                    truncation=True, padding=True, max_length=512
                )
                
                # Get embeddings
                with torch.no_grad():
                    ref_outputs = self.bert_model(**ref_inputs)
                    cand_outputs = self.bert_model(**cand_inputs)
                
                # Calculate cosine similarity between pooled outputs
                ref_embedding = ref_outputs.last_hidden_state.mean(dim=1)
                cand_embedding = cand_outputs.last_hidden_state.mean(dim=1)
                
                similarity = torch.cosine_similarity(ref_embedding, cand_embedding, dim=1)
                score = similarity.item()
                
                return QualityScore(
                    metric_type=QualityMetricType.BERTSCORE,
                    score=score,
                    explanation=f"BERTScore semantic similarity: {score:.3f}"
                )
                
            except Exception as e:
                logger.warning(f"BERTScore calculation failed: {e}")
                return QualityScore(
                    metric_type=QualityMetricType.BERTSCORE,
                    score=0.0,
                    confidence=0.0,
                    explanation="BERTScore calculation failed"
                )
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, calculate_bertscore)
    
    async def _calculate_semantic_similarity_async(self, reference: str, candidate: str) -> QualityScore:
        """Calculate semantic similarity using sentence transformers"""
        def calculate_semantic():
            if not self.semantic_model:
                return QualityScore(
                    metric_type=QualityMetricType.SEMANTIC_SIMILARITY,
                    score=0.0,
                    confidence=0.0,
                    explanation="Semantic similarity model not available"
                )
            
            try:
                # Encode both texts
                embeddings = self.semantic_model.encode([reference, candidate])
                
                # Calculate cosine similarity
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                
                return QualityScore(
                    metric_type=QualityMetricType.SEMANTIC_SIMILARITY,
                    score=float(similarity),
                    explanation=f"Semantic similarity using sentence transformers: {similarity:.3f}"
                )
                
            except Exception as e:
                logger.warning(f"Semantic similarity calculation failed: {e}")
                return QualityScore(
                    metric_type=QualityMetricType.SEMANTIC_SIMILARITY,
                    score=0.0,
                    confidence=0.0,
                    explanation="Semantic similarity calculation failed"
                )
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, calculate_semantic)
    
    async def _calculate_fluency_async(self, text: str) -> QualityScore:
        """Calculate fluency score"""
        def calculate_fluency():
            try:
                if not self.nlp:
                    # Fallback to simple heuristics
                    return self._calculate_fluency_heuristic(text)
                
                doc = self.nlp(text)
                
                # Count grammatical errors (simplified)
                error_count = 0
                total_tokens = len(doc)
                
                for token in doc:
                    # Check for obvious grammar issues
                    if token.pos_ == "VERB" and token.tag_ in ["VBZ", "VBP"]:
                        # Check subject-verb agreement (simplified)
                        for child in token.children:
                            if child.dep_ == "nsubj" and child.tag_ in ["NNS", "NNPS"]:
                                if token.tag_ == "VBZ":  # Singular verb with plural subject
                                    error_count += 1
                
                # Calculate fluency score
                error_rate = error_count / max(total_tokens, 1)
                fluency_score = max(0, 1 - error_rate * 5)  # Scale down errors
                
                return QualityScore(
                    metric_type=QualityMetricType.FLUENCY,
                    score=fluency_score,
                    details={
                        "error_count": error_count,
                        "total_tokens": total_tokens,
                        "error_rate": error_rate
                    },
                    explanation=f"Fluency score based on grammatical correctness: {fluency_score:.3f}"
                )
                
            except Exception as e:
                logger.warning(f"Fluency calculation failed: {e}")
                return self._calculate_fluency_heuristic(text)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, calculate_fluency)
    
    def _calculate_fluency_heuristic(self, text: str) -> QualityScore:
        """Calculate fluency using simple heuristics"""
        # Simple heuristics for fluency
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return QualityScore(
                metric_type=QualityMetricType.FLUENCY,
                score=0.0,
                explanation="No sentences found"
            )
        
        # Check average sentence length
        words = re.findall(r'\b\w+\b', text)
        avg_sentence_length = len(words) / len(sentences)
        
        # Penalize very short or very long sentences
        length_score = 1.0
        if avg_sentence_length < 5:
            length_score = avg_sentence_length / 5
        elif avg_sentence_length > 30:
            length_score = 30 / avg_sentence_length
        
        # Check for repeated words
        word_counts = Counter(word.lower() for word in words)
        repetition_penalty = sum(max(0, count - 2) for count in word_counts.values()) / len(words)
        repetition_score = max(0, 1 - repetition_penalty)
        
        # Combine scores
        fluency_score = (length_score + repetition_score) / 2
        
        return QualityScore(
            metric_type=QualityMetricType.FLUENCY,
            score=fluency_score,
            details={
                "avg_sentence_length": avg_sentence_length,
                "repetition_penalty": repetition_penalty,
                "length_score": length_score,
                "repetition_score": repetition_score
            },
            explanation=f"Heuristic fluency score: {fluency_score:.3f}"
        )
    
    async def _calculate_coherence_async(self, text: str, context: Optional[str] = None) -> QualityScore:
        """Calculate coherence score"""
        def calculate_coherence():
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                return QualityScore(
                    metric_type=QualityMetricType.COHERENCE,
                    score=1.0 if sentences else 0.0,
                    explanation="Single or no sentence - coherence not applicable"
                )
            
            # Calculate coherence using TF-IDF similarity between adjacent sentences
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            
            try:
                tfidf_matrix = vectorizer.fit_transform(sentences)
                
                # Calculate average cosine similarity between adjacent sentences
                similarities = []
                for i in range(len(sentences) - 1):
                    sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[i + 1])[0][0]
                    similarities.append(sim)
                
                coherence_score = statistics.mean(similarities) if similarities else 0.0
                
                return QualityScore(
                    metric_type=QualityMetricType.COHERENCE,
                    score=coherence_score,
                    details={
                        "sentence_count": len(sentences),
                        "avg_similarity": coherence_score,
                        "similarities": similarities
                    },
                    explanation=f"Coherence based on inter-sentence similarity: {coherence_score:.3f}"
                )
                
            except Exception as e:
                logger.warning(f"Coherence calculation failed: {e}")
                return QualityScore(
                    metric_type=QualityMetricType.COHERENCE,
                    score=0.5,
                    confidence=0.0,
                    explanation="Coherence calculation failed"
                )
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, calculate_coherence)
    
    async def _calculate_diversity_async(self, text: str) -> QualityScore:
        """Calculate diversity score"""
        def calculate_diversity():
            words = re.findall(r'\b\w+\b', text.lower())
            
            if not words:
                return QualityScore(
                    metric_type=QualityMetricType.DIVERSITY,
                    score=0.0,
                    explanation="No words found"
                )
            
            # Lexical diversity (Type-Token Ratio)
            unique_words = set(words)
            ttr = len(unique_words) / len(words)
            
            # N-gram diversity
            bigrams = [tuple(words[i:i+2]) for i in range(len(words) - 1)]
            unique_bigrams = set(bigrams)
            bigram_diversity = len(unique_bigrams) / len(bigrams) if bigrams else 0
            
            trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
            unique_trigrams = set(trigrams)
            trigram_diversity = len(unique_trigrams) / len(trigrams) if trigrams else 0
            
            # Combined diversity score
            diversity_score = (ttr + bigram_diversity + trigram_diversity) / 3
            
            return QualityScore(
                metric_type=QualityMetricType.DIVERSITY,
                score=diversity_score,
                details={
                    "type_token_ratio": ttr,
                    "bigram_diversity": bigram_diversity,
                    "trigram_diversity": trigram_diversity,
                    "unique_words": len(unique_words),
                    "total_words": len(words)
                },
                explanation=f"Lexical diversity score: {diversity_score:.3f}"
            )
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, calculate_diversity)
    
    async def _calculate_relevance_async(self, reference: str, candidate: str, 
                                       context: Optional[str] = None) -> QualityScore:
        """Calculate relevance score"""
        def calculate_relevance():
            # Combine reference and context for relevance calculation
            target_text = reference
            if context:
                target_text = f"{context} {reference}"
            
            # Use TF-IDF to calculate relevance
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            
            try:
                tfidf_matrix = vectorizer.fit_transform([target_text, candidate])
                similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
                
                return QualityScore(
                    metric_type=QualityMetricType.RELEVANCE,
                    score=similarity,
                    explanation=f"Content relevance score: {similarity:.3f}"
                )
                
            except Exception as e:
                logger.warning(f"Relevance calculation failed: {e}")
                return QualityScore(
                    metric_type=QualityMetricType.RELEVANCE,
                    score=0.0,
                    confidence=0.0,
                    explanation="Relevance calculation failed"
                )
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, calculate_relevance)
    
    async def _calculate_factuality_async(self, reference: str, candidate: str) -> QualityScore:
        """Calculate factuality score (simplified implementation)"""
        def calculate_factuality():
            # Simple factuality check based on entity overlap
            # In practice, this would use more sophisticated fact-checking
            
            if not self.nlp:
                return QualityScore(
                    metric_type=QualityMetricType.FACTUALITY,
                    score=0.5,
                    confidence=0.0,
                    explanation="Factuality model not available"
                )
            
            try:
                ref_doc = self.nlp(reference)
                cand_doc = self.nlp(candidate)
                
                # Extract entities
                ref_entities = set(ent.text.lower() for ent in ref_doc.ents)
                cand_entities = set(ent.text.lower() for ent in cand_doc.ents)
                
                if not ref_entities:
                    return QualityScore(
                        metric_type=QualityMetricType.FACTUALITY,
                        score=1.0,
                        explanation="No entities in reference to verify"
                    )
                
                # Calculate entity overlap
                common_entities = ref_entities & cand_entities
                entity_accuracy = len(common_entities) / len(ref_entities)
                
                # Check for contradictory entities (simplified)
                contradictions = 0
                for ent in cand_entities - ref_entities:
                    # This is a very simplified check
                    if any(ref_ent in ent or ent in ref_ent for ref_ent in ref_entities):
                        contradictions += 1
                
                contradiction_penalty = contradictions / max(len(cand_entities), 1)
                factuality_score = max(0, entity_accuracy - contradiction_penalty)
                
                return QualityScore(
                    metric_type=QualityMetricType.FACTUALITY,
                    score=factuality_score,
                    details={
                        "reference_entities": list(ref_entities),
                        "candidate_entities": list(cand_entities),
                        "common_entities": list(common_entities),
                        "entity_accuracy": entity_accuracy,
                        "contradictions": contradictions
                    },
                    explanation=f"Entity-based factuality score: {factuality_score:.3f}"
                )
                
            except Exception as e:
                logger.warning(f"Factuality calculation failed: {e}")
                return QualityScore(
                    metric_type=QualityMetricType.FACTUALITY,
                    score=0.0,
                    confidence=0.0,
                    explanation="Factuality calculation failed"
                )
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, calculate_factuality)
    
    def _calculate_overall_quality(self, scores: List[QualityScore]) -> float:
        """Calculate overall quality score from individual metrics"""
        if not scores:
            return 0.0
        
        # Weight different metrics
        metric_weights = {
            QualityMetricType.BLEU: 0.2,
            QualityMetricType.ROUGE_1: 0.15,
            QualityMetricType.ROUGE_2: 0.15,
            QualityMetricType.ROUGE_L: 0.15,
            QualityMetricType.SEMANTIC_SIMILARITY: 0.2,
            QualityMetricType.FLUENCY: 0.1,
            QualityMetricType.COHERENCE: 0.1,
            QualityMetricType.RELEVANCE: 0.1,
            QualityMetricType.FACTUALITY: 0.1,
            QualityMetricType.DIVERSITY: 0.05
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for score in scores:
            weight = metric_weights.get(score.metric_type, 0.05)
            weighted_sum += score.score * weight * score.confidence
            total_weight += weight * score.confidence
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    async def batch_evaluate(self, evaluation_pairs: List[Tuple[str, str]], 
                           context: Optional[str] = None) -> List[QualityEvaluation]:
        """Evaluate quality for multiple reference-candidate pairs"""
        tasks = [
            self.evaluate_quality(ref, cand, context) 
            for ref, cand in evaluation_pairs
        ]
        return await asyncio.gather(*tasks)
    
    def create_quality_report(self, evaluations: List[QualityEvaluation]) -> Dict[str, Any]:
        """Create comprehensive quality report"""
        if not evaluations:
            return {"message": "No evaluations provided"}
        
        # Aggregate scores by metric type
        metric_scores = defaultdict(list)
        for eval_result in evaluations:
            for score in eval_result.scores:
                metric_scores[score.metric_type.value].append(score.score)
        
        # Calculate statistics for each metric
        metric_stats = {}
        for metric_type, scores in metric_scores.items():
            metric_stats[metric_type] = {
                "mean": statistics.mean(scores),
                "median": statistics.median(scores),
                "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "min": min(scores),
                "max": max(scores),
                "count": len(scores)
            }
        
        # Overall quality statistics
        overall_scores = [eval_result.overall_quality for eval_result in evaluations]
        
        return {
            "summary": {
                "total_evaluations": len(evaluations),
                "overall_quality": {
                    "mean": statistics.mean(overall_scores),
                    "median": statistics.median(overall_scores),
                    "std": statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0.0,
                    "min": min(overall_scores),
                    "max": max(overall_scores)
                }
            },
            "metric_statistics": metric_stats,
            "performance": {
                "avg_evaluation_time": statistics.mean([e.evaluation_time for e in evaluations]),
                "total_evaluation_time": sum(e.evaluation_time for e in evaluations)
            }
        }
    
    def shutdown(self):
        """Shutdown the metrics calculator"""
        if self.executor:
            self.executor.shutdown(wait=True)


# Example usage
if __name__ == "__main__":
    async def example():
        # Initialize quality metrics calculator
        config = {
            "enabled_metrics": [
                QualityMetricType.BLEU, QualityMetricType.ROUGE_1, QualityMetricType.ROUGE_2,
                QualityMetricType.SEMANTIC_SIMILARITY, QualityMetricType.FLUENCY,
                QualityMetricType.COHERENCE, QualityMetricType.DIVERSITY
            ],
            "max_workers": 2
        }
        
        calculator = QualityMetricsCalculator(config)
        
        # Example evaluation
        reference = "The quick brown fox jumps over the lazy dog. This is a well-known pangram."
        candidate = "A fast brown fox leaps over a sleepy dog. This sentence contains all letters."
        
        evaluation = await calculator.evaluate_quality(reference, candidate)
        
        print("Quality Evaluation Results:")
        print(f"Overall Quality: {evaluation.overall_quality:.3f}")
        print(f"Evaluation Time: {evaluation.evaluation_time:.3f}s")
        
        for score in evaluation.scores:
            print(f"{score.metric_type.value}: {score.score:.3f} - {score.explanation}")
        
        # Batch evaluation example
        evaluation_pairs = [
            ("Hello world", "Hello universe"),
            ("Good morning", "Good evening"),
            ("The cat sat on the mat", "A feline rested on the rug")
        ]
        
        batch_results = await calculator.batch_evaluate(evaluation_pairs)
        
        # Create quality report
        report = calculator.create_quality_report(batch_results)
        print("\nQuality Report:")
        print(f"Total evaluations: {report['summary']['total_evaluations']}")
        print(f"Average overall quality: {report['summary']['overall_quality']['mean']:.3f}")
        
        calculator.shutdown()
    
    # Run example
    # asyncio.run(example())