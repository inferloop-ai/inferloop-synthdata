# sdk/validation/bleu_rouge.py
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class BLEUROUGEValidator:
    """BLEU and ROUGE score validation for generated text"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
    
    def calculate_bleu(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score between reference and candidate"""
        try:
            ref_tokens = nltk.word_tokenize(reference.lower())
            cand_tokens = nltk.word_tokenize(candidate.lower())
            
            score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=self.smoothing)
            return score
        except Exception as e:
            logger.error(f"BLEU calculation failed: {e}")
            return 0.0
    
    def calculate_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate ROUGE scores between reference and candidate"""
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.error(f"ROUGE calculation failed: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def validate_batch(self, references: List[str], candidates: List[str]) -> Dict[str, List[float]]:
        """Validate a batch of reference-candidate pairs"""
        if len(references) != len(candidates):
            raise ValueError("References and candidates must have the same length")
        
        bleu_scores = []
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for ref, cand in zip(references, candidates):
            bleu = self.calculate_bleu(ref, cand)
            rouge = self.calculate_rouge(ref, cand)
            
            bleu_scores.append(bleu)
            rouge1_scores.append(rouge['rouge1'])
            rouge2_scores.append(rouge['rouge2'])
            rougeL_scores.append(rouge['rougeL'])
        
        return {
            'bleu': bleu_scores,
            'rouge1': rouge1_scores,
            'rouge2': rouge2_scores,
            'rougeL': rougeL_scores
        }
