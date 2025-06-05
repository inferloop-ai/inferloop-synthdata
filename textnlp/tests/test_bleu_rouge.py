# tests/test_bleu_rouge.py
import unittest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdk.validation import BLEUROUGEValidator

class TestBLEUROUGEValidator(unittest.TestCase):
    def setUp(self):
        self.validator = BLEUROUGEValidator()
        self.reference = "The quick brown fox jumps over the lazy dog."
        self.perfect_match = "The quick brown fox jumps over the lazy dog."
        self.partial_match = "The brown fox jumps over the dog."
        self.no_match = "A completely different sentence."
    
    def test_calculate_bleu(self):
        """Test BLEU score calculation"""
        perfect_score = self.validator.calculate_bleu(self.reference, self.perfect_match)
        partial_score = self.validator.calculate_bleu(self.reference, self.partial_match)
        no_match_score = self.validator.calculate_bleu(self.reference, self.no_match)
        
        self.assertEqual(perfect_score, 1.0)
        self.assertGreater(perfect_score, partial_score)
        self.assertGreater(partial_score, no_match_score)
    
    def test_calculate_rouge(self):
        """Test ROUGE score calculation"""
        perfect_scores = self.validator.calculate_rouge(self.reference, self.perfect_match)
        partial_scores = self.validator.calculate_rouge(self.reference, self.partial_match)
        no_match_scores = self.validator.calculate_rouge(self.reference, self.no_match)
        
        # Perfect match should have scores close to 1.0
        self.assertAlmostEqual(perfect_scores['rouge1'], 1.0, places=1)
        self.assertAlmostEqual(perfect_scores['rouge2'], 1.0, places=1)
        self.assertAlmostEqual(perfect_scores['rougeL'], 1.0, places=1)
        
        # Perfect match should score higher than partial match
        self.assertGreater(perfect_scores['rouge1'], partial_scores['rouge1'])
        self.assertGreater(perfect_scores['rouge2'], partial_scores['rouge2'])
        self.assertGreater(perfect_scores['rougeL'], partial_scores['rougeL'])
        
        # Partial match should score higher than no match
        self.assertGreater(partial_scores['rouge1'], no_match_scores['rouge1'])
        self.assertGreater(partial_scores['rouge2'], no_match_scores['rouge2'])
        self.assertGreater(partial_scores['rougeL'], no_match_scores['rougeL'])
    
    def test_validate_batch(self):
        """Test batch validation"""
        references = [self.reference, self.reference]
        candidates = [self.perfect_match, self.partial_match]
        
        scores = self.validator.validate_batch(references, candidates)
        
        # Check that we get the expected score types
        self.assertIn('bleu', scores)
        self.assertIn('rouge1', scores)
        self.assertIn('rouge2', scores)
        self.assertIn('rougeL', scores)
        
        # Check that we get the expected number of scores
        self.assertEqual(len(scores['bleu']), 2)
        self.assertEqual(len(scores['rouge1']), 2)
        
        # First score (perfect match) should be higher than second score (partial match)
        self.assertGreater(scores['bleu'][0], scores['bleu'][1])
        self.assertGreater(scores['rouge1'][0], scores['rouge1'][1])
        
    def test_validate_batch_error(self):
        """Test validation with mismatched inputs"""
        references = [self.reference]
        candidates = [self.perfect_match, self.partial_match]  # One more candidate than reference
        
        with self.assertRaises(ValueError):
            self.validator.validate_batch(references, candidates)

if __name__ == "__main__":
    unittest.main()
