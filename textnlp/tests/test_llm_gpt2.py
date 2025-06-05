# tests/test_llm_gpt2.py
import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdk import GPT2Generator

class TestGPT2Generator(unittest.TestCase):
    @patch('sdk.llm_gpt2.GPT2Tokenizer')
    @patch('sdk.llm_gpt2.GPT2LMHeadModel')
    def setUp(self, mock_model, mock_tokenizer):
        # Mock the tokenizer and model
        self.mock_tokenizer = mock_tokenizer.from_pretrained.return_value
        self.mock_model = mock_model.from_pretrained.return_value
        
        # Set up mock return values
        self.mock_tokenizer.encode.return_value = MagicMock()
        self.mock_tokenizer.encode.return_value.to.return_value = MagicMock()
        self.mock_model.generate.return_value = MagicMock()
        self.mock_tokenizer.decode.return_value = "This is a generated text"
        
        # Create generator with mocked components
        self.generator = GPT2Generator("gpt2")
    
    def test_initialization(self):
        """Test that the generator initializes correctly"""
        self.assertEqual(self.generator.model_name, "gpt2")
        self.assertIsNotNone(self.generator.model)
        self.assertIsNotNone(self.generator.tokenizer)
    
    def test_validate_input(self):
        """Test input validation"""
        self.assertTrue(self.generator.validate_input("Valid prompt"))
        self.assertFalse(self.generator.validate_input(""))
        self.assertFalse(self.generator.validate_input(None))
        self.assertFalse(self.generator.validate_input(123))  # Not a string
    
    def test_generate(self):
        """Test text generation"""
        prompt = "Generate some text"
        result = self.generator.generate(prompt)
        
        # Check that the tokenizer and model were called
        self.mock_tokenizer.encode.assert_called_once_with(prompt, return_tensors="pt")
        self.mock_model.generate.assert_called_once()
        self.mock_tokenizer.decode.assert_called_once()
        
        # Check the result
        self.assertEqual(result, "This is a generated text")
    
    def test_batch_generate(self):
        """Test batch text generation"""
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        
        # Set up the mock to return different values for each call
        self.mock_tokenizer.decode.side_effect = [
            "Generated text 1",
            "Generated text 2",
            "Generated text 3"
        ]
        
        results = self.generator.batch_generate(prompts)
        
        # Check that we got the expected number of results
        self.assertEqual(len(results), 3)
        
        # Check that the generate method was called for each prompt
        self.assertEqual(self.mock_tokenizer.encode.call_count, 3)
        self.assertEqual(self.mock_model.generate.call_count, 3)
        self.assertEqual(self.mock_tokenizer.decode.call_count, 3)

if __name__ == "__main__":
    unittest.main()
