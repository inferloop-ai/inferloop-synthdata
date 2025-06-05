# sdk/llm_gpt2.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from typing import List, Dict, Any
import torch
import logging

logger = logging.getLogger(__name__)

class GPT2Generator(BaseGenerator):
    """GPT-2 based text generator (can be adapted for GPT-J, NeoX, LLaMA)"""
    
    def __init__(self, model_name: str = "gpt2", **kwargs):
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(model_name, **kwargs)
    
    def _setup_model(self):
        """Initialize GPT-2 model and tokenizer"""
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"Loaded {self.model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def generate(self, prompt: str, max_length: int = 100, **kwargs) -> str:
        """Generate text from a single prompt"""
        if not self.validate_input(prompt):
            return ""
        
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            generation_params = {
                'max_length': max_length,
                'num_return_sequences': 1,
                'temperature': kwargs.get('temperature', 0.8),
                'top_p': kwargs.get('top_p', 0.9),
                'do_sample': kwargs.get('do_sample', True),
                'pad_token_id': self.tokenizer.eos_token_id
            }
            
            with torch.no_grad():
                outputs = self.model.generate(inputs, **generation_params)
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the original prompt from output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
                
            return generated_text
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts"""
        results = []
        batch_size = kwargs.get('batch_size', 4)
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = []
            
            for prompt in batch:
                result = self.generate(prompt, **kwargs)
                batch_results.append(result)
            
            results.extend(batch_results)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}")
        
        return results
