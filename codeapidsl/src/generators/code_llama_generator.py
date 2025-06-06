# src/generators/code_llama_generator.py
import requests
import json
from typing import Dict, Any
from .base_generator import BaseCodeGenerator, GenerationConfig

class CodeLlamaGenerator(BaseCodeGenerator):
    """CodeLlama-based code generator"""
    
    def __init__(self, config: GenerationConfig, model_endpoint: str = None):
        super().__init__(config)
        self.model_endpoint = model_endpoint or "http://localhost:8000/v1/completions"
        self.model_name = "codellama/CodeLlama-13b-Instruct-hf"
    
    def _call_model(self, prompt: str, max_tokens: int = 512) -> str:
        """Call CodeLlama model API"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stop": ["\n\n", "```"]
        }
        
        response = requests.post(self.model_endpoint, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["text"].strip()
        else:
            raise Exception(f"Model API error: {response.status_code}")
    
    def generate_function(self, prompt: str, **kwargs) -> str:
        """Generate a function using CodeLlama"""
        enhanced_prompt = f"""
        Generate a {self.config.language} function based on this description:
        {prompt}
        
        Requirements:
        - Follow {self.config.language} best practices
        - Include type hints (if applicable)
        - Add docstring
        - Handle edge cases
        
        Function:
        ```{self.config.language}
        """
        
        return self._call_model(enhanced_prompt)
    
    def generate_class(self, prompt: str, **kwargs) -> str:
        """Generate a class using CodeLlama"""
        enhanced_prompt = f"""
        Create a {self.config.language} class based on this specification:
        {prompt}
        
        Requirements:
        - Include constructor and key methods
        - Add class and method docstrings
        - Follow {self.config.language} naming conventions
        - Include error handling
        
        Class:
        ```{self.config.language}
        """
        
        return self._call_model(enhanced_prompt, max_tokens=1024)
    
    def generate_module(self, prompt: str, **kwargs) -> str:
        """Generate a complete module"""
        enhanced_prompt = f"""
        Create a complete {self.config.language} module for:
        {prompt}
        
        Include:
        - Imports and dependencies
        - Multiple related functions/classes
        - Module-level docstring
        - Example usage
        
        Module:
        ```{self.config.language}
        """
        
        return self._call_model(enhanced_prompt, max_tokens=2048)
    
    def generate_api_endpoint(self, schema: Dict[str, Any]) -> str:
        """Generate API endpoint code"""
        framework = self.config.framework or "fastapi"
        
        prompt = f"""
        Create a {framework} API endpoint with this schema:
        {json.dumps(schema, indent=2)}
        
        Include:
        - Request/response models
        - Validation
        - Error handling
        - Documentation
        
        Endpoint:
        ```python
        """
        
        return self._call_model(prompt, max_tokens=1024)
