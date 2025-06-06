# src/generators/starcoder_generator.py
import requests
import json
from typing import Dict, Any
from .base_generator import BaseCodeGenerator, GenerationConfig

class StarCoderGenerator(BaseCodeGenerator):
    """StarCoder-based code generator for multiple languages"""
    
    def __init__(self, config: GenerationConfig, model_endpoint: str = None):
        super().__init__(config)
        self.model_endpoint = model_endpoint or "http://localhost:8001/v1/completions"
        self.model_name = "bigcode/starcoder"
    
    def _call_model(self, prompt: str, max_tokens: int = 512) -> str:
        """Call StarCoder model API"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.6,
            "top_p": 0.95,
            "stop": ["```", "\n\n"]
        }
        
        response = requests.post(self.model_endpoint, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["text"].strip()
        else:
            raise Exception(f"StarCoder API error: {response.status_code}")
    
    def generate_function(self, prompt: str, **kwargs) -> str:
        """Generate function using StarCoder"""
        enhanced_prompt = f"""
        Language: {self.config.language}
        Task: {prompt}
        
        Generate a well-structured function with:
        - Proper naming conventions
        - Type annotations (where applicable)
        - Error handling
        - Documentation
        
        Code:
        """
        
        return self._call_model(enhanced_prompt)
    
    def generate_class(self, prompt: str, **kwargs) -> str:
        """Generate class using StarCoder"""
        enhanced_prompt = f"""
        Language: {self.config.language}
        Create a class for: {prompt}
        
        Include:
        - Constructor
        - Key methods
        - Properties
        - Documentation
        
        Class definition:
        """
        
        return self._call_model(enhanced_prompt, max_tokens=1024)
    
    def generate_module(self, prompt: str, **kwargs) -> str:
        """Generate complete module"""
        enhanced_prompt = f"""
        Language: {self.config.language}
        Module specification: {prompt}
        
        Create a complete module with:
        - Imports
        - Classes and functions
        - Constants
        - Module docstring
        - Usage examples
        
        Module:
        """
        
        return self._call_model(enhanced_prompt, max_tokens=2048)
    
    def generate_api_endpoint(self, schema: Dict[str, Any]) -> str:
        """Generate API endpoint using StarCoder"""
        framework = self.config.framework or "express"
        
        prompt = f"""
        Language: {self.config.language}
        Framework: {framework}
        API Schema: {json.dumps(schema, indent=2)}
        
        Generate a complete API endpoint with:
        - Route definition
        - Input validation
        - Business logic
        - Error handling
        - Response formatting
        
        Endpoint code:
        """
        
        return self._call_model(prompt, max_tokens=1024)
