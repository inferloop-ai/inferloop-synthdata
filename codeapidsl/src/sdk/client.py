# src/sdk/client.py
import requests
import json
from typing import List, Dict, Any, Optional
from .exceptions import SynthCodeException, ValidationException

class SynthCodeSDK:
    """Python SDK for Synthetic Code Generation"""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def generate_code(
        self,
        prompts: List[str],
        language: str = "python",
        framework: Optional[str] = None,
        complexity: str = "medium",
        count: int = 100,
        include_tests: bool = True,
        include_validation: bool = True,
        output_format: str = "jsonl"
    ) -> Dict[str, Any]:
        """
        Generate synthetic code samples
        
        Args:
            prompts: List of generation prompts
            language: Programming language
            framework: Optional framework
            complexity: Code complexity level
            count: Number of samples to generate
            include_tests: Include unit tests
            include_validation: Validate generated code
            output_format: Output format (jsonl, csv, grpc)
        
        Returns:
            Dictionary containing generated code and metadata
        """
        payload = {
            "prompts": prompts,
            "language": language,
            "framework": framework,
            "complexity": complexity,
            "count": count,
            "include_tests": include_tests,
            "include_validation": include_validation,
            "output_format": output_format
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/generate/code",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise SynthCodeException(f"API request failed: {e}")
    
    def generate_tabular(self, **kwargs):
        """Generate tabular synthetic data (placeholder)"""
        return self._call_endpoint("/generate/tabular", **kwargs)
    
    def generate_text(self, **kwargs):
        """Generate text synthetic data (placeholder)"""
        return self._call_endpoint("/generate/text", **kwargs)
    
    def generate_audio(self, **kwargs):
        """Generate audio synthetic data (placeholder)"""
        return self._call_endpoint("/generate/audio", **kwargs)
    
    def generate_image(self, **kwargs):
        """Generate image synthetic data (placeholder)"""
        return self._call_endpoint("/generate/image", **kwargs)
    
    def generate_video(self, **kwargs):
        """Generate video synthetic data (placeholder)"""
        return self._call_endpoint("/generate/video", **kwargs)
    
    def generate_timeseries(self, **kwargs):
        """Generate time series synthetic data (placeholder)"""
        return self._call_endpoint("/generate/timeseries", **kwargs)
    
    def generate_graph(self, **kwargs):
        """Generate graph synthetic data (placeholder)"""
        return self._call_endpoint("/generate/graph", **kwargs)
    
    def generate_document(self, **kwargs):
        """Generate document synthetic data (placeholder)"""
        return self._call_endpoint("/generate/document", **kwargs)
    
    def generate_logs(self, **kwargs):
        """Generate log synthetic data (placeholder)"""
        return self._call_endpoint("/generate/logs", **kwargs)
    
    def generate_rag(self, **kwargs):
        """Generate RAG-ready synthetic data (placeholder)"""
        return self._call_endpoint("/generate/rag-data", **kwargs)
    
    def generate_multimodal(self, **kwargs):
        """Generate multimodal synthetic data (placeholder)"""
        return self._call_endpoint("/generate/multimodal", **kwargs)
    
    def generate_function_calls(self, **kwargs):
        """Generate function call synthetic data (placeholder)"""
        return self._call_endpoint("/generate/function-calls", **kwargs)
    
    def generate_dialogue(self, **kwargs):
        """Generate dialogue synthetic data (placeholder)"""
        return self._call_endpoint("/generate/dialogue", **kwargs)
    
    def generate_memory(self, **kwargs):
        """Generate memory scenario synthetic data (placeholder)"""
        return self._call_endpoint("/generate/memory", **kwargs)
    
    def generate_adversarial(self, **kwargs):
        """Generate adversarial synthetic data (placeholder)"""
        return self._call_endpoint("/generate/adversarial", **kwargs)
    
    def _call_endpoint(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Generic endpoint caller"""
        try:
            response = self.session.post(
                f"{self.base_url}{endpoint}",
                json=kwargs
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise SynthCodeException(f"API request to {endpoint} failed: {e}")
    
    def get_templates(self) -> Dict[str, Any]:
        """Get available code generation templates"""
        try:
            response = self.session.get(f"{self.base_url}/generate/code/templates")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise SynthCodeException(f"Failed to get templates: {e}")
    
    def validate_code(self, code: str, language: str) -> Dict[str, Any]:
        """Validate code syntax and compilation"""
        payload = {
            "code": code,
            "language": language
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/validate/code",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise ValidationException(f"Validation failed: {e}")
