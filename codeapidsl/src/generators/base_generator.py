# src/generators/base_generator.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass

@dataclass
class GenerationConfig:
    language: str
    framework: Optional[str] = None
    complexity: str = "medium"  # low, medium, high
    count: int = 100
    include_tests: bool = True
    include_docs: bool = True
    style_guide: Optional[str] = None

class BaseCodeGenerator(ABC):
    """Base class for all code generators"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def generate_function(self, prompt: str, **kwargs) -> str:
        """Generate a single function"""
        pass
    
    @abstractmethod
    def generate_class(self, prompt: str, **kwargs) -> str:
        """Generate a class with methods"""
        pass
    
    @abstractmethod
    def generate_module(self, prompt: str, **kwargs) -> str:
        """Generate a complete module"""
        pass
    
    @abstractmethod
    def generate_api_endpoint(self, schema: Dict[str, Any]) -> str:
        """Generate API endpoint code"""
        pass
    
    def generate_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Generate multiple code samples"""
        results = []
        for i, prompt in enumerate(prompts):
            try:
                code = self.generate_function(prompt)
                result = {
                    "id": f"sample_{i}",
                    "prompt": prompt,
                    "code": code,
                    "language": self.config.language,
                    "metadata": self._extract_metadata(code)
                }
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to generate code for prompt {i}: {e}")
                continue
        return results
    
    def _extract_metadata(self, code: str) -> Dict[str, Any]:
        """Extract metadata from generated code"""
        return {
            "lines_of_code": len(code.split('\n')),
            "estimated_complexity": self._estimate_complexity(code),
            "dependencies": self._extract_dependencies(code)
        }
    
    def _estimate_complexity(self, code: str) -> str:
        """Estimate code complexity based on patterns"""
        lines = len(code.split('\n'))
        if lines < 10:
            return "low"
        elif lines < 50:
            return "medium"
        else:
            return "high"
    
    def _extract_dependencies(self, code: str) -> List[str]:
        """Extract import statements and dependencies"""
        dependencies = []
        for line in code.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                dependencies.append(line)
        return dependencies
