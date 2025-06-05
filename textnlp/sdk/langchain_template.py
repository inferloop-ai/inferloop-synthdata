# sdk/langchain_template.py
import json
from typing import Dict, Any, List
from string import Template
import logging

logger = logging.getLogger(__name__)

class LangChainTemplate:
    """LangChain-style prompt template handler"""
    
    def __init__(self, template_path: str = None, template_dict: Dict = None):
        if template_path:
            self.template_data = self._load_template(template_path)
        elif template_dict:
            self.template_data = template_dict
        else:
            raise ValueError("Either template_path or template_dict must be provided")
        
        self.template = Template(self.template_data.get('template', ''))
        self.variables = self.template_data.get('variables', [])
    
    def _load_template(self, path: str) -> Dict:
        """Load template from JSON file"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load template from {path}: {e}")
            raise
    
    def format(self, **kwargs) -> str:
        """Format template with provided variables"""
        try:
            # Validate required variables
            missing_vars = set(self.variables) - set(kwargs.keys())
            if missing_vars:
                logger.warning(f"Missing variables: {missing_vars}")
            
            return self.template.safe_substitute(**kwargs)
        except Exception as e:
            logger.error(f"Template formatting failed: {e}")
            return ""
    
    def batch_format(self, variable_sets: List[Dict]) -> List[str]:
        """Format template with multiple variable sets"""
        return [self.format(**var_set) for var_set in variable_sets]
