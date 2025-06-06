# src/generators/openapi_generator.py
import yaml
import json
from typing import Dict, Any, List
from .base_generator import BaseCodeGenerator

class OpenAPIGenerator(BaseCodeGenerator):
    """Generate API definitions and mock implementations"""
    
    def generate_openapi_spec(self, api_description: str) -> Dict[str, Any]:
        """Generate OpenAPI 3.0 specification"""
        return {
            "openapi": "3.0.0",
            "info": {
                "title": "Generated API",
                "description": api_description,
                "version": "1.0.0"
            },
            "paths": self._generate_paths(api_description),
            "components": {
                "schemas": self._generate_schemas()
            }
        }
    
    def _generate_paths(self, description: str) -> Dict[str, Any]:
        """Generate API paths based on description"""
        # This would typically use an LLM to parse the description
        # For now, return a template
        return {
            "/users": {
                "get": {
                    "summary": "Get all users",
                    "responses": {
                        "200": {
                            "description": "Success",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {"$ref": "#/components/schemas/User"}
                                    }
                                }
                            }
                        }
                    }
                },
                "post": {
                    "summary": "Create user",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/User"}
                            }
                        }
                    },
                    "responses": {
                        "201": {"description": "Created"}
                    }
                }
            }
        }
    
    def _generate_schemas(self) -> Dict[str, Any]:
        """Generate data schemas"""
        return {
            "User": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "email": {"type": "string", "format": "email"}
                }
            }
        }
    
    def generate_function(self, prompt: str, **kwargs) -> str:
        """Generate OpenAPI spec as YAML string"""
        spec = self.generate_openapi_spec(prompt)
        return yaml.dump(spec, default_flow_style=False)
    
    def generate_class(self, prompt: str, **kwargs) -> str:
        """Generate API client class"""
        return f"""
class APIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    def get_users(self):
        # Implementation for {prompt}
        pass
    
    def create_user(self, user_data):
        # Implementation for {prompt}
        pass
"""
    
    def generate_module(self, prompt: str, **kwargs) -> str:
        """Generate complete API module"""
        return self.generate_function(prompt, **kwargs)
    
    def generate_api_endpoint(self, schema: Dict[str, Any]) -> str:
        """Generate API endpoint from schema"""
        return yaml.dump(schema, default_flow_style=False)
