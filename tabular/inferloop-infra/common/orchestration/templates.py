"""
Template engine for deployment configurations
"""

import os
import re
import json
import yaml
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import jinja2
from jinja2 import Environment, FileSystemLoader, select_autoescape


@dataclass
class TemplateVariable:
    """Definition of a template variable"""
    name: str
    description: str
    type: str
    default: Any = None
    required: bool = True
    validation: Optional[str] = None
    choices: Optional[List[Any]] = None


@dataclass
class DeploymentTemplate:
    """Deployment template definition"""
    name: str
    version: str
    description: str
    provider: str
    variables: List[TemplateVariable] = field(default_factory=list)
    resources: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_file(cls, file_path: str) -> 'DeploymentTemplate':
        """Load template from file"""
        with open(file_path, 'r') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        # Parse variables
        variables = []
        for var_data in data.get('variables', []):
            variables.append(TemplateVariable(**var_data))
        
        return cls(
            name=data['name'],
            version=data['version'],
            description=data['description'],
            provider=data['provider'],
            variables=variables,
            resources=data.get('resources', {}),
            outputs=data.get('outputs', {}),
            metadata=data.get('metadata', {})
        )
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate template inputs"""
        errors = []
        
        # Check required variables
        for var in self.variables:
            if var.required and var.name not in inputs:
                if var.default is None:
                    errors.append(f"Required variable '{var.name}' not provided")
            
            if var.name in inputs:
                value = inputs[var.name]
                
                # Type validation
                if var.type == 'string' and not isinstance(value, str):
                    errors.append(f"Variable '{var.name}' must be a string")
                elif var.type == 'number' and not isinstance(value, (int, float)):
                    errors.append(f"Variable '{var.name}' must be a number")
                elif var.type == 'boolean' and not isinstance(value, bool):
                    errors.append(f"Variable '{var.name}' must be a boolean")
                elif var.type == 'list' and not isinstance(value, list):
                    errors.append(f"Variable '{var.name}' must be a list")
                
                # Choices validation
                if var.choices and value not in var.choices:
                    errors.append(f"Variable '{var.name}' must be one of: {var.choices}")
                
                # Custom validation
                if var.validation:
                    try:
                        if not re.match(var.validation, str(value)):
                            errors.append(f"Variable '{var.name}' does not match pattern: {var.validation}")
                    except re.error:
                        errors.append(f"Invalid validation pattern for '{var.name}'")
        
        return errors


class TemplateEngine:
    """Engine for processing deployment templates"""
    
    def __init__(self, template_dirs: Optional[List[str]] = None):
        self.template_dirs = template_dirs or []
        self._templates: Dict[str, DeploymentTemplate] = {}
        self._jinja_env = self._create_jinja_environment()
    
    def _create_jinja_environment(self) -> Environment:
        """Create Jinja2 environment with custom filters"""
        env = Environment(
            loader=FileSystemLoader(self.template_dirs),
            autoescape=select_autoescape(['html', 'xml']),
            undefined=jinja2.StrictUndefined
        )
        
        # Add custom filters
        env.filters['to_json'] = json.dumps
        env.filters['to_yaml'] = yaml.dump
        env.filters['b64encode'] = lambda x: __import__('base64').b64encode(x.encode()).decode()
        env.filters['b64decode'] = lambda x: __import__('base64').b64decode(x.encode()).decode()
        
        # Add custom functions
        env.globals['env'] = os.environ.get
        env.globals['now'] = lambda: __import__('datetime').datetime.now().isoformat()
        env.globals['uuid'] = lambda: str(__import__('uuid').uuid4())
        
        return env
    
    def add_template_directory(self, directory: str) -> None:
        """Add template directory"""
        if os.path.isdir(directory) and directory not in self.template_dirs:
            self.template_dirs.append(directory)
            self._jinja_env = self._create_jinja_environment()
    
    def load_template(self, name: str, file_path: Optional[str] = None) -> DeploymentTemplate:
        """Load a deployment template"""
        if file_path:
            template = DeploymentTemplate.from_file(file_path)
        else:
            # Search in template directories
            for template_dir in self.template_dirs:
                path = Path(template_dir) / f"{name}.yaml"
                if path.exists():
                    template = DeploymentTemplate.from_file(str(path))
                    break
            else:
                raise FileNotFoundError(f"Template '{name}' not found")
        
        self._templates[template.name] = template
        return template
    
    def render_template(self, template_name: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Render a deployment template with variables"""
        if template_name not in self._templates:
            self.load_template(template_name)
        
        template = self._templates[template_name]
        
        # Validate inputs
        errors = template.validate_inputs(variables)
        if errors:
            raise ValueError(f"Template validation failed: {errors}")
        
        # Apply defaults
        context = {}
        for var in template.variables:
            if var.name in variables:
                context[var.name] = variables[var.name]
            elif var.default is not None:
                context[var.name] = var.default
        
        # Render resources
        rendered_resources = self._render_dict(template.resources, context)
        
        # Render outputs
        rendered_outputs = self._render_dict(template.outputs, context)
        
        return {
            'name': template.name,
            'version': template.version,
            'provider': template.provider,
            'resources': rendered_resources,
            'outputs': rendered_outputs,
            'metadata': {
                **template.metadata,
                'rendered_at': __import__('datetime').datetime.now().isoformat(),
                'variables': context
            }
        }
    
    def _render_dict(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively render dictionary with Jinja2 templates"""
        rendered = {}
        
        for key, value in data.items():
            rendered_key = self._render_string(key, context) if isinstance(key, str) else key
            
            if isinstance(value, dict):
                rendered[rendered_key] = self._render_dict(value, context)
            elif isinstance(value, list):
                rendered[rendered_key] = self._render_list(value, context)
            elif isinstance(value, str):
                rendered[rendered_key] = self._render_string(value, context)
            else:
                rendered[rendered_key] = value
        
        return rendered
    
    def _render_list(self, data: List[Any], context: Dict[str, Any]) -> List[Any]:
        """Recursively render list with Jinja2 templates"""
        rendered = []
        
        for item in data:
            if isinstance(item, dict):
                rendered.append(self._render_dict(item, context))
            elif isinstance(item, list):
                rendered.append(self._render_list(item, context))
            elif isinstance(item, str):
                rendered.append(self._render_string(item, context))
            else:
                rendered.append(item)
        
        return rendered
    
    def _render_string(self, value: str, context: Dict[str, Any]) -> Union[str, Any]:
        """Render string with Jinja2 template"""
        try:
            template = self._jinja_env.from_string(value)
            rendered = template.render(**context)
            
            # Try to parse as JSON if it looks like JSON
            if rendered.strip().startswith('{') or rendered.strip().startswith('['):
                try:
                    return json.loads(rendered)
                except json.JSONDecodeError:
                    pass
            
            return rendered
        except jinja2.exceptions.TemplateSyntaxError:
            # Not a template, return as-is
            return value
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List available templates"""
        templates = []
        
        # From loaded templates
        for name, template in self._templates.items():
            templates.append({
                'name': template.name,
                'version': template.version,
                'description': template.description,
                'provider': template.provider,
                'variables': len(template.variables)
            })
        
        # From template directories
        for template_dir in self.template_dirs:
            for file_path in Path(template_dir).glob('*.yaml'):
                if file_path.stem not in self._templates:
                    try:
                        template = DeploymentTemplate.from_file(str(file_path))
                        templates.append({
                            'name': template.name,
                            'version': template.version,
                            'description': template.description,
                            'provider': template.provider,
                            'variables': len(template.variables)
                        })
                    except Exception:
                        pass
        
        return templates
    
    def get_template_schema(self, template_name: str) -> Dict[str, Any]:
        """Get template schema including variables"""
        if template_name not in self._templates:
            self.load_template(template_name)
        
        template = self._templates[template_name]
        
        return {
            'name': template.name,
            'version': template.version,
            'description': template.description,
            'provider': template.provider,
            'variables': [
                {
                    'name': var.name,
                    'description': var.description,
                    'type': var.type,
                    'default': var.default,
                    'required': var.required,
                    'validation': var.validation,
                    'choices': var.choices
                }
                for var in template.variables
            ]
        }
    
    def validate_rendered_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate rendered configuration"""
        errors = []
        
        # Basic validation
        if 'resources' not in config:
            errors.append("Configuration missing 'resources' section")
        
        if 'provider' not in config:
            errors.append("Configuration missing 'provider'")
        
        # Validate each resource
        for name, resource in config.get('resources', {}).items():
            if 'type' not in resource:
                errors.append(f"Resource '{name}' missing 'type'")
            if 'config' not in resource:
                errors.append(f"Resource '{name}' missing 'config'")
        
        return errors