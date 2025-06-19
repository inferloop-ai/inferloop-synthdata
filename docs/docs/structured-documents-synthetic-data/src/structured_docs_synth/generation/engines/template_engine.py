"""
Template engine for structured document generation
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound
from faker import Faker

from ...core import (
    get_logger,
    get_config,
    TemplateNotFoundError,
    TemplateRenderingError,
    ValidationError,
    get_document_type_config
)


class TemplateEngine:
    """Core template engine for document generation"""
    
    def __init__(self, template_dir: Optional[str] = None):
        self.logger = get_logger(__name__)
        self.config = get_config()
        
        # Set up template directory
        if template_dir:
            self.template_dir = Path(template_dir)
        else:
            # Default to configs/schema_bank directory
            self.template_dir = Path(__file__).parent.parent.parent.parent.parent / "configs" / "schema_bank"
        
        if not self.template_dir.exists():
            self.template_dir.mkdir(parents=True, exist_ok=True)
            self.logger.warning(f"Template directory created: {self.template_dir}")
        
        # Set up Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=False,  # We control the content
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Initialize Faker for data generation
        self.faker = Faker()
        
        # Add custom filters and functions
        self._setup_template_functions()
        
        self.logger.info(f"Template engine initialized with directory: {self.template_dir}")
    
    def _setup_template_functions(self):
        """Set up custom Jinja2 filters and functions"""
        
        # Custom filters
        self.jinja_env.filters['currency'] = lambda x: f"${x:,.2f}"
        self.jinja_env.filters['phone'] = lambda x: f"({x[:3]}) {x[3:6]}-{x[6:]}" if len(x) == 10 else x
        self.jinja_env.filters['ssn'] = lambda x: f"{x[:3]}-{x[3:5]}-{x[5:]}" if len(x) == 9 else x
        self.jinja_env.filters['upper_first'] = lambda x: x.capitalize() if x else ""
        
        # Faker functions available in templates
        self.jinja_env.globals['fake'] = self.faker
        self.jinja_env.globals['fake_name'] = self.faker.name
        self.jinja_env.globals['fake_address'] = self.faker.address
        self.jinja_env.globals['fake_phone'] = self.faker.phone_number
        self.jinja_env.globals['fake_email'] = self.faker.email
        self.jinja_env.globals['fake_company'] = self.faker.company
        self.jinja_env.globals['fake_date'] = self.faker.date
        self.jinja_env.globals['fake_text'] = self.faker.text
        
        # Utility functions
        self.jinja_env.globals['range'] = range
        self.jinja_env.globals['len'] = len
        self.jinja_env.globals['enumerate'] = enumerate
    
    def load_template_config(self, document_type: str) -> Dict[str, Any]:
        """Load template configuration for document type"""
        try:
            doc_config = get_document_type_config(document_type)
            template_path = Path(self.template_dir) / doc_config['template_path']
            
            if not template_path.exists():
                # Create a default template if it doesn't exist
                self._create_default_template(document_type, template_path)
            
            with open(template_path, 'r') as f:
                template_config = yaml.safe_load(f)
            
            self.logger.debug(f"Loaded template config for {document_type}")
            return template_config
            
        except Exception as e:
            raise TemplateNotFoundError(
                f"Failed to load template for {document_type}: {str(e)}",
                details={'document_type': document_type, 'template_path': str(template_path)}
            )
    
    def _create_default_template(self, document_type: str, template_path: Path):
        """Create a default template for the document type"""
        
        default_templates = {
            'legal_contract': {
                'name': 'Legal Contract Template',
                'version': '1.0',
                'description': 'Basic legal contract template',
                'sections': [
                    {
                        'name': 'header',
                        'template': '''
                        CONTRACT AGREEMENT
                        
                        This Agreement is made on {{ effective_date }} between:
                        
                        Party 1: {{ parties[0] }}
                        Party 2: {{ parties[1] }}
                        '''
                    },
                    {
                        'name': 'terms',
                        'template': '''
                        TERMS AND CONDITIONS
                        
                        1. Scope of Work: {{ terms.scope | default("To be defined") }}
                        2. Duration: {{ terms.duration | default("12 months") }}
                        3. Payment: {{ terms.payment | default("As agreed") }}
                        4. Jurisdiction: {{ jurisdiction }}
                        '''
                    },
                    {
                        'name': 'signatures',
                        'template': '''
                        SIGNATURES
                        
                        Party 1: _________________    Date: _______
                        
                        Party 2: _________________    Date: _______
                        '''
                    }
                ],
                'required_fields': ['parties', 'effective_date', 'jurisdiction'],
                'optional_fields': ['terms']
            },
            
            'medical_form': {
                'name': 'Medical Form Template',
                'version': '1.0',
                'description': 'Basic medical form template',
                'sections': [
                    {
                        'name': 'patient_info',
                        'template': '''
                        PATIENT INFORMATION
                        
                        Name: {{ patient_name }}
                        Date of Birth: {{ date_of_birth }}
                        Medical Record Number: {{ medical_record_number }}
                        
                        {% if insurance_info %}
                        Insurance: {{ insurance_info.provider }} - {{ insurance_info.policy_number }}
                        {% endif %}
                        '''
                    },
                    {
                        'name': 'medical_history',
                        'template': '''
                        MEDICAL HISTORY
                        
                        {% if medical_history %}
                        {% for condition in medical_history %}
                        - {{ condition }}
                        {% endfor %}
                        {% else %}
                        No significant medical history reported.
                        {% endif %}
                        '''
                    }
                ],
                'required_fields': ['patient_name', 'date_of_birth', 'medical_record_number'],
                'optional_fields': ['insurance_info', 'medical_history']
            },
            
            'loan_application': {
                'name': 'Loan Application Template',
                'version': '1.0',
                'description': 'Basic loan application template',
                'sections': [
                    {
                        'name': 'applicant_info',
                        'template': '''
                        LOAN APPLICATION
                        
                        Applicant Name: {{ applicant_name }}
                        Loan Amount Requested: {{ loan_amount | currency }}
                        Annual Income: {{ income | currency }}
                        
                        {% if collateral %}
                        Collateral: {{ collateral }}
                        {% endif %}
                        '''
                    },
                    {
                        'name': 'terms',
                        'template': '''
                        LOAN TERMS
                        
                        Application Date: {{ fake_date() }}
                        Loan Purpose: {{ loan_purpose | default("General") }}
                        Requested Term: {{ term | default("60 months") }}
                        '''
                    }
                ],
                'required_fields': ['applicant_name', 'loan_amount', 'income'],
                'optional_fields': ['collateral', 'loan_purpose', 'term']
            },
            
            'tax_form': {
                'name': 'Tax Form Template', 
                'version': '1.0',
                'description': 'Basic tax form template',
                'sections': [
                    {
                        'name': 'taxpayer_info',
                        'template': '''
                        TAX FORM {{ tax_year }}
                        
                        Taxpayer Name: {{ taxpayer_name }}
                        SSN: {{ ssn | ssn }}
                        Filing Status: {{ filing_status | default("Single") }}
                        
                        {% if dependents %}
                        Number of Dependents: {{ dependents | length }}
                        {% endif %}
                        '''
                    },
                    {
                        'name': 'income',
                        'template': '''
                        INCOME INFORMATION
                        
                        Total Income: {{ total_income | currency }}
                        
                        {% if deductions %}
                        Total Deductions: {{ deductions | currency }}
                        {% endif %}
                        '''
                    }
                ],
                'required_fields': ['taxpayer_name', 'ssn', 'tax_year'],
                'optional_fields': ['filing_status', 'dependents', 'total_income', 'deductions']
            }
        }
        
        if document_type in default_templates:
            template_path.parent.mkdir(parents=True, exist_ok=True)
            with open(template_path, 'w') as f:
                yaml.dump(default_templates[document_type], f, default_flow_style=False)
            
            self.logger.info(f"Created default template: {template_path}")
    
    def validate_template_data(self, document_type: str, data: Dict[str, Any]) -> List[str]:
        """Validate that required fields are present in data"""
        errors = []
        
        try:
            doc_config = get_document_type_config(document_type)
            required_fields = doc_config.get('required_fields', [])
            
            for field in required_fields:
                if field not in data or data[field] is None:
                    errors.append(f"Missing required field: {field}")
                elif isinstance(data[field], str) and not data[field].strip():
                    errors.append(f"Required field '{field}' cannot be empty")
            
        except Exception as e:
            errors.append(f"Error validating template data: {str(e)}")
        
        return errors
    
    def render_template(self, document_type: str, data: Dict[str, Any]) -> str:
        """Render complete document from template"""
        
        try:
            # Validate input data
            validation_errors = self.validate_template_data(document_type, data)
            if validation_errors:
                raise ValidationError(
                    f"Template validation failed for {document_type}",
                    validation_errors=validation_errors
                )
            
            # Load template configuration
            template_config = self.load_template_config(document_type)
            
            # Render each section
            rendered_sections = []
            sections = template_config.get('sections', [])
            
            for section in sections:
                section_name = section.get('name', 'unknown')
                section_template = section.get('template', '')
                
                try:
                    # Create Jinja2 template
                    jinja_template = Template(section_template, environment=self.jinja_env)
                    
                    # Render with data
                    rendered_content = jinja_template.render(**data)
                    rendered_sections.append(rendered_content.strip())
                    
                    self.logger.debug(f"Rendered section: {section_name}")
                    
                except Exception as e:
                    self.logger.error(f"Error rendering section {section_name}: {str(e)}")
                    raise TemplateRenderingError(
                        f"Failed to render section '{section_name}': {str(e)}",
                        details={
                            'document_type': document_type,
                            'section': section_name,
                            'error': str(e)
                        }
                    )
            
            # Combine all sections
            final_document = '\n\n'.join(rendered_sections)
            
            self.logger.info(f"Successfully rendered {document_type} template")
            return final_document
            
        except (ValidationError, TemplateRenderingError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            raise TemplateRenderingError(
                f"Unexpected error rendering template for {document_type}: {str(e)}",
                details={'document_type': document_type, 'error': str(e)}
            )
    
    def get_template_fields(self, document_type: str) -> Dict[str, List[str]]:
        """Get required and optional fields for a document type"""
        try:
            doc_config = get_document_type_config(document_type)
            return {
                'required_fields': doc_config.get('required_fields', []),
                'optional_fields': doc_config.get('optional_fields', [])
            }
        except Exception as e:
            self.logger.error(f"Error getting template fields for {document_type}: {str(e)}")
            return {'required_fields': [], 'optional_fields': []}
    
    def generate_sample_data(self, document_type: str) -> Dict[str, Any]:
        """Generate sample data for a document type using Faker"""
        try:
            doc_config = get_document_type_config(document_type)
            required_fields = doc_config.get('required_fields', [])
            optional_fields = doc_config.get('optional_fields', [])
            
            sample_data = {}
            
            # Generate data for required fields
            for field in required_fields:
                sample_data[field] = self._generate_field_data(field, document_type)
            
            # Generate data for some optional fields (50% chance)
            for field in optional_fields:
                if self.faker.boolean(chance_of_getting_true=50):
                    sample_data[field] = self._generate_field_data(field, document_type)
            
            return sample_data
            
        except Exception as e:
            self.logger.error(f"Error generating sample data for {document_type}: {str(e)}")
            return {}
    
    def _generate_field_data(self, field_name: str, document_type: str) -> Any:
        """Generate appropriate fake data for a field"""
        
        # Common field patterns
        if 'name' in field_name.lower():
            if field_name == 'patient_name' or field_name == 'applicant_name' or field_name == 'taxpayer_name':
                return self.faker.name()
            elif field_name == 'company_name':
                return self.faker.company()
            else:
                return self.faker.name()
        
        elif 'date' in field_name.lower():
            if 'birth' in field_name.lower():
                return self.faker.date_of_birth(minimum_age=18, maximum_age=80).strftime('%Y-%m-%d')
            else:
                return self.faker.date(pattern='%Y-%m-%d')
        
        elif 'phone' in field_name.lower():
            return self.faker.phone_number()
        
        elif 'email' in field_name.lower():
            return self.faker.email()
        
        elif 'address' in field_name.lower():
            return self.faker.address()
        
        elif 'ssn' in field_name.lower():
            return self.faker.ssn()
        
        elif 'amount' in field_name.lower() or 'income' in field_name.lower():
            return self.faker.random_int(min=1000, max=100000)
        
        elif field_name == 'parties':
            return [self.faker.company(), self.faker.company()]
        
        elif field_name == 'jurisdiction':
            return self.faker.state()
        
        elif field_name == 'medical_record_number':
            return self.faker.random_number(digits=8)
        
        elif field_name == 'tax_year':
            return self.faker.random_int(min=2020, max=2024)
        
        else:
            # Default to text
            return self.faker.text(max_nb_chars=50)


# Global template engine instance
_template_engine: Optional[TemplateEngine] = None


def get_template_engine(template_dir: Optional[str] = None) -> TemplateEngine:
    """Get global template engine instance"""
    global _template_engine
    if _template_engine is None or template_dir:
        _template_engine = TemplateEngine(template_dir)
    return _template_engine