#!/usr/bin/env python3
"""
Template setup script for document generation templates.

Provides automated setup and management of document templates
for various document types and domains.
"""

import asyncio
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import yaml
from jinja2 import Environment, FileSystemLoader, Template

# Template configurations
TEMPLATE_CONFIGS = {
    'invoice': {
        'name': 'Invoice Template',
        'description': 'Standard business invoice template',
        'category': 'financial',
        'output_formats': ['pdf', 'html', 'docx'],
        'required_fields': ['invoice_number', 'date', 'customer', 'items', 'total'],
        'optional_fields': ['tax_rate', 'discount', 'payment_terms'],
        'template_files': {
            'html': 'invoice.html.jinja2',
            'css': 'invoice.css',
            'config': 'invoice_config.yaml'
        }
    },
    'receipt': {
        'name': 'Receipt Template',
        'description': 'Retail/restaurant receipt template',
        'category': 'financial',
        'output_formats': ['pdf', 'html'],
        'required_fields': ['receipt_number', 'date', 'items', 'total'],
        'optional_fields': ['tax', 'tip', 'payment_method'],
        'template_files': {
            'html': 'receipt.html.jinja2',
            'css': 'receipt.css',
            'config': 'receipt_config.yaml'
        }
    },
    'form': {
        'name': 'Form Template',
        'description': 'Generic form template with fields',
        'category': 'document',
        'output_formats': ['pdf', 'html', 'docx'],
        'required_fields': ['title', 'fields'],
        'optional_fields': ['instructions', 'signature_fields'],
        'template_files': {
            'html': 'form.html.jinja2',
            'css': 'form.css',
            'config': 'form_config.yaml'
        }
    },
    'report': {
        'name': 'Report Template',
        'description': 'Business report template',
        'category': 'document',
        'output_formats': ['pdf', 'html', 'docx'],
        'required_fields': ['title', 'sections'],
        'optional_fields': ['author', 'date', 'executive_summary'],
        'template_files': {
            'html': 'report.html.jinja2',
            'css': 'report.css',
            'config': 'report_config.yaml'
        }
    },
    'contract': {
        'name': 'Contract Template',
        'description': 'Legal contract template',
        'category': 'legal',
        'output_formats': ['pdf', 'docx'],
        'required_fields': ['parties', 'terms', 'date'],
        'optional_fields': ['witnesses', 'notary', 'appendices'],
        'template_files': {
            'html': 'contract.html.jinja2',
            'css': 'contract.css',
            'config': 'contract_config.yaml'
        }
    },
    'medical_record': {
        'name': 'Medical Record Template',
        'description': 'Healthcare medical record template',
        'category': 'healthcare',
        'output_formats': ['pdf', 'html'],
        'required_fields': ['patient_id', 'date', 'provider', 'diagnosis'],
        'optional_fields': ['medications', 'allergies', 'notes'],
        'template_files': {
            'html': 'medical_record.html.jinja2',
            'css': 'medical_record.css',
            'config': 'medical_record_config.yaml'
        }
    }
}

DEFAULT_TEMPLATE_DIR = Path.home() / '.structured_docs_synth' / 'templates'


class TemplateSetup:
    """Template setup and management system"""
    
    def __init__(self, template_dir: Optional[Path] = None):
        self.template_dir = template_dir or DEFAULT_TEMPLATE_DIR
        self.builtin_templates_dir = Path(__file__).parent / 'builtin_templates'
        
        # Ensure directories exist
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True
        )
    
    async def setup_all_templates(self, force_overwrite: bool = False) -> Dict[str, Any]:
        """
        Setup all available templates.
        
        Args:
            force_overwrite: Overwrite existing templates
        
        Returns:
            Setup results
        """
        print("=€ Setting up document templates...")
        
        results = {}
        successful = 0
        failed = 0
        
        for template_id in TEMPLATE_CONFIGS.keys():
            try:
                result = await self.setup_template(template_id, force_overwrite)
                results[template_id] = result
                
                if result['success']:
                    successful += 1
                    print(f" {template_id}: {result['message']}")
                else:
                    failed += 1
                    print(f"L {template_id}: {result['message']}")
                    
            except Exception as e:
                failed += 1
                results[template_id] = {'success': False, 'message': str(e)}
                print(f"L {template_id}: {e}")
        
        print(f"\n=Ê Setup Summary: {successful} successful, {failed} failed")
        
        return {
            'total_templates': len(TEMPLATE_CONFIGS),
            'successful': successful,
            'failed': failed,
            'results': results,
            'template_dir': str(self.template_dir)
        }
    
    async def setup_template(self, template_id: str, 
                           force_overwrite: bool = False) -> Dict[str, Any]:
        """
        Setup individual template.
        
        Args:
            template_id: Template identifier
            force_overwrite: Overwrite if exists
        
        Returns:
            Setup result
        """
        if template_id not in TEMPLATE_CONFIGS:
            return {
                'success': False,
                'message': f'Unknown template: {template_id}'
            }
        
        config = TEMPLATE_CONFIGS[template_id]
        template_path = self.template_dir / template_id
        
        # Check if template already exists
        if template_path.exists() and not force_overwrite:
            return {
                'success': True,
                'message': f'Template already exists at {template_path}'
            }
        
        try:
            # Create template directory
            template_path.mkdir(parents=True, exist_ok=True)
            
            # Generate template files
            await self._create_html_template(template_id, config, template_path)
            await self._create_css_file(template_id, config, template_path)
            await self._create_config_file(template_id, config, template_path)
            await self._create_sample_data(template_id, config, template_path)
            await self._create_readme(template_id, config, template_path)
            
            return {
                'success': True,
                'message': f'Template created at {template_path}',
                'files_created': len(config['template_files']) + 2  # +2 for sample and readme
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Failed to create template: {e}'
            }
    
    async def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all available templates.
        
        Returns:
            List of template information
        """
        templates = []
        
        for template_id, config in TEMPLATE_CONFIGS.items():
            template_path = self.template_dir / template_id
            
            template_info = {
                'id': template_id,
                'name': config['name'],
                'description': config['description'],
                'category': config['category'],
                'output_formats': config['output_formats'],
                'required_fields': config['required_fields'],
                'optional_fields': config['optional_fields'],
                'installed': template_path.exists(),
                'path': str(template_path) if template_path.exists() else None
            }
            
            if template_path.exists():
                # Get template metadata
                config_file = template_path / config['template_files']['config']
                if config_file.exists():
                    try:
                        with open(config_file, 'r') as f:
                            template_config = yaml.safe_load(f)
                        template_info['version'] = template_config.get('version', '1.0.0')
                        template_info['created_at'] = template_config.get('created_at')
                    except Exception:
                        pass
            
            templates.append(template_info)
        
        return templates
    
    async def validate_template(self, template_id: str) -> Dict[str, Any]:
        """
        Validate template installation and structure.
        
        Args:
            template_id: Template identifier
        
        Returns:
            Validation results
        """
        if template_id not in TEMPLATE_CONFIGS:
            return {
                'valid': False,
                'errors': [f'Unknown template: {template_id}']
            }
        
        config = TEMPLATE_CONFIGS[template_id]
        template_path = self.template_dir / template_id
        
        errors = []
        warnings = []
        
        # Check if template directory exists
        if not template_path.exists():
            errors.append(f'Template directory not found: {template_path}')
            return {'valid': False, 'errors': errors}
        
        # Check required files
        for file_type, filename in config['template_files'].items():
            file_path = template_path / filename
            if not file_path.exists():
                errors.append(f'Missing {file_type} file: {filename}')
        
        # Validate HTML template syntax
        html_file = template_path / config['template_files']['html']
        if html_file.exists():
            try:
                with open(html_file, 'r') as f:
                    template_content = f.read()
                
                # Try to parse template
                self.jinja_env.from_string(template_content)
                
            except Exception as e:
                errors.append(f'Invalid Jinja2 template syntax: {e}')
        
        # Validate config file
        config_file = template_path / config['template_files']['config']
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    yaml.safe_load(f)
            except Exception as e:
                errors.append(f'Invalid YAML config: {e}')
        
        # Check sample data
        sample_file = template_path / 'sample_data.json'
        if sample_file.exists():
            try:
                with open(sample_file, 'r') as f:
                    sample_data = json.load(f)
                
                # Check if required fields are present
                for field in config['required_fields']:
                    if field not in sample_data:
                        warnings.append(f'Sample data missing required field: {field}')
                        
            except Exception as e:
                errors.append(f'Invalid sample data: {e}')
        else:
            warnings.append('No sample data file found')
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'template_path': str(template_path)
        }
    
    async def test_template(self, template_id: str) -> Dict[str, Any]:
        """
        Test template rendering with sample data.
        
        Args:
            template_id: Template identifier
        
        Returns:
            Test results
        """
        validation = await self.validate_template(template_id)
        
        if not validation['valid']:
            return {
                'success': False,
                'message': 'Template validation failed',
                'errors': validation['errors']
            }
        
        try:
            config = TEMPLATE_CONFIGS[template_id]
            template_path = self.template_dir / template_id
            
            # Load sample data
            sample_file = template_path / 'sample_data.json'
            if not sample_file.exists():
                return {
                    'success': False,
                    'message': 'No sample data file found'
                }
            
            with open(sample_file, 'r') as f:
                sample_data = json.load(f)
            
            # Load and render template
            template_file = config['template_files']['html']
            template = self.jinja_env.get_template(f'{template_id}/{template_file}')
            
            rendered_html = template.render(**sample_data)
            
            # Save test output
            test_output_file = template_path / 'test_output.html'
            with open(test_output_file, 'w') as f:
                f.write(rendered_html)
            
            return {
                'success': True,
                'message': f'Template rendered successfully',
                'output_file': str(test_output_file),
                'output_size': len(rendered_html)
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Template rendering failed: {e}'
            }
    
    async def remove_template(self, template_id: str) -> Dict[str, Any]:
        """
        Remove template.
        
        Args:
            template_id: Template identifier
        
        Returns:
            Removal result
        """
        template_path = self.template_dir / template_id
        
        if not template_path.exists():
            return {
                'success': False,
                'message': f'Template not found: {template_id}'
            }
        
        try:
            shutil.rmtree(template_path)
            return {
                'success': True,
                'message': f'Template removed: {template_id}'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Failed to remove template: {e}'
            }
    
    # Template generation methods
    
    async def _create_html_template(self, template_id: str, config: Dict[str, Any], 
                                  template_path: Path):
        """Create HTML template file"""
        template_content = self._generate_html_template(template_id, config)
        
        html_file = template_path / config['template_files']['html']
        with open(html_file, 'w') as f:
            f.write(template_content)
    
    async def _create_css_file(self, template_id: str, config: Dict[str, Any], 
                             template_path: Path):
        """Create CSS file"""
        css_content = self._generate_css_styles(template_id, config)
        
        css_file = template_path / config['template_files']['css']
        with open(css_file, 'w') as f:
            f.write(css_content)
    
    async def _create_config_file(self, template_id: str, config: Dict[str, Any], 
                                template_path: Path):
        """Create template configuration file"""
        template_config = {
            'template_id': template_id,
            'name': config['name'],
            'description': config['description'],
            'category': config['category'],
            'version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'output_formats': config['output_formats'],
            'required_fields': config['required_fields'],
            'optional_fields': config['optional_fields'],
            'settings': {
                'page_size': 'A4',
                'margins': '2cm',
                'font_family': 'Arial, sans-serif',
                'font_size': '12pt'
            }
        }
        
        config_file = template_path / config['template_files']['config']
        with open(config_file, 'w') as f:
            yaml.dump(template_config, f, default_flow_style=False, indent=2)
    
    async def _create_sample_data(self, template_id: str, config: Dict[str, Any], 
                                template_path: Path):
        """Create sample data file"""
        sample_data = self._generate_sample_data(template_id, config)
        
        sample_file = template_path / 'sample_data.json'
        with open(sample_file, 'w') as f:
            json.dump(sample_data, f, indent=2, default=str)
    
    async def _create_readme(self, template_id: str, config: Dict[str, Any], 
                           template_path: Path):
        """Create README file"""
        readme_content = f"""# {config['name']}

{config['description']}

## Category
{config['category']}

## Output Formats
{', '.join(config['output_formats'])}

## Required Fields
{', '.join(config['required_fields'])}

## Optional Fields
{', '.join(config['optional_fields'])}

## Usage

1. Load the template:
   ```python
   from structured_docs_synth.generation.engines import TemplateEngine
   
   engine = TemplateEngine()
   template = engine.load_template('{template_id}')
   ```

2. Render with data:
   ```python
   data = {{
       # Your data here
   }}
   
   output = template.render(data)
   ```

## Files

- `{config['template_files']['html']}` - Main Jinja2 template
- `{config['template_files']['css']}` - Styling
- `{config['template_files']['config']}` - Configuration
- `sample_data.json` - Sample data for testing
- `README.md` - This file

## Generated

This template was automatically generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.
"""
        
        readme_file = template_path / 'README.md'
        with open(readme_file, 'w') as f:
            f.write(readme_content)
    
    def _generate_html_template(self, template_id: str, config: Dict[str, Any]) -> str:
        """Generate HTML template content based on template type"""
        
        if template_id == 'invoice':
            return '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Invoice #{{ invoice_number }}</title>
    <link rel="stylesheet" href="invoice.css">
</head>
<body>
    <div class="invoice">
        <header>
            <h1>INVOICE</h1>
            <div class="invoice-info">
                <p><strong>Invoice #:</strong> {{ invoice_number }}</p>
                <p><strong>Date:</strong> {{ date }}</p>
            </div>
        </header>
        
        <section class="customer-info">
            <h2>Bill To:</h2>
            <div class="customer">
                <p><strong>{{ customer.name }}</strong></p>
                <p>{{ customer.address }}</p>
                <p>{{ customer.city }}, {{ customer.state }} {{ customer.zip }}</p>
            </div>
        </section>
        
        <section class="items">
            <table>
                <thead>
                    <tr>
                        <th>Description</th>
                        <th>Quantity</th>
                        <th>Unit Price</th>
                        <th>Total</th>
                    </tr>
                </thead>
                <tbody>
                    {% for item in items %}
                    <tr>
                        <td>{{ item.description }}</td>
                        <td>{{ item.quantity }}</td>
                        <td>${{ "%.2f"|format(item.unit_price) }}</td>
                        <td>${{ "%.2f"|format(item.total) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </section>
        
        <section class="totals">
            <div class="total-line">
                <span>Subtotal:</span>
                <span>${{ "%.2f"|format(subtotal) }}</span>
            </div>
            {% if tax_rate %}
            <div class="total-line">
                <span>Tax ({{ tax_rate }}%):</span>
                <span>${{ "%.2f"|format(tax_amount) }}</span>
            </div>
            {% endif %}
            <div class="total-line total">
                <span>Total:</span>
                <span>${{ "%.2f"|format(total) }}</span>
            </div>
        </section>
        
        {% if payment_terms %}
        <section class="payment-terms">
            <h3>Payment Terms</h3>
            <p>{{ payment_terms }}</p>
        </section>
        {% endif %}
    </div>
</body>
</html>
'''
        
        elif template_id == 'receipt':
            return '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Receipt #{{ receipt_number }}</title>
    <link rel="stylesheet" href="receipt.css">
</head>
<body>
    <div class="receipt">
        <header>
            <h1>{{ store_name | default("Store Name") }}</h1>
            <p>{{ store_address | default("Store Address") }}</p>
            <p>Receipt #: {{ receipt_number }}</p>
            <p>Date: {{ date }}</p>
        </header>
        
        <section class="items">
            {% for item in items %}
            <div class="item">
                <span class="name">{{ item.name }}</span>
                <span class="price">${{ "%.2f"|format(item.price) }}</span>
            </div>
            {% endfor %}
        </section>
        
        <section class="totals">
            <div class="total-line">
                <span>Subtotal:</span>
                <span>${{ "%.2f"|format(subtotal) }}</span>
            </div>
            {% if tax %}
            <div class="total-line">
                <span>Tax:</span>
                <span>${{ "%.2f"|format(tax) }}</span>
            </div>
            {% endif %}
            {% if tip %}
            <div class="total-line">
                <span>Tip:</span>
                <span>${{ "%.2f"|format(tip) }}</span>
            </div>
            {% endif %}
            <div class="total-line total">
                <span>Total:</span>
                <span>${{ "%.2f"|format(total) }}</span>
            </div>
        </section>
        
        {% if payment_method %}
        <section class="payment">
            <p>Payment: {{ payment_method }}</p>
        </section>
        {% endif %}
        
        <footer>
            <p>Thank you for your business!</p>
        </footer>
    </div>
</body>
</html>
'''
        
        else:
            # Generic template
            return f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{{{{ title | default("{config['name']}") }}}}</title>
    <link rel="stylesheet" href="{template_id}.css">
</head>
<body>
    <div class="document">
        <header>
            <h1>{{{{ title }}</h1>
        </header>
        
        <main>
            <!-- Template content goes here -->
            <p>This is a generic template for {config['name']}.</p>
            
            <!-- Required fields -->
            {% for field in ["''' + '", "'.join(config['required_fields']) + '''"] %}
            <div class="field">
                <label>{{ field.replace('_', ' ').title() }}:</label>
                <span>{{{{ field }}}}</span>
            </div>
            {% endfor %}
        </main>
        
        <footer>
            <p>Generated on {{{{ date | default("today") }}}}</p>
        </footer>
    </div>
</body>
</html>
'''
    
    def _generate_css_styles(self, template_id: str, config: Dict[str, Any]) -> str:
        """Generate CSS styles for template"""
        return f'''
/* CSS for {config['name']} */

body {{
    font-family: Arial, sans-serif;
    font-size: 12pt;
    line-height: 1.4;
    margin: 0;
    padding: 20px;
    color: #333;
}}

.{template_id.replace('_', '-')} {{
    max-width: 800px;
    margin: 0 auto;
    background: white;
    padding: 40px;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
}}

header {{
    border-bottom: 2px solid #333;
    padding-bottom: 20px;
    margin-bottom: 30px;
}}

h1 {{
    margin: 0;
    font-size: 24pt;
    color: #333;
}}

h2 {{
    font-size: 16pt;
    margin: 20px 0 10px 0;
    color: #333;
}}

h3 {{
    font-size: 14pt;
    margin: 15px 0 8px 0;
    color: #333;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
}}

th, td {{
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
}}

th {{
    background-color: #f5f5f5;
    font-weight: bold;
}}

.field {{
    margin: 10px 0;
}}

.field label {{
    font-weight: bold;
    margin-right: 10px;
}}

.total-line {{
    display: flex;
    justify-content: space-between;
    margin: 5px 0;
}}

.total {{
    font-weight: bold;
    font-size: 14pt;
    border-top: 1px solid #333;
    padding-top: 5px;
}}

footer {{
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #ddd;
    text-align: center;
    color: #666;
}}

@media print {{
    body {{
        margin: 0;
        padding: 0;
    }}
    
    .{template_id.replace('_', '-')} {{
        box-shadow: none;
        padding: 0;
    }}
}}
'''
    
    def _generate_sample_data(self, template_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate sample data for template"""
        
        if template_id == 'invoice':
            return {
                'invoice_number': 'INV-2024-001',
                'date': '2024-01-15',
                'customer': {
                    'name': 'Acme Corporation',
                    'address': '123 Business St',
                    'city': 'Business City',
                    'state': 'BC',
                    'zip': '12345'
                },
                'items': [
                    {
                        'description': 'Consulting Services',
                        'quantity': 10,
                        'unit_price': 150.00,
                        'total': 1500.00
                    },
                    {
                        'description': 'Software License',
                        'quantity': 1,
                        'unit_price': 500.00,
                        'total': 500.00
                    }
                ],
                'subtotal': 2000.00,
                'tax_rate': 8.5,
                'tax_amount': 170.00,
                'total': 2170.00,
                'payment_terms': 'Net 30 days'
            }
        
        elif template_id == 'receipt':
            return {
                'receipt_number': 'REC-2024-001',
                'date': '2024-01-15 14:30:00',
                'store_name': 'Corner Café',
                'store_address': '456 Main St, City, ST 12345',
                'items': [
                    {'name': 'Coffee (Large)', 'price': 4.50},
                    {'name': 'Blueberry Muffin', 'price': 3.25},
                    {'name': 'Orange Juice', 'price': 2.75}
                ],
                'subtotal': 10.50,
                'tax': 0.84,
                'tip': 2.00,
                'total': 13.34,
                'payment_method': 'Credit Card'
            }
        
        else:
            # Generic sample data
            sample_data = {
                'title': f'Sample {config["name"]}',
                'date': datetime.now().strftime('%Y-%m-%d')
            }
            
            # Add required fields with sample values
            for field in config['required_fields']:
                if field not in sample_data:
                    sample_data[field] = f'Sample {field.replace("_", " ").title()}'
            
            return sample_data


async def main():
    """
    Main template setup script function.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Setup and manage document templates'
    )
    parser.add_argument(
        'action',
        choices=['setup', 'list', 'validate', 'test', 'remove'],
        help='Action to perform'
    )
    parser.add_argument(
        '--template', 
        help='Specific template ID'
    )
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Force overwrite existing templates'
    )
    parser.add_argument(
        '--template-dir', 
        type=Path,
        help='Custom template directory'
    )
    
    args = parser.parse_args()
    
    setup = TemplateSetup(template_dir=args.template_dir)
    
    if args.action == 'setup':
        if args.template:
            result = await setup.setup_template(args.template, args.force)
            if result['success']:
                print(f" {result['message']}")
            else:
                print(f"L {result['message']}")
                return 1
        else:
            await setup.setup_all_templates(args.force)
    
    elif args.action == 'list':
        templates = await setup.list_templates()
        print(f"=Ë Available Templates ({len(templates)}):")
        print("=" * 60)
        
        for template in templates:
            status = " Installed" if template['installed'] else " Not installed"
            print(f"=9 {template['id']}")
            print(f"   Name: {template['name']}")
            print(f"   Category: {template['category']}")
            print(f"   Description: {template['description']}")
            print(f"   Formats: {', '.join(template['output_formats'])}")
            print(f"   Status: {status}")
            if template['installed']:
                print(f"   Path: {template['path']}")
            print()
    
    elif args.action == 'validate':
        if not args.template:
            print("L Template ID required for validation")
            return 1
        
        result = await setup.validate_template(args.template)
        
        print(f"= Validation Results for {args.template}:")
        print(f"Valid: {' Yes' if result['valid'] else 'L No'}")
        
        if result['errors']:
            print("\nL Errors:")
            for error in result['errors']:
                print(f"  - {error}")
        
        if result['warnings']:
            print("\n  Warnings:")
            for warning in result['warnings']:
                print(f"  - {warning}")
        
        if result['valid']:
            print(f"\n=Á Template path: {result['template_path']}")
    
    elif args.action == 'test':
        if not args.template:
            print("L Template ID required for testing")
            return 1
        
        result = await setup.test_template(args.template)
        
        if result['success']:
            print(f" {result['message']}")
            print(f"=Ä Output saved to: {result['output_file']}")
            print(f"=È Output size: {result['output_size']} characters")
        else:
            print(f"L {result['message']}")
            if 'errors' in result:
                for error in result['errors']:
                    print(f"  - {error}")
            return 1
    
    elif args.action == 'remove':
        if not args.template:
            print("L Template ID required for removal")
            return 1
        
        result = await setup.remove_template(args.template)
        
        if result['success']:
            print(f" {result['message']}")
        else:
            print(f"L {result['message']}")
            return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(asyncio.run(main()))