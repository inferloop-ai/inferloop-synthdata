# tests/test_langchain_template.py
import unittest
import sys
import os
import tempfile
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdk import LangChainTemplate

class TestLangChainTemplate(unittest.TestCase):
    def setUp(self):
        # Create a temporary template file
        self.template_data = {
            "template": "Hello, ${name}! Welcome to ${service}.",
            "variables": ["name", "service"],
            "description": "Simple greeting template"
        }
        
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json')
        json.dump(self.template_data, self.temp_file)
        self.temp_file.close()
    
    def tearDown(self):
        # Clean up the temporary file
        os.unlink(self.temp_file.name)
    
    def test_load_template_from_file(self):
        """Test loading template from a file"""
        template = LangChainTemplate(template_path=self.temp_file.name)
        
        self.assertEqual(template.template.template, "Hello, ${name}! Welcome to ${service}.")
        self.assertEqual(template.variables, ["name", "service"])
    
    def test_load_template_from_dict(self):
        """Test loading template from a dictionary"""
        template = LangChainTemplate(template_dict=self.template_data)
        
        self.assertEqual(template.template.template, "Hello, ${name}! Welcome to ${service}.")
        self.assertEqual(template.variables, ["name", "service"])
    
    def test_format_template(self):
        """Test formatting a template with variables"""
        template = LangChainTemplate(template_dict=self.template_data)
        
        result = template.format(name="John", service="Inferloop")
        expected = "Hello, John! Welcome to Inferloop."
        
        self.assertEqual(result, expected)
    
    def test_format_with_missing_variables(self):
        """Test formatting with missing variables"""
        template = LangChainTemplate(template_dict=self.template_data)
        
        # Missing 'service' variable
        result = template.format(name="John")
        expected = "Hello, John! Welcome to ${service}."
        
        self.assertEqual(result, expected)
    
    def test_batch_format(self):
        """Test batch formatting"""
        template = LangChainTemplate(template_dict=self.template_data)
        
        variable_sets = [
            {"name": "John", "service": "Inferloop"},
            {"name": "Jane", "service": "NLP Service"}
        ]
        
        results = template.batch_format(variable_sets)
        expected = [
            "Hello, John! Welcome to Inferloop.",
            "Hello, Jane! Welcome to NLP Service."
        ]
        
        self.assertEqual(results, expected)

if __name__ == "__main__":
    unittest.main()
