#!/usr/bin/env python3
"""
Test script for PDF and DOCX generators
"""

import sys
import os
from pathlib import Path

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "docs/structured-documents-synthetic-data/src"))

try:
    from structured_docs_synth.generation.engines import PDFGenerator, DOCXGenerator, get_template_engine
    from structured_docs_synth.core import get_config
    
    def test_generators():
        """Test PDF and DOCX generators"""
        print("🧪 Testing PDF and DOCX Generators")
        print("=" * 50)
        
        # Initialize generators
        pdf_gen = PDFGenerator()
        docx_gen = DOCXGenerator()
        template_engine = get_template_engine()
        
        # Test document types
        document_types = ['legal_contract', 'medical_form', 'loan_application', 'tax_form']
        
        for doc_type in document_types:
            print(f"\n📄 Testing {doc_type}...")
            
            try:
                # Generate sample data
                sample_data = template_engine.generate_sample_data(doc_type)
                print(f"   ✅ Generated sample data: {len(sample_data)} fields")
                
                # Test PDF generation
                pdf_path = pdf_gen.generate_pdf(
                    document_type=doc_type,
                    data=sample_data,
                    metadata={
                        'title': f'Test {doc_type.replace("_", " ").title()}',
                        'date': '2024-01-15',
                        'reference': f'TEST-{doc_type.upper()}-001'
                    }
                )
                print(f"   ✅ Generated PDF: {pdf_path.name}")
                
                # Test DOCX generation
                docx_path = docx_gen.generate_docx(
                    document_type=doc_type,
                    data=sample_data,
                    metadata={
                        'title': f'Test {doc_type.replace("_", " ").title()}',
                        'date': '2024-01-15',
                        'reference': f'TEST-{doc_type.upper()}-001'
                    }
                )
                print(f"   ✅ Generated DOCX: {docx_path.name}")
                
            except Exception as e:
                print(f"   ❌ Error testing {doc_type}: {str(e)}")
        
        print(f"\n🎉 Test complete! Check output directories:")
        print(f"   📁 PDF files: {pdf_gen.output_dir}")
        print(f"   📁 DOCX files: {docx_gen.output_dir}")
    
    if __name__ == "__main__":
        test_generators()
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the correct directory and all dependencies are installed.")
except Exception as e:
    print(f"❌ Error: {e}")