#!/usr/bin/env python3
"""
Test script for PII detection
"""

import sys
from pathlib import Path

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "docs/structured-documents-synthetic-data/src"))

try:
    from structured_docs_synth.privacy import PIIDetector, PIIType
    from structured_docs_synth.generation.engines import get_template_engine
    
    def test_pii_detector():
        """Test PII detection functionality"""
        print("🔍 Testing PII Detector")
        print("=" * 50)
        
        # Initialize PII detector
        pii_detector = PIIDetector()
        
        # Test data with various PII types
        test_cases = [
            {
                "name": "Legal Contract Data",
                "data": {
                    "applicant_name": "John Smith",
                    "ssn": "123-45-6789",
                    "email": "john.smith@example.com",
                    "phone": "(555) 123-4567",
                    "address": "123 Main Street, Anytown, CA 90210"
                }
            },
            {
                "name": "Medical Form Data", 
                "data": {
                    "patient_name": "Jane Doe",
                    "date_of_birth": "01/15/1985",
                    "medical_record_number": "MRN: 12345678",
                    "insurance_email": "jane.doe@healthinsurance.com"
                }
            },
            {
                "name": "Text with multiple PII",
                "text": "Contact John Smith at john.smith@email.com or call (555) 123-4567. His SSN is 123-45-6789."
            }
        ]
        
        for test_case in test_cases:
            print(f"\n📄 Testing: {test_case['name']}")
            print("-" * 30)
            
            if "data" in test_case:
                # Test document-level PII detection
                results = pii_detector.detect_pii_in_document(test_case["data"])
                
                print(f"Fields with PII: {len(results)}")
                for field_name, result in results.items():
                    print(f"  🔸 {field_name}: {result.total_matches} matches ({result.risk_level} risk)")
                    for match in result.matches:
                        print(f"    - {match.pii_type.value}: {match.value} (confidence: {match.confidence:.2f})")
                
                # Generate report
                if results:
                    report = pii_detector.generate_pii_report(results)
                    print(f"\n📊 Overall Risk Level: {report['summary']['overall_risk_level']}")
                    print(f"📊 Total PII Types: {report['summary']['unique_pii_types']}")
                    
                    if report['recommendations']:
                        print("💡 Recommendations:")
                        for rec in report['recommendations']:
                            print(f"  - {rec}")
            
            elif "text" in test_case:
                # Test text-level PII detection
                result = pii_detector.detect_pii(test_case["text"])
                
                print(f"PII found: {result.has_pii}")
                print(f"Matches: {result.total_matches}")
                print(f"Risk level: {result.risk_level}")
                
                for match in result.matches:
                    print(f"  - {match.pii_type.value}: {match.value} (confidence: {match.confidence:.2f})")
                
                # Test masking
                masked_text = pii_detector.mask_pii(test_case["text"], result)
                print(f"\nOriginal: {test_case['text']}")
                print(f"Masked:   {masked_text}")
        
        # Test with sample generated data
        print(f"\n🤖 Testing with Generated Sample Data")
        print("-" * 40)
        
        template_engine = get_template_engine()
        sample_data = template_engine.generate_sample_data("medical_form")
        
        print("Generated sample data:")
        for key, value in sample_data.items():
            print(f"  {key}: {value}")
        
        # Detect PII in generated data
        pii_results = pii_detector.detect_pii_in_document(sample_data)
        
        if pii_results:
            print(f"\nPII detected in generated data:")
            for field_name, result in pii_results.items():
                print(f"  🔸 {field_name}: {result.total_matches} matches")
        else:
            print("✅ No PII detected in generated sample data")
        
        print(f"\n🎉 PII detection testing complete!")
    
    if __name__ == "__main__":
        test_pii_detector()
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the correct directory and all dependencies are installed.")
except Exception as e:
    print(f"❌ Error: {e}")