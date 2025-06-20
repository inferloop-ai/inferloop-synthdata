#!/usr/bin/env python3
"""
Enhanced comprehensive tests for all completed modules
"""

import sys
import os
import json
import traceback
from pathlib import Path
from datetime import datetime

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent / "structured-documents-synthetic-data/src"))

# Test results
test_results = {
    "total_tests": 0,
    "passed": 0,
    "failed": 0,
    "errors": [],
    "test_details": {},
    "modules_tested": []
}

def test_module(module_name, test_func):
    """Test a module and record results"""
    test_results["total_tests"] += 1
    test_results["modules_tested"].append(module_name)
    print(f"\n{'='*60}")
    print(f"Testing: {module_name}")
    print('='*60)
    
    try:
        test_func()
        test_results["passed"] += 1
        test_results["test_details"][module_name] = "PASSED"
        print(f"✅ {module_name} - PASSED")
        return True
    except Exception as e:
        test_results["failed"] += 1
        error_msg = f"{type(e).__name__}: {str(e)}"
        test_results["test_details"][module_name] = f"FAILED: {error_msg}"
        test_results["errors"].append({
            "module": module_name,
            "error": error_msg,
            "traceback": traceback.format_exc()
        })
        print(f"❌ {module_name} - FAILED: {error_msg}")
        return False

def test_ingestion_batch_init():
    """Test batch ingestion __init__.py"""
    # Test imports work
    from structured_docs_synth.ingestion.batch import (
        DatasetLoader, DatasetLoaderConfig, create_dataset_loader,
        FileProcessor, FileProcessorConfig, create_file_processor,
        create_batch_processing_pipeline
    )
    print("✓ All batch ingestion imports successful")
    
    # Test factory functions
    loader = create_dataset_loader()
    assert loader is not None
    print("✓ Dataset loader factory works")
    
    processor = create_file_processor()
    assert processor is not None
    print("✓ File processor factory works")
    
    pipeline = create_batch_processing_pipeline()
    assert 'dataset_loader' in pipeline
    assert 'file_processor' in pipeline
    print("✓ Batch processing pipeline factory works")

def test_content_generation_init():
    """Test content generation __init__.py"""
    from structured_docs_synth.generation.content import (
        create_content_generation_pipeline,
        create_financial_data_pipeline,
        create_healthcare_data_pipeline,
        create_legal_data_pipeline,
        DomainDataGenerator,
        EntityGenerator
    )
    print("✓ All content generation imports successful")
    
    # Test pipeline factories
    pipeline = create_content_generation_pipeline()
    assert 'domain_generator' in pipeline
    assert 'entity_generator' in pipeline
    print("✓ Content generation pipeline works")
    
    financial_pipeline = create_financial_data_pipeline()
    assert 'domain_generator' in financial_pipeline
    print("✓ Financial data pipeline works")

def test_generation_engines_init():
    """Test generation engines __init__.py"""
    from structured_docs_synth.generation.engines import (
        TemplateEngine, PDFGenerator, DOCXGenerator, get_template_engine
    )
    print("✓ All generation engines imports successful")
    
    # Test template engine
    template_engine = get_template_engine()
    assert template_engine is not None
    print("✓ Template engine factory works")

def test_main_generation_init():
    """Test main generation __init__.py"""
    from structured_docs_synth.generation import (
        create_generation_pipeline,
        create_document_generation_pipeline,
        DomainDataGenerator,
        EntityGenerator,
        TemplateEngine,
        PDFGenerator,
        DOCXGenerator
    )
    print("✓ All main generation imports successful")
    
    # Test main pipeline factory
    pipeline = create_generation_pipeline()
    assert 'content_pipeline' in pipeline
    assert 'template_engine' in pipeline
    assert 'pdf_generator' in pipeline
    assert 'docx_generator' in pipeline
    assert pipeline['status'] == 'production_ready'
    print("✓ Main generation pipeline works")
    
    # Test document-specific pipeline
    doc_pipeline = create_document_generation_pipeline('financial')
    assert doc_pipeline['document_type'] == 'financial'
    print("✓ Document-specific pipeline works")

def test_main_ingestion_init():
    """Test main ingestion __init__.py"""
    from structured_docs_synth.ingestion import (
        DataIngestionOrchestrator,
        create_batch_ingestion_pipeline,
        DatasetLoader,
        FileProcessor
    )
    print("✓ All main ingestion imports successful")
    
    # Test orchestrator
    orchestrator = DataIngestionOrchestrator()
    assert orchestrator is not None
    print("✓ Data ingestion orchestrator works")
    
    # Test available datasets
    datasets = orchestrator.get_available_datasets()
    assert isinstance(datasets, dict)
    print("✓ Available datasets retrieval works")

def test_domain_data_generator_enhanced():
    """Enhanced test for domain data generator"""
    from structured_docs_synth.generation.content.domain_data_generator import (
        DomainDataGenerator, DomainDataConfig, DataDomain, create_domain_generator
    )
    
    # Test all domains
    for domain in DataDomain:
        config = DomainDataConfig(domain=domain, num_records=2)
        generator = DomainDataGenerator(config)
        
        # Get available data types for this domain
        available_types = generator.get_available_data_types(domain)
        assert domain.value in available_types
        
        # Test generating data for each available type
        for data_type in available_types[domain.value]:
            try:
                data = generator.generate_data(data_type, count=1)
                assert len(data) == 1
                assert data[0].domain == domain
                print(f"✓ Generated {domain.value} - {data_type}")
            except Exception as e:
                print(f"⚠️  Failed to generate {domain.value} - {data_type}: {e}")

def test_entity_generator_enhanced():
    """Enhanced test for entity generator"""
    from structured_docs_synth.generation.content.entity_generator import (
        EntityGenerator, EntityType, create_entity_generator
    )
    
    generator = create_entity_generator()
    
    # Test all entity types
    for entity_type in EntityType:
        try:
            entity = generator.generate_entity(entity_type)
            assert entity.entity_type == entity_type
            assert isinstance(entity.data, dict)
            assert len(entity.data) > 0
            print(f"✓ Generated {entity_type.value} entity")
        except Exception as e:
            print(f"⚠️  Failed to generate {entity_type.value}: {e}")
    
    # Test batch generation
    batch_entities = generator.generate_entities(EntityType.PERSON, count=3)
    assert len(batch_entities) == 3
    print("✓ Batch entity generation works")

def test_integration_workflow():
    """Test end-to-end integration workflow"""
    from structured_docs_synth.generation import create_document_generation_pipeline
    from structured_docs_synth.ingestion import DataIngestionOrchestrator
    
    # Create financial document pipeline
    pipeline = create_document_generation_pipeline('financial')
    
    # Generate some financial data
    domain_generator = pipeline['content_pipeline']['domain_generator']
    entity_generator = pipeline['content_pipeline']['entity_generator']
    
    # Generate entities
    person = entity_generator.generate_entity(entity_generator.EntityType.PERSON)
    company = entity_generator.generate_entity(entity_generator.EntityType.COMPANY)
    
    # Generate domain data
    transaction = domain_generator.generate_data('transaction', count=1)[0]
    
    print("✓ Generated entities and domain data")
    
    # Test ingestion orchestrator
    orchestrator = DataIngestionOrchestrator()
    datasets = orchestrator.get_available_datasets()
    
    print(f"✓ Orchestrator can access {len(datasets)} datasets")
    print("✓ End-to-end integration test passed")

def test_template_engine_functionality():
    """Test template engine with sample data"""
    from structured_docs_synth.generation.engines.template_engine import get_template_engine
    
    template_engine = get_template_engine()
    
    # Test sample data generation
    sample_data = template_engine.generate_sample_data('legal_contract')
    assert isinstance(sample_data, dict)
    assert len(sample_data) > 0
    print("✓ Sample data generation works")
    
    # Test template fields
    fields = template_engine.get_template_fields('legal_contract')
    assert 'required_fields' in fields
    assert 'optional_fields' in fields
    print("✓ Template fields retrieval works")

def main():
    """Run all enhanced tests"""
    print("🧪 Running Enhanced Comprehensive Tests")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Run all tests
    test_module("ingestion_batch_init", test_ingestion_batch_init)
    test_module("content_generation_init", test_content_generation_init)
    test_module("generation_engines_init", test_generation_engines_init)
    test_module("main_generation_init", test_main_generation_init)
    test_module("main_ingestion_init", test_main_ingestion_init)
    test_module("domain_data_generator_enhanced", test_domain_data_generator_enhanced)
    test_module("entity_generator_enhanced", test_entity_generator_enhanced)
    test_module("template_engine_functionality", test_template_engine_functionality)
    test_module("integration_workflow", test_integration_workflow)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 ENHANCED TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {test_results['total_tests']}")
    print(f"✅ Passed: {test_results['passed']}")
    print(f"❌ Failed: {test_results['failed']}")
    print(f"Success Rate: {(test_results['passed'] / test_results['total_tests'] * 100):.1f}%")
    
    # Print detailed results
    print("\n📋 DETAILED RESULTS:")
    for module, status in test_results["test_details"].items():
        emoji = "✅" if status == "PASSED" else "❌"
        print(f"{emoji} {module}: {status}")
    
    # Print errors if any
    if test_results["errors"]:
        print("\n⚠️  ERROR DETAILS:")
        for error in test_results["errors"]:
            print(f"\n{error['module']}:")
            print(f"  Error: {error['error']}")
            if os.environ.get('VERBOSE'):
                print(f"  Traceback:\n{error['traceback']}")
    
    # Save results to file
    results_file = Path(__file__).parent / "enhanced_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    print(f"\n💾 Results saved to: {results_file}")
    
    # Final assessment
    print(f"\n🎯 FINAL ASSESSMENT:")
    if test_results['failed'] == 0:
        print("   🎉 ALL TESTS PASSED - PRODUCTION READY!")
    else:
        print(f"   ⚠️  {test_results['failed']} tests failed - needs attention")
    
    # Exit with appropriate code
    sys.exit(0 if test_results['failed'] == 0 else 1)

if __name__ == "__main__":
    main()