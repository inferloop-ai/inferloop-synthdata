#!/usr/bin/env python3
"""
Comprehensive QA tests for completed modules
"""

import sys
import os
import json
import traceback
from pathlib import Path
from datetime import datetime

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent / "docs/structured-documents-synthetic-data/src"))

# Test results
test_results = {
    "total_tests": 0,
    "passed": 0,
    "failed": 0,
    "errors": [],
    "test_details": {}
}

def test_module(module_name, test_func):
    """Test a module and record results"""
    test_results["total_tests"] += 1
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

def test_dataset_loader():
    """Test dataset_loader module"""
    from structured_docs_synth.ingestion.batch.dataset_loader import (
        DatasetLoader, DatasetLoaderConfig, DatasetFormat, DatasetSource,
        create_dataset_loader
    )
    
    # Test 1: Module imports
    print("✓ Module imports successful")
    
    # Test 2: Create loader with default config
    loader = create_dataset_loader()
    assert loader is not None
    print("✓ Created dataset loader with default config")
    
    # Test 3: Create loader with custom config
    config = DatasetLoaderConfig(
        cache_dir="/tmp/test_cache",
        max_dataset_size_gb=5.0,
        enable_caching=True
    )
    loader = DatasetLoader(config)
    assert loader.config.cache_dir == "/tmp/test_cache"
    print("✓ Created dataset loader with custom config")
    
    # Test 4: Get available datasets
    datasets = loader.get_available_datasets()
    assert isinstance(datasets, dict)
    assert len(datasets) > 0
    print(f"✓ Found {len(datasets)} available datasets")
    
    # Test 5: Get dataset registry
    registry = loader.get_dataset_registry()
    assert isinstance(registry, dict)
    print(f"✓ Dataset registry contains {len(registry)} entries")
    
    # Test 6: Test data types availability
    available_types = loader.get_available_data_types()
    assert isinstance(available_types, dict)
    print("✓ Data types retrieval successful")
    
    # Test 7: Test dataset discovery
    datasets = loader.discover_datasets(Path("/tmp"))
    assert isinstance(datasets, list)
    print("✓ Dataset discovery functional")

def test_file_processor():
    """Test file_processor module"""
    from structured_docs_synth.ingestion.batch.file_processor import (
        FileProcessor, FileProcessorConfig, FileType, ProcessingStatus,
        create_file_processor
    )
    
    # Test 1: Module imports
    print("✓ Module imports successful")
    
    # Test 2: Create processor with default config
    processor = create_file_processor()
    assert processor is not None
    print("✓ Created file processor with default config")
    
    # Test 3: Create processor with custom config
    config = FileProcessorConfig(
        max_file_size_mb=50.0,
        parallel_workers=2,
        extract_text=True,
        create_thumbnails=False
    )
    processor = FileProcessor(config)
    assert processor.config.max_file_size_mb == 50.0
    print("✓ Created file processor with custom config")
    
    # Test 4: Test file type detection
    test_files = {
        "test.pdf": FileType.PDF,
        "test.txt": FileType.TEXT,
        "test.json": FileType.JSON,
        "test.docx": FileType.DOCX,
        "test.jpg": FileType.IMAGE
    }
    
    for filename, expected_type in test_files.items():
        detected_type = processor._detect_file_type(Path(filename))
        assert detected_type == expected_type, f"Failed to detect {filename} as {expected_type}"
    print("✓ File type detection working correctly")
    
    # Test 5: Test hash calculation (with a dummy file)
    test_file = Path("/tmp/test_file.txt")
    test_file.write_text("Test content")
    hash_value = processor._calculate_hash(test_file)
    assert len(hash_value) == 32  # MD5 hash length
    test_file.unlink()
    print("✓ Hash calculation functional")
    
    # Test 6: Test discovery
    files = processor.discover_files(Path("/tmp"))
    assert isinstance(files, list)
    print(f"✓ File discovery found {len(files)} files")

def test_domain_data_generator():
    """Test domain_data_generator module"""
    from structured_docs_synth.generation.content.domain_data_generator import (
        DomainDataGenerator, DomainDataConfig, DataDomain, GenerationMode,
        create_domain_generator
    )
    
    # Test 1: Module imports
    print("✓ Module imports successful")
    
    # Test 2: Create generator with default config
    generator = create_domain_generator()
    assert generator is not None
    print("✓ Created domain generator with default config")
    
    # Test 3: Test each domain
    domains_to_test = [
        (DataDomain.FINANCIAL, "transaction"),
        (DataDomain.HEALTHCARE, "patient_record"),
        (DataDomain.LEGAL, "contract"),
        (DataDomain.GOVERNMENT, "tax_return"),
        (DataDomain.EDUCATION, "student_record"),
        (DataDomain.RETAIL, "order"),
        (DataDomain.TECHNOLOGY, "server_log"),
        (DataDomain.REAL_ESTATE, "property_listing"),
        (DataDomain.INSURANCE, "policy"),
        (DataDomain.MANUFACTURING, "work_order")
    ]
    
    for domain, data_type in domains_to_test:
        config = DomainDataConfig(domain=domain, num_records=5)
        generator = DomainDataGenerator(config)
        data = generator.generate_data(data_type, count=2)
        assert len(data) == 2
        assert all(d.domain == domain for d in data)
        print(f"✓ Generated {domain.value} - {data_type} data successfully")
    
    # Test 4: Test available data types
    generator = DomainDataGenerator()
    available = generator.get_available_data_types()
    assert isinstance(available, dict)
    assert len(available) > 0
    print("✓ Available data types retrieval successful")
    
    # Test 5: Test financial helpers
    generator = DomainDataGenerator(DomainDataConfig(domain=DataDomain.FINANCIAL))
    account_num = generator._generate_account_number()
    assert isinstance(account_num, str)
    
    holdings = generator._generate_investment_holdings()
    assert isinstance(holdings, list)
    assert len(holdings) > 0
    print("✓ Helper methods functional")

def test_entity_generator():
    """Test entity_generator module"""
    from structured_docs_synth.generation.content.entity_generator import (
        EntityGenerator, EntityGeneratorConfig, EntityType,
        create_entity_generator
    )
    
    # Test 1: Module imports
    print("✓ Module imports successful")
    
    # Test 2: Create generator with default config
    generator = create_entity_generator()
    assert generator is not None
    print("✓ Created entity generator with default config")
    
    # Test 3: Test each entity type
    entity_types = [
        EntityType.PERSON,
        EntityType.COMPANY,
        EntityType.ADDRESS,
        EntityType.FINANCIAL_ACCOUNT,
        EntityType.MEDICAL_IDENTIFIER,
        EntityType.LEGAL_ENTITY,
        EntityType.GOVERNMENT_ID,
        EntityType.CONTACT_INFO
    ]
    
    for entity_type in entity_types:
        entity = generator.generate_entity(entity_type)
        assert entity.entity_type == entity_type
        assert isinstance(entity.data, dict)
        assert len(entity.data) > 0
        print(f"✓ Generated {entity_type.value} entity successfully")
    
    # Test 4: Test batch generation
    entities = generator.generate_entities(EntityType.PERSON, count=5)
    assert len(entities) == 5
    assert all(e.entity_type == EntityType.PERSON for e in entities)
    print("✓ Batch entity generation successful")
    
    # Test 5: Test custom providers
    entity = generator.generate_entity(EntityType.FINANCIAL_ACCOUNT, account_type='credit_card')
    assert 'card_number' in entity.data
    assert 'cvv' in entity.data
    print("✓ Custom providers functional")
    
    # Test 6: Test address formatting
    entity = generator.generate_entity(EntityType.ADDRESS, address_type='residential')
    assert entity.data.get('formatted_address') is not None
    print("✓ Address formatting functional")

def main():
    """Run all tests"""
    print("🧪 Running Comprehensive QA Tests")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Run tests
    test_module("dataset_loader", test_dataset_loader)
    test_module("file_processor", test_file_processor)
    test_module("domain_data_generator", test_domain_data_generator)
    test_module("entity_generator", test_entity_generator)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
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
    results_file = Path(__file__).parent / "test_results.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    print(f"\n💾 Results saved to: {results_file}")
    
    # Exit with appropriate code
    sys.exit(0 if test_results['failed'] == 0 else 1)

if __name__ == "__main__":
    main()