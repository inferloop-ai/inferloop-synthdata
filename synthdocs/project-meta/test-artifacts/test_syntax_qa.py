#!/usr/bin/env python3
"""
Syntax and basic structure QA tests for completed modules
"""

import ast
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Test results
test_results = {
    "total_tests": 0,
    "passed": 0,
    "failed": 0,
    "errors": [],
    "test_details": {}
}

def test_module_syntax(module_path: Path):
    """Test Python syntax of a module"""
    test_results["total_tests"] += 1
    module_name = module_path.name
    
    print(f"\n{'='*60}")
    print(f"Testing syntax: {module_name}")
    print('='*60)
    
    try:
        # Read and parse the file
        with open(module_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse AST to check syntax
        tree = ast.parse(content, filename=str(module_path))
        
        # Count various elements
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        
        print(f"✓ Syntax is valid")
        print(f"✓ Found {len(classes)} classes")
        print(f"✓ Found {len(functions)} functions") 
        print(f"✓ Found {len(imports)} import statements")
        
        # Check for docstrings
        module_docstring = ast.get_docstring(tree)
        if module_docstring:
            print(f"✓ Module has docstring")
        
        # Check class and function docstrings
        documented_classes = sum(1 for cls in classes if ast.get_docstring(cls))
        documented_functions = sum(1 for func in functions if ast.get_docstring(func))
        
        print(f"✓ {documented_classes}/{len(classes)} classes have docstrings")
        print(f"✓ {documented_functions}/{len(functions)} functions have docstrings")
        
        # Check for type hints
        typed_functions = sum(1 for func in functions if func.returns or any(arg.annotation for arg in func.args.args))
        print(f"✓ {typed_functions}/{len(functions)} functions have type hints")
        
        test_results["passed"] += 1
        test_results["test_details"][module_name] = {
            "status": "PASSED",
            "classes": len(classes),
            "functions": len(functions),
            "imports": len(imports),
            "documented_classes": documented_classes,
            "documented_functions": documented_functions,
            "typed_functions": typed_functions
        }
        print(f"✅ {module_name} - SYNTAX VALID")
        return True
        
    except SyntaxError as e:
        error_msg = f"SyntaxError: {str(e)} at line {e.lineno}"
        test_results["failed"] += 1
        test_results["test_details"][module_name] = {"status": f"FAILED: {error_msg}"}
        test_results["errors"].append({
            "module": module_name,
            "error": error_msg,
            "line": e.lineno
        })
        print(f"❌ {module_name} - SYNTAX ERROR: {error_msg}")
        return False
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        test_results["failed"] += 1
        test_results["test_details"][module_name] = {"status": f"FAILED: {error_msg}"}
        test_results["errors"].append({
            "module": module_name,
            "error": error_msg
        })
        print(f"❌ {module_name} - ERROR: {error_msg}")
        return False

def check_module_completeness():
    """Check if all required methods are implemented"""
    print(f"\n{'='*60}")
    print("Checking Module Completeness")
    print('='*60)
    
    base_path = Path("structured-documents-synthetic-data/src/structured_docs_synth")
    
    # Expected modules and their key components
    expected_modules = {
        "ingestion/batch/dataset_loader.py": {
            "classes": ["DatasetLoader", "DatasetLoaderConfig"],
            "functions": ["create_dataset_loader"],
            "methods": ["load_dataset", "get_available_datasets"]
        },
        "ingestion/batch/file_processor.py": {
            "classes": ["FileProcessor", "FileProcessorConfig"],
            "functions": ["create_file_processor"],
            "methods": ["process_file", "process_files"]
        },
        "generation/content/domain_data_generator.py": {
            "classes": ["DomainDataGenerator", "DomainDataConfig"],
            "functions": ["create_domain_generator"],
            "methods": ["generate_data"]
        },
        "generation/content/entity_generator.py": {
            "classes": ["EntityGenerator", "EntityGeneratorConfig"],
            "functions": ["create_entity_generator"],
            "methods": ["generate_entity", "generate_entities"]
        }
    }
    
    completeness_results = {}
    
    for module_path, expected in expected_modules.items():
        full_path = base_path / module_path
        module_name = Path(module_path).name
        
        if not full_path.exists():
            print(f"❌ {module_name} - FILE NOT FOUND")
            completeness_results[module_name] = "MISSING"
            continue
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Extract class and function names
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            
            # Check for expected components
            missing_classes = set(expected["classes"]) - set(classes)
            missing_functions = set(expected["functions"]) - set(functions)
            
            # Check for methods within classes (simplified check)
            has_expected_methods = all(method in content for method in expected["methods"])
            
            if missing_classes or missing_functions or not has_expected_methods:
                issues = []
                if missing_classes:
                    issues.append(f"Missing classes: {missing_classes}")
                if missing_functions:
                    issues.append(f"Missing functions: {missing_functions}")
                if not has_expected_methods:
                    issues.append(f"Missing some expected methods")
                
                print(f"⚠️  {module_name} - INCOMPLETE: {'; '.join(issues)}")
                completeness_results[module_name] = f"INCOMPLETE: {'; '.join(issues)}"
            else:
                print(f"✅ {module_name} - COMPLETE")
                completeness_results[module_name] = "COMPLETE"
                
        except Exception as e:
            print(f"❌ {module_name} - ERROR: {e}")
            completeness_results[module_name] = f"ERROR: {e}"
    
    return completeness_results

def main():
    """Run all syntax tests"""
    print("🧪 Running Syntax QA Tests")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Find all Python files to test
    base_path = Path("structured-documents-synthetic-data/src/structured_docs_synth")
    
    test_files = [
        base_path / "ingestion/batch/dataset_loader.py",
        base_path / "ingestion/batch/file_processor.py", 
        base_path / "generation/content/domain_data_generator.py",
        base_path / "generation/content/entity_generator.py"
    ]
    
    # Test syntax for each file
    for file_path in test_files:
        if file_path.exists():
            test_module_syntax(file_path)
        else:
            print(f"❌ {file_path.name} - FILE NOT FOUND")
            test_results["total_tests"] += 1
            test_results["failed"] += 1
            test_results["test_details"][file_path.name] = "FILE NOT FOUND"
    
    # Check module completeness
    completeness_results = check_module_completeness()
    
    # Print summary
    print("\n" + "="*60)
    print("📊 SYNTAX TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {test_results['total_tests']}")
    print(f"✅ Passed: {test_results['passed']}")
    print(f"❌ Failed: {test_results['failed']}")
    print(f"Success Rate: {(test_results['passed'] / test_results['total_tests'] * 100):.1f}%")
    
    # Print detailed results
    print("\n📋 DETAILED RESULTS:")
    for module, details in test_results["test_details"].items():
        if isinstance(details, dict) and details["status"] == "PASSED":
            print(f"✅ {module}: SYNTAX OK - {details['classes']} classes, {details['functions']} functions")
        else:
            status = details["status"] if isinstance(details, dict) else details
            emoji = "✅" if status == "PASSED" else "❌"
            print(f"{emoji} {module}: {status}")
    
    print("\n📋 COMPLETENESS RESULTS:")
    for module, status in completeness_results.items():
        emoji = "✅" if status == "COMPLETE" else "⚠️" if "INCOMPLETE" in status else "❌"
        print(f"{emoji} {module}: {status}")
    
    # Print errors if any
    if test_results["errors"]:
        print("\n⚠️  ERROR DETAILS:")
        for error in test_results["errors"]:
            print(f"\n{error['module']}:")
            print(f"  Error: {error['error']}")
            if 'line' in error:
                print(f"  Line: {error['line']}")
    
    # Save results to file
    results_file = Path(__file__).parent / "syntax_test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "syntax_tests": test_results,
            "completeness_tests": completeness_results,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2, default=str)
    print(f"\n💾 Results saved to: {results_file}")
    
    # Final assessment
    all_complete = all(status == "COMPLETE" for status in completeness_results.values())
    syntax_ok = test_results['failed'] == 0
    
    print(f"\n🎯 FINAL ASSESSMENT:")
    print(f"   Syntax Quality: {'✅ PASS' if syntax_ok else '❌ FAIL'}")
    print(f"   Module Completeness: {'✅ COMPLETE' if all_complete else '⚠️ NEEDS WORK'}")
    print(f"   Overall Status: {'🎉 READY FOR PRODUCTION' if (syntax_ok and all_complete) else '🔧 NEEDS ATTENTION'}")
    
    # Exit with appropriate code
    sys.exit(0 if (syntax_ok and all_complete) else 1)

if __name__ == "__main__":
    main()