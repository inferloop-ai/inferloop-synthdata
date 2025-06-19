#!/usr/bin/env python3
"""
Final QA tests for all completed modules - syntax and completeness validation
"""

import ast
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Test results
test_results = {
    "total_files": 0,
    "syntax_passed": 0,
    "syntax_failed": 0,
    "completeness_passed": 0,
    "completeness_failed": 0,
    "errors": [],
    "file_details": {}
}

def test_file_syntax(file_path: Path):
    """Test Python syntax of a file"""
    test_results["total_files"] += 1
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse AST to check syntax
        tree = ast.parse(content, filename=str(file_path))
        
        # Count elements
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        
        # Check for placeholders
        has_placeholders = any(keyword in content.lower() for keyword in [
            'placeholder', 'todo', 'fixme', 'not implemented', 'pass  #', 'raise notimplementederror'
        ])
        
        test_results["syntax_passed"] += 1
        if not has_placeholders:
            test_results["completeness_passed"] += 1
        else:
            test_results["completeness_failed"] += 1
        
        test_results["file_details"][file_path.name] = {
            "syntax": "PASS",
            "completeness": "COMPLETE" if not has_placeholders else "HAS_PLACEHOLDERS",
            "classes": len(classes),
            "functions": len(functions),
            "imports": len(imports),
            "has_placeholders": has_placeholders
        }
        
        return True, not has_placeholders
        
    except SyntaxError as e:
        test_results["syntax_failed"] += 1
        test_results["completeness_failed"] += 1
        error_msg = f"SyntaxError: {str(e)} at line {e.lineno}"
        test_results["errors"].append({
            "file": str(file_path),
            "error": error_msg
        })
        test_results["file_details"][file_path.name] = {
            "syntax": f"FAIL: {error_msg}",
            "completeness": "FAIL"
        }
        return False, False
        
    except Exception as e:
        test_results["syntax_failed"] += 1
        test_results["completeness_failed"] += 1
        error_msg = f"{type(e).__name__}: {str(e)}"
        test_results["errors"].append({
            "file": str(file_path),
            "error": error_msg
        })
        test_results["file_details"][file_path.name] = {
            "syntax": f"FAIL: {error_msg}",
            "completeness": "FAIL"
        }
        return False, False

def main():
    """Run final QA tests"""
    print("🧪 Running Final QA Tests")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Test completed modules
    completed_files = [
        # Batch Ingestion
        "structured-documents-synthetic-data/src/structured_docs_synth/ingestion/batch/dataset_loader.py",
        "structured-documents-synthetic-data/src/structured_docs_synth/ingestion/batch/file_processor.py",
        "structured-documents-synthetic-data/src/structured_docs_synth/ingestion/batch/__init__.py",
        
        # Content Generation
        "structured-documents-synthetic-data/src/structured_docs_synth/generation/content/domain_data_generator.py",
        "structured-documents-synthetic-data/src/structured_docs_synth/generation/content/entity_generator.py",
        "structured-documents-synthetic-data/src/structured_docs_synth/generation/content/__init__.py",
        
        # Generation Engines
        "structured-documents-synthetic-data/src/structured_docs_synth/generation/engines/template_engine.py",
        "structured-documents-synthetic-data/src/structured_docs_synth/generation/engines/docx_generator.py",
        "structured-documents-synthetic-data/src/structured_docs_synth/generation/engines/pdf_generator.py",
        "structured-documents-synthetic-data/src/structured_docs_synth/generation/engines/__init__.py",
        
        # Main Module Inits
        "structured-documents-synthetic-data/src/structured_docs_synth/generation/__init__.py",
        "structured-documents-synthetic-data/src/structured_docs_synth/ingestion/__init__.py"
    ]
    
    print(f"Testing {len(completed_files)} completed modules...")
    print("-" * 80)
    
    for file_path_str in completed_files:
        file_path = Path(file_path_str)
        if file_path.exists():
            print(f"Testing: {file_path.name}")
            syntax_ok, complete = test_file_syntax(file_path)
            
            if syntax_ok and complete:
                print(f"✅ {file_path.name} - SYNTAX OK, COMPLETE")
            elif syntax_ok:
                print(f"⚠️  {file_path.name} - SYNTAX OK, HAS PLACEHOLDERS")
            else:
                print(f"❌ {file_path.name} - SYNTAX ERROR")
        else:
            print(f"❌ {file_path.name} - FILE NOT FOUND")
            test_results["syntax_failed"] += 1
            test_results["completeness_failed"] += 1
            test_results["total_files"] += 1
    
    # Calculate statistics
    syntax_rate = (test_results["syntax_passed"] / test_results["total_files"] * 100) if test_results["total_files"] > 0 else 0
    completeness_rate = (test_results["completeness_passed"] / test_results["total_files"] * 100) if test_results["total_files"] > 0 else 0
    
    # Print summary
    print("\n" + "="*80)
    print("📊 FINAL QA SUMMARY")
    print("="*80)
    print(f"Total Files Tested: {test_results['total_files']}")
    print(f"✅ Syntax Passed: {test_results['syntax_passed']} ({syntax_rate:.1f}%)")
    print(f"❌ Syntax Failed: {test_results['syntax_failed']}")
    print(f"🎯 Complete Modules: {test_results['completeness_passed']} ({completeness_rate:.1f}%)")
    print(f"⚠️  Modules with Placeholders: {test_results['completeness_failed']}")
    
    # Detailed breakdown
    print("\n📋 MODULE BREAKDOWN:")
    
    # Group by category
    categories = {
        "Batch Ingestion": [],
        "Content Generation": [],
        "Generation Engines": [],
        "Module Initializers": []
    }
    
    for file_path, details in test_results["file_details"].items():
        if "ingestion/batch" in file_path:
            categories["Batch Ingestion"].append((file_path, details))
        elif "generation/content" in file_path:
            categories["Content Generation"].append((file_path, details))
        elif "generation/engines" in file_path:
            categories["Generation Engines"].append((file_path, details))
        elif "__init__.py" in file_path:
            categories["Module Initializers"].append((file_path, details))
    
    for category, files in categories.items():
        if files:
            print(f"\n{category}:")
            for file_path, details in files:
                filename = Path(file_path).name
                syntax_status = "✅" if details["syntax"] == "PASS" else "❌"
                complete_status = "🎯" if details["completeness"] == "COMPLETE" else "⚠️"
                
                stats = ""
                if "classes" in details:
                    stats = f" ({details['classes']} classes, {details['functions']} functions)"
                
                print(f"  {syntax_status}{complete_status} {filename}{stats}")
    
    # Print errors if any
    if test_results["errors"]:
        print("\n⚠️  ERRORS:")
        for error in test_results["errors"]:
            filename = Path(error['file']).name
            print(f"  ❌ {filename}: {error['error']}")
    
    # Save detailed results
    results_file = Path(__file__).parent / "final_qa_results.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Module completion summary
    print(f"\n🎯 MODULE COMPLETION SUMMARY:")
    
    # Count by category
    batch_complete = sum(1 for f, d in test_results["file_details"].items() 
                        if "ingestion/batch" in f and d["completeness"] == "COMPLETE")
    content_complete = sum(1 for f, d in test_results["file_details"].items() 
                          if "generation/content" in f and d["completeness"] == "COMPLETE")
    engines_complete = sum(1 for f, d in test_results["file_details"].items() 
                          if "generation/engines" in f and d["completeness"] == "COMPLETE")
    
    print(f"   📦 Batch Ingestion: {batch_complete}/3 modules complete")
    print(f"   🎨 Content Generation: {content_complete}/3 modules complete") 
    print(f"   ⚙️  Generation Engines: {engines_complete}/4 modules complete")
    
    # Final assessment
    print(f"\n🏆 FINAL ASSESSMENT:")
    if test_results['syntax_failed'] == 0 and completeness_rate >= 90:
        print("   🎉 EXCELLENT - Production ready!")
    elif test_results['syntax_failed'] == 0 and completeness_rate >= 75:
        print("   ✅ GOOD - Ready with minor placeholders")
    elif test_results['syntax_failed'] == 0:
        print("   ⚠️  FAIR - Syntax OK but needs completion")
    else:
        print("   ❌ NEEDS WORK - Syntax errors present")
    
    # Exit code
    sys.exit(0 if test_results['syntax_failed'] == 0 else 1)

if __name__ == "__main__":
    main()