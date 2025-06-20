#!/usr/bin/env python3
"""
Test script to verify that unicode encoding issues have been fixed by checking syntax compilation.
"""

import ast
import sys
import os
from pathlib import Path

def test_file_syntax(filepath):
    """Test if a Python file can be parsed without syntax/unicode errors."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Try to parse the AST
        ast.parse(source, filename=str(filepath))
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    except UnicodeError as e:
        return False, f"UnicodeError: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def main():
    print("🧪 Testing Unicode/Syntax Fixes in Privacy Modules")
    print("=" * 55)
    
    # Define the files that were fixed
    source_dir = Path("docs/structured-documents-synthetic-data/src")
    privacy_files = [
        "structured_docs_synth/privacy/differential_privacy/privacy_budget.py",
        "structured_docs_synth/privacy/differential_privacy/exponential_mechanism.py", 
        "structured_docs_synth/privacy/differential_privacy/laplace_mechanism.py",
        "structured_docs_synth/privacy/differential_privacy/composition_analyzer.py"
    ]
    
    all_passed = True
    
    for rel_path in privacy_files:
        filepath = source_dir / rel_path
        if filepath.exists():
            success, error = test_file_syntax(filepath)
            if success:
                print(f"✅ {rel_path}")
            else:
                print(f"❌ {rel_path}: {error}")
                all_passed = False
        else:
            print(f"⚠️  {rel_path}: File not found")
            all_passed = False
    
    print("\n" + "=" * 55)
    if all_passed:
        print("🎉 All files passed syntax validation!")
        print("✅ Unicode encoding issues have been successfully fixed")
        return True
    else:
        print("💥 Some files still have syntax/unicode issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)