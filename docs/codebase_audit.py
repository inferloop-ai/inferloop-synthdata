#!/usr/bin/env python3
"""
Comprehensive Codebase Audit Script
Analyzes the structured documents synthetic data generation codebase for completeness.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json

def check_empty_files(base_path: str) -> List[Dict]:
    """Find all empty Python files"""
    empty_files = []
    for root, dirs, files in os.walk(base_path):
        # Skip virtual environment directories
        if 'test_env' in root or 'test_privacy_env' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if not content:
                            empty_files.append({
                                'path': file_path,
                                'type': 'empty',
                                'priority': 'high' if '__init__.py' in file else 'medium'
                            })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return empty_files

def check_stub_implementations(base_path: str) -> List[Dict]:
    """Find files with stub implementations, TODOs, or NotImplementedError"""
    stub_files = []
    patterns = [
        r'TODO',
        r'FIXME',
        r'NotImplementedError',
        r'raise NotImplementedError',
        r'pass\s*$',
        r'\.\.\.(?:\s*#.*)?$',  # Ellipsis
        r'def\s+\w+\([^)]*\):\s*$',  # Empty function definitions
        r'class\s+\w+[^:]*:\s*$',  # Empty class definitions
    ]
    
    compiled_patterns = [re.compile(pattern, re.MULTILINE) for pattern in patterns]
    
    for root, dirs, files in os.walk(base_path):
        # Skip virtual environment directories
        if 'test_env' in root or 'test_privacy_env' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    issues = []
                    for i, pattern in enumerate(compiled_patterns):
                        matches = pattern.findall(content)
                        if matches:
                            issues.extend([(patterns[i], match) for match in matches])
                    
                    if issues:
                        # Determine priority based on file type and content
                        priority = 'high'
                        if 'test' in file_path or '__init__.py' in file:
                            priority = 'medium'
                        if any('NotImplementedError' in issue[0] for issue in issues):
                            priority = 'high'
                        
                        stub_files.append({
                            'path': file_path,
                            'type': 'stub_implementation',
                            'issues': issues,
                            'priority': priority
                        })
                        
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return stub_files

def check_missing_init_files(base_path: str) -> List[Dict]:
    """Find directories that should have __init__.py files but don't"""
    missing_inits = []
    
    for root, dirs, files in os.walk(base_path):
        # Skip virtual environment directories
        if 'test_env' in root or 'test_privacy_env' in root:
            continue
            
        # Check if this is a Python package directory
        has_python_files = any(f.endswith('.py') for f in files)
        has_init = '__init__.py' in files
        
        # Skip certain directories
        skip_dirs = {'.git', '.github', '__pycache__', 'configs', 'data', 
                    'deployment', 'monitoring', 'notebooks', 'output', 'temp'}
        
        dir_name = os.path.basename(root)
        if dir_name in skip_dirs:
            continue
            
        if has_python_files and not has_init:
            # Check if this looks like a Python package
            parent_has_init = os.path.exists(os.path.join(os.path.dirname(root), '__init__.py'))
            if parent_has_init or 'src' in root:
                missing_inits.append({
                    'path': root,
                    'type': 'missing_init',
                    'priority': 'medium'
                })
    
    return missing_inits

def check_incomplete_modules(base_path: str) -> List[Dict]:
    """Check for modules that appear to have minimal implementations"""
    incomplete_modules = []
    
    for root, dirs, files in os.walk(base_path):
        # Skip virtual environment directories
        if 'test_env' in root or 'test_privacy_env' in root:
            continue
            
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    lines = content.split('\n')
                    non_empty_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
                    
                    # Check for minimal implementation indicators
                    if len(non_empty_lines) < 10:  # Very short files
                        incomplete_modules.append({
                            'path': file_path,
                            'type': 'minimal_implementation',
                            'line_count': len(non_empty_lines),
                            'priority': 'medium'
                        })
                    elif 'class' not in content and 'def' not in content:
                        # No classes or functions defined
                        incomplete_modules.append({
                            'path': file_path,
                            'type': 'no_implementations',
                            'priority': 'high'
                        })
                        
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return incomplete_modules

def check_documentation_files(base_path: str) -> List[Dict]:
    """Check for empty or incomplete documentation files"""
    incomplete_docs = []
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(('.md', '.rst', '.txt')) and 'README' in file.upper():
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    if not content:
                        incomplete_docs.append({
                            'path': file_path,
                            'type': 'empty_documentation',
                            'priority': 'low'
                        })
                    elif len(content) < 100:  # Very short documentation
                        incomplete_docs.append({
                            'path': file_path,
                            'type': 'minimal_documentation',
                            'priority': 'low'
                        })
                        
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return incomplete_docs

def check_configuration_files(base_path: str) -> List[Dict]:
    """Check for missing or incomplete configuration files"""
    missing_configs = []
    
    # Expected configuration files
    expected_configs = [
        'pyproject.toml',
        'requirements.txt',
        'setup.py',
        '.gitignore'
    ]
    
    for config_file in expected_configs:
        config_path = os.path.join(base_path, 'docs', 'structured-documents-synthetic-data', config_file)
        if not os.path.exists(config_path):
            missing_configs.append({
                'path': config_path,
                'type': 'missing_config',
                'priority': 'medium'
            })
        else:
            # Check if file is empty
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                if not content:
                    missing_configs.append({
                        'path': config_path,
                        'type': 'empty_config',
                        'priority': 'high'
                    })
            except Exception as e:
                print(f"Error reading {config_path}: {e}")
    
    return missing_configs

def generate_report(base_path: str) -> Dict:
    """Generate comprehensive audit report"""
    print("Starting comprehensive codebase audit...")
    
    report = {
        'audit_timestamp': str(Path(base_path).name),
        'base_path': base_path,
        'findings': {
            'empty_files': check_empty_files(base_path),
            'stub_implementations': check_stub_implementations(base_path),
            'missing_init_files': check_missing_init_files(base_path),
            'incomplete_modules': check_incomplete_modules(base_path),
            'incomplete_documentation': check_documentation_files(base_path),
            'missing_configurations': check_configuration_files(base_path)
        }
    }
    
    # Calculate summary statistics
    total_issues = sum(len(findings) for findings in report['findings'].values())
    high_priority = sum(
        len([item for item in findings if item.get('priority') == 'high'])
        for findings in report['findings'].values()
    )
    
    report['summary'] = {
        'total_issues': total_issues,
        'high_priority_issues': high_priority,
        'categories': {category: len(findings) for category, findings in report['findings'].items()}
    }
    
    return report

def print_detailed_report(report: Dict):
    """Print detailed audit report"""
    print("\n" + "="*80)
    print("COMPREHENSIVE CODEBASE AUDIT REPORT")
    print("="*80)
    
    print(f"\nBase Path: {report['base_path']}")
    print(f"Total Issues Found: {report['summary']['total_issues']}")
    print(f"High Priority Issues: {report['summary']['high_priority_issues']}")
    
    print("\n" + "-"*60)
    print("SUMMARY BY CATEGORY")
    print("-"*60)
    
    for category, count in report['summary']['categories'].items():
        print(f"{category.replace('_', ' ').title()}: {count}")
    
    # Detailed findings
    for category, findings in report['findings'].items():
        if findings:
            print(f"\n" + "="*60)
            print(f"{category.replace('_', ' ').title().upper()}")
            print("="*60)
            
            # Group by priority
            by_priority = {}
            for finding in findings:
                priority = finding.get('priority', 'unknown')
                if priority not in by_priority:
                    by_priority[priority] = []
                by_priority[priority].append(finding)
            
            for priority in ['high', 'medium', 'low']:
                if priority in by_priority:
                    print(f"\n{priority.upper()} PRIORITY ({len(by_priority[priority])} items):")
                    print("-" * 40)
                    
                    for finding in by_priority[priority][:10]:  # Limit to first 10
                        path = finding['path'].replace(report['base_path'], '.')
                        print(f"  • {path}")
                        if 'issues' in finding:
                            for issue_type, issue_content in finding['issues'][:3]:  # First 3 issues
                                print(f"    - {issue_type}: {issue_content[:50]}...")
                        if 'line_count' in finding:
                            print(f"    - Lines: {finding['line_count']}")
                    
                    if len(by_priority[priority]) > 10:
                        print(f"    ... and {len(by_priority[priority]) - 10} more")

if __name__ == "__main__":
    # Set base path to the docs directory
    base_path = "/Users/dattamiruke/INFERLOOP/GitHub/inferloop-synthdata/docs"
    
    # Generate comprehensive report
    report = generate_report(base_path)
    
    # Print detailed report
    print_detailed_report(report)
    
    # Save report to file
    output_file = os.path.join(base_path, "codebase_audit_report.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n\nDetailed report saved to: {output_file}")