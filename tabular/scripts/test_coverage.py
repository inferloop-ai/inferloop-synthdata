#!/usr/bin/env python
"""
Test coverage reporting script
"""

import subprocess
import sys
import os
from pathlib import Path
import json
from datetime import datetime
import argparse


class CoverageReporter:
    """Generate and analyze test coverage reports"""
    
    def __init__(self, min_coverage: float = 80.0):
        self.min_coverage = min_coverage
        self.project_root = Path(__file__).parent.parent
        
    def run_tests_with_coverage(self, test_path: str = None, markers: str = None):
        """Run tests with coverage collection"""
        
        print("Running tests with coverage...")
        print("=" * 80)
        
        # Build pytest command
        cmd = [
            sys.executable, '-m', 'pytest',
            '--cov=.',
            '--cov-config=.coveragerc',
            '--cov-report=term-missing',
            '--cov-report=html',
            '--cov-report=xml',
            '--cov-report=json',
            '-v'
        ]
        
        if test_path:
            cmd.append(test_path)
        
        if markers:
            cmd.extend(['-m', markers])
        
        # Run tests
        result = subprocess.run(cmd, cwd=self.project_root)
        
        return result.returncode == 0
    
    def analyze_coverage(self):
        """Analyze coverage results"""
        
        coverage_file = self.project_root / 'coverage.json'
        
        if not coverage_file.exists():
            print("Coverage file not found. Run tests first.")
            return False
        
        with open(coverage_file, 'r') as f:
            coverage_data = json.load(f)
        
        # Overall statistics
        total_coverage = coverage_data['totals']['percent_covered']
        
        print("\n" + "=" * 80)
        print("COVERAGE ANALYSIS")
        print("=" * 80)
        print(f"Overall Coverage: {total_coverage:.2f}%")
        print(f"Minimum Required: {self.min_coverage:.2f}%")
        print(f"Status: {'✓ PASS' if total_coverage >= self.min_coverage else '✗ FAIL'}")
        
        # File-by-file breakdown
        print("\nFile Coverage:")
        print("-" * 80)
        
        files = coverage_data['files']
        
        # Sort by coverage percentage
        sorted_files = sorted(
            files.items(),
            key=lambda x: x[1]['summary']['percent_covered']
        )
        
        for filepath, file_data in sorted_files:
            # Skip test files
            if 'test' in filepath or '__pycache__' in filepath:
                continue
            
            percent = file_data['summary']['percent_covered']
            missing = file_data['summary']['missing_lines']
            
            # Make path relative
            rel_path = Path(filepath).relative_to(self.project_root)
            
            status = "✓" if percent >= self.min_coverage else "✗"
            print(f"{status} {rel_path:<50} {percent:>6.2f}% ({missing} missing)")
        
        # Find uncovered code
        print("\nUncovered Code Summary:")
        print("-" * 80)
        
        uncovered_count = 0
        for filepath, file_data in files.items():
            if 'test' in filepath or '__pycache__' in filepath:
                continue
            
            missing_lines = file_data.get('missing_lines', [])
            if missing_lines and len(missing_lines) > 0:
                uncovered_count += 1
                rel_path = Path(filepath).relative_to(self.project_root)
                print(f"\n{rel_path}:")
                
                # Group consecutive lines
                if missing_lines:
                    ranges = self._group_lines(missing_lines)
                    print(f"  Missing lines: {', '.join(ranges)}")
        
        if uncovered_count == 0:
            print("All code is covered!")
        
        # Module coverage
        print("\n\nModule Coverage:")
        print("-" * 80)
        
        modules = {}
        for filepath, file_data in files.items():
            if 'test' in filepath or '__pycache__' in filepath:
                continue
            
            # Extract module name
            parts = Path(filepath).parts
            if 'inferloop_synthetic' in parts:
                idx = parts.index('inferloop_synthetic')
                if idx + 1 < len(parts):
                    module = parts[idx + 1]
                    if module not in modules:
                        modules[module] = {
                            'files': 0,
                            'covered': 0,
                            'total': 0
                        }
                    
                    modules[module]['files'] += 1
                    modules[module]['covered'] += file_data['summary']['covered_lines']
                    modules[module]['total'] += file_data['summary']['num_statements']
        
        for module, stats in sorted(modules.items()):
            if stats['total'] > 0:
                percent = (stats['covered'] / stats['total']) * 100
                print(f"{module:<20} {percent:>6.2f}% ({stats['files']} files)")
        
        return total_coverage >= self.min_coverage
    
    def _group_lines(self, lines):
        """Group consecutive line numbers into ranges"""
        if not lines:
            return []
        
        ranges = []
        start = lines[0]
        end = lines[0]
        
        for line in lines[1:]:
            if line == end + 1:
                end = line
            else:
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = line
                end = line
        
        # Add last range
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")
        
        return ranges
    
    def generate_badge(self):
        """Generate coverage badge"""
        
        print("\nGenerating coverage badge...")
        
        try:
            subprocess.run(
                ['coverage-badge', '-o', 'coverage.svg'],
                cwd=self.project_root,
                check=True
            )
            print("✓ Coverage badge generated: coverage.svg")
        except subprocess.CalledProcessError:
            print("✗ Failed to generate coverage badge")
        except FileNotFoundError:
            print("✗ coverage-badge not installed. Install with: pip install coverage-badge")
    
    def generate_report_summary(self, output_file: str = "coverage_summary.md"):
        """Generate markdown summary report"""
        
        coverage_file = self.project_root / 'coverage.json'
        
        if not coverage_file.exists():
            return
        
        with open(coverage_file, 'r') as f:
            coverage_data = json.load(f)
        
        total_coverage = coverage_data['totals']['percent_covered']
        
        report = []
        report.append("# Test Coverage Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n## Summary")
        report.append(f"- **Total Coverage**: {total_coverage:.2f}%")
        report.append(f"- **Lines Covered**: {coverage_data['totals']['covered_lines']}")
        report.append(f"- **Lines Missing**: {coverage_data['totals']['missing_lines']}")
        report.append(f"- **Total Statements**: {coverage_data['totals']['num_statements']}")
        
        # Add badge
        report.append("\n![Coverage](coverage.svg)")
        
        # Module breakdown
        report.append("\n## Module Coverage")
        report.append("\n| Module | Coverage | Files |")
        report.append("|--------|----------|-------|")
        
        modules = {}
        for filepath, file_data in coverage_data['files'].items():
            if 'test' in filepath or '__pycache__' in filepath:
                continue
            
            parts = Path(filepath).parts
            module = parts[0] if parts else 'root'
            
            if module not in modules:
                modules[module] = {
                    'files': 0,
                    'covered': 0,
                    'total': 0
                }
            
            modules[module]['files'] += 1
            modules[module]['covered'] += file_data['summary']['covered_lines']
            modules[module]['total'] += file_data['summary']['num_statements']
        
        for module, stats in sorted(modules.items()):
            if stats['total'] > 0:
                percent = (stats['covered'] / stats['total']) * 100
                report.append(f"| {module} | {percent:.2f}% | {stats['files']} |")
        
        # Write report
        output_path = self.project_root / output_file
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"\n✓ Coverage summary written to: {output_file}")
    
    def check_diff_coverage(self, base_branch: str = "main"):
        """Check coverage for changed files only"""
        
        print(f"\nChecking coverage for changes against {base_branch}...")
        
        # Get changed files
        result = subprocess.run(
            ['git', 'diff', '--name-only', base_branch],
            capture_output=True,
            text=True,
            cwd=self.project_root
        )
        
        if result.returncode != 0:
            print("Failed to get git diff")
            return
        
        changed_files = [
            f for f in result.stdout.strip().split('\n')
            if f.endswith('.py') and not f.startswith('tests/')
        ]
        
        if not changed_files:
            print("No Python files changed")
            return
        
        print(f"Changed files: {len(changed_files)}")
        
        # Load coverage data
        coverage_file = self.project_root / 'coverage.json'
        if not coverage_file.exists():
            print("Coverage data not found")
            return
        
        with open(coverage_file, 'r') as f:
            coverage_data = json.load(f)
        
        # Check coverage for changed files
        print("\nCoverage for changed files:")
        print("-" * 60)
        
        total_changed_covered = 0
        total_changed_statements = 0
        
        for filepath in changed_files:
            full_path = str(self.project_root / filepath)
            
            if full_path in coverage_data['files']:
                file_data = coverage_data['files'][full_path]
                percent = file_data['summary']['percent_covered']
                covered = file_data['summary']['covered_lines']
                total = file_data['summary']['num_statements']
                
                total_changed_covered += covered
                total_changed_statements += total
                
                status = "✓" if percent >= self.min_coverage else "✗"
                print(f"{status} {filepath:<40} {percent:>6.2f}%")
            else:
                print(f"✗ {filepath:<40} No coverage data")
        
        if total_changed_statements > 0:
            changed_coverage = (total_changed_covered / total_changed_statements) * 100
            print(f"\nOverall coverage for changed files: {changed_coverage:.2f}%")
            
            if changed_coverage < self.min_coverage:
                print(f"✗ Changed files coverage below minimum ({self.min_coverage}%)")
                return False
        
        return True


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Test coverage reporting')
    parser.add_argument(
        '--min-coverage',
        type=float,
        default=80.0,
        help='Minimum coverage percentage required'
    )
    parser.add_argument(
        '--test-path',
        help='Specific test path to run'
    )
    parser.add_argument(
        '--markers',
        help='Pytest markers to filter tests'
    )
    parser.add_argument(
        '--diff',
        action='store_true',
        help='Check coverage for changed files only'
    )
    parser.add_argument(
        '--badge',
        action='store_true',
        help='Generate coverage badge'
    )
    
    args = parser.parse_args()
    
    reporter = CoverageReporter(min_coverage=args.min_coverage)
    
    # Run tests
    success = reporter.run_tests_with_coverage(
        test_path=args.test_path,
        markers=args.markers
    )
    
    if not success:
        print("\n✗ Tests failed!")
        sys.exit(1)
    
    # Analyze coverage
    coverage_ok = reporter.analyze_coverage()
    
    # Generate reports
    reporter.generate_report_summary()
    
    if args.badge:
        reporter.generate_badge()
    
    if args.diff:
        diff_ok = reporter.check_diff_coverage()
        if not diff_ok:
            sys.exit(1)
    
    if not coverage_ok:
        print(f"\n✗ Coverage below minimum threshold ({args.min_coverage}%)")
        sys.exit(1)
    
    print("\n✓ All checks passed!")


if __name__ == "__main__":
    main()