#!/usr/bin/env python3
"""
Validate command for checking document quality and compliance.

Provides CLI interface for validating synthetic documents against
quality metrics, compliance rules, and structural requirements.
"""

import asyncio
import click
import json
from pathlib import Path
from typing import List, Optional

from ...core import get_logger
from ...quality import create_quality_engine
from ...privacy import create_privacy_engine
from ..utils import OutputFormatter, ProgressTracker


logger = get_logger(__name__)


@click.command(name='validate')
@click.argument(
    'paths',
    nargs=-1,
    required=True,
    type=click.Path(exists=True, path_type=Path)
)
@click.option(
    '--rules', '-r',
    multiple=True,
    type=click.Choice([
        'structure', 'completeness', 'quality', 'privacy', 
        'compliance', 'format', 'content', 'all'
    ]),
    default=['all'],
    help='Validation rules to apply'
)
@click.option(
    '--compliance-framework',
    type=click.Choice(['gdpr', 'hipaa', 'pci-dss', 'sox', 'all']),
    help='Compliance framework to validate against'
)
@click.option(
    '--threshold',
    type=float,
    default=0.8,
    help='Quality threshold for validation (0-1)'
)
@click.option(
    '--output-format',
    type=click.Choice(['json', 'table', 'summary']),
    default='table',
    help='Output format for results'
)
@click.option(
    '--output-file',
    type=click.Path(path_type=Path),
    help='Save results to file'
)
@click.option(
    '--recursive', '-R',
    is_flag=True,
    help='Process directories recursively'
)
@click.option(
    '--parallel/--sequential',
    default=True,
    help='Enable parallel validation'
)
@click.option(
    '--fail-fast',
    is_flag=True,
    help='Stop on first validation failure'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
@click.option(
    '--quiet', '-q',
    is_flag=True,
    help='Suppress non-error output'
)
def validate_command(
    paths: tuple,
    rules: tuple,
    compliance_framework: Optional[str],
    threshold: float,
    output_format: str,
    output_file: Optional[Path],
    recursive: bool,
    parallel: bool,
    fail_fast: bool,
    verbose: bool,
    quiet: bool
):
    """
    Validate synthetic documents for quality and compliance.
    
    Examples:
        # Validate all documents in directory
        synthdata validate ./output/
        
        # Check specific compliance
        synthdata validate docs/ --compliance-framework gdpr
        
        # Validate with specific rules
        synthdata validate file.pdf -r structure -r quality
        
        # Save detailed report
        synthdata validate ./dataset/ --output-format json -o report.json
    """
    asyncio.run(_validate_async(
        paths=paths,
        rules=rules,
        compliance_framework=compliance_framework,
        threshold=threshold,
        output_format=output_format,
        output_file=output_file,
        recursive=recursive,
        parallel=parallel,
        fail_fast=fail_fast,
        verbose=verbose,
        quiet=quiet
    ))


async def _validate_async(
    paths: tuple,
    rules: tuple,
    compliance_framework: Optional[str],
    threshold: float,
    output_format: str,
    output_file: Optional[Path],
    recursive: bool,
    parallel: bool,
    fail_fast: bool,
    verbose: bool,
    quiet: bool
):
    """Async implementation of validate command"""
    formatter = OutputFormatter(verbose=verbose, quiet=quiet)
    
    try:
        # Expand paths to find all files
        files_to_validate = []
        for path in paths:
            path = Path(path)
            if path.is_file():
                files_to_validate.append(path)
            elif path.is_dir():
                if recursive:
                    files_to_validate.extend(path.rglob('*'))
                else:
                    files_to_validate.extend(path.glob('*'))
        
        # Filter to supported file types
        supported_extensions = {'.pdf', '.docx', '.json', '.xml', '.html'}
        files_to_validate = [
            f for f in files_to_validate 
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]
        
        if not files_to_validate:
            formatter.warning("No valid files found to validate")
            return
        
        formatter.info(f"Found {len(files_to_validate)} files to validate")
        
        # Initialize validation engines
        quality_engine = create_quality_engine()
        privacy_engine = create_privacy_engine()
        
        # Prepare validation rules
        if 'all' in rules:
            validation_rules = [
                'structure', 'completeness', 'quality', 
                'privacy', 'compliance', 'format', 'content'
            ]
        else:
            validation_rules = list(rules)
        
        # Progress tracking
        progress = ProgressTracker(
            total=len(files_to_validate), 
            desc="Validating documents"
        )
        
        # Validate files
        results = []
        failed_count = 0
        
        if parallel:
            # Parallel validation
            tasks = [
                _validate_single_file(
                    file_path=file_path,
                    rules=validation_rules,
                    compliance_framework=compliance_framework,
                    threshold=threshold,
                    quality_engine=quality_engine,
                    privacy_engine=privacy_engine
                )
                for file_path in files_to_validate
            ]
            
            for task in asyncio.as_completed(tasks):
                result = await task
                results.append(result)
                progress.update(1)
                
                if not result['valid']:
                    failed_count += 1
                    if fail_fast:
                        formatter.error(f"Validation failed for {result['file']}")
                        break
        else:
            # Sequential validation
            for file_path in files_to_validate:
                result = await _validate_single_file(
                    file_path=file_path,
                    rules=validation_rules,
                    compliance_framework=compliance_framework,
                    threshold=threshold,
                    quality_engine=quality_engine,
                    privacy_engine=privacy_engine
                )
                results.append(result)
                progress.update(1)
                
                if not result['valid']:
                    failed_count += 1
                    if fail_fast:
                        formatter.error(f"Validation failed for {file_path}")
                        break
        
        progress.close()
        
        # Format and display results
        if output_format == 'json':
            output_data = {
                'summary': {
                    'total_files': len(results),
                    'valid_files': len(results) - failed_count,
                    'failed_files': failed_count,
                    'validation_rules': validation_rules,
                    'threshold': threshold
                },
                'results': results
            }
            
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
                formatter.success(f"Results saved to {output_file}")
            else:
                formatter.output(json.dumps(output_data, indent=2))
                
        elif output_format == 'table':
            _display_table_results(formatter, results, failed_count)
            
        elif output_format == 'summary':
            _display_summary_results(formatter, results, failed_count)
        
        # Exit code based on validation results
        if failed_count > 0:
            raise click.ClickException(
                f"Validation failed for {failed_count} file(s)"
            )
        else:
            formatter.success("All files passed validation")
            
    except KeyboardInterrupt:
        formatter.warning("\nValidation interrupted by user")
    except Exception as e:
        formatter.error(f"Validation failed: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise


async def _validate_single_file(
    file_path: Path,
    rules: List[str],
    compliance_framework: Optional[str],
    threshold: float,
    quality_engine,
    privacy_engine
) -> dict:
    """Validate a single file"""
    try:
        result = {
            'file': str(file_path),
            'valid': True,
            'score': 1.0,
            'issues': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Load document
        # In real implementation, this would use proper document loaders
        doc_data = {}
        if file_path.suffix == '.json':
            with open(file_path) as f:
                doc_data = json.load(f)
        
        # Apply validation rules
        for rule in rules:
            if rule == 'structure':
                validation = await quality_engine.validate_structure(doc_data)
            elif rule == 'completeness':
                validation = await quality_engine.validate_completeness(doc_data)
            elif rule == 'quality':
                validation = await quality_engine.validate_quality(doc_data)
            elif rule == 'privacy':
                validation = await privacy_engine.validate_privacy(doc_data)
            elif rule == 'compliance' and compliance_framework:
                validation = await privacy_engine.validate_compliance(
                    doc_data, framework=compliance_framework
                )
            else:
                continue
            
            # Process validation result
            if validation:
                result['metrics'][rule] = validation.get('score', 0)
                if validation.get('issues'):
                    result['issues'].extend(validation['issues'])
                if validation.get('warnings'):
                    result['warnings'].extend(validation['warnings'])
                
                if validation.get('score', 0) < threshold:
                    result['valid'] = False
        
        # Calculate overall score
        if result['metrics']:
            result['score'] = sum(result['metrics'].values()) / len(result['metrics'])
        
        return result
        
    except Exception as e:
        logger.error(f"Error validating {file_path}: {e}")
        return {
            'file': str(file_path),
            'valid': False,
            'score': 0.0,
            'issues': [f"Validation error: {str(e)}"],
            'warnings': [],
            'metrics': {}
        }


def _display_table_results(formatter: OutputFormatter, results: List[dict], 
                          failed_count: int):
    """Display results in table format"""
    formatter.output("\nValidation Results:")
    formatter.output("-" * 80)
    formatter.output(f"{'File':<40} {'Valid':<8} {'Score':<8} {'Issues':<10}")
    formatter.output("-" * 80)
    
    for result in results:
        file_name = Path(result['file']).name
        if len(file_name) > 37:
            file_name = file_name[:34] + "..."
        
        valid_str = " Yes" if result['valid'] else " No"
        score_str = f"{result['score']:.2f}"
        issues_str = str(len(result['issues']))
        
        formatter.output(
            f"{file_name:<40} {valid_str:<8} {score_str:<8} {issues_str:<10}"
        )
    
    formatter.output("-" * 80)
    formatter.output(
        f"Total: {len(results)} files, "
        f"Valid: {len(results) - failed_count}, "
        f"Failed: {failed_count}"
    )


def _display_summary_results(formatter: OutputFormatter, results: List[dict], 
                           failed_count: int):
    """Display summary of validation results"""
    total_files = len(results)
    valid_files = total_files - failed_count
    
    formatter.output("\nValidation Summary:")
    formatter.output(f"  Total files validated: {total_files}")
    formatter.output(f"  Valid files: {valid_files} ({valid_files/total_files*100:.1f}%)")
    formatter.output(f"  Failed files: {failed_count} ({failed_count/total_files*100:.1f}%)")
    
    # Aggregate metrics
    all_metrics = {}
    for result in results:
        for metric, score in result['metrics'].items():
            if metric not in all_metrics:
                all_metrics[metric] = []
            all_metrics[metric].append(score)
    
    if all_metrics:
        formatter.output("\nAverage Scores by Rule:")
        for metric, scores in all_metrics.items():
            avg_score = sum(scores) / len(scores)
            formatter.output(f"  {metric.capitalize()}: {avg_score:.2f}")
    
    # Top issues
    all_issues = []
    for result in results:
        all_issues.extend(result['issues'])
    
    if all_issues:
        formatter.output("\nTop Issues:")
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        for issue, count in sorted(issue_counts.items(), 
                                  key=lambda x: x[1], reverse=True)[:5]:
            formatter.output(f"  - {issue} ({count} occurrences)")


def create_validate_command():
    """Factory function to create validate command"""
    return validate_command


__all__ = ['validate_command', 'create_validate_command']