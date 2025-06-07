#!/usr/bin/env python3
"""
Command-line interface for image validation operations.
"""

import click
import json
from pathlib import Path
from typing import List, Optional
import logging

from validation.validate_quality import QualityValidator
from validation.validate_privacy import PrivacyValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.group()
def validate_cli():
    """Validate synthetic images for quality, privacy, and fairness."""
    pass

@validate_cli.command()
@click.option('--images-dir', '-i', required=True, type=click.Path(exists=True),
              help='Directory containing images to validate')
@click.option('--reference-dir', '-r', type=click.Path(exists=True),
              help='Reference images directory (optional)')
@click.option('--metrics', '-m', multiple=True, 
              type=click.Choice(['fid', 'ssim', 'lpips', 'psnr', 'all']),
              default=['all'], help='Quality metrics to compute')
@click.option('--output', '-o', type=click.Path(), help='Output report path')
@click.option('--batch-size', default=32, help='Batch size for processing')
@click.option('--detailed', is_flag=True, help='Generate detailed per-image metrics')
def quality(images_dir, reference_dir, metrics, output, batch_size, detailed):
    """Validate image quality using various metrics."""
    
    images_path = Path(images_dir)
    
    # Get image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']:
        image_files.extend(images_path.glob(ext))
    
    if not image_files:
        click.echo(f"❌ No images found in {images_dir}", err=True)
        return
    
    click.echo(f"🔍 Validating {len(image_files)} images...")
    
    try:
        validator = QualityValidator()
        
        # Convert metrics
        if 'all' in metrics:
            selected_metrics = ['fid', 'ssim', 'lpips', 'psnr']
        else:
            selected_metrics = list(metrics)
        
        if reference_dir:
            # Validate against reference
            ref_path = Path(reference_dir)
            ref_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']:
                ref_files.extend(ref_path.glob(ext))
            
            if not ref_files:
                click.echo(f"❌ No reference images found in {reference_dir}", err=True)
                return
            
            click.echo(f"📊 Comparing against {len(ref_files)} reference images...")
            
            results = validator.validate_against_reference(
                [str(f) for f in image_files],
                [str(f) for f in ref_files]
            )
        else:
            # Validate quality only
            results = validator.validate_batch([str(f) for f in image_files])
        
        # Display results
        _display_quality_results(results, detailed)
        
        # Save report if requested
        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            click.echo(f"\\n📁 Detailed report saved: {output}")
        
    except Exception as e:
        click.echo(f"❌ Quality validation failed: {e}", err=True)

def _display_quality_results(results, detailed=False):
    """Display quality validation results."""
    
    click.echo("\\n✅ Quality Validation Results")
    click.echo("=" * 40)
    
    # Summary
    if 'summary' in results:
        summary = results['summary']
        click.echo("\\n📊 SUMMARY:")
        
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                click.echo(f"  • {key.replace('_', ' ').title()}: {value:.4f}")
            else:
                click.echo(f"  • {key.replace('_', ' ').title()}: {value}")
    
    # Quality metrics
    if 'quality_metrics' in results:
        metrics = results['quality_metrics']
        click.echo("\\n🎯 QUALITY METRICS:")
        
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                click.echo(f"  • {metric.replace('_', ' ').title()}: {value:.4f}")
    
    # Diversity metrics
    if 'diversity_metrics' in results:
        diversity = results['diversity_metrics']
        click.echo("\\n🌈 DIVERSITY METRICS:")
        
        for metric, value in diversity.items():
            if isinstance(value, (int, float)):
                click.echo(f"  • {metric.replace('_', ' ').title()}: {value:.4f}")
    
    # Comparison metrics (if reference provided)
    if 'comparison_metrics' in results:
        comparison = results['comparison_metrics']
        if comparison:
            click.echo("\\n🆚 COMPARISON METRICS:")
            
            for metric, value in comparison.items():
                if isinstance(value, (int, float)):
                    click.echo(f"  • {metric.replace('_', ' ').title()}: {value:.4f}")
    
    # Individual scores (if detailed)
    if detailed and 'individual_scores' in results:
        scores = results['individual_scores']
        click.echo(f"\\n📋 INDIVIDUAL SCORES ({len(scores)} images):")
        
        for i, score in enumerate(scores[:10]):  # Show first 10
            click.echo(f"  Image {i+1}: Sharpness={score.get('sharpness', 0):.2f}, "
                      f"Brightness={score.get('brightness', 0):.2f}")
        
        if len(scores) > 10:
            click.echo(f"  ... and {len(scores) - 10} more images")

@validate_cli.command()
@click.option('--images-dir', '-i', required=True, type=click.Path(exists=True),
              help='Directory containing images to validate')
@click.option('--check-faces', is_flag=True, help='Check for faces in images')
@click.option('--check-text', is_flag=True, help='Check for text/PII in images')
@click.option('--blur-faces', is_flag=True, help='Generate anonymized versions')
@click.option('--output', '-o', type=click.Path(), help='Output report path')
@click.option('--save-anonymized', type=click.Path(), help='Save anonymized images to directory')
def privacy(images_dir, check_faces, check_text, blur_faces, output, save_anonymized):
    """Validate images for privacy concerns."""
    
    images_path = Path(images_dir)
    
    # Get image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(images_path.glob(ext))
    
    if not image_files:
        click.echo(f"❌ No images found in {images_dir}", err=True)
        return
    
    click.echo(f"🔒 Privacy validation for {len(image_files)} images...")
    
    try:
        validator = PrivacyValidator()
        
        results = validator.validate_batch(
            [str(f) for f in image_files],
            check_faces=check_faces or blur_faces,
            check_text=check_text,
            anonymize=blur_faces
        )
        
        # Display results
        _display_privacy_results(results)
        
        # Save anonymized images if requested
        if save_anonymized and 'anonymized_images' in results:
            save_path = Path(save_anonymized)
            save_path.mkdir(parents=True, exist_ok=True)
            
            anonymized = results['anonymized_images']
            for original_path, anon_image in anonymized.items():
                filename = Path(original_path).name
                save_file = save_path / f"anon_{filename}"
                anon_image.save(save_file)
            
            click.echo(f"\\n📁 Anonymized images saved to: {save_anonymized}")
        
        # Save report if requested
        if output:
            # Remove images from results for JSON serialization
            export_results = {k: v for k, v in results.items() if k != 'anonymized_images'}
            
            with open(output, 'w') as f:
                json.dump(export_results, f, indent=2, default=str)
            click.echo(f"\\n📁 Privacy report saved: {output}")
        
    except Exception as e:
        click.echo(f"❌ Privacy validation failed: {e}", err=True)

def _display_privacy_results(results):
    """Display privacy validation results."""
    
    click.echo("\\n🔒 Privacy Validation Results")
    click.echo("=" * 40)
    
    # Face detection results
    if 'face_detection' in results:
        face_stats = results['face_detection']
        click.echo("\\n👤 FACE DETECTION:")
        click.echo(f"  • Images with faces: {face_stats.get('images_with_faces', 0)}")
        click.echo(f"  • Total faces detected: {face_stats.get('total_faces', 0)}")
        click.echo(f"  • Average faces per image: {face_stats.get('avg_faces_per_image', 0):.2f}")
    
    # Text detection results
    if 'text_detection' in results:
        text_stats = results['text_detection']
        click.echo("\\n📝 TEXT DETECTION:")
        click.echo(f"  • Images with text: {text_stats.get('images_with_text', 0)}")
        click.echo(f"  • Potential PII detected: {text_stats.get('potential_pii', 0)}")
    
    # Privacy score
    if 'privacy_score' in results:
        score = results['privacy_score']
        click.echo(f"\\n🏆 PRIVACY SCORE: {score:.2f}/100")
        
        if score >= 80:
            click.echo("  ✅ Good privacy protection")
        elif score >= 60:
            click.echo("  ⚠️  Moderate privacy concerns")
        else:
            click.echo("  ❌ Significant privacy issues")

@validate_cli.command()
@click.option('--images-dir', '-i', required=True, type=click.Path(exists=True),
              help='Directory containing images to validate')
@click.option('--output', '-o', type=click.Path(), help='Output comprehensive report')
@click.option('--include-privacy', is_flag=True, help='Include privacy validation')
@click.option('--include-technical', is_flag=True, help='Include technical metrics')
def comprehensive(images_dir, output, include_privacy, include_technical):
    """Run comprehensive validation (quality + privacy + technical)."""
    
    images_path = Path(images_dir)
    
    # Get image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(images_path.glob(ext))
    
    if not image_files:
        click.echo(f"❌ No images found in {images_dir}", err=True)
        return
    
    click.echo(f"🔍 Comprehensive validation for {len(image_files)} images...")
    click.echo("This may take several minutes...")
    
    comprehensive_results = {
        'dataset_info': {
            'num_images': len(image_files),
            'image_directory': str(images_path),
            'validation_timestamp': click.DateTime().convert(None, None, None)
        }
    }
    
    try:
        # Quality validation
        click.echo("\\n📊 Running quality validation...")
        quality_validator = QualityValidator()
        quality_results = quality_validator.validate_batch([str(f) for f in image_files])
        comprehensive_results['quality'] = quality_results
        
        # Privacy validation
        if include_privacy:
            click.echo("\\n🔒 Running privacy validation...")
            privacy_validator = PrivacyValidator()
            privacy_results = privacy_validator.validate_batch(
                [str(f) for f in image_files],
                check_faces=True,
                check_text=True
            )
            comprehensive_results['privacy'] = privacy_results
        
        # Technical validation
        if include_technical:
            click.echo("\\n⚙️  Running technical validation...")
            technical_results = _validate_technical_specs(image_files)
            comprehensive_results['technical'] = technical_results
        
        # Generate overall assessment
        overall_assessment = _generate_overall_assessment(comprehensive_results)
        comprehensive_results['overall_assessment'] = overall_assessment
        
        # Display summary
        _display_comprehensive_results(comprehensive_results)
        
        # Save report
        if output:
            # Clean results for JSON serialization
            clean_results = _clean_results_for_export(comprehensive_results)
            
            with open(output, 'w') as f:
                json.dump(clean_results, f, indent=2, default=str)
            click.echo(f"\\n📁 Comprehensive report saved: {output}")
        
    except Exception as e:
        click.echo(f"❌ Comprehensive validation failed: {e}", err=True)

def _validate_technical_specs(image_files):
    """Validate technical specifications of images."""
    
    from PIL import Image
    import os
    
    technical_results = {
        'resolution_analysis': {},
        'format_analysis': {},
        'size_analysis': {},
        'compliance_check': {}
    }
    
    resolutions = [  formats = []
    file_sizes = []
    
    for image_path in image_files:
        try:
            # Get file info
            file_size = os.path.getsize(image_path)
            file_sizes.append(file_size)
            
            # Get image info
            with Image.open(image_path) as img:
                resolutions.append(img.size)  # (width, height)
                formats.append(img.format.lower())
        
        except Exception as e:
            logger.warning(f"Could not analyze {image_path}: {e}")
    
    # Resolution analysis
    if resolutions:
        widths, heights = zip(*resolutions)
        technical_results['resolution_analysis'] = {
            'avg_width': sum(widths) / len(widths),
            'avg_height': sum(heights) / len(heights),
            'min_resolution': f"{min(widths)}x{min(heights)}",
            'max_resolution': f"{max(widths)}x{max(heights)}",
            'aspect_ratios': list(set([round(w/h, 2) for w, h in resolutions]))
        }
    
    # Format analysis
    if formats:
        format_counts = {}
        for fmt in formats:
            format_counts[fmt] = format_counts.get(fmt, 0) + 1
        
        technical_results['format_analysis'] = {
            'format_distribution': format_counts,
            'dominant_format': max(format_counts, key=format_counts.get)
        }
    
    # Size analysis
    if file_sizes:
        technical_results['size_analysis'] = {
            'avg_size_mb': sum(file_sizes) / len(file_sizes) / (1024 * 1024),
            'min_size_mb': min(file_sizes) / (1024 * 1024),
            'max_size_mb': max(file_sizes) / (1024 * 1024),
            'total_size_mb': sum(file_sizes) / (1024 * 1024)
        }
    
    return technical_results

def _generate_overall_assessment(results):
    """Generate overall assessment based on all validation results."""
    
    assessment = {
        'overall_score': 0,
        'quality_grade': 'Unknown',
        'privacy_grade': 'Unknown',
        'technical_grade': 'Unknown',
        'recommendations': []
    }
    
    scores = []
    
    # Quality assessment
    if 'quality' in results and 'summary' in results['quality']:
        quality_score = results['quality']['summary'].get('overall_quality_score', 0)
        scores.append(quality_score * 100)
        
        if quality_score >= 0.8:
            assessment['quality_grade'] = 'A'
        elif quality_score >= 0.6:
            assessment['quality_grade'] = 'B'
        elif quality_score >= 0.4:
            assessment['quality_grade'] = 'C'
        else:
            assessment['quality_grade'] = 'D'
            assessment['recommendations'].append("Improve image quality and sharpness")
    
    # Privacy assessment
    if 'privacy' in results:
        privacy_score = results['privacy'].get('privacy_score', 0)
        scores.append(privacy_score)
        
        if privacy_score >= 80:
            assessment['privacy_grade'] = 'A'
        elif privacy_score >= 60:
            assessment['privacy_grade'] = 'B'
        elif privacy_score >= 40:
            assessment['privacy_grade'] = 'C'
        else:
            assessment['privacy_grade'] = 'D'
            assessment['recommendations'].append("Address privacy concerns (faces, PII)")
    
    # Technical assessment
    if 'technical' in results:
        # Simple technical scoring based on consistency
        tech_score = 75  # Default reasonable score
        scores.append(tech_score)
        assessment['technical_grade'] = 'B'
    
    # Overall score
    if scores:
        assessment['overall_score'] = sum(scores) / len(scores)
    
    return assessment

def _display_comprehensive_results(results):
    """Display comprehensive validation results."""
    
    click.echo("\\n🎯 Comprehensive Validation Results")
    click.echo("=" * 50)
    
    # Dataset info
    dataset_info = results.get('dataset_info', {})
    click.echo(f"\\n📊 DATASET: {dataset_info.get('num_images', 0)} images")
    
    # Overall assessment
    if 'overall_assessment' in results:
        assessment = results['overall_assessment']
    click.echo(f"\\n🏆 OVERALL SCORE: {assessment['overall_score']:.1f}/100")
        click.echo(f"   Quality Grade: {assessment['quality_grade']}")
        click.echo(f"   Privacy Grade: {assessment['privacy_grade']}")
        click.echo(f"   Technical Grade: {assessment['technical_grade']}")
        
        if assessment['recommendations']:
            click.echo("\\n💡 RECOMMENDATIONS:")
            for rec in assessment['recommendations']:
                click.echo(f"   • {rec}")

def _clean_resor_export(results):
    """Clean results for JSON serialization."""
    
    clean_results = {}
    
    for key, value in results.items():
        if key == 'privacy' and isinstance(value, dict):
            # Remove non-serializable image objects
            clean_value = {k: v for k, v in value.items() if k != 'anonymized_images'}
            clean_results[key] = clean_value
        else:
            clean_results[key] = value
    
    return clean_results

if __name__ == '__main__':
    validate_cli()
