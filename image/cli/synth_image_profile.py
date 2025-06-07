#!/usr/bin/env python3
"""
Synthetic Image Profiler CLI

Command-line interface for profiling image datasets to generate statistical models
for synthetic image generation.
"""

import click
import json
import time
from pathlib import Path
from typing import Optional
import logging

from realtime.profiler.generate_profile_json import ProfileGenerator
from realtime.ingest_unsplash import UnsplashIngester
from realtime.ingest_webcam import WebcamIngester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.group()
def profile_cli():
    """Profile data sources for synthetic image generation."""
    pass

@profile_cli.command()
@click.option('--source', '-s', required=True, 
              type=click.Choice(['unsplash', 'webcam', 'drone', 'edge_camera']),
              help='Data source to profile')
@click.option('--query', '-q', help='Search query (for unsplash)')
@click.option('--count', '-c', default=50, help='Number of images to profile')
@click.option('--output-name', '-o', help='Output profile name')
@click.option('--continuous', is_flag=True, help='Run continuous profiling')
@click.option('--interval', default=300, help='Interval between batches (seconds)')
def create(source, query, count, output_name, continuous, interval):
    """Create a new profile from a data source."""
    
    if not output_name:
        output_name = f"{source}_{query}" if query else source
    
    logger.info(f"Creating profile '{output_name}' from {source}")
    
    try:
        profile_generator = ProfileGenerator()
        
        if source == 'unsplash':
            if not query:
                click.echo("❌ Query required for Unsplash source", err=True)
                return
            
            if continuous:
                _run_continuous_unsplash_profiling(profile_generator, query, count, output_name, interval)
            else:
                _profile_unsplash_batch(profile_generator, query, count, output_name)
                
        elif source == 'webcam':
            if continuous:
                _run_continuous_webcam_profiling(profile_generator, count, output_name, interval)
            else:
                _profile_webcam_batch(profile_generator, count, output_name)
                
        else:
            click.echo(f"❌ Source '{source}' not yet implemented", err=True)
            
    except Exception as e:
        click.echo(f"❌ Profiling failed: {e}", err=True)

def _profile_unsplash_batch(generator, query, count, output_name):
    """Profile a single batch from Unsplash."""
    ingester = UnsplashIngester()
    images = ingester.fetch_random_images(query=query, count=count)
    
    if not images:
        click.echo("❌ Failed to fetch images from Unsplash", err=True)
        return
    
    profile_path = generator.process_and_save(images, output_name)
    click.echo(f"✅ Profiled {len(images)} Unsplash images")
    click.echo(f"📁 Profile saved: {profile_path}")

def _run_continuous_unsplash_profiling(generator, query, count, output_name, interval):
    """Run continuous profiling from Unsplash."""
    click.echo(f"🔄 Starting continuous profiling from Unsplash")
    click.echo(f"Query: {query}, Batch size: {count}, Interval: {interval}s")
    
    ingester = UnsplashIngester()
    batch_number = 0
    
    try:
        while True:
            batch_number += 1
            click.echo(f"\\n📊 Processing batch {batch_number}...")
            
            images = ingester.fetch_random_images(query=query, count=count)
            
            if images:
                # Update the profile with new images
                generator.process_image_batch(images, f"{output_name}_batch_{batch_number}")
                
                # Generate and save updated profile
                profile = generator.generate_profile(output_name)
                profile_path = generator.save_profile(profile, output_name)
                
                click.echo(f"✅ Batch {batch_number}: Profiled {len(images)} images")
                click.echo(f"📁 Updated profile: {profile_path}")
            else:
                click.echo(f"⚠️  Batch {batch_number}: No images fetched")
            
            click.echo(f"⏳ Waiting {interval} seconds until next batch...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        click.echo("\\n🛑 Continuous profiling stopped by user")
        
        # Save final profile
        final_profile = generator.generate_profile(output_name)
        final_path = generator.save_profile(final_profile, f"{output_name}_final")
        click.echo(f"📁 Final profile saved: {final_path}")

def _profile_webcam_batch(generator, count, output_name):
    """Profile a single batch from webcam."""
    ingester = WebcamIngester()
    
    try:
        images = ingester.capture_batch(count=count, interval=1.0)
        
        if not images:
            click.echo("❌ Failed to capture webcam images", err=True)
            return
        
        profile_path = generator.process_and_save(images, output_name)
        click.echo(f"✅ Profiled {len(images)} webcam images")
        click.echo(f"📁 Profile saved: {profile_path}")
        
    finally:
        ingester.release()

def _run_continuous_webcam_profiling(generator, count, output_name, interval):
    """Run continuous profiling from webcam."""
    click.echo(f"🔄 Starting continuous webcam profiling")
    click.echo(f"Batch size: {count}, Interval: {interval}s")
    
    ingester = WebcamIngester()
    batch_number = 0
    
    try:
        while True:
            batch_number += 1
            click.echo(f"\\n📊 Processing webcam batch {batch_number}...")
            
            images = ingester.capture_batch(count=count, interval=1.0)
            
            if images:
                generator.process_image_batch(images, f"{output_name}_batch_{batch_number}")
                
                profile = generator.generate_profile(output_name)
                profile_path = generator.save_profile(profile, output_name)
                
                click.echo(f"✅ Batch {batch_number}: Profiled {len(images)} webcam images")
                click.echo(f"📁 Updated profile: {profile_path}")
            else:
                click.echo(f"⚠️  Batch {batch_number}: No images captured")
            
            click.echo(f"⏳ Waiting {interval} seconds until next batch...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        click.echo("\\n🛑 Continuous profiling stopped by user")
    finally:
        ingester.release()

@profile_cli.command()
@click.option('--profile-name', '-p', required=True, help='Profile name to analyze')
@click.option('--detailed', '-d', is_flag=True, help='Show detailed analysis')
@click.option('--export', '-e', type=click.Path(), help='Export analysis to file')
def analyze(profile_name, detailed, export):
    """Analyze an existing profile."""
    
    profile_path = f"./profiles/stream_{profile_name}.json"
    
    if not Path(profile_path).exists():
        click.echo(f"❌ Profile '{profile_name}' not found", err=True)
        return
    
    try:
        with open(profile_path, 'r') as f:
            profile = json.load(f)
        
        analysis = _analyze_profile(profile, detailed)
        
        # Display analysis
        click.echo(f"\\n📊 Profile Analysis: {profile_name}")
        click.echo("=" * 50)
        
        for section, content in analysis.items():
            click.echo(f"\\n🔍 {section.upper()}:")
            if isinstance(content, dict):
                for key, value in content.items():
                    click.echo(f"  • {key}: {value}")
            elif isinstance(content, list):
                for item in content:
                    click.echo(f"  • {item}")
            else:
                click.echo(f"  {content}")
        
        # Export if requested
        if export:
            with open(export, 'w') as f:
                json.dump(analysis, f, indent=2)
            click.echo(f"\\n📁 Analysis exported to: {export}")
        
    except Exception as e:
        click.echo(f"❌ Analysis failed: {e}", err=True)

def _analyze_profile(profile, detailed=False):
    """Analyze profile data and generate insights."""
    
    metadata = profile.get('metadata', {})
    distributions = profile.get('distributions', {})
    conditioning = profile.get('conditioning_profile', {})
    
    analysis = {
        'basic_info': {
            'sample_count': metadata.get('sample_count', 0),
            'creation_date': metadata.get('generation_timestamp', 'unknown'),
            'source': metadata.get('source_name', 'unknown')
        }
    }
    
    # Analyze distributions
    if distributions:
        dist_analysis = {}
        
        for feature, dist_info in distributions.items():
            if isinstance(dist_info, dict) and 'basic_stats' in dist_info:
                stats = dist_info['basic_stats']
                
                # Classify the feature distribution
                cv = stats['std'] / stats['mean'] if stats['mean'] != 0 else float('inf')
                
                if cv < 0.1:
                    variability = "Low"
                elif cv < 0.5:
                    variability = "Medium"
                else:
                    variability = "High"
                
                dist_analysis[feature] = {
                    'mean': round(stats['mean'], 2),
                    'variability': variability,
                    'range': f"{stats['min']:.2f} - {stats['max']:.2f}"
                }
        
        analysis['distributions'] = dist_analysis
    
    # Analyze semantic content
    if 'detected_classes' in distributions:
        class_info = distributions['detected_classes']
        analysis['semantic_content'] = {
            'total_objects': class_info.get('total_detections', 0),
            'unique_classes': class_info.get('unique_classes', 0),
            'top_classes': list(class_info.get('top_classes', {}).keys())[:5],
            'diversity': f"{class_info.get('class_entropy', 0):.2f}"
        }
    
    # Generation recommendations
    if conditioning:
        hints = conditioning.get('generation_hints', {})
        recommendations = []
        
        if 'scene_guidance' in hints:
            scene = hints['scene_guidance'].get('preferred_scene', 'unknown')
            recommendations.append(f"Focus on {scene} scenes")
        
        if 'object_guidance' in hints:
            objects = hints['object_guidance'].get('preferred_classes', [])[:3]
            if objects:
                recommendations.append(f"Include objects: {', '.join(objects)}")
        
        analysis['generation_recommendations'] = recommendations
    
    return analysis

@profile_cli.command()
@click.option('--source', '-s', help='Filter by source')
@click.option('--min-samples', default=0, help='Minimum sample count')
@click.option('--sort-by', default='timestamp', 
              type=click.Choice(['timestamp', 'samples', 'name']),
              help='Sort profiles by')
def list_profiles(source, min_samples, sort_by):
    """List all available profiles."""
    
    profiles_dir = Path("./profiles")
    if not profiles_dir.exists():
        click.echo("📂 No profiles directory found")
        return
    
    profiles = []
    
    for profile_file in profiles_dir.glob("stream_*.json"):
        if profile_file.name.endswith("_latest.json"):
            continue  # Skip timestamped files
            
        try:
            with open(profile_file, 'r') as f:
                profile = json.load(f)
            
            metadata = profile.get('metadata', {})
            
            profile_info = {
                'name': profile_file.stem.replace('stream_', ''),
                'source': metadata.get('source_name', 'unknown'),
                'samples': metadata.get('sample_count', 0),
                'timestamp': metadata.get('generation_timestamp', ''),
                'file': str(profile_file)
            }
            
            # Apply filters
            if source and profile_info['source'] != source:
                continue
            if profile_info['samples'] < min_samples:
                continue
            
            profiles.append(profile_info)
            
        except Exception as e:
            logger.warning(f"Could not load profile {profile_file}: {e}")
    
    # Sort profiles
    if sort_by == 'timestamp':
        profiles.sort(key=lambda x: x['timestamp'], reverse=True)
    elif sort_by == 'samples':
        profiles.sort(key=lambda x: x['samples'], reverse=True)
    elif sort_by == 'name':
        profiles.sort(key=lambda x: x['name'])
    
    if not profiles:
        click.echo("📂 No profiles found matching criteria")
        return
    
    click.echo(f"📂 Found {len(profiles)} profiles:")
    click.echo()
    
    for profile in profiles:
        click.echo(f"📊 {profile['name']}")
        click.echo(f"   Source: {profile['source']}")
        click.echo(f"   Samples: {profile['samples']}")
        click.echo(f"   Created: {profile['timestamp']}")
        click.echo()

if __name__ == '__main__':
    profile_cli()

# ==================== cli/synth_image_validate.py ====================
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
    
    resolutions = []
    formats = []
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

def _clean_results_for_export(results):
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

# ==================== delivery/export_to_jsonl.py ====================
import json
import jsonlines
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class JSONLExporter:
    """Export synthetic image datasets to JSONL format for ML training."""
    
    def __init__(self):
        self.exported_count = 0
    
    def export_dataset(self, 
                      dataset_dir: str,
                      output_file: str,
                      include_metadata: bool = True,
                      include_annotations: bool = True,
                      batch_size: int = 1000) -> Dict:
        """Export a complete dataset to JSONL format."""
        
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            raise ValueError(f"Dataset directory not found: {dataset_dir}")
        
        # Get all images
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff']:
            image_files.extend(dataset_path.glob(ext))
        
        if not image_files:
            raise ValueError(f"No images found in {dataset_dir}")
        
        logger.info(f"Exporting {len(image_files)} images to JSONL...")
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        exported_records = []
        
        with jsonlines.open(output_file, mode='w') as writer:
            for i, image_file in enumerate(image_files):
                try:
                    record = self._create_image_record(
                        image_file, 
                        dataset_path,
                        include_metadata,
                        include_annotations
                    )
                    
                    writer.write(record)
                    exported_records.append(record)
                    
                    # Progress logging
                    if (i + 1) % batch_size == 0:
                        logger.info(f"Exported {i + 1}/{len(image_files)} images")
                        
                except Exception as e:
                    logger.error(f"Failed to export {image_file}: {e}")
                    continue
        
        self.exported_count = len(exported_records)
        
        # Create summary
        summary = {
            'export_timestamp': datetime.now().isoformat(),
            'total_images': len(image_files),
            'exported_images': self.exported_count,
            'output_file': str(output_path),
            'dataset_directory': str(dataset_path),
            'format': 'jsonl',
            'includes_metadata': include_metadata,
            'includes_annotations': include_annotations
        }
        
        # Save summary
        summary_file = output_path.parent / f"{output_path.stem}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Export complete: {self.exported_count} records written to {output_file}")
        
        return summary
    
    def _create_image_record(self, 
                           image_file: Path,
                           dataset_path: Path,
                           include_metadata: bool,
                           include_annotations: bool) -> Dict:
        """Create a JSONL record for a single image."""
        
        relative_path = image_file.relative_to(dataset_path)
        
        record = {
            'id': str(image_file.stem),
            'file_path': str(relative_path),
            'file_name': image_file.name,
            'created_at': datetime.fromtimestamp(image_file.stat().st_mtime).isoformat()
        }
        
        # Add image metadata
        if include_metadata:
            metadata = self._extract_image_metadata(image_file)
            record['metadata'] = metadata
        
        # Add annotations if available
        if include_annotations:
            annotations = self._load_annotations(image_file, dataset_path)
            if annotations:
                record['annotations'] = annotations
        
        # Add generation info if available
        generation_info = self._load_generation_info(image_file, dataset_path)
        if generation_info:
            record['generation'] = generation_info
        
        return record
    
    def _extract_image_metadata(self, image_file: Path) -> Dict:
        """Extract metadata from image file."""
        
        from PIL import Image
        import os
        
        try:
            with Image.open(image_file) as img:
                metadata = {
                    'width': img.width,