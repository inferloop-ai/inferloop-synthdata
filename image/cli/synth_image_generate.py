#!/usr/bin/env python3
"""
Command-line interface for synthetic image generation.
"""

import click
import json
import yaml
from pathlib import Path
from typing import List, Optional
import logging

from generation.generate_diffusion import DiffusionGenerator
from realtime.profiler.generate_profile_json import ProfileGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """Agentic AI Synthetic Image Generator CLI."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    ctx.ensure_object(dict)
    if config:
        with open(config, 'r') as f:
            if config.endswith('.yaml') or config.endswith('.yml'):
                ctx.obj['config'] = yaml.safe_load(f)
            else:
                ctx.obj['config'] = json.load(f)
    else:
        ctx.obj['config'] = {}

@cli.command()
@click.option('--method', '-m', default='diffusion', 
              type=click.Choice(['diffusion', 'gan', 'simulation']),
              help='Generation method to use')
@click.option('--num-images', '-n', default=10, help='Number of images to generate')
@click.option('--output-dir', '-o', default='./data/generated', help='Output directory')
@click.option('--profile', '-p', help='Profile source name to use for conditioning')
@click.option('--prompts', multiple=True, help='Custom prompts (can be used multiple times)')
@click.option('--batch-size', default=4, help='Batch size for generation')
@click.option('--steps', default=50, help='Number of inference steps (diffusion only)')
@click.option('--guidance-scale', default=7.5, help='Guidance scale (diffusion only)')
@click.pass_context
def generate(ctx, method, num_images, output_dir, profile, prompts, batch_size, steps, guidance_scale):
    """Generate synthetic images."""
    
    config = ctx.obj.get('config', {})
    generation_config = config.get('generation', {})
    
    logger.info(f"Generating {num_images} images using {method}")
    
    # Override config with CLI arguments
    if method == 'diffusion':
        # Initialize diffusion generator
        try:
            generator = DiffusionGenerator()
            
            if profile:
                # Generate from profile
                profile_path = f"./profiles/stream_{profile}.json"
                if not Path(profile_path).exists():
                    click.echo(f"Error: Profile '{profile}' not found at {profile_path}", err=True)
                    return
                
                images = generator.generate_from_profile(
                    profile_path=profile_path,
                    num_images=num_images,
                    batch_size=batch_size,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale
                )
                
            elif prompts:
                # Generate from custom prompts
                images = generator.generate_from_prompts(
                    prompts=list(prompts),
                    num_images=num_images,
                    batch_size=batch_size,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale
                )
                
            else:
                # Generate from default prompts
                default_prompts = generation_config.get('default_prompts', [
                    "high quality realistic image",
                    "professional photography, detailed"
                ])
                
                images = generator.generate_from_prompts(
                    prompts=default_prompts,
                    num_images=num_images,
                    batch_size=batch_size,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale
                )
            
            # Save images
            saved_paths = generator.save_images(images, output_dir, f"cli_{method}")
            
            click.echo(f"✅ Generated {len(images)} images")
            click.echo(f"📁 Saved to: {output_dir}")
            click.echo(f"📊 Files: {len(saved_paths)} images")
            
            generator.cleanup()
            
        except Exception as e:
            click.echo(f"❌ Generation failed: {e}", err=True)
            return
            
    else:
        click.echo(f"❌ Method '{method}' not yet implemented", err=True)

@cli.command()
@click.option('--source', '-s', required=True, help='Data source (unsplash, webcam, etc.)')
@click.option('--query', '-q', help='Search query (for unsplash)')
@click.option('--count', '-c', default=10, help='Number of images to process')
@click.option('--output-name', '-o', help='Output profile name (default: source name)')
@click.pass_context
def profile(ctx, source, query, count, output_name):
    """Create a distribution profile from data source."""
    
    if not output_name:
        output_name = f"{source}_{query}" if query else source
    
    logger.info(f"Creating profile '{output_name}' from {source}")
    
    try:
        profile_generator = ProfileGenerator()
        
        if source == 'unsplash':
            from realtime.ingest_unsplash import UnsplashIngester
            
            if not query:
                click.echo("❌ Query required for Unsplash source", err=True)
                return
            
            # Fetch images from Unsplash
            ingester = UnsplashIngester()
            images = ingester.fetch_random_images(query=query, count=count)
            
            if not images:
                click.echo("❌ Failed to fetch images from Unsplash", err=True)
                return
            
            # Generate profile
            profile_path = profile_generator.process_and_save(images, output_name)
            
            click.echo(f"✅ Profiled {len(images)} Unsplash images")
            click.echo(f"📁 Profile saved: {profile_path}")
            
        elif source == 'webcam':
            from realtime.ingest_webcam import WebcamIngester
            
            # Capture from webcam
            ingester = WebcamIngester()
            images = ingester.capture_batch(count=count, interval=1.0)
            
            if not images:
                click.echo("❌ Failed to capture webcam images", err=True)
                return
            
            # Generate profile
            profile_path = profile_generator.process_and_save(images, output_name)
            
            click.echo(f"✅ Profiled {len(images)} webcam images")
            click.echo(f"📁 Profile saved: {profile_path}")
            
            ingester.release()
            
        else:
            click.echo(f"❌ Unsupported source: {source}", err=True)
            
    except Exception as e:
        click.echo(f"❌ Profiling failed: {e}", err=True)

@cli.command()
@click.option('--profile-name', '-p', required=True, help='Profile name to display')
@click.option('--detailed', '-d', is_flag=True, help='Show detailed statistics')
def show_profile(profile_name, detailed):
    """Display profile information."""
    
    profile_path = f"./profiles/stream_{profile_name}.json"
    
    if not Path(profile_path).exists():
        click.echo(f"❌ Profile '{profile_name}' not found", err=True)
        return
    
    try:
        with open(profile_path, 'r') as f:
            profile = json.load(f)
        
        metadata = profile.get('metadata', {})
        conditioning = profile.get('conditioning_profile', {})
        hints = conditioning.get('generation_hints', {})
        
        click.echo(f"📊 Profile: {profile_name}")
        click.echo(f"📅 Created: {metadata.get('generation_timestamp', 'unknown')}")
        click.echo(f"🔢 Samples: {metadata.get('sample_count', 0)}")
        
        # Scene information
        scene_guidance = hints.get('scene_guidance', {})
        if scene_guidance:
            click.echo(f"🌆 Scene: {scene_guidance.get('preferred_scene', 'unknown')}")
        
        # Object information
        object_guidance = hints.get('object_guidance', {})
        if object_guidance:
            top_objects = object_guidance.get('preferred_classes', [])[:5]
            if top_objects:
                click.echo(f"🎯 Top Objects: {', '.join(top_objects)}")
        
        if detailed:
            click.echo("\\n📈 Detailed Statistics:")
            distributions = profile.get('distributions', {})
            for feature, dist_info in distributions.items():
                if isinstance(dist_info, dict) and 'basic_stats' in dist_info:
                    stats = dist_info['basic_stats']
                    click.echo(f"  {feature}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
        
    except Exception as e:
        click.echo(f"❌ Failed to load profile: {e}", err=True)

@cli.command()
def list_profiles():
    """List all available profiles."""
    
    profiles_dir = Path("./profiles")
    if not profiles_dir.exists():
        click.echo("📂 No profiles directory found")
        return
    
    profiles = list(profiles_dir.glob("stream_*.json"))
    
    if not profiles:
        click.echo("📂 No profiles found")
        return
    
    click.echo(f"📂 Found {len(profiles)} profiles:")
    
    for profile_path in sorted(profiles):
        try:
            with open(profile_path, 'r') as f:
                profile = json.load(f)
            
            name = profile_path.stem.replace('stream_', '')
            sample_count = profile.get('metadata', {}).get('sample_count', 0)
            timestamp = profile.get('metadata', {}).get('generation_timestamp', 'unknown')
            
            click.echo(f"  • {name} ({sample_count} samples) - {timestamp}")
            
        except Exception as e:
            click.echo(f"  • {profile_path.stem} (corrupted)")

@cli.command()
@click.option('--images-dir', '-i', required=True, help='Directory containing images to validate')
@click.option('--reference-dir', '-r', help='Reference images directory (optional)')
@click.option('--output', '-o', help='Output validation report path')
def validate(images_dir, reference_dir, output):
    """Validate generated images."""
    
    images_path = Path(images_dir)
    if not images_path.exists():
        click.echo(f"❌ Images directory not found: {images_dir}", err=True)
        return
    
    # Get image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(images_path.glob(ext))
    
    if not image_files:
        click.echo(f"❌ No images found in {images_dir}", err=True)
        return
    
    try:
        from validation.validate_quality import QualityValidator
        
        validator = QualityValidator()
        
        if reference_dir:
            # Validate against reference
            ref_path = Path(reference_dir)
            if not ref_path.exists():
                click.echo(f"❌ Reference directory not found: {reference_dir}", err=True)
                return
            
            ref_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                ref_files.extend(ref_path.glob(ext))
            
            results = validator.validate_against_reference(
                [str(f) for f in image_files],
                [str(f) for f in ref_files]
            )
        else:
            # Validate quality only
            results = validator.validate_batch([str(f) for f in image_files])
        
        # Display results
        click.echo(f"✅ Validated {len(image_files)} images")
        
        if 'quality_metrics' in results:
            metrics = results['quality_metrics']
            click.echo("📊 Quality Metrics:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    click.echo(f"  {metric}: {value:.4f}")
        
        # Save report if requested
        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            click.echo(f"📁 Report saved: {output}")
        
    except Exception as e:
        click.echo(f"❌ Validation failed: {e}", err=True)

if __name__ == '__main__':
    cli()
