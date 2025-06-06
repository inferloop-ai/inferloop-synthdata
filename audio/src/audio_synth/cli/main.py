# audio_synth/cli/main.py
"""
Command Line Interface for Audio Synthetic Data Framework
"""

import click
import yaml
import torch
import torchaudio
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
from datetime import datetime
import logging

from ..sdk.client import AudioSynthSDK
from ..core.utils.config import load_config
from ..core.utils.io import save_audio, load_audio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def cli(ctx, config, verbose):
    """Audio Synthetic Data Generation and Validation CLI"""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['verbose'] = verbose
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

@cli.command()
@click.option('--method', '-m', 
              type=click.Choice(['diffusion', 'gan', 'vae', 'tts', 'vocoder']),
              default='diffusion', help='Generation method')
@click.option('--prompt', '-p', type=str, help='Text prompt for generation')
@click.option('--num-samples', '-n', type=int, default=1, help='Number of samples to generate')
@click.option('--output-dir', '-o', type=click.Path(), default='./output', help='Output directory')
@click.option('--duration', '-d', type=float, default=5.0, help='Audio duration in seconds')
@click.option('--sample-rate', '-sr', type=int, default=22050, help='Sample rate')
@click.option('--privacy-level', type=click.Choice(['low', 'medium', 'high']), 
              default='medium', help='Privacy protection level')
@click.option('--seed', type=int, help='Random seed for reproducibility')
@click.option('--speaker-id', type=str, help='Target speaker ID for conditioning')
@click.option('--language', type=str, default='en', help='Language code')
@click.option('--gender', type=click.Choice(['male', 'female', 'other']), help='Target gender')
@click.option('--age-group', type=click.Choice(['child', 'adult', 'elderly']), help='Target age group')
@click.option('--accent', type=str, help='Target accent/dialect')
@click.pass_context
def generate(ctx, method, prompt, num_samples, output_dir, duration, sample_rate,
             privacy_level, seed, speaker_id, language, gender, age_group, accent):
    """Generate synthetic audio samples"""
    
    # Setup configuration
    config_path = ctx.obj.get('config')
    sdk = AudioSynthSDK(config_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare generation parameters
    generation_params = {
        'conditions': {
            'speaker_id': speaker_id,
            'language': language,
            'demographics': {}
        }
    }
    
    if gender:
        generation_params['conditions']['demographics']['gender'] = gender
    if age_group:
        generation_params['conditions']['demographics']['age_group'] = age_group
    if accent:
        generation_params['conditions']['demographics']['accent'] = accent
    
    if seed:
        generation_params['seed'] = seed
        torch.manual_seed(seed)
    
    try:
        click.echo(f"Generating {num_samples} audio samples using {method} method...")
        
        # Generate audio
        audios = sdk.generate(
            method=method,
            prompt=prompt,
            num_samples=num_samples,
            **generation_params
        )
        
        # Save generated audio files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_list = []
        
        for i, audio in enumerate(audios):
            filename = f"{method}_{timestamp}_{i+1:03d}.wav"
            filepath = output_path / filename
            
            # Save audio
            torchaudio.save(str(filepath), audio.unsqueeze(0), sample_rate)
            
            # Prepare metadata
            metadata = {
                'filename': filename,
                'method': method,
                'prompt': prompt,
                'duration': duration,
                'sample_rate': sample_rate,
                'privacy_level': privacy_level,
                'generation_params': generation_params,
                'timestamp': timestamp
            }
            metadata_list.append(metadata)
            
            click.echo(f"Saved: {filepath}")
        
        # Save metadata
        metadata_file = output_path / f"metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata_list, f, indent=2)
        
        click.echo(f"\nGeneration completed!")
        click.echo(f"Files saved to: {output_path}")
        click.echo(f"Metadata saved to: {metadata_file}")
        
    except Exception as e:
        click.echo(f"Error during generation: {str(e)}", err=True)
        raise click.Abort()

@cli.command()
@click.option('--input-dir', '-i', type=click.Path(exists=True), required=True,
              help='Directory containing audio files to validate')
@click.option('--output-file', '-o', type=click.Path(), 
              default='validation_results.json', help='Output validation results file')
@click.option('--validators', '-v', 
              type=click.Choice(['quality', 'privacy', 'fairness', 'all']),
              multiple=True, default=['all'], help='Validators to run')
@click.option('--metadata-file', '-m', type=click.Path(exists=True),
              help='Metadata file with generation parameters')
@click.option('--threshold-quality', type=float, default=0.7, help='Quality threshold')
@click.option('--threshold-privacy', type=float, default=0.8, help='Privacy threshold')
@click.option('--threshold-fairness', type=float, default=0.75, help='Fairness threshold')
@click.option('--generate-report', is_flag=True, help='Generate detailed HTML report')
@click.pass_context
def validate(ctx, input_dir, output_file, validators, metadata_file, 
             threshold_quality, threshold_privacy, threshold_fairness, generate_report):
    """Validate synthetic audio samples"""
    
    config_path = ctx.obj.get('config')
    sdk = AudioSynthSDK(config_path)
    
    input_path = Path(input_dir)
    
    # Load metadata if provided
    metadata_dict = {}
    if metadata_file:
        with open(metadata_file, 'r') as f:
            metadata_list = json.load(f)
            metadata_dict = {item['filename']: item for item in metadata_list}
    
    # Determine which validators to run
    if 'all' in validators:
        validators = ['quality', 'privacy', 'fairness']
    
    # Find audio files
    audio_files = list(input_path.glob('*.wav')) + list(input_path.glob('*.mp3'))
    
    if not audio_files:
        click.echo("No audio files found in input directory", err=True)
        raise click.Abort()
    
    click.echo(f"Validating {len(audio_files)} audio files...")
    
    try:
        # Load and validate audio files
        audios = []
        metadata_list = []
        
        for audio_file in audio_files:
            # Load audio
            audio, sample_rate = torchaudio.load(str(audio_file))
            audios.append(audio.squeeze())
            
            # Get metadata
            metadata = metadata_dict.get(audio_file.name, {})
            metadata['filename'] = audio_file.name
            metadata_list.append(metadata)
        
        # Run validation
        click.echo("Running validation...")
        validation_results = sdk.validate(
            audios=audios,
            metadata=metadata_list,
            validators=list(validators)
        )
        
        # Analyze results
        analysis = analyze_validation_results(
            validation_results, 
            threshold_quality, 
            threshold_privacy, 
            threshold_fairness
        )
        
        # Save results
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'validation_results': validation_results,
            'analysis': analysis,
            'thresholds': {
                'quality': threshold_quality,
                'privacy': threshold_privacy,
                'fairness': threshold_fairness
            },
            'files_validated': [f.name for f in audio_files]
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Print summary
        print_validation_summary(analysis)
        
        # Generate HTML report if requested
        if generate_report:
            report_file = output_file.replace('.json', '_report.html')
            generate_html_report(results_data, report_file)
            click.echo(f"HTML report generated: {report_file}")
        
        click.echo(f"\nValidation results saved to: {output_file}")
        
    except Exception as e:
        click.echo(f"Error during validation: {str(e)}", err=True)
        raise click.Abort()

@cli.command()
@click.option('--input-file', '-i', type=click.Path(exists=True), required=True,
              help='Input audio file')
@click.option('--output-file', '-o', type=click.Path(), help='Output processed audio file')
@click.option('--privacy-level', type=click.Choice(['low', 'medium', 'high']),
              default='medium', help='Privacy enhancement level')
@click.option('--target-speaker', type=str, help='Target speaker for voice conversion')
@click.option('--pitch-shift', type=float, help='Pitch shift in semitones')
@click.option('--time-stretch', type=float, help='Time stretch factor')
@click.pass_context
def enhance_privacy(ctx, input_file, output_file, privacy_level, target_speaker,
                   pitch_shift, time_stretch):
    """Enhance privacy of existing audio files"""
    
    click.echo(f"Enhancing privacy of {input_file}...")
    
    try:
        # Load audio
        audio, sample_rate = torchaudio.load(input_file)
        audio = audio.squeeze()
        
        # Apply privacy enhancements
        enhanced_audio = apply_privacy_enhancements(
            audio, privacy_level, target_speaker, pitch_shift, time_stretch
        )
        
        # Save enhanced audio
        if not output_file:
            input_path = Path(input_file)
            output_file = input_path.parent / f"{input_path.stem}_private{input_path.suffix}"
        
        torchaudio.save(str(output_file), enhanced_audio.unsqueeze(0), sample_rate)
        
        click.echo(f"Privacy-enhanced audio saved to: {output_file}")
        
    except Exception as e:
        click.echo(f"Error enhancing privacy: {str(e)}", err=True)
        raise click.Abort()

@cli.command()
@click.option('--input-dir', '-i', type=click.Path(exists=True), required=True,
              help='Directory containing validation results')
@click.option('--output-file', '-o', type=click.Path(), 
              default='benchmark_report.html', help='Output benchmark report')
@click.pass_context
def benchmark(ctx, input_dir, output_file):
    """Generate benchmark report from multiple validation runs"""
    
    input_path = Path(input_dir)
    result_files = list(input_path.glob('*.json'))
    
    if not result_files:
        click.echo("No validation result files found", err=True)
        raise click.Abort()
    
    click.echo(f"Generating benchmark report from {len(result_files)} result files...")
    
    try:
        # Load all results
        all_results = []
        for result_file in result_files:
            with open(result_file, 'r') as f:
                results = json.load(f)
                all_results.append(results)
        
        # Generate benchmark report
        generate_benchmark_report(all_results, output_file)
        
        click.echo(f"Benchmark report saved to: {output_file}")
        
    except Exception as e:
        click.echo(f"Error generating benchmark: {str(e)}", err=True)
        raise click.Abort()

@cli.command()
@click.option('--output-dir', '-o', type=click.Path(), default='./config',
              help='Output directory for configuration files')
@click.pass_context
def init_config(ctx, output_dir):
    """Initialize configuration files"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Default configuration
    default_config = {
        'audio': {
            'sample_rate': 22050,
            'duration': 5.0,
            'channels': 1,
            'format': 'wav',
            'bit_depth': 16
        },
        'generation': {
            'default_method': 'diffusion',
            'privacy_level': 'medium',
            'num_samples': 100,
            'seed': None
        },
        'validation': {
            'quality_threshold': 0.7,
            'privacy_threshold': 0.8,
            'fairness_threshold': 0.75,
            'protected_attributes': ['gender', 'age', 'accent', 'language']
        },
        'models': {
            'diffusion': {
                'model_path': './models/diffusion_model.pt',
                'denoising_steps': 50,
                'guidance_scale': 7.5
            },
            'gan': {
                'model_path': './models/gan_model.pt',
                'latent_dim': 128
            }
        },
        'output': {
            'default_format': 'wav',
            'normalize': True,
            'add_metadata': True
        }
    }
    
    # Save configuration
    config_file = output_path / 'default.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    click.echo(f"Configuration file created: {config_file}")
    click.echo("Edit this file to customize your settings.")

# Helper functions

def analyze_validation_results(validation_results: Dict[str, List[Dict]], 
                              threshold_quality: float,
                              threshold_privacy: float, 
                              threshold_fairness: float) -> Dict[str, Any]:
    """Analyze validation results against thresholds"""
    
    analysis = {
        'summary': {},
        'passed': {},
        'failed': {},
        'statistics': {}
    }
    
    for validator_name, results_list in validation_results.items():
        if not results_list:
            continue
            
        threshold = {
            'quality': threshold_quality,
            'privacy': threshold_privacy,
            'fairness': threshold_fairness
        }.get(validator_name, 0.5)
        
        # Calculate statistics
        all_metrics = {}
        for result in results_list:
            for metric, value in result.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        stats = {}
        for metric, values in all_metrics.items():
            stats[metric] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'std': (sum((x - sum(values) / len(values))**2 for x in values) / len(values))**0.5
            }
        
        analysis['statistics'][validator_name] = stats
        
        # Count passed/failed
        passed_count = 0
        failed_count = 0
        
        for result in results_list:
            avg_score = sum(result.values()) / len(result) if result else 0
            if avg_score >= threshold:
                passed_count += 1
            else:
                failed_count += 1
        
        analysis['passed'][validator_name] = passed_count
        analysis['failed'][validator_name] = failed_count
        analysis['summary'][validator_name] = {
            'total': len(results_list),
            'passed': passed_count,
            'failed': failed_count,
            'pass_rate': passed_count / len(results_list) if results_list else 0
        }
    
    return analysis

def print_validation_summary(analysis: Dict[str, Any]):
    """Print validation summary to console"""
    
    click.echo("\n" + "="*60)
    click.echo("VALIDATION SUMMARY")
    click.echo("="*60)
    
    for validator_name, summary in analysis['summary'].items():
        click.echo(f"\n{validator_name.upper()} Validation:")
        click.echo(f"  Total samples: {summary['total']}")
        click.echo(f"  Passed: {summary['passed']} ({summary['pass_rate']:.1%})")
        click.echo(f"  Failed: {summary['failed']}")
        
        if validator_name in analysis['statistics']:
            click.echo(f"  Key metrics:")
            stats = analysis['statistics'][validator_name]
            for metric, values in list(stats.items())[:3]:  # Show top 3 metrics
                click.echo(f"    {metric}: {values['mean']:.3f} ± {values['std']:.3f}")

def apply_privacy_enhancements(audio: torch.Tensor, 
                             privacy_level: str,
                             target_speaker: Optional[str],
                             pitch_shift: Optional[float],
                             time_stretch: Optional[float]) -> torch.Tensor:
    """Apply privacy enhancements to audio"""
    
    enhanced = audio.clone()
    
    # Apply transformations based on privacy level
    if privacy_level == 'high':
        # Strong voice conversion
        enhanced = enhanced + torch.randn_like(enhanced) * 0.1
        
    elif privacy_level == 'medium':
        # Moderate pitch and formant shifting
        enhanced = enhanced * 0.95 + torch.randn_like(enhanced) * 0.05
        
    elif privacy_level == 'low':
        # Light modifications
        enhanced = enhanced + torch.randn_like(enhanced) * 0.02
    
    # Apply specific transformations
    if pitch_shift:
        # Simulate pitch shifting (simplified)
        enhanced = enhanced * (1.0 + pitch_shift * 0.01)
    
    if time_stretch:
        # Simulate time stretching (simplified)
        if time_stretch != 1.0:
            new_length = int(len(enhanced) * time_stretch)
            enhanced = torch.nn.functional.interpolate(
                enhanced.unsqueeze(0).unsqueeze(0), 
                size=new_length, 
                mode='linear'
            ).squeeze()
    
    return enhanced

def generate_html_report(results_data: Dict[str, Any], output_file: str):
    """Generate HTML validation report"""
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Audio Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metrics {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
        .pass {{ color: green; font-weight: bold; }}
        .fail {{ color: red; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Audio Validation Report</h1>
        <p>Generated: {results_data['timestamp']}</p>
        <p>Files validated: {len(results_data['files_validated'])}</p>
    </div>
    
    <div class="section">
        <h2>Summary</h2>
        {_generate_summary_table(results_data['analysis'])}
    </div>
    
    <div class="section">
        <h2>Detailed Results</h2>
        {_generate_detailed_results(results_data['validation_results'])}
    </div>
</body>
</html>
"""
    
    with open(output_file, 'w') as f:
        f.write(html_content)

def _generate_summary_table(analysis: Dict[str, Any]) -> str:
    """Generate summary table HTML"""
    
    html = "<table><tr><th>Validator</th><th>Total</th><th>Passed</th><th>Failed</th><th>Pass Rate</th></tr>"
    
    for validator, summary in analysis['summary'].items():
        pass_rate = f"{summary['pass_rate']:.1%}"
        html += f"<tr><td>{validator}</td><td>{summary['total']}</td><td>{summary['passed']}</td><td>{summary['failed']}</td><td>{pass_rate}</td></tr>"
    
    html += "</table>"
    return html

def _generate_detailed_results(validation_results: Dict[str, List[Dict]]) -> str:
    """Generate detailed results HTML"""
    
    html = ""
    for validator, results in validation_results.items():
        html += f"<h3>{validator.title()} Metrics</h3>"
        
        if results:
            # Get all metric names
            all_metrics = set()
            for result in results:
                all_metrics.update(result.keys())
            
            html += "<table><tr><th>Sample</th>"
            for metric in sorted(all_metrics):
                html += f"<th>{metric}</th>"
            html += "</tr>"
            
            for i, result in enumerate(results):
                html += f"<tr><td>Sample {i+1}</td>"
                for metric in sorted(all_metrics):
                    value = result.get(metric, 0)
                    html += f"<td>{value:.3f}</td>"
                html += "</tr>"
            
            html += "</table>"
    
    return html

def generate_benchmark_report(all_results: List[Dict[str, Any]], output_file: str):
    """Generate benchmark comparison report"""
    
    # Aggregate results across all runs
    aggregated = {}
    
    for results in all_results:
        timestamp = results.get('timestamp', 'unknown')
        aggregated[timestamp] = results.get('analysis', {}).get('summary', {})
    
    # Generate HTML report
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Audio Synthesis Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Audio Synthesis Benchmark Report</h1>
        <p>Comparing {len(all_results)} validation runs</p>
    </div>
    
    <h2>Pass Rate Comparison</h2>
    <table>
        <tr><th>Timestamp</th><th>Quality</th><th>Privacy</th><th>Fairness</th></tr>
    """
    
    for timestamp, summary in aggregated.items():
        quality_rate = summary.get('quality', {}).get('pass_rate', 0)
        privacy_rate = summary.get('privacy', {}).get('pass_rate', 0)
        fairness_rate = summary.get('fairness', {}).get('pass_rate', 0)
        
        html_content += f"""
        <tr>
            <td>{timestamp}</td>
            <td>{quality_rate:.1%}</td>
            <td>{privacy_rate:.1%}</td>
            <td>{fairness_rate:.1%}</td>
        </tr>
        """
    
    html_content += """
    </table>
</body>
</html>
"""
    
    with open(output_file, 'w') as f:
        f.write(html_content)

if __name__ == '__main__':
    cli()