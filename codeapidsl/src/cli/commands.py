# src/cli/commands.py
import click
import json
import yaml
from pathlib import Path
from ..generators.code_llama_generator import CodeLlamaGenerator
from ..generators.starcoder_generator import StarCoderGenerator
from ..validators.syntax_validator import SyntaxValidator
from ..delivery.formatters import JSONLFormatter, CSVFormatter

@click.group()
def cli():
    """Synthetic Code Generation CLI"""
    pass

@cli.command()
@click.option('--prompts', '-p', required=True, help='Comma-separated prompts or path to file')
@click.option('--language', '-l', default='python', help='Programming language')
@click.option('--framework', '-f', help='Framework to use')
@click.option('--count', '-c', default=10, type=int, help='Number of samples to generate')
@click.option('--output', '-o', default='output.jsonl', help='Output file path')
@click.option('--format', 'output_format', default='jsonl', 
              type=click.Choice(['jsonl', 'csv']), help='Output format')
@click.option('--validate/--no-validate', default=True, help='Enable validation')
@click.option('--config', help='Path to configuration file')
def generate(prompts, language, framework, count, output, output_format, validate, config):
    """Generate synthetic code samples"""
    
    # Load configuration if provided
    config_data = {}
    if config:
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
    
    # Parse prompts
    if prompts.endswith('.txt') or prompts.endswith('.yaml'):
        # Read from file
        with open(prompts, 'r') as f:
            if prompts.endswith('.yaml'):
                prompt_data = yaml.safe_load(f)
                prompt_list = prompt_data.get('prompts', [])
            else:
                prompt_list = [line.strip() for line in f.readlines() if line.strip()]
    else:
        # Parse comma-separated
        prompt_list = [p.strip() for p in prompts.split(',')]
    
    # Initialize generator
    gen_config = GenerationConfig(
        language=language,
        framework=framework,
        count=count,
        **config_data.get('generation', {})
    )
    
    if language in ["python", "javascript", "typescript"]:
        generator = CodeLlamaGenerator(gen_config)
    else:
        generator = StarCoderGenerator(gen_config)
    
    click.echo(f"Generating {count} code samples in {language}...")
    
    # Generate code
    generated_code = generator.generate_batch(prompt_list[:count])
    
    # Validate if requested
    validation_results = []
    if validate:
        click.echo("Validating generated code...")
        validator = SyntaxValidator()
        
        for code_sample in generated_code:
            result = validator.validate_code(code_sample["code"], language)
            validation_results.append({
                "id": code_sample["id"],
                "validation": result
            })
    
    # Format and save output
    if output_format == 'jsonl':
        formatter = JSONLFormatter()
    else:
        formatter = CSVFormatter()
    
    formatted_data = formatter.format(generated_code, validation_results)
    
    # Write to file
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        f.write(formatted_data)
    
    click.echo(f"Generated code saved to {output}")
    
    # Summary
    valid_count = sum(1 for r in validation_results if r["validation"]["valid"])
    click.echo(f"Summary: {len(generated_code)} samples generated, {valid_count} passed validation")

@cli.command()
@click.option('--file', '-f', required=True, help='Code file to validate')
@click.option('--language', '-l', required=True, help='Programming language')
def validate_file(file, language):
    """Validate a code file"""
    validator = SyntaxValidator()
    
    with open(file, 'r') as f:
        code = f.read()
    
    result = validator.validate_code(code, language)
    
    if result["valid"]:
        click.echo(click.style("✓ Code is valid!", fg='green'))
    else:
        click.echo(click.style("✗ Validation failed:", fg='red'))
        for error in result["errors"]:
            click.echo(f"  {error}")
    
    if result["warnings"]:
        click.echo(click.style("Warnings:", fg='yellow'))
        for warning in result["warnings"]:
            click.echo(f"  {warning}")

if __name__ == '__main__':
    cli()
