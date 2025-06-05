# cli/main.py
import typer
from typing import List, Optional
import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdk import GPT2Generator, LangChainTemplate, DataFormatter
from sdk.validation import BLEUROUGEValidator

app = typer.Typer(help="Inferloop NLP Synthetic Data Generation CLI")

@app.command()
def generate(
    prompts: List[str] = typer.Argument(..., help="Text prompts to generate from"),
    model: str = typer.Option("gpt2", help="Model to use for generation"),
    max_length: int = typer.Option(100, help="Maximum length of generated text"),
    output: Optional[str] = typer.Option(None, help="Output file path"),
    format_type: str = typer.Option("jsonl", help="Output format (jsonl, csv, markdown)")
):
    """Generate synthetic text from prompts"""
    try:
        typer.echo(f"Loading model: {model}")
        generator = GPT2Generator(model)
        
        typer.echo(f"Generating text for {len(prompts)} prompts...")
        results = generator.batch_generate(prompts, max_length=max_length)
        
        # Format output
        data = [{"prompt": p, "generated": r} for p, r in zip(prompts, results)]
        
        if output:
            formatter = DataFormatter()
            if format_type == "jsonl":
                formatter.to_jsonl(data, output)
            elif format_type == "csv":
                formatter.to_csv(data, output)
            elif format_type == "markdown":
                formatter.to_markdown(data, output)
            
            typer.echo(f"Results saved to {output}")
        else:
            for item in data:
                typer.echo(f"Prompt: {item['prompt']}")
                typer.echo(f"Generated: {item['generated']}")
                typer.echo("---")
                
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def validate(
    references_file: str = typer.Argument(..., help="File containing reference texts"),
    candidates_file: str = typer.Argument(..., help="File containing candidate texts"),
    output: Optional[str] = typer.Option(None, help="Output file for validation scores")
):
    """Validate synthetic text quality using BLEU/ROUGE scores"""
    try:
        # Load data
        with open(references_file, 'r') as f:
            references = [line.strip() for line in f]
        
        with open(candidates_file, 'r') as f:
            candidates = [line.strip() for line in f]
        
        if len(references) != len(candidates):
            typer.echo("Error: References and candidates must have the same length", err=True)
            raise typer.Exit(1)
        
        # Validate
        validator = BLEUROUGEValidator()
        scores = validator.validate_batch(references, candidates)
        
        # Calculate averages
        avg_scores = {
            'avg_bleu': sum(scores['bleu']) / len(scores['bleu']),
            'avg_rouge1': sum(scores['rouge1']) / len(scores['rouge1']),
            'avg_rouge2': sum(scores['rouge2']) / len(scores['rouge2']),
            'avg_rougeL': sum(scores['rougeL']) / len(scores['rougeL'])
        }
        
        typer.echo("Validation Results:")
        for metric, score in avg_scores.items():
            typer.echo(f"{metric}: {score:.4f}")
        
        if output:
            with open(output, 'w') as f:
                json.dump({"individual_scores": scores, "averages": avg_scores}, f, indent=2)
            typer.echo(f"Detailed scores saved to {output}")
            
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def template(
    template_file: str = typer.Argument(..., help="LangChain template JSON file"),
    variables_file: str = typer.Argument(..., help="JSON file with template variables"),
    output: Optional[str] = typer.Option(None, help="Output file for formatted prompts")
):
    """Format prompts using LangChain templates"""
    try:
        # Load template
        template = LangChainTemplate(template_path=template_file)
        
        # Load variables
        with open(variables_file, 'r') as f:
            variable_sets = json.load(f)
        
        # Format prompts
        prompts = template.batch_format(variable_sets)
        
        if output:
            with open(output, 'w') as f:
                for prompt in prompts:
                    f.write(prompt + '\n')
            typer.echo(f"Formatted prompts saved to {output}")
        else:
            for i, prompt in enumerate(prompts, 1):
                typer.echo(f"Prompt {i}:")
                typer.echo(prompt)
                typer.echo("---")
                
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
