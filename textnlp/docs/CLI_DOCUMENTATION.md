# TextNLP CLI Documentation

## Overview

The TextNLP Command Line Interface (CLI) provides a powerful and user-friendly way to generate synthetic text data directly from your terminal. This documentation covers all commands, options, and features available in the CLI.

## Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Commands](#commands)
4. [Global Options](#global-options)
5. [Examples](#examples)
6. [Configuration](#configuration)
7. [Advanced Usage](#advanced-usage)

## Installation

```bash
# Install the package
pip install inferloop-textnlp

# Verify installation
inferloop-nlp --version

# Get help
inferloop-nlp --help
```

## Basic Usage

```bash
# Simple text generation
inferloop-nlp generate "Write a story about space"

# Generate with specific model
inferloop-nlp generate "Explain quantum computing" --model gpt2-large

# Save output to file
inferloop-nlp generate "Create a product description" --output product.txt

# Generate multiple samples
inferloop-nlp generate "Write a haiku" --num-samples 5
```

## Commands

### `generate` - Generate synthetic text

The main command for text generation.

```bash
inferloop-nlp generate [PROMPT] [OPTIONS]
```

#### Arguments

- `PROMPT` (required): The input prompt or path to prompt file

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | `gpt2` | Model to use for generation |
| `--max-tokens` | `-t` | `100` | Maximum number of tokens to generate |
| `--min-tokens` | | `0` | Minimum number of tokens |
| `--temperature` | | `0.7` | Sampling temperature (0.0-2.0) |
| `--top-p` | | `0.9` | Nucleus sampling threshold |
| `--top-k` | | `50` | Top-k sampling parameter |
| `--num-samples` | `-n` | `1` | Number of samples to generate |
| `--seed` | `-s` | `None` | Random seed for reproducibility |
| `--output` | `-o` | `stdout` | Output file path |
| `--format` | `-f` | `text` | Output format (text, json, jsonl) |
| `--stream` | | `False` | Stream output as it's generated |
| `--no-validation` | | `False` | Skip validation |

#### Examples

```bash
# Basic generation
inferloop-nlp generate "Write about artificial intelligence"

# With parameters
inferloop-nlp generate "Create a poem" \
  --model gpt2-xl \
  --max-tokens 200 \
  --temperature 0.9 \
  --top-p 0.95

# Multiple samples with JSON output
inferloop-nlp generate "Generate product names" \
  --num-samples 10 \
  --format json \
  --output names.json

# From file with streaming
inferloop-nlp generate prompts.txt \
  --stream \
  --model gpt2-large
```

### `validate` - Validate generated text

Validate text quality using various metrics.

```bash
inferloop-nlp validate [GENERATED_FILE] [OPTIONS]
```

#### Arguments

- `GENERATED_FILE` (required): Path to generated text file

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--reference` | `-r` | `None` | Reference text file for comparison |
| `--metrics` | `-m` | `all` | Metrics to calculate (bleu,rouge,quality) |
| `--output` | `-o` | `stdout` | Output file for results |
| `--format` | `-f` | `text` | Output format (text, json) |

#### Examples

```bash
# Validate with reference
inferloop-nlp validate generated.txt --reference original.txt

# Specific metrics
inferloop-nlp validate output.txt \
  --metrics bleu,rouge \
  --format json \
  --output scores.json

# Quality check only
inferloop-nlp validate generated.txt --metrics quality
```

### `template` - Generate from templates

Use predefined or custom templates for generation.

```bash
inferloop-nlp template [TEMPLATE_NAME] [OPTIONS]
```

#### Arguments

- `TEMPLATE_NAME` (required): Name of template or path to template file

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--variables` | `-v` | `{}` | Template variables (JSON or key=value) |
| `--model` | `-m` | `gpt2-large` | Model for generation |
| `--list` | `-l` | `False` | List available templates |
| `--output` | `-o` | `stdout` | Output file |

#### Examples

```bash
# List templates
inferloop-nlp template --list

# Use built-in template
inferloop-nlp template email \
  --variables '{"recipient": "John", "subject": "Meeting"}' \
  --output email.txt

# Custom template with variables
inferloop-nlp template my_template.txt \
  --variables "name=Alice,product=laptop,price=999"

# Interactive template filling
inferloop-nlp template blog_post --interactive
```

### `batch` - Batch processing

Process multiple prompts efficiently.

```bash
inferloop-nlp batch [INPUT_FILE] [OPTIONS]
```

#### Arguments

- `INPUT_FILE` (required): File containing prompts (one per line or JSON)

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output-dir` | `-o` | `./output` | Output directory |
| `--model` | `-m` | `gpt2` | Model for generation |
| `--batch-size` | `-b` | `32` | Batch size for processing |
| `--workers` | `-w` | `4` | Number of parallel workers |
| `--format` | `-f` | `jsonl` | Output format |
| `--resume` | | `False` | Resume from checkpoint |
| `--progress` | | `True` | Show progress bar |

#### Examples

```bash
# Process prompts file
inferloop-nlp batch prompts.txt \
  --output-dir results/ \
  --model gpt2-large \
  --batch-size 64

# JSON input with parallel processing
inferloop-nlp batch prompts.json \
  --workers 8 \
  --format json \
  --output-dir batch_results/

# Resume interrupted batch
inferloop-nlp batch large_prompts.txt \
  --resume \
  --output-dir continued_results/
```

### `serve` - Start API server

Launch the REST API server.

```bash
inferloop-nlp serve [OPTIONS]
```

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--host` | `-h` | `0.0.0.0` | Host to bind to |
| `--port` | `-p` | `8000` | Port to listen on |
| `--workers` | `-w` | `1` | Number of worker processes |
| `--reload` | | `False` | Auto-reload on code changes |
| `--model` | `-m` | `all` | Models to load (comma-separated) |
| `--auth` | | `False` | Enable authentication |

#### Examples

```bash
# Start basic server
inferloop-nlp serve

# Production server with multiple workers
inferloop-nlp serve \
  --port 8080 \
  --workers 4 \
  --model gpt2,gpt2-large

# Development server with reload
inferloop-nlp serve \
  --reload \
  --host localhost \
  --port 8000
```

### `chat` - Interactive chat mode

Start an interactive chat session.

```bash
inferloop-nlp chat [OPTIONS]
```

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | `gpt2-large` | Model for chat |
| `--system` | `-s` | `None` | System prompt |
| `--temperature` | | `0.7` | Response temperature |
| `--history` | | `10` | Number of turns to remember |
| `--save` | | `None` | Save conversation to file |

#### Examples

```bash
# Start chat
inferloop-nlp chat

# With custom system prompt
inferloop-nlp chat \
  --system "You are a helpful assistant" \
  --model gpt2-xl

# Save conversation
inferloop-nlp chat \
  --save conversation.json \
  --history 20
```

### `config` - Manage configuration

Configure default settings.

```bash
inferloop-nlp config [SUBCOMMAND] [OPTIONS]
```

#### Subcommands

- `show`: Display current configuration
- `set`: Set configuration values
- `reset`: Reset to defaults
- `export`: Export configuration
- `import`: Import configuration

#### Examples

```bash
# Show configuration
inferloop-nlp config show

# Set default model
inferloop-nlp config set default.model gpt2-large

# Set API key
inferloop-nlp config set api.key "your-api-key"

# Export configuration
inferloop-nlp config export --output config.json

# Import configuration
inferloop-nlp config import config.json
```

### `models` - Model management

Manage available models.

```bash
inferloop-nlp models [SUBCOMMAND] [OPTIONS]
```

#### Subcommands

- `list`: List available models
- `download`: Download a model
- `delete`: Delete a model
- `info`: Show model information

#### Examples

```bash
# List models
inferloop-nlp models list

# Download model
inferloop-nlp models download gpt2-xl

# Get model info
inferloop-nlp models info gpt2-large

# Delete model
inferloop-nlp models delete gpt2-medium
```

## Global Options

These options work with all commands:

| Option | Short | Description |
|--------|-------|-------------|
| `--help` | `-h` | Show help message |
| `--version` | `-V` | Show version |
| `--verbose` | `-v` | Increase verbosity (-vv for debug) |
| `--quiet` | `-q` | Suppress output |
| `--config` | `-c` | Path to config file |
| `--no-color` | | Disable colored output |
| `--json` | | Output in JSON format |

## Configuration

### Configuration File

Create a configuration file at `~/.inferloop/textnlp/config.yaml`:

```yaml
# Default model settings
default:
  model: gpt2-large
  temperature: 0.7
  max_tokens: 150
  top_p: 0.9
  top_k: 50

# API settings
api:
  key: your-api-key
  endpoint: https://api.inferloop.com

# Model paths
models:
  cache_dir: ~/.inferloop/models
  custom_models:
    - path: /path/to/model
      name: my-custom-model

# Generation defaults
generation:
  validate: true
  format: text
  streaming: false

# Logging
logging:
  level: INFO
  file: ~/.inferloop/logs/textnlp.log
```

### Environment Variables

Configure via environment variables:

```bash
# API configuration
export INFERLOOP_API_KEY="your-api-key"
export INFERLOOP_API_ENDPOINT="https://api.inferloop.com"

# Model settings
export INFERLOOP_MODEL_CACHE="~/.inferloop/models"
export INFERLOOP_DEFAULT_MODEL="gpt2-large"

# Generation defaults
export INFERLOOP_MAX_TOKENS="200"
export INFERLOOP_TEMPERATURE="0.8"

# Logging
export INFERLOOP_LOG_LEVEL="DEBUG"
```

## Advanced Usage

### Piping and Chaining

```bash
# Pipe input
echo "Write a poem about coding" | inferloop-nlp generate -

# Chain commands
inferloop-nlp generate "Create product names" -n 10 | \
  inferloop-nlp validate - --metrics quality

# Use with other tools
cat prompts.txt | \
  inferloop-nlp batch - | \
  jq '.[] | select(.quality_score > 0.8)'
```

### Custom Templates

Create a template file `email_template.txt`:

```
Subject: {{subject}}

Dear {{recipient}},

{{greeting}}

{{body}}

{{closing}},
{{sender}}
```

Use the template:

```bash
inferloop-nlp template email_template.txt \
  --variables '{
    "subject": "Project Update",
    "recipient": "Team",
    "greeting": "I hope this email finds you well.",
    "body": "I wanted to update you on our progress...",
    "closing": "Best regards",
    "sender": "John"
  }'
```

### Batch Processing with Custom Format

Create `batch_config.json`:

```json
{
  "prompts": [
    {
      "id": "001",
      "prompt": "Write a product description for a smartwatch",
      "params": {"temperature": 0.7, "max_tokens": 150}
    },
    {
      "id": "002", 
      "prompt": "Create a tagline for eco-friendly products",
      "params": {"temperature": 0.9, "max_tokens": 50}
    }
  ],
  "defaults": {
    "model": "gpt2-large",
    "top_p": 0.9
  }
}
```

Process the batch:

```bash
inferloop-nlp batch batch_config.json \
  --format custom \
  --output-dir results/
```

### Scripting and Automation

Create a generation script `generate_content.sh`:

```bash
#!/bin/bash

# Configuration
MODEL="gpt2-xl"
OUTPUT_DIR="generated"
DATE=$(date +%Y%m%d)

# Create output directory
mkdir -p "$OUTPUT_DIR/$DATE"

# Generate different content types
echo "Generating blog posts..."
inferloop-nlp generate "Write a tech blog post" \
  --model $MODEL \
  --max-tokens 500 \
  --num-samples 5 \
  --output "$OUTPUT_DIR/$DATE/blog_posts.json" \
  --format json

echo "Generating product descriptions..."
inferloop-nlp batch product_prompts.txt \
  --model $MODEL \
  --output-dir "$OUTPUT_DIR/$DATE/products" \
  --workers 4

echo "Validating generated content..."
inferloop-nlp validate "$OUTPUT_DIR/$DATE/blog_posts.json" \
  --metrics quality \
  --output "$OUTPUT_DIR/$DATE/validation_report.json"

echo "Content generation complete!"
```

### Integration with Other Tools

```bash
# Generate and analyze with Python
inferloop-nlp generate "Write 10 marketing slogans" -n 10 -f json | \
  python -c "
import json, sys
data = json.load(sys.stdin)
slogans = [item['text'] for item in data['results']]
print(f'Average length: {sum(len(s) for s in slogans) / len(slogans):.1f} chars')
print(f'Unique words: {len(set(' '.join(slogans).split()))}')
"

# Generate and translate
inferloop-nlp generate "Write a welcome message" | \
  trans -b :es

# Generate and speak
inferloop-nlp generate "Create a podcast intro" | \
  espeak
```

## Performance Tips

1. **Use appropriate models**: Smaller models for speed, larger for quality
2. **Batch processing**: Use batch command for multiple prompts
3. **Caching**: Models are cached after first use
4. **GPU acceleration**: Automatically used when available
5. **Streaming**: Use `--stream` for long generations

## Troubleshooting

### Common Issues

1. **Command not found**
   ```bash
   # Ensure pip bin directory is in PATH
   export PATH="$HOME/.local/bin:$PATH"
   ```

2. **Model download fails**
   ```bash
   # Set custom cache directory
   export INFERLOOP_MODEL_CACHE="/path/to/cache"
   
   # Download manually
   inferloop-nlp models download gpt2 --retry 3
   ```

3. **Out of memory**
   ```bash
   # Use smaller model or reduce batch size
   inferloop-nlp generate "prompt" --model gpt2
   inferloop-nlp batch file.txt --batch-size 8
   ```

4. **Slow generation**
   ```bash
   # Check if GPU is being used
   inferloop-nlp config show | grep device
   
   # Force CPU usage if needed
   CUDA_VISIBLE_DEVICES="" inferloop-nlp generate "prompt"
   ```

## Best Practices

1. **Always validate**: Use the validate command for quality assurance
2. **Use templates**: For consistent output formats
3. **Set seeds**: For reproducible results
4. **Monitor resources**: Use `--verbose` to see resource usage
5. **Save outputs**: Always save important generations
6. **Batch similar prompts**: Group similar prompts for efficiency

## Examples Gallery

### Marketing Content

```bash
# Product descriptions
inferloop-nlp template product_description \
  --variables "product=Wireless Earbuds,features=noise-cancelling;waterproof,price=$99"

# Email campaigns
inferloop-nlp batch email_prompts.txt \
  --model gpt2-xl \
  --output-dir campaigns/
```

### Creative Writing

```bash
# Story generation
inferloop-nlp generate "Once upon a time in a digital world" \
  --model gpt2-xl \
  --max-tokens 1000 \
  --temperature 0.9 \
  --output story.txt

# Poetry
inferloop-nlp generate "Write a haiku about programming" \
  --num-samples 10 \
  --format json | jq -r '.results[].text'
```

### Technical Documentation

```bash
# API documentation
inferloop-nlp template api_docs \
  --variables "endpoint=/users,method=GET,description=Retrieve user list"

# Code comments
echo "def calculate_fibonacci(n):" | \
  inferloop-nlp generate - \
  --model gpt2-large \
  --max-tokens 100
```

### Data Augmentation

```bash
# Generate variations
inferloop-nlp generate "The product works great" \
  --num-samples 20 \
  --temperature 0.8 \
  --output variations.json

# Sentiment variations
for sentiment in positive negative neutral; do
  inferloop-nlp generate "Write a $sentiment product review" \
    --num-samples 50 \
    --output "reviews_${sentiment}.txt"
done
```

## Support

- **Documentation**: Run `inferloop-nlp --help` or `inferloop-nlp [command] --help`
- **Examples**: See `/usr/share/inferloop/examples/` after installation
- **Issues**: Report at [github.com/inferloop/textnlp/issues](https://github.com/inferloop/textnlp/issues)
- **Community**: Join our Discord at [discord.gg/inferloop](https://discord.gg/inferloop)