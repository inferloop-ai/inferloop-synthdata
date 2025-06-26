# TextNLP SDK Documentation

## Overview

The TextNLP SDK provides a powerful Python interface for generating synthetic text and NLP data. This documentation covers all classes, methods, and features available in the SDK.

## Table of Contents

1. [Installation](#installation)
2. [Core Classes](#core-classes)
3. [Generators](#generators)
4. [Validators](#validators)
5. [Templates](#templates)
6. [Advanced Features](#advanced-features)
7. [API Reference](#api-reference)
8. [Examples](#examples)

## Installation

```bash
pip install inferloop-textnlp

# With specific extras
pip install inferloop-textnlp[langchain]  # For template support
pip install inferloop-textnlp[streaming]  # For streaming support
pip install inferloop-textnlp[all]       # All features
```

## Core Classes

### BaseGenerator

The abstract base class for all text generators.

```python
from textnlp.sdk.base_generator import BaseGenerator

class BaseGenerator(ABC):
    """Abstract base class for text generation"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    def generate_batch(self, prompts: List[str], **kwargs) -> List[GenerationResult]:
        """Generate text for multiple prompts"""
        pass
    
    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream text generation"""
        pass
```

### GenerationResult

Container for generation results with metadata.

```python
from textnlp.sdk.base_generator import GenerationResult

@dataclass
class GenerationResult:
    text: str                          # Generated text
    model: str                         # Model used
    prompt: str                        # Original prompt
    tokens_used: int                   # Number of tokens
    generation_time: float             # Time taken (seconds)
    parameters: Dict[str, Any]         # Generation parameters
    validation_results: Optional[Dict] # Validation metrics
```

## Generators

### GPT2Generator

Main generator for GPT-2 family models.

```python
from textnlp.sdk.llm_gpt2 import GPT2Generator

# Initialize generator
generator = GPT2Generator(
    model_name="gpt2-large",      # Model: gpt2, gpt2-medium, gpt2-large, gpt2-xl
    device="cuda",                # Device: cuda, cpu, auto
    cache_dir="./models",         # Model cache directory
    compile_model=True            # Compile for performance
)

# Generate text
result = generator.generate(
    prompt="Write a story about AI",
    max_length=200,               # Maximum tokens
    min_length=50,                # Minimum tokens
    temperature=0.7,              # Randomness (0.0-2.0)
    top_p=0.9,                   # Nucleus sampling
    top_k=50,                    # Top-k sampling
    repetition_penalty=1.2,       # Reduce repetition
    do_sample=True,              # Enable sampling
    num_return_sequences=1,       # Number of outputs
    seed=42                      # Random seed
)

# Access results
print(result.text)
print(f"Tokens used: {result.tokens_used}")
print(f"Generation time: {result.generation_time:.2f}s")
```

#### Batch Generation

```python
# Generate for multiple prompts
prompts = [
    "Write about space exploration",
    "Describe artificial intelligence",
    "Explain quantum computing"
]

results = generator.generate_batch(
    prompts,
    max_length=150,
    batch_size=2,        # Process 2 at a time
    show_progress=True   # Show progress bar
)

for i, result in enumerate(results):
    print(f"Prompt {i+1}: {result.text[:100]}...")
```

#### Streaming Generation

```python
# Stream tokens as they're generated
async def stream_example():
    async for token in generator.generate_stream(
        prompt="Tell me a long story",
        max_length=500,
        temperature=0.8
    ):
        print(token, end='', flush=True)

# Run streaming
import asyncio
asyncio.run(stream_example())
```

### LangChainTemplateGenerator

Advanced template-based generation using LangChain.

```python
from textnlp.sdk.langchain_template import LangChainTemplateGenerator

# Initialize template generator
template_gen = LangChainTemplateGenerator(
    model_name="gpt2-xl",
    temperature=0.7
)

# Define template
template = """
You are a {role} writing a {document_type}.

Topic: {topic}
Tone: {tone}
Length: {length} words

{document_type}:
"""

# Generate with template
result = template_gen.generate_from_template(
    template=template,
    variables={
        "role": "technical writer",
        "document_type": "API documentation",
        "topic": "RESTful web services",
        "tone": "professional",
        "length": "500"
    }
)

print(result.text)
```

#### Chain Templates

```python
# Create a chain of templates
from langchain import PromptTemplate, LLMChain

# First template: Generate outline
outline_template = PromptTemplate(
    input_variables=["topic"],
    template="Create an outline for an article about {topic}:"
)

# Second template: Expand outline
article_template = PromptTemplate(
    input_variables=["outline"],
    template="Write a detailed article based on this outline:\n{outline}"
)

# Chain them together
chain = template_gen.create_chain([outline_template, article_template])
article = chain.run(topic="Machine Learning in Healthcare")
```

#### Custom Templates

```python
# Load templates from file
template_gen.load_templates("./templates/")

# Use loaded template
result = template_gen.generate(
    template_name="email_template",
    variables={
        "sender": "John Doe",
        "recipient": "Jane Smith",
        "subject": "Project Update"
    }
)

# Save custom template
template_gen.save_template(
    name="product_description",
    template="""
    Product: {product_name}
    Category: {category}
    
    Write a compelling product description that highlights:
    - Key features: {features}
    - Target audience: {audience}
    - Unique selling points
    """,
    metadata={"author": "marketing_team", "version": "1.0"}
)
```

## Validators

### TextValidator

Comprehensive validation for generated text.

```python
from textnlp.sdk.validation import TextValidator

# Initialize validator
validator = TextValidator()

# Validate text
validation_results = validator.validate(
    text=generated_text,
    reference_text=original_text,  # Optional
    checks=[
        'grammar',      # Grammar checking
        'spelling',     # Spelling errors
        'readability',  # Readability scores
        'sentiment',    # Sentiment analysis
        'toxicity',     # Harmful content
        'relevance'     # Relevance to prompt
    ]
)

# Access results
print(f"Grammar score: {validation_results['grammar_score']}")
print(f"Readability: {validation_results['readability']['flesch_reading_ease']}")
print(f"Sentiment: {validation_results['sentiment']['label']}")
```

### BLEUROUGEValidator

Specialized validator for BLEU and ROUGE metrics.

```python
from textnlp.sdk.validation.bleu_rouge import BLEUROUGEValidator

# Initialize BLEU/ROUGE validator
br_validator = BLEUROUGEValidator()

# Calculate BLEU scores
bleu_scores = br_validator.calculate_bleu(
    hypothesis=generated_text,
    references=[ref1, ref2, ref3],
    weights=(0.25, 0.25, 0.25, 0.25)  # BLEU-1 to BLEU-4
)

print(f"BLEU-1: {bleu_scores['bleu_1']:.4f}")
print(f"BLEU-4: {bleu_scores['bleu_4']:.4f}")

# Calculate ROUGE scores
rouge_scores = br_validator.calculate_rouge(
    hypothesis=generated_text,
    references=[ref1, ref2, ref3]
)

print(f"ROUGE-1 F1: {rouge_scores['rouge_1']['f1']:.4f}")
print(f"ROUGE-L F1: {rouge_scores['rouge_l']['f1']:.4f}")

# Combined validation
combined_scores = br_validator.validate_all(
    generated_texts=[gen1, gen2, gen3],
    reference_texts=[ref1, ref2, ref3],
    aggregate=True  # Return aggregated scores
)
```

### GPT4Evaluator

Advanced evaluation using GPT-4 (requires API key).

```python
from textnlp.sdk.validation.gpt4_eval import GPT4Evaluator

# Initialize GPT-4 evaluator
evaluator = GPT4Evaluator(api_key="your-openai-api-key")

# Evaluate quality
quality_score = evaluator.evaluate_quality(
    generated_text=generated_text,
    prompt=original_prompt,
    criteria=[
        "relevance",      # Relevance to prompt
        "coherence",      # Logical flow
        "fluency",        # Language quality
        "creativity",     # Originality
        "accuracy"        # Factual correctness
    ]
)

print(f"Overall quality: {quality_score['overall']:.2f}/10")
print(f"Detailed feedback: {quality_score['feedback']}")

# Compare multiple generations
comparison = evaluator.compare_texts(
    texts=[gen1, gen2, gen3],
    prompt=original_prompt,
    return_best=True
)

print(f"Best generation: {comparison['best_index']}")
print(f"Ranking: {comparison['ranking']}")
```

### HumanEvaluationInterface

Interface for collecting human feedback.

```python
from textnlp.sdk.validation.human_interface import HumanEvaluationInterface

# Initialize human evaluation interface
human_eval = HumanEvaluationInterface(
    storage_backend="sqlite",  # or "postgresql", "mongodb"
    db_path="./evaluations.db"
)

# Create evaluation task
task_id = human_eval.create_task(
    text=generated_text,
    prompt=original_prompt,
    evaluation_criteria={
        "quality": {"type": "scale", "min": 1, "max": 10},
        "relevance": {"type": "scale", "min": 1, "max": 5},
        "issues": {"type": "multiselect", "options": ["grammar", "coherence", "factual", "other"]},
        "comments": {"type": "text"}
    }
)

# Get evaluation URL (for web interface)
eval_url = human_eval.get_evaluation_url(task_id)
print(f"Share this URL for evaluation: {eval_url}")

# Retrieve results
results = human_eval.get_results(task_id)
print(f"Average quality: {results['aggregated']['quality']['mean']:.2f}")
print(f"Comments: {results['individual_responses'][0]['comments']}")
```

## Templates

### Template Management

```python
from textnlp.sdk.template_manager import TemplateManager

# Initialize template manager
tm = TemplateManager()

# Register template
tm.register_template(
    name="email_campaign",
    template="""
    Subject: {subject_line}
    
    Dear {customer_name},
    
    {opening_paragraph}
    
    {main_content}
    
    {call_to_action}
    
    Best regards,
    {company_name}
    """,
    validators={
        "subject_line": lambda x: len(x) <= 100,
        "customer_name": lambda x: x.isalpha()
    }
)

# Generate from template
result = tm.generate(
    template_name="email_campaign",
    model="gpt2-large",
    variables={
        "subject_line": "Special Offer Just for You!",
        "customer_name": "Sarah",
        "opening_paragraph_prompt": "Write a friendly opening",
        "main_content_prompt": "Describe our summer sale",
        "call_to_action_prompt": "Create urgency to buy now",
        "company_name": "TechStore"
    }
)
```

### Dynamic Templates

```python
# Create dynamic template with conditionals
dynamic_template = """
{% if language == 'formal' %}
Dear Mr./Ms. {{last_name}},
{% else %}
Hi {{first_name}}!
{% endif %}

{{content}}

{% if include_signature %}
Sincerely,
{{sender_name}}
{{sender_title}}
{% endif %}
"""

# Use with Jinja2 rendering
result = tm.generate_dynamic(
    template=dynamic_template,
    model="gpt2-medium",
    variables={
        "language": "formal",
        "last_name": "Johnson",
        "content_prompt": "Write about our new product launch",
        "include_signature": True,
        "sender_name": "Alice Smith",
        "sender_title": "Product Manager"
    }
)
```

## Advanced Features

### Fine-tuning Support

```python
from textnlp.sdk.fine_tuning import FineTuner

# Initialize fine-tuner
fine_tuner = FineTuner(
    base_model="gpt2",
    output_dir="./fine_tuned_models"
)

# Prepare dataset
dataset = fine_tuner.prepare_dataset(
    texts=training_texts,
    validation_split=0.1
)

# Fine-tune model
fine_tuner.train(
    dataset=dataset,
    epochs=3,
    batch_size=4,
    learning_rate=5e-5,
    warmup_steps=100,
    save_steps=500
)

# Load and use fine-tuned model
ft_generator = GPT2Generator(
    model_name="./fine_tuned_models/checkpoint-final"
)
```

### Caching and Optimization

```python
from textnlp.sdk.cache import GenerationCache

# Initialize cache
cache = GenerationCache(
    backend="redis",  # or "memory", "disk"
    ttl=3600,        # Cache for 1 hour
    max_size=1000    # Maximum entries
)

# Use with generator
cached_generator = cache.wrap(generator)

# First call - generates and caches
result1 = cached_generator.generate("Write about AI")

# Second call - returns from cache
result2 = cached_generator.generate("Write about AI")

# Clear cache
cache.clear()
```

### Batch Processing

```python
from textnlp.sdk.batch_processor import BatchProcessor

# Initialize batch processor
processor = BatchProcessor(
    generator=generator,
    max_workers=4,         # Parallel workers
    batch_size=32,         # Items per batch
    progress_bar=True,     # Show progress
    error_handling="skip"  # or "retry", "fail"
)

# Process large dataset
input_prompts = load_prompts("prompts.txt")  # 10,000 prompts

results = processor.process(
    input_prompts,
    output_file="results.jsonl",
    checkpoint_interval=100,  # Save every 100 items
    resume_from_checkpoint=True
)

# Get statistics
stats = processor.get_stats()
print(f"Processed: {stats['success']}")
print(f"Failed: {stats['failed']}")
print(f"Average time: {stats['avg_time']:.2f}s")
```

### Model Management

```python
from textnlp.sdk.model_manager import ModelManager

# Initialize model manager
mm = ModelManager(cache_dir="./models")

# List available models
models = mm.list_models()
for model in models:
    print(f"{model['name']}: {model['size_mb']}MB")

# Download model
mm.download_model("gpt2-xl")

# Check model info
info = mm.get_model_info("gpt2-xl")
print(f"Parameters: {info['parameters']}")
print(f"License: {info['license']}")

# Delete model
mm.delete_model("gpt2-medium")

# Auto-download on first use
generator = GPT2Generator(
    model_name="gpt-j-6b",
    auto_download=True
)
```

## API Reference

### Configuration Classes

```python
@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_length: int = 100
    min_length: int = 0
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    do_sample: bool = True
    early_stopping: bool = False
    num_beams: int = 1
    num_return_sequences: int = 1
    seed: Optional[int] = None

@dataclass
class ValidationConfig:
    """Configuration for text validation"""
    check_grammar: bool = True
    check_spelling: bool = True
    check_toxicity: bool = True
    check_readability: bool = True
    calculate_bleu: bool = False
    calculate_rouge: bool = False
    custom_checks: List[Callable] = field(default_factory=list)
```

### Utility Functions

```python
from textnlp.sdk.utils import (
    tokenize,
    detokenize,
    count_tokens,
    truncate_text,
    clean_text,
    split_into_chunks
)

# Tokenization utilities
tokens = tokenize(text, model_name="gpt2")
text = detokenize(tokens, model_name="gpt2")
count = count_tokens(text, model_name="gpt2")

# Text processing
cleaned = clean_text(text, remove_urls=True, remove_emails=True)
truncated = truncate_text(text, max_tokens=100, model_name="gpt2")
chunks = split_into_chunks(text, chunk_size=500, overlap=50)
```

### Exceptions

```python
from textnlp.sdk.exceptions import (
    TextNLPException,
    ModelNotFoundError,
    GenerationError,
    ValidationError,
    TemplateError,
    RateLimitError
)

try:
    result = generator.generate(prompt)
except ModelNotFoundError as e:
    print(f"Model not found: {e}")
except GenerationError as e:
    print(f"Generation failed: {e}")
except RateLimitError as e:
    print(f"Rate limit exceeded. Retry after: {e.retry_after}s")
```

## Examples

### Example 1: Content Generation Pipeline

```python
from textnlp import Pipeline, GPT2Generator, TextValidator, TemplateManager

# Create pipeline
pipeline = Pipeline()

# Add components
pipeline.add_generator(GPT2Generator("gpt2-large"))
pipeline.add_validator(TextValidator())
pipeline.add_template_manager(TemplateManager())

# Configure pipeline
pipeline.configure({
    "generation": {
        "temperature": 0.8,
        "max_length": 300
    },
    "validation": {
        "min_quality_score": 0.7,
        "retry_on_fail": True,
        "max_retries": 3
    }
})

# Run pipeline
result = pipeline.run(
    template="blog_post",
    variables={
        "topic": "Artificial Intelligence",
        "tone": "informative",
        "audience": "general"
    }
)

print(result.text)
print(f"Quality score: {result.validation['quality_score']}")
```

### Example 2: A/B Testing

```python
from textnlp import ABTester

# Initialize A/B tester
tester = ABTester()

# Define variants
tester.add_variant("A", {
    "model": "gpt2-large",
    "temperature": 0.7,
    "top_p": 0.9
})

tester.add_variant("B", {
    "model": "gpt2-xl",
    "temperature": 0.8,
    "top_p": 0.95
})

# Run test
results = tester.run_test(
    prompts=test_prompts,
    num_generations_per_prompt=10,
    evaluation_criteria=["quality", "diversity", "relevance"]
)

# Get winner
winner = results.get_winner(metric="quality")
print(f"Winner: Variant {winner}")
print(f"Statistical significance: {results.p_value:.4f}")
```

### Example 3: Custom Generator

```python
from textnlp.sdk.base_generator import BaseGenerator

class CustomGenerator(BaseGenerator):
    """Custom generator implementation"""
    
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)
    
    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        # Custom generation logic
        start_time = time.time()
        
        # Your generation code here
        generated_text = self.model.generate(prompt, **kwargs)
        
        return GenerationResult(
            text=generated_text,
            model=self.model_name,
            prompt=prompt,
            tokens_used=len(generated_text.split()),
            generation_time=time.time() - start_time,
            parameters=kwargs
        )
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[GenerationResult]:
        return [self.generate(p, **kwargs) for p in prompts]
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        # Custom streaming logic
        for token in self.model.stream_generate(prompt, **kwargs):
            yield token

# Use custom generator
generator = CustomGenerator("./my_custom_model")
result = generator.generate("Hello world")
```

## Best Practices

1. **Resource Management**: Always close generators and validators when done
2. **Error Handling**: Wrap generation calls in try-except blocks
3. **Caching**: Use caching for repeated generations
4. **Batch Processing**: Use batch methods for multiple prompts
5. **Validation**: Always validate generated content
6. **Monitoring**: Track metrics and performance

## Performance Tips

1. **GPU Usage**: Use CUDA when available for 10-100x speedup
2. **Model Selection**: Choose the smallest model that meets your quality needs
3. **Batch Size**: Optimize batch size based on your GPU memory
4. **Caching**: Enable caching for frequently used prompts
5. **Compilation**: Use `compile_model=True` for PyTorch 2.0+

## Troubleshooting

Common issues and solutions:

1. **Import Errors**: Ensure all dependencies are installed with `pip install inferloop-textnlp[all]`
2. **CUDA Errors**: Check GPU availability with `torch.cuda.is_available()`
3. **Memory Issues**: Reduce batch size or use smaller models
4. **Slow Generation**: Enable GPU, model compilation, and caching
5. **Quality Issues**: Adjust temperature and sampling parameters

## Support

- GitHub Issues: [github.com/inferloop/textnlp/issues](https://github.com/inferloop/textnlp/issues)
- Documentation: [docs.inferloop.com/textnlp](https://docs.inferloop.com/textnlp)
- Examples: [github.com/inferloop/textnlp-examples](https://github.com/inferloop/textnlp-examples)