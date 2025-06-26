# TextNLP User Guide

## Welcome to TextNLP Synthetic Data Generation

TextNLP is a powerful platform for generating high-quality synthetic text and NLP data using state-of-the-art language models. This guide will help you get started and make the most of the platform.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Common Use Cases](#common-use-cases)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)
7. [Examples](#examples)

## Getting Started

### What is TextNLP?

TextNLP is a comprehensive synthetic data generation platform that helps you:
- Generate realistic text data for training NLP models
- Create test datasets for your applications
- Augment existing datasets with synthetic examples
- Generate content for various use cases (emails, reviews, articles, etc.)

### Key Features

- 🚀 **Multiple Language Models**: From GPT-2 to GPT-J and LLaMA
- 📊 **Quality Validation**: BLEU/ROUGE metrics and human evaluation
- 🔧 **Flexible Interfaces**: SDK, CLI, and REST API
- 🎯 **Template Support**: Structured generation with LangChain
- 🌐 **Streaming**: Real-time generation for large outputs
- 🔒 **Enterprise Ready**: Authentication, rate limiting, and monitoring

## Installation

### Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB+ recommended for large models)
- GPU optional but recommended for large models

### Install via pip

```bash
# Basic installation
pip install inferloop-textnlp

# With all dependencies
pip install inferloop-textnlp[all]

# For development
pip install inferloop-textnlp[dev]
```

### Install from source

```bash
git clone https://github.com/inferloop/textnlp.git
cd textnlp
pip install -e ".[all]"
```

### Verify Installation

```bash
# Check CLI
inferloop-nlp --version

# Test generation
inferloop-nlp generate "Hello world" --model gpt2
```

## Quick Start

### 1. Using the CLI

Generate text with a simple command:

```bash
# Basic generation
inferloop-nlp generate "Write a product review for a smartphone"

# Specify model and parameters
inferloop-nlp generate "Write a poem about AI" \
  --model gpt2-large \
  --max-tokens 200 \
  --temperature 0.8

# Save to file
inferloop-nlp generate "Create a news article" \
  --output article.txt
```

### 2. Using the SDK

```python
from textnlp import TextGenerator

# Initialize generator
generator = TextGenerator(model="gpt2-large")

# Generate text
result = generator.generate(
    prompt="Write a customer support email response",
    max_tokens=150,
    temperature=0.7
)

print(result.text)
print(f"Quality score: {result.validation['quality_score']}")
```

### 3. Using the REST API

Start the API server:

```bash
# Start server
inferloop-nlp serve --port 8000

# Or with Docker
docker run -p 8000:8000 inferloop/textnlp
```

Make API calls:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/textnlp/generate",
    json={
        "prompt": "Write a technical documentation",
        "model": "gpt2-large",
        "max_tokens": 200
    },
    headers={"X-API-Key": "your-api-key"}
)

result = response.json()
print(result['text'])
```

## Common Use Cases

### 1. Dataset Augmentation

Generate synthetic training data for NLP models:

```python
from textnlp import TextGenerator, DataAugmenter

# Original dataset
original_data = [
    "This product is amazing!",
    "Terrible experience, would not recommend.",
    "Average quality, nothing special."
]

# Augment dataset
augmenter = DataAugmenter(model="gpt2-medium")
synthetic_data = augmenter.augment(
    original_data,
    num_synthetic_per_original=3,
    preserve_sentiment=True
)

# Result: 12 samples (3 original + 9 synthetic)
```

### 2. Content Generation

Create various types of content:

```python
from textnlp import TemplateGenerator

# Email generator
email_gen = TemplateGenerator(
    template="""
    Subject: {subject}
    
    Dear {recipient},
    
    {body}
    
    Best regards,
    {sender}
    """,
    model="gpt2-large"
)

# Generate email
email = email_gen.generate(
    subject="Project Update",
    recipient="Team",
    body_prompt="Write about successful project completion",
    sender="Project Manager"
)
```

### 3. Test Data Creation

Generate test data for applications:

```python
from textnlp import TestDataGenerator

# Create test data generator
test_gen = TestDataGenerator()

# Generate customer reviews
reviews = test_gen.generate_reviews(
    product="Wireless Headphones",
    count=100,
    rating_distribution={
        5: 0.4,  # 40% 5-star
        4: 0.3,  # 30% 4-star
        3: 0.2,  # 20% 3-star
        2: 0.07, # 7% 2-star
        1: 0.03  # 3% 1-star
    }
)

# Generate support tickets
tickets = test_gen.generate_support_tickets(
    product="SaaS Platform",
    categories=["bug", "feature_request", "billing"],
    count=50
)
```

### 4. Multilingual Generation

Generate text in multiple languages:

```python
from textnlp import MultilingualGenerator

# Initialize multilingual generator
ml_gen = MultilingualGenerator(model="llama-7b")

# Generate in different languages
texts = ml_gen.generate_batch([
    {"prompt": "Write about AI", "language": "en"},
    {"prompt": "Écrire sur l'IA", "language": "fr"},
    {"prompt": "Escribir sobre IA", "language": "es"},
    {"prompt": "写关于人工智能", "language": "zh"}
])
```

## Best Practices

### 1. Model Selection

Choose the right model for your needs:

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| Quick prototypes | GPT-2 | Fast, lightweight |
| General content | GPT-2-Large | Good balance |
| High quality | GPT-J-6B | Best quality |
| Multilingual | LLaMA-7B | Language support |

### 2. Prompt Engineering

Write effective prompts:

```python
# ❌ Poor prompt
"Write something about cars"

# ✅ Good prompt
"Write a detailed review of a 2023 electric vehicle, focusing on:
- Battery life and charging time
- Driving experience
- Technology features
- Value for money
The review should be 200-300 words and written from a consumer perspective."
```

### 3. Parameter Tuning

Optimize generation parameters:

```python
# Conservative (factual content)
result = generator.generate(
    prompt=prompt,
    temperature=0.3,    # Low randomness
    top_p=0.9,         # Focused vocabulary
    top_k=40           # Limited choices
)

# Creative (artistic content)
result = generator.generate(
    prompt=prompt,
    temperature=0.9,    # High randomness
    top_p=0.95,        # Broader vocabulary
    top_k=100          # More choices
)

# Balanced (general use)
result = generator.generate(
    prompt=prompt,
    temperature=0.7,    # Moderate randomness
    top_p=0.9,         # Standard vocabulary
    top_k=50           # Balanced choices
)
```

### 4. Quality Validation

Always validate generated content:

```python
from textnlp import TextValidator

validator = TextValidator()

# Validate generated text
validation = validator.validate(
    generated_text,
    reference_texts=reference_samples,
    checks=['quality', 'relevance', 'grammar', 'toxicity']
)

if validation['quality_score'] < 0.7:
    # Regenerate with different parameters
    pass
```

### 5. Batch Processing

For large-scale generation:

```python
from textnlp import BatchGenerator

batch_gen = BatchGenerator(
    model="gpt2-large",
    batch_size=32,
    num_workers=4
)

# Generate 1000 samples
prompts = ["Generate review for product " + str(i) for i in range(1000)]
results = batch_gen.generate_batch(
    prompts,
    show_progress=True,
    save_intermediate=True
)
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory Error

```python
# Solution: Use smaller model or reduce batch size
generator = TextGenerator(
    model="gpt2",  # Use smaller model
    device="cpu",  # Use CPU if GPU OOM
    max_batch_size=1  # Process one at a time
)
```

#### 2. Slow Generation

```python
# Solution: Enable GPU and optimize settings
generator = TextGenerator(
    model="gpt2-large",
    device="cuda",  # Use GPU
    use_cache=True,  # Enable model caching
    compile_model=True  # Compile for speed
)
```

#### 3. Poor Quality Output

```python
# Solution: Improve prompts and adjust parameters
result = generator.generate(
    prompt=improved_prompt,
    temperature=0.7,  # Adjust temperature
    repetition_penalty=1.2,  # Reduce repetition
    min_length=50,  # Ensure minimum length
    num_return_sequences=3  # Generate multiple, pick best
)
```

#### 4. API Rate Limits

```python
# Solution: Implement retry logic
import time
from tenacity import retry, wait_exponential

@retry(wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_with_retry(prompt):
    return api_client.generate(prompt)
```

## Examples

### Example 1: Blog Post Generator

```python
from textnlp import BlogGenerator

# Create blog generator
blog_gen = BlogGenerator(model="gpt2-xl")

# Generate blog post
post = blog_gen.create_post(
    topic="The Future of AI in Healthcare",
    sections=["Introduction", "Current Applications", 
              "Challenges", "Future Outlook", "Conclusion"],
    words_per_section=200,
    tone="professional",
    include_examples=True
)

# Save as markdown
with open("ai_healthcare_blog.md", "w") as f:
    f.write(post.to_markdown())
```

### Example 2: Chatbot Training Data

```python
from textnlp import ConversationGenerator

# Create conversation generator
conv_gen = ConversationGenerator()

# Generate customer service conversations
conversations = conv_gen.generate_dialogues(
    scenario="customer_support",
    topics=["order_tracking", "returns", "technical_support"],
    num_conversations=100,
    turns_per_conversation=(3, 8),  # 3-8 turns
    include_emotions=True
)

# Export for training
conv_gen.export_to_jsonl(conversations, "chatbot_training.jsonl")
```

### Example 3: Product Description Generator

```python
from textnlp import ProductDescriptionGenerator

# Initialize generator
desc_gen = ProductDescriptionGenerator(
    model="gpt2-large",
    style="e-commerce"
)

# Product details
product = {
    "name": "Smart Fitness Tracker Pro",
    "category": "Wearables",
    "features": [
        "Heart rate monitoring",
        "Sleep tracking",
        "Water resistant",
        "7-day battery"
    ],
    "target_audience": "Fitness enthusiasts",
    "price_range": "Premium"
}

# Generate descriptions
descriptions = desc_gen.generate_variations(
    product,
    num_variations=5,
    lengths=["short", "medium", "long"],
    include_seo_keywords=True
)
```

### Example 4: Synthetic Survey Responses

```python
from textnlp import SurveyResponseGenerator

# Create survey response generator
survey_gen = SurveyResponseGenerator()

# Generate responses
responses = survey_gen.generate_responses(
    questions=[
        "What features would you like in our next product?",
        "How would you rate your experience?",
        "Any additional comments?"
    ],
    respondent_profiles=[
        {"type": "satisfied_customer", "count": 60},
        {"type": "neutral_customer", "count": 30},
        {"type": "dissatisfied_customer", "count": 10}
    ],
    response_style="natural",
    include_demographics=True
)

# Analyze synthetic responses
analysis = survey_gen.analyze_responses(responses)
print(analysis.sentiment_distribution)
print(analysis.common_themes)
```

## Next Steps

1. **Explore Advanced Features**: Check out template generation, fine-tuning, and custom models
2. **Join the Community**: Visit our [Discord](https://discord.gg/inferloop) for support
3. **Contribute**: We welcome contributions! See [CONTRIBUTING.md](../CONTRIBUTING.md)
4. **Enterprise**: Contact us for enterprise features and support

## Resources

- [API Documentation](./API_DOCUMENTATION.md)
- [SDK Reference](./SDK_DOCUMENTATION.md)
- [CLI Reference](./CLI_DOCUMENTATION.md)
- [Examples Repository](https://github.com/inferloop/textnlp-examples)
- [Video Tutorials](https://youtube.com/inferloop)

## Support

- **Documentation**: [docs.inferloop.com/textnlp](https://docs.inferloop.com/textnlp)
- **GitHub Issues**: [github.com/inferloop/textnlp/issues](https://github.com/inferloop/textnlp/issues)
- **Email**: support@inferloop.com
- **Discord**: [discord.gg/inferloop](https://discord.gg/inferloop)

Happy generating! 🚀