# TextNLP Documentation

Welcome to the TextNLP documentation! This comprehensive guide covers everything you need to know about generating synthetic text and NLP data using our platform.

## 📚 Documentation Overview

### [How Synthetic Data is Generated](./HOW_SYNTHETIC_DATA_IS_GENERATED.md)
Understand the technology and processes behind our text generation:
- Core generation pipeline and architecture
- Available language models (GPT-2, GPT-J, LLaMA)
- Generation techniques (sampling, beam search, constraints)
- Quality control and validation mechanisms

### [User Guide](./USER_GUIDE.md)
Get started with TextNLP and explore common use cases:
- Quick start and installation
- Basic and advanced usage examples
- Best practices for prompt engineering
- Common use cases (content generation, data augmentation, testing)
- Troubleshooting and tips

### [SDK Documentation](./SDK_DOCUMENTATION.md)
Deep dive into the Python SDK:
- Core classes and generators
- Validation and quality metrics
- Template management
- Advanced features (fine-tuning, caching, batch processing)
- Complete API reference

### [CLI Documentation](./CLI_DOCUMENTATION.md)
Master the command-line interface:
- All available commands and options
- Configuration management
- Batch processing and automation
- Integration with other tools
- Performance optimization

### [API Documentation](./API_DOCUMENTATION.md)
Build applications with our REST API:
- Authentication and rate limiting
- Complete endpoint reference
- Request/response formats
- WebSocket streaming
- Client examples in multiple languages

## 🚀 Quick Links

### Getting Started
1. **Install TextNLP**: `pip install inferloop-textnlp`
2. **Generate your first text**: `inferloop-nlp generate "Write a story about AI"`
3. **Start the API**: `inferloop-nlp serve`

### Common Tasks

**Generate Text (CLI)**
```bash
inferloop-nlp generate "Your prompt here" --model gpt2-large
```

**Generate Text (SDK)**
```python
from textnlp import TextGenerator
generator = TextGenerator()
result = generator.generate("Your prompt here")
```

**Generate Text (API)**
```bash
curl -X POST http://localhost:8000/api/textnlp/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Your prompt here"}'
```

## 📖 Learning Path

### For Beginners
1. Start with the [User Guide](./USER_GUIDE.md) - Quick Start section
2. Try basic CLI commands from [CLI Documentation](./CLI_DOCUMENTATION.md)
3. Explore example use cases in the User Guide

### For Developers
1. Review [How Synthetic Data is Generated](./HOW_SYNTHETIC_DATA_IS_GENERATED.md)
2. Deep dive into [SDK Documentation](./SDK_DOCUMENTATION.md)
3. Build applications using [API Documentation](./API_DOCUMENTATION.md)

### For Data Scientists
1. Understand the models in [How Synthetic Data is Generated](./HOW_SYNTHETIC_DATA_IS_GENERATED.md)
2. Learn validation techniques in SDK Documentation
3. Explore batch processing in CLI Documentation

## 🛠️ Key Features

### Multiple Interfaces
- **CLI**: Command-line for quick generation and automation
- **SDK**: Python library for programmatic access
- **API**: REST endpoints for any programming language

### Model Support
- **GPT-2 Family**: Fast, efficient generation (124M to 1.5B parameters)
- **GPT-J-6B**: High-quality generation for complex tasks
- **LLaMA**: Multi-lingual support
- **Commercial Models**: GPT-4, Claude (Enterprise tier)

### Generation Capabilities
- **Text Generation**: Articles, stories, descriptions
- **Template-based**: Structured content with variables
- **Streaming**: Real-time token generation
- **Batch Processing**: Efficient multi-prompt handling

### Quality Assurance
- **BLEU/ROUGE**: Automatic quality metrics
- **Grammar Checking**: Built-in validation
- **Human Evaluation**: Interface for manual review
- **Toxicity Filtering**: Safe content generation

## 📊 Performance Guidelines

### Model Selection
| Use Case | Recommended Model | Speed | Quality |
|----------|------------------|-------|---------|
| Prototyping | GPT-2 | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ |
| Production | GPT-2-Large | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| High Quality | GPT-J-6B | ⚡⚡ | ⭐⭐⭐⭐⭐ |

### Optimization Tips
1. **Use GPU**: 10-100x speedup for large models
2. **Enable Caching**: Avoid regenerating common prompts
3. **Batch Requests**: Process multiple prompts together
4. **Stream Large Outputs**: Better user experience

## 🔧 Configuration

### Environment Variables
```bash
export INFERLOOP_API_KEY="your-api-key"
export INFERLOOP_DEFAULT_MODEL="gpt2-large"
export INFERLOOP_CACHE_DIR="~/.inferloop/cache"
```

### Configuration File
Create `~/.inferloop/textnlp/config.yaml`:
```yaml
default:
  model: gpt2-large
  temperature: 0.7
  max_tokens: 200

api:
  endpoint: https://api.inferloop.com
  timeout: 30
```

## 🆘 Getting Help

### Resources
- **Examples**: See example code in each documentation file
- **Notebooks**: Check our [Jupyter notebooks](../examples/)
- **Videos**: Watch tutorials on [YouTube](https://youtube.com/inferloop)

### Support Channels
- **GitHub Issues**: [github.com/inferloop/textnlp/issues](https://github.com/inferloop/textnlp/issues)
- **Discord Community**: [discord.gg/inferloop](https://discord.gg/inferloop)
- **Email Support**: support@inferloop.com
- **Enterprise Support**: enterprise@inferloop.com

### Common Issues
1. **Installation Problems**: Check Python version (3.8+) and dependencies
2. **GPU Not Detected**: Verify CUDA installation
3. **Slow Generation**: Use smaller models or enable GPU
4. **API Connection**: Check firewall and network settings

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](../CONTRIBUTING.md) for:
- Code style guidelines
- Development setup
- Testing requirements
- Pull request process

## 📈 Stay Updated

- **Blog**: [blog.inferloop.com](https://blog.inferloop.com)
- **Twitter**: [@inferloop](https://twitter.com/inferloop)
- **Newsletter**: Subscribe at [inferloop.com/newsletter](https://inferloop.com/newsletter)
- **Changelog**: See [CHANGELOG.md](../CHANGELOG.md)

## 📄 License

TextNLP is licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.

---

**Happy Generating!** 🚀

If you find TextNLP useful, please ⭐ star our [GitHub repository](https://github.com/inferloop/textnlp)!