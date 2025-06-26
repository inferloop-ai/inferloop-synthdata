# Tabular Documentation

Welcome to the Tabular documentation! This comprehensive guide covers everything you need to know about generating synthetic tabular data using our platform.

## 📚 Documentation Overview

### [How Synthetic Data is Generated](./HOW_SYNTHETIC_DATA_IS_GENERATED.md)
Understand the technology and processes behind our data generation:
- Core generation pipeline and architecture
- Available algorithms (SDV, CTGAN, YData)
- Data processing and preprocessing techniques
- Quality assurance and privacy protection mechanisms

### [User Guide](./USER_GUIDE.md)
Get started with Tabular and explore common use cases:
- Quick start and installation
- Basic and advanced usage examples
- Best practices for different data types
- Common use cases (development, ML augmentation, privacy)
- Troubleshooting and tips

### [SDK Documentation](./SDK_DOCUMENTATION.md)
Deep dive into the Python SDK:
- Core classes and generators
- Algorithm-specific implementations
- Validation and quality metrics
- Privacy features and anonymization
- Advanced features (batch processing, streaming, caching)
- Complete API reference

### [CLI Documentation](./CLI_DOCUMENTATION.md)
Master the command-line interface:
- All available commands and options
- Configuration management
- Batch processing and automation
- Integration with data pipelines
- Performance optimization

### [API Documentation](./API_DOCUMENTATION.md)
Build applications with our REST API:
- Authentication and rate limiting
- Complete endpoint reference
- Request/response formats
- Asynchronous generation
- Client examples in multiple languages

## 🚀 Quick Links

### Getting Started
1. **Install Tabular**: `pip install inferloop-tabular`
2. **Generate your first data**: `inferloop-tabular generate data.csv --rows 1000`
3. **Start the API**: `inferloop-tabular serve`

### Common Tasks

**Generate Data (CLI)**
```bash
inferloop-tabular generate customers.csv --algorithm ctgan --rows 5000
```

**Generate Data (SDK)**
```python
from tabular import TabularGenerator
generator = TabularGenerator(model='ctgan')
generator.fit('customers.csv')
synthetic = generator.generate(num_rows=5000)
```

**Generate Data (API)**
```bash
curl -X POST http://localhost:8000/api/tabular/generate \
  -F "file=@customers.csv" \
  -F "algorithm=ctgan" \
  -F "num_rows=5000"
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
1. Understand the algorithms in [How Synthetic Data is Generated](./HOW_SYNTHETIC_DATA_IS_GENERATED.md)
2. Learn validation techniques in SDK Documentation
3. Explore batch processing and optimization in CLI Documentation

## 🛠️ Key Features

### Multiple Algorithms
- **SDV**: Fast, general-purpose synthesis with multiple models
- **CTGAN**: High-quality generation using GANs
- **YData**: Enterprise features with advanced privacy

### Data Types Support
- **Numerical**: Integer, float with distributions
- **Categorical**: Discrete categories with cardinality
- **Datetime**: Temporal data with patterns
- **Text**: Free-form text (limited support)
- **Geospatial**: Coordinates and locations

### Privacy Features
- **Differential Privacy**: Mathematical privacy guarantees
- **K-Anonymity**: Indistinguishability protection
- **Synthetic Guarantees**: No 1:1 mapping to real records
- **Privacy Metrics**: Comprehensive risk assessment

### Quality Assurance
- **Statistical Validation**: KS tests, correlations, distributions
- **ML Efficacy**: Train/test performance comparison
- **Visual Validation**: Plots and comparisons
- **Constraint Validation**: Business rule checking

## 📊 Performance Guidelines

### Algorithm Selection
| Use Case | Recommended Algorithm | Speed | Quality | Privacy |
|----------|---------------------|-------|---------|---------|
| Quick prototyping | SDV | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐ |
| Production data | CTGAN | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| High privacy | YData + DP | ⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Large datasets | SDV (sampling) | ⚡⚡⚡⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

### Optimization Tips
1. **Use GPU**: 10-100x speedup for neural models
2. **Batch Processing**: Handle large datasets efficiently
3. **Model Caching**: Avoid retraining for similar data
4. **Appropriate Parameters**: Balance quality vs speed

## 🔧 Configuration

### Environment Variables
```bash
export INFERLOOP_API_KEY="your-api-key"
export INFERLOOP_DEFAULT_ALGORITHM="ctgan"
export INFERLOOP_GPU_ENABLED="true"
```

### Configuration File
Create `~/.inferloop/tabular/config.yaml`:
```yaml
default:
  algorithm: ctgan
  output_format: csv
  gpu: auto
  seed: 42

api:
  endpoint: https://api.inferloop.com
  timeout: 300
```

## 🆘 Getting Help

### Resources
- **Examples**: See example code in each documentation file
- **Notebooks**: Check our [Jupyter notebooks](../examples/)
- **Videos**: Watch tutorials on [YouTube](https://youtube.com/inferloop)

### Support Channels
- **GitHub Issues**: [github.com/inferloop/tabular/issues](https://github.com/inferloop/tabular/issues)
- **Discord Community**: [discord.gg/inferloop](https://discord.gg/inferloop)
- **Email Support**: support@inferloop.com
- **Enterprise Support**: enterprise@inferloop.com

### Common Issues
1. **Memory Problems**: Use batch processing or sampling
2. **Slow Generation**: Enable GPU or use simpler algorithm
3. **Poor Quality**: Increase training epochs or try different algorithm
4. **Privacy Concerns**: Use YData with differential privacy

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

Tabular is licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.

---

**Happy Generating!** 🚀

If you find Tabular useful, please ⭐ star our [GitHub repository](https://github.com/inferloop/tabular)!