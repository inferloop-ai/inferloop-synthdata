c# README.md
# Inferloop Synthetic Data SDK

🚀 **A unified wrapper for synthetic data generation tools** - SDV, CTGAN, YData-Synthetic, and more!

## Overview

The Inferloop Synthetic Data SDK provides a consistent, easy-to-use interface across multiple synthetic data generation libraries. Whether you're using SDV's Gaussian Copula, CTGAN, or YData's WGAN-GP, our SDK abstracts away the complexity and provides unified APIs for generation, validation, and evaluation.

## ✨ Key Features

- 🎯 **Unified Interface**: Single API for multiple synthetic data libraries
- 🔧 **Flexible Configuration**: YAML/JSON-based configuration system
- 📊 **Comprehensive Validation**: Built-in quality assessment and privacy metrics
- 🖥️ **Multiple Interfaces**: SDK, CLI, and REST API
- 📈 **Rich Evaluation**: Statistical tests, correlation analysis, and utility metrics
- 🐳 **Docker Support**: Containerized deployment ready
- 📚 **Extensive Examples**: Jupyter notebooks and CLI examples

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install inferloop-synthetic

# With all supported libraries
pip install inferloop-synthetic[all]

# Specific libraries
pip install inferloop-synthetic[sdv,ctgan]
```

### SDK Usage

```python
from inferloop_synthetic.sdk import GeneratorFactory, SyntheticDataConfig
import pandas as pd

# Load your data
data = pd.read_csv("your_data.csv")

# Configure generation
config = SyntheticDataConfig(
    generator_type="sdv",
    model_type="gaussian_copula",
    num_samples=1000,
    categorical_columns=["category", "region"],
    continuous_columns=["age", "income"]
)

# Generate synthetic data
generator = GeneratorFactory.create_generator(config)
result = generator.fit_generate(data)

print(f"Generated {len(result.synthetic_data)} synthetic samples!")
```

### CLI Usage

```bash
# Generate synthetic data
inferloop-synthetic generate data.csv synthetic_output.csv \
  --generator-type sdv \
  --model-type gaussian_copula \
  --num-samples 1000

# Validate synthetic data
inferloop-synthetic validate original.csv synthetic.csv

# Get information about available generators
inferloop-synthetic info
```

### REST API Usage

```bash
# Start the API server
uvicorn inferloop_synthetic.api.app:app --host 0.0.0.0 --port 8000

# Upload data and generate (using curl)
curl -X POST "http://localhost:8000/data/upload" \
  -F "file=@your_data.csv"

curl -X POST "http://localhost:8000/generate/sync" \
  -H "Content-Type: application/json" \
  -d '{"config": {"generator_type": "sdv", "model_type": "gaussian_copula"}}'

  ## 🏗️ Architecture

```
📦 Inferloop Synthetic SDK
├── 🎛️ Base Generator Interface
├── 🔌 Library Wrappers (SDV, CTGAN, YData)
├── 📊 Validation Framework  
├── 🖥️ CLI Interface
├── 🌐 REST API
└── 📚 Examples & Documentation
```

## 📊 Supported Libraries

| Library | Models | Description |
|---------|--------|-------------|
| **SDV** | Gaussian Copula, CTGAN, CopulaGAN, TVAE | Comprehensive tabular synthesis |
| **CTGAN** | CTGAN, TVAE | Conditional Tabular GAN |
| **YData** | WGAN-GP, CramerGAN, DRAGAN | Advanced GAN-based synthesis |

## 🔍 Validation Metrics

Our comprehensive validation framework includes:

- **Statistical Tests**: KS tests, Chi-square tests
- **Distribution Analysis**: Mean, std, min, max comparisons  
- **Correlation Preservation**: Correlation matrix analysis
- **Privacy Metrics**: Distance-based privacy assessment
- **Utility Metrics**: ML model performance comparison
- **SDMetrics Integration**: Industry-standard quality metrics

## 🛠️ Development

```bash
# Clone repository
git clone https://github.com/inferloop/inferloop-synthetic.git
cd inferloop-synthetic

# Install in development mode
pip install -e ".[dev,all]"

# Run tests
pytest

# Run examples
python examples/notebook.py
```

## 📈 Performance Comparison

Based on our benchmarks across multiple datasets:

| Model | Generation Speed | Quality Score | Privacy Score |
|-------|-----------------|---------------|---------------|
| SDV Gaussian Copula | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| CTGAN | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| YData WGAN-GP | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of excellent libraries: SDV, CTGAN, YData-Synthetic
- Inspired by the need for unified synthetic data interfaces
- Thanks to the open-source community!

---

**Ready to generate high-quality synthetic data? Get started with Inferloop Synthetic SDK!** 🚀

