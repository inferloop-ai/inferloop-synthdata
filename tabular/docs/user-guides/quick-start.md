# Quick Start Guide

Get up and running with the Inferloop Synthetic Data SDK in 10 minutes! This guide will walk you through installation, basic usage, and your first synthetic dataset generation.

## Prerequisites

- Python 3.8 or higher
- 4GB RAM (minimum)
- Internet connection for package installation

## Step 1: Installation

### Quick Installation
```bash
pip install inferloop-synthetic[all]
```

### Verify Installation
```bash
# Check if installation was successful
python -c "import inferloop_synthetic; print('✅ Installation successful!')"

# Check CLI is available
inferloop-synthetic --help
```

## Step 2: Prepare Your Data

For this guide, we'll use a sample dataset. You can use your own CSV file or create a sample one:

```python
# Create a sample dataset (save as sample_data.csv)
import pandas as pd

data = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 50],
    'income': [50000, 60000, 70000, 80000, 90000, 100000],
    'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia'],
    'education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'Bachelor']
})

data.to_csv('sample_data.csv', index=False)
print("✅ Sample data created: sample_data.csv")
```

## Step 3: Generate Your First Synthetic Dataset

### Using the SDK (Python)

```python
from inferloop_synthetic.sdk import GeneratorFactory, SyntheticDataConfig
import pandas as pd

# Load your data
data = pd.read_csv('sample_data.csv')
print(f"📊 Loaded dataset with {len(data)} rows and {len(data.columns)} columns")

# Configure the generator
config = SyntheticDataConfig(
    generator_type="sdv",           # Use SDV library
    model_type="gaussian_copula",   # Use Gaussian Copula model
    num_samples=100,                # Generate 100 synthetic rows
    categorical_columns=["city", "education"],  # Specify categorical columns
    continuous_columns=["age", "income"]        # Specify continuous columns
)

# Create and use the generator
generator = GeneratorFactory.create_generator(config)
result = generator.fit_generate(data)

# Save synthetic data
result.synthetic_data.to_csv('synthetic_data.csv', index=False)
print(f"🎉 Generated {len(result.synthetic_data)} synthetic rows!")
print(f"💾 Saved to: synthetic_data.csv")

# View quality metrics
print("\n📈 Quality Metrics:")
for metric, value in result.quality_metrics.items():
    print(f"  {metric}: {value:.3f}")
```

### Using the CLI

```bash
# Generate synthetic data using command line
inferloop-synthetic generate sample_data.csv synthetic_output.csv \
  --generator-type sdv \
  --model-type gaussian_copula \
  --num-samples 100 \
  --categorical-columns city,education \
  --continuous-columns age,income

echo "🎉 Synthetic data generated successfully!"
```

### Using the REST API

First, start the API server:

```bash
# Start the API server
uvicorn inferloop_synthetic.api.app:app --host 0.0.0.0 --port 8000 &
echo "🚀 API server started at http://localhost:8000"
```

Then generate synthetic data:

```bash
# Upload your data
curl -X POST "http://localhost:8000/data/upload" \
  -F "file=@sample_data.csv" \
  -H "accept: application/json"

# Generate synthetic data
curl -X POST "http://localhost:8000/generate/sync" \
  -H "Content-Type: application/json" \
  -H "accept: application/json" \
  -d '{
    "config": {
      "generator_type": "sdv",
      "model_type": "gaussian_copula",
      "num_samples": 100,
      "categorical_columns": ["city", "education"],
      "continuous_columns": ["age", "income"]
    }
  }'

echo "🎉 API generation complete!"
```

## Step 4: Validate Your Synthetic Data

### Using the SDK

```python
from inferloop_synthetic.sdk.validator import SyntheticDataValidator

# Load original and synthetic data
original_data = pd.read_csv('sample_data.csv')
synthetic_data = pd.read_csv('synthetic_data.csv')

# Create validator and run validation
validator = SyntheticDataValidator()
validation_results = validator.validate(original_data, synthetic_data)

print("🔍 Validation Results:")
for test_name, result in validation_results.items():
    status = "✅ PASS" if result.get('passed', False) else "❌ FAIL"
    print(f"  {test_name}: {status}")
```

### Using the CLI

```bash
# Validate synthetic data quality
inferloop-synthetic validate sample_data.csv synthetic_data.csv

echo "🔍 Validation complete!"
```

## Step 5: Explore Different Models

Try different generators and models to see which works best for your data:

```python
# Try CTGAN model
config_ctgan = SyntheticDataConfig(
    generator_type="ctgan",
    model_type="ctgan",
    num_samples=100,
    epochs=10  # Use fewer epochs for quick testing
)

generator_ctgan = GeneratorFactory.create_generator(config_ctgan)
result_ctgan = generator_ctgan.fit_generate(data)

print(f"🧠 CTGAN generated {len(result_ctgan.synthetic_data)} rows")

# Try YData WGAN-GP model
config_ydata = SyntheticDataConfig(
    generator_type="ydata",
    model_type="wgan_gp",
    num_samples=100,
    epochs=10  # Use fewer epochs for quick testing
)

generator_ydata = GeneratorFactory.create_generator(config_ydata)
result_ydata = generator_ydata.fit_generate(data)

print(f"🎯 YData generated {len(result_ydata.synthetic_data)} rows")
```

## Step 6: Compare Results

```python
import matplotlib.pyplot as plt

# Compare distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Age distribution
axes[0, 0].hist(data['age'], alpha=0.7, label='Original', bins=10)
axes[0, 0].hist(result.synthetic_data['age'], alpha=0.7, label='Synthetic', bins=10)
axes[0, 0].set_title('Age Distribution')
axes[0, 0].legend()

# Income distribution
axes[0, 1].hist(data['income'], alpha=0.7, label='Original', bins=10)
axes[0, 1].hist(result.synthetic_data['income'], alpha=0.7, label='Synthetic', bins=10)
axes[0, 1].set_title('Income Distribution')
axes[0, 1].legend()

# City distribution
city_counts_orig = data['city'].value_counts()
city_counts_synth = result.synthetic_data['city'].value_counts()
axes[1, 0].bar(range(len(city_counts_orig)), city_counts_orig.values, alpha=0.7, label='Original')
axes[1, 0].bar(range(len(city_counts_synth)), city_counts_synth.values, alpha=0.7, label='Synthetic')
axes[1, 0].set_title('City Distribution')
axes[1, 0].legend()

# Education distribution
edu_counts_orig = data['education'].value_counts()
edu_counts_synth = result.synthetic_data['education'].value_counts()
axes[1, 1].bar(range(len(edu_counts_orig)), edu_counts_orig.values, alpha=0.7, label='Original')
axes[1, 1].bar(range(len(edu_counts_synth)), edu_counts_synth.values, alpha=0.7, label='Synthetic')
axes[1, 1].set_title('Education Distribution')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('comparison.png')
print("📊 Comparison chart saved as comparison.png")
```

## Common Use Cases

### 1. Data Privacy Protection
```python
# Generate privacy-preserving synthetic data
config = SyntheticDataConfig(
    generator_type="sdv",
    model_type="gaussian_copula",
    num_samples=1000,
    privacy_level="high",  # Add privacy protection
    differential_privacy=True
)
```

### 2. Data Augmentation
```python
# Augment small datasets
config = SyntheticDataConfig(
    generator_type="ctgan",
    model_type="ctgan",
    num_samples=len(data) * 3,  # Triple the dataset size
    augmentation_mode=True
)
```

### 3. Testing Data Generation
```python
# Generate test data with specific constraints
config = SyntheticDataConfig(
    generator_type="sdv",
    model_type="gaussian_copula",
    num_samples=500,
    constraints=[
        {"column": "age", "min": 18, "max": 65},
        {"column": "income", "min": 30000, "max": 200000}
    ]
)
```

## Next Steps

Congratulations! You've successfully generated your first synthetic dataset. Here's what to explore next:

### 📚 Learn More
- [SDK Usage Guide](sdk-usage.md) - Detailed SDK documentation
- [Configuration Guide](configuration.md) - All configuration options
- [Validation Framework](validation.md) - Understanding quality metrics

### 🔧 Advanced Features
- [Batch Processing](batch-processing.md) - Process large datasets
- [Privacy Features](privacy.md) - Advanced privacy protection
- [Streaming Data](streaming.md) - Real-time generation

### 🚀 Production Deployment
- [Production Deployment](../deployment/production-deployment.md) - Deploy to production
- [API Reference](api-usage.md) - Complete API documentation
- [Performance Tuning](performance-tuning.md) - Optimize for your use case

## Troubleshooting

### Common Issues

#### Installation Problems
```bash
# If you get permission errors
pip install --user inferloop-synthetic[all]

# If you get dependency conflicts
pip install --upgrade pip
pip install inferloop-synthetic[all]
```

#### Memory Issues
```python
# For large datasets, use batch processing
config = SyntheticDataConfig(
    generator_type="sdv",
    model_type="gaussian_copula",
    batch_size=1000,  # Process in smaller batches
    num_samples=10000
)
```

#### Quality Issues
```python
# Improve quality with more epochs (for neural networks)
config = SyntheticDataConfig(
    generator_type="ctgan",
    model_type="ctgan",
    epochs=300,  # Default is 100
    batch_size=500
)
```

### Getting Help

- 📖 Check our [FAQ](faq.md)
- 🐛 Report issues on [GitHub](https://github.com/inferloop/inferloop-synthetic/issues)
- 💬 Join our [Community Discord](https://discord.gg/inferloop)
- 📧 Email support: support@inferloop.com

## Summary

You've learned how to:
- ✅ Install the Inferloop Synthetic Data SDK
- ✅ Generate synthetic data using SDK, CLI, and API
- ✅ Validate synthetic data quality
- ✅ Compare different models and approaches
- ✅ Visualize and analyze results

The Inferloop Synthetic Data SDK makes it easy to generate high-quality synthetic data for any use case. Start with simple datasets and gradually explore more advanced features as your needs grow.

**Ready for more? Check out our [comprehensive tutorials](../tutorials/) for specific use cases and advanced techniques!** 🚀