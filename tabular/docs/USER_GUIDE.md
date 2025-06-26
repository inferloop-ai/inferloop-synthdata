# Tabular Synthetic Data User Guide

## Welcome to Tabular Synthetic Data Generation

Tabular is a powerful platform for generating high-quality synthetic tabular data using state-of-the-art machine learning algorithms. This guide will help you get started and make the most of the platform.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Common Use Cases](#common-use-cases)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)
7. [Examples](#examples)

## Getting Started

### What is Tabular?

Tabular is a comprehensive synthetic data generation platform that helps you:
- Generate realistic synthetic data that preserves statistical properties
- Create privacy-preserving datasets for development and testing
- Augment existing datasets for machine learning
- Share data without exposing sensitive information

### Key Features

- 🤖 **Multiple Algorithms**: SDV, CTGAN, and YData synthesizers
- 🔒 **Privacy First**: Built-in privacy protection and validation
- 📊 **Quality Metrics**: Comprehensive statistical validation
- 🚀 **Scalable**: Handle datasets from KBs to GBs
- 🎯 **Flexible**: Support for various data types and constraints
- 🌐 **Enterprise Ready**: Authentication, monitoring, and cloud deployment

## Installation

### Requirements

- Python 3.8 or higher
- 8GB RAM minimum (16GB+ recommended for large datasets)
- GPU optional but recommended for neural models

### Install via pip

```bash
# Basic installation
pip install inferloop-tabular

# With all dependencies
pip install inferloop-tabular[all]

# For development
pip install inferloop-tabular[dev]
```

### Install from source

```bash
git clone https://github.com/inferloop/tabular.git
cd tabular
pip install -e ".[all]"
```

### Verify Installation

```bash
# Check CLI
inferloop-tabular --version

# Test generation
inferloop-tabular generate examples/sample.csv --rows 100
```

## Quick Start

### 1. Using the CLI

Generate synthetic data with simple commands:

```bash
# Basic generation from CSV
inferloop-tabular generate data.csv --output synthetic.csv --rows 1000

# Specify algorithm and parameters
inferloop-tabular generate data.csv \
  --algorithm ctgan \
  --epochs 300 \
  --batch-size 500 \
  --output synthetic.csv

# Generate with privacy constraints
inferloop-tabular generate sensitive_data.csv \
  --privacy differential \
  --epsilon 1.0 \
  --output private_synthetic.csv
```

### 2. Using the SDK

```python
from tabular import TabularGenerator

# Initialize generator
generator = TabularGenerator(model='sdv')

# Load and generate data
generator.fit('sales_data.csv')
synthetic_data = generator.generate(num_rows=1000)

# Save results
synthetic_data.to_csv('synthetic_sales.csv', index=False)

# Check quality
metrics = generator.evaluate(synthetic_data)
print(f"Quality score: {metrics['overall_score']:.2f}")
```

### 3. Using the REST API

Start the API server:

```bash
# Start server
inferloop-tabular serve --port 8000

# Or with Docker
docker run -p 8000:8000 inferloop/tabular
```

Make API calls:

```python
import requests
import pandas as pd

# Upload and generate
with open('data.csv', 'rb') as f:
    response = requests.post(
        "http://localhost:8000/api/tabular/generate",
        files={'file': f},
        data={
            'algorithm': 'ctgan',
            'num_rows': 1000
        },
        headers={'X-API-Key': 'your-api-key'}
    )

# Download results
result = response.json()
synthetic_df = pd.DataFrame(result['data'])
```

## Common Use Cases

### 1. Development and Testing

Create realistic test data without using production data:

```python
from tabular import TabularGenerator, DataProfiler

# Profile production data structure
profiler = DataProfiler()
schema = profiler.extract_schema('production_orders.csv')

# Generate test data matching schema
generator = TabularGenerator(model='sdv')
test_data = generator.generate_from_schema(
    schema=schema,
    num_rows=10000,
    include_edge_cases=True
)

# Add specific test scenarios
test_data = generator.add_scenarios(test_data, [
    {'order_status': 'cancelled', 'count': 100},
    {'payment_failed': True, 'count': 50},
    {'order_amount': '>10000', 'count': 20}
])

test_data.to_csv('test_orders.csv', index=False)
```

### 2. Machine Learning Data Augmentation

Augment training data to improve model performance:

```python
from tabular import DataAugmenter

# Load original training data
original_data = pd.read_csv('training_data.csv')

# Augment with synthetic samples
augmenter = DataAugmenter(
    model='ctgan',
    preserve_distribution=True
)

# Generate synthetic samples for minority classes
augmented_data = augmenter.balance_dataset(
    original_data,
    target_column='fraud',
    strategy='oversample_minority',
    synthetic_ratio=0.5
)

# Verify class balance
print(augmented_data['fraud'].value_counts())
```

### 3. Privacy-Preserving Data Sharing

Share data with partners while protecting privacy:

```python
from tabular import PrivacyPreservingGenerator

# Initialize with privacy guarantees
generator = PrivacyPreservingGenerator(
    model='ydata',
    privacy_level='high',
    epsilon=0.5
)

# Load sensitive data
sensitive_data = pd.read_csv('customer_data.csv')

# Generate privacy-safe synthetic version
safe_data = generator.generate(
    sensitive_data,
    num_rows=len(sensitive_data),
    remove_outliers=True,
    generalize_rare_categories=True
)

# Validate privacy
privacy_report = generator.validate_privacy(
    original=sensitive_data,
    synthetic=safe_data
)

print(f"Privacy score: {privacy_report['privacy_score']:.2f}")
print(f"Utility score: {privacy_report['utility_score']:.2f}")

# Export with metadata
generator.export_with_metadata(
    safe_data,
    'shareable_data.csv',
    include_generation_report=True
)
```

### 4. Time Series Data Generation

Generate temporal data with realistic patterns:

```python
from tabular import TimeSeriesGenerator

# Load historical data
historical = pd.read_csv('sales_history.csv', parse_dates=['date'])

# Initialize time series generator
ts_generator = TimeSeriesGenerator(
    model='ydata-timegan',
    sequence_length=30
)

# Fit on historical patterns
ts_generator.fit(
    historical,
    temporal_cols=['date'],
    value_cols=['sales', 'inventory']
)

# Generate future scenarios
future_data = ts_generator.generate_future(
    periods=90,  # 90 days ahead
    scenarios={
        'optimistic': {'growth_rate': 1.1},
        'baseline': {'growth_rate': 1.0},
        'pessimistic': {'growth_rate': 0.9}
    }
)

# Visualize scenarios
ts_generator.plot_scenarios(future_data)
```

### 5. Multi-table Generation

Generate related tables maintaining relationships:

```python
from tabular import MultiTableGenerator

# Define relationships
relationships = [
    {
        'parent': 'customers',
        'child': 'orders',
        'key': 'customer_id',
        'cardinality': 'one_to_many'
    },
    {
        'parent': 'orders',
        'child': 'order_items',
        'key': 'order_id',
        'cardinality': 'one_to_many'
    }
]

# Load related tables
tables = {
    'customers': pd.read_csv('customers.csv'),
    'orders': pd.read_csv('orders.csv'),
    'order_items': pd.read_csv('order_items.csv')
}

# Generate synthetic database
mt_generator = MultiTableGenerator(model='sdv')
synthetic_db = mt_generator.generate(
    tables=tables,
    relationships=relationships,
    scale_factor=2.0  # 2x the original size
)

# Validate relationships
validation = mt_generator.validate_relationships(synthetic_db)
print(f"Referential integrity: {validation['integrity_score']:.2%}")
```

## Best Practices

### 1. Data Preparation

Prepare your data for optimal results:

```python
from tabular import DataPreparer

preparer = DataPreparer()

# Clean and standardize
clean_data = preparer.clean(
    raw_data,
    handle_missing='smart',  # Intelligent imputation
    remove_duplicates=True,
    standardize_types=True
)

# Remove PII before generation
safe_data = preparer.remove_pii(
    clean_data,
    columns=['name', 'email', 'ssn'],
    hash_ids=True  # Keep relationships
)

# Document constraints
constraints = preparer.extract_constraints(safe_data)
preparer.save_constraints(constraints, 'data_constraints.json')
```

### 2. Algorithm Selection

Choose the right algorithm for your needs:

| Use Case | Recommended Algorithm | Why |
|----------|---------------------|-----|
| Quick prototypes | SDV GaussianCopula | Fast, good quality |
| Complex relationships | CTGAN | Handles non-linear patterns |
| High privacy needs | YData with DP | Built-in privacy features |
| Large datasets | SDV with sampling | Memory efficient |
| Time series | YData TimeGAN | Temporal patterns |

### 3. Parameter Tuning

Optimize generation parameters:

```python
from tabular import HyperparameterTuner

# Automatic tuning
tuner = HyperparameterTuner(
    model='ctgan',
    metric='statistical_similarity'
)

best_params = tuner.tune(
    training_data,
    param_space={
        'epochs': [100, 300, 500],
        'batch_size': [100, 500, 1000],
        'embedding_dim': [64, 128, 256]
    },
    cv_folds=3
)

print(f"Best parameters: {best_params}")

# Use tuned parameters
generator = TabularGenerator(model='ctgan', **best_params)
```

### 4. Quality Validation

Always validate synthetic data quality:

```python
from tabular import QualityValidator

validator = QualityValidator()

# Comprehensive validation
report = validator.validate(
    real_data=original_data,
    synthetic_data=synthetic_data,
    tests=[
        'statistical_similarity',
        'correlation_preservation',
        'ml_efficacy',
        'privacy_metrics'
    ]
)

# Visual validation
validator.create_report(
    report,
    output_path='validation_report.html',
    include_plots=True
)

# Automated quality gates
if report['overall_score'] < 0.8:
    print("Warning: Quality below threshold")
    validator.suggest_improvements(report)
```

### 5. Production Deployment

Deploy generation pipelines:

```python
from tabular import GenerationPipeline

# Create production pipeline
pipeline = GenerationPipeline(
    name='daily_synthetic_generation',
    model='sdv',
    schedule='0 2 * * *'  # 2 AM daily
)

# Add data source
pipeline.add_source(
    type='database',
    connection_string='postgresql://...',
    query='SELECT * FROM transactions WHERE date = CURRENT_DATE'
)

# Add generation step
pipeline.add_generation(
    num_rows='match_source',
    constraints_file='constraints.json'
)

# Add validation
pipeline.add_validation(
    min_quality_score=0.85,
    alert_on_failure=True
)

# Add output
pipeline.add_output(
    type='s3',
    bucket='synthetic-data',
    format='parquet'
)

# Deploy pipeline
pipeline.deploy(environment='production')
```

## Troubleshooting

### Common Issues

#### 1. Memory Errors

```python
# Solution: Use batch processing
generator = TabularGenerator(
    model='ctgan',
    batch_size=100,  # Smaller batches
    use_gpu=False    # CPU uses less memory
)

# Or use sampling
sample = data.sample(n=10000)
generator.fit(sample)
```

#### 2. Slow Generation

```python
# Solution: Optimize parameters
generator = TabularGenerator(
    model='sdv',  # Faster than neural models
    parallel=True,  # Use multiple cores
    verbose=True    # Monitor progress
)

# Enable GPU for neural models
generator = TabularGenerator(
    model='ctgan',
    use_gpu=True,
    device='cuda:0'
)
```

#### 3. Poor Quality

```python
# Solution: Increase training
generator = TabularGenerator(
    model='ctgan',
    epochs=500,      # More training
    patience=50      # Early stopping
)

# Try different algorithms
for model in ['sdv', 'ctgan', 'ydata']:
    generator = TabularGenerator(model=model)
    generator.fit(data)
    quality = generator.evaluate()
    print(f"{model}: {quality['score']:.2f}")
```

#### 4. Constraint Violations

```python
# Solution: Add explicit constraints
constraints = {
    'age': {'min': 0, 'max': 120},
    'salary': {'min': 0},
    'percentage': {'min': 0, 'max': 100},
    'email': {'regex': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'}
}

generator = TabularGenerator(
    model='sdv',
    constraints=constraints,
    enforce_constraints=True
)
```

## Examples

### Example 1: E-commerce Data Generation

```python
from tabular import TabularGenerator, DataProfiler

# Profile real e-commerce data
profiler = DataProfiler()
profile = profiler.analyze('ecommerce_orders.csv')

print(f"Dataset shape: {profile['shape']}")
print(f"Column types: {profile['dtypes']}")
print(f"Missing values: {profile['missing']}")

# Generate synthetic e-commerce data
generator = TabularGenerator(model='ctgan')
generator.fit('ecommerce_orders.csv')

# Generate with business rules
synthetic_orders = generator.generate(
    num_rows=10000,
    constraints={
        'order_total': lambda x: x['quantity'] * x['unit_price'],
        'ship_date': lambda x: x['order_date'] + pd.Timedelta(days=2)
    }
)

# Add realistic patterns
synthetic_orders = generator.add_seasonality(
    synthetic_orders,
    date_column='order_date',
    seasonal_columns=['quantity', 'order_total'],
    pattern='retail'  # Black Friday, Christmas peaks
)

synthetic_orders.to_csv('synthetic_ecommerce.csv', index=False)
```

### Example 2: Healthcare Data Anonymization

```python
from tabular import HealthcareGenerator

# Specialized healthcare generator
hc_generator = HealthcareGenerator(
    privacy_standard='hipaa',
    preserve_clinical_validity=True
)

# Load patient data
patient_data = pd.read_csv('patient_records.csv')

# Generate HIPAA-compliant synthetic data
synthetic_patients = hc_generator.generate(
    patient_data,
    preserve_distributions={
        'diagnosis_codes': True,
        'lab_values': True,
        'demographics': 'generalize'
    },
    remove_identifiers=True
)

# Validate clinical validity
clinical_report = hc_generator.validate_clinical_rules(
    synthetic_patients,
    rules_file='clinical_constraints.yaml'
)

print(f"Clinical validity: {clinical_report['validity_score']:.2%}")
```

### Example 3: Financial Data Generation

```python
from tabular import FinancialDataGenerator

# Initialize with financial constraints
fin_generator = FinancialDataGenerator(
    model='ydata',
    ensure_accounting_rules=True
)

# Generate transaction data
transactions = fin_generator.generate_transactions(
    num_accounts=1000,
    num_transactions=50000,
    date_range=('2023-01-01', '2023-12-31'),
    transaction_types=['deposit', 'withdrawal', 'transfer'],
    ensure_balance_consistency=True
)

# Generate related customer data
customers = fin_generator.generate_customers(
    num_customers=1000,
    link_to_transactions=transactions,
    include_risk_scores=True
)

# Validate financial consistency
validation = fin_generator.validate_financial_rules(
    transactions=transactions,
    customers=customers
)

print(f"Balance consistency: {validation['balance_check']}")
print(f"Double-entry validity: {validation['double_entry_check']}")
```

### Example 4: IoT Sensor Data

```python
from tabular import IoTDataGenerator

# Generate sensor data
iot_generator = IoTDataGenerator(
    sensor_types=['temperature', 'humidity', 'pressure'],
    anomaly_rate=0.01  # 1% anomalies
)

# Generate time series sensor data
sensor_data = iot_generator.generate(
    num_sensors=100,
    duration_hours=24*7,  # One week
    sampling_rate='1min',
    include_failures=True,
    realistic_patterns={
        'temperature': 'daily_cycle',
        'humidity': 'weather_based'
    }
)

# Add correlated anomalies
sensor_data = iot_generator.add_anomalies(
    sensor_data,
    anomaly_types=['spike', 'drift', 'stuck'],
    correlation_groups=[
        ['sensor_1', 'sensor_2'],  # Nearby sensors
        ['sensor_10', 'sensor_11', 'sensor_12']
    ]
)

sensor_data.to_parquet('synthetic_iot_data.parquet')
```

## Next Steps

1. **Explore Advanced Features**: Check out multi-table generation, custom constraints, and streaming
2. **Join the Community**: Visit our [Discord](https://discord.gg/inferloop) for support
3. **Contribute**: We welcome contributions! See [CONTRIBUTING.md](../CONTRIBUTING.md)
4. **Enterprise**: Contact us for enterprise features and support

## Resources

- [API Documentation](./API_DOCUMENTATION.md)
- [SDK Reference](./SDK_DOCUMENTATION.md)
- [CLI Reference](./CLI_DOCUMENTATION.md)
- [Examples Repository](https://github.com/inferloop/tabular-examples)
- [Video Tutorials](https://youtube.com/inferloop)

## Support

- **Documentation**: [docs.inferloop.com/tabular](https://docs.inferloop.com/tabular)
- **GitHub Issues**: [github.com/inferloop/tabular/issues](https://github.com/inferloop/tabular/issues)
- **Email**: support@inferloop.com
- **Discord**: [discord.gg/inferloop](https://discord.gg/inferloop)

Happy generating! 🚀