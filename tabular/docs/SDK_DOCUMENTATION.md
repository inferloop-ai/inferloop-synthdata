# Tabular SDK Documentation

## Overview

The Tabular SDK provides a powerful Python interface for generating synthetic tabular data using state-of-the-art algorithms. This documentation covers all classes, methods, and features available in the SDK.

## Table of Contents

1. [Installation](#installation)
2. [Core Classes](#core-classes)
3. [Generators](#generators)
4. [Validators](#validators)
5. [Privacy Features](#privacy-features)
6. [Advanced Features](#advanced-features)
7. [API Reference](#api-reference)
8. [Examples](#examples)

## Installation

```bash
pip install inferloop-tabular

# With specific extras
pip install inferloop-tabular[gpu]      # GPU support
pip install inferloop-tabular[privacy]  # Privacy features
pip install inferloop-tabular[all]      # All features
```

## Core Classes

### TabularGenerator

The main class for synthetic data generation.

```python
from tabular import TabularGenerator

class TabularGenerator:
    """Main interface for tabular synthetic data generation"""
    
    def __init__(self, 
                 model: str = 'sdv',
                 **kwargs):
        """
        Initialize generator.
        
        Args:
            model: Algorithm to use ('sdv', 'ctgan', 'ydata')
            **kwargs: Model-specific parameters
        """
        
    def fit(self, 
            data: Union[pd.DataFrame, str],
            column_types: Optional[Dict[str, str]] = None,
            constraints: Optional[List[Constraint]] = None):
        """
        Train the model on data.
        
        Args:
            data: Training data (DataFrame or file path)
            column_types: Override auto-detected types
            constraints: Business rules to enforce
        """
        
    def generate(self, 
                 num_rows: int,
                 conditions: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Generate synthetic data.
        
        Args:
            num_rows: Number of rows to generate
            conditions: Conditional generation constraints
            
        Returns:
            Generated synthetic data
        """
        
    def evaluate(self, 
                 synthetic_data: pd.DataFrame,
                 metrics: List[str] = None) -> Dict[str, float]:
        """
        Evaluate synthetic data quality.
        
        Args:
            synthetic_data: Generated data to evaluate
            metrics: Specific metrics to calculate
            
        Returns:
            Dictionary of metric scores
        """
```

### BaseGenerator

Abstract base class for all generators.

```python
from tabular.sdk.base import BaseGenerator

class BaseGenerator(ABC):
    """Abstract base class for synthetic data generators"""
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Train the generator on data"""
        pass
    
    @abstractmethod
    def generate(self, num_samples: int, **kwargs) -> pd.DataFrame:
        """Generate synthetic samples"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save trained model"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load trained model"""
        pass
    
    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Check if model is trained"""
        pass
```

## Generators

### SDVGenerator

Synthetic Data Vault implementation with multiple models.

```python
from tabular.sdk.sdv_generator import SDVGenerator

# Initialize with GaussianCopula
generator = SDVGenerator(
    model_type='gaussian_copula',  # or 'ctgan', 'copulagan', 'tvae'
    default_distribution='gaussian',
    enforce_min_max_values=True,
    locales=['en_US']  # For realistic fake data
)

# Fit the model
generator.fit(
    data=training_data,
    discrete_columns=['category', 'status'],
    update_transformers={
        'email': 'email',  # Use email transformer
        'phone': 'phone_number'
    }
)

# Generate with conditions
synthetic = generator.generate(
    num_samples=1000,
    conditions={
        'category': 'premium',
        'balance': lambda x: x > 1000
    }
)

# Use different SDV models
models = {
    'fast': SDVGenerator('gaussian_copula'),
    'quality': SDVGenerator('ctgan'),
    'balanced': SDVGenerator('copulagan')
}
```

#### Model-specific parameters

```python
# GaussianCopula parameters
gc_generator = SDVGenerator(
    model_type='gaussian_copula',
    default_distribution='beta',  # Better for bounded data
    categorical_transformer='one_hot_encoding',
    numerical_distributions={
        'age': 'uniform',
        'income': 'gamma',
        'score': 'norm'
    }
)

# CTGAN parameters (via SDV)
ctgan_generator = SDVGenerator(
    model_type='ctgan',
    epochs=300,
    batch_size=500,
    discriminator_steps=1,
    log_frequency=True,
    verbose=True
)

# CopulaGAN parameters
copulagan_generator = SDVGenerator(
    model_type='copulagan',
    epochs=300,
    discriminator_lr=2e-4,
    generator_lr=2e-4,
    cuda=True
)
```

### CTGANGenerator

Standalone CTGAN implementation with advanced features.

```python
from tabular.sdk.ctgan_generator import CTGANGenerator

# Initialize CTGAN
generator = CTGANGenerator(
    embedding_dim=128,       # Size of embeddings
    generator_dim=(256, 256),  # Generator architecture
    discriminator_dim=(256, 256),  # Discriminator architecture
    generator_lr=2e-4,       # Learning rate
    discriminator_lr=2e-4,
    batch_size=500,
    epochs=300,
    pac=10,                  # Pac-GAN for mode coverage
    cuda=True
)

# Advanced training with callbacks
generator.fit(
    data,
    discrete_columns=['col1', 'col2'],
    callbacks=[
        EarlyStopping(patience=50),
        ModelCheckpoint('checkpoints/'),
        TensorBoard('logs/')
    ]
)

# Conditional generation
conditional_synthetic = generator.sample(
    n=1000,
    condition_column='category',
    condition_value='A'
)

# Progressive training for large datasets
generator.fit_progressive(
    data,
    stages=[100, 200, 300],  # Epochs per stage
    sample_sizes=[0.1, 0.5, 1.0]  # Data fraction per stage
)
```

#### Mode-specific normalization

```python
# Custom normalization for multimodal distributions
generator = CTGANGenerator(
    embedding_dim=128,
    generator_activation='relu',
    discriminator_activation='leaky_relu',
    batch_norm=True,
    dropout=0.1
)

# Handle specific column types
generator.set_column_converter(
    column='salary',
    converter='mode_specific_normalization',
    n_modes=3  # For trimodal distribution
)
```

### YDataGenerator

Enterprise-grade generator with privacy features.

```python
from tabular.sdk.ydata_generator import YDataGenerator

# Initialize YData synthesizer
generator = YDataGenerator(
    model_type='regular',  # 'regular', 'wgan', 'wgangp', 'dragan', 'cramer'
    noise_dim=128,
    layers_dim=128,
    batch_size=128,
    epochs=300,
    learning_rate=5e-4,
    beta_1=0.5,
    beta_2=0.9
)

# Fit with privacy budget
generator.fit(
    data,
    privacy_level='high',  # Automatic privacy configuration
    num_teachers=100,      # For PATE-GAN
    epsilon=1.0            # Differential privacy
)

# Time series generation
from tabular.sdk.ydata_generator import TimeGANGenerator

timegan = TimeGANGenerator(
    seq_len=24,           # Sequence length
    n_seq=1,              # Number of sequences
    hidden_dim=24,        # Hidden state size
    gamma=1,              # Time-series loss weight
    noise_dim=32,
    layers_dim=128,
    batch_size=128,
    epochs=50000
)

# Fit time series data
timegan.fit(
    data,
    temporal_cols=['timestamp'],
    value_cols=['value1', 'value2'],
    sequence_index='entity_id'
)

# Generate future sequences
future_data = timegan.generate_sequences(
    n_sequences=100,
    seq_len=48,  # Generate longer sequences
    conditions={'entity_type': 'A'}
)
```

#### Advanced YData features

```python
# Conditional WGAN-GP
wgan_generator = YDataGenerator(
    model_type='wgangp',
    gradient_penalty_weight=10,
    n_critic=5,  # Critic updates per generator update
    conditional=True,
    condition_columns=['category', 'region']
)

# DRAGAN for improved stability
dragan_generator = YDataGenerator(
    model_type='dragan',
    lambda_reg=10,  # Regularization weight
    perturb_scale=0.5
)

# Privacy-preserving generation
private_generator = YDataGenerator(
    model_type='regular',
    differential_privacy=True,
    epsilon=1.0,
    delta=1e-5,
    clip_value=1.0,
    noise_multiplier=0.1
)
```

## Validators

### QualityValidator

Comprehensive validation of synthetic data quality.

```python
from tabular.sdk.validator import QualityValidator

validator = QualityValidator()

# Statistical validation
stats_report = validator.validate_statistics(
    real_data=original_df,
    synthetic_data=synthetic_df,
    tests=[
        'ks_test',          # Kolmogorov-Smirnov test
        'chi_squared',      # For categorical columns
        'correlation',      # Correlation preservation
        'mutual_info'       # Mutual information
    ]
)

print(f"Statistical similarity: {stats_report['overall_score']:.2f}")

# Machine learning efficacy
ml_report = validator.validate_ml_efficacy(
    real_data=original_df,
    synthetic_data=synthetic_df,
    target_column='target',
    models=['random_forest', 'logistic_regression', 'xgboost'],
    cv_folds=5
)

print(f"ML efficacy score: {ml_report['efficacy_score']:.2f}")

# Privacy validation
privacy_report = validator.validate_privacy(
    real_data=original_df,
    synthetic_data=synthetic_df,
    quasi_identifiers=['age', 'zipcode', 'gender'],
    sensitive_attributes=['income', 'health_status']
)

print(f"Privacy score: {privacy_report['privacy_score']:.2f}")
print(f"K-anonymity: {privacy_report['k_anonymity']}")
```

### ConstraintValidator

Validate business rules and constraints.

```python
from tabular.sdk.validator import ConstraintValidator

# Define constraints
constraints = [
    {
        'type': 'range',
        'column': 'age',
        'min': 0,
        'max': 120
    },
    {
        'type': 'unique',
        'column': 'customer_id'
    },
    {
        'type': 'relationship',
        'columns': ['start_date', 'end_date'],
        'condition': lambda df: (df['end_date'] >= df['start_date']).all()
    },
    {
        'type': 'sum',
        'columns': ['pct_1', 'pct_2', 'pct_3'],
        'total': 100
    }
]

validator = ConstraintValidator(constraints)
violations = validator.validate(synthetic_data)

if violations:
    print(f"Found {len(violations)} constraint violations")
    validator.fix_violations(synthetic_data, inplace=True)
```

### VisualValidator

Visual comparison of real and synthetic data.

```python
from tabular.sdk.validator import VisualValidator

visual_validator = VisualValidator()

# Generate comparison report
report = visual_validator.create_report(
    real_data=original_df,
    synthetic_data=synthetic_df,
    output_path='validation_report.html',
    include_plots=[
        'distributions',     # Univariate distributions
        'correlations',      # Correlation matrices
        'pca',              # PCA visualization
        'tsne',             # t-SNE visualization
        'pairplot'          # Pairwise relationships
    ]
)

# Individual plot generation
fig = visual_validator.plot_distributions(
    real_data=original_df,
    synthetic_data=synthetic_df,
    columns=['age', 'income', 'score'],
    plot_type='overlay'  # or 'side_by_side'
)
fig.savefig('distributions.png')
```

## Privacy Features

### PrivacyEngine

Comprehensive privacy protection mechanisms.

```python
from tabular.sdk.privacy import PrivacyEngine

# Initialize privacy engine
privacy_engine = PrivacyEngine(
    epsilon=1.0,              # Privacy budget
    delta=1e-5,               # Failure probability
    mechanism='gaussian',     # or 'laplace'
    bounds_strategy='clip'    # or 'normalize'
)

# Apply differential privacy to generator
private_generator = privacy_engine.make_private(
    generator,
    noise_multiplier=1.1,
    max_grad_norm=1.0,
    secure_mode=True
)

# Generate with privacy guarantees
private_data = private_generator.generate(
    num_rows=1000,
    enforce_privacy=True
)

# Get privacy accounting
spent_budget = privacy_engine.get_privacy_spent()
print(f"Privacy budget spent: ε={spent_budget['epsilon']:.2f}")
```

### AnonymizationEngine

Advanced anonymization techniques.

```python
from tabular.sdk.privacy import AnonymizationEngine

anonymizer = AnonymizationEngine()

# Configure anonymization strategies
anonymizer.configure({
    'name': 'redact',
    'email': 'hash',
    'phone': 'generalize',
    'ssn': 'synthetic',
    'address': 'generalize_geographic',
    'dob': 'generalize_temporal'
})

# K-anonymity
k_anon_data = anonymizer.k_anonymize(
    data,
    quasi_identifiers=['age', 'zipcode', 'gender'],
    k=5,
    suppression_limit=0.1
)

# L-diversity
l_diverse_data = anonymizer.l_diversify(
    k_anon_data,
    sensitive_attribute='diagnosis',
    l=3
)

# T-closeness
t_close_data = anonymizer.t_closeness(
    l_diverse_data,
    sensitive_attribute='income',
    t=0.15
)
```

## Advanced Features

### BatchProcessor

Efficient batch processing for large datasets.

```python
from tabular.sdk.batch import BatchProcessor

# Initialize batch processor
processor = BatchProcessor(
    generator=generator,
    batch_size=10000,
    n_jobs=4,  # Parallel processing
    memory_limit='8GB'
)

# Process large file in batches
processor.process_file(
    input_file='large_dataset.csv',
    output_file='synthetic_large.csv',
    chunksize=50000,
    callback=lambda i, total: print(f"Progress: {i}/{total}")
)

# Streaming generation
for batch in processor.generate_stream(
    total_rows=1000000,
    batch_size=10000
):
    # Process each batch
    process_batch(batch)
```

### StreamingGenerator

Real-time streaming data generation.

```python
from tabular.sdk.streaming import StreamingGenerator

# Initialize streaming generator
stream_gen = StreamingGenerator(
    base_generator=generator,
    buffer_size=1000,
    rate_limit=100  # Records per second
)

# Start streaming
stream = stream_gen.start_stream(
    duration=3600,  # Stream for 1 hour
    output_format='json'
)

# Consume stream
for record in stream:
    # Process each record
    kafka_producer.send('synthetic_topic', record)
    
# Async streaming
async def consume_stream():
    async for batch in stream_gen.async_stream(batch_size=100):
        await process_batch_async(batch)
```

### ModelProfiler

Profile and optimize model performance.

```python
from tabular.sdk.profiler import ModelProfiler

profiler = ModelProfiler()

# Profile model performance
profile = profiler.profile_model(
    generator,
    test_data=sample_data,
    metrics=[
        'training_time',
        'generation_time',
        'memory_usage',
        'quality_score'
    ]
)

# Compare different models
comparison = profiler.compare_models({
    'sdv': SDVGenerator(),
    'ctgan': CTGANGenerator(),
    'ydata': YDataGenerator()
}, test_data=sample_data)

# Get recommendations
recommendations = profiler.recommend_model(
    data_profile={
        'num_rows': 100000,
        'num_columns': 50,
        'categorical_columns': 10,
        'privacy_required': True
    }
)

print(f"Recommended model: {recommendations['best_model']}")
print(f"Reason: {recommendations['reason']}")
```

### CacheManager

Caching for improved performance.

```python
from tabular.sdk.cache import CacheManager

# Initialize cache
cache = CacheManager(
    backend='redis',  # or 'memory', 'disk'
    ttl=3600,        # Time to live
    max_size='1GB'
)

# Cache trained models
cached_generator = cache.cached_generator(generator)

# First call trains and caches
cached_generator.fit(data)

# Subsequent calls use cache
synthetic = cached_generator.generate(1000)  # Fast!

# Manual cache control
cache.save_model('model_key', generator)
loaded_generator = cache.load_model('model_key')

# Clear cache
cache.clear()
```

### VersionManager

Model versioning and tracking.

```python
from tabular.sdk.versioning import VersionManager

version_manager = VersionManager(
    storage_backend='s3',  # or 'local', 'azure', 'gcs'
    bucket='model-versions'
)

# Save model version
version_id = version_manager.save_version(
    generator,
    metadata={
        'dataset': 'customer_data',
        'algorithm': 'ctgan',
        'quality_score': 0.92,
        'training_date': '2024-01-15'
    },
    tags=['production', 'v2.0']
)

# Load specific version
generator = version_manager.load_version(version_id)

# List versions
versions = version_manager.list_versions(
    filter_tags=['production'],
    sort_by='quality_score',
    limit=10
)

# Compare versions
comparison = version_manager.compare_versions(
    version_ids=['v1', 'v2'],
    test_data=sample_data
)
```

## API Reference

### Configuration Classes

```python
@dataclass
class GenerationConfig:
    """Configuration for data generation"""
    algorithm: str = 'sdv'
    num_samples: int = 1000
    batch_size: int = 500
    random_state: Optional[int] = None
    constraints: Optional[List[Dict]] = None
    conditions: Optional[Dict] = None
    output_format: str = 'dataframe'  # or 'csv', 'parquet', 'json'

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    epochs: int = 300
    learning_rate: float = 2e-4
    batch_size: int = 500
    early_stopping: bool = True
    patience: int = 50
    validation_split: float = 0.2
    verbose: bool = True
    log_frequency: int = 10
    checkpoint_dir: Optional[str] = None
```

### Utility Functions

```python
from tabular.sdk.utils import (
    detect_column_types,
    infer_constraints,
    validate_data,
    profile_data,
    sample_stratified,
    create_metadata
)

# Automatic column type detection
column_types = detect_column_types(data)

# Infer constraints from data
constraints = infer_constraints(
    data,
    include_statistical=True,
    confidence=0.95
)

# Validate data before training
issues = validate_data(
    data,
    check_duplicates=True,
    check_missing=True,
    check_types=True
)

# Create detailed data profile
profile = profile_data(
    data,
    include_correlations=True,
    include_distributions=True
)

# Stratified sampling
sample = sample_stratified(
    data,
    stratify_columns=['category', 'region'],
    sample_size=1000
)

# Create metadata for SDV
metadata = create_metadata(
    data,
    primary_key='id',
    datetime_format='%Y-%m-%d',
    locales=['en_US']
)
```

### Exceptions

```python
from tabular.sdk.exceptions import (
    TabularException,
    ModelNotFittedError,
    InvalidDataError,
    ConstraintViolationError,
    PrivacyBudgetExceededError,
    GenerationError
)

try:
    synthetic = generator.generate(1000)
except ModelNotFittedError:
    print("Model must be fitted before generation")
except ConstraintViolationError as e:
    print(f"Constraint violated: {e.constraint}")
except PrivacyBudgetExceededError as e:
    print(f"Privacy budget exceeded. Remaining: {e.remaining_budget}")
```

## Examples

### Example 1: Complete Pipeline

```python
from tabular import Pipeline, TabularGenerator, QualityValidator, PrivacyEngine

# Create pipeline
pipeline = Pipeline()

# Add components
pipeline.add_generator(TabularGenerator('ctgan'))
pipeline.add_validator(QualityValidator())
pipeline.add_privacy_engine(PrivacyEngine(epsilon=1.0))

# Configure pipeline
pipeline.configure({
    'generator': {
        'epochs': 300,
        'batch_size': 500
    },
    'validator': {
        'min_quality': 0.8,
        'privacy_check': True
    },
    'output': {
        'format': 'parquet',
        'compression': 'snappy'
    }
})

# Run pipeline
result = pipeline.run(
    input_data='customer_data.csv',
    output_path='synthetic_customers.parquet',
    num_samples=10000
)

print(f"Quality score: {result.quality_score:.2f}")
print(f"Privacy guarantee: ε={result.privacy_epsilon:.2f}")
```

### Example 2: Custom Generator

```python
from tabular.sdk.base import BaseGenerator

class CustomGenerator(BaseGenerator):
    """Custom generator implementation"""
    
    def __init__(self, custom_param=0.5):
        super().__init__()
        self.custom_param = custom_param
        self.model = None
        
    def fit(self, data: pd.DataFrame, **kwargs):
        """Custom training logic"""
        # Your training code here
        self.model = train_custom_model(data, self.custom_param)
        self._is_fitted = True
        
    def generate(self, num_samples: int, **kwargs) -> pd.DataFrame:
        """Custom generation logic"""
        if not self.is_fitted:
            raise ModelNotFittedError()
            
        # Your generation code here
        return generate_with_custom_model(self.model, num_samples)
        
    def save(self, path: str):
        """Save custom model"""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load(self, path: str):
        """Load custom model"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self._is_fitted = True

# Use custom generator
generator = CustomGenerator(custom_param=0.7)
generator.fit(data)
synthetic = generator.generate(1000)
```

### Example 3: Multi-table Generation

```python
from tabular import MultiTableSynthesizer

# Define table relationships
metadata = {
    'tables': {
        'users': {
            'primary_key': 'user_id',
            'columns': {
                'user_id': {'sdtype': 'id'},
                'age': {'sdtype': 'numerical'},
                'city': {'sdtype': 'categorical'}
            }
        },
        'transactions': {
            'primary_key': 'transaction_id',
            'columns': {
                'transaction_id': {'sdtype': 'id'},
                'user_id': {'sdtype': 'id'},
                'amount': {'sdtype': 'numerical'},
                'date': {'sdtype': 'datetime'}
            }
        }
    },
    'relationships': [
        {
            'parent_table': 'users',
            'child_table': 'transactions',
            'parent_primary_key': 'user_id',
            'child_foreign_key': 'user_id'
        }
    ]
}

# Initialize synthesizer
synthesizer = MultiTableSynthesizer(metadata, model='ctgan')

# Fit on multiple tables
tables = {
    'users': users_df,
    'transactions': transactions_df
}
synthesizer.fit(tables)

# Generate synthetic database
synthetic_tables = synthesizer.generate(scale=1.5)

# Validate relationships
assert len(synthetic_tables['transactions']['user_id'].unique()) <= len(synthetic_tables['users'])
```

## Best Practices

1. **Always validate**: Use QualityValidator after generation
2. **Start simple**: Begin with SDV before trying complex models
3. **Monitor privacy**: Track privacy budget consumption
4. **Use caching**: Cache trained models for production
5. **Profile first**: Understand your data before generation
6. **Test constraints**: Verify business rules are preserved
7. **Batch large datasets**: Use BatchProcessor for scalability

## Performance Tips

1. **GPU acceleration**: Use CUDA for 10-100x speedup
2. **Optimize batch size**: Larger batches are faster but use more memory
3. **Use appropriate algorithm**: SDV for speed, CTGAN for quality
4. **Enable caching**: Avoid retraining models
5. **Parallel processing**: Use n_jobs parameter when available

## Troubleshooting

Common issues and solutions:

1. **Import errors**: Ensure all dependencies installed with `pip install inferloop-tabular[all]`
2. **CUDA errors**: Check GPU availability with `torch.cuda.is_available()`
3. **Memory errors**: Reduce batch size or use BatchProcessor
4. **Slow training**: Enable GPU or reduce model complexity
5. **Poor quality**: Increase epochs or try different algorithm

## Support

- GitHub Issues: [github.com/inferloop/tabular/issues](https://github.com/inferloop/tabular/issues)
- Documentation: [docs.inferloop.com/tabular](https://docs.inferloop.com/tabular)
- Examples: [github.com/inferloop/tabular-examples](https://github.com/inferloop/tabular-examples)