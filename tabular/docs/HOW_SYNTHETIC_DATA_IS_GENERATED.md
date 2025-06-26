# How Synthetic Tabular Data is Generated

## Overview

The Tabular Synthetic Data Generation platform uses state-of-the-art machine learning algorithms to create realistic, privacy-preserving synthetic data that maintains the statistical properties and relationships of your original dataset. This document explains the technical details of our generation process.

## Table of Contents

1. [Generation Pipeline](#generation-pipeline)
2. [Available Algorithms](#available-algorithms)
3. [Data Processing](#data-processing)
4. [Quality Assurance](#quality-assurance)
5. [Privacy Protection](#privacy-protection)
6. [Performance Optimization](#performance-optimization)

## Generation Pipeline

### 1. Data Ingestion

```
Original Data → Validation → Profiling → Preprocessing → Model Training → Generation → Post-processing → Synthetic Data
```

The generation pipeline consists of several stages:

1. **Data Ingestion**: Accept CSV, Parquet, JSON, or database connections
2. **Validation**: Check data types, missing values, and consistency
3. **Profiling**: Analyze statistical properties and relationships
4. **Preprocessing**: Handle missing values, encode categoricals, normalize numerics
5. **Model Training**: Train the selected algorithm on processed data
6. **Generation**: Create synthetic samples
7. **Post-processing**: Ensure constraints and business rules
8. **Quality Check**: Validate synthetic data quality

### 2. Column Type Detection

Our system automatically detects and handles different column types:

- **Numerical**: Continuous values (float, int)
- **Categorical**: Discrete categories (string, enum)
- **Datetime**: Temporal data with various formats
- **Boolean**: Binary true/false values
- **Text**: Free-form text (if enabled)
- **Geospatial**: Coordinates and locations

## Available Algorithms

### SDV (Synthetic Data Vault)

**Best for**: General-purpose synthetic data generation with good balance of speed and quality

The SDV library provides multiple model options:

#### GaussianCopula
- Models dependencies between columns using copulas
- Preserves marginal distributions
- Handles mixed data types well
- Fast training and generation

```python
# How it works internally
model = GaussianCopula(
    enforce_min_max_values=True,
    default_distribution='gaussian'
)
model.fit(data)
synthetic_data = model.sample(num_rows=1000)
```

#### CTGAN (Included in SDV)
- Uses Generative Adversarial Networks
- Better for complex relationships
- Higher quality but slower

#### CopulaGAN
- Combines copulas with GANs
- Best of both approaches
- Good for complex distributions

### CTGAN (Standalone)

**Best for**: High-quality generation with complex relationships and distributions

CTGAN uses a conditional GAN architecture specifically designed for tabular data:

1. **Mode-specific normalization**: Handles multimodal distributions
2. **Conditional generator**: Ensures even sampling of discrete values
3. **PacGAN discriminator**: Prevents mode collapse
4. **Training process**:
   - Generator creates fake samples
   - Discriminator learns to distinguish real vs fake
   - Both improve through adversarial training

```python
# Architecture overview
Generator: Noise → Hidden Layers → Synthetic Sample
Discriminator: Sample → Hidden Layers → Real/Fake Score
```

Key features:
- Handles imbalanced categorical columns
- Preserves complex multimodal distributions
- Learns non-linear relationships
- Privacy-preserving by design

### YData Synthetic

**Best for**: Enterprise use cases requiring advanced privacy guarantees and customization

YData provides several synthesizers:

#### Regular Synthesizer
- Fast baseline model
- Good for simple datasets
- Minimal configuration

#### WGAN-GP Synthesizer
- Wasserstein GAN with gradient penalty
- More stable training
- Better convergence properties

#### DRAGAN Synthesizer
- Deep Regret Analytic GAN
- Improved mode coverage
- Better for rare categories

#### TimeGAN Synthesizer
- Specialized for time-series data
- Preserves temporal dynamics
- Handles irregular time intervals

## Data Processing

### Preprocessing Steps

1. **Missing Value Handling**
   ```python
   # Strategies used:
   - Mean/median imputation for numerics
   - Mode imputation for categoricals
   - Forward/backward fill for time series
   - Model-based imputation for complex patterns
   ```

2. **Encoding**
   ```python
   # Categorical encoding:
   - One-hot encoding for low cardinality
   - Target encoding for high cardinality
   - Ordinal encoding for ordered categories
   
   # Numerical transformations:
   - Standard scaling
   - Min-max normalization
   - Box-Cox transformation for skewed data
   ```

3. **Constraint Learning**
   ```python
   # Automatically detected constraints:
   - Min/max bounds
   - Positive/negative only
   - Sum constraints (e.g., percentages = 100)
   - Business rules
   ```

### Feature Engineering

The system automatically creates features to improve generation:

- **Interaction terms**: Captures relationships between columns
- **Temporal features**: Extracts date/time components
- **Aggregations**: Statistical summaries for grouped data
- **Embeddings**: Learned representations for high-cardinality categories

## Quality Assurance

### Statistical Validation

1. **Univariate Metrics**
   - Mean, median, mode comparison
   - Standard deviation and variance
   - Distribution shape (skewness, kurtosis)
   - Range and percentiles

2. **Multivariate Metrics**
   - Correlation matrices
   - Mutual information
   - Joint distributions
   - Conditional distributions

3. **Machine Learning Efficacy**
   - Train classifier on real vs synthetic
   - Compare model performance
   - Feature importance similarity

### Visual Validation

- **Distribution plots**: Compare real vs synthetic
- **Correlation heatmaps**: Relationship preservation
- **PCA/t-SNE plots**: Overall structure similarity
- **QQ plots**: Distribution matching

### Privacy Metrics

1. **Distance to Closest Record (DCR)**
   - Measures minimum distance to real records
   - Ensures no memorization

2. **Membership Inference Risk**
   - Tests if records can be identified
   - Validates privacy protection

3. **Attribute Inference Risk**
   - Checks if hidden attributes can be inferred
   - Protects sensitive information

## Privacy Protection

### Differential Privacy

Optional differential privacy can be applied:

```python
# Noise addition mechanism
epsilon = 1.0  # Privacy budget
sensitivity = max_value - min_value
noise_scale = sensitivity / epsilon
synthetic_value = real_value + laplace_noise(scale=noise_scale)
```

### K-Anonymity

Ensures each record is indistinguishable from at least k-1 others:

```python
# Generalization hierarchy
Age: 25 → 20-30 → 20-40 → Adult
Location: Street → City → State → Country
```

### Synthetic Data Guarantees

1. **No 1:1 mapping**: Synthetic records don't correspond to real individuals
2. **Statistical similarity**: Preserves patterns without copying data
3. **Plausible deniability**: Can't determine if specific record was in training data

## Performance Optimization

### Scalability Features

1. **Batch Processing**
   - Process large datasets in chunks
   - Parallel training for independent models
   - Distributed generation

2. **GPU Acceleration**
   - CUDA support for neural models
   - 10-100x speedup for training
   - Optimized tensor operations

3. **Incremental Learning**
   - Update models with new data
   - No need to retrain from scratch
   - Maintains consistency

### Memory Management

```python
# Strategies for large datasets:
- Chunked reading and processing
- Out-of-core training
- Model checkpointing
- Efficient data structures
```

### Caching and Optimization

1. **Model Caching**: Trained models are cached for reuse
2. **Preprocessing Cache**: Transformed data stored for speed
3. **Result Caching**: Common queries cached
4. **JIT Compilation**: Critical paths optimized

## Generation Parameters

### Key Parameters by Algorithm

**SDV Parameters:**
- `default_distribution`: Distribution assumption for numerics
- `enforce_min_max_values`: Respect original bounds
- `locales`: For generating locale-specific data
- `verbose`: Detailed logging

**CTGAN Parameters:**
- `epochs`: Training iterations (default: 300)
- `batch_size`: Samples per batch (default: 500)
- `discriminator_steps`: D updates per G update
- `embedding_dim`: Size of embeddings (default: 128)
- `generator_dim`: Hidden layer sizes
- `discriminator_dim`: Discriminator architecture

**YData Parameters:**
- `noise_dim`: Latent space dimension
- `num_cols`: Numerical column indices
- `cat_cols`: Categorical column indices
- `learning_rate`: Training speed
- `beta_1`, `beta_2`: Adam optimizer parameters

### Tuning Guidelines

1. **For Speed**: Use SDV GaussianCopula, reduce epochs
2. **For Quality**: Use CTGAN/YData, increase epochs and dimensions
3. **For Privacy**: Enable differential privacy, reduce batch size
4. **For Large Data**: Use batch processing, GPU acceleration

## Advanced Features

### Conditional Generation

Generate data matching specific conditions:

```python
# Example: Generate only high-value customers
conditions = {
    'customer_segment': 'premium',
    'annual_revenue': '>100000'
}
synthetic_data = generator.sample(
    num_rows=1000,
    conditions=conditions
)
```

### Time Series Generation

For temporal data with dependencies:

```python
# Preserves:
- Trends and seasonality
- Autocorrelation
- Cross-series dependencies
- Irregular intervals
```

### Multi-table Generation

Generate related tables maintaining referential integrity:

```python
# Parent-child relationships
# Foreign key constraints
# Cardinality preservation
# Consistent business logic
```

## Best Practices

1. **Data Preparation**
   - Clean data before generation
   - Remove or mask PII
   - Ensure consistent formats
   - Document constraints

2. **Algorithm Selection**
   - Start with SDV for baseline
   - Use CTGAN for complex data
   - Try YData for enterprise needs
   - Compare multiple algorithms

3. **Validation Strategy**
   - Always validate statistical properties
   - Test downstream ML tasks
   - Check privacy metrics
   - Get domain expert review

4. **Iterative Improvement**
   - Start with small samples
   - Tune parameters based on metrics
   - Increase complexity gradually
   - Monitor generation time

## Troubleshooting

### Common Issues

1. **Poor Quality**: Increase epochs, try different algorithm
2. **Slow Generation**: Reduce model complexity, use GPU
3. **Memory Errors**: Enable batch processing, reduce dimensions
4. **Constraint Violations**: Add explicit constraints, post-process
5. **Mode Collapse**: Adjust GAN parameters, try different algorithm

### Debug Mode

Enable detailed logging:
```python
generator = TabularGenerator(
    model='ctgan',
    verbose=True,
    debug=True
)
```

## Future Enhancements

Planned improvements:
1. **AutoML Integration**: Automatic algorithm selection
2. **Federated Learning**: Train on distributed data
3. **Real-time Generation**: Stream synthetic data
4. **Custom Constraints**: User-defined business rules
5. **Explainable AI**: Understand generation decisions