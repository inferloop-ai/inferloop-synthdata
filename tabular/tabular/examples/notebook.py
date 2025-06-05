# examples/notebook.ipynb (Python version for demonstration)
"""
Inferloop Synthetic Data SDK - Complete Example
This notebook demonstrates the full workflow of the Inferloop Synthetic Data SDK
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import Inferloop SDK
from inferloop_synthetic.sdk import (
    GeneratorFactory, 
    SyntheticDataConfig, 
    SyntheticDataValidator
)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_sample_dataset():
    """Create a sample dataset for demonstration"""
    np.random.seed(42)
    
    # Generate sample customer data
    n_samples = 5000
    
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.normal(40, 15, n_samples).astype(int),
        'income': np.random.lognormal(10.5, 0.5, n_samples),
        'credit_score': np.random.normal(650, 100, n_samples).astype(int),
        'num_purchases': np.random.poisson(5, n_samples),
        'total_spent': np.random.exponential(500, n_samples),
        'gender': np.random.choice(['M', 'F', 'Other'], n_samples, p=[0.45, 0.45, 0.1]),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic'], n_samples, p=[0.2, 0.5, 0.3])
    }
    
    df = pd.DataFrame(data)
    
    # Add some realistic constraints
    df['age'] = np.clip(df['age'], 18, 85)
    df['credit_score'] = np.clip(df['credit_score'], 300, 850)
    df['income'] = np.clip(df['income'], 20000, 500000)
    
    # Create correlation between income and spending
    df['total_spent'] = df['total_spent'] + df['income'] * 0.02 * np.random.uniform(0.5, 1.5, n_samples)
    
    return df

def example_1_basic_generation():
    """Example 1: Basic synthetic data generation with SDV"""
    print("🚀 Example 1: Basic Synthetic Data Generation with SDV")
    print("=" * 60)
    
    # Create sample data
    print("📊 Creating sample dataset...")
    original_data = create_sample_dataset()
    print(f"Original data shape: {original_data.shape}")
    print(f"Columns: {list(original_data.columns)}")
    
    # Configure synthetic data generation
    config = SyntheticDataConfig(
        generator_type="sdv",
        model_type="gaussian_copula",
        num_samples=3000,
        categorical_columns=['gender', 'region', 'customer_segment'],
        continuous_columns=['age', 'income', 'credit_score', 'total_spent'],
        validate_output=True
    )
    
    # Create generator and generate data
    print("\n🤖 Creating SDV generator...")
    generator = GeneratorFactory.create_generator(config)
    
    print("🏋️ Training model...")
    generator.fit(original_data)
    
    print("🎲 Generating synthetic data...")
    result = generator.generate()
    
    print(f"\n✅ Generated {len(result.synthetic_data)} synthetic samples!")
    print(f"Generation time: {result.generation_time:.2f} seconds")
    print(f"Model info: {result.model_info['library']} - {result.model_info['model_type']}")
    
    return original_data, result.synthetic_data

def example_2_ctgan_generation():
    """Example 2: Synthetic data generation with CTGAN"""
    print("\n🚀 Example 2: Synthetic Data Generation with CTGAN")
    print("=" * 60)
    
    # Create sample data
    original_data = create_sample_dataset()
    
    # Configure CTGAN generation
    config = SyntheticDataConfig(
        generator_type="ctgan",
        model_type="ctgan",
        num_samples=2000,
        epochs=100,  # Reduced for demo
        batch_size=500,
        categorical_columns=['gender', 'region', 'customer_segment']
    )
    
    # Generate data
    print("🤖 Creating CTGAN generator...")
    generator = GeneratorFactory.create_generator(config)
    
    print("🏋️ Training CTGAN model...")
    result = generator.fit_generate(original_data)
    
    print(f"\n✅ Generated {len(result.synthetic_data)} synthetic samples!")
    print(f"Generation time: {result.generation_time:.2f} seconds")
    
    return original_data, result.synthetic_data

def example_3_validation_and_comparison():
    """Example 3: Comprehensive validation and comparison"""
    print("\n🚀 Example 3: Validation and Quality Assessment")
    print("=" * 60)
    
    # Generate data using multiple methods
    original_data = create_sample_dataset()
    
    # SDV Gaussian Copula
    sdv_config = SyntheticDataConfig(
        generator_type="sdv",
        model_type="gaussian_copula",
        num_samples=2000,
        categorical_columns=['gender', 'region', 'customer_segment']
    )
    
    sdv_generator = GeneratorFactory.create_generator(sdv_config)
    sdv_result = sdv_generator.fit_generate(original_data)
    
    # Run comprehensive validation
    print("🔍 Running comprehensive validation...")
    validator = SyntheticDataValidator(original_data, sdv_result.synthetic_data)
    validation_results = validator.validate_all()
    
    # Display results
    print("\n📊 Validation Results:")
    print("-" * 40)
    print(f"Overall Quality Score: {validation_results['overall_quality']:.3f}")
    print(f"Basic Statistics Score: {validation_results['basic_stats']['score']:.3f}")
    print(f"Distribution Similarity: {validation_results['distribution_similarity']['score']:.3f}")
    print(f"Correlation Preservation: {validation_results['correlation_preservation']['score']:.3f}")
    print(f"Privacy Score: {validation_results['privacy_metrics']['score']:.3f}")
    print(f"Utility Score: {validation_results['utility_metrics']['score']:.3f}")
    
    # Generate detailed report
    print("\n📄 Detailed Validation Report:")
    print(validator.generate_report())
    
    return original_data, sdv_result.synthetic_data, validation_results

def example_4_advanced_configuration():
    """Example 4: Advanced configuration with hyperparameter tuning"""
    print("\n🚀 Example 4: Advanced Configuration")
    print("=" * 60)
    
    original_data = create_sample_dataset()
    
    # Advanced configuration with hyperparameters
    advanced_config = SyntheticDataConfig(
        generator_type="ydata",
        model_type="wgan_gp",
        num_samples=1500,
        categorical_columns=['gender', 'region', 'customer_segment'],
        continuous_columns=['age', 'income', 'credit_score', 'total_spent'],
        epochs=200,
        batch_size=128,
        learning_rate=1e-4,
        hyperparameters={
            'noise_dim': 64,
            'layers_dim': 256
        },
        quality_threshold=0.8
    )
    
    try:
        print("🤖 Creating YData generator with advanced config...")
        generator = GeneratorFactory.create_generator(advanced_config)
        result = generator.fit_generate(original_data)
        
        print(f"✅ Generated {len(result.synthetic_data)} samples")
        print(f"Configuration: {result.config.to_dict()}")
        
        return original_data, result.synthetic_data
    
    except ImportError:
        print("⚠️ YData Synthetic not available, skipping this example")
        return None, None

def visualize_comparison(original_data, synthetic_data, title="Data Comparison"):
    """Visualize comparison between original and synthetic data"""
    print(f"\n📊 Visualizing: {title}")
    
    # Set up subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=16)
    
    # Age distribution
    axes[0, 0].hist(original_data['age'], alpha=0.7, label='Original', bins=30)
    axes[0, 0].hist(synthetic_data['age'], alpha=0.7, label='Synthetic', bins=30)
    axes[0, 0].set_title('Age Distribution')
    axes[0, 0].legend()
    
    # Income distribution (log scale)
    axes[0, 1].hist(np.log(original_data['income']), alpha=0.7, label='Original', bins=30)
    axes[0, 1].hist(np.log(synthetic_data['income']), alpha=0.7, label='Synthetic', bins=30)
    axes[0, 1].set_title('Income Distribution (log)')
    axes[0, 1].legend()
    
    # Credit score distribution
    axes[0, 2].hist(original_data['credit_score'], alpha=0.7, label='Original', bins=30)
    axes[0, 2].hist(synthetic_data['credit_score'], alpha=0.7, label='Synthetic', bins=30)
    axes[0, 2].set_title('Credit Score Distribution')
    axes[0, 2].legend()
    
    # Gender distribution
    orig_gender = original_data['gender'].value_counts()
    synth_gender = synthetic_data['gender'].value_counts()
    x = range(len(orig_gender))
    axes[1, 0].bar([i-0.2 for i in x], orig_gender.values, 0.4, label='Original')
    axes[1, 0].bar([i+0.2 for i in x], synth_gender.values, 0.4, label='Synthetic')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(orig_gender.index)
    axes[1, 0].set_title('Gender Distribution')
    axes[1, 0].legend()
    
    # Region distribution
    orig_region = original_data['region'].value_counts()
    synth_region = synthetic_data['region'].value_counts()
    x = range(len(orig_region))
    axes[1, 1].bar([i-0.2 for i in x], orig_region.values, 0.4, label='Original')
    axes[1, 1].bar([i+0.2 for i in x], synth_region.values, 0.4, label='Synthetic')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(orig_region.index)
    axes[1, 1].set_title('Region Distribution')
    axes[1, 1].legend()
    
    # Correlation heatmap
    corr_diff = (original_data.select_dtypes(include=[np.number]).corr() - 
                 synthetic_data.select_dtypes(include=[np.number]).corr())
    sns.heatmap(corr_diff, annot=True, cmap='RdBu', center=0, ax=axes[1, 2])
    axes[1, 2].set_title('Correlation Difference\n(Original - Synthetic)')
    
    plt.tight_layout()
    plt.show()

def run_all_examples():
    """Run all examples in sequence"""
    print("🎯 Inferloop Synthetic Data SDK - Complete Examples")
    print("=" * 80)
    
    # Example 1: Basic SDV generation
    orig_data, sdv_synthetic = example_1_basic_generation()
    if orig_data is not None:
        visualize_comparison(orig_data, sdv_synthetic, "SDV Gaussian Copula")
    
    # Example 2: CTGAN generation
    try:
        orig_data, ctgan_synthetic = example_2_ctgan_generation()
        if orig_data is not None:
            visualize_comparison(orig_data, ctgan_synthetic, "CTGAN")
    except ImportError:
        print("⚠️ CTGAN not available, skipping example 2")
    
    # Example 3: Validation
    orig_data, synthetic_data, validation = example_3_validation_and_comparison()
    
    # Example 4: Advanced configuration
    try:
        example_4_advanced_configuration()
    except ImportError:
        print("⚠️ YData Synthetic not available, skipping example 4")
    
    print("\n🎉 All examples completed!")
    print("Check the generated plots and validation results above.")

if __name__ == "__main__":
    run_all_examples()


