🎯 Core Architecture
The SDK provides a unified interface across SDV, CTGAN, YData-Synthetic, and other libraries through:

Base abstractions with SyntheticDataConfig and BaseSyntheticGenerator
Library-specific wrappers that handle the complexity of each tool
Factory pattern for easy generator creation
Comprehensive validation framework with statistical tests and quality metrics

🚀 Multiple Interfaces
1. Python SDK

from inferloop_synthetic.sdk import GeneratorFactory, SyntheticDataConfig

config = SyntheticDataConfig(
    generator_type="sdv",
    model_type="gaussian_copula", 
    num_samples=1000,
    categorical_columns=["category", "region"]
)

generator = GeneratorFactory.create_generator(config)
result = generator.fit_generate(data)

2. CLI Interface

inferloop-synthetic generate --config config.json

3. API Interface

from inferloop_synthetic.api import create_app

app = create_app()

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    config = SyntheticDataConfig(**data['config'])
    generator = GeneratorFactory.create_generator(config)
    result = generator.fit_generate(data['data'])
    return result


📊 Comprehensive Validation
The validation framework includes:

Statistical tests: KS tests, Chi-square tests
Distribution analysis: Mean, std, correlation preservation
Privacy metrics: Distance-based privacy assessment
Utility metrics: ML model performance comparison
Overall quality scoring with actionable recommendations

🔧 Key Benefits

Consistency: Same API regardless of underlying library
Flexibility: Easy to switch between SDV, CTGAN, YData approaches
Quality: Built-in validation and quality assessment
Scalability: Async generation with job tracking
Extensibility: Easy to add new synthetic data libraries

📈 Usage Scenarios

Data Scientists: Use SDK directly for experimentation
MLOps Teams: Deploy via REST API for production pipelines
Analysts: Use CLI for quick data generation tasks
Researchers: Leverage validation framework for quality assessment

