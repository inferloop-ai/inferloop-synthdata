# examples/usage_examples.py
from src.sdk.client import SynthCodeSDK

# SDK Usage Example
def sdk_example():
    # Initialize SDK
    sdk = SynthCodeSDK(base_url="http://localhost:8000")
    
    # Generate Python functions
    prompts = [
        "Create a function to calculate fibonacci numbers",
        "Implement a binary search algorithm",
        "Write a function to validate email addresses",
        "Create a decorator for timing function execution"
    ]
    
    result = sdk.generate_code(
        prompts=prompts,
        language="python",
        framework="fastapi",
        count=20,
        include_validation=True
    )
    
    print(f"Generated {len(result['generated_code'])} code samples")
    print(f"Validation results: {len(result['validation_results'])}")
    
    # Example generated code sample
    sample = result['generated_code'][0]
    print(f"Sample code:\n{sample['code']}")

# CLI Usage Examples
"""
# Generate Python functions
synth code generate -p "fibonacci function,binary search,email validator" -l python -c 10 -o output.jsonl

# Generate with configuration file
synth code generate -p prompts.txt -l javascript -f express --config config.yaml -o js_samples.jsonl

# Validate existing code
synth validate-file -f my_code.py -l python
"""

# API Usage Examples
"""
# Generate code via API
curl -X POST "http://localhost:8000/generate/code" \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["fibonacci function", "binary search"],
    "language": "python",
    "count": 5,
    "include_validation": true
  }'

# Get available templates
curl -X GET "http://localhost:8000/generate/code/templates"
"""

if __name__ == "__main__":
    sdk_example()
