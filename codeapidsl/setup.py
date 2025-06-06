# setup.py
from setuptools import setup, find_packages

setup(
    name="synth-code-generator",
    version="1.0.0",
    description="Synthetic Code Generation System",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn[standard]>=0.15.0",
        "click>=8.0.0",
        "requests>=2.25.0",
        "pydantic>=1.8.0",
        "pyyaml>=5.4.0",
        "boto3>=1.18.0",
        "grpcio>=1.39.0",
        "grpcio-tools>=1.39.0",
        "transformers>=4.12.0",
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.7b0",
            "flake8>=3.9.0",
            "mypy>=0.910"
        ]
    },
    entry_points={
        "console_scripts": [
            "synth=src.cli.commands:cli",
        ],
    },
    python_requires=">=3.8",
)
