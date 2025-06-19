"""
Structured Documents Synthetic Data Generator - MVP Setup
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="structured-docs-synth",
    version="0.1.0",
    author="InferLoop",
    author_email="contact@inferloop.ai",
    description="Synthetic structured document generation platform for Legal, Healthcare, Banking, and Government domains",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/inferloop-ai/inferloop-synthdata",
    packages=find_packages(where="docs/structured-documents-synthetic-data/src"),
    package_dir={"": "docs/structured-documents-synthetic-data/src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Markup",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.7.0",
            "pre-commit>=3.6.0",
        ],
        "ocr": [
            "pytesseract>=0.3.10",
            "opencv-python>=4.8.0",
        ],
        "cloud": [
            "boto3>=1.35.0",
            "azure-storage-blob>=12.19.0",
            "google-cloud-storage>=2.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "synth-doc=structured_docs_synth.cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "structured_docs_synth": [
            "configs/*.yaml",
            "configs/**/*.yaml",
            "templates/*.json",
            "templates/**/*.json",
        ],
    },
    zip_safe=False,
)