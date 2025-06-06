# conftest.py
"""
Pytest configuration and shared fixtures
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path

# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )

@pytest.fixture(scope="session")
def temp_models_dir():
    """Create temporary directory for model files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        models_path = Path(tmpdir) / "models"
        models_path.mkdir()
        
        # Create dummy model files
        for model_name in ["diffusion_model.pt", "tts_model.pt", "gan_model.pt"]:
            model_file = models_path / model_name
            torch.save({"dummy": "model"}, model_file)
        
        yield str(models_path)

@pytest.fixture
def sample_audio_batch():
    """Generate batch of sample audio for testing"""
    batch_size = 5
    duration = 1.0
    sample_rate = 22050
    samples = int(duration * sample_rate)
    
    audios = []
    for i in range(batch_size):
        # Generate different frequencies for each sample
        frequency = 220 * (2 ** (i / 12))  # Musical intervals
        t = torch.linspace(0, duration, samples)
        audio = torch.sin(2 * torch.pi * frequency * t) * 0.5
        audios.append(audio)
    
    return audios

@pytest.fixture
def sample_metadata_batch():
    """Generate batch of metadata for testing"""
    demographics = [
        {"gender": "male", "age_group": "adult", "accent": "american"},
        {"gender": "female", "age_group": "adult", "accent": "british"},
        {"gender": "other", "age_group": "elderly", "accent": "australian"},
        {"gender": "male", "age_group": "child", "accent": "canadian"},
        {"gender": "female", "age_group": "adult", "accent": "indian"}
    ]
    
    metadata = []
    for i, demo in enumerate(demographics):
        meta = {
            "sample_id": i,
            "demographics": demo,
            "prompt": f"Test sample {i}",
            "method": "diffusion"
        }
        metadata.append(meta)
    
    return metadata

# Test data fixtures
@pytest.fixture
def clean_speech_audio():
    """Generate clean speech-like audio"""
    duration = 2.0
    sample_rate = 22050
    samples = int(duration * sample_rate)
    
    # Generate formant-like structure
    t = torch.linspace(0, duration, samples)
    
    # Multiple harmonics to simulate speech
    fundamental = 150  # Typical male voice
    audio = (
        torch.sin(2 * torch.pi * fundamental * t) * 0.3 +
        torch.sin(2 * torch.pi * fundamental * 2 * t) * 0.2 +
        torch.sin(2 * torch.pi * fundamental * 3 * t) * 0.1
    )
    
    # Add some noise for realism
    audio += torch.randn_like(audio) * 0.05
    
    return audio

@pytest.fixture
def noisy_audio():
    """Generate noisy audio for testing"""
    duration = 2.0
    sample_rate = 22050
    samples = int(duration * sample_rate)
    
    # High noise content
    audio = torch.randn(samples) * 0.5
    
    return audio

# Performance monitoring fixture
@pytest.fixture
def performance_monitor():
    """Monitor test performance"""
    import time
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    yield
    
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    execution_time = end_time - start_time
    memory_delta = end_memory - start_memory
    
    print(f"\nPerformance metrics:")
    print(f"  Execution time: {execution_time:.2f} seconds")
    print(f"  Memory delta: {memory_delta:.1f} MB")
    
    # Optional: Assert performance constraints
    if execution_time > 30:  # 30 second warning
        print(f"WARNING: Test took {execution_time:.2f} seconds")
    
    if memory_delta > 100:  # 100 MB warning
        print(f"WARNING: Test used {memory_delta:.1f} MB additional memory")

# Skip markers for missing dependencies
def pytest_runtest_setup(item):
    """Skip tests based on available dependencies"""
    # Skip slow tests unless specifically requested
    if "slow" in item.keywords and not item.config.getoption("--runslow"):
        pytest.skip("need --runslow option to run")
    
    # Skip integration tests in unit test mode
    if "integration" in item.keywords and item.config.getoption("--unit-only"):
        pytest.skip("skipping integration test in unit-only mode")

def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--unit-only", action="store_true", default=False, help="run only unit tests"
    )
    parser.addoption(
        "--integration-only", action="store_true", default=False, help="run only integration tests"
    )

# ============================================================================

