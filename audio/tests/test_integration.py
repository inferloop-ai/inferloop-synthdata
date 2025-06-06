# tests/test_integration.py
"""
Integration tests for the complete audio synthesis pipeline
"""

import pytest
import torch
import torchaudio
import tempfile
import os
from pathlib import Path
import json

from audio_synth.sdk.client import AudioSynthSDK
from audio_synth.cli.main import cli
from click.testing import CliRunner

@pytest.mark.integration
class TestEndToEndPipeline:
    """End-to-end integration tests"""
    
    @pytest.fixture
    def sdk(self):
        """Initialize SDK for testing"""
        return AudioSynthSDK()
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_complete_generation_pipeline(self, sdk, temp_output_dir):
        """Test complete generation and validation pipeline"""
        # Step 1: Generate audio
        result = sdk.generate_and_validate(
            method="diffusion",
            prompt="Integration test audio sample",
            num_samples=2,
            validators=["quality", "privacy", "fairness"]
        )
        
        # Verify generation
        assert len(result["audios"]) == 2
        for audio in result["audios"]:
            assert isinstance(audio, torch.Tensor)
            assert len(audio.shape) == 1
        
        # Verify validation
        assert "quality" in result["validation"]
        assert "privacy" in result["validation"]
        assert "fairness" in result["validation"]
        
        # Step 2: Save generated audio
        for i, audio in enumerate(result["audios"]):
            filename = temp_output_dir / f"integration_test_{i}.wav"
            torchaudio.save(str(filename), audio.unsqueeze(0), 22050)
            assert filename.exists()
        
        # Step 3: Re-validate saved audio
        audios = []
        for i in range(2):
            filename = temp_output_dir / f"integration_test_{i}.wav"
            audio, _ = torchaudio.load(str(filename))
            audios.append(audio.squeeze())
        
        revalidation_results = sdk.validate(
            audios=audios,
            validators=["quality"]
        )
        
        assert len(revalidation_results["quality"]) == 2
    
    def test_privacy_preservation_pipeline(self, sdk):
        """Test privacy preservation across the pipeline"""
        # Generate audio with different privacy levels
        privacy_levels = ["low", "medium", "high"]
        results = {}
        
        for level in privacy_levels:
            result = sdk.generate_and_validate(
                method="diffusion",
                prompt="Privacy test audio",
                num_samples=1,
                validators=["privacy"],
                conditions={"privacy_level": level}
            )
            
            privacy_metrics = result["validation"]["privacy"][0]
            results[level] = privacy_metrics
        
        # Verify privacy increases with level
        assert results["high"]["speaker_anonymity"] >= results["medium"]["speaker_anonymity"]
        assert results["medium"]["speaker_anonymity"] >= results["low"]["speaker_anonymity"]
    
    def test_fairness_across_demographics(self, sdk):
        """Test fairness across different demographic groups"""
        demographics = [
            {"gender": "male", "age_group": "adult"},
            {"gender": "female", "age_group": "adult"},
            {"gender": "other", "age_group": "adult"}
        ]
        
        all_audios = []
        all_metadata = []
        
        for demo in demographics:
            result = sdk.generate(
                method="diffusion",
                prompt="Fairness test audio",
                num_samples=2,
                conditions={"demographics": demo}
            )
            
            all_audios.extend(result)
            all_metadata.extend([{"demographics": demo}] * len(result))
        
        # Validate fairness across all samples
        fairness_results = sdk.validate(
            audios=all_audios,
            metadata=all_metadata,
            validators=["fairness"]
        )
        
        # Check that fairness metrics are calculated
        assert len(fairness_results["fairness"]) == len(all_audios)
        
        # Verify demographic parity is included
        first_result = fairness_results["fairness"][0]
        assert "demographic_parity" in first_result

@pytest.mark.integration
class TestCLIIntegration:
    """Test CLI integration"""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner"""
        return CliRunner()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for CLI tests"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_cli_generate_command(self, runner, temp_dir):
        """Test CLI generate command"""
        result = runner.invoke(cli, [
            'generate',
            '--method', 'diffusion',
            '--prompt', 'CLI test audio',
            '--num-samples', '1',
            '--output-dir', temp_dir,
            '--duration', '1.0'
        ])
        
        assert result.exit_code == 0
        
        # Check that output files were created
        output_path = Path(temp_dir)
        wav_files = list(output_path.glob('*.wav'))
        assert len(wav_files) >= 1
        
        # Check metadata file
        metadata_files = list(output_path.glob('metadata_*.json'))
        assert len(metadata_files) == 1
    
    def test_cli_validate_command(self, runner, temp_dir):
        """Test CLI validate command"""
        # First generate some audio to validate
        temp_path = Path(temp_dir)
        
        # Create a dummy audio file
        audio = torch.randn(1, 22050)
        audio_file = temp_path / "test_audio.wav"
        torchaudio.save(str(audio_file), audio, 22050)
        
        # Run validation
        result = runner.invoke(cli, [
            'validate',
            '--input-dir', temp_dir,
            '--output-file', str(temp_path / 'validation_results.json'),
            '--validators', 'quality'
        ])
        
        assert result.exit_code == 0
        
        # Check that validation results were created
        results_file = temp_path / 'validation_results.json'
        assert results_file.exists()
        
        with open(results_file) as f:
            results = json.load(f)
            assert 'validation_results' in results
    
    def test_cli_init_config(self, runner, temp_dir):
        """Test CLI config initialization"""
        result = runner.invoke(cli, [
            'init-config',
            '--output-dir', temp_dir
        ])
        
        assert result.exit_code == 0
        
        # Check that config file was created
        config_file = Path(temp_dir) / 'default.yaml'
        assert config_file.exists()

@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Performance and stress tests"""
    
    def test_batch_generation_performance(self):
        """Test performance with batch generation"""
        sdk = AudioSynthSDK()
        
        import time
        start_time = time.time()
        
        # Generate multiple samples
        result = sdk.generate(
            method="diffusion",
            prompt="Performance test",
            num_samples=10
        )
        
        generation_time = time.time() - start_time
        
        # Verify all samples generated
        assert len(result) == 10
        
        # Performance check (should generate 10 samples in reasonable time)
        assert generation_time < 60  # 60 seconds max for 10 samples
        
        print(f"Generated 10 samples in {generation_time:.2f} seconds")
        print(f"Average time per sample: {generation_time/10:.2f} seconds")
    
    def test_memory_usage(self):
        """Test memory usage during generation"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        sdk = AudioSynthSDK()
        
        # Generate multiple samples
        for i in range(5):
            _ = sdk.generate(
                method="diffusion",
                prompt=f"Memory test {i}",
                num_samples=2
            )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage: {initial_memory:.1f} MB -> {final_memory:.1f} MB")
        print(f"Memory increase: {memory_increase:.1f} MB")
        
        # Memory increase should be reasonable (less than 1GB)
        assert memory_increase < 1024
    
    def test_concurrent_generation(self):
        """Test concurrent generation requests"""
        import threading
        import time
        
        sdk = AudioSynthSDK()
        results = []
        errors = []
        
        def generate_audio(prompt_id):
            try:
                result = sdk.generate(
                    method="diffusion",
                    prompt=f"Concurrent test {prompt_id}",
                    num_samples=1
                )
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=generate_audio, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all succeeded
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 3
        
        for result in results:
            assert len(result) == 1
            assert isinstance(result[0], torch.Tensor)

# ============================================================================
