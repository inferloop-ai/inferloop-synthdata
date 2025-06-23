"""
Streaming support for large dataset processing
"""

import os
import tempfile
from typing import Iterator, Optional, Dict, Any, Tuple, Callable, AsyncIterator
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

import pandas as pd
import numpy as np
from pandas.io.parsers import TextFileReader

from .base import BaseSyntheticGenerator, GenerationResult, SyntheticDataConfig
from .factory import GeneratorFactory
from .validator import SyntheticDataValidator


class StreamingDataProcessor:
    """Process large datasets in chunks with streaming"""
    
    def __init__(self, 
                 chunk_size: int = 10000,
                 max_memory_mb: int = 1024,
                 parallel_chunks: int = None):
        self.chunk_size = chunk_size
        self.max_memory_mb = max_memory_mb
        self.parallel_chunks = parallel_chunks or mp.cpu_count()
        self.temp_dir = None
        
    def __enter__(self):
        """Context manager entry"""
        self.temp_dir = tempfile.mkdtemp(prefix='inferloop_streaming_')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup temp files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def estimate_chunk_size(self, file_path: str, target_memory_mb: int = 100) -> int:
        """Estimate optimal chunk size based on file structure"""
        # Read small sample to estimate row size
        sample_df = pd.read_csv(file_path, nrows=1000)
        
        # Estimate memory usage per row
        memory_per_row = sample_df.memory_usage(deep=True).sum() / len(sample_df)
        
        # Calculate chunk size to fit in target memory
        rows_per_chunk = int((target_memory_mb * 1024 * 1024) / memory_per_row)
        
        # Apply bounds
        return max(1000, min(rows_per_chunk, 100000))
    
    def read_csv_chunks(self, file_path: str, 
                       chunk_size: Optional[int] = None) -> Iterator[pd.DataFrame]:
        """Read CSV file in chunks"""
        chunk_size = chunk_size or self.chunk_size
        
        # Use pandas chunked reader
        reader = pd.read_csv(file_path, chunksize=chunk_size)
        
        for i, chunk in enumerate(reader):
            yield i, chunk
    
    def process_chunk(self, chunk_data: Tuple[int, pd.DataFrame], 
                     generator_config: SyntheticDataConfig,
                     fit_metadata: Optional[Dict[str, Any]] = None) -> Tuple[int, pd.DataFrame]:
        """Process a single chunk"""
        chunk_id, chunk_df = chunk_data
        
        # Create generator for this chunk
        generator = GeneratorFactory.create_generator(generator_config)
        
        if fit_metadata:
            # Use pre-fitted model metadata if available
            # Note: This would require implementing _apply_metadata in base generators
            # For now, fit and generate for each chunk
            result = generator.fit_generate(chunk_df)
        else:
            # Fit and generate for this chunk
            result = generator.fit_generate(chunk_df)
        
        return chunk_id, result.synthetic_data
    
    async def process_chunks_async(self, 
                                  chunks: Iterator[Tuple[int, pd.DataFrame]],
                                  generator_config: SyntheticDataConfig,
                                  fit_metadata: Optional[Dict[str, Any]] = None) -> Iterator[pd.DataFrame]:
        """Process chunks asynchronously"""
        with ThreadPoolExecutor(max_workers=self.parallel_chunks) as executor:
            # Submit all chunks for processing
            futures = []
            for chunk_data in chunks:
                future = executor.submit(
                    self.process_chunk, 
                    chunk_data, 
                    generator_config,
                    fit_metadata
                )
                futures.append(future)
            
            # Yield results as they complete
            for future in futures:
                chunk_id, synthetic_chunk = future.result()
                yield synthetic_chunk
    
    def save_chunk(self, chunk: pd.DataFrame, chunk_id: int) -> str:
        """Save chunk to temporary file"""
        if not self.temp_dir:
            raise RuntimeError("StreamingDataProcessor must be used as context manager")
        
        chunk_path = os.path.join(self.temp_dir, f'chunk_{chunk_id:06d}.parquet')
        chunk.to_parquet(chunk_path, compression='snappy')
        return chunk_path
    
    def merge_chunks(self, chunk_paths: list, output_path: str, 
                    output_format: str = 'csv') -> None:
        """Merge temporary chunk files into final output"""
        if output_format == 'csv':
            # Write header from first chunk
            first_chunk = pd.read_parquet(chunk_paths[0])
            first_chunk.to_csv(output_path, index=False, mode='w')
            
            # Append remaining chunks
            for chunk_path in chunk_paths[1:]:
                chunk = pd.read_parquet(chunk_path)
                chunk.to_csv(output_path, index=False, mode='a', header=False)
                
        elif output_format == 'parquet':
            # Merge parquet files
            dfs = []
            for chunk_path in chunk_paths:
                dfs.append(pd.read_parquet(chunk_path))
            
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_df.to_parquet(output_path, compression='snappy')
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")


class StreamingSyntheticGenerator:
    """Synthetic data generator with streaming support"""
    
    def __init__(self, config: SyntheticDataConfig, 
                 chunk_size: int = 10000,
                 parallel_chunks: int = None):
        self.config = config
        self.chunk_size = chunk_size
        self.parallel_chunks = parallel_chunks
        self.processor = StreamingDataProcessor(
            chunk_size=chunk_size,
            parallel_chunks=parallel_chunks
        )
    
    def fit_sample(self, data_path: str, sample_size: int = 10000) -> Dict[str, Any]:
        """Fit generator on data sample"""
        # Read sample of data
        sample_df = pd.read_csv(data_path, nrows=sample_size)
        
        # Create generator and fit
        generator = GeneratorFactory.create_generator(self.config)
        generator.fit(sample_df)
        
        # Extract model metadata (would need to be implemented in base generators)
        # For now, return None to indicate metadata extraction not supported
        return None
    
    def generate_streaming(self, 
                          input_path: str,
                          output_path: str,
                          sample_ratio: float = 1.0,
                          output_format: str = 'csv',
                          progress_callback: Optional[Callable[[int, int], None]] = None) -> GenerationResult:
        """Generate synthetic data with streaming for large files"""
        
        with self.processor:
            # Estimate optimal chunk size
            chunk_size = self.processor.estimate_chunk_size(input_path)
            
            # Count total rows for progress tracking
            total_rows = sum(1 for _ in open(input_path)) - 1  # Subtract header
            total_chunks = (total_rows + chunk_size - 1) // chunk_size
            
            # Fit on sample if needed
            fit_metadata = None
            if self.config.model_params.get('use_sample_fitting', True):
                fit_metadata = self.fit_sample(input_path)
            
            # Process chunks
            chunks = self.processor.read_csv_chunks(input_path, chunk_size)
            chunk_paths = []
            processed_chunks = 0
            
            # Process chunks in parallel
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def process_all():
                async for synthetic_chunk in self.processor.process_chunks_async(
                    chunks, self.config, fit_metadata
                ):
                    # Apply sampling if needed
                    if sample_ratio < 1.0:
                        sample_size = int(len(synthetic_chunk) * sample_ratio)
                        synthetic_chunk = synthetic_chunk.sample(n=sample_size)
                    
                    # Save chunk
                    chunk_path = self.processor.save_chunk(
                        synthetic_chunk, 
                        len(chunk_paths)
                    )
                    chunk_paths.append(chunk_path)
                    
                    # Update progress
                    nonlocal processed_chunks
                    processed_chunks += 1
                    if progress_callback:
                        progress_callback(processed_chunks, total_chunks)
            
            loop.run_until_complete(process_all())
            
            # Merge chunks into final output
            self.processor.merge_chunks(chunk_paths, output_path, output_format)
            
            # Calculate final statistics
            total_synthetic_rows = sum(
                len(pd.read_parquet(p)) for p in chunk_paths
            )
            
            return GenerationResult(
                synthetic_data=None,  # Too large to return in memory
                metadata={
                    'output_path': output_path,
                    'total_rows': total_synthetic_rows,
                    'chunks_processed': len(chunk_paths),
                    'chunk_size': chunk_size,
                    'generator_type': self.config.generator_type.value,
                    'model_type': self.config.model_type.value
                },
                validation_metrics={}
            )
    
    async def generate_streaming_async(self,
                                     input_path: str,
                                     output_path: str,
                                     sample_ratio: float = 1.0,
                                     output_format: str = 'csv') -> AsyncIterator[Dict[str, Any]]:
        """Async generator for streaming synthetic data generation"""
        
        with self.processor:
            # Estimate chunk size
            chunk_size = self.processor.estimate_chunk_size(input_path)
            
            # Fit on sample
            fit_metadata = None
            if self.config.model_params.get('use_sample_fitting', True):
                fit_metadata = self.fit_sample(input_path)
            
            # Process chunks and yield progress
            chunks = self.processor.read_csv_chunks(input_path, chunk_size)
            chunk_count = 0
            
            async for synthetic_chunk in self.processor.process_chunks_async(
                chunks, self.config, fit_metadata
            ):
                # Apply sampling
                if sample_ratio < 1.0:
                    sample_size = int(len(synthetic_chunk) * sample_ratio)
                    synthetic_chunk = synthetic_chunk.sample(n=sample_size)
                
                # Save chunk
                chunk_path = self.processor.save_chunk(synthetic_chunk, chunk_count)
                
                # Yield progress update
                yield {
                    'type': 'progress',
                    'chunk': chunk_count,
                    'rows_generated': len(synthetic_chunk),
                    'chunk_path': chunk_path
                }
                
                chunk_count += 1
            
            # Final merge
            chunk_paths = [
                os.path.join(self.processor.temp_dir, f'chunk_{i:06d}.parquet')
                for i in range(chunk_count)
            ]
            
            self.processor.merge_chunks(chunk_paths, output_path, output_format)
            
            # Yield completion
            yield {
                'type': 'completed',
                'output_path': output_path,
                'total_chunks': chunk_count,
                'format': output_format
            }


class StreamingValidator:
    """Validate synthetic data in streaming fashion"""
    
    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size
        self.validator = SyntheticDataValidator()
    
    def validate_streaming(self,
                          original_path: str,
                          synthetic_path: str,
                          sample_size: int = 10000) -> Dict[str, Any]:
        """Validate synthetic data against original using sampling"""
        
        # Read samples from both files
        original_sample = pd.read_csv(original_path, nrows=sample_size)
        synthetic_sample = pd.read_csv(synthetic_path, nrows=sample_size)
        
        # Run validation on samples
        validation_results = {
            'statistical_similarity': self.validator.statistical_similarity(
                original_sample, synthetic_sample
            ),
            'privacy_metrics': self.validator.privacy_metrics(
                original_sample, synthetic_sample
            ),
            'utility_metrics': self.validator.utility_metrics(
                original_sample, synthetic_sample
            )
        }
        
        # Add streaming-specific metrics
        validation_results['streaming_metrics'] = {
            'original_sample_size': len(original_sample),
            'synthetic_sample_size': len(synthetic_sample),
            'validation_method': 'sampling',
            'sample_ratio': sample_size / sum(1 for _ in open(original_path))
        }
        
        return validation_results
    
    def validate_chunks(self,
                       original_chunks: Iterator[pd.DataFrame],
                       synthetic_chunks: Iterator[pd.DataFrame]) -> Dict[str, Any]:
        """Validate data chunk by chunk"""
        
        aggregated_metrics = {
            'statistical_similarity': [],
            'privacy_scores': [],
            'utility_scores': []
        }
        
        for orig_chunk, synth_chunk in zip(original_chunks, synthetic_chunks):
            # Validate chunk
            chunk_stats = self.validator.statistical_similarity(orig_chunk, synth_chunk)
            chunk_privacy = self.validator.privacy_metrics(orig_chunk, synth_chunk)
            chunk_utility = self.validator.utility_metrics(orig_chunk, synth_chunk)
            
            # Aggregate metrics
            aggregated_metrics['statistical_similarity'].append(
                chunk_stats['overall_score']
            )
            aggregated_metrics['privacy_scores'].append(
                chunk_privacy.get('overall_privacy_score', 0)
            )
            aggregated_metrics['utility_scores'].append(
                chunk_utility.get('overall_utility_score', 0)
            )
        
        # Calculate final scores
        return {
            'statistical_similarity': np.mean(aggregated_metrics['statistical_similarity']),
            'privacy_score': np.mean(aggregated_metrics['privacy_scores']),
            'utility_score': np.mean(aggregated_metrics['utility_scores']),
            'chunks_validated': len(aggregated_metrics['statistical_similarity'])
        }


# Utility functions for streaming operations
def estimate_file_chunks(file_path: str, chunk_size: int = 10000) -> int:
    """Estimate number of chunks in a file"""
    total_rows = sum(1 for _ in open(file_path)) - 1  # Subtract header
    return (total_rows + chunk_size - 1) // chunk_size


def create_streaming_generator(config: SyntheticDataConfig, 
                             chunk_size: int = 10000) -> StreamingSyntheticGenerator:
    """Factory function to create streaming generator"""
    return StreamingSyntheticGenerator(
        config=config,
        chunk_size=chunk_size,
        parallel_chunks=config.model_params.get('parallel_chunks', mp.cpu_count())
    )