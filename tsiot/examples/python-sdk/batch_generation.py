#!/usr/bin/env python3
"""
Batch Generation Example for TSIOT Python SDK

This example demonstrates batch processing capabilities of the TSIOT Python SDK including:
- Batch time series generation
- Parallel processing
- Progress tracking
- Memory management
- Output organization
- Performance optimization
"""

import asyncio
import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

# Add the parent directory to the path to import the SDK
sys.path.insert(0, '../../internal/sdk/python')

from client import TSIOTClient, AsyncTSIOTClient
from timeseries import TimeSeries, TimeSeriesMetadata
from generators import ARIMAGenerator, LSTMGenerator, StatisticalGenerator
from utils import TSIOTError, setup_logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchGenerationConfig:
    """Configuration for batch generation operations"""
    
    def __init__(self):
        # Generation parameters
        self.total_series = 100
        self.series_length = 1000
        self.batch_size = 10
        
        # Algorithm distribution
        self.algorithm_weights = {
            'arima': 0.4,
            'lstm': 0.3,
            'statistical': 0.3
        }
        
        # Parallel processing
        self.max_workers = min(8, multiprocessing.cpu_count())
        self.use_async = True
        
        # Output configuration
        self.output_dir = Path("batch_output")
        self.save_individual_files = True
        self.save_summary = True
        
        # Performance settings
        self.chunk_size = 50  # For memory management
        self.progress_interval = 10  # Progress update frequency


class ProgressTracker:
    """Track and display progress of batch operations"""
    
    def __init__(self, total_items: int, description: str = "Processing"):
        self.total_items = total_items
        self.completed_items = 0
        self.description = description
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, increment: int = 1):
        """Update progress counter"""
        self.completed_items += increment
        current_time = time.time()
        
        # Update every second or when completed
        if (current_time - self.last_update > 1.0) or (self.completed_items >= self.total_items):
            self.display_progress()
            self.last_update = current_time
    
    def display_progress(self):
        """Display current progress"""
        elapsed = time.time() - self.start_time
        percent = (self.completed_items / self.total_items) * 100
        
        if self.completed_items > 0:
            eta = (elapsed / self.completed_items) * (self.total_items - self.completed_items)
            eta_str = f", ETA: {eta:.1f}s"
        else:
            eta_str = ""
        
        logger.info(f"{self.description}: {self.completed_items}/{self.total_items} "
                   f"({percent:.1f}%, {elapsed:.1f}s elapsed{eta_str})")


class BatchGenerator:
    """Main class for batch time series generation"""
    
    def __init__(self, config: BatchGenerationConfig, 
                 base_url: str = "http://localhost:8080", 
                 api_key: Optional[str] = None):
        self.config = config
        self.base_url = base_url
        self.api_key = api_key
        
        # Initialize clients
        self.sync_client = TSIOTClient(
            base_url=base_url,
            api_key=api_key,
            timeout=30,
            max_retries=3
        )
        
        self.async_client = AsyncTSIOTClient(
            base_url=base_url,
            api_key=api_key,
            timeout=30,
            max_retries=3
        )
        
        # Initialize generators
        self.generators = self._create_generators()
        
        # Create output directory
        self.config.output_dir.mkdir(exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'total_generated': 0,
            'successful': 0,
            'failed': 0,
            'by_algorithm': {},
            'start_time': None,
            'end_time': None,
            'generation_times': [],
            'total_data_points': 0
        }
    
    def _create_generators(self) -> Dict[str, Any]:
        """Create generator instances for different algorithms"""
        return {
            'arima': ARIMAGenerator(
                ar_params=[0.5, -0.3, 0.1],
                ma_params=[0.2, -0.1],
                noise_std=0.5
            ),
            'lstm': LSTMGenerator(
                hidden_size=32,
                sequence_length=15,
                num_layers=2
            ),
            'statistical': StatisticalGenerator(
                distribution='normal',
                parameters={'mean': 0.0, 'std': 1.0}
            )
        }
    
    def _select_algorithm(self, index: int) -> str:
        """Select algorithm based on weights and index"""
        algorithms = list(self.config.algorithm_weights.keys())
        weights = list(self.config.algorithm_weights.values())
        
        # Use modulo for deterministic distribution
        weight_sum = sum(weights)
        normalized_weights = [w / weight_sum for w in weights]
        
        cumulative = 0
        position = (index % 100) / 100.0
        
        for i, weight in enumerate(normalized_weights):
            cumulative += weight
            if position < cumulative:
                return algorithms[i]
        
        return algorithms[-1]  # Fallback
    
    def _create_metadata(self, index: int, algorithm: str) -> TimeSeriesMetadata:
        """Create metadata for a time series"""
        return TimeSeriesMetadata(
            series_id=f"batch_{algorithm}_{index:04d}",
            name=f"Batch Generated {algorithm.upper()} Series {index}",
            description=f"Batch generated {algorithm} time series",
            frequency="1min",
            tags={
                "batch_id": "batch_demo",
                "algorithm": algorithm,
                "index": str(index),
                "generated_at": datetime.now().isoformat()
            }
        )
    
    def generate_single_series(self, index: int) -> Optional[TimeSeries]:
        """Generate a single time series"""
        try:
            # Select algorithm
            algorithm = self._select_algorithm(index)
            generator = self.generators[algorithm]
            
            # Create metadata
            metadata = self._create_metadata(index, algorithm)
            
            # Generate series
            start_time = time.time()
            
            ts = generator.generate(
                length=self.config.series_length,
                start_time=datetime.now() - timedelta(hours=24),
                metadata=metadata
            )
            
            generation_time = time.time() - start_time
            
            # Update statistics
            self.stats['generation_times'].append(generation_time)
            self.stats['total_data_points'] += len(ts)
            
            if algorithm not in self.stats['by_algorithm']:
                self.stats['by_algorithm'][algorithm] = 0
            self.stats['by_algorithm'][algorithm] += 1
            
            return ts
            
        except Exception as e:
            logger.error(f"Failed to generate series {index}: {e}")
            return None
    
    async def generate_single_series_async(self, index: int) -> Optional[TimeSeries]:
        """Generate a single time series asynchronously"""
        try:
            # Select algorithm and create request
            algorithm = self._select_algorithm(index)
            metadata = self._create_metadata(index, algorithm)
            
            # Create generation request
            request = {
                'type': algorithm,
                'length': self.config.series_length,
                'parameters': self._get_algorithm_parameters(algorithm),
                'metadata': metadata.to_dict()
            }
            
            # Generate via API
            start_time = time.time()
            ts = await self.async_client.generate(request)
            generation_time = time.time() - start_time
            
            # Update statistics
            self.stats['generation_times'].append(generation_time)
            self.stats['total_data_points'] += len(ts)
            
            if algorithm not in self.stats['by_algorithm']:
                self.stats['by_algorithm'][algorithm] = 0
            self.stats['by_algorithm'][algorithm] += 1
            
            return ts
            
        except Exception as e:
            logger.error(f"Failed to generate series {index} async: {e}")
            return None
    
    def _get_algorithm_parameters(self, algorithm: str) -> Dict[str, Any]:
        """Get parameters for algorithm"""
        if algorithm == 'arima':
            return {
                'ar_params': [0.5, -0.3, 0.1],
                'ma_params': [0.2, -0.1],
                'noise': 0.5
            }
        elif algorithm == 'lstm':
            return {
                'hidden_size': 32,
                'sequence_length': 15,
                'num_layers': 2
            }
        elif algorithm == 'statistical':
            return {
                'distribution': 'normal',
                'mean': 0.0,
                'std': 1.0
            }
        else:
            return {}
    
    def save_series(self, ts: TimeSeries, index: int) -> bool:
        """Save a time series to file"""
        try:
            if self.config.save_individual_files:
                filename = f"series_{index:04d}_{ts.metadata.series_id}.json"
                filepath = self.config.output_dir / filename
                
                with open(filepath, 'w') as f:
                    json.dump(ts.to_dict(), f, indent=2, default=str)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to save series {index}: {e}")
            return False
    
    def generate_batch_threaded(self) -> List[TimeSeries]:
        """Generate batch using threading"""
        logger.info(f"Starting threaded batch generation: {self.config.total_series} series")
        
        self.stats['start_time'] = time.time()
        progress = ProgressTracker(self.config.total_series, "Threaded Generation")
        
        successful_series = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.generate_single_series, i): i
                for i in range(self.config.total_series)
            }
            
            # Process completed tasks
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                
                try:
                    ts = future.result()
                    if ts is not None:
                        successful_series.append(ts)
                        self.save_series(ts, index)
                        self.stats['successful'] += 1
                    else:
                        self.stats['failed'] += 1
                        
                except Exception as e:
                    logger.error(f"Task {index} generated exception: {e}")
                    self.stats['failed'] += 1
                
                progress.update()
        
        self.stats['end_time'] = time.time()
        self.stats['total_generated'] = len(successful_series)
        
        return successful_series
    
    async def generate_batch_async(self) -> List[TimeSeries]:
        """Generate batch using async operations"""
        logger.info(f"Starting async batch generation: {self.config.total_series} series")
        
        self.stats['start_time'] = time.time()
        progress = ProgressTracker(self.config.total_series, "Async Generation")
        
        successful_series = []
        
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def generate_with_semaphore(index: int) -> Optional[TimeSeries]:
            async with semaphore:
                return await self.generate_single_series_async(index)
        
        # Create all tasks
        tasks = [generate_with_semaphore(i) for i in range(self.config.total_series)]
        
        # Process tasks in batches for memory management
        for i in range(0, len(tasks), self.config.batch_size):
            batch_tasks = tasks[i:i + self.config.batch_size]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for j, result in enumerate(batch_results):
                index = i + j
                
                if isinstance(result, Exception):
                    logger.error(f"Task {index} failed: {result}")
                    self.stats['failed'] += 1
                elif result is not None:
                    successful_series.append(result)
                    self.save_series(result, index)
                    self.stats['successful'] += 1
                else:
                    self.stats['failed'] += 1
                
                progress.update()
        
        self.stats['end_time'] = time.time()
        self.stats['total_generated'] = len(successful_series)
        
        return successful_series
    
    def generate_batch_sequential(self) -> List[TimeSeries]:
        """Generate batch sequentially (for comparison)"""
        logger.info(f"Starting sequential batch generation: {self.config.total_series} series")
        
        self.stats['start_time'] = time.time()
        progress = ProgressTracker(self.config.total_series, "Sequential Generation")
        
        successful_series = []
        
        for i in range(self.config.total_series):
            ts = self.generate_single_series(i)
            
            if ts is not None:
                successful_series.append(ts)
                self.save_series(ts, i)
                self.stats['successful'] += 1
            else:
                self.stats['failed'] += 1
            
            progress.update()
        
        self.stats['end_time'] = time.time()
        self.stats['total_generated'] = len(successful_series)
        
        return successful_series
    
    def save_summary(self, series_list: List[TimeSeries]):
        """Save batch generation summary"""
        if not self.config.save_summary:
            return
        
        try:
            # Calculate additional statistics
            total_duration = self.stats['end_time'] - self.stats['start_time']
            avg_generation_time = (
                sum(self.stats['generation_times']) / len(self.stats['generation_times'])
                if self.stats['generation_times'] else 0
            )
            
            throughput = self.stats['successful'] / total_duration if total_duration > 0 else 0
            
            summary = {
                'batch_config': {
                    'total_series': self.config.total_series,
                    'series_length': self.config.series_length,
                    'batch_size': self.config.batch_size,
                    'max_workers': self.config.max_workers,
                    'algorithm_weights': self.config.algorithm_weights
                },
                'results': {
                    'total_generated': self.stats['total_generated'],
                    'successful': self.stats['successful'],
                    'failed': self.stats['failed'],
                    'success_rate': (self.stats['successful'] / self.config.total_series) * 100,
                    'by_algorithm': self.stats['by_algorithm'],
                    'total_data_points': self.stats['total_data_points']
                },
                'performance': {
                    'total_duration': total_duration,
                    'avg_generation_time': avg_generation_time,
                    'throughput_series_per_sec': throughput,
                    'throughput_points_per_sec': self.stats['total_data_points'] / total_duration if total_duration > 0 else 0
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Save summary
            summary_path = self.config.output_dir / "batch_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Summary saved to {summary_path}")
            
            # Log key metrics
            logger.info(f"Batch Generation Summary:")
            logger.info(f"  Success Rate: {summary['results']['success_rate']:.1f}%")
            logger.info(f"  Total Duration: {total_duration:.2f}s")
            logger.info(f"  Throughput: {throughput:.2f} series/sec")
            logger.info(f"  Total Data Points: {self.stats['total_data_points']:,}")
            
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")
    
    def run_batch_generation(self, method: str = "async") -> List[TimeSeries]:
        """Run batch generation with specified method"""
        logger.info(f"Starting batch generation with method: {method}")
        
        # Reset statistics
        self.stats = {
            'total_generated': 0,
            'successful': 0,
            'failed': 0,
            'by_algorithm': {},
            'start_time': None,
            'end_time': None,
            'generation_times': [],
            'total_data_points': 0
        }
        
        # Run generation
        if method == "async":
            series_list = asyncio.run(self.generate_batch_async())
        elif method == "threaded":
            series_list = self.generate_batch_threaded()
        elif method == "sequential":
            series_list = self.generate_batch_sequential()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Save summary
        self.save_summary(series_list)
        
        return series_list


def compare_methods(config: BatchGenerationConfig) -> Dict[str, Any]:
    """Compare different batch generation methods"""
    logger.info("=== Comparing Batch Generation Methods ===")
    
    methods = ["sequential", "threaded", "async"]
    results = {}
    
    for method in methods:
        logger.info(f"\nTesting {method} method...")
        
        # Create fresh generator for each test
        generator = BatchGenerator(config)
        
        try:
            # Run with smaller batch for comparison
            original_total = config.total_series
            config.total_series = 20  # Smaller for comparison
            
            series_list = generator.run_batch_generation(method)
            
            results[method] = {
                'successful': generator.stats['successful'],
                'failed': generator.stats['failed'],
                'duration': generator.stats['end_time'] - generator.stats['start_time'],
                'throughput': generator.stats['successful'] / (generator.stats['end_time'] - generator.stats['start_time'])
            }
            
            # Restore original total
            config.total_series = original_total
            
        except Exception as e:
            logger.error(f"Method {method} failed: {e}")
            results[method] = {'error': str(e)}
    
    return results


def main():
    """Main function to run the batch generation demo"""
    try:
        # Configuration
        config = BatchGenerationConfig()
        
        # You can override configuration here
        config.total_series = 50
        config.series_length = 500
        config.max_workers = 4
        
        base_url = "http://localhost:8080"
        api_key = None  # Set if your service requires authentication
        
        logger.info("TSIOT Batch Generation Demo")
        logger.info("=" * 50)
        
        # Method comparison
        comparison_results = compare_methods(config)
        
        print("\nMethod Comparison Results:")
        print("-" * 30)
        for method, result in comparison_results.items():
            if 'error' in result:
                print(f"{method:12}: ERROR - {result['error']}")
            else:
                print(f"{method:12}: {result['successful']:2d} series in {result['duration']:.2f}s "
                      f"({result['throughput']:.2f} series/sec)")
        
        # Full batch generation
        logger.info(f"\nRunning full batch generation: {config.total_series} series")
        
        generator = BatchGenerator(config, base_url=base_url, api_key=api_key)
        series_list = generator.run_batch_generation("async")
        
        print(f"\nBatch generation completed!")
        print(f"Generated {len(series_list)} time series")
        print(f"Output saved to: {config.output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)