"""
Batch processing for multiple datasets
"""

import os
import time
import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import json
import pandas as pd
from enum import Enum

from .base import SyntheticDataConfig, GenerationResult, BaseSyntheticGenerator
from .factory import GeneratorFactory
from .progress import MultiProgressTracker, ProgressInfo, ProgressStage


class BatchStatus(Enum):
    """Status of batch job"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some succeeded, some failed
    CANCELLED = "cancelled"


@dataclass
class BatchDataset:
    """Information about a dataset in batch"""
    id: str
    input_path: str
    output_path: str
    config: Union[SyntheticDataConfig, Dict[str, Any]]
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result of batch processing"""
    job_id: str
    status: BatchStatus
    started_at: datetime
    completed_at: Optional[datetime]
    total_datasets: int
    successful: int
    failed: int
    results: Dict[str, Union[GenerationResult, Exception]]
    processing_time: float
    errors: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'job_id': self.job_id,
            'status': self.status.value,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'total_datasets': self.total_datasets,
            'successful': self.successful,
            'failed': self.failed,
            'processing_time': self.processing_time,
            'errors': self.errors
        }


class BatchProcessor:
    """Process multiple datasets in batch"""
    
    def __init__(self,
                 max_workers: int = 4,
                 use_async: bool = False,
                 cache_models: bool = True,
                 fail_fast: bool = False):
        self.max_workers = max_workers
        self.use_async = use_async
        self.cache_models = cache_models
        self.fail_fast = fail_fast
        self.progress_tracker = MultiProgressTracker()
        self._model_cache: Dict[str, BaseSyntheticGenerator] = {}
        self._cancelled = False
    
    def process_batch(self,
                     datasets: List[BatchDataset],
                     progress_callback: Optional[Callable[[str, ProgressInfo], None]] = None) -> BatchResult:
        """Process multiple datasets"""
        if progress_callback:
            self.progress_tracker.callback = progress_callback
        
        # Sort by priority
        datasets.sort(key=lambda x: x.priority, reverse=True)
        
        # Initialize result
        job_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        started_at = datetime.now()
        results = {}
        errors = {}
        
        try:
            if self.use_async:
                # Async processing
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results, errors = loop.run_until_complete(
                    self._process_async(datasets)
                )
            else:
                # Sync processing with thread pool
                results, errors = self._process_sync(datasets)
        
        except KeyboardInterrupt:
            self._cancelled = True
            self.progress_tracker.cancel_all()
        
        # Calculate final status
        completed_at = datetime.now()
        processing_time = (completed_at - started_at).total_seconds()
        
        successful = sum(1 for r in results.values() if isinstance(r, GenerationResult))
        failed = len(errors)
        
        if self._cancelled:
            status = BatchStatus.CANCELLED
        elif failed == 0:
            status = BatchStatus.COMPLETED
        elif successful == 0:
            status = BatchStatus.FAILED
        else:
            status = BatchStatus.PARTIAL
        
        return BatchResult(
            job_id=job_id,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            total_datasets=len(datasets),
            successful=successful,
            failed=failed,
            results=results,
            processing_time=processing_time,
            errors=errors
        )
    
    def _process_sync(self, datasets: List[BatchDataset]) -> Tuple[Dict, Dict]:
        """Process datasets synchronously with thread pool"""
        results = {}
        errors = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_dataset = {
                executor.submit(self._process_single_dataset, dataset): dataset
                for dataset in datasets
            }
            
            # Process completed tasks
            for future in concurrent.futures.as_completed(future_to_dataset):
                dataset = future_to_dataset[future]
                
                try:
                    result = future.result()
                    results[dataset.id] = result
                except Exception as e:
                    errors[dataset.id] = str(e)
                    
                    if self.fail_fast:
                        # Cancel remaining tasks
                        for f in future_to_dataset:
                            f.cancel()
                        break
                
                if self._cancelled:
                    break
        
        return results, errors
    
    async def _process_async(self, datasets: List[BatchDataset]) -> Tuple[Dict, Dict]:
        """Process datasets asynchronously"""
        results = {}
        errors = {}
        
        # Create tasks
        tasks = []
        for dataset in datasets:
            task = asyncio.create_task(
                self._process_single_dataset_async(dataset)
            )
            tasks.append((dataset, task))
        
        # Wait for completion
        for dataset, task in tasks:
            try:
                result = await task
                results[dataset.id] = result
            except Exception as e:
                errors[dataset.id] = str(e)
                
                if self.fail_fast:
                    # Cancel remaining tasks
                    for _, t in tasks:
                        if not t.done():
                            t.cancel()
                    break
        
        return results, errors
    
    def _process_single_dataset(self, dataset: BatchDataset) -> GenerationResult:
        """Process a single dataset"""
        # Create progress tracker for this dataset
        tracker = self.progress_tracker.create_tracker(dataset.id)
        
        try:
            # Load data
            tracker.set_stage(ProgressStage.LOADING_DATA, f"Loading {dataset.input_path}")
            data = self._load_data(dataset.input_path)
            tracker.update(1, 1)
            
            # Create configuration
            if isinstance(dataset.config, dict):
                config = SyntheticDataConfig(**dataset.config)
            else:
                config = dataset.config
            
            # Set progress callback for generator
            config.progress_callback = lambda info: tracker.update(
                info.current, info.total, info.message
            )
            
            # Get or create generator
            generator = self._get_generator(config)
            
            # Fit and generate
            tracker.set_stage(ProgressStage.TRAINING, "Training model")
            result = generator.fit_generate(data)
            
            # Save output
            tracker.set_stage(ProgressStage.SAVING, f"Saving to {dataset.output_path}")
            result.save(dataset.output_path)
            
            # Complete
            tracker.complete(f"Dataset {dataset.id} processed successfully")
            
            return result
            
        except Exception as e:
            tracker.fail(f"Error processing dataset {dataset.id}: {str(e)}")
            raise
        finally:
            # Clean up tracker after delay
            time.sleep(1)
            self.progress_tracker.remove_tracker(dataset.id)
    
    async def _process_single_dataset_async(self, dataset: BatchDataset) -> GenerationResult:
        """Process a single dataset asynchronously"""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._process_single_dataset, dataset)
    
    def _load_data(self, path: str) -> pd.DataFrame:
        """Load dataset from file"""
        path = Path(path)
        
        if path.suffix.lower() == '.csv':
            return pd.read_csv(path)
        elif path.suffix.lower() == '.parquet':
            return pd.read_parquet(path)
        elif path.suffix.lower() == '.json':
            return pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _get_generator(self, config: SyntheticDataConfig) -> BaseSyntheticGenerator:
        """Get or create generator with caching"""
        if not self.cache_models:
            return GeneratorFactory.create_generator(config)
        
        # Create cache key
        cache_key = f"{config.generator_type}_{config.model_type}"
        
        if cache_key not in self._model_cache:
            self._model_cache[cache_key] = GeneratorFactory.create_generator(config)
        
        return self._model_cache[cache_key]
    
    def cancel(self):
        """Cancel batch processing"""
        self._cancelled = True
        self.progress_tracker.cancel_all()


class BatchBuilder:
    """Builder for creating batch jobs"""
    
    def __init__(self):
        self.datasets: List[BatchDataset] = []
        self._default_config: Optional[SyntheticDataConfig] = None
        self._id_counter = 0
    
    def set_default_config(self, config: Union[SyntheticDataConfig, Dict[str, Any]]) -> 'BatchBuilder':
        """Set default configuration for datasets"""
        if isinstance(config, dict):
            self._default_config = SyntheticDataConfig(**config)
        else:
            self._default_config = config
        return self
    
    def add_dataset(self,
                   input_path: str,
                   output_path: str,
                   config: Optional[Union[SyntheticDataConfig, Dict[str, Any]]] = None,
                   dataset_id: Optional[str] = None,
                   priority: int = 0,
                   **metadata) -> 'BatchBuilder':
        """Add dataset to batch"""
        if dataset_id is None:
            dataset_id = f"dataset_{self._id_counter:04d}"
            self._id_counter += 1
        
        # Use provided config or default
        dataset_config = config or self._default_config
        if dataset_config is None:
            raise ValueError("No configuration provided and no default set")
        
        dataset = BatchDataset(
            id=dataset_id,
            input_path=input_path,
            output_path=output_path,
            config=dataset_config,
            priority=priority,
            metadata=metadata
        )
        
        self.datasets.append(dataset)
        return self
    
    def add_directory(self,
                     input_dir: str,
                     output_dir: str,
                     pattern: str = "*.csv",
                     config: Optional[Union[SyntheticDataConfig, Dict[str, Any]]] = None,
                     recursive: bool = False) -> 'BatchBuilder':
        """Add all matching files from directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Ensure output directory exists
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find matching files
        if recursive:
            files = input_path.rglob(pattern)
        else:
            files = input_path.glob(pattern)
        
        for file_path in files:
            # Create output path maintaining directory structure
            relative_path = file_path.relative_to(input_path)
            out_path = output_path / relative_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.add_dataset(
                input_path=str(file_path),
                output_path=str(out_path),
                config=config,
                dataset_id=file_path.stem
            )
        
        return self
    
    def build(self) -> List[BatchDataset]:
        """Build and return dataset list"""
        if not self.datasets:
            raise ValueError("No datasets added to batch")
        
        return self.datasets


def create_batch_config_file(output_path: str,
                           datasets: List[Dict[str, Any]],
                           default_config: Optional[Dict[str, Any]] = None):
    """Create a batch configuration file"""
    batch_config = {
        'version': '1.0',
        'created_at': datetime.now().isoformat(),
        'default_config': default_config or {
            'generator_type': 'sdv',
            'model_type': 'gaussian_copula',
            'num_samples': 1000
        },
        'datasets': datasets
    }
    
    with open(output_path, 'w') as f:
        json.dump(batch_config, f, indent=2)


def load_batch_from_config(config_path: str) -> List[BatchDataset]:
    """Load batch configuration from file"""
    with open(config_path, 'r') as f:
        batch_config = json.load(f)
    
    default_config = batch_config.get('default_config', {})
    datasets = []
    
    for dataset_info in batch_config['datasets']:
        # Merge with default config
        config = {**default_config, **dataset_info.get('config', {})}
        
        dataset = BatchDataset(
            id=dataset_info.get('id', f"dataset_{len(datasets)}"),
            input_path=dataset_info['input_path'],
            output_path=dataset_info['output_path'],
            config=config,
            priority=dataset_info.get('priority', 0),
            metadata=dataset_info.get('metadata', {})
        )
        datasets.append(dataset)
    
    return datasets