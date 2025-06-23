"""
Benchmark API endpoints
"""

import uuid
import tempfile
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import json

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, UploadFile, File
from pydantic import BaseModel, Field
import pandas as pd

from sdk.benchmark import GeneratorBenchmark, BenchmarkResult
from api.deps import get_current_user
from api.middleware.rate_limiter import rate_limit

router = APIRouter(prefix="/benchmark", tags=["benchmark"])

# Global storage for benchmark jobs
benchmark_jobs = {}


class BenchmarkRequest(BaseModel):
    """Request model for benchmark job"""
    generators: Optional[List[Dict[str, str]]] = Field(
        None, 
        description="List of generators to benchmark [{generator_type, model_type}]"
    )
    dataset_sizes: Optional[List[int]] = Field(
        None,
        description="Dataset sizes to test for scalability"
    )
    num_samples: Optional[int] = Field(
        None,
        description="Number of samples to generate"
    )
    test_scalability: bool = Field(
        False,
        description="Test scalability with different dataset sizes"
    )


class BenchmarkJobResponse(BaseModel):
    """Response model for benchmark job"""
    job_id: str
    status: str
    message: str


class BenchmarkJobStatus(BaseModel):
    """Status of benchmark job"""
    job_id: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    progress: int
    total: int
    current_task: str
    results_available: bool


class BenchmarkSummary(BaseModel):
    """Summary of benchmark results"""
    total_generators: int
    successful: int
    failed: int
    fastest_generator: Optional[Dict[str, Any]] = None
    most_memory_efficient: Optional[Dict[str, Any]] = None
    best_quality: Optional[Dict[str, Any]] = None
    average_metrics: Dict[str, float]


@router.post("/start", response_model=BenchmarkJobResponse)
@rate_limit(requests_per_minute=5)
async def start_benchmark(
    file: UploadFile = File(...),
    request: BenchmarkRequest = BenchmarkRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user = Depends(get_current_user)
):
    """Start a benchmark job with uploaded dataset"""
    
    # Validate file type
    if not file.filename.endswith(('.csv', '.json', '.parquet')):
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        file_path = tmp.name
    
    # Create job
    job_id = f"benchmark_{uuid.uuid4().hex[:8]}"
    
    # Initialize job
    benchmark_jobs[job_id] = {
        'status': 'pending',
        'started_at': datetime.now(),
        'completed_at': None,
        'progress': 0,
        'total': 0,
        'current_task': 'Initializing',
        'results': None,
        'error': None,
        'file_path': file_path,
        'dataset_name': Path(file.filename).stem
    }
    
    # Start background task
    background_tasks.add_task(
        run_benchmark_job,
        job_id,
        file_path,
        request,
        Path(file.filename).stem
    )
    
    return BenchmarkJobResponse(
        job_id=job_id,
        status="pending",
        message="Benchmark job started"
    )


async def run_benchmark_job(
    job_id: str,
    file_path: str,
    request: BenchmarkRequest,
    dataset_name: str
):
    """Run benchmark job in background"""
    
    try:
        # Update status
        benchmark_jobs[job_id]['status'] = 'running'
        
        # Load data
        benchmark_jobs[job_id]['current_task'] = 'Loading dataset'
        
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            data = pd.read_json(file_path)
        elif file_path.endswith('.parquet'):
            data = pd.read_parquet(file_path)
        
        # Create benchmark
        benchmark = GeneratorBenchmark()
        
        # Determine generators to test
        if request.generators:
            generators = [(g['generator_type'], g['model_type']) 
                         for g in request.generators]
        else:
            # Default generators
            generators = [
                ('sdv', 'gaussian_copula'),
                ('sdv', 'ctgan'),
                ('ctgan', 'ctgan'),
                ('ydata', 'wgan_gp')
            ]
        
        # Calculate total tasks
        if request.test_scalability and request.dataset_sizes:
            total_tasks = len(generators) * len(request.dataset_sizes)
        else:
            total_tasks = len(generators)
        
        benchmark_jobs[job_id]['total'] = total_tasks
        
        # Run benchmarks
        completed = 0
        
        if request.test_scalability and request.dataset_sizes:
            # Test scalability
            for gen_type, model_type in generators:
                benchmark_jobs[job_id]['current_task'] = f'Testing {gen_type}/{model_type} scalability'
                
                results = benchmark.benchmark_dataset_sizes(
                    data=data,
                    generator_type=gen_type,
                    model_type=model_type,
                    sizes=request.dataset_sizes,
                    dataset_name=dataset_name
                )
                
                completed += len(request.dataset_sizes)
                benchmark_jobs[job_id]['progress'] = completed
        else:
            # Standard benchmark
            for gen_type, model_type in generators:
                benchmark_jobs[job_id]['current_task'] = f'Benchmarking {gen_type}/{model_type}'
                
                benchmark.benchmark_generator(
                    data=data,
                    generator_type=gen_type,
                    model_type=model_type,
                    num_samples=request.num_samples,
                    dataset_name=dataset_name
                )
                
                completed += 1
                benchmark_jobs[job_id]['progress'] = completed
        
        # Save results
        benchmark_jobs[job_id]['current_task'] = 'Saving results'
        
        results_file = f"benchmark_results_{job_id}.json"
        benchmark.save_results(results_file)
        
        # Store results in job
        benchmark_jobs[job_id]['results'] = [r.to_dict() for r in benchmark.results]
        benchmark_jobs[job_id]['status'] = 'completed'
        benchmark_jobs[job_id]['completed_at'] = datetime.now()
        
    except Exception as e:
        benchmark_jobs[job_id]['status'] = 'failed'
        benchmark_jobs[job_id]['error'] = str(e)
        benchmark_jobs[job_id]['completed_at'] = datetime.now()
    
    finally:
        # Clean up temp file
        import os
        if os.path.exists(file_path):
            os.unlink(file_path)


@router.get("/jobs/{job_id}/status", response_model=BenchmarkJobStatus)
async def get_job_status(job_id: str, current_user = Depends(get_current_user)):
    """Get benchmark job status"""
    
    if job_id not in benchmark_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = benchmark_jobs[job_id]
    
    return BenchmarkJobStatus(
        job_id=job_id,
        status=job['status'],
        started_at=job['started_at'],
        completed_at=job.get('completed_at'),
        progress=job['progress'],
        total=job['total'],
        current_task=job['current_task'],
        results_available=job['status'] == 'completed' and job['results'] is not None
    )


@router.get("/jobs/{job_id}/results")
async def get_job_results(job_id: str, current_user = Depends(get_current_user)):
    """Get benchmark job results"""
    
    if job_id not in benchmark_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = benchmark_jobs[job_id]
    
    if job['status'] != 'completed':
        raise HTTPException(
            status_code=400, 
            detail=f"Job not completed. Current status: {job['status']}"
        )
    
    if not job['results']:
        raise HTTPException(status_code=404, detail="Results not found")
    
    # Calculate summary
    results = job['results']
    successful_results = [r for r in results if r['error'] is None]
    
    summary = BenchmarkSummary(
        total_generators=len(results),
        successful=len(successful_results),
        failed=len(results) - len(successful_results),
        average_metrics={}
    )
    
    if successful_results:
        # Find best performers
        fastest = min(successful_results, key=lambda r: r['total_time'])
        summary.fastest_generator = {
            'generator': f"{fastest['generator_type']}/{fastest['model_type']}",
            'time': fastest['total_time']
        }
        
        min_memory = min(successful_results, key=lambda r: r['peak_memory_mb'])
        summary.most_memory_efficient = {
            'generator': f"{min_memory['generator_type']}/{min_memory['model_type']}",
            'memory_mb': min_memory['peak_memory_mb']
        }
        
        best_quality = max(successful_results, key=lambda r: r['quality_score'])
        summary.best_quality = {
            'generator': f"{best_quality['generator_type']}/{best_quality['model_type']}",
            'quality_score': best_quality['quality_score']
        }
        
        # Calculate averages
        summary.average_metrics = {
            'fit_time': sum(r['fit_time'] for r in successful_results) / len(successful_results),
            'generate_time': sum(r['generate_time'] for r in successful_results) / len(successful_results),
            'total_time': sum(r['total_time'] for r in successful_results) / len(successful_results),
            'peak_memory_mb': sum(r['peak_memory_mb'] for r in successful_results) / len(successful_results),
            'quality_score': sum(r['quality_score'] for r in successful_results) / len(successful_results)
        }
    
    return {
        'job_id': job_id,
        'dataset_name': job['dataset_name'],
        'completed_at': job['completed_at'],
        'summary': summary,
        'results': results
    }


@router.get("/jobs/{job_id}/report")
async def get_job_report(
    job_id: str,
    format: str = "text",
    current_user = Depends(get_current_user)
):
    """Get formatted benchmark report"""
    
    if job_id not in benchmark_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = benchmark_jobs[job_id]
    
    if job['status'] != 'completed' or not job['results']:
        raise HTTPException(status_code=400, detail="Results not available")
    
    # Create benchmark object and load results
    benchmark = GeneratorBenchmark()
    
    # Convert results back to BenchmarkResult objects
    for result_data in job['results']:
        result = BenchmarkResult(
            generator_type=result_data['generator_type'],
            model_type=result_data['model_type'],
            dataset_name=result_data['dataset_name'],
            dataset_rows=result_data['dataset_rows'],
            dataset_columns=result_data['dataset_columns'],
            num_samples=result_data['num_samples'],
            fit_time=result_data['fit_time'],
            generate_time=result_data['generate_time'],
            total_time=result_data['total_time'],
            peak_memory_mb=result_data['peak_memory_mb'],
            memory_delta_mb=result_data['memory_delta_mb'],
            quality_score=result_data['quality_score'],
            basic_stats_score=result_data['basic_stats_score'],
            distribution_score=result_data['distribution_score'],
            correlation_score=result_data['correlation_score'],
            privacy_score=result_data['privacy_score'],
            utility_score=result_data['utility_score'],
            cpu_count=result_data['cpu_count'],
            cpu_usage_avg=result_data['cpu_usage_avg'],
            timestamp=datetime.fromisoformat(result_data['timestamp']),
            error=result_data['error'],
            config=result_data['config']
        )
        benchmark.results.append(result)
    
    # Generate report
    if format == "text":
        report = benchmark.generate_report()
        return {"report": report, "format": "text"}
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")


@router.post("/compare")
async def compare_generators(
    file: UploadFile = File(...),
    generators: List[Dict[str, str]] = None,
    num_samples: int = 1000,
    current_user = Depends(get_current_user)
):
    """Quick comparison of generators on uploaded dataset"""
    
    # Validate file
    if not file.filename.endswith(('.csv', '.json', '.parquet')):
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        file_path = tmp.name
    
    try:
        # Load data
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            data = pd.read_json(file_path)
        elif file_path.endswith('.parquet'):
            data = pd.read_parquet(file_path)
        
        # Default generators if not specified
        if not generators:
            generators = [
                {'generator_type': 'sdv', 'model_type': 'gaussian_copula'},
                {'generator_type': 'sdv', 'model_type': 'ctgan'},
                {'generator_type': 'ctgan', 'model_type': 'ctgan'}
            ]
        
        # Run quick benchmark
        benchmark = GeneratorBenchmark()
        results = []
        
        for gen in generators:
            result = benchmark.benchmark_generator(
                data=data,
                generator_type=gen['generator_type'],
                model_type=gen['model_type'],
                num_samples=num_samples,
                dataset_name=Path(file.filename).stem
            )
            results.append(result.to_dict())
        
        # Return comparison
        return {
            'dataset_info': {
                'name': Path(file.filename).stem,
                'rows': len(data),
                'columns': len(data.columns)
            },
            'results': results
        }
        
    finally:
        # Clean up temp file
        import os
        if os.path.exists(file_path):
            os.unlink(file_path)


@router.get("/templates")
async def get_benchmark_templates(current_user = Depends(get_current_user)):
    """Get benchmark configuration templates"""
    
    return {
        'quick_benchmark': {
            'generators': [
                {'generator_type': 'sdv', 'model_type': 'gaussian_copula'},
                {'generator_type': 'sdv', 'model_type': 'ctgan'},
                {'generator_type': 'ctgan', 'model_type': 'ctgan'}
            ],
            'num_samples': 1000
        },
        'full_benchmark': {
            'generators': [
                {'generator_type': 'sdv', 'model_type': 'gaussian_copula'},
                {'generator_type': 'sdv', 'model_type': 'ctgan'},
                {'generator_type': 'sdv', 'model_type': 'copula_gan'},
                {'generator_type': 'sdv', 'model_type': 'tvae'},
                {'generator_type': 'ctgan', 'model_type': 'ctgan'},
                {'generator_type': 'ctgan', 'model_type': 'tvae'},
                {'generator_type': 'ydata', 'model_type': 'wgan_gp'},
                {'generator_type': 'ydata', 'model_type': 'cramer_gan'},
                {'generator_type': 'ydata', 'model_type': 'dragan'}
            ]
        },
        'scalability_test': {
            'generators': [
                {'generator_type': 'sdv', 'model_type': 'gaussian_copula'},
                {'generator_type': 'sdv', 'model_type': 'ctgan'}
            ],
            'dataset_sizes': [100, 500, 1000, 5000, 10000],
            'test_scalability': True
        }
    }