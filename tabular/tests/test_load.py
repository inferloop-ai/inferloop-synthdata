"""
Load testing for API endpoints
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiohttp
import requests
from pathlib import Path
import tempfile

import pytest
from locust import HttpUser, task, between, events
from locust.env import Environment
from locust.stats import StatsEntry


@dataclass
class LoadTestResult:
    """Results from load test"""
    endpoint: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    percentile_50: float
    percentile_95: float
    percentile_99: float
    requests_per_second: float
    error_rate: float
    errors: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'endpoint': self.endpoint,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'avg_response_time': self.avg_response_time,
            'min_response_time': self.min_response_time,
            'max_response_time': self.max_response_time,
            'percentile_50': self.percentile_50,
            'percentile_95': self.percentile_95,
            'percentile_99': self.percentile_99,
            'requests_per_second': self.requests_per_second,
            'error_rate': self.error_rate,
            'errors': self.errors
        }


class SyntheticDataAPIUser(HttpUser):
    """Locust user for load testing Synthetic Data API"""
    
    wait_time = between(1, 3)
    
    def on_start(self):
        """Setup before running tasks"""
        # Create test data file
        self.test_data = pd.DataFrame({
            'col1': np.random.rand(100),
            'col2': np.random.choice(['A', 'B', 'C'], 100),
            'col3': range(100)
        })
        
        # Upload test data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_data.to_csv(f, index=False)
            self.test_file_path = f.name
        
        # Upload and get file_id
        with open(self.test_file_path, 'rb') as f:
            response = self.client.post(
                "/data/upload",
                files={"file": ("test.csv", f, "text/csv")}
            )
            if response.status_code == 200:
                self.file_id = response.json()['file_id']
            else:
                self.file_id = None
    
    def on_stop(self):
        """Cleanup after tests"""
        import os
        if hasattr(self, 'test_file_path') and os.path.exists(self.test_file_path):
            os.unlink(self.test_file_path)
    
    @task(1)
    def health_check(self):
        """Test health endpoint"""
        self.client.get("/health")
    
    @task(2)
    def list_generators(self):
        """Test generators listing"""
        self.client.get("/generators")
    
    @task(5)
    def generate_sync_small(self):
        """Test synchronous generation with small dataset"""
        if not self.file_id:
            return
        
        request_data = {
            "config": {
                "generator_type": "sdv",
                "model_type": "gaussian_copula",
                "num_samples": 100
            },
            "validate_output": False
        }
        
        self.client.post(
            f"/generate/sync?file_id={self.file_id}",
            json=request_data,
            name="/generate/sync"
        )
    
    @task(3)
    def generate_async(self):
        """Test asynchronous generation"""
        if not self.file_id:
            return
        
        request_data = {
            "config": {
                "generator_type": "sdv",
                "model_type": "gaussian_copula",
                "num_samples": 1000
            }
        }
        
        response = self.client.post(
            f"/generate/async?file_id={self.file_id}",
            json=request_data,
            name="/generate/async"
        )
        
        if response.status_code == 200:
            job_id = response.json()['job_id']
            
            # Poll for completion (with timeout)
            for _ in range(10):
                status_response = self.client.get(
                    f"/jobs/{job_id}",
                    name="/jobs/[job_id]"
                )
                
                if status_response.status_code == 200:
                    status = status_response.json()['status']
                    if status in ['completed', 'failed']:
                        break
                
                time.sleep(1)
    
    @task(2)
    def profile_data(self):
        """Test data profiling"""
        if not self.file_id:
            return
        
        self.client.get(
            f"/profile/{self.file_id}",
            name="/profile/[file_id]"
        )
    
    @task(1)
    def config_templates(self):
        """Test configuration templates"""
        self.client.get("/config/templates")


class LoadTester:
    """Load testing utility for API endpoints"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[LoadTestResult] = []
    
    def run_concurrent_requests(self,
                              endpoint: str,
                              method: str = "GET",
                              num_requests: int = 100,
                              max_workers: int = 10,
                              **kwargs) -> LoadTestResult:
        """Run concurrent requests to an endpoint"""
        
        response_times = []
        errors = {}
        successful = 0
        
        start_time = time.time()
        
        def make_request():
            try:
                start = time.time()
                
                if method == "GET":
                    response = requests.get(f"{self.base_url}{endpoint}", **kwargs)
                elif method == "POST":
                    response = requests.post(f"{self.base_url}{endpoint}", **kwargs)
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                elapsed = time.time() - start
                response_times.append(elapsed)
                
                if response.status_code >= 200 and response.status_code < 300:
                    return True, None
                else:
                    error_key = f"HTTP_{response.status_code}"
                    return False, error_key
                    
            except Exception as e:
                error_key = type(e).__name__
                return False, error_key
        
        # Run requests concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            
            for future in as_completed(futures):
                success, error = future.result()
                if success:
                    successful += 1
                elif error:
                    errors[error] = errors.get(error, 0) + 1
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        if response_times:
            result = LoadTestResult(
                endpoint=endpoint,
                total_requests=num_requests,
                successful_requests=successful,
                failed_requests=num_requests - successful,
                avg_response_time=statistics.mean(response_times),
                min_response_time=min(response_times),
                max_response_time=max(response_times),
                percentile_50=np.percentile(response_times, 50),
                percentile_95=np.percentile(response_times, 95),
                percentile_99=np.percentile(response_times, 99),
                requests_per_second=num_requests / total_time,
                error_rate=(num_requests - successful) / num_requests,
                errors=errors
            )
        else:
            # All requests failed
            result = LoadTestResult(
                endpoint=endpoint,
                total_requests=num_requests,
                successful_requests=0,
                failed_requests=num_requests,
                avg_response_time=0,
                min_response_time=0,
                max_response_time=0,
                percentile_50=0,
                percentile_95=0,
                percentile_99=0,
                requests_per_second=num_requests / total_time,
                error_rate=1.0,
                errors=errors
            )
        
        self.results.append(result)
        return result
    
    async def run_async_requests(self,
                               endpoint: str,
                               method: str = "GET",
                               num_requests: int = 100,
                               **kwargs) -> LoadTestResult:
        """Run async requests to an endpoint"""
        
        response_times = []
        errors = {}
        successful = 0
        
        start_time = time.time()
        
        async def make_request(session):
            try:
                start = time.time()
                
                if method == "GET":
                    async with session.get(f"{self.base_url}{endpoint}", **kwargs) as response:
                        await response.text()
                        elapsed = time.time() - start
                        response_times.append(elapsed)
                        
                        if 200 <= response.status < 300:
                            return True, None
                        else:
                            return False, f"HTTP_{response.status}"
                            
                elif method == "POST":
                    async with session.post(f"{self.base_url}{endpoint}", **kwargs) as response:
                        await response.text()
                        elapsed = time.time() - start
                        response_times.append(elapsed)
                        
                        if 200 <= response.status < 300:
                            return True, None
                        else:
                            return False, f"HTTP_{response.status}"
                            
            except Exception as e:
                return False, type(e).__name__
        
        # Run requests asynchronously
        async with aiohttp.ClientSession() as session:
            tasks = [make_request(session) for _ in range(num_requests)]
            results = await asyncio.gather(*tasks)
            
            for success, error in results:
                if success:
                    successful += 1
                elif error:
                    errors[error] = errors.get(error, 0) + 1
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        if response_times:
            result = LoadTestResult(
                endpoint=endpoint,
                total_requests=num_requests,
                successful_requests=successful,
                failed_requests=num_requests - successful,
                avg_response_time=statistics.mean(response_times),
                min_response_time=min(response_times),
                max_response_time=max(response_times),
                percentile_50=np.percentile(response_times, 50),
                percentile_95=np.percentile(response_times, 95),
                percentile_99=np.percentile(response_times, 99),
                requests_per_second=num_requests / total_time,
                error_rate=(num_requests - successful) / num_requests,
                errors=errors
            )
        else:
            result = LoadTestResult(
                endpoint=endpoint,
                total_requests=num_requests,
                successful_requests=0,
                failed_requests=num_requests,
                avg_response_time=0,
                min_response_time=0,
                max_response_time=0,
                percentile_50=0,
                percentile_95=0,
                percentile_99=0,
                requests_per_second=num_requests / total_time,
                error_rate=1.0,
                errors=errors
            )
        
        self.results.append(result)
        return result
    
    def run_stress_test(self,
                       endpoint: str,
                       initial_load: int = 10,
                       max_load: int = 1000,
                       step: int = 10,
                       duration_per_step: int = 10) -> List[LoadTestResult]:
        """Run stress test with increasing load"""
        
        stress_results = []
        
        current_load = initial_load
        while current_load <= max_load:
            print(f"Testing with {current_load} concurrent requests...")
            
            # Run test for specified duration
            start_time = time.time()
            requests_sent = 0
            
            while time.time() - start_time < duration_per_step:
                result = self.run_concurrent_requests(
                    endpoint=endpoint,
                    num_requests=current_load,
                    max_workers=min(current_load, 50)
                )
                requests_sent += current_load
                
                # Check if system is failing
                if result.error_rate > 0.5:  # More than 50% errors
                    print(f"System failing at {current_load} concurrent requests")
                    stress_results.append(result)
                    return stress_results
            
            stress_results.append(result)
            current_load += step
        
        return stress_results
    
    def run_endurance_test(self,
                          endpoint: str,
                          load: int = 50,
                          duration_minutes: int = 60) -> List[LoadTestResult]:
        """Run endurance test with sustained load"""
        
        endurance_results = []
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        interval_minutes = 5
        next_checkpoint = start_time + (interval_minutes * 60)
        
        while time.time() < end_time:
            # Run load test
            result = self.run_concurrent_requests(
                endpoint=endpoint,
                num_requests=load,
                max_workers=min(load, 20)
            )
            
            # Check if it's time for checkpoint
            if time.time() >= next_checkpoint:
                endurance_results.append(result)
                elapsed_minutes = (time.time() - start_time) / 60
                print(f"Checkpoint at {elapsed_minutes:.1f} minutes: "
                      f"Avg response time: {result.avg_response_time:.3f}s, "
                      f"Error rate: {result.error_rate:.2%}")
                next_checkpoint += (interval_minutes * 60)
        
        return endurance_results
    
    def generate_report(self) -> str:
        """Generate load test report"""
        
        report = []
        report.append("=" * 80)
        report.append("LOAD TEST REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        for result in self.results:
            report.append(f"\nEndpoint: {result.endpoint}")
            report.append("-" * 40)
            report.append(f"Total Requests: {result.total_requests}")
            report.append(f"Successful: {result.successful_requests} ({result.successful_requests/result.total_requests*100:.1f}%)")
            report.append(f"Failed: {result.failed_requests} ({result.error_rate*100:.1f}%)")
            report.append(f"Requests/Second: {result.requests_per_second:.1f}")
            report.append(f"\nResponse Times:")
            report.append(f"  Average: {result.avg_response_time*1000:.1f} ms")
            report.append(f"  Min: {result.min_response_time*1000:.1f} ms")
            report.append(f"  Max: {result.max_response_time*1000:.1f} ms")
            report.append(f"  50th percentile: {result.percentile_50*1000:.1f} ms")
            report.append(f"  95th percentile: {result.percentile_95*1000:.1f} ms")
            report.append(f"  99th percentile: {result.percentile_99*1000:.1f} ms")
            
            if result.errors:
                report.append(f"\nErrors:")
                for error, count in result.errors.items():
                    report.append(f"  {error}: {count}")
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)


# Test functions for pytest
class TestLoadAPI:
    """Load tests for API endpoints"""
    
    @pytest.fixture
    def load_tester(self):
        """Create load tester instance"""
        return LoadTester()
    
    @pytest.fixture
    def test_file_id(self):
        """Upload test file and return file_id"""
        # Create test data
        df = pd.DataFrame({
            'col1': np.random.rand(100),
            'col2': np.random.choice(['A', 'B', 'C'], 100),
            'col3': range(100)
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f, index=False)
            temp_file = f.name
        
        # Upload file
        try:
            with open(temp_file, 'rb') as f:
                response = requests.post(
                    "http://localhost:8000/data/upload",
                    files={"file": ("test.csv", f, "text/csv")}
                )
                
            if response.status_code == 200:
                yield response.json()['file_id']
            else:
                yield None
        finally:
            import os
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_health_endpoint_load(self, load_tester):
        """Test health endpoint under load"""
        result = load_tester.run_concurrent_requests(
            endpoint="/health",
            num_requests=1000,
            max_workers=50
        )
        
        assert result.successful_requests > 950  # Allow 5% failure
        assert result.avg_response_time < 0.1  # Should respond in < 100ms
        assert result.percentile_95 < 0.2  # 95% should respond in < 200ms
    
    def test_generators_endpoint_load(self, load_tester):
        """Test generators listing under load"""
        result = load_tester.run_concurrent_requests(
            endpoint="/generators",
            num_requests=500,
            max_workers=25
        )
        
        assert result.error_rate < 0.05  # Less than 5% errors
        assert result.avg_response_time < 0.2
    
    @pytest.mark.asyncio
    async def test_async_generation_load(self, load_tester):
        """Test async generation endpoints"""
        # This would need a pre-uploaded file_id
        # Skipping actual test as it requires running API
        pass
    
    def test_stress_health_endpoint(self, load_tester):
        """Stress test health endpoint"""
        results = load_tester.run_stress_test(
            endpoint="/health",
            initial_load=10,
            max_load=500,
            step=50,
            duration_per_step=5
        )
        
        # Find breaking point
        breaking_point = None
        for result in results:
            if result.error_rate > 0.1:  # More than 10% errors
                breaking_point = result.total_requests
                break
        
        # Should handle at least 200 concurrent requests
        assert breaking_point is None or breaking_point > 200
    
    @pytest.mark.slow
    def test_endurance(self, load_tester):
        """Test API endurance with sustained load"""
        results = load_tester.run_endurance_test(
            endpoint="/health",
            load=50,
            duration_minutes=10  # Shorter for testing
        )
        
        # Response times should not degrade significantly
        if len(results) >= 2:
            initial_avg = results[0].avg_response_time
            final_avg = results[-1].avg_response_time
            
            # Should not degrade by more than 50%
            assert final_avg < initial_avg * 1.5


def run_locust_test(host: str = "http://localhost:8000",
                   users: int = 100,
                   spawn_rate: int = 10,
                   run_time: int = 60):
    """Run Locust load test programmatically"""
    
    # Setup Locust environment
    env = Environment(user_classes=[SyntheticDataAPIUser])
    env.create_local_runner()
    
    # Start test
    env.runner.start(users, spawn_rate=spawn_rate)
    
    # Run for specified time
    time.sleep(run_time)
    
    # Stop test
    env.runner.quit()
    
    # Get statistics
    stats = env.stats
    
    print("\nLocust Test Results:")
    print("-" * 80)
    
    for stat in stats.entries.values():
        print(f"\n{stat.name}:")
        print(f"  Requests: {stat.num_requests}")
        print(f"  Failures: {stat.num_failures}")
        print(f"  Avg Response Time: {stat.avg_response_time:.1f} ms")
        print(f"  Min Response Time: {stat.min_response_time:.1f} ms")
        print(f"  Max Response Time: {stat.max_response_time:.1f} ms")
        print(f"  RPS: {stat.current_rps:.1f}")


if __name__ == "__main__":
    # Run basic load test
    tester = LoadTester()
    
    print("Running load tests...")
    
    # Test various endpoints
    endpoints = [
        ("/health", "GET"),
        ("/generators", "GET"),
        ("/config/templates", "GET")
    ]
    
    for endpoint, method in endpoints:
        print(f"\nTesting {endpoint}...")
        result = tester.run_concurrent_requests(
            endpoint=endpoint,
            method=method,
            num_requests=100,
            max_workers=10
        )
        
        print(f"  Success rate: {(1 - result.error_rate) * 100:.1f}%")
        print(f"  Avg response time: {result.avg_response_time * 1000:.1f} ms")
        print(f"  Requests/second: {result.requests_per_second:.1f}")
    
    # Generate report
    report = tester.generate_report()
    print("\n" + report)
    
    # Save report
    with open("load_test_report.txt", "w") as f:
        f.write(report)