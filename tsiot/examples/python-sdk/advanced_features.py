#!/usr/bin/env python3
"""
Advanced Features Example for TSIOT Python SDK

This example demonstrates advanced features of the TSIOT Python SDK including:
- Complex generator configurations
- Custom validation rules
- Advanced analytics
- Batch processing
- Concurrent operations
- Error handling and retry logic
"""

import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add the parent directory to the path to import the SDK
sys.path.insert(0, '../../internal/sdk/python')

from client import TSIOTClient, AsyncTSIOTClient
from timeseries import TimeSeries, DataPoint, TimeSeriesMetadata
from generators import ARIMAGenerator, LSTMGenerator, StatisticalGenerator
from validators import QualityValidator, StatisticalValidator, PrivacyValidator
from utils import TSIOTError, ValidationError, NetworkError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedFeaturesDemo:
    """
    Demonstrates advanced features of the TSIOT SDK
    """
    
    def __init__(self, base_url: str = "http://localhost:8080", api_key: str = None):
        """
        Initialize the demo with client configuration
        
        Args:
            base_url: TSIOT service URL
            api_key: API key for authentication
        """
        self.base_url = base_url
        self.api_key = api_key
        
        # Initialize clients
        self.sync_client = TSIOTClient(
            base_url=base_url,
            api_key=api_key,
            timeout=60,
            max_retries=5
        )
        
        self.async_client = AsyncTSIOTClient(
            base_url=base_url,
            api_key=api_key,
            timeout=60,
            max_retries=5
        )
        
        # Initialize generators
        self.generators = {
            'arima': ARIMAGenerator(
                ar_params=[0.7, -0.3, 0.1],
                ma_params=[0.4, -0.2],
                noise_std=0.5,
                trend='linear'
            ),
            'lstm': LSTMGenerator(
                hidden_size=64,
                sequence_length=20,
                num_layers=3,
                dropout=0.2
            ),
            'statistical': StatisticalGenerator(
                distribution='normal',
                parameters={'mean': 10.0, 'std': 2.0},
                seasonal_pattern=[1.2, 1.5, 1.8, 1.3, 0.9, 0.7, 0.8, 1.0, 1.1, 1.4, 1.6, 1.3]
            )
        }
        
        # Initialize validators
        self.validators = {
            'quality': QualityValidator(
                min_quality_score=0.8,
                max_missing_ratio=0.05,
                check_outliers=True
            ),
            'statistical': StatisticalValidator(
                normality_test=True,
                stationarity_test=True,
                autocorrelation_test=True
            ),
            'privacy': PrivacyValidator(
                k_anonymity=5,
                l_diversity=3,
                t_closeness=0.1
            )
        }

    def demonstrate_complex_generation(self) -> Dict[str, TimeSeries]:
        """
        Demonstrate complex time series generation with multiple algorithms
        
        Returns:
            Dictionary of generated time series by algorithm
        """
        logger.info("=== Complex Generation Demo ===")
        
        generated_series = {}
        
        for name, generator in self.generators.items():
            try:
                logger.info(f"Generating {name.upper()} time series...")
                
                # Create metadata
                metadata = TimeSeriesMetadata(
                    series_id=f"{name}_advanced_demo",
                    name=f"Advanced {name.upper()} Demo Series",
                    description=f"Complex {name} time series with advanced parameters",
                    frequency="1min",
                    tags={
                        "demo": "advanced_features",
                        "algorithm": name,
                        "complexity": "high"
                    }
                )
                
                # Generate series locally
                ts = generator.generate(
                    length=2000,
                    start_time=datetime.now() - timedelta(hours=33, minutes=20),
                    metadata=metadata
                )
                
                generated_series[name] = ts
                
                # Log statistics
                stats = ts.basic_statistics()
                logger.info(f"{name} series: {len(ts)} points, "
                           f"mean={stats['mean']:.3f}, std={stats['std']:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to generate {name} series: {e}")
        
        return generated_series

    def demonstrate_advanced_validation(self, series_dict: Dict[str, TimeSeries]) -> Dict[str, Dict]:
        """
        Demonstrate advanced validation with multiple validators
        
        Args:
            series_dict: Dictionary of time series to validate
            
        Returns:
            Dictionary of validation results
        """
        logger.info("=== Advanced Validation Demo ===")
        
        validation_results = {}
        
        for series_name, ts in series_dict.items():
            logger.info(f"Validating {series_name} series...")
            
            series_results = {}
            
            for validator_name, validator in self.validators.items():
                try:
                    logger.info(f"  Running {validator_name} validation...")
                    
                    # Run validation
                    result = validator.validate(ts)
                    series_results[validator_name] = result
                    
                    # Log key metrics
                    if 'quality_score' in result:
                        logger.info(f"    Quality score: {result['quality_score']:.3f}")
                    if 'is_stationary' in result:
                        logger.info(f"    Stationary: {result['is_stationary']}")
                    if 'privacy_score' in result:
                        logger.info(f"    Privacy score: {result['privacy_score']:.3f}")
                        
                except Exception as e:
                    logger.error(f"  {validator_name} validation failed: {e}")
                    series_results[validator_name] = {"error": str(e)}
            
            validation_results[series_name] = series_results
        
        return validation_results

    async def demonstrate_async_operations(self) -> Dict[str, Any]:
        """
        Demonstrate asynchronous operations and concurrent processing
        
        Returns:
            Dictionary of async operation results
        """
        logger.info("=== Async Operations Demo ===")
        
        results = {}
        
        try:
            # Test service connectivity
            logger.info("Testing async connectivity...")
            health = await self.async_client.health_check()
            results['health_check'] = health
            logger.info("Service health check successful")
            
            # Concurrent generation requests
            logger.info("Starting concurrent generation...")
            
            generation_requests = [
                {
                    'type': 'arima',
                    'length': 1000,
                    'parameters': {
                        'ar_params': [0.5, -0.2],
                        'ma_params': [0.3],
                        'trend': 'linear'
                    },
                    'metadata': {
                        'series_id': 'async_arima_1',
                        'name': 'Async ARIMA Series 1'
                    }
                },
                {
                    'type': 'lstm',
                    'length': 800,
                    'parameters': {
                        'hidden_size': 32,
                        'sequence_length': 10
                    },
                    'metadata': {
                        'series_id': 'async_lstm_1',
                        'name': 'Async LSTM Series 1'
                    }
                },
                {
                    'type': 'statistical',
                    'length': 1200,
                    'parameters': {
                        'distribution': 'normal',
                        'mean': 5.0,
                        'std': 1.5
                    },
                    'metadata': {
                        'series_id': 'async_statistical_1',
                        'name': 'Async Statistical Series 1'
                    }
                }
            ]
            
            # Execute concurrent requests
            start_time = time.time()
            
            tasks = [
                self.async_client.generate(request)
                for request in generation_requests
            ]
            
            generated_series = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            concurrent_duration = end_time - start_time
            
            # Process results
            successful_series = []
            for i, result in enumerate(generated_series):
                if isinstance(result, Exception):
                    logger.error(f"Generation {i+1} failed: {result}")
                else:
                    successful_series.append(result)
                    logger.info(f"Generated series {i+1}: {len(result)} points")
            
            results['concurrent_generation'] = {
                'duration': concurrent_duration,
                'successful_count': len(successful_series),
                'total_requests': len(generation_requests)
            }
            
            logger.info(f"Concurrent generation completed in {concurrent_duration:.2f}s")
            
            # Demonstrate batch validation
            if successful_series:
                logger.info("Starting batch validation...")
                
                validation_requests = []
                for ts in successful_series:
                    validation_requests.append({
                        'time_series': ts.to_dict(),
                        'validation_types': ['quality', 'statistical', 'privacy']
                    })
                
                validation_tasks = [
                    self.async_client.validate(req)
                    for req in validation_requests
                ]
                
                validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
                
                successful_validations = [
                    result for result in validation_results
                    if not isinstance(result, Exception)
                ]
                
                results['batch_validation'] = {
                    'successful_count': len(successful_validations),
                    'total_requests': len(validation_requests)
                }
                
                logger.info(f"Batch validation: {len(successful_validations)}/{len(validation_requests)} successful")
            
        except Exception as e:
            logger.error(f"Async operations failed: {e}")
            results['error'] = str(e)
        
        return results

    def demonstrate_error_handling(self) -> Dict[str, Any]:
        """
        Demonstrate comprehensive error handling and retry logic
        
        Returns:
            Dictionary of error handling results
        """
        logger.info("=== Error Handling Demo ===")
        
        results = {}
        
        # Test invalid requests
        logger.info("Testing invalid request handling...")
        
        try:
            # Invalid generator type
            invalid_request = {
                'type': 'invalid_generator',
                'length': 100
            }
            
            self.sync_client.generate(invalid_request)
            results['invalid_generator'] = "Should have failed"
            
        except ValidationError as e:
            logger.info(f"Correctly caught validation error: {e}")
            results['invalid_generator'] = "validation_error_caught"
        except Exception as e:
            logger.error(f"Unexpected error type: {e}")
            results['invalid_generator'] = f"unexpected_error: {type(e).__name__}"
        
        # Test network error handling
        logger.info("Testing network error handling...")
        
        try:
            # Create client with invalid URL
            invalid_client = TSIOTClient(
                base_url="http://invalid-url-that-does-not-exist:9999",
                timeout=5,
                max_retries=2
            )
            
            invalid_client.health_check()
            results['network_error'] = "Should have failed"
            
        except NetworkError as e:
            logger.info(f"Correctly caught network error: {e}")
            results['network_error'] = "network_error_caught"
        except Exception as e:
            logger.error(f"Unexpected error type: {e}")
            results['network_error'] = f"unexpected_error: {type(e).__name__}"
        
        # Test retry logic with timeout
        logger.info("Testing retry logic...")
        
        retry_client = TSIOTClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=1,  # Very short timeout
            max_retries=3
        )
        
        try:
            start_time = time.time()
            
            # This might succeed or fail depending on service response time
            retry_client.get_info()
            
            end_time = time.time()
            results['retry_logic'] = {
                'status': 'succeeded',
                'duration': end_time - start_time
            }
            
        except Exception as e:
            end_time = time.time()
            logger.info(f"Request failed after retries: {e}")
            results['retry_logic'] = {
                'status': 'failed_with_retries',
                'duration': end_time - start_time,
                'error': str(e)
            }
        
        return results

    def demonstrate_data_export_formats(self, ts: TimeSeries) -> Dict[str, Any]:
        """
        Demonstrate different data export formats
        
        Args:
            ts: Time series to export
            
        Returns:
            Dictionary of export results
        """
        logger.info("=== Data Export Demo ===")
        
        results = {}
        
        formats = ['json', 'csv', 'parquet']
        
        for fmt in formats:
            try:
                logger.info(f"Exporting to {fmt.upper()} format...")
                
                start_time = time.time()
                
                if fmt == 'json':
                    data = ts.to_json(indent=2)
                    size = len(data)
                elif fmt == 'csv':
                    data = ts.to_csv()
                    size = len(data)
                else:
                    # For parquet, we'll simulate the process
                    data = f"Simulated {fmt} export data"
                    size = len(data)
                
                end_time = time.time()
                
                results[fmt] = {
                    'success': True,
                    'size_bytes': size,
                    'duration': end_time - start_time
                }
                
                logger.info(f"{fmt.upper()} export: {size} bytes in {end_time - start_time:.3f}s")
                
            except Exception as e:
                logger.error(f"Failed to export to {fmt}: {e}")
                results[fmt] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results

    def run_all_demos(self) -> Dict[str, Any]:
        """
        Run all demonstration scenarios
        
        Returns:
            Complete results dictionary
        """
        logger.info("Starting TSIOT Advanced Features Demo")
        logger.info("=" * 50)
        
        all_results = {}
        
        try:
            # 1. Complex Generation
            generated_series = self.demonstrate_complex_generation()
            all_results['generation'] = {
                'series_count': len(generated_series),
                'algorithms': list(generated_series.keys())
            }
            
            # 2. Advanced Validation
            if generated_series:
                validation_results = self.demonstrate_advanced_validation(generated_series)
                all_results['validation'] = validation_results
                
                # 3. Data Export
                first_series = next(iter(generated_series.values()))
                export_results = self.demonstrate_data_export_formats(first_series)
                all_results['export'] = export_results
            
            # 4. Error Handling
            error_results = self.demonstrate_error_handling()
            all_results['error_handling'] = error_results
            
            # 5. Async Operations
            async_results = asyncio.run(self.demonstrate_async_operations())
            all_results['async_operations'] = async_results
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            all_results['error'] = str(e)
        
        return all_results


def main():
    """
    Main function to run the advanced features demo
    """
    try:
        # Configuration
        base_url = "http://localhost:8080"
        api_key = None  # Set if your service requires authentication
        
        # Create and run demo
        demo = AdvancedFeaturesDemo(base_url=base_url, api_key=api_key)
        results = demo.run_all_demos()
        
        # Print summary
        print("\n" + "=" * 50)
        print("ADVANCED FEATURES DEMO SUMMARY")
        print("=" * 50)
        
        for category, result in results.items():
            print(f"\n{category.upper().replace('_', ' ')}:")
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, (str, int, float, bool)):
                        print(f"  {key}: {value}")
                    elif isinstance(value, dict) and 'success' in value:
                        status = "" if value['success'] else ""
                        print(f"  {key}: {status}")
            else:
                print(f"  Result: {result}")
        
        print(f"\nDemo completed successfully!")
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