#!/usr/bin/env python3
"""
Streaming Data Example for TSIOT Python SDK

This example demonstrates real-time streaming capabilities of the TSIOT Python SDK including:
- Real-time data generation and consumption
- Streaming validation
- Live analytics
- WebSocket/SSE integration
- Data buffering and windowing
- Real-time visualization (simulation)
"""

import asyncio
import logging
import sys
import time
import json
import queue
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, AsyncGenerator, Callable
from collections import deque
import statistics

# Add the parent directory to the path to import the SDK
sys.path.insert(0, '../../internal/sdk/python')

from client import TSIOTClient, AsyncTSIOTClient
from timeseries import TimeSeries, DataPoint, TimeSeriesMetadata
from generators import ARIMAGenerator, LSTMGenerator, StatisticalGenerator
from validators import QualityValidator, StatisticalValidator
from utils import TSIOTError, ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StreamingBuffer:
    """Buffer for managing streaming time series data"""
    
    def __init__(self, max_size: int = 1000, window_size: int = 100):
        self.max_size = max_size
        self.window_size = window_size
        self.buffer = deque(maxlen=max_size)
        self.windows = deque(maxlen=10)  # Keep last 10 windows
        self.lock = threading.Lock()
        
        # Statistics
        self.total_points = 0
        self.last_update = None
        
    def add_point(self, point: DataPoint):
        """Add a data point to the buffer"""
        with self.lock:
            self.buffer.append(point)
            self.total_points += 1
            self.last_update = datetime.now()
            
            # Create windows for analysis
            if len(self.buffer) >= self.window_size:
                window_data = list(self.buffer)[-self.window_size:]
                self.windows.append(window_data)
    
    def get_latest_window(self) -> Optional[List[DataPoint]]:
        """Get the latest window of data"""
        with self.lock:
            if self.windows:
                return list(self.windows[-1])
            return None
    
    def get_current_buffer(self) -> List[DataPoint]:
        """Get current buffer contents"""
        with self.lock:
            return list(self.buffer)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self.lock:
            if not self.buffer:
                return {}
            
            values = [p.value for p in self.buffer]
            
            return {
                'count': len(self.buffer),
                'total_points': self.total_points,
                'mean': statistics.mean(values) if values else 0,
                'stdev': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values) if values else 0,
                'max': max(values) if values else 0,
                'last_update': self.last_update.isoformat() if self.last_update else None
            }


class StreamingGenerator:
    """Generate streaming time series data"""
    
    def __init__(self, generator_type: str = 'arima', interval: float = 1.0):
        self.generator_type = generator_type
        self.interval = interval
        self.is_running = False
        
        # Initialize generator
        if generator_type == 'arima':
            self.generator = ARIMAGenerator(
                ar_params=[0.7, -0.2],
                ma_params=[0.3],
                noise_std=0.5
            )
        elif generator_type == 'lstm':
            self.generator = LSTMGenerator(
                hidden_size=32,
                sequence_length=10
            )
        elif generator_type == 'statistical':
            self.generator = StatisticalGenerator(
                distribution='normal',
                parameters={'mean': 0.0, 'std': 1.0}
            )
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")
        
        # Internal state for continuous generation
        self.previous_values = deque(maxlen=10)
        self.time_index = 0
    
    def _generate_next_point(self) -> DataPoint:
        """Generate the next data point in the sequence"""
        current_time = datetime.now()
        
        # For simplicity, we'll generate based on previous values and some randomness
        if self.generator_type == 'arima':
            # Simple ARIMA-like generation
            value = 0.0
            if len(self.previous_values) > 0:
                value += 0.7 * self.previous_values[-1]
            if len(self.previous_values) > 1:
                value -= 0.2 * self.previous_values[-2]
            
            # Add noise
            import random
            value += random.gauss(0, 0.5)
            
        elif self.generator_type == 'lstm':
            # Simple pattern-based generation
            pattern_value = 5 * (self.time_index % 20) / 20.0
            noise = random.gauss(0, 0.2)
            value = pattern_value + noise
            
        else:  # statistical
            value = random.gauss(0, 1.0)
        
        # Add trend or seasonality
        trend = 0.01 * self.time_index
        seasonal = 2.0 * (self.time_index % 100) / 100.0
        value += trend + seasonal
        
        point = DataPoint(
            timestamp=current_time,
            value=value,
            quality=random.uniform(0.8, 1.0),
            metadata={'generator': self.generator_type, 'index': self.time_index}
        )
        
        self.previous_values.append(value)
        self.time_index += 1
        
        return point
    
    async def stream_data(self, duration_seconds: int) -> AsyncGenerator[DataPoint, None]:
        """Stream data points for specified duration"""
        self.is_running = True
        start_time = time.time()
        
        logger.info(f"Starting {self.generator_type} streaming for {duration_seconds}s")
        
        try:
            while self.is_running and (time.time() - start_time) < duration_seconds:
                point = self._generate_next_point()
                yield point
                await asyncio.sleep(self.interval)
                
        except Exception as e:
            logger.error(f"Streaming error: {e}")
        finally:
            self.is_running = False
            logger.info(f"Streaming stopped after {time.time() - start_time:.2f}s")
    
    def stop(self):
        """Stop streaming"""
        self.is_running = False


class StreamingValidator:
    """Validate streaming data in real-time"""
    
    def __init__(self, buffer: StreamingBuffer):
        self.buffer = buffer
        self.quality_validator = QualityValidator(min_quality_score=0.7)
        self.statistical_validator = StatisticalValidator()
        
        # Validation state
        self.validation_history = deque(maxlen=100)
        self.alerts = deque(maxlen=50)
    
    async def validate_stream(self, check_interval: int = 10):
        """Continuously validate streaming data"""
        logger.info("Starting streaming validation")
        
        while True:
            try:
                await asyncio.sleep(check_interval)
                
                # Get latest window
                window_data = self.buffer.get_latest_window()
                if not window_data or len(window_data) < 10:
                    continue
                
                # Create temporary time series for validation
                metadata = TimeSeriesMetadata(
                    series_id='streaming_validation',
                    name='Streaming Validation Window'
                )
                ts = TimeSeries(data_points=window_data, metadata=metadata)
                
                # Run validations
                validation_result = {
                    'timestamp': datetime.now(),
                    'window_size': len(window_data),
                    'quality': {},
                    'statistical': {},
                    'alerts': []
                }
                
                # Quality validation
                try:
                    quality_result = self.quality_validator.validate(ts)
                    validation_result['quality'] = quality_result
                    
                    if quality_result.get('quality_score', 1.0) < 0.8:
                        alert = {
                            'type': 'quality',
                            'message': f"Low quality score: {quality_result['quality_score']:.3f}",
                            'timestamp': datetime.now()
                        }
                        validation_result['alerts'].append(alert)
                        self.alerts.append(alert)
                        
                except Exception as e:
                    logger.warning(f"Quality validation failed: {e}")
                
                # Statistical validation
                try:
                    stats_result = self.statistical_validator.validate(ts)
                    validation_result['statistical'] = stats_result
                    
                    # Check for anomalies
                    if not stats_result.get('is_stationary', True):
                        alert = {
                            'type': 'statistical',
                            'message': "Non-stationary data detected",
                            'timestamp': datetime.now()
                        }
                        validation_result['alerts'].append(alert)
                        self.alerts.append(alert)
                        
                except Exception as e:
                    logger.warning(f"Statistical validation failed: {e}")
                
                # Store validation result
                self.validation_history.append(validation_result)
                
                # Log alerts
                for alert in validation_result['alerts']:
                    logger.warning(f"ALERT: {alert['message']}")
                
                # Log periodic summary
                if len(self.validation_history) % 6 == 0:  # Every minute if check_interval=10
                    self._log_validation_summary()
                    
            except Exception as e:
                logger.error(f"Validation loop error: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    def _log_validation_summary(self):
        """Log validation summary"""
        if not self.validation_history:
            return
        
        recent_validations = list(self.validation_history)[-6:]  # Last 6 validations
        
        quality_scores = []
        alert_count = 0
        
        for validation in recent_validations:
            if 'quality' in validation and 'quality_score' in validation['quality']:
                quality_scores.append(validation['quality']['quality_score'])
            alert_count += len(validation.get('alerts', []))
        
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0
        
        logger.info(f"Validation Summary (last {len(recent_validations)} windows): "
                   f"Avg Quality: {avg_quality:.3f}, Alerts: {alert_count}")
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        return list(self.alerts)[-count:]


class StreamingAnalyzer:
    """Perform real-time analytics on streaming data"""
    
    def __init__(self, buffer: StreamingBuffer):
        self.buffer = buffer
        self.analysis_history = deque(maxlen=100)
    
    async def analyze_stream(self, analysis_interval: int = 5):
        """Continuously analyze streaming data"""
        logger.info("Starting streaming analytics")
        
        while True:
            try:
                await asyncio.sleep(analysis_interval)
                
                # Get buffer statistics
                buffer_stats = self.buffer.get_statistics()
                if buffer_stats.get('count', 0) < 10:
                    continue
                
                # Get current buffer data
                current_data = self.buffer.get_current_buffer()
                if len(current_data) < 10:
                    continue
                
                # Perform analysis
                analysis_result = {
                    'timestamp': datetime.now(),
                    'buffer_stats': buffer_stats,
                    'trend_analysis': self._analyze_trend(current_data),
                    'anomaly_detection': self._detect_anomalies(current_data),
                    'pattern_analysis': self._analyze_patterns(current_data)
                }
                
                self.analysis_history.append(analysis_result)
                
                # Log interesting findings
                if analysis_result['anomaly_detection']['anomaly_count'] > 0:
                    logger.info(f"Detected {analysis_result['anomaly_detection']['anomaly_count']} anomalies")
                
                # Log periodic summary
                if len(self.analysis_history) % 12 == 0:  # Every minute if interval=5
                    self._log_analysis_summary()
                    
            except Exception as e:
                logger.error(f"Analysis loop error: {e}")
                await asyncio.sleep(5)
    
    def _analyze_trend(self, data_points: List[DataPoint]) -> Dict[str, Any]:
        """Analyze trend in the data"""
        if len(data_points) < 10:
            return {'trend': 'insufficient_data'}
        
        values = [p.value for p in data_points[-50:]]  # Last 50 points
        
        # Simple linear regression
        n = len(values)
        x = list(range(n))
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(xi * yi for xi, yi in zip(x, values))
        sum_x2 = sum(xi * xi for xi in x)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            slope = 0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        trend_direction = 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable'
        
        return {
            'trend': trend_direction,
            'slope': slope,
            'strength': abs(slope)
        }
    
    def _detect_anomalies(self, data_points: List[DataPoint]) -> Dict[str, Any]:
        """Detect anomalies in the data"""
        if len(data_points) < 20:
            return {'anomaly_count': 0, 'anomalies': []}
        
        values = [p.value for p in data_points]
        
        # Simple z-score based anomaly detection
        mean_val = statistics.mean(values)
        stdev_val = statistics.stdev(values) if len(values) > 1 else 1.0
        
        anomalies = []
        threshold = 2.5  # z-score threshold
        
        for i, point in enumerate(data_points):
            if stdev_val > 0:
                z_score = abs((point.value - mean_val) / stdev_val)
                if z_score > threshold:
                    anomalies.append({
                        'index': i,
                        'value': point.value,
                        'z_score': z_score,
                        'timestamp': point.timestamp.isoformat()
                    })
        
        return {
            'anomaly_count': len(anomalies),
            'anomalies': anomalies[-10:],  # Last 10 anomalies
            'threshold': threshold
        }
    
    def _analyze_patterns(self, data_points: List[DataPoint]) -> Dict[str, Any]:
        """Analyze patterns in the data"""
        if len(data_points) < 20:
            return {'pattern': 'insufficient_data'}
        
        values = [p.value for p in data_points[-100:]]  # Last 100 points
        
        # Simple autocorrelation analysis
        def autocorrelation(data, lag):
            n = len(data)
            if lag >= n:
                return 0
            
            mean_val = statistics.mean(data)
            numerator = sum((data[i] - mean_val) * (data[i + lag] - mean_val) 
                          for i in range(n - lag))
            denominator = sum((x - mean_val) ** 2 for x in data)
            
            return numerator / denominator if denominator > 0 else 0
        
        # Check for periodicity
        max_lag = min(20, len(values) // 2)
        autocorrs = [autocorrelation(values, lag) for lag in range(1, max_lag + 1)]
        
        if autocorrs:
            max_autocorr = max(autocorrs)
            best_lag = autocorrs.index(max_autocorr) + 1
        else:
            max_autocorr = 0
            best_lag = 0
        
        pattern_type = 'periodic' if max_autocorr > 0.3 else 'random'
        
        return {
            'pattern': pattern_type,
            'best_lag': best_lag,
            'max_autocorrelation': max_autocorr,
            'periodicity_strength': max_autocorr
        }
    
    def _log_analysis_summary(self):
        """Log analysis summary"""
        if not self.analysis_history:
            return
        
        recent_analyses = list(self.analysis_history)[-12:]  # Last 12 analyses
        
        # Aggregate trends
        trend_counts = {'increasing': 0, 'decreasing': 0, 'stable': 0}
        total_anomalies = 0
        
        for analysis in recent_analyses:
            trend = analysis.get('trend_analysis', {}).get('trend', 'unknown')
            if trend in trend_counts:
                trend_counts[trend] += 1
            
            total_anomalies += analysis.get('anomaly_detection', {}).get('anomaly_count', 0)
        
        dominant_trend = max(trend_counts, key=trend_counts.get)
        
        logger.info(f"Analysis Summary (last {len(recent_analyses)} windows): "
                   f"Trend: {dominant_trend}, Total Anomalies: {total_anomalies}")
    
    def get_latest_analysis(self) -> Optional[Dict[str, Any]]:
        """Get the latest analysis result"""
        return self.analysis_history[-1] if self.analysis_history else None


class StreamingDemo:
    """Main demo class for streaming functionality"""
    
    def __init__(self, base_url: str = "http://localhost:8080", api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key
        
        # Initialize buffer
        self.buffer = StreamingBuffer(max_size=2000, window_size=50)
        
        # Initialize components
        self.generator = StreamingGenerator('arima', interval=0.5)
        self.validator = StreamingValidator(self.buffer)
        self.analyzer = StreamingAnalyzer(self.buffer)
        
        # Control flags
        self.is_running = False
    
    async def run_streaming_demo(self, duration_seconds: int = 60):
        """Run the complete streaming demo"""
        logger.info("=== TSIOT Streaming Data Demo ===")
        logger.info(f"Running for {duration_seconds} seconds")
        
        self.is_running = True
        
        try:
            # Start background tasks
            tasks = [
                asyncio.create_task(self._consume_stream(duration_seconds)),
                asyncio.create_task(self.validator.validate_stream()),
                asyncio.create_task(self.analyzer.analyze_stream()),
                asyncio.create_task(self._periodic_summary())
            ]
            
            # Wait for streaming to complete
            await asyncio.wait_for(tasks[0], timeout=duration_seconds + 10)
            
            # Cancel other tasks
            for task in tasks[1:]:
                task.cancel()
            
            # Wait a bit for cleanup
            await asyncio.sleep(1)
            
            # Final summary
            self._log_final_summary()
            
        except asyncio.TimeoutError:
            logger.info("Demo completed (timeout)")
        except Exception as e:
            logger.error(f"Demo failed: {e}")
        finally:
            self.is_running = False
    
    async def _consume_stream(self, duration_seconds: int):
        """Consume streaming data and add to buffer"""
        async for point in self.generator.stream_data(duration_seconds):
            if not self.is_running:
                break
            
            self.buffer.add_point(point)
            
            # Log every 100 points
            if self.buffer.total_points % 100 == 0:
                logger.debug(f"Consumed {self.buffer.total_points} data points")
    
    async def _periodic_summary(self):
        """Provide periodic summaries"""
        while self.is_running:
            await asyncio.sleep(15)  # Every 15 seconds
            
            if not self.is_running:
                break
            
            # Buffer stats
            buffer_stats = self.buffer.get_statistics()
            
            # Recent alerts
            recent_alerts = self.validator.get_recent_alerts(5)
            
            # Latest analysis
            latest_analysis = self.analyzer.get_latest_analysis()
            
            logger.info("=== PERIODIC SUMMARY ===")
            logger.info(f"Buffer: {buffer_stats.get('count', 0)} points, "
                       f"Mean: {buffer_stats.get('mean', 0):.3f}")
            
            if recent_alerts:
                logger.info(f"Recent Alerts: {len(recent_alerts)}")
            
            if latest_analysis:
                trend = latest_analysis.get('trend_analysis', {}).get('trend', 'unknown')
                anomalies = latest_analysis.get('anomaly_detection', {}).get('anomaly_count', 0)
                logger.info(f"Trend: {trend}, Anomalies: {anomalies}")
    
    def _log_final_summary(self):
        """Log final demo summary"""
        buffer_stats = self.buffer.get_statistics()
        all_alerts = self.validator.get_recent_alerts(100)
        latest_analysis = self.analyzer.get_latest_analysis()
        
        logger.info("=== FINAL SUMMARY ===")
        logger.info(f"Total Points Generated: {buffer_stats.get('total_points', 0)}")
        logger.info(f"Final Buffer Size: {buffer_stats.get('count', 0)}")
        logger.info(f"Mean Value: {buffer_stats.get('mean', 0):.3f}")
        logger.info(f"Std Dev: {buffer_stats.get('stdev', 0):.3f}")
        logger.info(f"Total Alerts: {len(all_alerts)}")
        
        if latest_analysis:
            trend = latest_analysis.get('trend_analysis', {}).get('trend', 'unknown')
            pattern = latest_analysis.get('pattern_analysis', {}).get('pattern', 'unknown')
            logger.info(f"Final Trend: {trend}")
            logger.info(f"Pattern Type: {pattern}")


async def demo_multiple_streams():
    """Demonstrate multiple concurrent streams"""
    logger.info("=== Multiple Streams Demo ===")
    
    # Create multiple generators
    generators = [
        StreamingGenerator('arima', interval=0.3),
        StreamingGenerator('lstm', interval=0.4),
        StreamingGenerator('statistical', interval=0.2)
    ]
    
    # Create separate buffers
    buffers = [
        StreamingBuffer(max_size=500, window_size=30) for _ in generators
    ]
    
    async def consume_stream(gen, buf, stream_id, duration):
        count = 0
        async for point in gen.stream_data(duration):
            buf.add_point(point)
            count += 1
            if count % 50 == 0:
                logger.info(f"Stream {stream_id}: {count} points")
    
    # Run streams concurrently
    duration = 30
    tasks = [
        asyncio.create_task(consume_stream(gen, buf, i, duration))
        for i, (gen, buf) in enumerate(zip(generators, buffers))
    ]
    
    await asyncio.gather(*tasks)
    
    # Summary
    for i, buf in enumerate(buffers):
        stats = buf.get_statistics()
        logger.info(f"Stream {i}: {stats.get('total_points', 0)} points, "
                   f"mean={stats.get('mean', 0):.3f}")


def main():
    """Main function to run the streaming demo"""
    try:
        # Configuration
        base_url = "http://localhost:8080"
        api_key = None
        duration = 60  # seconds
        
        logger.info("TSIOT Streaming Data Demo")
        logger.info("=" * 50)
        
        # Single stream demo
        demo = StreamingDemo(base_url=base_url, api_key=api_key)
        asyncio.run(demo.run_streaming_demo(duration))
        
        print("\n" + "=" * 50)
        print("Single stream demo completed!")
        
        # Multiple streams demo
        print("\nStarting multiple streams demo...")
        asyncio.run(demo_multiple_streams())
        
        print("\nMultiple streams demo completed!")
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