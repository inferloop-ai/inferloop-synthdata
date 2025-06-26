"""
Generation Metrics Implementation for TextNLP
Comprehensive metrics collection and analysis for text generation
"""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import statistics
import json
import hashlib
from collections import defaultdict, Counter
import re
import numpy as np
import torch
from transformers import AutoTokenizer
import psutil
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of generation metrics"""
    # Performance Metrics
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    TOKENS_PER_SECOND = "tokens_per_second"
    
    # Quality Metrics
    PERPLEXITY = "perplexity"
    COHERENCE = "coherence"
    FLUENCY = "fluency"
    DIVERSITY = "diversity"
    REPETITION = "repetition"
    
    # Content Metrics
    LENGTH = "length"
    VOCABULARY_SIZE = "vocabulary_size"
    READABILITY = "readability"
    SENTIMENT = "sentiment"
    
    # Model Metrics
    GPU_UTILIZATION = "gpu_utilization"
    MEMORY_USAGE = "memory_usage"
    CPU_UTILIZATION = "cpu_utilization"
    
    # Safety Metrics
    SAFETY_SCORE = "safety_score"
    TOXICITY_SCORE = "toxicity_score"
    BIAS_SCORE = "bias_score"
    
    # Business Metrics
    SUCCESS_RATE = "success_rate"
    ERROR_RATE = "error_rate"
    USER_SATISFACTION = "user_satisfaction"


@dataclass
class MetricValue:
    """Individual metric value"""
    metric_type: MetricType
    value: Union[float, int, str]
    unit: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationMetrics:
    """Complete metrics for a generation request"""
    request_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    model_name: str = ""
    prompt_length: int = 0
    generated_length: int = 0
    
    # Timing metrics
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_latency: float = 0.0
    first_token_latency: float = 0.0
    tokens_per_second: float = 0.0
    
    # Quality metrics
    perplexity: Optional[float] = None
    coherence_score: Optional[float] = None
    fluency_score: Optional[float] = None
    diversity_score: Optional[float] = None
    repetition_ratio: Optional[float] = None
    
    # Content metrics
    vocabulary_size: int = 0
    readability_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    
    # Resource metrics
    peak_gpu_memory: Optional[float] = None
    peak_cpu_usage: Optional[float] = None
    peak_ram_usage: Optional[float] = None
    
    # Safety metrics
    safety_violations: List[str] = field(default_factory=list)
    toxicity_score: Optional[float] = None
    bias_score: Optional[float] = None
    
    # Status
    success: bool = True
    error_message: Optional[str] = None
    
    # Additional metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, (list, dict)):
                result[key] = value
            else:
                result[key] = value
        return result


class MetricsCollector:
    """Comprehensive metrics collection system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.enabled_metrics = set(self.config.get("enabled_metrics", [
            MetricType.LATENCY, MetricType.THROUGHPUT, MetricType.LENGTH,
            MetricType.SUCCESS_RATE, MetricType.TOXICITY_SCORE
        ]))
        
        # Storage for metrics
        self.metrics_history: List[GenerationMetrics] = []
        self.aggregated_metrics: Dict[str, Any] = {}
        self.real_time_metrics: Dict[str, Any] = {}
        
        # Tokenizer for text analysis
        self._initialize_tokenizer()
        
        # Background monitoring
        self.monitoring_enabled = self.config.get("enable_monitoring", True)
        self.monitoring_interval = self.config.get("monitoring_interval", 10)  # seconds
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        
        if self.monitoring_enabled:
            self._start_monitoring()
    
    def _initialize_tokenizer(self):
        """Initialize tokenizer for text analysis"""
        try:
            tokenizer_name = self.config.get("tokenizer", "gpt2")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Initialized tokenizer: {tokenizer_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize tokenizer: {e}")
            self.tokenizer = None
    
    def _start_monitoring(self):
        """Start background monitoring thread"""
        self.monitoring_thread = threading.Thread(target=self._monitor_system_metrics, daemon=True)
        self.monitoring_thread.start()
        logger.info("Started system metrics monitoring")
    
    def _monitor_system_metrics(self):
        """Monitor system metrics in background"""
        while not self.stop_monitoring.wait(self.monitoring_interval):
            try:
                # CPU utilization
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # GPU metrics (if available)
                gpu_metrics = self._get_gpu_metrics()
                
                # Update real-time metrics
                self.real_time_metrics.update({
                    "cpu_utilization": cpu_percent,
                    "memory_utilization": memory_percent,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                if gpu_metrics:
                    self.real_time_metrics.update(gpu_metrics)
                
            except Exception as e:
                logger.error(f"Error in system metrics monitoring: {e}")
    
    def _get_gpu_metrics(self) -> Dict[str, Any]:
        """Get GPU metrics if CUDA is available"""
        gpu_metrics = {}
        
        try:
            if torch.cuda.is_available():
                # GPU memory
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
                    memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                    
                    gpu_metrics[f"gpu_{i}_memory_allocated"] = memory_allocated
                    gpu_metrics[f"gpu_{i}_memory_reserved"] = memory_reserved
                    gpu_metrics[f"gpu_{i}_memory_total"] = memory_total
                    gpu_metrics[f"gpu_{i}_utilization"] = (memory_allocated / memory_total) * 100
        
        except Exception as e:
            logger.debug(f"Could not get GPU metrics: {e}")
        
        return gpu_metrics
    
    @contextmanager
    def track_generation(self, request_id: str, user_id: Optional[str] = None, 
                        model_name: str = "", prompt: str = ""):
        """Context manager for tracking generation metrics"""
        metrics = GenerationMetrics(
            request_id=request_id,
            user_id=user_id,
            model_name=model_name,
            prompt_length=len(prompt) if prompt else 0,
            start_time=datetime.now(timezone.utc)
        )
        
        # Initial resource snapshot
        initial_memory = psutil.virtual_memory().used
        initial_gpu_memory = self._get_current_gpu_memory()
        
        start_time = time.time()
        first_token_time = None
        
        class TokenTracker:
            def __init__(self):
                self.first_token_recorded = False
            
            def record_first_token(self):
                nonlocal first_token_time
                if not self.first_token_recorded:
                    first_token_time = time.time()
                    self.first_token_recorded = True
        
        token_tracker = TokenTracker()
        
        try:
            yield metrics, token_tracker
            metrics.success = True
        except Exception as e:
            metrics.success = False
            metrics.error_message = str(e)
            logger.error(f"Generation failed for request {request_id}: {e}")
        finally:
            # Calculate timing metrics
            end_time = time.time()
            metrics.end_time = datetime.now(timezone.utc)
            metrics.total_latency = end_time - start_time
            
            if first_token_time:
                metrics.first_token_latency = first_token_time - start_time
            
            if metrics.generated_length > 0 and metrics.total_latency > 0:
                metrics.tokens_per_second = metrics.generated_length / metrics.total_latency
            
            # Calculate resource usage
            final_memory = psutil.virtual_memory().used
            final_gpu_memory = self._get_current_gpu_memory()
            
            metrics.peak_ram_usage = (final_memory - initial_memory) / 1024**3  # GB
            
            if initial_gpu_memory and final_gpu_memory:
                metrics.peak_gpu_memory = max(0, final_gpu_memory - initial_gpu_memory)
            
            # Store metrics
            self.metrics_history.append(metrics)
            self._update_aggregated_metrics()
    
    def _get_current_gpu_memory(self) -> Optional[float]:
        """Get current GPU memory usage"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024**3  # GB
        except Exception:
            pass
        return None
    
    def calculate_text_metrics(self, text: str, prompt: str = "") -> Dict[str, Any]:
        """Calculate various text quality metrics"""
        metrics = {}
        
        if not text:
            return metrics
        
        # Basic metrics
        metrics["length"] = len(text)
        metrics["word_count"] = len(text.split())
        metrics["sentence_count"] = len(re.split(r'[.!?]+', text))
        
        # Vocabulary metrics
        words = re.findall(r'\b\w+\b', text.lower())
        unique_words = set(words)
        metrics["vocabulary_size"] = len(unique_words)
        metrics["vocabulary_diversity"] = len(unique_words) / len(words) if words else 0
        
        # Repetition metrics
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            metrics["token_count"] = len(tokens)
            
            # Calculate repetition ratio
            if len(tokens) > 1:
                repetitions = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i-1])
                metrics["repetition_ratio"] = repetitions / (len(tokens) - 1)
            else:
                metrics["repetition_ratio"] = 0.0
        
        # N-gram diversity
        metrics.update(self._calculate_ngram_diversity(text))
        
        # Readability metrics
        metrics.update(self._calculate_readability(text))
        
        # Coherence metrics (if prompt provided)
        if prompt:
            metrics.update(self._calculate_coherence(prompt, text))
        
        return metrics
    
    def _calculate_ngram_diversity(self, text: str) -> Dict[str, float]:
        """Calculate n-gram diversity metrics"""
        words = re.findall(r'\b\w+\b', text.lower())
        
        if len(words) < 2:
            return {"bigram_diversity": 0.0, "trigram_diversity": 0.0}
        
        # Bigram diversity
        bigrams = [tuple(words[i:i+2]) for i in range(len(words) - 1)]
        bigram_diversity = len(set(bigrams)) / len(bigrams) if bigrams else 0
        
        # Trigram diversity
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        trigram_diversity = len(set(trigrams)) / len(trigrams) if trigrams else 0
        
        return {
            "bigram_diversity": bigram_diversity,
            "trigram_diversity": trigram_diversity
        }
    
    def _calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics"""
        words = re.findall(r'\b\w+\b', text)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not words or not sentences:
            return {"flesch_score": 0.0, "avg_sentence_length": 0.0}
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Syllable count (approximation)
        def count_syllables(word):
            word = word.lower()
            vowels = "aeiouy"
            syllable_count = 0
            prev_was_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = is_vowel
            
            # Handle silent e
            if word.endswith('e') and syllable_count > 1:
                syllable_count -= 1
            
            return max(1, syllable_count)
        
        total_syllables = sum(count_syllables(word) for word in words)
        avg_syllables_per_word = total_syllables / len(words)
        
        # Flesch Reading Ease Score
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        flesch_score = max(0, min(100, flesch_score))  # Clamp to 0-100
        
        return {
            "flesch_score": flesch_score,
            "avg_sentence_length": avg_sentence_length,
            "avg_syllables_per_word": avg_syllables_per_word
        }
    
    def _calculate_coherence(self, prompt: str, text: str) -> Dict[str, float]:
        """Calculate coherence metrics between prompt and generated text"""
        if not self.tokenizer:
            return {"coherence_score": 0.0}
        
        try:
            # Simple word overlap coherence
            prompt_words = set(re.findall(r'\b\w+\b', prompt.lower()))
            text_words = set(re.findall(r'\b\w+\b', text.lower()))
            
            overlap = len(prompt_words & text_words)
            total_unique = len(prompt_words | text_words)
            
            coherence_score = overlap / total_unique if total_unique > 0 else 0.0
            
            return {"coherence_score": coherence_score}
        
        except Exception as e:
            logger.warning(f"Failed to calculate coherence: {e}")
            return {"coherence_score": 0.0}
    
    def calculate_perplexity(self, text: str, model) -> Optional[float]:
        """Calculate perplexity using a language model"""
        try:
            if not self.tokenizer:
                return None
            
            # Encode text
            inputs = self.tokenizer.encode(text, return_tensors="pt")
            
            # Calculate perplexity
            with torch.no_grad():
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            
            return perplexity
        
        except Exception as e:
            logger.warning(f"Failed to calculate perplexity: {e}")
            return None
    
    def add_safety_metrics(self, metrics: GenerationMetrics, 
                          safety_scores: Dict[str, float]):
        """Add safety-related metrics"""
        metrics.toxicity_score = safety_scores.get("toxicity", 0.0)
        metrics.bias_score = safety_scores.get("bias", 0.0)
        
        # Identify violations
        violations = []
        if metrics.toxicity_score > 0.5:
            violations.append("toxicity")
        if metrics.bias_score > 0.5:
            violations.append("bias")
        
        metrics.safety_violations = violations
    
    def _update_aggregated_metrics(self):
        """Update aggregated metrics from history"""
        if not self.metrics_history:
            return
        
        recent_metrics = self.metrics_history[-100:]  # Last 100 requests
        
        # Success rate
        successful = sum(1 for m in recent_metrics if m.success)
        self.aggregated_metrics["success_rate"] = successful / len(recent_metrics)
        
        # Average latency
        latencies = [m.total_latency for m in recent_metrics if m.total_latency > 0]
        if latencies:
            self.aggregated_metrics["avg_latency"] = statistics.mean(latencies)
            self.aggregated_metrics["p95_latency"] = np.percentile(latencies, 95)
            self.aggregated_metrics["p99_latency"] = np.percentile(latencies, 99)
        
        # Average throughput
        throughputs = [m.tokens_per_second for m in recent_metrics if m.tokens_per_second > 0]
        if throughputs:
            self.aggregated_metrics["avg_throughput"] = statistics.mean(throughputs)
        
        # Safety metrics
        toxicity_scores = [m.toxicity_score for m in recent_metrics if m.toxicity_score is not None]
        if toxicity_scores:
            self.aggregated_metrics["avg_toxicity"] = statistics.mean(toxicity_scores)
        
        bias_scores = [m.bias_score for m in recent_metrics if m.bias_score is not None]
        if bias_scores:
            self.aggregated_metrics["avg_bias"] = statistics.mean(bias_scores)
        
        # Violation rate
        total_violations = sum(len(m.safety_violations) for m in recent_metrics)
        self.aggregated_metrics["violation_rate"] = total_violations / len(recent_metrics)
        
        # Update timestamp
        self.aggregated_metrics["updated_at"] = datetime.now(timezone.utc).isoformat()
    
    def get_metrics_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get summary of metrics for a given time window"""
        cutoff_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_time = cutoff_time.replace(hour=cutoff_time.hour - time_window_hours)
        
        recent_metrics = [
            m for m in self.metrics_history 
            if m.start_time and m.start_time >= cutoff_time
        ]
        
        if not recent_metrics:
            return {"message": "No metrics in time window", "count": 0}
        
        summary = {
            "time_window_hours": time_window_hours,
            "total_requests": len(recent_metrics),
            "successful_requests": sum(1 for m in recent_metrics if m.success),
            "failed_requests": sum(1 for m in recent_metrics if not m.success),
        }
        
        # Performance metrics
        latencies = [m.total_latency for m in recent_metrics if m.total_latency > 0]
        if latencies:
            summary["performance"] = {
                "avg_latency": statistics.mean(latencies),
                "min_latency": min(latencies),
                "max_latency": max(latencies),
                "p50_latency": np.percentile(latencies, 50),
                "p95_latency": np.percentile(latencies, 95),
                "p99_latency": np.percentile(latencies, 99)
            }
        
        # Throughput metrics
        throughputs = [m.tokens_per_second for m in recent_metrics if m.tokens_per_second > 0]
        if throughputs:
            summary["throughput"] = {
                "avg_tokens_per_second": statistics.mean(throughputs),
                "min_tokens_per_second": min(throughputs),
                "max_tokens_per_second": max(throughputs)
            }
        
        # Content metrics
        lengths = [m.generated_length for m in recent_metrics if m.generated_length > 0]
        if lengths:
            summary["content"] = {
                "avg_length": statistics.mean(lengths),
                "min_length": min(lengths),
                "max_length": max(lengths)
            }
        
        # Safety metrics
        toxicity_scores = [m.toxicity_score for m in recent_metrics if m.toxicity_score is not None]
        if toxicity_scores:
            summary["safety"] = {
                "avg_toxicity": statistics.mean(toxicity_scores),
                "max_toxicity": max(toxicity_scores),
                "high_toxicity_count": sum(1 for score in toxicity_scores if score > 0.5)
            }
        
        # Model usage
        model_counts = Counter(m.model_name for m in recent_metrics if m.model_name)
        summary["model_usage"] = dict(model_counts.most_common(10))
        
        # Error analysis
        error_messages = [m.error_message for m in recent_metrics if m.error_message]
        error_counts = Counter(error_messages)
        summary["top_errors"] = dict(error_counts.most_common(5))
        
        return summary
    
    def export_metrics(self, format: str = "json", 
                      time_window_hours: Optional[int] = None) -> str:
        """Export metrics in specified format"""
        if time_window_hours:
            cutoff_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_time = cutoff_time.replace(hour=cutoff_time.hour - time_window_hours)
            metrics_to_export = [
                m for m in self.metrics_history 
                if m.start_time and m.start_time >= cutoff_time
            ]
        else:
            metrics_to_export = self.metrics_history
        
        if format.lower() == "json":
            data = {
                "metrics": [m.to_dict() for m in metrics_to_export],
                "summary": self.get_metrics_summary(time_window_hours or 24),
                "real_time_metrics": self.real_time_metrics,
                "aggregated_metrics": self.aggregated_metrics,
                "export_timestamp": datetime.now(timezone.utc).isoformat()
            }
            return json.dumps(data, indent=2)
        
        elif format.lower() == "csv":
            # Convert to CSV format
            import csv
            import io
            
            output = io.StringIO()
            if metrics_to_export:
                fieldnames = metrics_to_export[0].to_dict().keys()
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for metric in metrics_to_export:
                    writer.writerow(metric.to_dict())
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics"""
        return {
            **self.real_time_metrics,
            **self.aggregated_metrics,
            "active_requests": len([m for m in self.metrics_history if m.end_time is None])
        }
    
    def stop(self):
        """Stop the metrics collector"""
        if self.monitoring_thread:
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Metrics collector stopped")


# Example usage
if __name__ == "__main__":
    async def example():
        # Initialize metrics collector
        config = {
            "enabled_metrics": [
                MetricType.LATENCY, MetricType.THROUGHPUT, MetricType.LENGTH,
                MetricType.TOXICITY_SCORE, MetricType.SUCCESS_RATE
            ],
            "enable_monitoring": True,
            "monitoring_interval": 5
        }
        
        collector = MetricsCollector(config)
        
        # Simulate some generation requests
        for i in range(5):
            request_id = f"req_{i}"
            prompt = f"Write a story about {['cats', 'dogs', 'robots', 'space', 'magic'][i]}"
            
            with collector.track_generation(request_id, f"user_{i}", "gpt2", prompt) as (metrics, token_tracker):
                # Simulate generation process
                await asyncio.sleep(0.1)  # Simulate processing time
                token_tracker.record_first_token()
                
                # Simulate generated text
                generated_text = f"This is a generated story about {prompt.split()[-1]}. " * 10
                metrics.generated_length = len(generated_text.split())
                
                # Calculate text metrics
                text_metrics = collector.calculate_text_metrics(generated_text, prompt)
                metrics.vocabulary_size = text_metrics.get("vocabulary_size", 0)
                metrics.repetition_ratio = text_metrics.get("repetition_ratio", 0.0)
                
                # Add safety scores (simulated)
                collector.add_safety_metrics(metrics, {
                    "toxicity": 0.1,
                    "bias": 0.05
                })
                
                await asyncio.sleep(0.2)  # Simulate more processing
        
        # Get metrics summary
        summary = collector.get_metrics_summary(1)  # Last hour
        print("Metrics Summary:")
        print(json.dumps(summary, indent=2))
        
        # Get real-time metrics
        real_time = collector.get_real_time_metrics()
        print("\nReal-time Metrics:")
        print(json.dumps(real_time, indent=2))
        
        # Export metrics
        json_export = collector.export_metrics("json", 1)
        print("\nExported metrics length:", len(json_export))
        
        collector.stop()
    
    # Run example
    # asyncio.run(example())