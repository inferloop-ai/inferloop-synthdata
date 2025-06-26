"""
Batch Inference System for TextNLP
High-performance batch processing for text generation with dynamic batching and optimization
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from queue import Queue, Empty
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class BatchingStrategy(Enum):
    """Batching strategies for inference"""
    DYNAMIC = "dynamic"  # Dynamic batching with timeout
    FIXED = "fixed"  # Fixed batch size
    ADAPTIVE = "adaptive"  # Adaptive based on model load
    CONTINUOUS = "continuous"  # Continuous batching (streaming)


class PaddingStrategy(Enum):
    """Padding strategies for batch processing"""
    LONGEST = "longest"  # Pad to longest sequence in batch
    MAX_LENGTH = "max_length"  # Pad to maximum model length
    ADAPTIVE = "adaptive"  # Adaptive padding based on batch statistics


@dataclass
class BatchRequest:
    """Individual request in a batch"""
    request_id: str
    prompt: str
    max_length: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    num_return_sequences: int = 1
    do_sample: bool = True
    user_id: Optional[str] = None
    priority: int = 0  # Higher priority = processed first
    timestamp: float = field(default_factory=time.time)
    callback: Optional[Callable] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResponse:
    """Response from batch processing"""
    request_id: str
    generated_text: str
    success: bool
    error_message: str = ""
    processing_time: float = 0.0
    queue_time: float = 0.0
    generation_time: float = 0.0
    tokens_generated: int = 0
    model_name: str = ""
    batch_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchConfig:
    """Configuration for batch inference"""
    max_batch_size: int = 8
    max_wait_time: float = 0.1  # seconds
    padding_strategy: PaddingStrategy = PaddingStrategy.LONGEST
    batching_strategy: BatchingStrategy = BatchingStrategy.DYNAMIC
    enable_kv_cache: bool = True
    max_queue_size: int = 1000
    num_workers: int = 1
    device: str = "auto"
    fp16: bool = True
    compile_model: bool = False
    
    # Adaptive batching parameters
    target_latency: float = 0.5  # Target latency in seconds
    min_batch_size: int = 1
    latency_sla: float = 2.0  # Maximum acceptable latency
    
    # Continuous batching parameters
    iteration_level_batching: bool = False
    max_iterations_per_batch: int = 10


class BatchInferenceEngine:
    """High-performance batch inference engine"""
    
    def __init__(self, model_name_or_path: str, config: BatchConfig):
        self.model_name = model_name_or_path
        self.config = config
        
        # Initialize model and tokenizer
        self._load_model_and_tokenizer()
        
        # Request queues
        self.request_queue = Queue(maxsize=config.max_queue_size)
        self.priority_queue = Queue(maxsize=config.max_queue_size)
        self.response_futures = {}
        
        # Batch processing
        self.current_batch = []
        self.batch_lock = threading.Lock()
        self.processing = False
        
        # Workers
        self.workers = []
        self.executor = ThreadPoolExecutor(max_workers=config.num_workers)
        
        # Performance tracking
        self.performance_stats = {
            "total_requests": 0,
            "total_batches": 0,
            "average_batch_size": 0.0,
            "average_latency": 0.0,
            "throughput": 0.0,
            "queue_length": 0
        }
        
        # Adaptive batching state
        self.recent_latencies = deque(maxlen=100)
        self.adaptive_batch_size = config.max_batch_size
        
        logger.info(f"Batch inference engine initialized for {model_name_or_path}")
    
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer with optimizations"""
        
        # Device setup
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            "torch_dtype": torch.float16 if self.config.fp16 and torch.cuda.is_available() else torch.float32,
            "device_map": "auto" if torch.cuda.is_available() else None,
            "low_cpu_mem_usage": True
        }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Model optimizations
        if self.config.enable_kv_cache:
            self.model.config.use_cache = True
        
        if self.config.compile_model and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model, mode="reduce-overhead")
            logger.info("Model compiled with PyTorch 2.0")
        
        self.model.eval()
        logger.info(f"Model loaded on {self.device}")
    
    def start(self):
        """Start the batch inference engine"""
        if self.processing:
            logger.warning("Engine already running")
            return
        
        self.processing = True
        
        # Start worker threads based on batching strategy
        if self.config.batching_strategy == BatchingStrategy.CONTINUOUS:
            self._start_continuous_batching()
        else:
            self._start_dynamic_batching()
        
        logger.info("Batch inference engine started")
    
    def stop(self):
        """Stop the batch inference engine"""
        self.processing = False
        
        # Stop workers
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        self.executor.shutdown(wait=True)
        logger.info("Batch inference engine stopped")
    
    def _start_dynamic_batching(self):
        """Start dynamic batching workers"""
        
        def batch_worker():
            while self.processing:
                try:
                    batch = self._collect_batch()
                    if batch:
                        self._process_batch(batch)
                    else:
                        time.sleep(0.001)  # Short sleep if no requests
                except Exception as e:
                    logger.error(f"Batch worker error: {e}")
        
        # Start worker threads
        for i in range(self.config.num_workers):
            worker = threading.Thread(target=batch_worker, daemon=True)
            worker.start()
            self.workers.append(worker)
    
    def _start_continuous_batching(self):
        """Start continuous batching for iteration-level processing"""
        
        def continuous_worker():
            active_sequences = {}  # request_id -> sequence state
            
            while self.processing:
                try:
                    # Add new requests to active sequences
                    new_requests = self._collect_new_requests()
                    for request in new_requests:
                        active_sequences[request.request_id] = {
                            "request": request,
                            "input_ids": None,
                            "past_key_values": None,
                            "generated_tokens": 0,
                            "start_time": time.time()
                        }
                    
                    if active_sequences:
                        # Process one iteration for all active sequences
                        completed = self._process_continuous_iteration(active_sequences)
                        
                        # Remove completed sequences
                        for request_id in completed:
                            del active_sequences[request_id]
                    else:
                        time.sleep(0.001)
                        
                except Exception as e:
                    logger.error(f"Continuous batching error: {e}")
        
        worker = threading.Thread(target=continuous_worker, daemon=True)
        worker.start()
        self.workers.append(worker)
    
    async def generate_async(self, request: BatchRequest) -> BatchResponse:
        """Asynchronously generate text for a single request"""
        
        # Create future for response
        future = asyncio.get_event_loop().create_future()
        request.callback = lambda response: future.set_result(response)
        
        # Add to appropriate queue
        try:
            if request.priority > 0:
                self.priority_queue.put_nowait(request)
            else:
                self.request_queue.put_nowait(request)
        except:
            return BatchResponse(
                request_id=request.request_id,
                generated_text="",
                success=False,
                error_message="Queue full"
            )
        
        # Wait for response
        return await future
    
    def generate_sync(self, request: BatchRequest) -> BatchResponse:
        """Synchronously generate text for a single request"""
        
        # Create event for synchronization
        response_event = threading.Event()
        response_holder = {"response": None}
        
        def callback(response):
            response_holder["response"] = response
            response_event.set()
        
        request.callback = callback
        
        # Add to queue
        try:
            if request.priority > 0:
                self.priority_queue.put_nowait(request)
            else:
                self.request_queue.put_nowait(request)
        except:
            return BatchResponse(
                request_id=request.request_id,
                generated_text="",
                success=False,
                error_message="Queue full"
            )
        
        # Wait for response
        if response_event.wait(timeout=self.config.latency_sla):
            return response_holder["response"]
        else:
            return BatchResponse(
                request_id=request.request_id,
                generated_text="",
                success=False,
                error_message="Timeout"
            )
    
    def _collect_batch(self) -> List[BatchRequest]:
        """Collect requests into a batch"""
        
        batch = []
        batch_size = self._get_adaptive_batch_size()
        start_time = time.time()
        
        # First, check priority queue
        while len(batch) < batch_size and not self.priority_queue.empty():
            try:
                request = self.priority_queue.get_nowait()
                batch.append(request)
            except Empty:
                break
        
        # Then, fill from regular queue
        while len(batch) < batch_size:
            try:
                # Wait for requests with timeout
                if len(batch) == 0:
                    # If no requests yet, wait longer
                    timeout = self.config.max_wait_time
                else:
                    # If we have some requests, shorter timeout
                    elapsed = time.time() - start_time
                    timeout = max(0.001, self.config.max_wait_time - elapsed)
                
                request = self.request_queue.get(timeout=timeout)
                batch.append(request)
                
            except Empty:
                break
        
        return batch
    
    def _collect_new_requests(self) -> List[BatchRequest]:
        """Collect new requests for continuous batching"""
        
        requests = []
        max_new_requests = self.config.max_batch_size // 2  # Leave room for existing sequences
        
        # Check priority queue first
        while len(requests) < max_new_requests and not self.priority_queue.empty():
            try:
                request = self.priority_queue.get_nowait()
                requests.append(request)
            except Empty:
                break
        
        # Then regular queue
        while len(requests) < max_new_requests and not self.request_queue.empty():
            try:
                request = self.request_queue.get_nowait()
                requests.append(request)
            except Empty:
                break
        
        return requests
    
    def _get_adaptive_batch_size(self) -> int:
        """Get adaptive batch size based on recent performance"""
        
        if self.config.batching_strategy != BatchingStrategy.ADAPTIVE:
            return self.config.max_batch_size
        
        if len(self.recent_latencies) < 10:
            return self.config.max_batch_size
        
        recent_avg_latency = np.mean(list(self.recent_latencies)[-10:])
        
        if recent_avg_latency > self.config.target_latency * 1.2:
            # Latency too high, reduce batch size
            self.adaptive_batch_size = max(
                self.config.min_batch_size,
                int(self.adaptive_batch_size * 0.8)
            )
        elif recent_avg_latency < self.config.target_latency * 0.8:
            # Latency good, can increase batch size
            self.adaptive_batch_size = min(
                self.config.max_batch_size,
                int(self.adaptive_batch_size * 1.2)
            )
        
        return self.adaptive_batch_size
    
    def _process_batch(self, batch: List[BatchRequest]):
        """Process a batch of requests"""
        
        if not batch:
            return
        
        batch_id = f"batch_{int(time.time() * 1000)}"
        start_time = time.time()
        
        try:
            # Prepare inputs
            inputs = self._prepare_batch_inputs(batch)
            
            # Generate
            with torch.no_grad():
                generation_start = time.time()
                outputs = self.model.generate(
                    **inputs,
                    max_length=max(req.max_length for req in batch),
                    temperature=batch[0].temperature,  # Use first request's params for simplicity
                    top_p=batch[0].top_p,
                    top_k=batch[0].top_k,
                    do_sample=batch[0].do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=self.config.enable_kv_cache
                )
                generation_time = time.time() - generation_start
            
            # Decode responses
            responses = self._decode_batch_outputs(batch, inputs, outputs, batch_id, start_time, generation_time)
            
            # Send responses
            for response in responses:
                if batch[responses.index(response)].callback:
                    batch[responses.index(response)].callback(response)
            
            # Update performance stats
            self._update_performance_stats(batch, responses, start_time)
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Send error responses
            for request in batch:
                error_response = BatchResponse(
                    request_id=request.request_id,
                    generated_text="",
                    success=False,
                    error_message=str(e),
                    batch_id=batch_id
                )
                if request.callback:
                    request.callback(error_response)
    
    def _process_continuous_iteration(self, active_sequences: Dict[str, Any]) -> List[str]:
        """Process one iteration of continuous batching"""
        
        completed_sequences = []
        
        if not active_sequences:
            return completed_sequences
        
        # Prepare batch inputs for current iteration
        batch_inputs = []
        sequence_ids = []
        
        for request_id, seq_state in active_sequences.items():
            request = seq_state["request"]
            
            if seq_state["input_ids"] is None:
                # First iteration - encode prompt
                inputs = self.tokenizer(
                    request.prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=request.max_length
                )
                seq_state["input_ids"] = inputs["input_ids"]
                seq_state["attention_mask"] = inputs["attention_mask"]
            
            batch_inputs.append({
                "input_ids": seq_state["input_ids"],
                "attention_mask": seq_state["attention_mask"],
                "past_key_values": seq_state["past_key_values"]
            })
            sequence_ids.append(request_id)
        
        # Pad and batch inputs
        batched_inputs = self._batch_continuous_inputs(batch_inputs)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**batched_inputs)
        
        # Process outputs for each sequence
        for i, request_id in enumerate(sequence_ids):
            seq_state = active_sequences[request_id]
            request = seq_state["request"]
            
            # Get next token
            logits = outputs.logits[i, -1, :]
            
            # Apply temperature and sampling
            if request.temperature != 1.0:
                logits = logits / request.temperature
            
            # Sample next token
            if request.do_sample:
                # Top-k and top-p sampling
                if request.top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, request.top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(0, top_k_indices, top_k_logits)
                
                if request.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > request.top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Update sequence state
            seq_state["input_ids"] = torch.cat([seq_state["input_ids"], next_token.unsqueeze(0)], dim=1)
            seq_state["attention_mask"] = torch.cat([
                seq_state["attention_mask"],
                torch.ones((1, 1), dtype=torch.long, device=seq_state["attention_mask"].device)
            ], dim=1)
            seq_state["generated_tokens"] += 1
            
            # Update past key values if using cache
            if self.config.enable_kv_cache and hasattr(outputs, 'past_key_values'):
                seq_state["past_key_values"] = outputs.past_key_values
            
            # Check completion conditions
            if (next_token.item() == self.tokenizer.eos_token_id or
                seq_state["generated_tokens"] >= request.max_length or
                seq_state["input_ids"].shape[1] >= request.max_length):
                
                # Generate response
                generated_text = self.tokenizer.decode(
                    seq_state["input_ids"][0],
                    skip_special_tokens=True
                )
                
                # Remove original prompt
                if generated_text.startswith(request.prompt):
                    generated_text = generated_text[len(request.prompt):].strip()
                
                response = BatchResponse(
                    request_id=request_id,
                    generated_text=generated_text,
                    success=True,
                    processing_time=time.time() - seq_state["start_time"],
                    generation_time=time.time() - seq_state["start_time"],
                    tokens_generated=seq_state["generated_tokens"],
                    model_name=self.model_name
                )
                
                if request.callback:
                    request.callback(response)
                
                completed_sequences.append(request_id)
        
        return completed_sequences
    
    def _prepare_batch_inputs(self, batch: List[BatchRequest]) -> Dict[str, torch.Tensor]:
        """Prepare inputs for batch processing"""
        
        prompts = [request.prompt for request in batch]
        
        # Tokenize with appropriate padding
        if self.config.padding_strategy == PaddingStrategy.LONGEST:
            padding = True
            max_length = None
        elif self.config.padding_strategy == PaddingStrategy.MAX_LENGTH:
            padding = "max_length"
            max_length = max(req.max_length for req in batch)
        else:  # ADAPTIVE
            # Calculate adaptive max length
            lengths = [len(self.tokenizer.encode(prompt)) for prompt in prompts]
            max_length = min(max(lengths) + 50, max(req.max_length for req in batch))
            padding = "max_length"
        
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=padding,
            truncation=True,
            max_length=max_length
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def _batch_continuous_inputs(self, batch_inputs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Batch inputs for continuous processing"""
        
        # Find maximum sequence length
        max_length = max(inputs["input_ids"].shape[1] for inputs in batch_inputs)
        
        # Pad sequences
        batched = {
            "input_ids": [],
            "attention_mask": []
        }
        
        for inputs in batch_inputs:
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            # Pad if necessary
            if input_ids.shape[1] < max_length:
                pad_length = max_length - input_ids.shape[1]
                input_ids = torch.cat([
                    torch.full((1, pad_length), self.tokenizer.pad_token_id, dtype=input_ids.dtype, device=input_ids.device),
                    input_ids
                ], dim=1)
                attention_mask = torch.cat([
                    torch.zeros((1, pad_length), dtype=attention_mask.dtype, device=attention_mask.device),
                    attention_mask
                ], dim=1)
            
            batched["input_ids"].append(input_ids)
            batched["attention_mask"].append(attention_mask)
        
        # Stack tensors
        batched["input_ids"] = torch.cat(batched["input_ids"], dim=0)
        batched["attention_mask"] = torch.cat(batched["attention_mask"], dim=0)
        
        return batched
    
    def _decode_batch_outputs(self, batch: List[BatchRequest], inputs: Dict[str, torch.Tensor],
                            outputs: torch.Tensor, batch_id: str, start_time: float,
                            generation_time: float) -> List[BatchResponse]:
        """Decode batch outputs into responses"""
        
        responses = []
        
        for i, request in enumerate(batch):
            try:
                # Extract generated tokens (remove input)
                input_length = inputs["input_ids"][i].shape[0]
                generated_ids = outputs[i][input_length:]
                
                # Decode text
                generated_text = self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True
                )
                
                response = BatchResponse(
                    request_id=request.request_id,
                    generated_text=generated_text,
                    success=True,
                    processing_time=time.time() - start_time,
                    queue_time=start_time - request.timestamp,
                    generation_time=generation_time,
                    tokens_generated=len(generated_ids),
                    model_name=self.model_name,
                    batch_id=batch_id
                )
                
            except Exception as e:
                response = BatchResponse(
                    request_id=request.request_id,
                    generated_text="",
                    success=False,
                    error_message=str(e),
                    batch_id=batch_id
                )
            
            responses.append(response)
        
        return responses
    
    def _update_performance_stats(self, batch: List[BatchRequest], 
                                responses: List[BatchResponse], start_time: float):
        """Update performance statistics"""
        
        processing_time = time.time() - start_time
        
        self.performance_stats["total_requests"] += len(batch)
        self.performance_stats["total_batches"] += 1
        
        # Update running averages
        total_requests = self.performance_stats["total_requests"]
        
        # Average batch size
        self.performance_stats["average_batch_size"] = (
            (self.performance_stats["average_batch_size"] * (total_requests - len(batch)) + len(batch)) / 
            total_requests
        )
        
        # Average latency
        successful_responses = [r for r in responses if r.success]
        if successful_responses:
            avg_response_time = np.mean([r.processing_time for r in successful_responses])
            self.performance_stats["average_latency"] = (
                (self.performance_stats["average_latency"] * (total_requests - len(batch)) + 
                 avg_response_time * len(batch)) / total_requests
            )
            
            # Update recent latencies for adaptive batching
            self.recent_latencies.append(avg_response_time)
        
        # Throughput (requests per second)
        self.performance_stats["throughput"] = len(batch) / processing_time
        
        # Queue length
        self.performance_stats["queue_length"] = (
            self.request_queue.qsize() + self.priority_queue.qsize()
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            **self.performance_stats,
            "adaptive_batch_size": self.adaptive_batch_size,
            "device": str(self.device),
            "model_name": self.model_name,
            "config": {
                "max_batch_size": self.config.max_batch_size,
                "batching_strategy": self.config.batching_strategy.value,
                "padding_strategy": self.config.padding_strategy.value
            }
        }
    
    @asynccontextmanager
    async def batch_context(self):
        """Context manager for batch inference engine"""
        try:
            self.start()
            yield self
        finally:
            self.stop()


# Example usage and testing
async def example_usage():
    """Example usage of the batch inference engine"""
    
    config = BatchConfig(
        max_batch_size=4,
        max_wait_time=0.1,
        batching_strategy=BatchingStrategy.DYNAMIC,
        padding_strategy=PaddingStrategy.LONGEST,
        num_workers=1,
        fp16=False  # For CPU testing
    )
    
    engine = BatchInferenceEngine("gpt2", config)
    
    async with engine.batch_context():
        # Test batch processing
        requests = [
            BatchRequest(
                request_id=f"req_{i}",
                prompt=f"Write a story about {topic}",
                max_length=100
            )
            for i, topic in enumerate(["cats", "space", "robots", "magic", "ocean"])
        ]
        
        # Process requests
        print("Processing batch requests...")
        tasks = [engine.generate_async(req) for req in requests]
        responses = await asyncio.gather(*tasks)
        
        # Print results
        for response in responses:
            if response.success:
                print(f"Request {response.request_id}: {response.generated_text[:50]}...")
                print(f"  Processing time: {response.processing_time:.3f}s")
                print(f"  Tokens generated: {response.tokens_generated}")
            else:
                print(f"Request {response.request_id} failed: {response.error_message}")
        
        # Print stats
        stats = engine.get_stats()
        print(f"\nEngine Statistics:")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Average batch size: {stats['average_batch_size']:.2f}")
        print(f"  Average latency: {stats['average_latency']:.3f}s")
        print(f"  Throughput: {stats['throughput']:.2f} req/s")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())