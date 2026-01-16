"""Benchmarking tools for training and inference."""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    timestamp: str
    
    # Training metrics
    tokens_per_second: Optional[float] = None
    samples_per_second: Optional[float] = None
    peak_memory_gb: Optional[float] = None
    average_memory_gb: Optional[float] = None
    
    # Inference metrics
    inference_latency_ms: Optional[float] = None
    inference_throughput_tokens_per_sec: Optional[float] = None
    
    # Model metrics
    model_size_gb: Optional[float] = None
    num_parameters: Optional[int] = None
    trainable_parameters: Optional[int] = None
    
    # Extra info
    batch_size: Optional[int] = None
    sequence_length: Optional[int] = None
    device: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dict."""
        return asdict(self)
    
    def save(self, path: str):
        """Save to JSON."""
        try:
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
            logger.info(f"Saved benchmark to {path}")
        except Exception as e:
            logger.error(f"Failed to save benchmark: {e}")


class PerformanceBenchmark:
    """Performance profiling for models."""
    
    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device
        self.results: List[BenchmarkResult] = []
    
    def benchmark_training_speed(
        self,
        dataloader: DataLoader,
        max_batches: int = 10,
        warmup_batches: int = 2
    ) -> Dict[str, float]:
        """
        Measure training throughput.
        
        Args:
            dataloader: Training data
            max_batches: Test on this many batches
            warmup_batches: Skip these batches (for warmup)
        
        Returns:
            Dict with tokens/sec, samples/sec, etc
        """
        self.model.train()
        
        times = []
        total_tokens = 0
        total_samples = 0
        
        # Warmup (not timed)
        for i, batch in enumerate(dataloader):
            if i >= warmup_batches:
                break
            
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            else:
                batch = batch.to(self.device)
            
            with torch.no_grad():
                _ = self.model(**batch) if isinstance(batch, dict) else self.model(batch)
            
            torch.cuda.synchronize() if self.device == 'cuda' else None
        
        # Actual benchmark
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
                
                if isinstance(batch, dict):
                    seq_len = batch.get('input_ids', torch.zeros(1)).shape[1]
                    batch_size = batch.get('input_ids', torch.zeros(1)).shape[0]
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                else:
                    seq_len = batch.shape[1] if len(batch.shape) > 1 else 1
                    batch_size = batch.shape[0]
                    batch = batch.to(self.device)
                
                start = time.time()
                _ = self.model(**batch) if isinstance(batch, dict) else self.model(batch)
                torch.cuda.synchronize() if self.device == 'cuda' else None
                end = time.time()
                
                times.append(end - start)
                total_tokens += batch_size * seq_len
                total_samples += batch_size
        
        if len(times) == 0:
            return {}
        
        avg_time = np.mean(times)
        
        return {
            'throughput_tokens_per_second': total_tokens / sum(times),
            'throughput_samples_per_second': total_samples / sum(times),
            'avg_batch_time_ms': avg_time * 1000,
            'batches_tested': len(times)
        }
    
    def benchmark_inference(
        self,
        batch_size: int = 4,
        seq_len: int = 512,
        vocab_size: int = 32000,
        max_new_tokens: int = 128,
        num_runs: int = 5
    ) -> Dict[str, float]:
        """
        Measure inference speed.
        
        Uses random data, so this is just a speed test, not accuracy test.
        
        Args:
            batch_size: Batch size for inference
            seq_len: Sequence length
            vocab_size: Vocab size (for random data)
            max_new_tokens: Generate this many tokens
            num_runs: Average over this many runs
        
        Returns:
            Dict with latency, throughput, etc
        """
        self.model.eval()
        
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                # Create random input
                input_ids = torch.randint(
                    0, vocab_size, (batch_size, seq_len), device=self.device
                )
                attention_mask = torch.ones_like(input_ids)
                
                start = time.time()
                _ = self.model(input_ids, attention_mask=attention_mask)
                torch.cuda.synchronize() if self.device == 'cuda' else None
                end = time.time()
                
                times.append(end - start)
        
        if len(times) == 0:
            return {}
        
        total_tokens = max_new_tokens * batch_size
        avg_time = np.mean(times)
        
        return {
            'latency_ms': avg_time * 1000,
            'throughput_tokens_per_sec': total_tokens / avg_time,
            'runs': num_runs
        }
    
    def measure_model_size(self) -> Dict[str, float]:
        """
        Measure model size in memory.
        
        Returns:
            Dict with size info
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Estimate size (4 bytes per float32 param)
        size_gb = (total_params * 4) / (1024 ** 3)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'estimated_size_gb': size_gb,
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0.0
        }
    
    def measure_memory_usage(
        self,
        dataloader: DataLoader,
        max_batches: int = 5
    ) -> Dict[str, float]:
        """
        Measure peak and average GPU memory during training.
        
        Args:
            dataloader: Training data
            max_batches: Test on this many batches
        
        Returns:
            Dict with memory info
        """
        if self.device != 'cuda':
            logger.warning("Memory measurement only works on CUDA")
            return {}
        
        self.model.train()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        memory_readings = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
                
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                else:
                    batch = batch.to(self.device)
                
                _ = self.model(**batch) if isinstance(batch, dict) else self.model(batch)
                
                # Record memory usage
                memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                memory_readings.append(memory_mb)
        
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        avg_mb = np.mean(memory_readings) if memory_readings else 0.0
        
        return {
            'peak_memory_gb': peak_mb / 1024,
            'average_memory_gb': avg_mb / 1024,
            'batches_measured': len(memory_readings)
        }
    
    def run_all_benchmarks(
        self,
        train_dataloader: Optional[DataLoader] = None,
        vocab_size: int = 32000,
        save_path: Optional[str] = None
    ) -> BenchmarkResult:
        """
        Run all benchmarks and return summary.
        
        Args:
            train_dataloader: Optional training data for training benchmark
            vocab_size: Vocab size for inference benchmark
            save_path: Save results to JSON file
        
        Returns:
            BenchmarkResult with all measurements
        """
        logger.info("Running benchmarks...")
        
        result = BenchmarkResult(
            name=self.model.__class__.__name__,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            device=self.device
        )
        
        # Model size
        size_info = self.measure_model_size()
        result.model_size_gb = size_info.get('estimated_size_gb')
        result.num_parameters = size_info.get('total_parameters')
        result.trainable_parameters = size_info.get('trainable_parameters')
        
        # Training speed (if data provided)
        if train_dataloader:
            try:
                logger.info("Benchmarking training speed...")
                speed_info = self.benchmark_training_speed(train_dataloader)
                result.tokens_per_second = speed_info.get('throughput_tokens_per_second')
                result.samples_per_second = speed_info.get('throughput_samples_per_second')
            except Exception as e:
                logger.warning(f"Training benchmark failed: {e}")
        
        # Memory (if CUDA)
        if self.device == 'cuda' and train_dataloader:
            try:
                logger.info("Measuring memory usage...")
                mem_info = self.measure_memory_usage(train_dataloader)
                result.peak_memory_gb = mem_info.get('peak_memory_gb')
                result.average_memory_gb = mem_info.get('average_memory_gb')
            except Exception as e:
                logger.warning(f"Memory benchmark failed: {e}")
        
        # Inference
        try:
            logger.info("Benchmarking inference...")
            inf_info = self.benchmark_inference(vocab_size=vocab_size)
            result.inference_latency_ms = inf_info.get('latency_ms')
            result.inference_throughput_tokens_per_sec = inf_info.get('throughput_tokens_per_sec')
        except Exception as e:
            logger.warning(f"Inference benchmark failed: {e}")
        
        # Save if requested
        if save_path:
            result.save(save_path)
        
        return result
    
    def print_summary(self, result: BenchmarkResult):
        """Print results nicely."""
        print(f"\nBenchmark Results: {result.name}")
        print(f"Device: {result.device}")
        print(f"Time: {result.timestamp}")
        print()
        
        if result.num_parameters:
            print(f"Parameters: {result.num_parameters:,}")
            print(f"Trainable: {result.trainable_parameters:,}")
            if result.model_size_gb:
                print(f"Estimated Size: {result.model_size_gb:.2f} GB")
        
        if result.tokens_per_second:
            print(f"Training Throughput: {result.tokens_per_second:.0f} tokens/sec")
        
        if result.peak_memory_gb:
            print(f"Peak Memory: {result.peak_memory_gb:.2f} GB")
            if result.average_memory_gb:
                print(f"Average Memory: {result.average_memory_gb:.2f} GB")
        
        if result.inference_throughput_tokens_per_sec:
            print(f"Inference Throughput: {result.inference_throughput_tokens_per_sec:.0f} tokens/sec")
            if result.inference_latency_ms:
                print(f"Inference Latency: {result.inference_latency_ms:.2f} ms")
        
        print()
