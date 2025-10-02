#!/usr/bin/env python3
"""Test script for GPU memory scheduler and torch.multiprocessing implementation."""

import sys
import os
import logging
import time
import torch

# Add project paths
sys.path.append("src")
from common.libs import utilities

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_gpu_scheduler():
    """Test GPU memory scheduler functionality."""
    print("=== Testing GPU Memory Scheduler ===")
    
    scheduler = utilities.get_gpu_scheduler()
    
    # Test memory estimation
    memory_4k = scheduler.estimate_memory_usage(4096, 4096, 3)
    memory_6k = scheduler.estimate_memory_usage(6144, 6144, 3)
    
    print(f"Estimated memory for 4K image: {memory_4k / 1e9:.2f} GB")
    print(f"Estimated memory for 6K image: {memory_6k / 1e9:.2f} GB")
    
    # Test memory acquisition
    task_id = "test_task_1"
    if scheduler.acquire_memory(task_id, memory_4k):
        print(f"Successfully acquired {memory_4k / 1e6:.1f} MB for {task_id}")
        scheduler.release_memory(task_id)
        print(f"Released memory for {task_id}")
    else:
        print(f"Could not acquire memory for {task_id}")

def dummy_worker(args):
    """Dummy worker function for testing multiprocessing."""
    task_id, sleep_time = args
    
    # Simulate some work
    time.sleep(sleep_time)
    
    # Test CUDA availability in worker process
    cuda_available = torch.cuda.is_available() if utilities.TORCH_AVAILABLE else False
    
    return {
        'task_id': task_id,
        'process_id': os.getpid(),
        'cuda_available': cuda_available,
        'torch_threads': torch.get_num_threads() if utilities.TORCH_AVAILABLE else None,
        'sleep_time': sleep_time
    }

def dummy_memory_estimator(args):
    """Dummy memory estimator for testing."""
    task_id, sleep_time = args
    # Simulate different memory requirements
    memory_estimate = int(1e8 * sleep_time)  # 100MB per second of sleep
    return task_id, memory_estimate

def test_torch_multiprocessing():
    """Test torch.multiprocessing implementation."""
    print("\n=== Testing Torch Multiprocessing ===")
    
    # Create test tasks
    test_tasks = [
        ("task_1", 0.1),
        ("task_2", 0.2),
        ("task_3", 0.1),
        ("task_4", 0.3),
    ]
    
    print(f"Running {len(test_tasks)} tasks with torch.multiprocessing...")
    
    start_time = time.time()
    results = utilities.mt_runner(
        dummy_worker,
        test_tasks,
        num_threads=2,
        progress_desc="Testing multiprocessing",
        gpu_memory_estimator=dummy_memory_estimator,
    )
    elapsed = time.time() - start_time
    
    print(f"Completed in {elapsed:.2f} seconds")
    print("Results:")
    for result in results:
        print(f"  {result}")

def test_device_info():
    """Test device information functions."""
    print("\n=== Testing Device Information ===")
    
    if utilities.TORCH_AVAILABLE:
        # Import rawproc to test device functions
        from rawnind.libs import rawproc
        
        device_info = rawproc.get_device_info()
        print(f"Device info: {device_info}")
        print(f"Device type: {rawproc.get_device_type()}")
        print(f"Device count: {rawproc.get_device_count()}")
        print(f"Device name: {rawproc.get_device_name()}")
        print(f"Accelerator available: {rawproc.is_accelerator_available()}")
    else:
        print("PyTorch not available, skipping device tests")

if __name__ == "__main__":
    print("GPU Memory Scheduler and Torch Multiprocessing Test")
    print("=" * 60)
    
    test_device_info()
    test_gpu_scheduler()
    test_torch_multiprocessing()
    
    print("\n=== Test Complete ===")