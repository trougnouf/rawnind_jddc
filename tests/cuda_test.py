#!/usr/bin/env python3
"""Quick CUDA functionality test"""

def test_cuda():
    print("=== CUDA Functionality Test ===")
    
    # Test 1: CuPy import
    try:
        import cupy as cp
        print("✓ CuPy imported successfully")
    except ImportError as e:
        print(f"✗ CuPy import failed: {e}")
        return False
    
    # Test 2: Device count
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
        print(f"✓ Found {device_count} CUDA device(s)")
        if device_count == 0:
            print("✗ No CUDA devices available")
            return False
    except Exception as e:
        print(f"✗ Device count check failed: {type(e).__name__}: {e}")
        return False
    
    # Test 3: Basic array operations
    try:
        test_array = cp.array([1, 2, 3, 4, 5])
        result = cp.sum(test_array)
        print(f"✓ Basic array operation: sum([1,2,3,4,5]) = {result}")
    except Exception as e:
        print(f"✗ Basic array operation failed: {type(e).__name__}: {e}")
        return False
    
    # Test 4: Memory allocation
    try:
        large_array = cp.zeros((1000, 1000), dtype=cp.float32)
        mean_val = cp.mean(large_array)
        print(f"✓ Memory allocation test: 1000x1000 array mean = {mean_val}")
    except Exception as e:
        print(f"✗ Memory allocation failed: {type(e).__name__}: {e}")
        return False
    
    print("✓ All CUDA tests passed!")
    return True

if __name__ == "__main__":
    success = test_cuda()
    exit(0 if success else 1)
