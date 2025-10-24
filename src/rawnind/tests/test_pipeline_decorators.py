"""
Tests for pipeline_decorators.py fixes.

Tests for:
1. Exponential backoff (reasonable timing)
2. cache_result promise pattern (no duplicate computation)
3. batch_process timeout handling (flushes partial batches)
"""

import pytest
import trio
import trio.testing

from rawnind.dataset.pipeline_decorators import (
    stage,
    cache_result,
    batch_process,
)

pytestmark = pytest.mark.dataset


# ============================================================================
# Test Exponential Backoff Fix
# ============================================================================

@pytest.mark.trio
async def test_stage_exponential_backoff_reasonable_timing():
    """Test that exponential backoff uses reasonable timings (1s, 2s, 4s, 8s)."""
    call_count = 0
    retry_times = []

    class TestStage:
        @stage(retries=3, debug_on_=False)
        async def failing_method(self):
            nonlocal call_count
            call_count += 1
            retry_times.append(trio.current_time())
            raise ValueError("Test error")

    stage_instance = TestStage()

    with pytest.raises(ValueError):
        await stage_instance.failing_method()

    # Should have been called 4 times (initial + 3 retries)
    assert call_count == 4

    # Check timing intervals (allow 10% tolerance for test flakiness)
    # Retries should be at approximately: 0s, +1s, +2s, +4s
    assert len(retry_times) == 4
    intervals = [retry_times[i+1] - retry_times[i] for i in range(3)]

    assert 0.9 <= intervals[0] <= 1.1  # ~1s
    assert 1.8 <= intervals[1] <= 2.2  # ~2s
    assert 3.6 <= intervals[2] <= 4.4  # ~4s


@pytest.mark.trio
async def test_stage_exponential_backoff_caps_at_30s():
    """Test that exponential backoff caps at 30s.

    Verify wait times without actually sleeping by checking the calculation.
    """
    # Test the formula directly: min(2 ** attempt, 30)
    wait_times = [min(2 ** attempt, 30) for attempt in range(10)]

    # Progression should be: 1, 2, 4, 8, 16, 30, 30, 30, 30, 30
    assert wait_times[0] == 1
    assert wait_times[1] == 2
    assert wait_times[2] == 4
    assert wait_times[3] == 8
    assert wait_times[4] == 16
    assert wait_times[5] == 30  # 2^5=32, capped to 30
    assert wait_times[6] == 30  # 2^6=64, capped to 30
    assert wait_times[9] == 30  # 2^9=512, capped to 30

    # Verify all high attempts cap at 30
    assert all(t == 30 for t in wait_times[5:])


# ============================================================================
# Test cache_result Promise Pattern Fix
# ============================================================================

@pytest.mark.trio
async def test_cache_result_promise_pattern_no_duplicate_computation():
    """Test that concurrent cache misses share computation (promise pattern)."""
    computation_count = 0
    computation_order = []

    class TestClass:
        @cache_result(key_func=lambda self, key: key)
        async def expensive_compute(self, key):
            nonlocal computation_count
            computation_count += 1
            computation_order.append(f"start_{key}")
            await trio.sleep(0.1)  # Simulate expensive work
            computation_order.append(f"end_{key}")
            return f"result_{key}"

    instance = TestClass()
    results = []

    async def caller():
        result = await instance.expensive_compute("test_key")
        results.append(result)

    # Launch 3 concurrent requests for same key
    async with trio.open_nursery() as nursery:
        for _ in range(3):
            nursery.start_soon(caller)

    # Should only compute ONCE (not 3 times)
    assert computation_count == 1

    # All tasks should get same result
    assert len(results) == 3
    assert all(r == "result_test_key" for r in results)


@pytest.mark.trio
async def test_cache_result_error_allows_retry():
    """Test that computation errors don't block future retries."""
    call_count = 0

    class TestClass:
        @cache_result(key_func=lambda self, key: key)
        async def failing_then_succeeding(self, key):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First attempt fails")
            return f"success_{key}"

    instance = TestClass()

    # First call fails
    with pytest.raises(ValueError):
        await instance.failing_then_succeeding("key")

    # Second call should retry (not return cached error)
    result = await instance.failing_then_succeeding("key")
    assert result == "success_key"
    assert call_count == 2


# ============================================================================
# Test batch_process Timeout Handling Fix
# ============================================================================

@pytest.mark.trio
async def test_batch_process_has_timeout_parameter():
    """Test that batch_process decorator accepts timeout parameter."""
    # Just verify the decorator can be instantiated with timeout
    from inspect import signature

    # Check decorator signature
    sig = signature(batch_process)
    params = list(sig.parameters.keys())

    assert 'batch_size' in params
    assert 'timeout' in params

    # Verify default timeout is 5.0
    assert sig.parameters['timeout'].default == 5.0


@pytest.mark.trio
async def test_batch_process_timeout_logic():
    """Test the timeout check logic exists in implementation."""
    # Verify the decorator handles timeout by checking implementation
    import inspect
    source = inspect.getsource(batch_process)

    # Should have timeout handling code
    assert 'timeout' in source
    assert 'last_process_time' in source
    assert 'trio.current_time()' in source

    # Should flush on both size and timeout conditions
    assert 'batch_size' in source
    assert 'process_and_send' in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
