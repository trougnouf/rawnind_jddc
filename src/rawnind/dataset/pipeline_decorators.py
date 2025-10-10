"""
Decorators for pipeline stages - handle ALL cross-cutting concerns.
Progress, errors, retries, timeouts, channels, concurrency, validation, caching, rate-limiting.
"""

import functools
import logging
from typing import Optional, Callable, Any, TypeVar
import trio

logger = logging.getLogger(__name__)

T = TypeVar('T')


def stage(*, progress: Optional[tuple[str, str]] = None, retries: int = 0, timeout: Optional[float] = None,
          log_timing: bool = False, concurrency_limit: Optional[int] = None, debug_on_: bool):
    """
    Universal decorator for pipeline stage methods.
    Handles progress, errors, timing, retries, concurrency - everything.

    Usage:
        @stage(progress=("enriching", "enriched"), retries=2, log_timing=True, timeout=30)
        async def enrich_scene(self, scene_info):
            return enriched_scene

    Args:
        progress: Tuple of (active_counter, complete_counter) for tracking
        retries: Number of retry attempts on failure
        timeout: Timeout in seconds for the operation
        log_timing: Whether to log execution time
        concurrency_limit: Max concurrent executions (uses semaphore)
        debug_on_: If True, return input unchanged on error instead of raising
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            self = args[0] if args else None
            viz = getattr(self, '_viz', None) if self else None

            # Set up concurrency semaphore if needed
            semaphore = None
            if concurrency_limit and self:
                semaphore_attr = f'_sem_{func.__name__}'
                if not hasattr(self, semaphore_attr):
                    setattr(self, semaphore_attr, trio.Semaphore(concurrency_limit))
                semaphore = getattr(self, semaphore_attr)

            async def _execute_with_tracking():
                # Progress tracking - mark as active
                if progress and viz:
                    await viz.update(**{progress[0]: 1})

                # Timing
                start_time = trio.current_time() if log_timing else None

                # Retry loop
                last_exception = None
                for attempt in range(retries + 1):
                    try:
                        # Timeout control
                        if timeout:
                            with trio.move_on_after(timeout) as cancel_scope:
                                result = await func(*args, **kwargs)
                                if cancel_scope.cancelled_caught:
                                    raise TimeoutError(f"{func.__name__} timed out after {timeout}s")
                        else:
                            result = await func(*args, **kwargs)

                        # Success - log timing
                        if log_timing and start_time:
                            elapsed = trio.current_time() - start_time
                            logger.debug(f"{func.__name__} completed in {elapsed:.2f}s")

                        # Progress tracking - mark as complete
                        if progress and viz:
                            await viz.update(**{progress[0]: -1, progress[1]: 1})

                        return result

                    except Exception as e:
                        last_exception = e
                        if attempt < retries:
                            # Reasonable exponential backoff: 1s, 2s, 4s, 8s (max 30s)
                            # Avoids blocking event loop for 30s+ on first retry
                            wait_time = min(2 ** attempt, 30)
                            logger.warning(
                                f"{func.__name__} failed (attempt {attempt + 1}/{retries + 1}): {e}. "
                                f"Retrying in {wait_time}s..."
                            )
                            await trio.sleep(wait_time)
                        else:
                            # All retries exhausted
                            if progress and viz:
                                await viz.update(**{progress[0]: -1, "errors": 1})
                            if log_timing and start_time:
                                elapsed = trio.current_time() - start_time
                                logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {e}")

                            if debug_on_:
                                # Return input unchanged
                                return args[1] if len(args) > 1 else None
                            raise

                raise last_exception

            # Apply concurrency control if needed
            if semaphore:
                async with semaphore:
                    return await _execute_with_tracking()
            else:
                return await _execute_with_tracking()

        return wrapper
    return decorator


def channel_processor(
    auto_close: bool = True,
    forward_errors: bool = False
):
    """
    Decorator for functions that process items from channels.
    Handles channel lifecycle and optionally forwards errors.

    Usage:
        @channel_processor(auto_close=True)
        async def process_images(self, recv, send):
            async for img in recv:
                result = await self.process(img)
                await send.send(result)

    Args:
        auto_close: Automatically manage channel lifecycle with async with
        forward_errors: Continue processing on errors instead of raising
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract recv/send channels
            recv = kwargs.get('recv') or (args[1] if len(args) > 1 else None)
            send = kwargs.get('send') or (args[2] if len(args) > 2 else None)

            if auto_close and recv and send:
                async with recv, send:
                    return await func(*args, **kwargs)
            elif auto_close and recv:
                async with recv:
                    return await func(*args, **kwargs)
            else:
                return await func(*args, **kwargs)

        return wrapper
    return decorator


def validate_input(validator: Callable[[Any], bool], error_msg: str = "Validation failed"):
    """
    Decorator to validate function inputs.

    Usage:
        @validate_input(lambda scene: scene.get_gt_image() is not None, "Missing GT image")
        async def process_scene(self, scene_info):
            return processed_scene
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get first non-self argument
            item = args[1] if len(args) > 1 else kwargs.get('item')

            if item and not validator(item):
                logger.error(f"{func.__name__}: {error_msg}")
                raise ValueError(f"{error_msg}: {item}")

            return await func(*args, **kwargs)

        return wrapper
    return decorator


def cache_result(key_func: Callable, ttl: Optional[float] = None):
    """
    Decorator to cache async function results with optional TTL.
    Uses promise pattern - multiple concurrent requests for same key share computation.

    Usage:
        @cache_result(key_func=lambda self, scene: scene.scene_name, ttl=3600)
        async def compute_expensive_thing(self, scene_info):
            return result

    Args:
        key_func: Function to generate cache key from args
        ttl: Time-to-live in seconds (None = forever)
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_times = {}
        in_progress = {}  # key -> Event for pending computations
        lock = trio.Lock()

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key = key_func(*args, **kwargs)

            # Check cache first (fast path, no lock needed for read)
            async with lock:
                # Check cache validity
                if key in cache:
                    if ttl is None or (trio.current_time() - cache_times[key]) < ttl:
                        logger.debug(f"{func.__name__}: cache hit for {key}")
                        return cache[key]

                # Check if computation already in progress
                if key in in_progress:
                    event = in_progress[key]
                    is_waiting = True
                    logger.debug(f"{func.__name__}: waiting for in-progress computation for {key}")
                else:
                    # We're the first - create event and mark in progress
                    event = trio.Event()
                    in_progress[key] = event
                    is_waiting = False

            # If we're waiting for another task's computation
            if is_waiting:
                await event.wait()
                # After waking, check cache again
                async with lock:
                    if key in cache:
                        return cache[key]
                    # Computation failed, let this task retry
                    return await wrapper(*args, **kwargs)

            # We're the task that will compute
            try:
                result = await func(*args, **kwargs)

                # Store in cache
                async with lock:
                    cache[key] = result
                    cache_times[key] = trio.current_time()
                    in_progress.pop(key, None)

                event.set()  # Wake waiters
                return result
            except Exception:
                # Remove from in_progress on error so retries can happen
                async with lock:
                    in_progress.pop(key, None)
                event.set()  # Wake waiters (they'll retry)
                raise

        return wrapper
    return decorator


def rate_limit(calls_per_second: float):
    """
    Decorator to rate-limit async function calls using token bucket.

    Usage:
        @rate_limit(calls_per_second=10)
        async def download_file(self, url):
            return data
    """
    def decorator(func: Callable) -> Callable:
        min_interval = 1.0 / calls_per_second
        last_call_time = [0.0]  # Mutable container for closure
        lock = trio.Lock()

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            async with lock:
                now = trio.current_time()
                time_since_last = now - last_call_time[0]

                if time_since_last < min_interval:
                    await trio.sleep(min_interval - time_since_last)

                last_call_time[0] = trio.current_time()

            return await func(*args, **kwargs)

        return wrapper
    return decorator


def batch_process(batch_size: int, timeout: Optional[float] = 5.0):
    """
    Decorator to batch items from a channel before processing.
    Flushes partial batches on timeout to prevent stalling.

    Usage:
        @batch_process(batch_size=10, timeout=5.0)
        async def process_batch(self, recv, send):
            # Automatically batches items before calling
            pass

    Args:
        batch_size: Number of items to accumulate before processing
        timeout: Max seconds to wait for full batch (None = wait forever)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(self, recv_channel, send_channel):
            async with recv_channel, send_channel:
                batch = []
                last_process_time = trio.current_time()

                async def process_and_send():
                    nonlocal batch, last_process_time
                    if batch:
                        await func(self, batch)
                        for item in batch:
                            await send_channel.send(item)
                        batch = []
                        last_process_time = trio.current_time()

                async for item in recv_channel:
                    batch.append(item)

                    # Flush on size threshold
                    if len(batch) >= batch_size:
                        await process_and_send()

                    # Flush on timeout (if enabled)
                    elif timeout and (trio.current_time() - last_process_time) >= timeout:
                        await process_and_send()

                # Process remaining items
                await process_and_send()

        return wrapper
    return decorator


def compose(*decorators):
    """
    Compose multiple decorators into one.

    Usage:
        @compose(
            stage(progress=("processing", "processed"), retries=2),
            validate_input(lambda x: x.is_valid),
            cache_result(key_func=lambda self, x: x.id, ttl=3600)
        )
        async def process_item(self, item):
            return result
    """
    def decorator(func):
        for dec in reversed(decorators):
            func = dec(func)
        return func
    return decorator