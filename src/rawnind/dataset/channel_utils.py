"""Utilities for trio channel operations in pipeline architectures."""

import trio
from typing import Dict, Optional, Callable, TypeVar, Any

T = TypeVar('T')


def create_channel_dict(names: list[str], buffer_size: int) -> Dict[str, Any]:
    """
    Create a dictionary of send/recv channel pairs.

    Args:
        names: List of channel names (e.g., ['scene', 'verified', 'enriched'])
        buffer_size: Buffer size for memory channels

    Returns:
        Dictionary with '{name}_send' and '{name}_recv' keys

    Example:
        channels = create_channel_dict(['scene', 'verified'], buffer_size=10)
        # Creates: scene_send, scene_recv, verified_send, verified_recv
    """
    channels = {}
    for name in names:
        send, recv = trio.open_memory_channel(buffer_size)
        channels[f'{name}_send'] = send
        channels[f'{name}_recv'] = recv
    return channels


async def merge_channels(recv1, recv2, send):
    """
    Merge two receive channels into one send channel.

    Items from both input channels are forwarded to the output channel.
    Closes send channel when both inputs are exhausted.

    Args:
        recv1: First receive channel
        recv2: Second receive channel
        send: Output send channel
    """
    async with send:
        async with trio.open_nursery() as nursery:
            async def forward(recv):
                async with recv:
                    async for item in recv:
                        await send.send(item)

            nursery.start_soon(forward, recv1)
            nursery.start_soon(forward, recv2)


async def limit_producer(
    producer_func: Callable,
    max_items: Optional[int],
    send_channel,
    buffer_size: int,
    *args,
    **kwargs
):
    """
    Wrap a producer function to limit output to max_items.

    If max_items is None, forwards all items. Otherwise, cancels
    the producer after max_items have been sent.

    Args:
        producer_func: Async function that produces items (takes send channel)
        max_items: Maximum items to produce (None = unlimited)
        send_channel: Destination channel for limited output
        buffer_size: Buffer size for internal channel
        *args, **kwargs: Additional arguments for producer_func

    Example:
        await limit_producer(
            ingestor.produce_scenes,
            max_items=10,
            send_channel=output_channel,
            buffer_size=50
        )
    """
    if max_items is None:
        await producer_func(*args, send_channel, **kwargs)
        return

    internal_send, internal_recv = trio.open_memory_channel(buffer_size)

    async with trio.open_nursery() as nursery:
        nursery.start_soon(producer_func, *args, internal_send, **kwargs)

        item_count = 0
        async with internal_recv, send_channel:
            async for item in internal_recv:
                await send_channel.send(item)
                item_count += 1
                if item_count >= max_items:
                    nursery.cancel_scope.cancel()
                    break


async def consume_until(
    recv_channel,
    max_items: Optional[int],
    nursery,
    on_item: Optional[Callable[[T], Any]] = None
):
    """
    Consume from a channel until max_items reached, then cancel nursery.

    Args:
        recv_channel: Channel to consume from
        max_items: Maximum items to consume before canceling (None = unlimited)
        nursery: Trio nursery to cancel when limit reached
        on_item: Optional async callback to invoke for each item

    Returns:
        Number of items consumed

    Example:
        async with nursery:
            # ... start pipeline stages
            consumed = await consume_until(
                final_channel,
                max_items=100,
                nursery=nursery,
                on_item=lambda scene: viz.update(complete=1)
            )
    """
    async with recv_channel:
        consumed = 0
        async for item in recv_channel:
            if on_item:
                await on_item(item)
            consumed += 1
            if max_items is not None and consumed >= max_items:
                nursery.cancel_scope.cancel()
                break
        return consumed