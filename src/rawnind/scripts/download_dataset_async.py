#!/usr/bin/env python3
"""
Trio-based dataset downloader script for RawNIND.

This thin wrapper delegates to DatasetIndex.async_download_missing_files, which
handles index loading, local discovery, progress reporting, downloads, and
final validation. A summary is printed afterward for convenience.
"""

import sys

import trio

from rawnind.dataset.manager import DatasetIndex


async def main() -> None:
    """Entrypoint executed within a Trio event loop."""
    index = DatasetIndex()
    await index.async_download_missing_files(max_concurrent=5)
    index.print_summary()


if __name__ == "__main__":
    trio.run(main)
    sys.exit(0)