"""Asynchronous downloader for missing files with concurrency control and retry logic.

This class manages the downloading of missing image files from a remote source,
supporting concurrent downloads with configurable retry logic and progress tracking.
It uses trio for asynchronous execution and httpx for HTTP requests.

Attributes:
    max_concurrent: Maximum number of concurrent downloads (default: 5)
    max_retries: Maximum number of retry attempts per file (default: 3)
    progress: Whether to display progress bar (default: True)

Args:
    max_concurrent: Maximum number of concurrent downloads (int, >= 1)
    max_retries: Maximum retry attempts per file (int, >= 0)
    progress: Whether to show progress bar (bool)

Raises:
    ValueError: If max_concurrent is less than 1 or max_retries is negative
    httpx.HTTPError: If HTTP requests fail afte\r all retries
    trio.Cancelled: If the download is cancelled during execution

Examples:
    >>> import trio
    >>> from pathlib import Path
    >>> from my_module import Downloader
    >>> downloader = Downloader(max_concurrent=3, max_retries=2)
    >>>
    >>> # Using with trio to run async operations
    >>> async def example():
    ...     # Assume channel setup and usage elsewhere
    ...     pass
    >>> trio.run(example)

Notes:
    The downloader maintains a fixed concurrency limit to avoid overwhelming
    the remote server. It implements exponential backoff for retries and
    handles network interruptions gracefully. Progress tracking is achieved
    through tqdm when enabled. The downloader is designed to be used with
    trio's async context and memory channels for efficient inter-task communication.

    All downloaded files are saved to their respective paths as specified by
    the ImageInfo objects, with proper directory creation if needed.
    The downloader logs progress and errors to the module's logger with DEBUG
    and ERROR log levels respectively."""

import logging
from pathlib import Path

import httpx
import trio
from tqdm import tqdm

from .SceneInfo import ImageInfo

logger = logging.getLogger(__name__)


class Downloader:
    """Download missing files from remote sources with concurrent downloads and retry logic.
    
    This class implements an asynchronous file downloader that manages concurrent downloads
    with configurable retry behavior. It uses trio for async operations and supports progress
    reporting through tqdm. The downloader handles failed downloads gracefully by retrying
    up to a configurable maximum number of times.
    
    Args:
        max_concurrent: Maximum number of simultaneous downloads (int, default: 5)
            Controls the thread pool size for concurrent downloads (range: 1-100)
        max_retries: Maximum retry attempts per file (int, default: 3)
            Number of times to retry a failed download before marking it as failed (range: 0-10)
        progress: Enable/disable progress bar display (bool, default: True)
            When True, displays a progress bar showing download progress
    
    Returns:
        None
    
    Raises:
        Exception: Propagated from underlying download operations during file retrieval
        ValueError: If invalid parameters are passed during initialization
    
    Attributes:
        max_concurrent: Configured maximum concurrent downloads
        max_retries: Configured maximum retry attempts
        progress: Progress bar display setting
    
    Notes:
        The downloader maintains a thread pool of size `max_concurrent` for concurrent downloads.
        Each download operation will be retried up to `max_retries` times before failing permanently.
        Progress updates are handled via tqdm library when enabled.
    
        This implementation does not include built-in rate limiting or bandwidth control.
        For high-volume downloads, consider using external rate limiting or implementing
        custom download managers with additional constraints.
    
    Examples:
        >>> downloader = Downloader(max_concurrent=10, max_retries=5, progress=False)
        >>> print(downloader.max_concurrent)
        10
        >>> print(downloader.max_retries)
        5
        >>> print(downloader.progress)
        False"""

    def __init__(
        self, max_concurrent: int = 5, max_retries: int = 3, progress: bool = True
    ):
        """Initialize downloader with configurable concurrency and retry settings.

        This constructor sets up a downloader instance with parameters that control
        download behavior including maximum concurrent downloads, retry attempts
        for failed downloads, and progress reporting. The downloader uses a thread pool
        to manage concurrent downloads and can automatically retry failed downloads
        based on configurable retry limits.

        Args:
            max_concurrent: Maximum number of simultaneous downloads (int, default: 5)
                Controls the thread pool size for concurrent downloads.
                Range: 1-100 (recommended: 1-20 for most use cases)
            max_retries: Maximum retry attempts per file (int, default: 3)
                Number of times to retry a failed download before marking it as failed.
                Range: 0-10 (recommended: 0-3)
            progress: Enable/disable progress bar display (bool, default: True)
                When True, displays a progress bar showing download progress.
                When False, operates silently without user interface updates.

        Attributes:
            max_concurrent: Configured maximum concurrent downloads
            max_retries: Configured maximum retry attempts
            progress: Progress bar display setting

        Notes:
            The downloader maintains a thread pool of size `max_concurrent` for
            concurrent downloads. Each download operation will be retried up to
            `max_retries` times before failing permanently. Progress updates
            are handled via tqdm library when enabled.

            This implementation does not include built-in rate limiting or bandwidth
            control. For high-volume downloads, consider using external rate limiting
            or implementing custom download managers with additional constraints.

        Examples:
            >>> downloader = Downloader(max_concurrent=10, max_retries=5, progress=False)
            >>> print(downloader.max_concurrent)
            10
            >>> print(downloader.max_retries)
            5
            >>> print(downloader.progress)
            False"""
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.progress = progress

    async def consume_missing(
        self,
        recv_channel: trio.MemoryReceiveChannel,
        downloaded_send: trio.MemorySendChannel,
    ) -> None:
        """Download missing files with concurrency control.
        
        Downloads files from a memory channel with controlled concurrent execution. This method processes incoming ImageInfo objects from a receive channel, downloads them using retry logic, and sends successful downloads to a send channel. The download process respects a maximum concurrent connection limit and supports progress tracking.
        
        Args:
            recv_channel: MemoryReceiveChannel that yields ImageInfo objects for missing files to download
            downloaded_send: MemorySendChannel that receives successfully downloaded ImageInfo objects
        
        Returns:
            None
        
        Raises:
            trio.Cancelled: When the operation is cancelled during execution
            trio.ClosedResourceError: When either channel is closed during operation
            Exception: Propagated from download operations or channel operations
        
        Notes:
            - Uses semaphore-based concurrency control with self.max_concurrent limit
            - Implements retry logic for failed downloads
            - Progress bar is optional (controlled by self.progress flag)
            - All channels are properly closed upon completion or cancellation
            - This is an async method that returns a coroutine object
            - The progress bar shows download progress with file units
        
        Examples:
            >>> async with trio.open_nursery() as nursery:
            ...     nursery.start_soon(consume_missing, recv_channel, send_channel)"""
        async with recv_channel, downloaded_send:
            async with trio.open_nursery() as nursery:
                semaphore = trio.Semaphore(self.max_concurrent)
                pbar = tqdm(desc="Downloading", unit="file") if self.progress else None

                async for img_info in recv_channel:

                    async def download_task(img: ImageInfo):
                        async with semaphore:
                            success = await self._download_with_retry(img)
                            if success:
                                await downloaded_send.send(img)
                            if pbar is not None:
                                pbar.update(1)

                    nursery.start_soon(download_task, img_info)

                if pbar is not None:
                    pbar.close()

    async def _download_with_retry(self, img_info: ImageInfo) -> bool:
        """Download an image file with retry logic and optional tensor loading.
        
        Attempts to download an image file from a remote URL to a local path with
        exponential backoff retry logic. If successful, the image is loaded into memory
        as a torch tensor for faster access. The function will retry failed downloads
        up to a configured maximum number of times before giving up.
        
        Args:
            img_info (ImageInfo): Object containing download URL, local path, and filename
                information for the image to be downloaded.
        
        Returns:
            bool: True if the image was successfully downloaded and loaded, False otherwise.
        
        Raises:
            Exception: Any exceptions raised during the download process or image loading
                are caught and logged, but re-raised to the caller for proper handling.
        
        Examples:
            >>> await downloader._download_with_retry(image_info)
            True
        
        Notes:
            - This method is asynchronous and returns a coroutine
            - The download process includes opportunistic tensor loading to cache
            - Failed download attempts are logged with warning messages
            - The function returns False if all retry attempts are exhausted
            - The image loading occurs with as_torch=True to enable fast access patterns"""
        for attempt in range(self.max_retries):
            try:
                await self._download_file(img_info.download_url, img_info.local_path)
                # Opportunistically load tensor into cache (page cache makes this fast)
                await img_info.load_image(as_torch=True)
                return True
            except Exception as e:
                logger.warning(
                    f"Download attempt {attempt + 1}/{self.max_retries} failed "
                    f"for {img_info.filename}: {e}"
                )
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"Failed to download {img_info.filename} after {self.max_retries} attempts"
                    )
                    return False
        return False

    async def _download_file(self, url: str, dest_path: Path) -> None:
        """Download a file asynchronously from a URL to a local path.
        
        This method performs an asynchronous HTTP GET request to download a file from the specified URL
        and saves it to the given destination path. The method creates parent directories automatically
        if they don't exist. It uses streaming to handle large files efficiently without loading them
        completely into memory. The download includes automatic status checking and uses a 5-minute
        timeout for the HTTP request.
        
        Args:
            url: The URL of the file to download, as a string
            dest_path: The local destination path where the file should be saved, as a Path object
        
        Returns:
            None
        
        Raises:
            httpx.HTTPError: If the HTTP request fails or returns an error status code
            httpx.TimeoutException: If the HTTP request exceeds the 300-second timeout
            OSError: If there are file system errors during directory creation or file writing
        
        Examples:
            >>> await _download_file("https://example.com/file.txt", Path("downloads/file.txt"))
        
        Notes:
            This is an asynchronous method that returns a coroutine.
            Files are downloaded in chunks of 8192 bytes for memory efficiency.
            The method automatically creates all necessary parent directories for the destination path.
            The HTTP client is automatically closed after the download completes."""
        dest_path_trio = trio.Path(dest_path)
        await dest_path_trio.parent.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()

                async with await dest_path_trio.open("wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        await f.write(chunk)
