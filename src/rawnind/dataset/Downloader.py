import httpx
import trio
from pathlib import Path
from tqdm import tqdm
import logging

from .SceneInfo import ImageInfo

logger = logging.getLogger(__name__)

class Downloader:
    """Downloads missing files from remote source."""
    
    def __init__(
        self,
        max_concurrent: int = 5,
        max_retries: int = 3,
        progress: bool = True
    ):
        """
        Initialize downloader.
        
        Args:
            max_concurrent: Maximum concurrent downloads
            max_retries: Maximum retry attempts per file
            progress: Show progress bar
        """
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.progress = progress
    
    async def consume_missing(
        self,
        recv_channel: trio.MemoryReceiveChannel,
        downloaded_send: trio.MemorySendChannel
    ) -> None:
        """
        Download missing files with concurrency control.
        
        Args:
            recv_channel: Receives ImageInfo for missing files
            downloaded_send: Sends ImageInfo for successfully downloaded files
        """
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
        """Download file with retry logic."""
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
                    logger.error(f"Failed to download {img_info.filename} after {self.max_retries} attempts")
                    return False
        return False
    
    async def _download_file(self, url: str, dest_path: Path) -> None:
        """Download a file asynchronously."""
        dest_path_trio = trio.Path(dest_path)
        await dest_path_trio.parent.mkdir(parents=True, exist_ok=True)
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                
                async with await dest_path_trio.open("wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        await f.write(chunk)
