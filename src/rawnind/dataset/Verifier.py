import hashlib
import logging
from pathlib import Path

import trio

logger = logging.getLogger(__name__)

def hash_sha1(file_path: Path) -> str:
    """Compute SHA-1 hash of a file (blocking)."""
    sha1 = hashlib.sha1()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha1.update(chunk)
    return sha1.hexdigest()


class Verifier:
    """Verifies file integrity using SHA-1 hash validation."""

    def __init__(self, max_retries: int = 3):
        """Initialize verifier.

        Args:
            max_retries: Maximum number of verification retries per file
        """
        self.max_retries = max_retries
    
    async def consume_new_files(
        self,
        recv_channel: trio.MemoryReceiveChannel,
        verified_send: trio.MemorySendChannel,
        missing_send: trio.MemorySendChannel
    ) -> None:
        """
        Verify files and route to verified or missing channels.
        
        Args:
            recv_channel: Receives ImageInfo with local_path set
            verified_send: Sends verified ImageInfo
            missing_send: Sends ImageInfo for corrupted/missing files
        """
        async with recv_channel, verified_send, missing_send:
            async for img_info in recv_channel:
                if img_info.local_path is None or not img_info.local_path.exists():
                    if img_info.retry_count < self.max_retries:
                        img_info.retry_count += 1
                        await missing_send.send(img_info)
                    else:
                        logger.error(
                            f"File {img_info.filename} failed verification after "
                            f"{self.max_retries} retries. Skipping."
                        )
                    continue

                # Verify hash (run in thread to avoid blocking)
                computed_hash = await trio.to_thread.run_sync(
                    hash_sha1,
                    img_info.local_path
                )

                if computed_hash == img_info.sha1:
                    img_info.validated = True
                    await verified_send.send(img_info)
                else:
                    logger.warning(
                        f"Hash mismatch for {img_info.filename}: "
                        f"expected {img_info.sha1}, got {computed_hash}"
                    )

                    img_info.local_path.unlink()
                    logger.info(f"Deleted corrupted file: {img_info.local_path}")

                    img_info.validated = False
                    img_info.local_path = None

                    if img_info.retry_count < self.max_retries:
                        img_info.retry_count += 1
                        await missing_send.send(img_info)
                    else:
                        logger.error(
                            f"File {img_info.filename} failed verification after "
                            f"{self.max_retries} retries. Skipping."
                        )
