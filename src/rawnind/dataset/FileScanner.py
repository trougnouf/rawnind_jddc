from pathlib import Path
from typing import List

import trio

from .SceneInfo import ImageInfo, SceneInfo


class FileScanner:
    """Scans filesystem for files mentioned in new_item image_info."""

    def __init__(self, dataset_root: Path):
        """
        Initialize filesystem scanner.
        
        Args:
            dataset_root: Root directory for dataset files
        """
        self.dataset_root = dataset_root

    async def consume_new_items(
            self,
            recv_channel: trio.MemoryReceiveChannel,
            new_file_send: trio.MemorySendChannel,
            missing_send: trio.MemorySendChannel
    ) -> None:
        """
        Scan filesystem for files and route accordingly.

        Args:
            recv_channel: Receives SceneInfo objects
            new_file_send: Sends ImageInfo for discovered files
            missing_send: Sends ImageInfo for missing files
        """
        async with recv_channel, new_file_send, missing_send:
            async for scene_info in recv_channel:
                # Compute scene directory once for efficiency
                scene_dir = self.dataset_root / scene_info.cfa_type / scene_info.scene_name
                gt_dir = scene_dir / "gt"

                # Process all images in the scene
                for img_info in scene_info.all_images():
                    candidates = self._candidate_paths_for_scene(scene_dir, gt_dir, img_info)

                    found = False
                    for candidate in candidates:
                        candidate_trio = trio.Path(candidate)
                        if await candidate_trio.exists():
                            img_info.local_path = candidate
                            await new_file_send.send(img_info)
                            found = True
                            break

                    if not found:
                        # Set primary candidate as download target
                        img_info.local_path = candidates[0] if candidates else None
                        await missing_send.send(img_info)

                await trio.sleep(0)

    def _candidate_paths_for_scene(
            self,
            scene_dir: Path,
            gt_dir: Path,
            img_info: ImageInfo
    ) -> List[Path]:
        """Generate candidate paths for an image file within a scene directory."""
        if img_info.is_clean:
            return [
                gt_dir / img_info.filename,
                scene_dir / img_info.filename
            ]
        else:
            return [scene_dir / img_info.filename]
