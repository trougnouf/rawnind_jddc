from pathlib import Path
from typing import Dict, Generator, Optional, Set, Tuple

import trio

from .SceneInfo import SceneInfo, ImageInfo


class SceneIndexer:
    """Consumes images and produces complete scenes when all images arrive."""

    def __init__(self, dataset_root: Path):
        """
        Initialize dataset index.

        Args:
            dataset_root: Root directory for dataset files
        """
        self.dataset_root = dataset_root
        self._scenes: Dict[str, Dict[str, SceneInfo]] = {}
        self._images: Dict[str, ImageInfo] = {}  # Indexed by sha1
        self._scene_completion_tracker: Set[Tuple[str, str]] = set()

    async def consume_images_produce_scenes(
            self,
            image_recv_channel: trio.MemoryReceiveChannel,
            scene_send_channel: trio.MemorySendChannel
    ) -> None:
        """
        Consume images and produce complete scenes.

        Args:
            image_recv_channel: Receives ImageInfo objects
            scene_send_channel: Sends complete SceneInfo objects
        """
        async with image_recv_channel, scene_send_channel:
            async for img_info in image_recv_channel:
                self._add_image_to_index(img_info)

                # Check if scene is complete
                scene_key = (img_info.cfa_type, img_info.scene_name)
                if scene_key not in self._scene_completion_tracker:
                    if self._is_scene_complete(img_info):
                        scene_info = self._construct_scene(img_info)
                        self._scene_completion_tracker.add(scene_key)
                        self._move_scene_to_complete(scene_info)
                        await scene_send_channel.send(scene_info)

    def _add_image_to_index(self, img_info: ImageInfo) -> None:
        """Add an image to the index."""
        # Add to image index (incomplete scenes tracked here)
        self._images[img_info.sha1] = img_info

    def _is_scene_complete(self, img_info: ImageInfo) -> bool:
        """Check if all images in the scene are present in the image index."""
        # Check if all SHA1s from scene_images list are in the image index
        expected_sha1s = set(img_info.scene_images)

        for sha1 in expected_sha1s:
            if sha1 not in self._images:
                return False

        return True

    def _construct_scene(self, img_info: ImageInfo) -> SceneInfo:
        """Construct a SceneInfo from images in the image index."""
        clean_images = []
        noisy_images = []

        # Gather all images for this scene from the image index
        for sha1 in img_info.scene_images:
            img = self._images.get(sha1)
            if img:
                if img.is_clean:
                    clean_images.append(img)
                else:
                    noisy_images.append(img)

        scene_info = SceneInfo(
            scene_name=img_info.scene_name,
            cfa_type=img_info.cfa_type,
            unknown_sensor=False,  # TODO: get from metadata
            test_reserve=False,  # TODO: get from metadata
            clean_images=clean_images,
            noisy_images=noisy_images
        )

        # Add to scene index
        if img_info.cfa_type not in self._scenes:
            self._scenes[img_info.cfa_type] = {}
        self._scenes[img_info.cfa_type][img_info.scene_name] = scene_info

        return scene_info

    def _move_scene_to_complete(self, scene_info: SceneInfo) -> None:
        """Remove images of a complete scene from the image index."""
        for img in scene_info.all_images():
            self._images.pop(img.sha1, None)

    def iter_complete_scenes(self) -> Generator[SceneInfo, None, None]:
        """Yield only complete scenes."""
        for cfa_type, scenes in self._scenes.items():
            for scene_info in scenes.values():
                scene_key = (cfa_type, scene_info.scene_name)
                if scene_key in self._scene_completion_tracker:
                    yield scene_info

    def iter_incomplete_images(self, verified_only: bool = False) -> Generator[ImageInfo, None, None]:
        """Yield images from incomplete scenes, optionally filtering to verified only."""
        for img in self._images.values():
            if not verified_only or img.validated:
                yield img
