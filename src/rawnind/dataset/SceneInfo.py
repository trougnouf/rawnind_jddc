from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, ClassVar, Any
import trio


@dataclass
class ImageInfo:
    """Information about a single image file."""

    DATAVERSE_BASE_URL: ClassVar[str] = "https://dataverse.uclouvain.be"

    filename: str
    sha1: str
    is_clean: bool
    scene_name: str
    scene_images: list[str]
    cfa_type: str  # 'Bayer' or 'X-Trans'
    local_path: Optional[Path] = None
    validated: bool = False
    file_id: str = ""  # Dataverse file ID for downloads
    retry_count: int = 0  # Track verification retry attempts
    metadata: dict = field(default_factory=dict)  # Computed metadata
    xmp_path: Optional[Path] = None  # Path to associated .xmp sidecar file
    _image_tensor: Optional[Any] = field(default=None, repr=False)  # Cached tensor (torch or numpy)

    @property
    def download_url(self) -> str:
        """Construct download URL for this file."""
        if not self.file_id:
            raise ValueError(
                f"Cannot construct download URL: file_id not set for {self.filename}"
            )
        return f"{self.DATAVERSE_BASE_URL}/api/access/datafile/{self.file_id}"

    @property
    def image_tensor(self) -> Optional[Any]:
        """Get cached image tensor, or None if not loaded."""
        return self._image_tensor

    @property
    def aligned_image_tensor(self) -> Optional[Any]:
        """
        Get image tensor with alignment applied if alignment metadata exists.
        Returns None if image not loaded. Returns unshifted image if no alignment in metadata.
        """
        if self._image_tensor is None:
            return None

        alignment = self.metadata.get("alignment")
        if not alignment or alignment == [0, 0]:
            return self._image_tensor

        # Apply alignment shift
        y_offset, x_offset = alignment
        if y_offset == 0 and x_offset == 0:
            return self._image_tensor

        # Crop to aligned region (handles negative offsets)
        if self._is_torch_tensor(self._image_tensor):
            # torch tensor
            if len(self._image_tensor.shape) == 2:  # (H, W)
                h, w = self._image_tensor.shape
            else:  # (C, H, W)
                _, h, w = self._image_tensor.shape

            y_start = max(0, y_offset)
            y_end = min(h, h + y_offset)
            x_start = max(0, x_offset)
            x_end = min(w, w + x_offset)

            if len(self._image_tensor.shape) == 2:
                return self._image_tensor[y_start:y_end, x_start:x_end]
            else:
                return self._image_tensor[:, y_start:y_end, x_start:x_end]
        else:
            # numpy array
            import numpy as np
            if len(self._image_tensor.shape) == 2:
                h, w = self._image_tensor.shape
            else:
                _, h, w = self._image_tensor.shape

            y_start = max(0, y_offset)
            y_end = min(h, h + y_offset)
            x_start = max(0, x_offset)
            x_end = min(w, w + x_offset)

            if len(self._image_tensor.shape) == 2:
                return self._image_tensor[y_start:y_end, x_start:x_end]
            else:
                return self._image_tensor[:, y_start:y_end, x_start:x_end]

    async def load_image(self, as_torch: bool = True) -> Any:
        """
        Load RAW image as Bayer/X-Trans array and cache in memory.

        Args:
            as_torch: If True, return torch.Tensor (zero-copy from numpy). If False, return numpy array.

        Returns:
            Cached tensor (torch.Tensor or numpy.ndarray)
        """
        if self._image_tensor is not None:
            if as_torch and not self._is_torch_tensor(self._image_tensor):
                import torch
                self._image_tensor = torch.from_numpy(self._image_tensor)
            return self._image_tensor

        local_path_trio = trio.Path(self.local_path)
        if not self.local_path or not await local_path_trio.exists():
            raise ValueError(f"Cannot load image: local_path not set or doesn't exist for {self.filename}")

        def load_sync():
            from rawnind.libs.rawproc import img_fpath_to_np_mono_flt_and_metadata
            data, img_metadata = img_fpath_to_np_mono_flt_and_metadata(str(self.local_path))
            self.metadata.update(img_metadata)
            return data

        np_data = await trio.to_thread.run_sync(load_sync)

        if as_torch:
            import torch
            self._image_tensor = torch.from_numpy(np_data)
        else:
            self._image_tensor = np_data

        return self._image_tensor

    def unload_image(self) -> None:
        """Free cached image tensor from memory."""
        self._image_tensor = None

    def cache_image_data(self, data: Any) -> None:
        """Manually set cached image data."""
        self._image_tensor = data

    @staticmethod
    def _is_torch_tensor(obj: Any) -> bool:
        """Check if object is a torch.Tensor without importing torch."""
        return type(obj).__module__.startswith('torch') and type(obj).__name__ == 'Tensor'


@dataclass
class SceneInfo:
    """Information about a scene (collection of clean and noisy images)."""

    scene_name: str
    cfa_type: str  # 'Bayer' or 'X-Trans'
    unknown_sensor: bool
    test_reserve: bool
    clean_images: List[ImageInfo] = field(default_factory=list)
    noisy_images: List[ImageInfo] = field(default_factory=list)

    def all_images(self) -> List[ImageInfo]:
        """Get all images (clean + noisy) for this scene."""
        return self.clean_images + self.noisy_images

    def get_gt_image(self) -> Optional[ImageInfo]:
        """Get the first clean (GT) image, or None if none exist."""
        return self.clean_images[0] if self.clean_images else None
