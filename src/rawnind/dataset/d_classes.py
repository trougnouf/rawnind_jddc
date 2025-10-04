from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


@dataclass
class ImageInfo:
    """Information about a single image file."""

    filename: str
    sha1: str
    is_clean: bool
    local_path: Optional[Path] = None
    validated: bool = False
    file_id: str = ""  # Dataverse file ID for downloads

    @property
    def download_url(self) -> str:
        """Construct download URL for this file.

        Returns:
            Download URL for the file
        """
        if not self.file_id:
            raise ValueError(
                f"Cannot construct download URL: file_id not set for {self.filename}"
            )
        return f"https://dataverse.uclouvain.be/api/access/datafile/{self.file_id}"


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
