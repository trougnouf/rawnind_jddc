"""Unit tests for dataset manager module."""

import sys
import pytest
import yaml
from pathlib import Path

sys.path.insert(0, "src")
from rawnind.dataset import DatasetIndex, ImageInfo, SceneInfo, compute_sha1


@pytest.fixture
def mock_dataset_yaml():
    """Create a mock dataset.yaml structure."""
    return {
        "Bayer": {
            "TestScene1": {
                "unknown_sensor": False,
                "test_reserve": False,
                "clean_images": [
                    {
                        "filename": "Bayer_TestScene1_GT_ISO100_sha1=abc123.cr2",
                        "sha1": "abc123def456",
                    }
                ],
                "noisy_images": [
                    {
                        "filename": "Bayer_TestScene1_ISO800_sha1=def456.cr2",
                        "sha1": "def456ghi789",
                    },
                    {
                        "filename": "Bayer_TestScene1_ISO1600_sha1=ghi789.cr2",
                        "sha1": "ghi789jkl012",
                    },
                ],
            },
            "TestScene2": {
                "unknown_sensor": False,
                "test_reserve": True,
                "clean_images": [
                    {
                        "filename": "Bayer_TestScene2_GT_ISO100_sha1=jkl012.cr2",
                        "sha1": "jkl012mno345",
                    }
                ],
                "noisy_images": [],
            },
        },
        "X-Trans": {
            "TestSceneXT1": {
                "unknown_sensor": False,
                "test_reserve": False,
                "clean_images": [
                    {
                        "filename": "XTrans_TestSceneXT1_GT_ISO200_sha1=mno345.raf",
                        "sha1": "mno345pqr678",
                    }
                ],
                "noisy_images": [
                    {
                        "filename": "XTrans_TestSceneXT1_ISO3200_sha1=pqr678.raf",
                        "sha1": "pqr678stu901",
                    }
                ],
            }
        },
    }


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def dataset_index_with_mock(mock_dataset_yaml, temp_cache_dir):
    """Create a DatasetIndex with mock data."""
    cache_path = temp_cache_dir / "dataset_index.yaml"

    # Write mock data to cache
    with open(cache_path, "w") as f:
        yaml.dump(mock_dataset_yaml, f)

    # Create index
    index = DatasetIndex(cache_path=cache_path)
    index.load_index()

    return index


class TestImageInfo:
    """Tests for ImageInfo dataclass."""

    def test_creation(self):
        """Test ImageInfo creation."""
        img = ImageInfo(filename="test.cr2", sha1="abc123", is_clean=True)
        assert img.filename == "test.cr2"
        assert img.sha1 == "abc123"
        assert img.is_clean is True
        assert img.local_path is None
        assert img.validated is False

    def test_with_local_path(self):
        """Test ImageInfo with local path."""
        img = ImageInfo(
            filename="test.cr2",
            sha1="abc123",
            is_clean=False,
            local_path=Path("/path/to/test.cr2"),
        )
        assert img.local_path == Path("/path/to/test.cr2")


class TestSceneInfo:
    """Tests for SceneInfo dataclass."""

    def test_creation(self):
        """Test SceneInfo creation."""
        scene = SceneInfo(
            scene_name="TestScene",
            cfa_type="Bayer",
            unknown_sensor=False,
            test_reserve=False,
        )
        assert scene.scene_name == "TestScene"
        assert scene.cfa_type == "Bayer"
        assert len(scene.clean_images) == 0
        assert len(scene.noisy_images) == 0

    def test_all_images(self):
        """Test all_images() method."""
        clean_img = ImageInfo("clean.cr2", "abc", True)
        noisy_img1 = ImageInfo("noisy1.cr2", "def", False)
        noisy_img2 = ImageInfo("noisy2.cr2", "ghi", False)

        scene = SceneInfo(
            scene_name="Test",
            cfa_type="Bayer",
            unknown_sensor=False,
            test_reserve=False,
            clean_images=[clean_img],
            noisy_images=[noisy_img1, noisy_img2],
        )

        all_imgs = scene.all_images()
        assert len(all_imgs) == 3
        assert clean_img in all_imgs
        assert noisy_img1 in all_imgs
        assert noisy_img2 in all_imgs

    def test_get_gt_image(self):
        """Test get_gt_image() method."""
        clean_img = ImageInfo("clean.cr2", "abc", True)
        scene = SceneInfo(
            scene_name="Test",
            cfa_type="Bayer",
            unknown_sensor=False,
            test_reserve=False,
            clean_images=[clean_img],
        )

        gt = scene.get_gt_image()
        assert gt is clean_img

    def test_get_gt_image_none(self):
        """Test get_gt_image() with no clean images."""
        scene = SceneInfo(
            scene_name="Test",
            cfa_type="Bayer",
            unknown_sensor=False,
            test_reserve=False,
        )

        gt = scene.get_gt_image()
        assert gt is None


class TestDatasetIndex:
    """Tests for DatasetIndex class."""

    def test_initialization(self, temp_cache_dir):
        """Test DatasetIndex initialization."""
        cache_path = temp_cache_dir / "test_index.yaml"
        index = DatasetIndex(cache_path=cache_path)

        assert index.cache_path == cache_path
        assert index._loaded is False
        assert len(index.scenes) == 0

    def test_load_from_yaml(self, dataset_index_with_mock):
        """Test loading index from YAML."""
        index = dataset_index_with_mock

        assert index._loaded is True
        assert "Bayer" in index.scenes
        assert "X-Trans" in index.scenes
        assert "TestScene1" in index.scenes["Bayer"]
        assert "TestScene2" in index.scenes["Bayer"]
        assert "TestSceneXT1" in index.scenes["X-Trans"]

    def test_get_all_scenes(self, dataset_index_with_mock):
        """Test get_all_scenes()."""
        index = dataset_index_with_mock
        scenes = index.get_all_scenes()

        assert len(scenes) == 3
        scene_names = [s.scene_name for s in scenes]
        assert "TestScene1" in scene_names
        assert "TestScene2" in scene_names
        assert "TestSceneXT1" in scene_names

    def test_get_scenes_by_cfa_bayer(self, dataset_index_with_mock):
        """Test get_scenes_by_cfa() for Bayer."""
        index = dataset_index_with_mock
        bayer_scenes = index.get_scenes_by_cfa("Bayer")

        assert len(bayer_scenes) == 2
        scene_names = [s.scene_name for s in bayer_scenes]
        assert "TestScene1" in scene_names
        assert "TestScene2" in scene_names

    def test_get_scenes_by_cfa_xtrans(self, dataset_index_with_mock):
        """Test get_scenes_by_cfa() for X-Trans."""
        index = dataset_index_with_mock
        xtrans_scenes = index.get_scenes_by_cfa("X-Trans")

        assert len(xtrans_scenes) == 1
        assert xtrans_scenes[0].scene_name == "TestSceneXT1"

    def test_get_scene(self, dataset_index_with_mock):
        """Test get_scene()."""
        index = dataset_index_with_mock
        scene = index.get_scene("Bayer", "TestScene1")

        assert scene is not None
        assert scene.scene_name == "TestScene1"
        assert scene.cfa_type == "Bayer"
        assert len(scene.clean_images) == 1
        assert len(scene.noisy_images) == 2

    def test_get_scene_not_found(self, dataset_index_with_mock):
        """Test get_scene() with non-existent scene."""
        index = dataset_index_with_mock
        scene = index.get_scene("Bayer", "NonExistent")

        assert scene is None

    def test_get_missing_files(self, dataset_index_with_mock):
        """Test get_missing_files() - all files should be missing."""
        index = dataset_index_with_mock
        missing = index.get_missing_files()

        # All files should be missing since we haven't discovered any
        # Bayer: TestScene1 (1 clean + 2 noisy), TestScene2 (1 clean + 0 noisy)
        # X-Trans: TestSceneXT1 (1 clean + 1 noisy)
        # Total: 3 clean + 3 noisy = 6
        assert len(missing) == 6

    def test_get_available_scenes_none(self, dataset_index_with_mock):
        """Test get_available_scenes() with no local files."""
        index = dataset_index_with_mock
        available = index.get_available_scenes()

        # No files discovered yet, so no scenes available
        assert len(available) == 0

    def test_scene_image_counts(self, dataset_index_with_mock):
        """Test that scene image counts are correct."""
        index = dataset_index_with_mock

        scene1 = index.get_scene("Bayer", "TestScene1")
        assert len(scene1.clean_images) == 1
        assert len(scene1.noisy_images) == 2
        assert (
            scene1.clean_images[0].filename
            == "Bayer_TestScene1_GT_ISO100_sha1=abc123.cr2"
        )
        assert scene1.clean_images[0].sha1 == "abc123def456"

        scene2 = index.get_scene("Bayer", "TestScene2")
        assert len(scene2.clean_images) == 1
        assert len(scene2.noisy_images) == 0
        assert scene2.test_reserve is True

        scenext = index.get_scene("X-Trans", "TestSceneXT1")
        assert len(scenext.clean_images) == 1
        assert len(scenext.noisy_images) == 1


class TestComputeSha1:
    """Tests for compute_sha1 function."""

    def test_compute_sha1(self, tmp_path):
        """Test SHA1 computation."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)

        # Compute SHA1
        computed = compute_sha1(test_file)

        # Expected SHA1 for "Hello, World!"
        import hashlib

        expected = hashlib.sha1(test_content).hexdigest()

        assert computed == expected

    def test_compute_sha1_empty_file(self, tmp_path):
        """Test SHA1 computation for empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_bytes(b"")

        computed = compute_sha1(test_file)

        # SHA1 of empty string
        import hashlib

        expected = hashlib.sha1(b"").hexdigest()

        assert computed == expected


class TestValidation:
    """Tests for file validation."""

    def test_validate_file_success(self, tmp_path):
        """Test successful file validation."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_content = b"Test content"
        test_file.write_bytes(test_content)

        # Compute actual SHA1
        import hashlib

        actual_sha1 = hashlib.sha1(test_content).hexdigest()

        # Create ImageInfo
        img_info = ImageInfo(
            filename="test.txt", sha1=actual_sha1, is_clean=True, local_path=test_file
        )

        # Create index and validate
        index = DatasetIndex()
        result = index.validate_file(img_info)

        assert result is True
        assert img_info.validated is True

    def test_validate_file_mismatch(self, tmp_path):
        """Test file validation with SHA1 mismatch."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"Test content")

        # Create ImageInfo with wrong SHA1
        img_info = ImageInfo(
            filename="test.txt",
            sha1="wrongsha1hash",
            is_clean=True,
            local_path=test_file,
        )

        # Create index and validate
        index = DatasetIndex()
        result = index.validate_file(img_info)

        assert result is False
        assert img_info.validated is False

    def test_validate_file_not_found(self):
        """Test file validation when file doesn't exist."""
        img_info = ImageInfo(
            filename="nonexistent.txt",
            sha1="abc123",
            is_clean=True,
            local_path=Path("/nonexistent/path/file.txt"),
        )

        index = DatasetIndex()
        result = index.validate_file(img_info)

        assert result is False
        assert img_info.validated is False

    def test_validate_file_no_path(self):
        """Test file validation when local_path is None."""
        img_info = ImageInfo(filename="test.txt", sha1="abc123", is_clean=True)

        index = DatasetIndex()
        result = index.validate_file(img_info)

        assert result is False
        assert img_info.validated is False


class TestSceneGenerators:
    """Tests for scene generator methods."""

    def test_sequential_scene_generator_order(self, dataset_index_with_mock):
        """Test that sequential generator yields scenes in sorted order."""
        index = dataset_index_with_mock
        scenes = list(index.sequential_scene_generator())

        assert len(scenes) == 3
        scene_identifiers = [(s.cfa_type, s.scene_name) for s in scenes]

        # Should be sorted by CFA type, then scene name
        assert scene_identifiers[0] == ("Bayer", "TestScene1")
        assert scene_identifiers[1] == ("Bayer", "TestScene2")
        assert scene_identifiers[2] == ("X-Trans", "TestSceneXT1")

    def test_random_scene_generator_count(self, dataset_index_with_mock):
        """Test random generator with count parameter."""
        index = dataset_index_with_mock

        # Request fewer scenes than available
        scenes = list(index.random_scene_generator(count=2, seed=42))
        assert len(scenes) == 2

        # Request all scenes
        scenes = list(index.random_scene_generator(count=3, seed=42))
        assert len(scenes) == 3

    def test_random_scene_generator_reproducibility(self, dataset_index_with_mock):
        """Test that random generator is reproducible with seed."""
        index = dataset_index_with_mock

        # Generate with same seed twice
        scenes1 = list(index.random_scene_generator(seed=123))
        scenes2 = list(index.random_scene_generator(seed=123))

        # Should be identical
        assert len(scenes1) == len(scenes2)
        for s1, s2 in zip(scenes1, scenes2):
            assert s1.scene_name == s2.scene_name
            assert s1.cfa_type == s2.cfa_type


class TestCaching:
    """Tests for caching functionality."""

    def test_sorted_cfa_types_cached(self, dataset_index_with_mock):
        """Test that sorted_cfa_types property is cached."""
        index = dataset_index_with_mock

        # First access
        types1 = index.sorted_cfa_types
        assert types1 == ["Bayer", "X-Trans"]

        # Second access - should return cached value
        types2 = index.sorted_cfa_types
        assert types2 is types1

    def test_cache_invalidation_on_build_index(self, dataset_index_with_mock):
        """Test that caches are invalidated when index is rebuilt."""
        index = dataset_index_with_mock

        # Access cached properties
        _ = index.sorted_cfa_types

        # Cache should be populated
        assert index._sorted_cfa_types is not None

        # Rebuild index with new data
        new_data = {
            "Bayer": {
                "NewScene": {
                    "unknown_sensor": False,
                    "test_reserve": False,
                    "clean_images": [],
                    "noisy_images": [],
                }
            }
        }
        index._build_index_from_data(new_data)

        # Cache should be cleared
        assert index._sorted_cfa_types is None


class TestSceneGenerators:
    """Tests for scene generator methods."""

    def test_sequential_scene_generator_order(self, dataset_index_with_mock):
        """Test that sequential generator yields scenes in sorted order."""
        index = dataset_index_with_mock
        scenes = list(index.sequential_scene_generator())

        assert len(scenes) == 3
        scene_identifiers = [(s.cfa_type, s.scene_name) for s in scenes]

        # Should be sorted by CFA type, then scene name
        assert scene_identifiers[0] == ("Bayer", "TestScene1")
        assert scene_identifiers[1] == ("Bayer", "TestScene2")
        assert scene_identifiers[2] == ("X-Trans", "TestSceneXT1")

    def test_random_scene_generator_count(self, dataset_index_with_mock):
        """Test random generator with count parameter."""
        index = dataset_index_with_mock

        # Request fewer scenes than available
        scenes = list(index.random_scene_generator(count=2, seed=42))
        assert len(scenes) == 2

        # Request all scenes
        scenes = list(index.random_scene_generator(count=3, seed=42))
        assert len(scenes) == 3

    def test_random_scene_generator_reproducibility(self, dataset_index_with_mock):
        """Test that random generator is reproducible with seed."""
        index = dataset_index_with_mock

        # Generate with same seed twice
        scenes1 = list(index.random_scene_generator(seed=123))
        scenes2 = list(index.random_scene_generator(seed=123))

        # Should be identical
        assert len(scenes1) == len(scenes2)
        for s1, s2 in zip(scenes1, scenes2):
            assert s1.scene_name == s2.scene_name
            assert s1.cfa_type == s2.cfa_type


class TestCaching:
    """Tests for caching functionality."""

    def test_sorted_cfa_types_cached(self, dataset_index_with_mock):
        """Test that sorted_cfa_types property is cached."""
        index = dataset_index_with_mock

        # First access
        types1 = index.sorted_cfa_types
        assert types1 == ["Bayer", "X-Trans"]

        # Second access - should return cached value
        types2 = index.sorted_cfa_types
        assert types2 is types1

    def test_cache_invalidation_on_build_index(self, dataset_index_with_mock):
        """Test that caches are invalidated when index is rebuilt."""
        index = dataset_index_with_mock

        # Access cached properties
        _ = index.sorted_cfa_types

        # Cache should be populated
        assert index._sorted_cfa_types is not None

        # Rebuild index with new data
        new_data = {
            "Bayer": {
                "NewScene": {
                    "unknown_sensor": False,
                    "test_reserve": False,
                    "clean_images": [],
                    "noisy_images": [],
                }
            }
        }
        index._build_index_from_data(new_data)

        # Cache should be cleared
        assert index._sorted_cfa_types is None
