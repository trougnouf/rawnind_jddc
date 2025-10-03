"""Tests for DatasetIndex trio/httpx asynchronous functionality."""

import json
import os
from pathlib import Path

import httpx
import pytest
import yaml

from rawnind.dataset.manager import (
    DATASET_API_URL,
    DATASET_YAML_URL,
    DatasetIndex,
    ImageInfo,
    compute_sha1,
)


@pytest.fixture
def mock_index(tmp_path):
    cache_path = tmp_path / "dataset_cache.yaml"
    index = DatasetIndex(cache_path=cache_path)
    yield index
    if cache_path.exists():
        cache_path.unlink()


@pytest.mark.trio
async def test_async_load_index_from_cache(mock_index):
    mock_data = {"Bayer": {"test_scene": {"clean_images": [], "noisy_images": []}}}
    mock_index.cache_path.write_text(yaml.dump(mock_data))

    await mock_index.async_load_index(force_update=False)

    assert mock_index._loaded is True
    assert "Bayer" in mock_index.scenes


@pytest.mark.trio
async def test_async_load_index_force_update(monkeypatch, mock_index):
    async def mock_async_update() -> None:
        mock_index._loaded = True

    monkeypatch.setattr(mock_index, "async_update_index", mock_async_update)

    await mock_index.async_load_index(force_update=True)

    assert mock_index._loaded is True


@pytest.mark.trio
async def test_async_update_index(monkeypatch, mock_index):
    mock_text = yaml.dump({"Bayer": {"test_scene": {"clean_images": [], "noisy_images": []}}})

    class MockResponse:
        def __init__(self, text: str):
            self._text = text

        def raise_for_status(self) -> None:
            pass

        @property
        def text(self) -> str:
            return self._text

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def get(self, url: str):
            assert url == DATASET_YAML_URL
            return MockResponse(mock_text)

    monkeypatch.setattr("rawnind.dataset.manager.httpx.AsyncClient", MockClient)

    await mock_index.async_update_index()

    assert mock_index._loaded is True
    assert "Bayer" in mock_index.scenes
    assert mock_index.cache_path.exists()


@pytest.mark.trio
async def test_async_discover_local_files(mock_index, tmp_path, monkeypatch):
    mock_data = {
        "Bayer": {
            "test_scene": {
                "clean_images": [{"filename": "test.tif", "sha1": "abc"}],
                "noisy_images": [],
            }
        }
    }
    mock_index._build_index_from_data(mock_data)
    mock_index._loaded = True

    temp_root = tmp_path / "src" / "rawnind" / "datasets" / "RawNIND" / "src"
    bayer_dir = temp_root / "Bayer"
    scene_dir = bayer_dir / "test_scene" / "gt"
    scene_dir.mkdir(parents=True, exist_ok=True)
    test_file = scene_dir / "test.tif"
    test_file.touch()

    monkeypatch.setattr("rawnind.dataset.manager.DATASET_ROOT", temp_root)

    found, total = await mock_index.async_discover_local_files()

    assert found == 1
    assert total == 1
    assert mock_index.scenes["Bayer"]["test_scene"].clean_images[0].local_path == test_file


def test_iter_missing_files_assigns_expected_path(mock_index, tmp_path, monkeypatch):
    mock_data = {
        "Bayer": {
            "test_scene": {
                "clean_images": [{"filename": "missing.tif", "sha1": "abc"}],
                "noisy_images": [],
            }
        }
    }
    mock_index._build_index_from_data(mock_data)
    mock_index._loaded = True

    temp_root = tmp_path / "src" / "rawnind" / "datasets" / "RawNIND" / "src"
    monkeypatch.setattr("rawnind.dataset.manager.DATASET_ROOT", temp_root)

    missing = list(mock_index.iter_missing_files())

    assert len(missing) == 1
    expected_path = temp_root / "Bayer" / "test_scene" / "gt" / "missing.tif"
    assert missing[0].local_path == expected_path
    assert mock_index.missing_file_count == 1
    assert mock_index.dataset_complete is False


@pytest.mark.trio
async def test_async_get_dataset_files(monkeypatch, mock_index):
    mock_json = {
        "status": "OK",
        "data": [
            {
                "dataFile": {
                    "filename": "test.tif",
                    "id": 123,
                }
            }
        ],
    }

    class MockResponse:
        def __init__(self, payload: dict):
            self._payload = payload

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return self._payload

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def get(self, url: str):
            assert url == DATASET_API_URL
            return MockResponse(mock_json)

    monkeypatch.setattr("rawnind.dataset.manager.httpx.AsyncClient", MockClient)

    files_dict = await mock_index.async_get_dataset_files()

    assert files_dict == {"test.tif": 123}


@pytest.mark.trio
async def test_ensure_remote_file_index_handles_modern_schema(monkeypatch, mock_index):
    modern_payload = {
        "status": "OK",
        "totalCount": 2,
        "data": [
            {
                "label": "foo.arw",
                "dataFile": {
                    "id": 111,
                    "filename": "foo.arw",
                },
            },
            {
                "label": "bar.arw",
                "dataFile": {
                    "id": 222,
                    "filename": "bar.arw",
                },
            },
        ],
    }

    class MockResponse:
        def __init__(self, payload: dict):
            self._payload = payload

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return self._payload

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def get(self, url: str):
            assert url == DATASET_API_URL
            return MockResponse(modern_payload)

    monkeypatch.setattr("rawnind.dataset.manager.httpx.AsyncClient", MockClient)

    files_dict = await mock_index.async_get_dataset_files()

    assert files_dict == {"foo.arw": 111, "bar.arw": 222}


@pytest.mark.trio
async def test_async_get_dataset_files_single_scene(monkeypatch, tmp_path):
    """Smoke test using a reduced remote index populated via monkeypatch."""
    index = DatasetIndex(cache_path=tmp_path / "single_scene_cache.yaml")

    single_scene_data = {
        "Bayer": {
            "2pilesofplates": {
                "clean_images": [
                    {
                        "filename": "Bayer_2pilesofplates_GT_ISO100_sha1=854554a34b339413462eb1538d4cb0fa95d468b5.arw",
                        "sha1": "854554a34b339413462eb1538d4cb0fa95d468b5",
                    },
                    {
                        "filename": "Bayer_2pilesofplates_GT_ISO100_sha1=8593a92d470192bac4434f3abeb2f0bcd8cf03ad.arw",
                        "sha1": "8593a92d470192bac4434f3abeb2f0bcd8cf03ad",
                    },
                ],
                "noisy_images": [
                    {
                        "filename": "Bayer_2pilesofplates_ISO65535_sha1=b5c1ca049cd080144d2853ef5a78b0247a50cf3f.arw",
                        "sha1": "b5c1ca049cd080144d2853ef5a78b0247a50cf3f",
                    },
                    {
                        "filename": "Bayer_2pilesofplates_ISO12800_sha1=7c5f212c12b86bd8c4ecde91ba37de415b3744de.arw",
                        "sha1": "7c5f212c12b86bd8c4ecde91ba37de415b3744de",
                    },
                    {
                        "filename": "Bayer_2pilesofplates_ISO2000_sha1=31fac4e2aeaa235193744d7e49c503734b6346b2.arw",
                        "sha1": "31fac4e2aeaa235193744d7e49c503734b6346b2",
                    },
                    {
                        "filename": "Bayer_2pilesofplates_ISO320_sha1=080e42669f295009fcd3fb9dda0ac72b1148f93a.arw",
                        "sha1": "080e42669f295009fcd3fb9dda0ac72b1148f93a",
                    },
                    {
                        "filename": "Bayer_2pilesofplates_ISO800_sha1=1353e69380a30ec67df3cd21be3cf8ce3c6db222.arw",
                        "sha1": "1353e69380a30ec67df3cd21be3cf8ce3c6db222",
                    },
                    {
                        "filename": "Bayer_2pilesofplates_ISO16000_sha1=3b38449e04f75c3b68443e888c9c02fa99eddde6.arw",
                        "sha1": "3b38449e04f75c3b68443e888c9c02fa99eddde6",
                    },
                    {
                        "filename": "Bayer_2pilesofplates_ISO125_sha1=58c56d2ccf6cfcc49292f8b09ccba1609111380a.arw",
                        "sha1": "58c56d2ccf6cfcc49292f8b09ccba1609111380a",
                    },
                    {
                        "filename": "Bayer_2pilesofplates_ISO640_sha1=e11074913f99cdc64f7f41368a5bc99421a4ad23.arw",
                        "sha1": "e11074913f99cdc64f7f41368a5bc99421a4ad23",
                    },
                    {
                        "filename": "Bayer_2pilesofplates_ISO160_sha1=17018efcc6228c02fd3a024abc70ef08c64a7622.arw",
                        "sha1": "17018efcc6228c02fd3a024abc70ef08c64a7622",
                    },
                    {
                        "filename": "Bayer_2pilesofplates_ISO1250_sha1=9121efbd50e2f2392665cb17b435b2526df8e9ae.arw",
                        "sha1": "9121efbd50e2f2392665cb17b435b2526df8e9ae",
                    },
                    {
                        "filename": "Bayer_2pilesofplates_ISO400_sha1=6637f1f7485046ba8e6e09978bd26b91042568e6.arw",
                        "sha1": "6637f1f7485046ba8e6e09978bd26b91042568e6",
                    },
                    {
                        "filename": "Bayer_2pilesofplates_ISO2500_sha1=c08ac26ca1d57b6c258f4c4706ad312de0936ec9.arw",
                        "sha1": "c08ac26ca1d57b6c258f4c4706ad312de0936ec9",
                    },
                ],
            }
        },
        "X-Trans": {},
    }

    index._build_index_from_data(single_scene_data)
    index._loaded = True

    target_filenames = {
        img["filename"]
        for img in single_scene_data["Bayer"]["2pilesofplates"]["clean_images"]
    }
    target_filenames.update(
        img["filename"]
        for img in single_scene_data["Bayer"]["2pilesofplates"]["noisy_images"]
    )

    original_ensure = DatasetIndex._ensure_remote_file_index

    async def reduced_remote_index(self, client, *, force_refresh=False):
        mapping = await original_ensure(self, client, force_refresh=force_refresh)
        kept = {name: file_id for name, file_id in mapping.items() if name in target_filenames}
        self._remote_file_index = kept
        return kept

    monkeypatch.setattr(DatasetIndex, "_ensure_remote_file_index", reduced_remote_index)

    files = await index.async_get_dataset_files()

    assert set(files.keys()) == target_filenames
    assert all(isinstance(value, int) and value > 0 for value in files.values())
    assert len(files) == len(target_filenames)


@pytest.mark.trio
async def test_async_download_missing_files_smoke(tmp_path, monkeypatch):
    root_dir = tmp_path / "src" / "rawnind" / "datasets" / "RawNIND" / "src"
    bayer_dir = root_dir / "Bayer"
    scene_dir = bayer_dir / "smoke_scene"
    noise_dir = scene_dir
    noise_dir.mkdir(parents=True, exist_ok=True)

    mock_data = {
        "Bayer": {
            "smoke_scene": {
                "clean_images": [
                    {
                        "filename": "smoke_clean.arw",
                        "sha1": "dummy",
                    }
                ],
                "noisy_images": [
                    {
                        "filename": "smoke_noisy.arw",
                        "sha1": "dummy",
                    }
                ],
            }
        }
    }

    def fake_sha1(path: Path) -> str:
        return "dummy"

    class FakeResponse:
        def __init__(self, data: bytes):
            self._data = data

        def raise_for_status(self) -> None:
            pass

        async def aread(self) -> bytes:
            return self._data

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def get(self, url: str, *_, **__):
            filename = url.split("/")[-1]
            return FakeResponse(filename.encode("utf-8"))

    index = DatasetIndex(cache_path=tmp_path / "smoke_cache.yaml")
    index._build_index_from_data(mock_data)
    index._loaded = True

    monkeypatch.setattr("rawnind.dataset.manager.DATASET_ROOT", root_dir)
    monkeypatch.setattr("rawnind.dataset.manager.httpx.AsyncClient", FakeClient)
    monkeypatch.setattr("rawnind.dataset.manager.compute_sha1", fake_sha1)

    fake_mapping = {
        "smoke_clean.arw": 1,
        "smoke_noisy.arw": 2,
    }

    async def fake_remote_index(*_args, **_kwargs):
        return fake_mapping

    monkeypatch.setattr(index, "_ensure_remote_file_index", fake_remote_index)

    await index.async_download_missing_files(max_concurrent=1, limit=1)

    clean_path = scene_dir / "smoke_clean.arw"
    assert clean_path.exists()
    assert clean_path.read_bytes() == b"1"


@pytest.mark.trio
async def test_download_and_validate_success(mock_index, tmp_path):
    content = b"test content"
    correct_sha1 = "1eebdf4fdc9fc7bf283031b93f9aef3338de9052"
    img_info = ImageInfo("test.tif", correct_sha1, is_clean=False)
    img_info.local_path = tmp_path / "test.tif"

    class DummyResponse:
        def __init__(self, body: bytes):
            self._content = body

        def raise_for_status(self) -> None:
            pass

        async def aread(self) -> bytes:
            return self._content

    class DummyClient:
        async def get(self, url: str, timeout=None):
            return DummyResponse(content)

    await mock_index._download_and_validate(
        DummyClient(),
        img_info,
        999,
        timeout=httpx.Timeout(30.0),
    )

    assert img_info.local_path.exists()
    assert compute_sha1(img_info.local_path) == correct_sha1
    assert img_info.validated is True


@pytest.mark.trio
async def test_async_download_missing_files(monkeypatch, mock_index, tmp_path):
    mock_data = {
        "Bayer": {
            "test_scene": {
                "clean_images": [{"filename": "missing.tif", "sha1": "abc"}],
                "noisy_images": [],
            }
        }
    }
    mock_index._build_index_from_data(mock_data)
    mock_index._loaded = True

    temp_root = tmp_path / "src" / "rawnind" / "datasets" / "RawNIND" / "src"
    monkeypatch.setattr("rawnind.dataset.manager.DATASET_ROOT", temp_root)

    async def fake_discover():
        return (0, 1)

    monkeypatch.setattr(mock_index, "async_discover_local_files", fake_discover)

    async def fake_remote_index(*_args, **_kwargs):
        return {"missing.tif": 456}

    monkeypatch.setattr(mock_index, "_ensure_remote_file_index", fake_remote_index)

    downloads = []

    async def fake_download(client, image_info, file_id, *, timeout):
        downloads.append((image_info.filename, file_id))

    monkeypatch.setattr(mock_index, "_download_and_validate", fake_download)

    class DummyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr("rawnind.dataset.manager.httpx.AsyncClient", lambda **kwargs: DummyClient())

    await mock_index.async_download_missing_files(max_concurrent=1)

    assert downloads == [("missing.tif", 456)]


@pytest.mark.trio
async def test_async_download_missing_files_no_missing(monkeypatch, mock_index, tmp_path):
    mock_data = {
        "Bayer": {
            "test_scene": {
                "clean_images": [{"filename": "local.tif", "sha1": "abc"}],
                "noisy_images": [],
            }
        }
    }
    mock_index._build_index_from_data(mock_data)
    mock_index._loaded = True

    temp_root = tmp_path / "src" / "rawnind" / "datasets" / "RawNIND" / "src"
    gt_dir = temp_root / "Bayer" / "test_scene" / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / "local.tif").write_bytes(b"dummy")

    monkeypatch.setattr("rawnind.dataset.manager.DATASET_ROOT", temp_root)

    await mock_index.async_discover_local_files()

    async def fail_remote_index(*_args, **_kwargs):
        raise AssertionError("should not fetch remote index when dataset is complete")

    monkeypatch.setattr(mock_index, "_ensure_remote_file_index", fail_remote_index)

    await mock_index.async_download_missing_files()

    assert mock_index.dataset_complete is True


def test_compute_sha1(tmp_path):
    test_file = tmp_path / "test.txt"
    test_content = b"test content"

    test_file.write_bytes(test_content)

    computed_hash = compute_sha1(test_file)
    expected_hash = "1eebdf4fdc9fc7bf283031b93f9aef3338de9052"

    assert computed_hash == expected_hash