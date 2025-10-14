import numpy as np
import pytest

try:
    import torch
except ImportError:  # torch is in deps, but guard anyway
    torch = None

from rawnind.libs.rawproc import make_overexposure_mask_bayer


@pytest.mark.parametrize("shape,threshold,expected_true_frac", [
    # 2D Bayer-like mosaic
    ((6, 8), 0.5, 0.5),
    # 2D X-Trans-like mosaic (multiple of 6)
    ((12, 12), 0.5, 0.5),
])
def test_overexposure_mask_bayer_2d(shape, threshold, expected_true_frac):
    rng = np.random.default_rng(42)
    img = rng.random(shape, dtype=np.float32)
    mask = make_overexposure_mask_bayer(img, threshold)

    # Assertions
    assert mask.shape == shape
    assert mask.dtype == np.bool_
    # Roughly half should be below threshold (random uniform)
    frac_true = mask.mean()
    assert 0.35 <= frac_true <= 0.65


@pytest.mark.parametrize("channels,down_factor", [
    (4, 2),  # Bayer RGGB stacked channels
    (9, 3),  # X-Trans 3x3 color grouping (e.g., 9 planes) stacked channels
])
def test_overexposure_mask_bayer_3d(channels, down_factor):
    h, w = 18, 24
    ch, hh, ww = channels, h // down_factor, w // down_factor
    rng = np.random.default_rng(0)
    img = rng.random((ch, hh, ww), dtype=np.float32)

    threshold = 0.6
    mask = make_overexposure_mask_bayer(img, threshold)

    # Assertions
    assert mask.shape == (hh, ww)
    assert mask.dtype == np.bool_

    # All-channel criterion: create a pixel where one channel exceeds threshold
    img2 = np.full((ch, hh, ww), 0.4, dtype=np.float32)
    img2[0, 0, 0] = 0.99  # one channel too bright
    mask2 = make_overexposure_mask_bayer(img2, 0.6)
    # any channel >= threshold disables that pixel
    assert not bool(mask2[0, 0])
    assert bool(mask2[1, 1])


@pytest.mark.skipif(torch is None, reason="torch not available")
def test_overexposure_mask_bayer_accepts_torch_tensor():
    h, w = 10, 14
    img = torch.rand((h, w), dtype=torch.float32)
    threshold = 0.5
    mask = make_overexposure_mask_bayer(img, threshold)

    # Return type should be numpy bool array to integrate with downstream numpy ops
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.bool_
    assert mask.shape == (h, w)

    # 3D tensor variant
    img3 = torch.full((4, h // 2, w // 2), 0.4)
    img3[0, 0, 0] = 0.99
    mask3 = make_overexposure_mask_bayer(img3, 0.6)
    assert mask3.shape == (h // 2, w // 2)
    assert not bool(mask3[0, 0])
    assert bool(mask3[1, 1])
