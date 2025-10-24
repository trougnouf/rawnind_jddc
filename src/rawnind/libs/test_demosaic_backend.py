"""RED test: Verify demosaic() uses OIIO backend instead of cv2.

TDD WORKFLOW:
1. RED: This test will fail because demosaic() currently uses cv2
2. GREEN: Implement OIIO-based demosaic to make it pass
3. REFACTOR: Clean up if needed
"""

import pytest
import numpy as np
from unittest.mock import patch


def test_demosaic_uses_oiio_not_cv2():
    """demosaic() must use OpenImageIO backend, not cv2.

    RATIONALE: cv2 demosaic has limitations:
    - Requires uint16 conversion (precision loss)
    - Limited algorithm options
    - Not designed for HDR workflows

    OIIO provides:
    - Native float32 support (no conversions)
    - Better HDR handling
    - Consistent with rest of pipeline (EXR I/O uses OIIO)
    """
    from rawnind.libs import raw

    # Create test bayer image (RGGB pattern)
    h, w = 256, 256
    bayer_data = np.random.rand(1, h, w).astype(np.float32) * 0.5
    metadata = {"bayer_pattern": "RGGB"}

    # Mock cv2.demosaicing to track if it's called
    with patch('cv2.demosaicing', side_effect=Exception("cv2.demosaicing should not be called!")) as mock_cv2:
        # Call demosaic - should use OIIO, not cv2
        result = raw.demosaic(bayer_data, metadata)

        # Verify cv2 was NOT called
        assert not mock_cv2.called, "demosaic() must use OIIO backend, not cv2!"

    # Verify output shape and type
    assert result.shape == (3, h, w), f"Wrong output shape: {result.shape}"
    assert result.dtype == np.float32, f"Wrong dtype: {result.dtype}"
    assert np.isfinite(result).all(), "Output contains NaN/Inf"


def test_demosaic_oiio_preserves_hdr_range():
    """OIIO demosaic should preserve HDR values (>1.0 and <0.0).

    DESIGN: Linear sensor data can exceed [0,1] for bright highlights.
    cv2 requires uint16 conversion which adds complexity and risk.
    OIIO handles float32 natively.
    """
    from rawnind.libs import raw

    # Create HDR test data with values outside [0,1]
    h, w = 256, 256
    bayer_data = np.random.rand(1, h, w).astype(np.float32) * 2.0 - 0.5  # Range: [-0.5, 1.5]
    metadata = {"bayer_pattern": "RGGB"}

    # Mock cv2 to ensure OIIO is used
    with patch('cv2.demosaicing', side_effect=Exception("Must use OIIO!")):
        result = raw.demosaic(bayer_data, metadata)

    # Verify HDR range preserved (values outside [0,1] should exist)
    assert result.min() < 0, "HDR negative values should be preserved"
    assert result.max() > 1.0, "HDR highlights (>1.0) should be preserved"
    assert np.isfinite(result).all(), "No NaN/Inf allowed"


def test_demosaic_oiio_backend_specified():
    """demosaic() should accept backend='oiio' parameter.

    DESIGN: Explicit backend selection for transparency.
    Default should be 'oiio' but allow override for testing.
    """
    from rawnind.libs import raw

    h, w = 128, 128
    bayer_data = np.random.rand(1, h, w).astype(np.float32) * 0.5
    metadata = {"bayer_pattern": "RGGB"}

    # Should accept backend parameter
    with patch('cv2.demosaicing', side_effect=Exception("Must use OIIO!")):
        result = raw.demosaic(bayer_data, metadata, backend='oiio')

    assert result.shape == (3, h, w)
    assert result.dtype == np.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])