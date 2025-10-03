1. Ground Truth Validation: Test on synthetically shifted images where the true shift is known, verifying both methods detect the correct shift (not just that they agree with each other).

2. Alignment Improvement: Verify that applying the detected shift actually reduces error compared to unaligned images (MAE/MSE before alignment > MAE/MSE after alignment).

3. Shift Application Consistency: The same shift value, when applied using the method's own `shift_images()` function, should produce the claimed alignment quality.

4. CFA Pattern Preservation: For RAW data, verify that shifted images maintain valid Bayer/X-Trans patterns (especially for odd shifts that could break 2x2 Bayer alignment).

5. Edge Case Coverage:
   - Zero shift (already aligned)
   - Large shifts (near image boundaries)
   - Odd vs even shifts
   - Sub-pixel behavior (if applicable)

6. Cross-Domain Validation: If alignment is performed in RAW domain, verify the shift also produces good alignment in RGB domain (and vice versa).

7. Method Agreement: FFT-CFA and Bruteforce-RGB should produce identical or nearly identical shifts (current test does this).

8. Performance Bounds: FFT-CFA should be demonstrably faster than Bruteforce-RGB (quantify the speedup).

9. Failure Modes: Test on images that are difficult to align (low texture, repetitive patterns) and verify graceful degradation or appropriate error handling.
