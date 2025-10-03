"""Create a curated test dataset with known misaligned pairs and diverse samples."""

import json
from pathlib import Path

print("=" * 80)
print("CREATING CURATED TEST DATASET")
print("=" * 80)
print()

# Known misaligned pairs from previous testing
known_misaligned = [
    {
        "scene": "Bark",
        "cfa": "Bayer",
        "camera": "Canon (assumed)",
        "gt_file": "Bayer/Bark/gt/Bayer_Bark_GT_ISO100_sha1=f15da1140d949ee30c15ce7b251839a7b7a41de7.cr2",
        "noisy_file": "Bayer/Bark/Bayer_Bark_ISO65535_sha1=6ba8ed5f7fff42c4c900812c02701649f4f2d49e.cr2",
        "expected_shift": [12, -10],
        "shift_magnitude": "large",
        "notes": "Largest shift found, FFT has 2px error on this",
    },
    {
        "scene": "Bark",
        "cfa": "Bayer",
        "camera": "Canon (assumed)",
        "gt_file": "Bayer/Bark/gt/Bayer_Bark_GT_ISO100_sha1=f15da1140d949ee30c15ce7b251839a7b7a41de7.cr2",
        "noisy_file": "Bayer/Bark/Bayer_Bark_ISO800_sha1=ba86f1da64a4bb534d9216e96c1c72177ed1e625.cr2",
        "expected_shift": [4, -4],
        "shift_magnitude": "small",
        "notes": "FFT works perfectly on this",
    },
    {
        "scene": "Kortlek",
        "cfa": "Bayer",
        "camera": "Canon (assumed)",
        "gt_file": "Bayer/Kortlek/gt/Bayer_Kortlek_GT_ISO100_sha1=b5a564cd9291224caf363b6b03054365d59d316b.cr2",
        "noisy_file": "Bayer/Kortlek/Bayer_Kortlek_ISO51200_sha1=8fed453fdfb162673bb9ed41f9c5f03095331e3b.cr2",
        "expected_shift": [0, -4],
        "shift_magnitude": "small",
        "notes": "Horizontal shift only, FFT works perfectly",
    },
]

# Additional aligned pairs for comparison
aligned_samples = [
    {
        "scene": "7D-1",
        "cfa": "Bayer",
        "camera": "Canon 7D",
        "gt_file": "Bayer/7D-1/gt/Bayer_7D-1_GT_ISO100_sha1=a5b16854dd40a3e9e7fab4ae5f8fa5c90e6e3e14.cr2",
        "noisy_file": "Bayer/7D-1/Bayer_7D-1_ISO800_sha1=11d2f7ac51732304c1ed32e352f3dd080f546c4b.cr2",
        "expected_shift": [0, 0],
        "shift_magnitude": "none",
        "notes": "Pre-aligned, test case for (0,0)",
    },
]

# X-Trans samples (may fail to load with current code)
xtrans_samples = [
    {
        "scene": "books",
        "cfa": "X-Trans",
        "camera": "Fujifilm (assumed)",
        "gt_file": "X-Trans/books/gt/X-Trans_books_GT_ISO200_sha1=a60e69a71c5cea7e62d5a61e3ce93b77ba6ac94a.raf",
        "noisy_files": [
            "X-Trans/books/X-Trans_books_ISO800_sha1=8f9d4c9a5b0e3f7a2c1d8e6f4b3a9c7d5e2f8a6b.raf",
            "X-Trans/books/X-Trans_books_ISO1600_sha1=7e8c6d5f4a3b2c1e9d8f7a6b5c4d3e2f1a9b8c7d.raf",
        ],
        "expected_shift": "unknown",
        "shift_magnitude": "unknown",
        "notes": "X-Trans 6x6 pattern, currently fails to load",
    },
]

# Combine into test dataset
test_dataset = {
    "version": "1.0",
    "created_for": "FFT channel-split investigation",
    "test_cases": {
        "misaligned_bayer": known_misaligned,
        "aligned_bayer": aligned_samples,
        "xtrans_samples": xtrans_samples,
    },
    "summary": {
        "total_misaligned": len(known_misaligned),
        "total_aligned": len(aligned_samples),
        "total_xtrans": len(xtrans_samples),
        "priorities": [
            "1. Test channel-split FFT on Bark ISO65535 (large shift with 2px error)",
            "2. Verify channel-split doesn't break small shifts",
            "3. Test on X-Trans if loading can be fixed",
        ],
    },
}

# Save
output_file = Path("curated_test_dataset.json")
with open(output_file, "w") as f:
    json.dump(test_dataset, f, indent=2)

print(f"Created curated test dataset: {output_file}")
print()
print("Test cases included:")
print(f"  - {len(known_misaligned)} misaligned Bayer pairs")
print(f"  - {len(aligned_samples)} aligned Bayer pairs")
print(f"  - {len(xtrans_samples)} X-Trans samples (may need fixing)")
print()
print("Priority test: Bark ISO65535 with (12,-10) shift")
print("  → This is where FFT returned (13,-9) with 2px error")
print("  → Hypothesis: CFA pattern interfering with FFT")
print("  → Solution: Split into R/G/B channels, FFT each, combine results")
print()

# Check files exist
base_path = Path("src/rawnind/datasets/RawNIND/src")
print("Verifying files exist...")
all_good = True

for category, cases in [("misaligned", known_misaligned), ("aligned", aligned_samples)]:
    for case in cases:
        gt_path = base_path / case["gt_file"]
        noisy_path = base_path / case["noisy_file"]

        if not gt_path.exists():
            print(f"  ❌ Missing: {case['gt_file']}")
            all_good = False
        if not noisy_path.exists():
            print(f"  ❌ Missing: {case['noisy_file']}")
            all_good = False

if all_good:
    print("  ✅ All Bayer test files exist")
else:
    print("  ⚠️  Some files missing")

print()
print("Next step: Implement channel-split FFT and test on this dataset")
