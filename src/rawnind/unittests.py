import unittest
import sys
import torch

sys.path.append("..")

from rawnind.libs import raw
from rawnind.libs import rawproc


class TestRawproc(unittest.TestCase):
    def test_scenelin_to_pq(self):
        """Test that scenelin_to_pq pytorch matches numpy colour_science library."""
        ptbatch = torch.rand(5, 3, 256, 256) * 1.1 - 0.1
        npbatch = ptbatch.numpy()
        nptransformed = rawproc.scenelin_to_pq(npbatch)
        pttransformed = rawproc.scenelin_to_pq(ptbatch)
        self.assertTrue(
            torch.allclose(
                pttransformed,
                torch.from_numpy(nptransformed).to(torch.float32),
                atol=1e-05,
            )
        )

    def test_match_gain(self):
        """Check that the gain is matched for 0 and 1 and that the dimensions remain the same for single and batched images."""
        anchor_single_0 = torch.zeros(3, 256, 256, dtype=torch.float32)
        anchor_single_1 = torch.ones_like(anchor_single_0)
        anchor_batch_0 = torch.zeros(5, 3, 256, 256, dtype=torch.float32)
        anchor_batch_1 = torch.ones_like(anchor_batch_0)
        rand_single = torch.rand_like(anchor_single_0)
        rand_batch = torch.rand_like(anchor_batch_0)
        matched_single_0 = rawproc.match_gain(
            anchor_img=anchor_single_0, other_img=rand_single
        )
        matched_single_1 = rawproc.match_gain(
            anchor_img=anchor_single_1, other_img=rand_single
        )
        matched_batch_0 = rawproc.match_gain(
            anchor_img=anchor_batch_0, other_img=rand_batch
        )
        matched_batch_1 = rawproc.match_gain(
            anchor_img=anchor_batch_1, other_img=rand_batch
        )
        self.assertAlmostEqual(matched_single_0.mean().item(), 0.0, places=5)
        self.assertAlmostEqual(matched_single_1.mean().item(), 1.0, places=5)
        self.assertAlmostEqual(matched_batch_0.mean().item(), 0.0, places=5)
        self.assertAlmostEqual(matched_batch_1.mean().item(), 1.0, places=5)
        self.assertAlmostEqual(matched_single_0.shape, (3, 256, 256))
        self.assertAlmostEqual(matched_batch_1.shape, (5, 3, 256, 256))

    def test_camRGB_to_rec2020_batch_conversion(self):
        """Ensure the output of rawproc.camRGB_to_lin_rec2020_images (pytorch batched) is the same as that of raw.camRGB_to_profiledRGB_img (numpy single)."""
        rgb_xyz_matrix_1 = torch.tensor(
            [
                [0.9943, -0.3269, -0.0839],
                [-0.5323, 1.3269, 0.2259],
                [-0.1198, 0.2083, 0.7557],
                [0.0000, 0.0000, 0.0000],
            ]
        )
        rgb_xyz_matrix_2 = torch.tensor(
            [
                [0.6988, -0.1384, -0.0714],
                [-0.5631, 1.3410, 0.2447],
                [-0.1485, 0.2204, 0.7318],
                [0.0000, 0.0000, 0.0000],
            ]
        )
        rgb_xyz_matrix_3 = torch.tensor(
            [
                [0.7888, -0.1902, -0.1011],
                [-0.8106, 1.6085, 0.2099],
                [-0.2353, 0.2866, 0.7330],
                [0.0000, 0.0000, 0.0000],
            ]
        )
        rgb_xyz_matrix_4 = rgb_xyz_matrix_2
        rgb_xyz_matrices = torch.stack(
            (rgb_xyz_matrix_1, rgb_xyz_matrix_2, rgb_xyz_matrix_3, rgb_xyz_matrix_4)
        )
        img1_camRGB = torch.rand(3, 256, 256)
        img2_camRGB = torch.rand(3, 256, 256)
        img3_camRGB = torch.rand(3, 256, 256)
        img4_camRGB = torch.rand(3, 256, 256)
        images_camRGB = torch.stack(
            (img1_camRGB, img2_camRGB, img3_camRGB, img4_camRGB)
        )
        img1_rec2020 = torch.from_numpy(
            raw.camRGB_to_profiledRGB_img(
                img1_camRGB.numpy(),
                {"rgb_xyz_matrix": rgb_xyz_matrix_1.numpy()},
                "lin_rec2020",
            )
        )
        img2_rec2020 = torch.from_numpy(
            raw.camRGB_to_profiledRGB_img(
                img2_camRGB.numpy(),
                {"rgb_xyz_matrix": rgb_xyz_matrix_2.numpy()},
                "lin_rec2020",
            )
        )
        img3_rec2020 = torch.from_numpy(
            raw.camRGB_to_profiledRGB_img(
                img3_camRGB.numpy(),
                {"rgb_xyz_matrix": rgb_xyz_matrix_3.numpy()},
                "lin_rec2020",
            )
        )
        img4_rec2020 = torch.from_numpy(
            raw.camRGB_to_profiledRGB_img(
                img4_camRGB.numpy(),
                {"rgb_xyz_matrix": rgb_xyz_matrix_4.numpy()},
                "lin_rec2020",
            )
        )
        images_rec2020 = rawproc.camRGB_to_lin_rec2020_images(
            images_camRGB, rgb_xyz_matrices
        )
        individual_outputs = torch.stack(
            (img1_rec2020, img2_rec2020, img3_rec2020, img4_rec2020)
        )

        self.assertTrue(torch.allclose(individual_outputs, images_rec2020, atol=1e-06))


if __name__ == "__main__":
    unittest.main()
