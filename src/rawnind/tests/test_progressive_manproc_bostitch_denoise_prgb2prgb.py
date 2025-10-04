import sys
import os

sys.path.append("..")
from rawnind.libs import rawds_manproc
from rawnind.libs import rawtestlib

RAWNIND_BOSTITCH_TEST_DESCRIPTOR_FPATH = os.path.join(
    "..", "..", "datasets", "RawNIND_Bostitch", "manproc_test_descriptor.yaml"
)
MS_SSIM_VALUES = {
    "le": {0.9975, 0.97, 0.99},
    "ge": [0.5, 0.9, 0.99, 1.00],
}
if __name__ == "__main__":
    preset_args = {"test_only": True, "init_step": None}
    if "--load_path" not in sys.argv:
        preset_args["load_path"] = None
    denoiserTraining = rawtestlib.DenoiseTestCustomDataloaderProfiledRGBToProfiledRGB(
        preset_args=preset_args
    )
    for operator, msssim_values in MS_SSIM_VALUES.items():
        for msssim_value in msssim_values:
            if operator == "le":
                kwargs = {"max_msssim_score": msssim_value}
            elif operator == "ge":
                kwargs = {"min_msssim_score": msssim_value}
            if any(
                score_key in denoiserTraining.json_saver.results["best_val"]
                for score_key in [
                    f"progressive_test_manproc_bostitch_rawnind_msssim_{operator}_{msssim_value}_msssim_loss.None",
                    f"progressive_test_manproc_bostitch_rawnind_msssim_{operator}_{msssim_value}_msssim_loss",
                ]
            ):
                print("Skipping test, best_val is known")
                continue
            dataset = rawds_manproc.ManuallyProcessedImageTestDataHandler(
                net_input_type="lin_rec2020",
                test_descriptor_fpath=RAWNIND_BOSTITCH_TEST_DESCRIPTOR_FPATH,
                **kwargs,
            )
            dataloader = dataset.batched_iterator()
            denoiserTraining.offline_custom_test(
                dataloader=dataloader,
                test_name=f"progressive_test_manproc_bostitch_rawnind_msssim_{operator}_{msssim_value}",
                save_individual_images=True,
            )
