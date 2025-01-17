"""Adapted from: https://chatgpt.com/c/66f42139-5b74-800f-bbdf-b7a697e4f393"""

import os

# Base directory
base_dir = "/orb/benoit_phd/models/rawnind_dc"

# Subdirectories to be created
color_spaces = ["linRGB", "sRGB"]
compression_levels = [
    "1",
    "5",
    "10",
    "20",
    "30",
    "40",
    "50",
    "60",
]

# YAML content template
yaml_content = """
arbitrary_proc_method: {arbitrary_proc_method}
arch: JPEGXL
batch_size_clean: 1
batch_size_noisy: 4
bayer_only: true
bitEstimator_lr_multiplier: 10.0
bitstream_out_channels: 320
clean_dataset_yamlfpaths:
- ../../datasets/extraraw/trougnouf/crops_metadata.yaml
- ../../datasets/extraraw/raw-pixls/crops_metadata.yaml
comment: JPEGXL_{color_space}_{compression_level}
config: config/train_dc_{proc_method}.yaml
continue_training_from_last_model_if_exists: true
crop_size: 256
data_pairing: x_y
debug_options:
- minimize_threads
device: null
expname: JPEGXL_{proc_method}_{compression_level}
fallback_load_path: JPEGXL_{proc_method}_{compression_level}
funit: {compression_level}
hidden_out_channels: 192
in_channels: 3
init_lr: 0.0001
init_step: 1
load_path: ../../models/rawnind_dc/JPEGXL_{color_space}_{compression_level}/saved_models/iter_1.pt
loss: msssim_loss
lr_multiplier: 0.85
match_gain: input
metrics:
- msssim_loss
- mse
noise_dataset_yamlfpaths:
- /scratch/brummer/39509156/RawNIND/RawNIND_masks_and_alignments.yaml
num_crops_per_image: 2
patience: 100000
reset_lr: false
reset_optimizer_on_fallback_load_path: false
save_dpath: ../../models/rawnind_dc/JPEGXL_{color_space}_{compression_level}
test_crop_size: 1024
test_interval: 1500000
test_reserve:
- 7D-2
- Vaxt-i-trad
- Pen-pile
- MuseeL-vases-A7C
- D60-1
- MuseeL-Saint-Pierre-C500D
- TitusToys
- boardgames_top
- Laura_Lemons_platformer
- MuseeL-bluebirds-A7C
tot_steps: 6000000
train_lambda: 4.0
transfer_function: None
transfer_function_valtest: None
val_crop_size: 1024
val_interval: 15000
warmup_nsteps: 100000
"""

# Create args.yaml files in each directory
for color_space in color_spaces:
    for compression_level in compression_levels:
        # Define the directory
        dir_path = f"{base_dir}/JPEGXL_{color_space}_{compression_level}"
        os.makedirs(dir_path, exist_ok=True)

        # Define the file path
        file_path = os.path.join(dir_path, "args.yaml")

        # Replace placeholders in the template
        proc_method = "proc2proc" if color_space == "sRGB" else "prgb2prgb"
        yaml_data = yaml_content.format(
            color_space=color_space,
            compression_level=compression_level,
            proc_method=proc_method,
            arbitrary_proc_method="opencv" if color_space == "sRGB" else "null",
        )

        # Create the args.yaml file
        with open(file_path, "w") as file:
            file.write(yaml_data)

        print(f"Created: {file_path}")
