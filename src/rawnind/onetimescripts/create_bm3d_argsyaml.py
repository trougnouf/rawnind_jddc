"""See also: https://chatgpt.com/c/66f42139-5b74-800f-bbdf-b7a697e4f393"""
import os

# Base directory
base_dir = "/orb/benoit_phd/models/rawnind_denoise"

# Subdirectories to be created
color_spaces = ["linRGB", "sRGB"]
noise_levels = ["5", "10", "20", "30", "40", "50", "60", "70", "80", "90", "93", "95", "97", "99"]

# YAML content template
yaml_content = """
arbitrary_proc_method: null
arch: bm3d
batch_size_clean: 1
batch_size_noisy: 4
bayer_only: true
clean_dataset_yamlfpaths:
- ../../datasets/extraraw/trougnouf/crops_metadata.yaml
- ../../datasets/extraraw/raw-pixls/crops_metadata.yaml
comment: bm3d_{color_space}_{noise_level}
config: null
continue_training_from_last_model_if_exists: true
crop_size: 256
data_pairing: x_y
debug_options:
- minimize_threads
device: null
expname: bm3d_{color_space}_{noise_level}
fallback_load_path: null
funit: {noise_level}
in_channels: 3
init_lr: 0.0003
init_step: 1
load_path: ../../models/rawnind_denoise/bm3d_{color_space}_{noise_level}/saved_models/iter_1.pt
loss: msssim_loss
lr_multiplier: 0.85
match_gain: output
metrics:
- msssim_loss
- mse
noise_dataset_yamlfpaths:
- /scratch/brummer/39509156/RawNIND/RawNIND_masks_and_alignments.yaml
num_crops_per_image: 4
patience: 100000
preupsample: false
reset_lr: false
reset_optimizer_on_fallback_load_path: false
save_dpath: ../../models/rawnind_denoise/bm3d_{color_space}_{noise_level}
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
- LucieB_bw_drawing1
- LucieB_bw_drawing2
- LucieB_board
- LucieB_painted_wallpaper
- LucieB_painted_plants
- LucieB_groceries
tot_steps: 6000000
transfer_function: None
transfer_function_valtest: None
val_crop_size: 1024
val_interval: 15000
warmup_nsteps: 0
"""

# Create args.yaml files in each directory
for color_space in color_spaces:
    for noise_level in noise_levels:
        # Define the directory
        dir_path = f"{base_dir}/bm3d_{color_space}_{noise_level}"
        
        # Define the file path
        file_path = os.path.join(dir_path, "args.yaml")
        
        # Replace placeholders in the template
        yaml_data = yaml_content.format(color_space=color_space, noise_level=noise_level)
        
        # Create the args.yaml file
        with open(file_path, "w") as file:
            file.write(yaml_data)
        
        print(f"Created: {file_path}")


