import os
import yaml

# Directory containing the models
base_dir = "/orb/benoit_phd/models/rawnind_denoise/"

# List of models to check
bm3d_models = [
    model for model in os.listdir(base_dir) if model.startswith("bm3d_sRGB_")
]

# Pictures and their directories
pictures = {
    "manproc": [
        "MuseeL-bluebirds-A7C_MuseeL-bluebirds-A7C_ISO50_capt0015.arw.tif_aligned_to_ISO50_capt0015.arw.tif",
        "MuseeL-bluebirds-A7C_MuseeL-bluebirds-A7C_ISO16000_capt0002.arw.tif_aligned_to_ISO50_capt0015.arw.tif",
        "MuseeL-bluebirds-A7C_MuseeL-bluebirds-A7C_ISO64000_capt0007.arw.tif_aligned_to_ISO50_capt0015.arw.tif",
        "MuseeL-bluebirds-A7C_MuseeL-bluebirds-A7C_ISO204800_capt0010.arw.tif_aligned_to_ISO50_capt0015.arw.tif",
    ],
    "manproc_bostitch": [
        "LucieB_bw_drawing1_LucieB_bw_drawing1_IMG_7692.CR2.tif_aligned_to_IMG_7692.CR2.tif",
        "LucieB_bw_drawing1_LucieB_bw_drawing1_IMG_7688.CR2.tif_aligned_to_IMG_7692.CR2.tif",
        "LucieB_bw_drawing1_LucieB_bw_drawing1_IMG_7689.CR2.tif_aligned_to_IMG_7692.CR2.tif",
        "LucieB_bw_drawing1_LucieB_bw_drawing1_IMG_7691.CR2.tif_aligned_to_IMG_7692.CR2.tif",
    ],
}

# Dictionary to store best results
best_results = {}

# Iterate through models and pictures to find the best MS-SSIM
for model in bm3d_models:
    for dir_name, pic_list in pictures.items():
        yaml_path = os.path.join(base_dir, model, dir_name, "iter_1.yaml")
        if not os.path.exists(yaml_path):
            continue

        # Load YAML file
        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)

        # Check each picture
        for picture in pic_list:
            if picture in data:
                msssim_loss = data[picture]["msssim_loss"]
                if (
                    picture not in best_results
                    or msssim_loss < best_results[picture]["msssim_loss"]
                ):
                    best_results[picture] = {
                        "msssim_loss": msssim_loss,
                        "model": model,
                        "dir": dir_name,
                    }

# Format the results
results = []
for picture, info in best_results.items():
    msssim = 1 - info["msssim_loss"]
    model_path = f"{info['model']}/{info['dir']}/iter_1/{picture}"
    results.append(f"MS-SSIM: {msssim:.6f}\n{model_path}")

# Output the results
output = "\n\n".join(results)
print(output)
