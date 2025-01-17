import yaml
import matplotlib.pyplot as plt
import numpy as np

# Load YAML file
file_path = "../../datasets/RawNIND/RawNIND_masks_and_alignments.yaml"
with open(file_path, "r") as file:
    data = yaml.safe_load(file)

# Select images with specific image_set values
selected_images = [
    item
    for item in data
    if item.get("image_set")
    in [
        "7D-2",
        "Vaxt-i-trad",
        "Pen-pile",
        "MuseeL-vases-A7C",
        "D60-1",
        "MuseeL-Saint-Pierre-C500D",
        "TitusToys",
        "boardgames_top",
        "Laura_Lemons_platformer",
        "MuseeL-bluebirds-A7C",
    ]
]

# Get the "best_alignment_loss" values
#loss_values = [item.get("best_alignment_loss", 0.0) for item in selected_images]
loss_values = [item.get("rgb_msssim_score", 0.0) for item in selected_images]

# Sort the values for cumulative distribution plot
loss_values.sort()


# Plotting histogram
fig, ax = plt.subplots()
ax.hist(loss_values, bins=50, density=True)
ax.set_xlabel("MS-SSIM score")
ax.legend()


# # Plotting cumulative distribution
# fig, ax = plt.subplots()
# counts, bin_edges = np.histogram(loss_values, bins=50, density=True)
# cdf = np.cumsum(counts)
# ax.plot(bin_edges[1:], cdf / cdf[-1], label="Cumulative Distribution")
# ax.set_xlabel("Best Alignment Loss")
# ax.set_ylabel("Cumulative Probability")
# ax.set_title("Cumulative Distribution of Best Alignment Loss")
# ax.legend()

# Display the plot
plt.show()
