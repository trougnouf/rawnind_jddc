import yaml
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
import math
import numpy as np
from imageio import imwrite
import hashlib
import imageio.v3 as iio  # Modern imageio API for loading images

# REC2020_ICC_FPATH = (
#     "/orb/benoit_phd/src/ext/image_color_processing/data/lin_rec2020_from_dt.icc"
# )

# also in grapher.py
LITERATURE = {
    "jddc": {
        "[BM3D]": "[3]",
        "[NIND]": "[10]",
        "[OURS]": None,
        "[JPEGXL]": "[17]",
        "[COMPDENOISE]": "[24]",
        "[MANYPRIORS]": "[27]",
    },
    "thesis": {
        "[BM3D]": None,
        "[NIND]": None,
        "[OURS]": "(Ch. 5)",
        "[JPEGXL]": None,
        "[COMPDENOISE]": "(Ch. 4)",
        "[MANYPRIORS]": "(Ch. 3)",
    },
    "wiki": {
        "[BM3D]": None,
        "[NIND]": None,
        "[OURS]": None,
        "[JPEGXL]": None,
        "[COMPDENOISE]": None,
        "[MANYPRIORS]": None,
    },
}

YAML_FILES = [
    "/orb/benoit_phd/src/rawnind/plot_cfg/Picture2_32.yaml",
    "/orb/benoit_phd/src/rawnind/plot_cfg/Picture1_32.yaml",
]

# Rec. 2020 to Rec. 709 transformation matrix
REC2020_TO_REC709_MATRIX = np.array(
    [
        [1.6605, -0.5876, -0.0728],
        [-0.1246, 1.1329, -0.0083],
        [-0.0182, -0.1006, 1.1187],
    ]
)


def generate_unique_filename(filepath, prefix="closeup_"):
    """
    Generate a unique filename based on the file path using a checksum.

    Args:
        filepath (str): The original file path.
        prefix (str): A prefix for the new filename.

    Returns:
        str: A unique filename with the checksum.
    """
    # Generate a SHA-256 hash of the file path
    checksum = hashlib.sha256(filepath.encode()).hexdigest()[
        :8
    ]  # Use the first 8 characters
    filename = os.path.basename(filepath)
    unique_filename = f"{prefix}{checksum}_{filename}"
    return unique_filename


def linear_to_srgb(linear_rgb):
    """Apply gamma correction to convert linear RGB to sRGB."""
    # Ensure values are in the range [0, 1] (clip invalid values)
    linear_rgb = np.clip(linear_rgb, 0, 1)

    # Apply gamma correction
    srgb = np.where(
        linear_rgb <= 0.0031308,
        12.92 * linear_rgb,
        1.055 * np.power(linear_rgb, 1 / 2.4) - 0.055,
    )
    return np.clip(srgb, 0, 1)


def convert_rec2020_to_srgb(image):
    """
    Convert an image from Linear Rec. 2020 RGB to sRGB.

    Args:
        image (PIL.Image.Image): Input image in Linear Rec. 2020 RGB.

    Returns:
        PIL.Image.Image: Output image in sRGB.
    """
    # Convert PIL image to NumPy array (normalize to [0, 1])
    img_array = np.array(image).astype(np.float32) / 255.0

    # Apply Rec. 2020 to Rec. 709 transformation
    rec709_rgb = np.dot(img_array, REC2020_TO_REC709_MATRIX.T)

    # Apply gamma correction to get sRGB
    srgb = linear_to_srgb(rec709_rgb)

    # Convert back to uint8 and PIL.Image
    srgb_image = (srgb * 255).astype(np.uint8)
    return srgb_image


# Prepare placeholders for all images
output_dir = "tmp/processed_images"
os.makedirs(output_dir, exist_ok=True)


def create_placeholder_with_closeups(image_path, save_path):
    WHITESPACE = 50

    # Load 16-bit float TIFF image using imageio
    img_data = iio.imread(image_path)  # Reads as a NumPy array
    # rotate the image if image_path is in "gt" folder and endswith .tif and contains bluebirds
    if (
        (image_path.endswith(".tif"))
        and "bluebirds" in image_path
        and ("lin_rec2020" in image_path or "faux_Bayer" in image_path)
    ):
        img_data = np.rot90(img_data, 1)
    if img_data.dtype in [np.float32, np.float16]:  # Handle float images
        img_data = np.clip(img_data, 0.0, 1.0)  # Clip to valid range [0, 1]
        img_data = img_data * 255  # Normalize
    elif img_data.dtype == np.uint16:  # Handle 16-bit integer images
        img_data = img_data / 65535.0 * 255
    # if "bluebirds" in image_path or not image_path.endswith(".png") or True:
    img = convert_rec2020_to_srgb(img_data)
    # else:
    #     print(img_data.mean())
    #     # ensure mean is 128
    #     img_data = img_data - np.mean(img_data) + 34
    #     # clip to 0,255
    #     img_data = np.clip(img_data, 0, 255)
    #     img = (img_data).astype(np.uint8)
    # Convert to PIL Image
    img = Image.fromarray(img)
    # Convert to sRGB for better visualization

    width, height = img.size

    # Determine if it's landscape (width > height) or portrait
    is_landscape = width > height

    if is_landscape:
        new_height = int(height * 1.5)  # Increase height by 50% for bottom closeups
        placeholder_width = width
        placeholder = Image.new("RGB", (placeholder_width, new_height), "white")
        placeholder.paste(img, (0, 0))
        closeup_dimension = height // 2  # Height for all close-ups
        x_offset = 0
        paste_location = lambda x, y: (x, height)  # paste on the bottom
        resized_dimension_calculation = lambda w, h: int(closeup_dimension * (w / h))
    else:  # portrait
        new_width = int(width * 1.5)  # Increase width by 50% for right side closeups
        placeholder_height = height
        placeholder = Image.new("RGB", (new_width, placeholder_height), "white")
        placeholder.paste(img, (0, 0))
        closeup_dimension = width // 2  # Width for all close-ups
        y_offset = 0
        paste_location = lambda x, y: (width, y)  # paste to the right
        resized_dimension_calculation = lambda w, h: int(closeup_dimension * (h / w))

    if is_landscape:
        closeup_regions = [
            (2731, 1954, 2890, 2114),
            (1075, 1262, 1758, 1945),
            (1745, 23, 1976, 254),
        ]
    else:
        # Define close-up regions and properties
        closeup_regions = [
            (2755, 1873, 3172, 2232),  # Rectangle 1
            (3318, 4586, 3440, 4670),  # Rectangle 2
            (2035, 2677, 2265, 3017),  # Rectangle 3
        ]
        if (
            image_path.endswith(".tif")
            and "bluebirds" in image_path
            and "lin_rec2020" in image_path
        ):
            closeup_regions = [
                (3001, 2382, 3407, 2734),  # Rectangle 1
                (3687, 5067, 3809, 5151),  # Rectangle 2
                (2341, 3160, 2560, 3482),  # Rectangle 3
            ]
        elif (
            (image_path.endswith(".png"))
            and "bluebirds" in image_path
            and "faux_Bayer" in image_path
        ):
            closeup_regions = [
                (3001, 2402, 3417, 2754),  # Rectangle 1
                (3687, 5092, 3809, 5176),  # Rectangle 2
                (2341, 3180, 2560, 3512),  # Rectangle 3 / claw
            ]
    colors = ["cyan", "magenta", "yellow"]  # Colors for each close-up

    def draw_dashed_rectangle(draw_obj, x1, y1, x2, y2, color, dash_length, width):
        """Draw a dashed rectangle by manually creating dashed lines."""
        # Top edge
        for i in range(x1, x2, dash_length * 2):
            draw_obj.line(
                [(i, y1), (min(i + dash_length, x2), y1)], fill=color, width=width
            )
        # Left edge
        for i in range(y1, y2, dash_length * 2):
            draw_obj.line(
                [(x1, i), (x1, min(i + dash_length, y2))], fill=color, width=width
            )
        # Bottom edge
        for i in range(x1, x2, dash_length * 2):
            draw_obj.line(
                [(i, y2), (min(i + dash_length, x2), y2)], fill=color, width=width
            )
        # Right edge
        for i in range(y1, y2, dash_length * 2):
            draw_obj.line(
                [(x2, i), (x2, min(i + dash_length, y2))], fill=color, width=width
            )

    # Make a copy of the main image to ensure close-ups are independent
    main_image_with_rectangles = img.copy()
    main_image_draw = ImageDraw.Draw(main_image_with_rectangles)

    # Draw rectangles on the main image and prepare close-ups
    for idx, (x1, y1, x2, y2) in enumerate(closeup_regions):
        # Draw rectangles on the main image
        draw_dashed_rectangle(
            main_image_draw, x1, y1, x2, y2, colors[idx], dash_length=20, width=25
        )

        # Prepare close-ups
        cropped = img.crop((x1, y1, x2, y2))

        if is_landscape:
            resized_width = resized_dimension_calculation(cropped.width, cropped.height)
            resized = cropped.resize((resized_width, closeup_dimension), Image.LANCZOS)
        else:
            resized_height = resized_dimension_calculation(
                cropped.width, cropped.height
            )
            resized = cropped.resize((closeup_dimension, resized_height), Image.LANCZOS)

        # Create a new image for the resized close-up to draw dashed lines
        if is_landscape:
            resized_with_border = Image.new(
                "RGB", (resized_width, closeup_dimension), "white"
            )
        else:
            resized_with_border = Image.new(
                "RGB", (closeup_dimension, resized_height), "white"
            )
        resized_with_border.paste(resized, (0, 0))

        # Draw a dashed rectangle on the resized close-up
        closeup_draw = ImageDraw.Draw(resized_with_border)
        if is_landscape:
            draw_dashed_rectangle(
                closeup_draw,
                0,
                0,
                resized_width - 1,
                closeup_dimension - 1,
                colors[idx],
                dash_length=40,
                width=50,
            )
        else:
            draw_dashed_rectangle(
                closeup_draw,
                0,
                0,
                closeup_dimension - 1,
                resized_height - 1,
                colors[idx],
                dash_length=40,
                width=50,
            )

        # Add the resized close-up with dashed borders to the placeholder
        if is_landscape:
            placeholder.paste(resized_with_border, paste_location(x_offset, 0))
            x_offset += resized_width  # next x offset
        else:
            placeholder.paste(resized_with_border, paste_location(0, y_offset))
            y_offset += resized_height  # next y offset

    # Update placeholder size to include whitespace on the side/bottom
    if is_landscape:
        total_width = max(x_offset, width) + WHITESPACE
        placeholder_with_whitespace = Image.new(
            "RGB", (total_width, new_height), "white"
        )
    else:
        total_height = max(y_offset, height) + WHITESPACE
        placeholder_with_whitespace = Image.new(
            "RGB", (new_width, total_height), "white"
        )

    placeholder_with_whitespace.paste(placeholder, (0, 0))

    # Paste the updated main image (with rectangles) back into the placeholder
    placeholder_with_whitespace.paste(main_image_with_rectangles, (0, 0))

    # Save the placeholder with close-ups
    placeholder_with_whitespace.save(save_path)


def plot_section(
    fig,
    gs,
    section_data,
    section_name,
    row_offset,
    columns,  # Pass columns as a parameter
    show_only_first=False,
    show_only_last=False,
    include_bpp=True,
    yaml_file: str = "",
):
    """
    Plot a section (Input, Denoising, or Compression) with conditional caption display.
    """
    font_size = 16
    font_size_method = 19

    total_items = len(section_data)
    rows = math.ceil(total_items / (columns - 1))  # Subtract 1 for the method column

    for i, image_data in enumerate(section_data):
        row, col = divmod(i, columns - 1)
        row += row_offset

        # Determine when to show captions
        if show_only_first:
            show_caption = row == row_offset
        elif show_only_last:
            show_caption = row == row_offset + rows - 1
        else:
            show_caption = True

        # Plot method name in the first column
        if col == 0:
            ax = fig.add_subplot(gs[row, 0])
            ax.text(
                x=0.5,
                y=0.5,
                s=image_data.get("method", ""),
                fontsize=font_size_method,
                ha="center",
                va="center",
                rotation=90,
                weight=(
                    "bold" if "Input" in section_name else "normal"
                ),  # Make 'Input' bold
            )
            ax.axis("off")

        caption = f"MS-SSIM: {float(image_data.get('msssim', 0)):.3f}"
        if include_bpp:
            caption += f", {image_data['bpp']} bpp"
        if image_data.get("parameters"):
            if len(image_data["parameters"]) + len(caption) > 32:
                caption += "\n"
            else:
                caption += ", "
            caption += f"{image_data['parameters']}"

        ax = fig.add_subplot(gs[row, col + 1])  # Skip the method column
        img = Image.open(image_data["image_path"])
        ax.imshow(img)
        if show_caption:
            ax.set_title(label=caption, fontsize=font_size, pad=10)
        ax.axis("off")

        # Add a dashed line below the last image in the row
        show_dashed_line = col == columns - 2
        if (
            (section_name == "Input" and "Developed" not in image_data["method"])
            or (row == 5 and "Picture1" in yaml_file)
            or (row == 4 and "Picture2" in yaml_file)
        ):
            show_dashed_line = False

        if show_dashed_line:
            fig.add_subplot(gs[row, :])  # Span the entire row
            line_ax = plt.gca()
            line_ax.plot(
                [0, 1],
                [0, 0],
                transform=line_ax.transAxes,
                color="gray",
                linestyle="--",
                linewidth=0.5,
            )
            line_ax.axis("off")
    return rows


# Function to create and save a figure
def create_figure(section_data, file_suffix_1: str, file_suffix_2: str, yaml_file: str):
    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)

    # Define columns based on the YAML file name
    columns = 5  # if "Picture1" in yaml_file else 4
    fig_width = 23 if "Picture1" in yaml_file else 23  # Adjust for larger font size
    fig_height_per_row = 5  # if "Figure1" in yaml_file else 6
    hspace = (
        0.10
        if "denoising" in file_suffix_1.lower() and "Picture1" in yaml_file
        else 0.11
    )

    # Prepare placeholders for processed images
    processed_data = {"Input": [], "Denoising": [], "Compression": []}
    # Process images from YAML data
    for section, items in data.items():
        for method_item in items:
            method = method_item["method"]
            for method_acro, reference in LITERATURE[file_suffix_2].items():
                method = method.replace(method_acro, reference or "")

            for img in method_item["images"]:
                if (
                    section == "Compression"
                    and "Input" in method
                    and "Developed" not in method
                ):
                    continue

                src_fpath = img["src_fpath"]
                unique_filename = generate_unique_filename(filepath=src_fpath)
                placeholder_path = os.path.join(output_dir, unique_filename)

                create_placeholder_with_closeups(
                    image_path=src_fpath, save_path=placeholder_path
                )

                processed_data[section].append(
                    {
                        "method": method,
                        "image_path": placeholder_path,
                        "caption": img.get("caption"),
                        "bpp": img.get("bpp", None),
                        "msssim": img.get("msssim", None),
                        "parameters": img.get("parameters", None),
                    }
                )

    # Calculate total rows
    total_rows = math.ceil(len(processed_data["Input"]) / (columns - 1)) + math.ceil(
        len(section_data) / (columns - 1)
    )
    if "compression" in file_suffix_1.lower():
        if total_rows > 0:
            total_rows -= 2
    total_rows = max(1, total_rows)

    # Create the figure and GridSpec
    fig = plt.figure(
        figsize=(fig_width, total_rows * fig_height_per_row),
        tight_layout=True,
    )
    gs = plt.GridSpec(
        nrows=total_rows,
        ncols=columns,
        figure=fig,
        width_ratios=[0.1]
        + [1] * (columns - 1),  # Narrower first column for method names
        hspace=hspace,
    )

    row_offset = 0
    # Plot input section
    if "compression" in file_suffix_1.lower():
        input_data = [
            row for row in processed_data["Input"] if "Developed" in row["method"]
        ]
        row_offset += plot_section(
            fig=fig,
            gs=gs,
            section_data=input_data,
            section_name="Input",
            row_offset=row_offset,
            columns=columns,
            show_only_last=True,
            include_bpp=True,
            yaml_file=yaml_file,
        )
    else:
        row_offset += plot_section(
            fig=fig,
            gs=gs,
            section_data=processed_data["Input"],
            section_name="Input",
            row_offset=row_offset,
            columns=columns,
            show_only_first=True,
            include_bpp=False,
            yaml_file=yaml_file,
        )

    plot_section(
        fig=fig,
        gs=gs,
        section_data=section_data,
        section_name=file_suffix_1,
        row_offset=row_offset,
        columns=columns,
        include_bpp="compression" in file_suffix_1.lower(),
        yaml_file=yaml_file,
    )

    output_path = f"/orb/benoit_phd/wiki/Papers/JDDC/journal_paper/figures/{os.path.basename(yaml_file)}_{file_suffix_1}_{file_suffix_2}.pdf"
    # output_path = f"/orb/benoit_phd/wiki/Papers/JDDC/journal_paper/figures/{os.path.basename(yaml_file)}_{file_suffix_1}_{file_suffix_2}_w{fig_width}_h{fig_height_per_row}_hs{hspace}.pdf"
    plt.savefig(fname=output_path, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {output_path}")


# Iterate over YAML files and create figures
for a_paper, method_reference in LITERATURE.items():
    for yaml_file in YAML_FILES:
        with open(yaml_file, "r") as file:
            data = yaml.safe_load(file)

        processed_data = {"Input": [], "Denoising": [], "Compression": []}

        for section, items in data.items():
            for method_item in items:
                method = method_item["method"]
                for method_acro, reference in method_reference.items():
                    method = method.replace(method_acro, reference or "")

                for img in method_item["images"]:
                    if (
                        section == "Compression"
                        and "Input" in method
                        and "Developed" not in method
                    ):
                        continue

                    src_fpath = img["src_fpath"]
                    unique_filename = generate_unique_filename(filepath=src_fpath)
                    placeholder_path = os.path.join(output_dir, unique_filename)
                    create_placeholder_with_closeups(
                        image_path=src_fpath, save_path=placeholder_path
                    )

                    processed_data[section].append(
                        {
                            "method": method,
                            "image_path": placeholder_path,
                            "caption": img.get("caption"),
                            "bpp": img.get("bpp", None),
                            "msssim": img.get("msssim", None),
                            "parameters": img.get("parameters", None),
                        }
                    )
        create_figure(
            section_data=processed_data["Denoising"],
            file_suffix_1="denoising",
            file_suffix_2=a_paper,
            yaml_file=yaml_file,
        )
        create_figure(
            section_data=processed_data["Compression"],
            file_suffix_1="compression",
            file_suffix_2=a_paper,
            yaml_file=yaml_file,
        )
