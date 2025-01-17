import os
import hashlib
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import imageio.v3 as iio


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

    # Handle RGBA images by removing the alpha channel
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    # Apply Rec. 2020 to Rec. 709 transformation
    rec709_rgb = np.dot(img_array, REC2020_TO_REC709_MATRIX.T)

    # Apply gamma correction to get sRGB
    srgb = linear_to_srgb(rec709_rgb)

    # Convert back to uint8 and PIL.Image
    srgb_image = (srgb * 255).astype(np.uint8)
    return srgb_image


def create_placeholder_with_closeups(image_path, save_path):
    WHITESPACE = 0  # Set to 0 here

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
    # img = convert_rec2020_to_srgb(img_data)
    img = img_data.astype(np.uint8)
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
            # (2728, 2014, 2831, 2117),
            (2792, 1777, 2911, 1895),
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
        # if (
        #     image_path.endswith(".tif")
        #     and "bluebirds" in image_path
        #     and "lin_rec2020" in image_path
        # ):
        #     closeup_regions = [
        #         (3001, 2382, 3407, 2734),  # Rectangle 1
        #         (3687, 5067, 3809, 5151),  # Rectangle 2
        #         (2341, 3160, 2560, 3482),  # Rectangle 3
        #     ]
        # elif (
        #     (image_path.endswith(".png"))
        #     and "bluebirds" in image_path
        #     and "faux_Bayer" in image_path
        # ):
        if (
            "/0_" in image_path
            or "/1_" in image_path
            or "/2_" in image_path
            or "/3_" in image_path
        ):
            closeup_regions = [
                (3025, 2383, 3378, 2702),  # Rectangle 1
                (3681, 5061, 3807, 5143),  # Rectangle 2
                (2350, 3158, 2562, 3479),  # Rectangle 3 / claw
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
    return placeholder_with_whitespace


def create_numbered_image(image_path, index):
    """Creates a processed image with the number on top-left"""
    output_dir = "tmp/processed_images"
    os.makedirs(output_dir, exist_ok=True)

    unique_filename = generate_unique_filename(filepath=image_path, prefix="numbered_")
    placeholder_path = os.path.join(output_dir, unique_filename)
    img_with_closeups = create_placeholder_with_closeups(image_path, placeholder_path)

    # Add number overlay
    draw = ImageDraw.Draw(img_with_closeups)
    # Use a font size that scales with the image dimensions
    font_size = int(min(img_with_closeups.width, img_with_closeups.height) * 0.05)
    font = ImageFont.truetype(
        "DejaVuSans.ttf", size=font_size
    )  # Use a better font and size
    draw.text((10, 10), " " + str(index), font=font, fill="white")  # position and color
    return img_with_closeups


def create_grid_figure(image_paths, output_path, grid_shape, start_index=0):
    """Creates and saves the grid figure with a given shape and start index."""

    # Calculate the max height
    max_height = 0
    for path in image_paths:
        img_data = iio.imread(path)  # Reads as a NumPy array
        # rotate the image if image_path is in "gt" folder and endswith .tif and contains bluebirds
        if (
            (path.endswith(".tif"))
            and "bluebirds" in path
            and ("lin_rec2020" in path or "faux_Bayer" in path)
        ):
            img_data = np.rot90(img_data, 1)
        if img_data.dtype in [np.float32, np.float16]:  # Handle float images
            img_data = np.clip(img_data, 0.0, 1.0)  # Clip to valid range [0, 1]
            img_data = img_data * 255  # Normalize
        elif img_data.dtype == np.uint16:  # Handle 16-bit integer images
            img_data = img_data / 65535.0 * 255
        # img = convert_rec2020_to_srgb(img_data)
        img = img_data.astype(np.uint8)
        img = Image.fromarray(img)
        max_height = max(max_height, img.height)

    images = []
    for i, path in enumerate(image_paths):
        images.append(create_numbered_image(path, i + start_index))

    # Calculate the max size for uniform layout
    max_width = max(img.width for img in images)

    # Create a new list of images with uniform size
    resized_images = []
    for img in images:
        resized = img.resize((max_width, max_height))
        resized_images.append(resized)

    # Create the grid layout
    rows, cols = grid_shape
    grid_width = max_width * cols
    grid_height = max_height * rows
    grid_image = Image.new("RGB", (grid_width, grid_height), "black")

    for i, img in enumerate(resized_images):
        row = i // cols
        col = i % cols
        x = col * max_width
        y = row * max_height
        grid_image.paste(img, (x, y))

    # Resize the final grid image to the target width
    target_width = 1500
    wpercent = target_width / float(grid_image.size[0])
    target_height = int((float(grid_image.size[1]) * float(wpercent)))
    resized_grid_image = grid_image.resize((target_width, target_height), Image.LANCZOS)

    resized_grid_image.save(output_path, "JPEG")
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    image_paths = [
        "/orb/benoit_phd/wiki/Papers/JDDC/journal_paper/figures/src/pipeline/darktable_exported/0_nothing_ISO16000_capt0002.png",
        "/orb/benoit_phd/wiki/Papers/JDDC/journal_paper/figures/src/pipeline/darktable_exported/1_bw_points_ISO16000_capt0002_01.png",
        "/orb/benoit_phd/wiki/Papers/JDDC/journal_paper/figures/src/pipeline/darktable_exported/2_demosaic_linrec2020_d65_ISO16000_capt0002_02.png",
        "/orb/benoit_phd/wiki/Papers/JDDC/journal_paper/figures/src/pipeline/darktable_exported/3_denoise_ISO16000_capt0002.arw.png",
        "/orb/benoit_phd/wiki/Papers/JDDC/journal_paper/figures/src/pipeline/darktable_exported/4_lenscorrection_perspective_cropISO16000_capt0002.arw_01.png",
        "/orb/benoit_phd/wiki/Papers/JDDC/journal_paper/figures/src/pipeline/darktable_exported/5_exposure_ISO16000_capt0002.arw.png",
        "/orb/benoit_phd/wiki/Papers/JDDC/journal_paper/figures/src/pipeline/darktable_exported/6_color_calibration_ISO16000_capt0002.arw_02.png",
        "/orb/benoit_phd/wiki/Papers/JDDC/journal_paper/figures/src/pipeline/darktable_exported/7_diffuseorsharpen_ISO16000_capt0002.arw_03.png",
        "/orb/benoit_phd/wiki/Papers/JDDC/journal_paper/figures/src/pipeline/darktable_exported/8_colorbalance_filmicISO16000_capt0002.arw_05.png",
    ]

    # Create the 3x3 grid with all images, named _thesis
    output_path_thesis = "/orb/benoit_phd/wiki/Papers/JDDC/journal_paper/figures/pipeline_stages_figure_thesis.jpg"
    create_grid_figure(image_paths, output_path_thesis, (3, 3))

    # Create a 4x2 grid skipping the first image, named _jddc
    output_path_jddc = "/orb/benoit_phd/wiki/Papers/JDDC/journal_paper/figures/pipeline_stages_figure_jddc.jpg"
    create_grid_figure(image_paths[1:], output_path_jddc, (2, 4), start_index=1)
