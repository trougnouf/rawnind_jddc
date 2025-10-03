#!/usr/bin/env python3

"""
prereq: extract_dataset_thumbnails.py (adjust hardcoded MOSAICFORMAT for X-Trans and Bayer)

for jddc IEEEtran paper:

Bayer:
python onetimescripts/mk_mosaic.py --source_dir /orb/benoit_phd/datasets/RawNIND/Thumbnails/Bayer --height 640 --columns 27 --rows 7 --output /orb/benoit_phd/datasets/RawNIND/Thumbnails/mosaic_Bayer.png
X-Trans:
python onetimescripts/mk_mosaic.py --source_dir /orb/benoit_phd/datasets/RawNIND/Thumbnails/X-Trans --height 274 --columns 30 --rows 3 --output /orb/benoit_phd/datasets/RawNIND/Thumbnails/mosaic_X-Trans.png

then run mk_combined_mosaic.py

for thesis (16x20):
# aim for 1500 x 1500 px
# 189+111=300 pictures
# 3/2 format
# 30 rows x 20 columns
20 + 10 rows
9-10 columns + 11 columns
Bayer:
python onetimescripts/mk_mosaic.py --source_dir /orb/benoit_phd/datasets/RawNIND/Thumbnails/Bayer --height 1600 --width 1500 --columns 11 --rows 17 --output /orb/benoit_phd/datasets/RawNIND/Thumbnails/mosaic_Bayer_thesis.pngX-Trans:
python onetimescripts/mk_mosaic.py --source_dir /orb/benoit_phd/datasets/RawNIND/Thumbnails/X-Trans --width 1500 --columns 22 --rows 5 --height 225 --output /orb/benoit_phd/datasets/RawNIND/Thumbnails/mosaic_X-Trans_thesis.png
python onetimescripts/mk_combined_mosaic.py --mosaic_bayer /orb/benoit_phd/datasets/RawNIND/Thumbnails/mosaic_Bayer_thesis.png --mosaic_xtrans /orb/benoit_phd/datasets/RawNIND/Thumbnails/mosaic_X-Trans_thesis.png --output /orb/benoit_phd/wiki/Thesis/thesis_from_scratch/figures/jddc/mosaic.png
"""

import os
from PIL import Image, ImageChops, ImageEnhance, ImageOps


def crop_whitespace(img, bg_color=(255, 255, 255)):
    """
    Crops the white (background) space from the edges of an image.

    Parameters:
    - img (PIL.Image.Image): The image to crop.
    - bg_color (tuple): The RGB color of the background to remove. Default is white.

    Returns:
    - PIL.Image.Image: The cropped image.
    """
    # Create a background image of the same size and color
    bg = Image.new(img.mode, img.size, bg_color)

    # Compute the difference between the image and the background
    diff = ImageChops.difference(img, bg)

    # Convert the difference image to grayscale
    diff = diff.convert("L")

    # Get the bounding box of the non-background areas
    bbox = diff.getbbox()

    if bbox:
        # Crop the image to the bounding box
        return img.crop(bbox)
    else:
        # If no difference is found, return the original image
        return img


def normalize_image(img):
    """
    Normalizes the image to enhance brightness and contrast.

    Parameters:
    - img (PIL.Image.Image): The image to normalize.

    Returns:
    - PIL.Image.Image: The normalized image.
    """
    # Apply auto-contrast to enhance global contrast
    img = ImageOps.autocontrast(img)

    # Optional: Further enhance brightness and contrast
    # Uncomment the following lines if you want more control

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.2)  # Adjust the factor as needed

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)  # Adjust the factor as needed

    return img


def create_mosaic(
    source_dir,
    output_path,
    mosaic_width=3840,
    mosaic_height=640,
    rows=7,
    columns=27,
    overlap=0.05,  # Reduced overlap to 5%
):
    """
    Creates a mosaic from images in the source directory.

    Parameters:
    - source_dir (str): Path to the directory containing thumbnail images.
    - output_path (str): Path where the mosaic image will be saved.
    - mosaic_width (int): Width of the final mosaic in pixels.
    - mosaic_height (int): Height of the final mosaic in pixels.
    - rows (int): Number of rows in the mosaic grid.
    - columns (int): Number of columns in the mosaic grid.
    - overlap (float): Fraction of overlap between images (0 to 1).
    """
    # Collect all image file paths
    supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")
    image_files = [
        os.path.join(source_dir, file)
        for file in os.listdir(source_dir)
        if file.lower().endswith(supported_formats)
        and os.path.isfile(os.path.join(source_dir, file))
    ]

    if not image_files:
        print(f"No supported image files found in {source_dir}.")
        return

    total_images = len(image_files)
    print(f"Found {total_images} images.")

    # Calculate number of images per grid cell
    grid_capacity = rows * columns
    if total_images > grid_capacity:
        print(
            f"Warning: Number of images ({total_images}) exceeds grid capacity ({grid_capacity}). Some images will be skipped."
        )
        image_files = image_files[:grid_capacity]
        total_images = grid_capacity
    elif total_images < grid_capacity:
        print(
            f"Note: Number of images ({total_images}) is less than grid capacity ({grid_capacity}). Some grid cells will be empty."
        )

    # Calculate cell size with overlap
    cell_width = mosaic_width / (columns - (columns - 1) * overlap)
    cell_height = mosaic_height / (rows - (rows - 1) * overlap)
    cell_width = int(cell_width)
    cell_height = int(cell_height)
    print(
        f"Each grid cell size: {cell_width}x{cell_height} pixels with {overlap * 100}% overlap."
    )

    # Create a blank canvas
    mosaic_image = Image.new(
        "RGB", (mosaic_width, mosaic_height), color=(255, 255, 255)
    )

    # Iterate over grid and paste images
    for idx, image_path in enumerate(image_files):
        row = idx // columns
        col = idx % columns

        if row >= rows:
            print("Reached maximum grid capacity.")
            break

        # Calculate position with overlap
        x = int(col * (cell_width * (1 - overlap)))
        y = int(row * (cell_height * (1 - overlap)))

        try:
            with Image.open(image_path) as img:
                # Normalize the image to enhance brightness and contrast
                img = normalize_image(img)

                # Maintain aspect ratio with updated resampling filter
                try:
                    resample_filter = Image.Resampling.LANCZOS
                except AttributeError:
                    resample_filter = Image.ANTIALIAS

                img.thumbnail((cell_width, cell_height), resample=resample_filter)

                # Calculate position to paste the image (centered in the cell)
                paste_x = x + (cell_width - img.width) // 2
                paste_y = y + (cell_height - img.height) // 2

                # Paste the image onto the mosaic
                mosaic_image.paste(img, (paste_x, paste_y))
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    # Crop the white space from the top and bottom
    cropped_mosaic = crop_whitespace(mosaic_image, bg_color=(255, 255, 255))
    print("White space cropped from the mosaic.")

    # Save the cropped mosaic image
    try:
        cropped_mosaic.save(output_path)
        print(f"Mosaic saved to {output_path}.")
        # If the output path is .png, also save as .jpg
        if output_path.lower().endswith(".png"):
            jpg_output = output_path[:-4] + ".jpg"
            cropped_mosaic.save(jpg_output, "JPEG")
            print(f"Mosaic also saved to {jpg_output}.")
    except Exception as e:
        print(f"Failed to save mosaic image: {e}")


if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Create a mosaic from thumbnail images with normalization."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="/orb/benoit_phd/datasets/RawNIND/Thumbnails/Bayer",
        help="Directory containing thumbnail images.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/orb/benoit_phd/datasets/RawNIND/Thumbnails/mosaic.png",
        help="Output path for the mosaic image.",
    )
    parser.add_argument(
        "--width", type=int, default=3840, help="Width of the mosaic in pixels."
    )
    parser.add_argument(
        "--height", type=int, default=640, help="Height of the mosaic in pixels."
    )
    parser.add_argument(
        "--rows", type=int, default=7, help="Number of rows in the mosaic grid."
    )
    parser.add_argument(
        "--columns", type=int, default=27, help="Number of columns in the mosaic grid."
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.05,
        help="Fraction of overlap between images (0 to 1).",
    )

    args = parser.parse_args()

    # Call the create_mosaic function with provided arguments
    create_mosaic(
        source_dir=args.source_dir,
        output_path=args.output,
        mosaic_width=args.width,
        mosaic_height=args.height,
        rows=args.rows,
        columns=args.columns,
        overlap=args.overlap,
    )
