#!/usr/bin/env python3

from PIL import Image, ImageDraw


def load_image(path):
    """
    Loads an image from the given path.

    Parameters:
    - path (str): Path to the image file.

    Returns:
    - PIL.Image.Image: Loaded image.
    """
    try:
        img = Image.open(path).convert("RGBA")
        return img
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None


def crop_to_width(image, target_width):
    """
    Crops the width of the image to match the target width (center-aligned).

    Parameters:
    - image (PIL.Image.Image): Image to crop.
    - target_width (int): Desired width.

    Returns:
    - PIL.Image.Image: Cropped image.
    """
    if image.width > target_width:
        left = (image.width - target_width) // 2
        right = left + target_width
        return image.crop((left, 0, right, image.height))
    return image


def create_dotted_line(
    draw, start, end, line_width=2, dot_spacing=10, fill=(0, 0, 0, 255)
):
    """
    Draws a horizontal dotted line between start and end points.

    Parameters:
    - draw (PIL.ImageDraw.Draw): ImageDraw object.
    - start (tuple): (x, y) starting point.
    - end (tuple): (x, y) ending point.
    - line_width (int): Width of the dotted line.
    - dot_spacing (int): Space between dots.
    - fill (tuple): Color of the line.
    """
    x_start, y = start
    x_end, _ = end
    for x in range(x_start, x_end, dot_spacing * 2):
        draw.line([(x, y), (x + dot_spacing, y)], fill=fill, width=line_width)


def combine_images_with_dotted_line(mosaic_bayer_path, mosaic_xtrans_path, output_path):
    """
    Combines two mosaic images stacked vertically with a horizontal dotted line.

    Parameters:
    - mosaic_bayer_path (str): Path to mosaic_Bayer.png.
    - mosaic_xtrans_path (str): Path to mosaic_X-Trans.png.
    - output_path (str): Path to save the combined image.
    """
    # Load images
    img_bayer = load_image(mosaic_bayer_path)
    img_xtrans = load_image(mosaic_xtrans_path)

    if img_bayer is None or img_xtrans is None:
        print("One or both images could not be loaded. Exiting.")
        return

    # Find the smaller width and crop both images to match
    target_width = min(img_bayer.width, img_xtrans.width)
    img_bayer = crop_to_width(img_bayer, target_width)
    img_xtrans = crop_to_width(img_xtrans, target_width)

    # Determine the width and height of the combined image
    dotted_line_height = 4  # Thickness of the dotted line
    combined_width = target_width
    combined_height = (
        img_bayer.height + img_xtrans.height + dotted_line_height - 1
    )  # Adjust for seamless alignment

    # Create a new transparent image
    combined_image = Image.new(
        "RGBA", (combined_width, combined_height), (255, 255, 255, 0)
    )

    # Paste bayer image on the top
    combined_image.paste(img_bayer, (0, 0), img_bayer)

    # Draw the dotted line
    draw = ImageDraw.Draw(combined_image)
    line_y = (
        img_bayer.height + 1
    )  # Slight overlap with the bottom row of the bayer image
    create_dotted_line(
        draw,
        (0, line_y),
        (combined_width, line_y),
        line_width=dotted_line_height,
        dot_spacing=15,
    )

    # Paste X-Trans image on the bottom
    xtrans_y_position = (
        img_bayer.height + dotted_line_height - 1
    )  # Align perfectly with the dotted line
    combined_image.paste(img_xtrans, (0, xtrans_y_position), img_xtrans)

    # Save the combined image
    try:
        final_image = combined_image.convert("RGB")
        final_image.save(output_path)
        print(f"Combined image saved to {output_path}.")
    except Exception as e:
        print(f"Failed to save combined image: {e}")


if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Combine two mosaic images stacked vertically with a dotted line."
    )
    parser.add_argument(
        "--mosaic_bayer",
        type=str,
        default="/orb/benoit_phd/datasets/RawNIND/Thumbnails/mosaic_Bayer.png",
        help="Path to the bayer mosaic image.",
    )
    parser.add_argument(
        "--mosaic_xtrans",
        type=str,
        default="/orb/benoit_phd/datasets/RawNIND/Thumbnails/mosaic_X-Trans.png",
        help="Path to the x-trans mosaic image.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/orb/benoit_phd/datasets/RawNIND/Thumbnails/combined_vertical_mosaic.png",
        help="Output path for the combined mosaic image.",
    )

    args = parser.parse_args()

    # Call the function to combine images
    combine_images_with_dotted_line(
        mosaic_bayer_path=args.mosaic_bayer,
        mosaic_xtrans_path=args.mosaic_xtrans,
        output_path=args.output,
    )
