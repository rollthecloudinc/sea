import colorsys
from PIL import Image
import numpy as np

def extract_and_overlay_with_transparency_violet(
    image_path, grayscale_path, overlay_path, hue_min=270, hue_max=300, highlight_color=(128, 0, 128, 128)
):
    """
    Generate both a grayscale image based on violet saturation and a transparent overlay image.

    Args:
        image_path (str): Path to the input image.
        grayscale_path (str): Path to save the grayscale output image.
        overlay_path (str): Path to save the transparent overlay image.
        hue_min (int): Minimum hue value for violet detection (default is 270).
        hue_max (int): Maximum hue value for violet detection (default is 300).
        highlight_color (tuple): RGBA color to highlight detected violet areas (default is semi-transparent violet).
    """
    # Load the image
    image = Image.open(image_path).convert('RGBA')
    img_array = np.array(image, dtype=np.uint8)

    # Extract RGB channels and normalize to [0, 1]
    img_rgb = img_array[..., :3] / 255.0
    R, G, B = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]

    # Convert RGB to HSV
    H, S, V = np.vectorize(colorsys.rgb_to_hsv)(R, G, B)

    # Create a mask for pixels in the violet hue range
    violet_mask = (H >= hue_min / 360) & (H <= hue_max / 360)

    # Generate the grayscale image based on saturation
    grayscale = np.where(violet_mask, (S * 255).astype(np.uint8), 255)  # Non-violet pixels are white
    grayscale_image = Image.fromarray(grayscale, mode='L')
    grayscale_image.save(grayscale_path)
    print(f"Grayscale image saved to {grayscale_path}")

    # Generate the transparent overlay image
    overlay_array = np.copy(img_array)
    overlay_array[violet_mask] = highlight_color  # Apply highlight color to detected violet pixels
    overlay_image = Image.fromarray(overlay_array, mode='RGBA')
    overlay_image.save(overlay_path)
    print(f"Transparent overlay image saved to {overlay_path}")

# Examp
if __name__ == "__main__":
    # Paths for input and output images
    input_image_path = "input_image.jpg"  # Replace with the path to your input image
    grayscale_output_path = "grayscale_violet_output.jpg"  # Path for grayscale output
    transparent_overlay_output_path = "transparent_violet_overlay_output.png"  # Path for transparent overlay output

    # Call the function
    extract_and_overlay_with_transparency_violet(
        image_path=input_image_path,
        grayscale_path=grayscale_output_path,
        overlay_path=transparent_overlay_output_path,
        hue_min=270,  # Minimum hue value for violet detection (270°)
        hue_max=300,  # Maximum hue value for violet detection (300°)
        highlight_color=(128, 0, 128, 128)  # Semi-transparent violet highlight
    )