import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from skimage.feature import local_binary_pattern
from skimage.segmentation import slic # Import SLIC
from skimage.segmentation import mark_boundaries # For visualization
import colorsys
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict # <--- ADD THIS LINE
from util import check_file_exists

# Color-Based Overlay Functions
def extract_and_overlay_with_transparency(
    image_path, grayscale_path, overlay_path, hue_min, hue_max, highlight_color
):
    """
    Generic function to handle transparency overlays for specific color ranges.
    """
    check_file_exists(image_path)
    image = Image.open(image_path).convert('RGBA')
    img_array = np.array(image, dtype=np.uint8)

    # Extract RGB and normalize to [0, 1]
    img_rgb = img_array[..., :3] / 255.0
    R, G, B = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]

    # Convert RGB to HSV
    H, S, V = np.vectorize(colorsys.rgb_to_hsv)(R, G, B)

    # Create a mask for pixels in the specified hue range
    color_mask = (H >= hue_min / 360) & (H <= hue_max / 360)

    # Generate the grayscale image
    grayscale = np.where(color_mask, (S * 255).astype(np.uint8), 255)
    grayscale_image = Image.fromarray(grayscale, mode='L')
    grayscale_image.save(grayscale_path)
    print(f"Grayscale image saved to {grayscale_path}")

    # Generate the transparent overlay image
    overlay_array = np.copy(img_array)
    overlay_array[color_mask] = highlight_color
    overlay_image = Image.fromarray(overlay_array, mode='RGBA')
    overlay_image.save(overlay_path)
    print(f"Transparent overlay image saved to {overlay_path}")

def generate_temperature_heatmap(
        red_grayscale_path,
        orange_grayscale_path,
        yellow_grayscale_path,
        green_grayscale_path,
        blue_grayscale_path,
        violet_grayscale_path,
        output_heatmap_path,
):
    """
    Generate a visual heatmap that accounts for the intensity (saturation) of colors,
    transitioning smoothly from warm (red) to cool (blue) with less saturated regions appearing lighter.

    Args:
        red_grayscale_path (str): Path to the grayscale image for red.
        orange_grayscale_path (str): Path to the grayscale image for orange.
        yellow_grayscale_path (str): Path to the grayscale image for yellow.
        green_grayscale_path (str): Path to the grayscale image for green.
        blue_grayscale_path (str): Path to the grayscale image for blue.
        violet_grayscale_path (str): Path to the grayscale image for violet.
        output_heatmap_path (str): Path to save the resulting temperature heatmap.
    """
    # Load all grayscale images as numpy arrays
    red = cv2.imread(red_grayscale_path, cv2.IMREAD_GRAYSCALE)
    orange = cv2.imread(orange_grayscale_path, cv2.IMREAD_GRAYSCALE)
    yellow = cv2.imread(yellow_grayscale_path, cv2.IMREAD_GRAYSCALE)
    green = cv2.imread(green_grayscale_path, cv2.IMREAD_GRAYSCALE)
    blue = cv2.imread(blue_grayscale_path, cv2.IMREAD_GRAYSCALE)
    violet = cv2.imread(violet_grayscale_path, cv2.IMREAD_GRAYSCALE)

    # Verify that all images are loaded
    if red is None or orange is None or yellow is None or green is None or blue is None or violet is None:
        raise ValueError("Error: One or more grayscale images could not be loaded!")

    # Normalize grayscale values to the range [0, 1]
    red = red / 255.0
    orange = orange / 255.0
    yellow = yellow / 255.0
    green = green / 255.0
    blue = blue / 255.0
    violet = violet / 255.0

    # Combine warm and cool colors, scaling by their intensities
    warm_colors = (red + orange + yellow) / 3  # Average warm saturation
    cool_colors = (green + blue + violet) / 3  # Average cool saturation

    # Create the temperature map based on warm and cool contributions
    # Areas with less saturation will naturally have intermediate/lighter values
    temperature_map = warm_colors - cool_colors  # Positive = warm, Negative = cool

    # Normalize the temperature map to range [0, 255] for visualization
    temperature_map_normalized = cv2.normalize(temperature_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply a colormap for visualization
    # COLORMAP_JET transitions from blue (cool) to red (warm) smoothly
    temperature_heatmap = cv2.applyColorMap(temperature_map_normalized, cv2.COLORMAP_JET)

    # Save the resulting heatmap
    cv2.imwrite(output_heatmap_path, temperature_heatmap)
    print(f"Temperature heatmap saved to {output_heatmap_path}")

    # Display the heatmap using matplotlib for better visualization
    plt.imshow(cv2.cvtColor(temperature_heatmap, cv2.COLOR_BGR2RGB))
    plt.title("Temperature Heatmap (Warm to Cool with Saturation Intensity)")
    plt.axis("off")
    plt.show()

    # Return the temperature map as a numpy array for further analysis if needed
    return temperature_map_normalized