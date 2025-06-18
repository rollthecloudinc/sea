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

# Helper function to check if a file exists
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

def apply_boolean_mask_by_color(depth_heatmap_path, output_mask_path):
    """
    Apply a boolean mask to the depth_heatmap image, marking green, blue, and violet as True,
    and orange, red, and yellow as False.

    Args:
        depth_heatmap_path (str): Path to the depth heatmap image.
        output_mask_path (str): Path to save the resulting boolean mask image.
    """
    check_file_exists(depth_heatmap_path)

    # Load the heatmap image
    heatmap = cv2.imread(depth_heatmap_path)
    if heatmap is None:
        raise ValueError(f"Error: Unable to load the depth heatmap image at {depth_heatmap_path}!")

    # --- ADD THIS DEBUG PRINT LINE ---
    print(f"DEBUG (apply_boolean_mask_by_color): Loaded heatmap dtype: {heatmap.dtype}, shape: {heatmap.shape}")
    # --- END DEBUG PRINT ---

    # Convert the heatmap to HSV color space
    heatmap_hsv = cv2.cvtColor(heatmap, cv2.COLOR_BGR2HSV)

    # Define the ranges for green, blue, and violet hues in HSV
    # Hue ranges are approximate and may need adjustment depending on the colormap used
    green_range = ((35, 50, 50), (85, 255, 255))  # Approx green hue range
    blue_range = ((100, 50, 50), (140, 255, 255))  # Approx blue hue range
    violet_range = ((140, 50, 50), (170, 255, 255))  # Approx violet hue range

    # Create masks for each color range
    green_mask = cv2.inRange(heatmap_hsv, green_range[0], green_range[1])
    blue_mask = cv2.inRange(heatmap_hsv, blue_range[0], blue_range[1])
    violet_mask = cv2.inRange(heatmap_hsv, violet_range[0], violet_range[1])

    # Combine masks for green, blue, and violet (True)
    true_mask = cv2.bitwise_or(cv2.bitwise_or(green_mask, blue_mask), violet_mask)

    # Create a mask for orange, red, and yellow hues (False)
    orange_range = ((10, 50, 50), (25, 255, 255))  # Approx orange hue range
    red_range = ((0, 50, 50), (10, 255, 255))  # Approx red hue range
    yellow_range = ((25, 50, 50), (35, 255, 255))  # Approx yellow hue range

    orange_mask = cv2.inRange(heatmap_hsv, orange_range[0], orange_range[1])
    red_mask = cv2.inRange(heatmap_hsv, red_range[0], red_range[1])
    yellow_mask = cv2.inRange(heatmap_hsv, yellow_range[0], yellow_range[1])

    false_mask = cv2.bitwise_or(cv2.bitwise_or(orange_mask, red_mask), yellow_mask)

    # Create the final boolean mask
    boolean_mask = np.zeros_like(true_mask, dtype=np.uint8)
    boolean_mask[true_mask > 0] = 1  # True for green, blue, violet
    boolean_mask[false_mask > 0] = 0  # False for orange, red, yellow

    # Save the resulting mask as a visual image (optional)
    visual_mask = (boolean_mask * 255).astype(np.uint8)  # Convert boolean mask to grayscale for visualization
    cv2.imwrite(output_mask_path, visual_mask)
    print(f"Boolean mask saved to {output_mask_path}")

    # Display the boolean mask using matplotlib for better visualization
    plt.imshow(boolean_mask, cmap="gray")
    plt.title("Boolean Mask (Green/Blue/Violet = True, Orange/Red/Yellow = False)")
    plt.axis("off")
    plt.show()

    return boolean_mask  # Return the boolean mask as a NumPy array for further analysis