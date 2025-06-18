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