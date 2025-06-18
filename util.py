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

# --- CORRECTED AND COMPLETE identify_negative_superpixel FUNCTION ---
def identify_negative_superpixel(
        original_image_path: str,
        boolean_mask_path: str,
        output_superpixel_visualization_path: str,
        desired_pixel_coverage_percent: float = 80.0,
        visualize: bool = True,
        segments: np.ndarray = None  # <--- CRITICAL: ACCEPTS PRE-COMPUTED SEGMENTS
) -> list[int]:  # <--- CRITICAL: NOW RETURNS A LIST OF INTEGER LABELS
    """
    Identifies superpixels that predominantly fall within the 'negative' (0) areas
    of a boolean mask and optionally visualizes them.

    Args:
        original_image_path (str): Path to the original RGB image (for visualization).
        boolean_mask_path (str): Path to the *already existing* boolean mask image (0 for negative, 255 for positive).
        output_superpixel_visualization_path (str): Path to save the superpixel visualization.
        desired_pixel_coverage_percent (float): The percentage of a superpixel's pixels
                                                that must be 'negative' (0) for the superpixel to be identified as negative.
        visualize (bool): If True, a visualization image will be saved.
        segments (np.ndarray, optional): Pre-computed superpixel segments array. This is REQUIRED
                                         for consistent fusion across multiple analyses.
                                         If None, a ValueError will be raised.

    Returns:
        list[int]: A list of superpixel labels (integers) that are identified as predominantly negative.
                   Returns an empty list if no such superpixels are found.
    """
    print(f"\n--- Identifying negative space superpixels for '{original_image_path}' ---")

    check_file_exists(original_image_path)
    check_file_exists(boolean_mask_path)

    image = cv2.imread(original_image_path)
    if image is None:
        raise ValueError(f"Error: Unable to load original image at {original_image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # For visualization

    mask = cv2.imread(boolean_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Error: Unable to load boolean mask at {boolean_mask_path}")

    # Ensure mask is binary (0 or 1). Assuming 0 in the visual mask means negative space.
    # If the boolean_mask_visual is 0 for obstacles/negative, and 255 for free space:
    boolean_mask_processed = (mask == 0).astype(np.uint8)  # 0 for negative, 1 for positive

    # Ensure provided segments exist and match image dimensions
    if segments is None:
        raise ValueError(
            "Error: Pre-computed 'segments' array is required for identify_negative_superpixel. Please pass it from main.py.")

    # Resize boolean mask if dimensions don't match (should ideally be handled during mask generation)
    if boolean_mask_processed.shape[:2] != image.shape[:2]:
        print(
            f"Warning: Boolean mask dimensions {boolean_mask_processed.shape} do not match original image {image.shape[:2]}. Resizing mask.")
        boolean_mask_processed = cv2.resize(boolean_mask_processed, (image.shape[1], image.shape[0]),
                                            interpolation=cv2.INTER_NEAREST)

    # Ensure segments match image dimensions (critical consistency check)
    if segments.shape[:2] != image.shape[:2]:
        raise ValueError(f"Provided segments shape {segments.shape} does not match image shape {image.shape}. "
                         "Please ensure consistent dimensions for SLIC generation in main.py.")

    # --- DEBUGGING PRINTS ---
    print(f"DEBUG (identify_negative_superpixel): Original image shape: {image.shape}, Dtype: {image.dtype}")
    print(
        f"DEBUG (identify_negative_superpixel): Processed boolean mask shape: {boolean_mask_processed.shape}, Dtype: {boolean_mask_processed.dtype}")
    print(f"DEBUG (identify_negative_superpixel): Segments shape: {segments.shape}, Dtype: {segments.dtype}")
    print(f"DEBUG (identify_negative_superpixel): Unique processed mask values: {np.unique(boolean_mask_processed)}")
    print(f"DEBUG (identify_negative_superpixel): Unique segment values (first few): {np.unique(segments)[:5]} ...")
    # --- END DEBUGGING PRINTS ---

    # Analyze each superpixel for its negative space coverage
    negative_superpixel_labels = []

    superpixel_pixel_counts = defaultdict(lambda: {'negative': 0, 'total': 0})

    # Iterate through all pixels and assign them to their superpixel label
    # This approach is generally more robust for collecting counts by label
    for (row, col), label in np.ndenumerate(segments):
        # Basic bounds check, though if images are same size, it should pass
        if row >= boolean_mask_processed.shape[0] or col >= boolean_mask_processed.shape[1]:
            print(
                f"DEBUG WARNING: Segments produced out-of-bounds index: ({row}, {col}) for mask of shape {boolean_mask_processed.shape}. Skipping pixel.")
            continue

        superpixel_pixel_counts[label]['total'] += 1

        # Check if the pixel is 'negative' (value 0 in our boolean_mask_processed)
        # Use .item() to ensure you get a scalar value from a 0-d array if that's what indexing returns
        if boolean_mask_processed[row, col].item() == 0:
            superpixel_pixel_counts[label]['negative'] += 1

    # Now, determine which superpixels are predominantly negative
    for label, counts in superpixel_pixel_counts.items():
        if counts['total'] > 0:  # Avoid division by zero
            negative_coverage = (counts['negative'] / counts['total']) * 100
            if negative_coverage >= desired_pixel_coverage_percent:
                negative_superpixel_labels.append(label)

    # Visualization
    if visualize:
        # --- ADDED DEBUG PRINTS FOR OVERLAY INITIALIZATION ---
        print(
            f"DEBUG (identify_negative_superpixel - VISUALIZE BLOCK): Entering visualize block for '{original_image_path}'.")
        print(
            f"DEBUG (identify_negative_superpixel - VISUALIZE BLOCK): State of image_rgb: shape={image_rgb.shape}, dtype={image_rgb.dtype}")
        # --- END ADDED DEBUG PRINTS ---

        try:
            # Create an empty overlay for highlighting on the original image
            # Ensure the overlay has the same dimensions and type as the original image_rgb
            overlay = np.zeros_like(image_rgb, dtype=np.uint8)
            print(
                f"DEBUG (identify_negative_superpixel - VISUALIZE BLOCK): 'overlay' variable successfully initialized. Shape: {overlay.shape}, Dtype: {overlay.dtype}")
        except Exception as e:
            print(f"ERROR (identify_negative_superpixel - VISUALIZE BLOCK): Failed to initialize 'overlay'. Error: {e}")
            raise  # Re-raise to immediately stop and see the initial error

        # Highlight negative superpixels with a distinct color (e.g., red)
        for label in negative_superpixel_labels:
            mask_segment = (segments == label)
            # Ensure mask_segment aligns with overlay dimensions
            if mask_segment.shape != overlay.shape[:2]:
                print(
                    f"DEBUG WARNING: Segment mask shape {mask_segment.shape} does not match overlay shape {overlay.shape[:2]}. Skipping visualization for label {label}.")
                continue
            # Ensure assigning a 3-channel color to a 3-channel overlay
            if overlay.ndim == 3 and mask_segment.ndim == 2:
                # Need to expand mask_segment to 3 dimensions to assign color correctly
                # or iterate for each channel if mask_segment is for a single channel
                # This ensures the assignment matches dimensions
                # Example: overlay[mask_segment] = [255, 0, 0] works if mask_segment is 2D boolean array
                overlay[mask_segment] = [255, 0, 0]  # Red color for negative superpixels (RGB)
            else:
                print(
                    f"DEBUG WARNING: Cannot apply color overlay due to dimension mismatch. Overlay ndim: {overlay.ndim}, Mask ndim: {mask_segment.ndim}")

        # Blend the overlay with the original image
        alpha = 0.6  # Transparency factor for the overlay
        # This output_image_viz is uint8
        output_image_viz = cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0)

        # Optionally, draw superpixel boundaries (on the blended image)
        if segments.shape[:2] == output_image_viz.shape[:2]:
            # mark_boundaries returns a float array (typically float64) with values in [0, 1]
            output_image_viz_float = mark_boundaries(output_image_viz, segments,
                                                     color=(0, 255, 0))  # Green boundaries (RGB)

            # --- CRITICAL FIX: Convert the float image back to uint8 (0-255) ---
            # Ensure values are clipped to 0-1 range before multiplying, as mark_boundaries can sometimes exceed 1.0 slightly
            output_image_viz = (np.clip(output_image_viz_float, 0, 1) * 255).astype(np.uint8)

        else:
            print(
                f"DEBUG WARNING: Segments shape {segments.shape} does not match visualization image shape {output_image_viz.shape}. Skipping boundary marking.")

        # Convert back to BGR for saving with OpenCV
        # This line will now receive a uint8 array, resolving the previous CV_64F error
        output_image_bgr = cv2.cvtColor(output_image_viz, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_superpixel_visualization_path, output_image_bgr)
        print(f"Negative superpixel visualization saved to {output_superpixel_visualization_path}")

    return negative_superpixel_labels