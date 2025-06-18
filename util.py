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

def find_overlapping_negative_superpixels(
    list_of_negative_superpixel_labels: list[list[int]],
    min_overlap_sources: int = 2 # X amount of images/sources
) -> list[int]:
    """
    Identifies superpixels that are deemed 'negative' by a minimum number of sources.
    This effectively finds the intersection of negative superpixels across multiple boolean masks.

    Args:
        list_of_negative_superpixel_labels (list[list[int]]): A list where each inner list
                                                               contains the labels of negative superpixels
                                                               from one source (e.g., depth, temp, obstacle).
        min_overlap_sources (int): The minimum number of sources that must identify a superpixel
                                   as negative for it to be considered a 'fused' negative superpixel.

    Returns:
        list[int]: A list of superpixel labels that meet the minimum overlap criteria.
    """
    if not list_of_negative_superpixel_labels:
        return []

    # Count how many times each superpixel label appears across all lists
    superpixel_counts = defaultdict(int)
    for superpixel_list in list_of_negative_superpixel_labels:
        # Use a set to count each superpixel only once per source
        for label in set(superpixel_list):
            superpixel_counts[label] += 1

    fused_negative_superpixels = []
    for label, count in superpixel_counts.items():
        if count >= min_overlap_sources:
            fused_negative_superpixels.append(label)

    return fused_negative_superpixels

def visualize_specific_superpixels(
        image_path: str,
        segments: np.ndarray,
        superpixel_labels_to_highlight: list[int],
        output_path: str,
        highlight_color: tuple[int, int, int] = (0, 0, 255),  # Default to Red (RGB)
        transparency_alpha: float = 0.5  # For blending the highlight
):
    """
    Highlights a specific set of superpixels on an image and saves the visualization.

    Args:
        image_path (str): Path to the original image.
        segments (np.ndarray): The superpixel segmentation map (output from SLIC).
        superpixel_labels_to_highlight (list[int]): A list of integer labels of superpixels to highlight.
        output_path (str): Path to save the visualized image.
        highlight_color (tuple[int, int, int]): The RGB color to use for highlighting (e.g., (255, 0, 0) for red).
        transparency_alpha (float): Transparency factor for the highlight overlay (0.0 to 1.0).
    """
    check_file_exists(image_path)  # Assuming check_file_exists is available

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Unable to load image for visualization at {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for consistency with highlight_color

    # Create an empty overlay for highlighting
    overlay = np.zeros_like(image_rgb, dtype=np.uint8)

    # Highlight the specified superpixels
    for label in superpixel_labels_to_highlight:
        # Create a boolean mask for the current superpixel label
        mask_segment = (segments == label)

        # Ensure mask_segment dimensions match the overlay's first two dimensions (height, width)
        if mask_segment.shape[:2] == overlay.shape[:2]:
            overlay[mask_segment] = list(highlight_color)  # Apply the highlight color
        else:
            print(
                f"DEBUG WARNING (visualize_specific_superpixels): Segment mask shape {mask_segment.shape} does not match image/overlay shape {overlay.shape}. Skipping highlight for label {label}.")

    # Blend the overlay with the original image
    # The image_rgb is uint8, overlay is uint8. addWeighted is appropriate.
    visualized_image_rgb = cv2.addWeighted(image_rgb, 1 - transparency_alpha, overlay, transparency_alpha, 0)

    # Optionally, draw superpixel boundaries on the final image for better context
    # This part can be resource intensive; remove if not strictly needed or for speed
    if segments.shape[:2] == visualized_image_rgb.shape[:2]:
        # mark_boundaries returns float, so convert back to uint8
        boundaries_image_float = mark_boundaries(visualized_image_rgb, segments, color=(0, 255, 0))  # Green boundaries
        # Ensure values are within 0-1 range before multiplying
        visualized_image_rgb = (np.clip(boundaries_image_float, 0, 1) * 255).astype(np.uint8)
    else:
        print(
            f"DEBUG WARNING (visualize_specific_superpixels): Segments shape {segments.shape} does not match visualization image shape {visualized_image_rgb.shape}. Skipping boundary drawing.")

    # Convert the final visualization back to BGR for saving with OpenCV
    visualized_image_bgr = cv2.cvtColor(visualized_image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, visualized_image_bgr)
    print(f"Specific superpixel visualization saved to {output_path}")

# (Ensure check_file_exists is defined or imported in this file)
# def check_file_exists(file_path):
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"File not found: {file_path}")

# Assuming check_file_exists is defined elsewhere in this file or imported.
# If not, you might need to add it here:
# def check_file_exists(file_path):
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"File not found: {file_path}")

def create_fused_boolean_mask(segments: np.ndarray, fused_labels: list[int], output_mask_path: str) -> np.ndarray:
    """
    Creates a binary boolean mask image based on a list of fused superpixel labels.

    Args:
        segments (np.ndarray): The superpixel segmentation map (output from SLIC),
                               where each pixel's value is its superpixel label.
        fused_labels (list[int]): A list of superpixel labels that are considered 'fused negative'.
        output_mask_path (str): Path to save the resulting binary mask image.

    Returns:
        np.ndarray: The resulting boolean mask (0s and 1s) as a NumPy array.
    """
    if segments is None or segments.ndim < 2:
        raise ValueError("Segments array must be a valid NumPy array with at least 2 dimensions.")
    if not isinstance(fused_labels, list):
        raise TypeError("Fused labels must be a list of integers.")

    print(f"\n--- Creating Fused Boolean Mask ---")

    # Get the dimensions from the segments array
    height, width = segments.shape[:2]

    # Initialize a blank boolean mask (all zeros/False)
    fused_boolean_mask = np.zeros((height, width), dtype=np.uint8)

    # Iterate through the fused labels and set corresponding pixels to 1 (True)
    for label in fused_labels:
        # Create a boolean mask for the current superpixel label
        # Set pixels belonging to this label in the fused_boolean_mask to 1
        fused_boolean_mask[segments == label] = 1

    # Save the resulting mask as a visual grayscale image (0=black, 255=white)
    visual_output_mask = (fused_boolean_mask * 255).astype(np.uint8)

    # --- DEBUG PRINTS FOR IMWRITE ISSUE ---
    print(f"DEBUG (create_fused_boolean_mask): Attempting to save to path: '{output_mask_path}'")

    # Get the directory part of the path
    output_dir = os.path.dirname(output_mask_path)
    if not output_dir:  # If output_mask_path is just a filename (e.g., "mask.jpg"), dir is current '.'
        output_dir = "."

    print(f"DEBUG (create_fused_boolean_mask): Parent directory for saving: '{output_dir}'")
    print(f"DEBUG (create_fused_boolean_mask): Parent directory exists: {os.path.exists(output_dir)}")
    print(
        f"DEBUG (create_fused_boolean_mask): Image data to save - shape: {visual_output_mask.shape}, Dtype: {visual_output_mask.dtype}, Channels: {visual_output_mask.ndim}")
    # --- END DEBUG PRINTS ---

    # Attempt to write the image
    cv2.imwrite(output_mask_path, visual_output_mask)
    print(f"Fused boolean mask saved to {output_mask_path}")

    # Display the boolean mask using matplotlib
    plt.imshow(fused_boolean_mask, cmap="gray")
    plt.title("Fused Negative Superpixels Boolean Mask")
    plt.axis("off")
    plt.show()

    return fused_boolean_mask

# If you don't have it already in this file:
# def check_file_exists(file_path):
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"File not found: {file_path}")
def find_closest_negative_pixel_spiral(boolean_map: np.ndarray) -> tuple[int, int] | None:
    """
    Finds the coordinates (row, col) of the closest 'negative space' pixel (value 1)
    in a boolean map, by spiraling outward from the center.

    Args:
        boolean_map (np.ndarray): A 2D NumPy array (uint8 or bool) where 1 represents
                                  negative space and 0 represents non-negative space.

    Returns:
        tuple[int, int] | None: A tuple (row, col) of the closest negative pixel,
                                or None if no negative pixel is found in the map.
    """
    if boolean_map is None or boolean_map.ndim != 2 or boolean_map.dtype not in [np.uint8, np.bool_]:
        raise ValueError("Input boolean_map must be a 2D NumPy array of type uint8 or bool.")

    rows, cols = boolean_map.shape

    # Calculate the starting center coordinates
    center_row = rows // 2
    center_col = cols // 2

    # Define directions: Right, Down, Left, Up (dx, dy)
    dr = [0, 1, 0, -1]  # Change in row
    dc = [1, 0, -1, 0]  # Change in column

    current_row, current_col = center_row, center_col
    direction_idx = 0  # Start moving right
    step_length = 1  # Number of steps to take in the current direction
    steps_taken_in_segment = 0

    # Max possible distance to check is roughly the diagonal from center to corner
    max_steps_overall = rows * cols  # A safe upper bound for total pixels to check

    print(f"\n--- Searching for closest negative pixel spiraling from center ({center_row}, {center_col}) ---")

    # Loop for spiral traversal
    # Continue as long as we are within bounds
    for _ in range(max_steps_overall):  # Prevent infinite loop on all-zero maps
        # Check the current pixel
        if 0 <= current_row < rows and 0 <= current_col < cols:
            if boolean_map[current_row, current_col] == 1:
                print(f"DEBUG: Found closest negative pixel at ({current_row}, {current_col}).")
                return (current_row, current_col)
        else:
            # If we go out of bounds, no need to check further in this direction without a turn
            # This 'else' branch helps prune the search if the map is small and we exit quickly.
            # However, the subsequent checks on `current_row, current_col` will also catch this.
            pass

        # Move to the next pixel in the current direction
        current_row += dr[direction_idx]
        current_col += dc[direction_idx]
        steps_taken_in_segment += 1

        # Check if we need to change direction and potentially increase step length
        if steps_taken_in_segment == step_length:
            steps_taken_in_segment = 0  # Reset steps for new segment
            direction_idx = (direction_idx + 1) % 4  # Change direction (0->1->2->3->0...)

            # Increase step length after every two turns (e.g., after Right then Down, or Left then Up)
            if direction_idx % 2 == 0:  # This means we've just completed a segment in the 'vertical' or 'horizontal' direction (Down or Up)
                # and are about to start a new 'horizontal' or 'vertical' (Right or Left).
                step_length += 1

        # Additional check to break if we are completely out of bounds and not likely to re-enter
        # This is more robust than just checking for `_ in range(max_steps_overall)`
        if not (0 <= current_row < rows and 0 <= current_col < cols):
            # We've spiraled off the map. If the center was the only point checked, and it's not 1,
            # and we go out of bounds, no need to continue.
            # This check prevents unnecessary loops after leaving the map.
            # However, the loop `for _ in range(max_steps_overall)` provides a hard limit.
            # A more precise check would involve checking if the entire map has been covered by the spiral.
            # For simplicity, `max_steps_overall` is usually sufficient for bounded maps.
            # A more robust solution might use a visited set for extremely sparse maps if desired.
            pass  # Keep looping to ensure all reachable pixels are checked if it was a very large map.

    print("DEBUG: No negative pixel found within the map using spiral search.")
    return None  # No negative pixel found