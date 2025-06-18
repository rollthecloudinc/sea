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
from depth import depth_estimation_heatmap
from util import check_file_exists
from shapes import detect_shapes
from color import extract_and_overlay_with_transparency

# Texture Analysis using LBP
def analyze_texture(image_path, output_path, radius=3, num_points=24, threshold=0.2):
    check_file_exists(image_path)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Unable to read the image!")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Performing texture analysis using Local Binary Patterns (LBP)...")
    lbp = local_binary_pattern(gray, num_points, radius, method="uniform")
    lbp_normalized = lbp / np.max(lbp)
    proximity_mask = np.where(lbp_normalized > threshold, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(proximity_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated_image = image.copy()

    for contour in contours:
        cv2.drawContours(annotated_image, [contour], -1, (0, 0, 255), 2)  # Red for close objects

    cv2.imwrite(output_path, annotated_image)
    print(f"Texture-based proximity detection saved to {output_path}")

    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.title("Texture Analysis")
    plt.axis("off")
    plt.show()

# Navigable Heatmap Generation
def generate_navigable_heatmap(image_path, output_heatmap_path):

    check_file_exists(image_path)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Unable to read the image!")

    input_height, input_width = 256, 256
    image_resized = cv2.resize(image, (input_width, input_height))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # Load MiDaS model
    print("Loading MiDaS model...")
    model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((input_height, input_width)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    input_tensor = transform(image_rgb).unsqueeze(0).to(device)

    print("Performing depth estimation...")
    with torch.no_grad():
        depth_map = model(input_tensor).squeeze().cpu().numpy()

    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Threshold depth map for navigable space
    depth_threshold = 50
    navigable_space = np.where(depth_map > depth_threshold, 255, 0).astype(np.uint8)

    # Detect obstacles
    edges = cv2.Canny(depth_map_normalized, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    obstacle_mask = np.zeros_like(depth_map_normalized)
    cv2.drawContours(obstacle_mask, contours, -1, 255, thickness=cv2.FILLED)

    # Create buffer zones
    buffer_zone_mask = cv2.dilate(obstacle_mask, kernel=np.ones((15, 15), np.uint8), iterations=1)

    # Generate heatmap
    heatmap = np.zeros_like(depth_map_normalized, dtype=np.uint8)
    heatmap[navigable_space == 255] = 128  # Green for navigable space
    heatmap[buffer_zone_mask == 255] = 200  # Yellow for buffer zones
    heatmap[obstacle_mask == 255] = 255    # Red for obstacles

    # Overlay the heatmap on the original image
    heatmap_bgr = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_resized = cv2.resize(heatmap_bgr, (image.shape[1], image.shape[0]))
    overlay = cv2.addWeighted(image, 0.6, heatmap_resized, 0.4, 0)

    cv2.imwrite(output_heatmap_path, overlay)
    print(f"Navigable space heatmap saved to {output_heatmap_path}")

    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Navigable Space Heatmap")
    plt.axis("off")
    plt.show()


def generate_temperature_heatmap_v2(
        red_grayscale_path,
        orange_grayscale_path,
        yellow_grayscale_path,
        green_grayscale_path,
        blue_grayscale_path,
        violet_grayscale_path,
        output_heatmap_path,
):
    """
    Generate a visual heatmap that transitions smoothly from warm colors (red) to cool colors (blue).

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

    # Define warm and cool color contributions
    warm_colors = red + orange + yellow  # Warm: Red, Orange, Yellow
    cool_colors = green + blue + violet  # Cool: Green, Blue, Violet

    # Normalize warm and cool intensities to [0, 1]
    warm_colors = np.clip(warm_colors, 0, 1)
    cool_colors = np.clip(cool_colors, 0, 1)

    # Combine warm and cool contributions into a single map
    # Warm colors are mapped to higher values, cool colors to lower values
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
    plt.title("Temperature Heatmap (Warm to Cool)")
    plt.axis("off")
    plt.show()

    # Return the temperature map as a numpy array for further analysis if needed
    return temperature_map_normalized


def generate_temperature_heatmap_v3(
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

def generate_obstacle_heatmap(detected_shapes_image_path, output_heatmap_path):
    """
    Generate a heatmap where areas with obstacles or hard edges are highlighted as warm (red),
    and free space is highlighted as cool (blue).

    Args:
        detected_shapes_image_path (str): Path to the image with detected shapes (output of detect_shapes).
        output_heatmap_path (str): Path to save the resulting obstacle heatmap.
    """
    # Load the detected shapes image
    image = cv2.imread(detected_shapes_image_path)
    if image is None:
        raise ValueError(f"Error: Unable to load the image at {detected_shapes_image_path}!")

    # Convert the image to grayscale for contour detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours in the edge map
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank obstacle heatmap
    obstacle_map = np.zeros_like(gray, dtype=np.float32)

    # Fill the obstacle map based on contours
    for contour in contours:
        # Assign high heat value (e.g., 1.0) to pixels inside obstacles (contours)
        cv2.drawContours(obstacle_map, [contour], -1, 1.0, thickness=cv2.FILLED)

    # Normalize the obstacle map to range [0, 255] for visualization
    obstacle_map_normalized = cv2.normalize(obstacle_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply a colormap for visualization
    # COLORMAP_JET transitions from blue (cool) to red (warm) smoothly
    obstacle_heatmap = cv2.applyColorMap(obstacle_map_normalized, cv2.COLORMAP_JET)

    # Save the resulting obstacle heatmap
    cv2.imwrite(output_heatmap_path, obstacle_heatmap)
    print(f"Obstacle heatmap saved to {output_heatmap_path}")

    # Display the heatmap using matplotlib for better visualization
    plt.imshow(cv2.cvtColor(obstacle_heatmap, cv2.COLOR_BGR2RGB))
    plt.title("Obstacle Heatmap")
    plt.axis("off")
    plt.show()

    # Return the obstacle map as a numpy array for further analysis if needed
    return obstacle_map

def generate_smooth_obstacle_heatmap(detected_shapes_image_path, output_heatmap_path):
    """
    Generate a smooth obstacle heatmap where:
      - Inside shapes (obstacles): Transition from red (edges) to orange/yellow (center).
      - Outside shapes (free space): Transition from violet (near edges) to blue/green (far from obstacles).

    Args:
        detected_shapes_image_path (str): Path to the image with detected shapes (output of detect_shapes).
        output_heatmap_path (str): Path to save the resulting smooth obstacle heatmap.
    """
    # Load the detected shapes image
    image = cv2.imread(detected_shapes_image_path)
    if image is None:
        raise ValueError(f"Error: Unable to load the image at {detected_shapes_image_path}!")

    # Convert the image to grayscale for contour detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Create a binary mask for obstacle regions
    obstacle_mask = np.zeros_like(gray, dtype=np.uint8)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(obstacle_mask, contours, -1, 255, thickness=cv2.FILLED)

    # Compute the distance transform for inside and outside the shapes
    inside_distance = cv2.distanceTransform(obstacle_mask, cv2.DIST_L2, 5)
    outside_distance = cv2.distanceTransform(cv2.bitwise_not(obstacle_mask), cv2.DIST_L2, 5)

    # Normalize distances to range [0, 1]
    inside_distance_normalized = inside_distance / np.max(inside_distance)
    outside_distance_normalized = outside_distance / np.max(outside_distance)

    # Combine inside and outside distances into a single temperature map
    temperature_map = np.zeros_like(inside_distance_normalized, dtype=np.float32)
    temperature_map[obstacle_mask > 0] = 0.5 + 0.5 * inside_distance_normalized[obstacle_mask > 0]  # Red to Yellow
    temperature_map[obstacle_mask == 0] = 0.5 * (1 - outside_distance_normalized[obstacle_mask == 0])  # Violet to Blue/Green

    # Normalize the temperature map to range [0, 255] for visualization
    temperature_map_normalized = cv2.normalize(temperature_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply a colormap for visualization
    # COLORMAP_JET transitions smoothly from violet (coolest) to red (warmest)
    smooth_obstacle_heatmap = cv2.applyColorMap(temperature_map_normalized, cv2.COLORMAP_JET)

    # Save the resulting heatmap
    cv2.imwrite(output_heatmap_path, smooth_obstacle_heatmap)
    print(f"Smooth obstacle heatmap saved to {output_heatmap_path}")

    # Display the heatmap using matplotlib for better visualization
    plt.imshow(cv2.cvtColor(smooth_obstacle_heatmap, cv2.COLOR_BGR2RGB))
    plt.title("Smooth Obstacle Heatmap (Inside & Outside Transition)")
    plt.axis("off")
    plt.show()

    # Return the temperature map as a numpy array for further analysis if needed
    return temperature_map

def generate_navigable_heatmap_v2(image_path, output_heatmap_path):
    """
    Generate a navigable heatmap where:
      - Navigable areas are highlighted in cool tones (blue/green).
      - Buffer zones are highlighted in intermediate tones (yellow).
      - Obstacles are highlighted in warm tones (red).

    Args:
        image_path (str): Path to the input image.
        output_heatmap_path (str): Path to save the resulting navigable heatmap image.
    """
    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Unable to read the image at {image_path}!")

    # Resize the image for compatibility with MiDaS
    input_height, input_width = 256, 256
    image_resized = cv2.resize(image, (input_width, input_height))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # Load MiDaS model
    print("Loading MiDaS model...")
    model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
    model.eval()

    # Use CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare the image for the MiDaS model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((input_height, input_width)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    input_tensor = transform(image_rgb).unsqueeze(0).to(device)

    print("Performing depth estimation...")
    with torch.no_grad():
        depth_map = model(input_tensor).squeeze().cpu().numpy()

    # Normalize the depth map for processing
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Threshold depth map to identify navigable space
    depth_threshold = 50  # Minimum clearance distance
    navigable_space = np.where(depth_map > depth_threshold, 255, 0).astype(np.uint8)

    # Detect obstacles using edge detection
    edges = cv2.Canny(depth_map_normalized, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create obstacle mask
    obstacle_mask = np.zeros_like(depth_map_normalized)
    cv2.drawContours(obstacle_mask, contours, -1, 255, thickness=cv2.FILLED)

    # Create buffer zones
    buffer_zone_mask = cv2.dilate(obstacle_mask, kernel=np.ones((15, 15), np.uint8), iterations=1)

    # Create a combined heatmap
    heatmap = np.zeros_like(depth_map_normalized, dtype=np.float32)
    heatmap[navigable_space == 255] = 0.3

    # Navigable areas (cool tones: blue/green)
    heatmap[buffer_zone_mask == 255] = 0.6  # Buffer zones (intermediate tones: yellow)
    heatmap[obstacle_mask == 255] = 1.0  # Obstacles (warm tones: red)

    # Normalize the heatmap to range [0, 255] for visualization
    heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply a colormap for visualization
    # COLORMAP_JET transitions smoothly from blue (coolest) to red (warmest)
    navigable_heatmap = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

    # Resize the heatmap back to the original image size
    heatmap_resized = cv2.resize(navigable_heatmap, (image.shape[1], image.shape[0]))

    # Save the resulting navigable heatmap
    cv2.imwrite(output_heatmap_path, heatmap_resized)
    print(f"Navigable heatmap saved to {output_heatmap_path}")

    # Display the heatmap using matplotlib for better visualization
    plt.imshow(cv2.cvtColor(heatmap_resized, cv2.COLOR_BGR2RGB))
    plt.title("Navigable Heatmap (Smooth Transitions)")
    plt.axis("off")
    plt.show()

    # Return the heatmap as a NumPy array for further analysis if needed
    return heatmap

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


# --- UPDATED FUNCTION: identify_negative_superpixel ---
def identify_negative_superpixel(
        original_image_path: str,
        boolean_mask_path: str,  # Now takes the path to the *pre-computed* boolean mask
        output_superpixel_visualization_path: str,  # New path to save the superpixel visualization
        num_superpixels: int = 1000,
        compactness: float = 10.0,
        min_size_factor: float = 0.05,
        desired_pixel_coverage_percent: float = 100.0,
        visualize: bool = True
) -> list:
    """
    Identifies superpixels that are entirely (or mostly) "negative space" (0 values)
    in the boolean depth heatmap.

    Args:
        original_image_path (str): Path to the original input image (for SLIC and visualization).
        boolean_mask_path (str): Path to the *already existing* boolean mask image (e.g., from apply_boolean_mask_by_color).
        output_superpixel_visualization_path (str): Path to save the superpixel visualization image.
        num_superpixels (int): Approximate number of superpixels to generate.
        compactness (float): Balances color proximity and space proximity for SLIC.
        min_size_factor (float): Superpixels smaller than this fraction of the average size are merged.
        desired_pixel_coverage_percent (float): The percentage of pixels within a superpixel that must be 0
                                                 (negative space) for it to be identified. Default 100%.
        visualize (bool): If True, displays the superpixel segmentation and the identified superpixels.

    Returns:
        list: A list of dictionaries, each representing an identified superpixel
              with its 'label', 'centroid' (y, x), and 'bbox' (min_row, min_col, max_row, max_col).
              Returns an empty list if no such superpixel is found.
    """
    print(f"\n--- Identifying negative space superpixels for '{original_image_path}' ---")

    check_file_exists(original_image_path)
    check_file_exists(boolean_mask_path)

    original_image = cv2.imread(original_image_path)
    if original_image is None:
        raise ValueError(f"Error: Unable to read the original image at '{original_image_path}'!")
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Load the boolean mask directly
    print(f"Loading boolean mask from {boolean_mask_path}...")
    boolean_mask_visual = cv2.imread(boolean_mask_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    if boolean_mask_visual is None:
        raise ValueError(f"Error: Unable to load boolean mask from '{boolean_mask_path}'!")

    # Ensure the loaded mask is truly boolean (0 or 1) and of correct type.
    # The `apply_boolean_mask_by_color` saves it as 0 or 255. Convert to 0 or 1.
    boolean_mask = (boolean_mask_visual == 0).astype(np.uint8)  # 0 (black) in saved image means False (0) for us.
    # 255 (white) in saved image means True (1) for us.
    # We want 'negative space' to be 0. So map original 0 to 0.
    # If you saved 0 as True, and 255 as False, adjust this line.
    # Assuming 0 is obstacle/negative and 255 is navigable/positive

    # Check if the dimensions match
    if original_image.shape[:2] != boolean_mask.shape:
        raise ValueError(
            f"Dimension mismatch: Original image {original_image.shape[:2]} vs Boolean mask {boolean_mask.shape}")

    # 1. Generate Superpixels using SLIC
    print(f"Generating {num_superpixels} superpixels...")
    segments = slic(original_image_rgb, n_segments=num_superpixels, compactness=compactness,
                    sigma=1, start_label=1,
                    min_size_factor=min_size_factor, enforce_connectivity=True)

    identified_superpixels = []

    # 2. Iterate through each superpixel and check its corresponding region in the boolean mask
    unique_labels = np.unique(segments)
    print(f"Checking {len(unique_labels)} superpixels for negative space...")

    required_negative_pixels_count = (desired_pixel_coverage_percent / 100.0)

    for label in unique_labels:
        superpixel_mask = (segments == label)
        boolean_region = boolean_mask[superpixel_mask]

        negative_pixel_count = np.sum(boolean_region == 0)  # Count pixels that are 0 (negative space)
        total_pixel_count = boolean_region.size

        if total_pixel_count == 0:
            continue

        percentage_negative = (negative_pixel_count / total_pixel_count) * 100

        if percentage_negative >= desired_pixel_coverage_percent:
            rows, cols = np.where(superpixel_mask)

            if rows.size == 0 or cols.size == 0:
                continue

            centroid_y = int(np.mean(rows))
            centroid_x = int(np.mean(cols))

            min_row, max_row = np.min(rows), np.max(rows)
            min_col, max_col = np.min(cols), np.max(cols)

            identified_superpixels.append({
                'label': label,
                'centroid': (centroid_y, centroid_x),
                'bbox': (min_row, min_col, max_row, max_col),
                'percentage_negative': percentage_negative
            })
            print(f"  Found negative superpixel: Label {label}, Centroid ({centroid_y},{centroid_x}), "
                  f"BBox {min_row},{min_col},{max_row},{max_col}, {percentage_negative:.2f}% negative.")

    if visualize:
        fig, ax = plt.subplots(1, 2, figsize=(18, 9))  # Increased figure size

        # Original image with superpixel boundaries
        ax[0].imshow(mark_boundaries(original_image_rgb, segments))
        ax[0].set_title(f"Superpixel Segmentation ({num_superpixels} segments)")
        ax[0].axis('off')

        # Identified negative superpixels highlighted on the original image with BBox and Centroid
        overlay_image = np.copy(original_image_rgb)

        # Create a blank mask for highlighting identified superpixels
        highlight_mask = np.zeros_like(segments, dtype=bool)

        for sp in identified_superpixels:
            label = sp['label']
            highlight_mask[segments == label] = True  # Mark pixels belonging to this superpixel

            # Draw bounding box and centroid on the original image for identified superpixels
            min_r, min_c, max_r, max_c = sp['bbox']
            cv2.rectangle(overlay_image, (min_c, min_r), (max_c, max_r), (0, 255, 0), 2)  # Green bbox
            cv2.circle(overlay_image, (sp['centroid'][1], sp['centroid'][0]), 5, (255, 0, 0), -1)  # Red centroid

        # Create a transparent yellow overlay for identified superpixels
        alpha = 0.4  # Transparency level
        yellow_color = [255, 255, 0]  # Yellow in RGB

        # Create an overlay layer
        overlay_layer = np.zeros_like(original_image_rgb, dtype=np.uint8)
        overlay_layer[highlight_mask] = yellow_color

        # Combine the original image with the overlay layer
        final_visualization_image = cv2.addWeighted(overlay_image, 1.0, overlay_layer, alpha, 0)

        ax[1].imshow(final_visualization_image)
        ax[1].set_title(f"Identified Negative Superpixels (Coverage >= {desired_pixel_coverage_percent:.0f}%)")
        ax[1].axis('off')

        plt.tight_layout()

        # Save the visualization image
        plt.savefig(output_superpixel_visualization_path, bbox_inches='tight', pad_inches=0)
        print(f"Superpixel visualization saved to {output_superpixel_visualization_path}")

        plt.show()  # Display the plot

    return identified_superpixels


# --- CORRECTED AND COMPLETE identify_negative_superpixel FUNCTION ---
def identify_negative_superpixel_v2(
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


def combine_boolean_masks_union(
        mask_paths: list[str],
        output_combined_mask_path: str
) -> np.ndarray:
    """
    Combines multiple boolean masks using a logical OR operation (union).
    If a pixel is marked as 'negative' (0) in *any* of the input masks,
    it will be marked as 'negative' (0) in the combined mask.

    Args:
        mask_paths (list[str]): A list of paths to the input boolean mask images (0s and 255s).
        output_combined_mask_path (str): Path to save the resulting combined boolean mask image.

    Returns:
        np.ndarray: The combined boolean mask as a NumPy array (0s and 1s).
    """
    if not mask_paths:
        raise ValueError("No mask paths provided for combination.")

    # Load the first mask to get dimensions
    check_file_exists(mask_paths[0])
    first_mask_raw = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)
    if first_mask_raw is None:
        raise ValueError(f"Error: Unable to load the first mask image at {mask_paths[0]}!")

    # Convert to boolean (0 for negative, 1 for positive).
    # Since our masks are 0 (negative) and 255 (positive), convert 255 to 1.
    combined_mask = (first_mask_raw == 255).astype(
        np.uint8)  # Initialize with the first mask (1s where 255, 0s where 0)

    # Process subsequent masks
    for i, path in enumerate(mask_paths[1:]):
        check_file_exists(path)
        current_mask_raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if current_mask_raw is None:
            print(f"Warning: Unable to load mask image at {path}. Skipping this mask.")
            continue

        if current_mask_raw.shape != first_mask_raw.shape:
            raise ValueError(f"Mask {path} has different dimensions ({current_mask_raw.shape}) "
                             f"than the first mask ({first_mask_raw.shape}). All masks must have the same size.")

        current_mask_bool = (current_mask_raw == 255).astype(np.uint8)  # Convert 255 to 1

        # Logical OR: if either mask has a '0' (negative), the result is '0' (negative)
        # This means: (NOT A) OR (NOT B) --> NOT (A AND B)
        # We want: If A is negative (0) OR B is negative (0) --> Combined is negative (0)
        # So, if any pixel is 0 (negative) in an input mask, we want it to be 0 in the output.
        # If it's 1 (positive) in both, we want it to be 1.
        combined_mask = np.logical_and(combined_mask, current_mask_bool).astype(np.uint8)

    # Invert the combined_mask to match the 0 (negative) and 1 (positive) convention
    # where 0 means "problematic" (our target for superpixel identification)
    # and 1 means "safe/navigable".
    # Since np.logical_and(True, True) is True (1), and we want 0 for negative,
    # we need to invert the boolean result if it was based on 1 for positive.
    # Let's clarify: if 255 means positive (safe) and 0 means negative (hazard).
    # We want the union of hazards.
    # combined_mask initial logic: 1s for safe, 0s for hazard.
    # So, if ANY pixel in ANY mask is 0 (hazard), we want the final pixel to be 0 (hazard).
    # If a pixel is 1 (safe) IN ALL masks, then the final pixel is 1 (safe).
    # This is equivalent to: combined_mask = mask1 AND mask2 AND mask3 ... (where 1 is safe, 0 is hazard)
    # The current `np.logical_and` achieves this:
    # (1 AND 1) = 1 (safe)
    # (1 AND 0) = 0 (hazard)
    # (0 AND 1) = 0 (hazard)
    # (0 AND 0) = 0 (hazard)
    # This logic correctly finds "safe only if ALL are safe".

    # The `identify_negative_superpixel` function looks for 0s as "negative".
    # Our `combined_mask` currently has 0s for negative, 1s for positive. This matches.

    visual_mask = (combined_mask * 255).astype(np.uint8)  # Convert 1s to 255 for visualization
    cv2.imwrite(output_combined_mask_path, visual_mask)
    print(f"Combined boolean mask (union of negative spaces) saved to {output_combined_mask_path}")

    return combined_mask  # Returns 0s and 1s

# src/superpixel_analysis.py
# ( ... existing imports and functions above ... )

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

# Main Function
def main():
    # Input and output paths
    input_image_path = "input_image.jpg"  # Replace with your image file
    output_dir = "output_results"
    os.makedirs(output_dir, exist_ok=True)

    # ... (input/output path definitions) ...

    # --- Create a dummy image for testing if it doesn't exist ---
    # ... (dummy image creation code) ...

    # --- Generate common superpixel segments ONCE for consistent analysis ---
    # This is crucial for comparing superpixels across different masks
    image_for_slic = cv2.imread(input_image_path)  # <--- Image loaded here
    if image_for_slic is None:
        raise ValueError(f"Error: Unable to load original image for SLIC at {input_image_path}")

    # Parameters for SLIC (can be tuned)
    num_superpixels = 500
    compactness = 20.0
    # The `slic` function from `skimage.segmentation` generates the superpixel segments
    segments = slic(image_for_slic, n_segments=num_superpixels, compactness=compactness, sigma=1,
                    enforce_connectivity=True, slic_zero=False)  # <--- 'segments' is created here!
    print(f"\nGenerated {segments.max() + 1} superpixel segments for the image.")

    # Lists to store the identified negative superpixel labels from each source
    all_negative_superpixel_label_lists = [] # <--- It starts here as an empty list

    # Output file paths
    output_image_path_depth = os.path.join(output_dir, "depth_heatmap.jpg")
    output_image_path_shapes = os.path.join(output_dir, "shapes_detected.jpg")
    output_image_path_texture = os.path.join(output_dir, "texture_analysis.jpg")
    output_image_path_navigable = os.path.join(output_dir, "navigable_heatmap.jpg")

    # Color overlay paths
    grayscale_output_path_red = os.path.join(output_dir, "grayscale_red.jpg")
    transparent_overlay_output_path_red = os.path.join(output_dir, "transparent_red_overlay.png")

    grayscale_output_path_blue = os.path.join(output_dir, "grayscale_blue.jpg")
    transparent_overlay_output_path_blue = os.path.join(output_dir, "transparent_blue_overlay.png")

    grayscale_output_path_yellow = os.path.join(output_dir, "grayscale_yellow.jpg")
    transparent_overlay_output_path_yellow = os.path.join(output_dir, "transparent_yellow_overlay.png")

    grayscale_output_path_orange = os.path.join(output_dir, "grayscale_orange.jpg")
    transparent_overlay_output_path_orange = os.path.join(output_dir, "transparent_orange_overlay.png")

    grayscale_output_path_green = os.path.join(output_dir, "grayscale_green.jpg")
    transparent_overlay_output_path_green = os.path.join(output_dir, "transparent_green_overlay.png")

    grayscale_output_path_violet = os.path.join(output_dir, "grayscale_violet.jpg")
    transparent_overlay_output_path_violet = os.path.join(output_dir, "transparent_violet_overlay.png")

    # Run all processing functions
    try:
        # Depth Estimation
        print("Running depth estimation heatmap...")
        depth_estimation_heatmap(input_image_path, output_image_path_depth)

        # Shape Detection
        print("Running shape detection...")
        detect_shapes(input_image_path, output_image_path_shapes)

        # Texture Analysis
        print("Running texture analysis...")
        analyze_texture(input_image_path, output_image_path_texture)

        # Navigable Heatmap Generation
        print("Running navigable heatmap generation...")
        generate_navigable_heatmap(input_image_path, output_image_path_navigable)

        # Color Overlays
        print("Running red color overlay...")
        extract_and_overlay_with_transparency(
            image_path=input_image_path,
            grayscale_path=grayscale_output_path_red,
            overlay_path=transparent_overlay_output_path_red,
            hue_min=0, hue_max=30, highlight_color=(255, 0, 0, 128)
        )

        print("Running blue color overlay...")
        extract_and_overlay_with_transparency(
            image_path=input_image_path,
            grayscale_path=grayscale_output_path_blue,
            overlay_path=transparent_overlay_output_path_blue,
            hue_min=200, hue_max=240, highlight_color=(0, 0, 255, 128)
        )

        print("Running yellow color overlay...")
        extract_and_overlay_with_transparency(
            image_path=input_image_path,
            grayscale_path=grayscale_output_path_yellow,
            overlay_path=transparent_overlay_output_path_yellow,
            hue_min=45, hue_max=75, highlight_color=(255, 255, 0, 128)
        )

        print("Running orange color overlay...")
        extract_and_overlay_with_transparency(
            image_path=input_image_path,
            grayscale_path=grayscale_output_path_orange,
            overlay_path=transparent_overlay_output_path_orange,
            hue_min=30, hue_max=60, highlight_color=(255, 165, 0, 128)
        )

        print("Running green color overlay...")
        extract_and_overlay_with_transparency(
            image_path=input_image_path,
            grayscale_path=grayscale_output_path_green,
            overlay_path=transparent_overlay_output_path_green,
            hue_min=90, hue_max=150, highlight_color=(0, 255, 0, 128)
        )

        print("Running violet color overlay...")
        extract_and_overlay_with_transparency(
            image_path=input_image_path,
            grayscale_path=grayscale_output_path_violet,
            overlay_path=transparent_overlay_output_path_violet,
            hue_min=270,
            hue_max=300, highlight_color=(128, 0, 128, 128)
        )

        red_grayscale_path = grayscale_output_path_red
        orange_grayscale_path = grayscale_output_path_orange
        yellow_grayscale_path = grayscale_output_path_yellow
        green_grayscale_path = grayscale_output_path_green
        blue_grayscale_path = grayscale_output_path_blue
        violet_grayscale_path = grayscale_output_path_violet
        output_temp_heatmap_path = os.path.join(output_dir, "color_temp_heatmap.jpg")

        generate_temperature_heatmap_v3(
            red_grayscale_path,
            orange_grayscale_path,
            yellow_grayscale_path,
            green_grayscale_path,
            blue_grayscale_path,
            violet_grayscale_path,
            output_temp_heatmap_path,
        )

        output_obstacle_heatmap_path = os.path.join(output_dir, "obstacle_heatmap.jpg")
        generate_smooth_obstacle_heatmap(output_image_path_shapes, output_obstacle_heatmap_path)

        #output_navigatable_v2_heatmap_path = os.path.join(output_dir, "navigatable_heatmap_v2.jpg")
        #generate_navigable_heatmap_v2(output_image_path_navigable, output_navigatable_v2_heatmap_path)

        output_image_path_depth_bool = os.path.join(output_dir, "depth_heatmap_bool.jpg")
        output_temp_heatmap_path_bool = os.path.join(output_dir, "color_temp_heatmap_bool.jpg")
        output_obstacle_heatmap_path_bool = os.path.join(output_dir, "obstacle_heatmap_bool.jpg")
        apply_boolean_mask_by_color(output_image_path_depth, output_image_path_depth_bool)
        apply_boolean_mask_by_color(output_temp_heatmap_path, output_temp_heatmap_path_bool)
        apply_boolean_mask_by_color(output_obstacle_heatmap_path, output_obstacle_heatmap_path_bool)

        # 3. Identify Negative Superpixels (now uses the pre-computed boolean mask)
        output_superpixel_visualization_path = os.path.join(output_dir, "depth_heatmap_superpixel.jpg")
        print("\n--- Identifying Negative Space Superpixels ---")
        identified_superpixels = identify_negative_superpixel_v2(
            original_image_path=input_image_path,
            boolean_mask_path=output_image_path_depth_bool, # Pass the path to the already saved boolean mask
            output_superpixel_visualization_path=output_superpixel_visualization_path, # Save the new visualization
            #num_superpixels=500,
            #compactness=20.0,
            desired_pixel_coverage_percent=95.0, # e.g., 95% of pixels must be 0 for this superpixel
            visualize=True,
            segments=segments
        )
        if identified_superpixels:
            print("\nFound the following negative space superpixels:")
            all_negative_superpixel_label_lists.append(identified_superpixels)
        else:
            print("\nNo superpixels found that are predominantly negative space.")

       # 6. Superpixel Analysis from Temperature Boolean Mask
        output_superpixel_visualization_path_temp = os.path.join(output_dir, "color_temp_heatmap_superpixel.jpg")
        print("\n--- Identifying Negative Space Superpixels (from Temperature Mask) ---")
        identified_superpixels_temp = identify_negative_superpixel_v2(
            original_image_path=input_image_path,
            boolean_mask_path=output_temp_heatmap_path_bool, # <--- Using the new boolean mask
            output_superpixel_visualization_path=output_superpixel_visualization_path_temp, # <--- New output path
            #num_superpixels=500,
            #compactness=20.0,
            desired_pixel_coverage_percent=95.0, # Can be adjusted
            visualize=True,
            segments=segments
        )
        if identified_superpixels_temp:
            print(f"\nFound {len(identified_superpixels_temp)} negative superpixels from Temperature Mask.")
            all_negative_superpixel_label_lists.append(identified_superpixels_temp)
        else:
            print("\nNo superpixels found that are predominantly negative space from Temperature Mask.")

        # 9. Superpixel Analysis from Obstacle Boolean Mask
        print("\n--- Identifying Negative Space Superpixels (from Obstacle Mask) ---")
        output_superpixel_visualization_path_obstacle = os.path.join(output_dir, "obstacle_heatmap_superpixel.jpg")
        identified_superpixels_obstacle = identify_negative_superpixel_v2(
            original_image_path=input_image_path,
            boolean_mask_path=output_obstacle_heatmap_path_bool,  # <--- Using the new boolean mask
            output_superpixel_visualization_path=output_superpixel_visualization_path_obstacle,
            # <--- New output path
            #num_superpixels=500,
            #compactness=20.0,
            desired_pixel_coverage_percent=95.0,  # Can be adjusted
            visualize=True,
            segments=segments
        )
        if identified_superpixels_obstacle:
            print(f"\nFound {len(identified_superpixels_obstacle)} negative superpixels from Obstacle Mask.")
            all_negative_superpixel_label_lists.append(identified_superpixels_obstacle)
        else:
            print("\nNo superpixels found that are predominantly negative space from Obstacle Mask.")

        # --- 4. Combined (Union) Negative Superpixel Analysis ---
        print("\n--- Combining Boolean Masks for Union of Negative Space ---")
        # List all the boolean mask paths you want to combine
        boolean_mask_paths_to_combine = [
            output_superpixel_visualization_path,
            output_superpixel_visualization_path_temp,
            output_superpixel_visualization_path_obstacle
        ]

        output_combined_boolean_mask_path = os.path.join(output_dir, "superpixel_fusion_map.jpg")
        #combine_boolean_masks_union(
        #    mask_paths=boolean_mask_paths_to_combine,
        #    output_combined_mask_path=output_combined_boolean_mask_path
        #)

        # --- NEW: Superpixel-level Overlap/Intersection Fusion ---
        print("\n--- Performing Superpixel-level Overlap/Intersection Fusion ---")
        # Define the minimum number of sources that must identify a superpixel as negative
        min_overlap_sources = 2  # e.g., at least 2 out  of 3 sources must agree

        fused_negative_superpixel_labels = find_overlapping_negative_superpixels(
            list_of_negative_superpixel_labels=all_negative_superpixel_label_lists,
            min_overlap_sources=min_overlap_sources
        )

        output_fused_superpixel_visualization_path = os.path.join(output_dir, "superpixel_fusion_map.jpg")
        if fused_negative_superpixel_labels:
            print(
                f"\nFound {len(fused_negative_superpixel_labels)} fused negative superpixels (overlap from >= {min_overlap_sources} sources).")
            # Visualize the fused superpixels
            visualize_specific_superpixels(
                image_path=input_image_path,
                output_path=output_fused_superpixel_visualization_path,
                segments=segments,  # Pass the original segments array
                superpixel_labels_to_highlight=fused_negative_superpixel_labels,
                highlight_color=(0, 0, 255),  # Red for danger
                #show_boundaries=True
            )
        else:
            print("\nNo superpixels found that meet the minimum overlap criteria for fusion.")

        # --- NEW: Create and save the fused boolean mask ---
        output_fused_boolean_mask_path = os.path.join(output_dir, "superpixel_fusion_map_bool.jpg")
        fused_boolean_array = create_fused_boolean_mask(
            segments,
            fused_negative_superpixel_labels,
            output_fused_boolean_mask_path
        )

        # --- NEW: Find the closest negative pixel using the spiral algorithm ---
        closest_pixel_coords = find_closest_negative_pixel_spiral(fused_boolean_array)

        if closest_pixel_coords:
            print(f"Closest negative space pixel found at (row, col): {closest_pixel_coords}")
            # You can visualize this point on the original image if you like
            # e.g., using cv2.circle on image_rgb or a copy of it
        else:
            print("No negative space pixels found in the fused boolean mask.")


    except FileNotFoundError as e:
        print(f"File not found error: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()