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

# Helper function to check if a file exists
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

# Depth Estimation Heatmap
def depth_estimation_heatmap(image_path, output_path):
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
    heatmap = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    cv2.imwrite(output_path, heatmap_resized)

    plt.imshow(cv2.cvtColor(heatmap_resized, cv2.COLOR_BGR2RGB))
    plt.title("Depth Heat Map")
    plt.axis("off")
    plt.show()

# Shape Detection
def detect_shapes(image_path, output_path):
    check_file_exists(image_path)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Unable to read the image!")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)

        num_vertices = len(approx)
        if num_vertices == 3:
            shape_name = "Triangle"
        elif num_vertices == 4:
            aspect_ratio = float(w) / h
            shape_name = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
        elif num_vertices == 5:
            shape_name = "Pentagon"
        elif num_vertices == 6:
            shape_name = "Hexagon"
        else:
            area = cv2.contourArea(contour)
            radius = w / 2
            if abs(1 - (area / (np.pi * radius ** 2))) < 0.2:
                shape_name = "Circle"
            else:
                shape_name = "Unknown"

        # Draw the contour and label the shape
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        #cv2.putText(image, shape_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imwrite(output_path, image)
    print(f"Shapes detected and saved to {output_path}")

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Shape Detection")
    plt.axis("off")
    plt.show()

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

# Main Function
def main():
    # Input and output paths
    input_image_path = "input_image.jpg"  # Replace with your image file
    output_dir = "output_results"
    os.makedirs(output_dir, exist_ok=True)

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
        identified_superpixels = identify_negative_superpixel(
            original_image_path=input_image_path,
            boolean_mask_path=output_image_path_depth_bool, # Pass the path to the already saved boolean mask
            output_superpixel_visualization_path=output_superpixel_visualization_path, # Save the new visualization
            num_superpixels=500,
            compactness=20.0,
            desired_pixel_coverage_percent=95.0, # e.g., 95% of pixels must be 0 for this superpixel
            visualize=True
        )
        if identified_superpixels:
            print("\nFound the following negative space superpixels:")
            for sp in identified_superpixels:
                print(f"  Label: {sp['label']}, Centroid: {sp['centroid']}, BBox: {sp['bbox']}, Neg%:{sp['percentage_negative']:.2f}")
        else:
            print("\nNo superpixels found that are predominantly negative space.")

    except FileNotFoundError as e:
        print(f"File not found error: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()