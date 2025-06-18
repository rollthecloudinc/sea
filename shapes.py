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