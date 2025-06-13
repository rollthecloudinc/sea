import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from skimage.feature import local_binary_pattern
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
        cv2.putText(image, shape_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

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

    except FileNotFoundError as e:
        print(f"File not found error: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()