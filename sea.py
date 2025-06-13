import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import timm
from skimage.feature import local_binary_pattern
import colorsys
from PIL import Image

def depth_estimation_heatmap(image_path, output_path):
    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return

    # Resize the image for compatibility with the MiDaS model
    input_height, input_width = 256, 256
    image_resized = cv2.resize(image, (input_width, input_height))

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # Normalize the image
    image_normalized = image_rgb / 255.0

    # Load MiDaS model from torch.hub
    print("Loading MiDaS model...")
    model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
    model.eval()

    # Ensure the model runs on CPU
    device = torch.device("cpu")  # Force model to run on CPU
    model.to(device)

    # Prepare the image for the model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((input_height, input_width)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    input_tensor = transform(image_rgb).unsqueeze(0).to(device)

    # Perform depth estimation
    print("Performing depth estimation...")
    with torch.no_grad():
        depth_map = model(input_tensor)
    depth_map = depth_map.squeeze().cpu().numpy()

    # Normalize the depth map for visualization
    print("Creating heat map...")
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply a heat map to the depth map
    heatmap = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

    # Resize the heat map back to the original image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Save the heat map image
    cv2.imwrite(output_path, heatmap_resized)
    print(f"Depth heat map saved to {output_path}")

    # Display the heat map
    plt.imshow(cv2.cvtColor(heatmap_resized, cv2.COLOR_BGR2RGB))
    plt.title("Depth Heat Map")
    plt.axis("off")
    plt.show()

def detect_shapes(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours to detect shapes
    for contour in contours:
        # Approximate the contour to reduce the number of points
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(approx)

        # Classify the shape based on the number of vertices
        num_vertices = len(approx)
        if num_vertices == 3:
            shape_name = "Triangle"
        elif num_vertices == 4:
            # Check if the shape is square or rectangle
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                shape_name = "Square"
            else:
                shape_name = "Rectangle"
        elif num_vertices == 5:
            shape_name = "Pentagon"
        elif num_vertices == 6:
            shape_name = "Hexagon"
        else:
            # Check for circles
            area = cv2.contourArea(contour)
            radius = w / 2
            if abs(1 - (area / (np.pi * radius ** 2))) < 0.2:
                shape_name = "Circle"
            else:
                shape_name = "Unknown"

        # Draw the contour and label the shape
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        cv2.putText(image, shape_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the output image
    cv2.imwrite(output_path, image)
    print(f"Shapes detected and saved to {output_path}")

def generate_navigable_heatmap(image_path, output_heatmap_path):
    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return

    # Resize the image for compatibility with the MiDaS model
    input_height, input_width = 256, 256
    image_resized = cv2.resize(image, (input_width, input_height))

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # Normalize the image
    image_normalized = image_rgb / 255.0

    # Load the MiDaS model
    print("Loading MiDaS model...")
    model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
    model.eval()

    # Ensure the model runs on CPU
    device = torch.device("cpu")
    model.to(device)

    # Prepare the image for the model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((input_height, input_width)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    input_tensor = transform(image_rgb).unsqueeze(0).to(device)

    print("Performing depth estimation...")
    with torch.no_grad():
        depth_map = model(input_tensor)
    depth_map = depth_map.squeeze().cpu().numpy()

    # Debugging depth map
    print("Depth Map Stats:")
    print("Min Depth Value:", np.min(depth_map))
    print("Max Depth Value:", np.max(depth_map))

    # Normalize the depth map for visualization
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Threshold depth map to identify navigable space
    depth_threshold = 50  # Minimum clearance distance
    print(f"Thresholding depth map with clearance distance: {depth_threshold}")
    navigable_space = np.where(depth_map > depth_threshold, 255, 0).astype(np.uint8)

    # Detect obstacles using edge detection
    print("Detecting obstacles...")
    edges = cv2.Canny(depth_map_normalized, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create obstacle mask
    obstacle_mask = np.zeros_like(depth_map_normalized)
    cv2.drawContours(obstacle_mask, contours, -1, 255, thickness=cv2.FILLED)

    # Debugging obstacle mask
    print("Obstacle Mask Stats:")
    print("Non-Zero Values in Obstacle Mask:", np.count_nonzero(obstacle_mask))

    # Create buffer zones
    print("Creating buffer zones around obstacles...")
    buffer_zone_mask = cv2.dilate(obstacle_mask, kernel=np.ones((15, 15), np.uint8), iterations=1)  # Adjust dilation size for larger buffer

    # Debugging buffer zones
    print("Buffer Zone Stats:")
    print("Non-Zero Values in Buffer Zone Mask:", np.count_nonzero(buffer_zone_mask))

    # Create heatmap
    print("Generating heatmap...")
    heatmap = np.zeros_like(depth_map_normalized, dtype=np.uint8)
    heatmap[navigable_space == 255] = 128  # Green for navigable space
    heatmap[buffer_zone_mask == 255] = 200  # Yellow for buffer zones
    heatmap[obstacle_mask == 255] = 255    # Red for obstacles

    # Overlay the heatmap on the original image
    print("Overlaying heatmap on the original image...")
    heatmap_bgr = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_resized = cv2.resize(heatmap_bgr, (image.shape[1], image.shape[0]))
    overlay = cv2.addWeighted(image, 0.6, heatmap_resized, 0.4, 0)

    # Save the heatmap and overlay
    cv2.imwrite(output_heatmap_path, overlay)
    print(f"Navigable space heatmap saved to {output_heatmap_path}")

    # Display the overlay
    cv2.imshow("Navigable Space Heatmap", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def analyze_texture(image_path, output_path, radius=3, num_points=24, threshold=0.2):
    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return

    # Convert to grayscale for texture analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute Local Binary Pattern (LBP)
    print("Performing texture analysis using Local Binary Patterns (LBP)...")
    lbp = local_binary_pattern(gray, num_points, radius, method="uniform")

    # Normalize LBP to range [0, 1]
    lbp_normalized = lbp / np.max(lbp)

    # Threshold for proximity detection (high texture density indicates proximity)
    print(f"Applying proximity threshold: {threshold}")
    proximity_mask = np.where(lbp_normalized > threshold, 255, 0).astype(np.uint8)

    # Find contours of "close" regions
    contours, _ = cv2.findContours(proximity_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    annotated_image = image.copy()
    for contour in contours:
        cv2.drawContours(annotated_image, [contour], -1, (0, 0, 255), 2)  # Red for close objects

    # Save the annotated image
    cv2.imwrite(output_path, annotated_image)
    print(f"Texture-based proximity detection saved to {output_path}")

    # Display the annotated image
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.title("Texture-Based Proximity Detection")
    plt.axis("off")
    plt.show()

def extract_and_overlay_with_transparency_red(
    image_path, grayscale_path, overlay_path, highlight_color=(255, 0, 0, 128)
):
    """
    Generate both a grayscale image based on red saturation and a transparent overlay image.

    Args:
        image_path (str): Path to the input image.
        grayscale_path (str): Path to save the grayscale output image.
        overlay_path (str): Path to save the transparent overlay image.
        highlight_color (tuple): RGBA color to highlight detected red areas (default is semi-transparent red).
    """
    # Load the image
    image = Image.open(image_path).convert('RGBA')
    img_array = np.array(image, dtype=np.uint8)

    # Extract RGB channels and normalize to [0, 1]
    img_rgb = img_array[..., :3] / 255.0
    R, G, B = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]

    # Convert RGB to HSV
    H, S, V = np.vectorize(colorsys.rgb_to_hsv)(R, G, B)

    # Create a mask for pixels in the red hue range (two ranges: 0°–30° and 330°–360°)
    red_mask = ((H >= 0.0) & (H <= 30 / 360)) | ((H >= 330 / 360) & (H <= 1.0))

    # Generate the grayscale image based on saturation
    grayscale = np.where(red_mask, (S * 255).astype(np.uint8), 255)  # Non-red pixels are white
    grayscale_image = Image.fromarray(grayscale, mode='L')
    grayscale_image.save(grayscale_path)
    print(f"Grayscale image saved to {grayscale_path}")

    # Generate the transparent overlay image
    overlay_array = np.copy(img_array)
    overlay_array[red_mask] = highlight_color  # Apply highlight color to detected red pixels
    overlay_image = Image.fromarray(overlay_array, mode='RGBA')
    overlay_image.save(overlay_path)
    print(f"Transparent overlay image saved to {overlay_path}")

def extract_and_overlay_with_transparency_blue(
    image_path, grayscale_path, overlay_path, hue_min=200, hue_max=240, highlight_color=(0, 0, 255, 128)
):
    """
    Generate both a grayscale image based on blue saturation and a transparent overlay image.

    Args:
        image_path (str): Path to the input image.
        grayscale_path (str): Path to save the grayscale output image.
        overlay_path (str): Path to save the transparent overlay image.
        hue_min (int): Minimum hue value for blue detection (default is 200).
        hue_max (int): Maximum hue value for blue detection (default is 240).
        highlight_color (tuple): RGBA color to highlight detected blue areas (default is semi-transparent blue).
    """
    # Load the image
    image = Image.open(image_path).convert('RGBA')
    img_array = np.array(image, dtype=np.uint8)

    # Extract RGB channels and normalize to [0, 1]
    img_rgb = img_array[..., :3] / 255.0
    R, G, B = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]

    # Convert RGB to HSV
    H, S, V = np.vectorize(colorsys.rgb_to_hsv)(R, G, B)

    # Create a mask for pixels in the blue hue range
    blue_mask = (H >= hue_min / 360) & (H <= hue_max / 360)

    # Generate the grayscale image based on saturation
    grayscale = np.where(blue_mask, (S * 255).astype(np.uint8), 255)  # Non-blue pixels are white
    grayscale_image = Image.fromarray(grayscale, mode='L')
    grayscale_image.save(grayscale_path)
    print(f"Grayscale image saved to {grayscale_path}")

    # Generate the transparent overlay image
    overlay_array = np.copy(img_array)
    overlay_array[blue_mask] = highlight_color  # Apply highlight color to detected blue pixels
    overlay_image = Image.fromarray(overlay_array, mode='RGBA')
    overlay_image.save(overlay_path)
    print(f"Transparent overlay image saved to {overlay_path}")

def extract_and_overlay_with_transparency_yellow(
    image_path, grayscale_path, overlay_path, hue_min=45, hue_max=75, highlight_color=(255, 255, 0, 128)
):
    """
    Generate both a grayscale image based on yellow saturation and a transparent overlay image.

    Args:
        image_path (str): Path to the input image.
        grayscale_path (str): Path to save the grayscale output image.
        overlay_path (str): Path to save the transparent overlay image.
        hue_min (int): Minimum hue value for yellow detection (default is 45).
        hue_max (int): Maximum hue value for yellow detection (default is 75).
        highlight_color (tuple): RGBA color to highlight detected yellow areas (default is semi-transparent yellow).
    """
    # Load the image
    image = Image.open(image_path).convert('RGBA')
    img_array = np.array(image, dtype=np.uint8)

    # Extract RGB channels and normalize to [0, 1]
    img_rgb = img_array[..., :3] / 255.0
    R, G, B = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]

    # Convert RGB to HSV
    H, S, V = np.vectorize(colorsys.rgb_to_hsv)(R, G, B)

    # Create a mask for pixels in the yellow hue range
    yellow_mask = (H >= hue_min / 360) & (H <= hue_max / 360)

    # Generate the grayscale image based on saturation
    grayscale = np.where(yellow_mask, (S * 255).astype(np.uint8), 255)  # Non-yellow pixels are white
    grayscale_image = Image.fromarray(grayscale, mode='L')
    grayscale_image.save(grayscale_path)
    print(f"Grayscale image saved to {grayscale_path}")

    # Generate the transparent overlay image
    overlay_array = np.copy(img_array)
    overlay_array[yellow_mask] = highlight_color  # Apply highlight color to detected yellow pixels
    overlay_image = Image.fromarray(overlay_array, mode='RGBA')
    overlay_image.save(overlay_path)
    print(f"Transparent overlay image saved to {overlay_path}")

def extract_and_overlay_with_transparency_orange(
    image_path, grayscale_path, overlay_path, hue_min=30, hue_max=60, highlight_color=(255, 165, 0, 128)
):
    """
    Generate both a grayscale image based on orange saturation and a transparent overlay image.

    Args:
        image_path (str): Path to the input image.
        grayscale_path (str): Path to save the grayscale output image.
        overlay_path (str): Path to save the transparent overlay image.
        hue_min (int): Minimum hue value for orange detection (default is 30).
        hue_max (int): Maximum hue value for orange detection (default is 60).
        highlight_color (tuple): RGBA color to highlight detected orange areas (default is semi-transparent orange).
    """
    # Load the image
    image = Image.open(image_path).convert('RGBA')
    img_array = np.array(image, dtype=np.uint8)

    # Extract RGB channels and normalize to [0, 1]
    img_rgb = img_array[..., :3] / 255.0
    R, G, B = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]

    # Convert RGB to HSV
    H, S, V = np.vectorize(colorsys.rgb_to_hsv)(R, G, B)

    # Create a mask for pixels in the orange hue range
    orange_mask = (H >= hue_min / 360) & (H <= hue_max / 360)

    # Generate the grayscale image based on saturation
    grayscale = np.where(orange_mask, (S * 255).astype(np.uint8), 255)  # Non-orange pixels are white
    grayscale_image = Image.fromarray(grayscale, mode='L')
    grayscale_image.save(grayscale_path)
    print(f"Grayscale image saved to {grayscale_path}")

    # Generate the transparent overlay image
    overlay_array = np.copy(img_array)
    overlay_array[orange_mask] = highlight_color  # Apply highlight color to detected orange pixels
    overlay_image = Image.fromarray(overlay_array, mode='RGBA')
    overlay_image.save(overlay_path)
    print(f"Transparent overlay image saved to {overlay_path}")

def extract_and_overlay_with_transparency_green(
        image_path, grayscale_path, overlay_path, hue_min=90, hue_max=150, highlight_color=(0, 255, 0, 128)
):
    """
    Generate both a grayscale image based on green saturation and a transparent overlay image.

    Args:
        image_path (str): Path to the input image.
        grayscale_path (str): Path to save the grayscale output image.
        overlay_path (str): Path to save the transparent overlay image.
        hue_min (int): Minimum hue value for green detection (default is 90).
        hue_max (int): Maximum hue value for green detection (default is 150).
        highlight_color (tuple): RGBA color to highlight detected green areas (default is semi-transparent green).
    """
    # Load the image
    image = Image.open(image_path).convert('RGBA')
    img_array = np.array(image, dtype=np.uint8)

    # Extract RGB channels and normalize to [0, 1]
    img_rgb = img_array[..., :3] / 255.0
    R, G, B = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]

    # Convert RGB to HSV
    H, S, V = np.vectorize(colorsys.rgb_to_hsv)(R, G, B)

    # Create a mask for pixels in the green hue range
    green_mask = (H >= hue_min / 360) & (H <= hue_max / 360)

    # Generate the grayscale image based on saturation
    grayscale = np.where(green_mask, (S * 255).astype(np.uint8), 255)  # Non-green pixels are white
    grayscale_image = Image.fromarray(grayscale, mode='L')
    grayscale_image.save(grayscale_path)
    print(f"Grayscale image saved to {grayscale_path}")

    # Generate the transparent overlay image
    overlay_array = np.copy(img_array)
    overlay_array[green_mask] = highlight_color  # Apply highlight color to detected green pixels
    overlay_image = Image.fromarray(overlay_array, mode='RGBA')
    overlay_image.save(overlay_path)
    print(f"Transparent overlay image saved to {overlay_path}")

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

if __name__ == "__main__":
    # Provide the input image path and output image path
    input_image_path = "input_image.jpg"  # Replace with your image file path

    output_image_path_depth = "sea_output_depth.jpg"  # Replace with your desired output file path
    output_image_path_shapes = "sea_output_shapes.jpg"  # Replace with your desired output file path
    output_image_path_edges = "sea_output_edges.jpg"  # Replace with your desired output file path
    output_image_path_texture = "sea_output_texture.jpg"  # Replace with your desired output file path

    # Run the function
    depth_estimation_heatmap(input_image_path, output_image_path_depth)
    detect_shapes(input_image_path, output_image_path_shapes)
    analyze_texture(input_image_path, output_image_path_texture)

    grayscale_output_path_red = "sea_grayscale_red_output.jpg"  # Path for grayscale output
    transparent_overlay_output_path_red = "sea_transparent_red_overlay_output.png"  # Path for transparent overlay output

    # Call the function
    extract_and_overlay_with_transparency_red(
        image_path=input_image_path,
        grayscale_path=grayscale_output_path_red,
        overlay_path=transparent_overlay_output_path_red,
        highlight_color=(255, 0, 0, 128)  # Semi-transparent red highlight
    )

    grayscale_output_path_blue = "sea_grayscale_blue_output.jpg"  # Path for grayscale output
    transparent_overlay_output_path_blue = "sea_transparent_blue_overlay_output.png"  # Path for transparent overlay output

    # Call the function with adjusted parameters for blue detection
    extract_and_overlay_with_transparency_blue(
        image_path=input_image_path,
        grayscale_path=grayscale_output_path_blue,
        overlay_path=transparent_overlay_output_path_blue,
        hue_min=200,  # Minimum hue value for blue detection (200°)
        hue_max=240,  # Maximum hue value for blue detection (240°)
        highlight_color=(0, 0, 255, 128)  # Semi-transparent blue highlight
    )

    grayscale_output_path_yellow = "sea_grayscale_yellow_output.jpg"  # Path for grayscale output
    transparent_overlay_output_path_yellow = "sea_transparent_yellow_overlay_output.png"  # Path for transparent overlay output

    # Call the function
    extract_and_overlay_with_transparency_yellow(
        image_path=input_image_path,
        grayscale_path=grayscale_output_path_yellow,
        overlay_path=transparent_overlay_output_path_yellow,
        hue_min=45,  # Minimum hue value for yellow detection (45°)
        hue_max=75,  # Maximum hue value for yellow detection (75°)
        highlight_color=(255, 255, 0, 128)  # Semi-transparent yellow highlight
    )

    grayscale_output_path_orange = "sea_grayscale_orange_output.jpg"  # Path for grayscale output
    transparent_overlay_output_path_orange = "sea_transparent_orange_overlay_output.png"  # Path for transparent overlay output

    # Call the function
    extract_and_overlay_with_transparency_orange(
        image_path=input_image_path,
        grayscale_path=grayscale_output_path_orange,
        overlay_path=transparent_overlay_output_path_orange,
        hue_min=30,  # Minimum hue value for orange detection (30°)
        hue_max=60,  # Maximum hue value for orange detection (60°)
        highlight_color=(255, 165, 0, 128)  # Semi-transparent orange highlight
    )

    grayscale_output_path_green = "sea_grayscale_green_output.jpg"  # Path for grayscale output
    transparent_overlay_output_path_green = "sea_transparent_green_overlay_output.png"  # Path for transparent overlay output

    # Call the function
    extract_and_overlay_with_transparency_green(
        image_path=input_image_path,
        grayscale_path=grayscale_output_path_green,
        overlay_path=transparent_overlay_output_path_green,
        hue_min=90,  # Minimum hue value for green detection (90°)
        hue_max=150,  # Maximum hue value for green detection (150°)
        highlight_color=(0, 255, 0, 128)  # Semi-transparent green highlight
    )

    grayscale_output_path_violet = "sea_grayscale_violet_output.jpg"  # Path for grayscale output
    transparent_overlay_output_path_violet = "sea_transparent_violet_overlay_output.png"  # Path for transparent overlay output

    # Call the function
    extract_and_overlay_with_transparency_violet(
        image_path=input_image_path,
        grayscale_path=grayscale_output_path_violet,
        overlay_path=transparent_overlay_output_path_violet,
        hue_min=270,  # Minimum hue value for violet detection (270°)
        hue_max=300,  # Maximum hue value for violet detection (300°)
        highlight_color=(128, 0, 128, 128)  # Semi-transparent violet highlight
    )

    # @todo: This is erroring
    # cv2.error: OpenCV(4.11.0) /io/opencv/modules/highgui/src/window.cpp:1301: error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvShowImage'
    generate_navigable_heatmap(input_image_path, output_image_path_edges)