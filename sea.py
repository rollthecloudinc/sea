import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import timm

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

if __name__ == "__main__":
    # Provide the input image path and output image path
    input_image_path = "input_image.jpg"  # Replace with your image file path

    output_image_path_depth = "sea_output_depth.jpg"  # Replace with your desired output file path
    output_image_path_shapes = "sea_output_shapes.jpg"  # Replace with your desired output file path
    output_image_path_edges = "sea_output_edges.jpg"  # Replace with your desired output file path

    # Run the function
    depth_estimation_heatmap(input_image_path, output_image_path_depth)
    detect_shapes(input_image_path, output_image_path_shapes)
    generate_navigable_heatmap(input_image_path, output_image_path_edges)