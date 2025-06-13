import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

def detect_shapes_with_depth(image_path, output_depth_path, output_shapes_path):
    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return

    # Resize the image for compatibility with MiDaS model
    input_height, input_width = 256, 256
    image_resized = cv2.resize(image, (input_width, input_height))

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # Normalize the image
    image_normalized = image_rgb / 255.0

    # Load MiDaS model for depth estimation
    print("Loading MiDaS model...")
    model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
    model.eval()

    # Ensure the model runs on CPU
    device = torch.device("cpu")  # Force CPU usage
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
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Save depth heat map
    heatmap = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    cv2.imwrite(output_depth_path, heatmap_resized)
    print(f"Depth heat map saved to {output_depth_path}")

    # Detect edges and contours for shape detection
    print("Detecting shapes...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours to detect shapes and calculate relative size
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
            # Check if the shape is a square or rectangle
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
            shape_name = "Unknown"

        # Calculate the average depth of the shape (using the depth map)
        mask = np.zeros(depth_map_normalized.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)  # Fill the contour area
        average_depth = cv2.mean(depth_map, mask=mask)[0]  # Average depth within the shape

        # Draw the contour and annotate the shape with depth information
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        cv2.putText(image, f"{shape_name}, Depth: {int(average_depth)}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the image with shape annotations
    cv2.imwrite(output_shapes_path, image)
    print(f"Shapes with depth annotations saved to {output_shapes_path}")

    # Display the final image with shapes and depth annotations
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Shapes with Depth Information")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # Provide the input image path and output paths
    input_image_path = "input_image.jpg"  # Replace with your image file path
    output_depth_path = "output_heatmap.jpg"  # Replace with your desired depth heatmap file path
    output_shapes_path = "output_shapes_with_depth.jpg"  # Replace with your desired shape annotation file path

    # Run the function
    detect_shapes_with_depth(input_image_path, output_depth_path, output_shapes_path)