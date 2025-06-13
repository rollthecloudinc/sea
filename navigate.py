import cv2
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt


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

    # Normalize the depth map for visualization
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

    # Create heatmap
    heatmap = np.zeros_like(depth_map_normalized, dtype=np.uint8)
    heatmap[navigable_space == 255] = 128  # Green for navigable space
    heatmap[obstacle_mask == 255] = 255  # Red for obstacles

    # Overlay the heatmap on the original image
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
    # Provide the input image path and output heatmap path
    input_image_path = "input_image.jpg"  # Replace with your image file path
    output_heatmap_path = "navigable_space_heatmap.jpg"  # Replace with your desired heatmap output file path

    # Run the function
    generate_navigable_heatmap(input_image_path, output_heatmap_path)
