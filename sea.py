import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import timm

def read_image(image_path):
    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return

    return image

def create_depth_map(image):
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
    return depth_map

def create_depth_heatmap(image, depth_map):

    # Normalize the depth map for visualization
    print("Creating heat map...")
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply a heat map to the depth map
    heatmap = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

    # Resize the heat map back to the original image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    return heatmap_resized

def visualize_depth_heatmap(heatmap_resized, output_path):

    # Save the heat map image
    cv2.imwrite(output_path, heatmap_resized)
    print(f"Depth heat map saved to {output_path}")

    # Display the heat map
    plt.imshow(cv2.cvtColor(heatmap_resized, cv2.COLOR_BGR2RGB))
    plt.title("Depth Heat Map")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # Provide the input image path and output image path
    input_image_path = "input_image.jpg"  # Replace with your image file path
    output_image_path = "sea_heatmap.jpg"  # Replace with your desired output file path

    # Run the function pipeline
    image = read_image(input_image_path)

    # Depth Pipeline
    depth_map = create_depth_map(image)
    depth_heatmap = create_depth_heatmap(depth_map, image)
    visualize_depth_heatmap(depth_heatmap, output_image_path)