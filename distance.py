import cv2
import numpy as np
import torch
from torchvision import transforms


def depth(image_path):
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

    return depth_map


def distance(depth_map, threshold):
    """
    Converts a depth map into a boolean array, indicating whether each pixel is near or far.

    Args:
        depth_map (numpy.ndarray): The input depth map as a 2D numpy array.
        threshold (float): The configurable threshold value to determine near or far.

    Returns:
        numpy.ndarray: A boolean array where True indicates the pixel is "near"
                       and False indicates the pixel is "far".
    """
    # Ensure the depth map is a numpy array
    if not isinstance(depth_map, np.ndarray):
        raise ValueError("depth_map must be a numpy array.")

    # Create a boolean array based on the threshold
    near_far_map = depth_map <= threshold

    return near_far_map


def visualize_near_far(near_far_map, output_path):
    """
    Visualizes the near/far boolean array and saves the visualization as an image.

    Args:
        near_far_map (numpy.ndarray): The boolean array indicating near/far pixels.
        output_path (str): Path to save the output visualization image.
    """
    # Convert the boolean array to a format suitable for visualization (0 for False, 255 for True)
    visualization = (near_far_map * 255).astype(np.uint8)

    # Ensure the proportions match the original image
    visualization_resized = cv2.resize(visualization, (near_far_map.shape[1], near_far_map.shape[0]))

    # Save the visualization image
    cv2.imwrite(output_path, visualization_resized)
    print(f"Near/Far visualization saved to {output_path}")


if __name__ == "__main__":
    # Input image path
    input_image_path = "input_image.jpg"  # Replace with your image file path

    # Output paths
    output_depth_path = "depth_map.jpg"  # Optional: Save the depth map for debugging
    output_visualization_path = "near_far_visualization.jpg"  # Replace with your desired output

    try:
        # Step 1: Generate the depth map using the `depth` function
        depth_map = depth(input_image_path)
        print("Depth map generated successfully.")

        # Optional: Save the depth map as a grayscale image for debugging
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(output_depth_path, depth_map_normalized)
        print(f"Depth map saved to {output_depth_path}")
    except Exception as e:
        print(f"Error generating depth map: {e}")
        exit()

    try:
        # Step 2: Transform the depth map into a near/far boolean array using the `distance` function
        threshold = 0.9  # Replace with your desired threshold value
        near_far_map = distance(depth_map, threshold)
        print("Near/Far map generated successfully.")
    except Exception as e:
        print(f"Error generating near/far map: {e}")
        exit()

    try:
        # Step 3: Visualize the near/far map using the `visualize_near_far` function
        visualize_near_far(near_far_map, output_visualization_path)
    except Exception as e:
        print(f"Error visualizing near/far map: {e}")
        exit()
