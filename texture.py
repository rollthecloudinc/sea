import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

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

if __name__ == "__main__":
    # Provide the input image path and output path
    input_image_path = "input_image.jpg"  # Replace with your image file path
    output_image_path = "output_texture_proximity.jpg"  # Replace with your desired output file path

    # Run the function
    analyze_texture(input_image_path, output_image_path)