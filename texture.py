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