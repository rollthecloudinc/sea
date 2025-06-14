import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import colormaps


def create_spiral_heatmap(size):
    # Create an empty array to hold the spiral heatmap values
    heatmap = np.zeros((size, size))

    # Define boundaries for the spiral traversal
    top, bottom = 0, size - 1
    left, right = 0, size - 1

    # Initialize value for the spiral traversal
    value = 0

    # Create the spiral order
    while top <= bottom and left <= right:
        # Traverse from left to right along the top row
        for col in range(left, right + 1):
            heatmap[top, col] = value
            value += 1
        top += 1

        # Traverse from top to bottom along the right column
        for row in range(top, bottom + 1):
            heatmap[row, right] = value
            value += 1
        right -= 1

        # Traverse from right to left along the bottom row (if not collapsed)
        if top <= bottom:
            for col in range(right, left - 1, -1):
                heatmap[bottom, col] = value
                value += 1
            bottom -= 1

        # Traverse from bottom to top along the left column (if not collapsed)
        if left <= right:
            for row in range(bottom, top - 1, -1):
                heatmap[row, left] = value
                value += 1
            left += 1

    # Reverse values so center pixel is lowest and outermost pixel is highest
    heatmap = size * size - heatmap

    # Normalize values to range [0, 1]
    heatmap = heatmap / np.max(heatmap)
    return heatmap


def visualize_heatmap(heatmap, colormap='coolwarm', filename='true_spiral_heatmap.png'):
    # Normalize heatmap values to [0, 1]
    norm = Normalize(vmin=0, vmax=1)

    # Get the colormap using the new recommended method
    cmap = colormaps.get_cmap(colormap)

    # Apply the colormap directly to the normalized heatmap data
    colored_heatmap = cmap(heatmap)  # Map normalized values to RGBA colors

    # Create the figure and axes explicitly
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(colored_heatmap, interpolation='nearest')

    # Add the colorbar explicitly linked to the plot
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', label="Normalized Distance")

    # Add title and save the image
    ax.set_title("True Spiral Heatmap")
    ax.axis("off")
    plt.savefig(filename, dpi=300)
    print(f"Heatmap saved as '{filename}'")


# Example Usage
size = 100  # Change this to any size for the matrix
spiral_heatmap = create_spiral_heatmap(size)

# Visualize and save the true heatmap with full color range
visualize_heatmap(spiral_heatmap, colormap='coolwarm', filename='true_spiral_heatmap.png')