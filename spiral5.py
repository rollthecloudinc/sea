import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap


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


def visualize_heatmap(heatmap, filename='custom_spiral_heatmap.png'):
    # Create a custom colormap with the specified color sequence
    custom_colors = [
        (0.0, 'green'),  # Green at the lowest value
        (0.2, 'blue'),  # Blue
        (0.4, 'violet'),  # Violet
        (0.6, 'red'),  # Red
        (0.8, 'orange'),  # Orange
        (1.0, 'yellow')  # Yellow at the highest value
    ]
    cmap = LinearSegmentedColormap.from_list("CustomColormap", [color[1] for color in custom_colors])

    # Normalize heatmap values to [0, 1]
    norm = Normalize(vmin=0, vmax=1)

    # Create the figure and axes explicitly
    fig, ax = plt.subplots(figsize=(6, 6))

    # Display the heatmap
    im = ax.imshow(heatmap, cmap=cmap, norm=norm, interpolation='nearest')

    # Add the colorbar explicitly linked to the plot
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', label="Normalized Distance")

    # Add title and save the image
    ax.set_title("Spiral Heatmap with Custom Color Gradient")
    ax.axis("off")

    # Save the heatmap image to a file
    plt.savefig(filename, dpi=300)
    print(f"Heatmap saved as '{filename}'")

# Example Usage
size = 25  # Specify the matrix size
spiral_heatmap = create_spiral_heatmap(size)

# Visualize and save the heatmap with custom color mapping
visualize_heatmap(spiral_heatmap, filename='custom_spiral5.png')