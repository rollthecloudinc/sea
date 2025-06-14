import numpy as np
import matplotlib.pyplot as plt

def create_spiral_heatmap(size):
    # Create an empty array to hold the spiral heatmap values
    heatmap = np.zeros((size, size))

    # Define boundaries for the spiral traversal
    top, bottom = 0, size - 1
    left, right = 0, size - 1

    # Start at the center pixel (for odd-sized grids)
    center_x, center_y = size // 2, size // 2

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

    # Normalize values to range [0, 1]
    heatmap = heatmap / np.max(heatmap)
    return heatmap

# Create a 9x9 heatmap
size = 20
spiral_heatmap = create_spiral_heatmap(size)

# Save the heatmap as an image file
plt.figure(figsize=(6, 6))
plt.imshow(spiral_heatmap, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label="Normalized Distance")
plt.title("Spiral Heatmap")
plt.axis("off")

# Save the image to a file
plt.savefig("spiral_heatmap.png", dpi=300)
print("Heatmap saved as 'spiral_heatmap.png'")