import numpy as np
import matplotlib.pyplot as plt

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

# Create a 9x9 heatmap
size = 9
spiral_heatmap = create_spiral_heatmap(size)

# Visualize the heatmap using matplotlib and save it as an image
plt.figure(figsize=(6, 6))
plt.imshow(spiral_heatmap, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label="Normalized Distance")
plt.title("Spiral Heatmap (Center to Outward)")
plt.axis("off")

# Save the image to a file
plt.savefig("spiral_heatmap_center.png", dpi=300)
print("Heatmap saved as 'spiral_heatmap_center.png'")