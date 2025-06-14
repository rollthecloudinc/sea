import numpy as np
import matplotlib.pyplot as plt


def spiral_indices(array_shape):
    """
    Generate the indices of a 2D array in spiral order, starting from the center.

    Parameters:
        array_shape (tuple): Shape of the array as (rows, cols).

    Returns:
        list: List of (row, col) tuples in spiral order.
    """
    rows, cols = array_shape
    indices = []  # List to store the spiral order

    # Calculate the center of the array
    center_row, center_col = rows // 2, cols // 2

    # Initialize variables for traversal
    row, col = center_row, center_col  # Start at the center
    indices.append((row, col))  # Add the center pixel

    # Spiral outward
    steps = 1  # Number of steps to move in each direction
    direction = 0  # Direction: 0=right, 1=down, 2=left, 3=up
    while len(indices) < rows * cols:
        for _ in range(2):  # Repeat twice to complete a layer of the spiral
            for _ in range(steps):
                if len(indices) >= rows * cols:  # Stop if all pixels are processed
                    break
                # Update row and column based on direction
                if direction == 0:  # Move right
                    col += 1
                elif direction == 1:  # Move down
                    row += 1
                elif direction == 2:  # Move left
                    col -= 1
                elif direction == 3:  # Move up
                    row -= 1
                # Add the current index to the spiral order
                indices.append((row, col))
            direction = (direction + 1) % 4  # Change direction
        steps += 1  # Increase the number of steps for the next layer

    return indices


def visualize_processing_sequence(array_shape, filename='spiral_processing_sequence.png'):
    """
    Visualize and save the processing sequence of pixels in spiral order.

    Parameters:
        array_shape (tuple): Shape of the array as (rows, cols).
        filename (str): Name of the file to save the visualization.
    """
    rows, cols = array_shape

    # Create a matrix to store the processing order
    processing_order = np.zeros((rows, cols), dtype=int)

    # Get the spiral indices
    spiral_order = spiral_indices(array_shape)

    # Assign processing order values to the matrix
    for index, (row, col) in enumerate(spiral_order):
        processing_order[row, col] = index + 1  # Start counting from 1

    # Visualize the matrix as a heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(processing_order, cmap='viridis', interpolation='nearest')

    # Add a colorbar to indicate the processing order
    plt.colorbar(label="Processing Order")
    plt.title("Spiral Processing Sequence (Starting from Center)")
    plt.axis("off")

    # Save the heatmap to a file
    plt.savefig(filename, dpi=300)
    print(f"Visualization saved as '{filename}'")

# Define the array dimensions
array_shape = (9, 9)  # Change this to any shape

# Visualize and save the spiral processing order as a heatmap
visualize_processing_sequence(array_shape, filename='spiral_processing_sequence.png')