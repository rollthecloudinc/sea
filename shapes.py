import cv2
import numpy as np


def detect_shapes(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours to detect shapes
    for contour in contours:
        # Approximate the contour to reduce the number of points
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(approx)

        # Classify the shape based on the number of vertices
        num_vertices = len(approx)
        if num_vertices == 3:
            shape_name = "Triangle"
        elif num_vertices == 4:
            # Check if the shape is square or rectangle
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                shape_name = "Square"
            else:
                shape_name = "Rectangle"
        elif num_vertices == 5:
            shape_name = "Pentagon"
        elif num_vertices == 6:
            shape_name = "Hexagon"
        else:
            # Check for circles
            area = cv2.contourArea(contour)
            radius = w / 2
            if abs(1 - (area / (np.pi * radius ** 2))) < 0.2:
                shape_name = "Circle"
            else:
                shape_name = "Unknown"

        # Draw the contour and label the shape
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        cv2.putText(image, shape_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the output image
    cv2.imwrite(output_path, image)
    print(f"Shapes detected and saved to {output_path}")


if __name__ == "__main__":
    # Provide the input image path and output image path
    input_image_path = "input_image.jpg"  # Replace with your image file path
    output_image_path = "output_shapes.jpg"  # Replace with your desired output file path

    detect_shapes(input_image_path, output_image_path)