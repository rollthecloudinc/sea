import cv2
import numpy as np


def detect_lines_with_depth(image_path, output_path):
    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection (Canny)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Apply Hough Line Transform to detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

    # Draw the lines on the original image
    if lines is not None:
        # Normalize rho values for line thickness scaling
        rho_values = [rho for rho, theta in lines[:, 0]]
        min_rho = min(rho_values)
        max_rho = max(rho_values)

        for rho, theta in lines[:, 0]:
            # Convert polar coordinates to Cartesian
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # Scale line thickness based on rho value (depth)
            thickness = int(np.interp(rho, [min_rho, max_rho], [1, 10]))  # Scale thickness between 1 and 10

            # Draw the line on the image
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), thickness)

    # Save the output image
    cv2.imwrite(output_path, image)
    print(f"Lines detected with depth differentiation and saved to {output_path}")


if __name__ == "__main__":
    # Provide the input image path and output image path
    input_image_path = "input_image.jpg"  # Replace with your image file path
    output_image_path = "output_lines2.jpg"  # Replace with your desired output file path

    detect_lines_with_depth(input_image_path, output_image_path)