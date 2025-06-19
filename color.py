import os
import cv2  # Used in generate_temperature_heatmap, keeping for consistency with your code
import numpy as np
# import torch # Not used in this specific task, but keeping as it was in your original snippet
# from torchvision import transforms # Not used in this specific task
# from skimage.feature import local_binary_pattern # Not used in this specific task
# from skimage.segmentation import slic # Not used in this specific task
# from skimage.segmentation import mark_boundaries # Not used in this specific task
import colorsys  # As per your request, for HSV conversion
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Mock check_file_exists if it's from a separate util.py ---
# If you have a real util.py, ensure it's in your Python path or remove this mock
try:
    from util import check_file_exists
except ImportError:
    print("Warning: 'util.py' not found. Using a mock 'check_file_exists' function.")


    def check_file_exists(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error: File not found at '{path}'")


# --- End of mock ---


# Color-Based Overlay Functions (from your provided snippet)
def extract_and_overlay_with_transparency(
        image_path, grayscale_path, overlay_path, hue_min, hue_max, highlight_color
):
    """
    Generic function to handle transparency overlays for specific color ranges.
    """
    check_file_exists(image_path)
    image = Image.open(image_path).convert('RGBA')
    img_array = np.array(image, dtype=np.uint8)

    # Extract RGB and normalize to [0, 1]
    img_rgb = img_array[..., :3] / 255.0
    R, G, B = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]

    # Convert RGB to HSV using colorsys.rgb_to_hsv (returns H, S, V in [0, 1])
    # Apply vectorize for element-wise operation on arrays
    H, S, V = np.vectorize(colorsys.rgb_to_hsv)(R, G, B)

    # Create a mask for pixels in the specified hue range
    # Note: hue_min and hue_max are expected in degrees (0-360) here,
    # so we divide by 360 for colorsys's [0, 1] range.
    color_mask = (H >= hue_min / 360) & (H <= hue_max / 360)

    # Generate the grayscale image based on saturation (as per your initial concept)
    grayscale = np.where(color_mask, (S * 255).astype(np.uint8), 255)  # 255 for non-masked areas
    grayscale_image = Image.fromarray(grayscale, mode='L')
    grayscale_image.save(grayscale_path)
    print(f"Grayscale image saved to {grayscale_path}")

    # The transparent overlay part was commented out in your snippet, keeping it that way
    # overlay_array = np.copy(img_array)
    # overlay_array[color_mask] = highlight_color
    # overlay_image = Image.fromarray(overlay_array, mode='RGBA')
    # overlay_image.save(overlay_path)
    # print(f"Transparent overlay image saved to {overlay_path}")


def generate_temperature_heatmap(
        red_grayscale_path,
        orange_grayscale_path,
        yellow_grayscale_path,
        green_grayscale_path,
        blue_grayscale_path,
        violet_grayscale_path,
        output_heatmap_path,
):
    """
    Generate a visual heatmap that accounts for the intensity (saturation) of colors,
    transitioning smoothly from warm (red) to cool (blue) with less saturated regions appearing lighter.
    (This function uses OpenCV, ensure it's installed: pip install opencv-python)
    """
    # Load all grayscale images as numpy arrays
    red = cv2.imread(red_grayscale_path, cv2.IMREAD_GRAYSCALE)
    orange = cv2.imread(orange_grayscale_path, cv2.IMREAD_GRAYSCALE)
    yellow = cv2.imread(yellow_grayscale_path, cv2.IMREAD_GRAYSCALE)
    green = cv2.imread(green_grayscale_path, cv2.IMREAD_GRAYSCALE)
    blue = cv2.imread(blue_grayscale_path, cv2.IMREAD_GRAYSCALE)
    violet = cv2.imread(violet_grayscale_path, cv2.IMREAD_GRAYSCALE)

    # Verify that all images are loaded
    if red is None or orange is None or yellow is None or green is None or blue is None or violet is None:
        raise ValueError("Error: One or more grayscale images could not be loaded!")

    # Normalize grayscale values to the range [0, 1]
    red = red / 255.0
    orange = orange / 255.0
    yellow = yellow / 255.0
    green = green / 255.0
    blue = blue / 255.0
    violet = violet / 255.0

    # Combine warm and cool colors, scaling by their intensities
    warm_colors = (red + orange + yellow) / 3  # Average warm saturation
    cool_colors = (green + blue + violet) / 3  # Average cool saturation

    # Create the temperature map based on warm and cool contributions
    # Areas with less saturation will naturally have intermediate/lighter values
    temperature_map = warm_colors - cool_colors  # Positive = warm, Negative = cool

    # Normalize the temperature map to range [0, 255] for visualization
    temperature_map_normalized = cv2.normalize(temperature_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply a colormap for visualization
    # COLORMAP_JET transitions from blue (cool) to red (warm) smoothly
    temperature_heatmap = cv2.applyColorMap(temperature_map_normalized, cv2.COLORMAP_JET)

    # Save the resulting heatmap
    cv2.imwrite(output_heatmap_path, temperature_heatmap)
    print(f"Temperature heatmap saved to {output_heatmap_path}")

    # Display the heatmap using matplotlib for better visualization
    plt.imshow(cv2.cvtColor(temperature_heatmap, cv2.COLOR_BGR2RGB))
    plt.title("Temperature Heatmap (Warm to Cool with Saturation Intensity)")
    plt.axis("off")
    plt.show()

    # Return the temperature map as a numpy array for further analysis if needed
    return temperature_map_normalized


# --- NEW FUNCTION: Create Warm and Cool Plates based on S*V ---
def create_warm_cool_plates(image_path, min_colorfulness_threshold=0.05):
    """
    Creates grayscale "warm" and "cool" color plates where intensity reflects S*V.
    Utilizes colorsys for HSV conversion, as in extract_and_overlay_with_transparency.

    Args:
        image_path (str): The path to the image file.
        min_colorfulness_threshold (float): A minimum threshold for S*V (0-1).
                                            Pixels with S*V below this will be black
                                            in both warm/cool plates, filtering out
                                            near-achromatic areas.

    Returns:
        tuple: (warm_plate, cool_plate) as NumPy arrays (0-255 grayscale).
               Returns (None, None) if an error occurs.
    """
    check_file_exists(image_path)
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image, dtype=np.uint8)

    # Normalize RGB to [0, 1] for colorsys
    img_rgb_norm = img_array / 255.0
    R, G, B = img_rgb_norm[..., 0], img_rgb_norm[..., 1], img_rgb_norm[..., 2]

    # Convert RGB to HSV using colorsys.rgb_to_hsv (returns H, S, V in [0, 1])
    h_channel, s_channel, v_channel = np.vectorize(colorsys.rgb_to_hsv)(R, G, B)

    # Calculate combined saturation and value (S*V) as our 'colorfulness' metric
    colorfulness = s_channel * v_channel

    # Initialize warm and cool plates as all zeros (black)
    warm_plate_data = np.zeros_like(colorfulness, dtype=np.float32)
    cool_plate_data = np.zeros_like(colorfulness, dtype=np.float32)

    # Define warm and cool hue masks (Hue in [0, 1] from colorsys)
    # A common split point: Hues from ~0.83 (Magenta) to 0/1 (Red) to ~0.16 (Yellow) are Warm.
    # Hues from ~0.16 (Yellow) to ~0.83 (Magenta) are Cool.
    # These boundaries are adaptable based on perceptual preference.

    # Warm colors: (0.83 to 1.0) and (0.0 to 0.16) - covers Red, Orange, Yellow
    warm_hue_mask = (h_channel >= 0.83) | (h_channel < 0.16)

    # Cool colors: (0.16 to 0.83) - covers Green, Cyan, Blue, Violet/Magenta-ish
    cool_hue_mask = (h_channel >= 0.16) & (h_channel < 0.83)

    # Apply the colorfulness threshold to filter out achromatic areas
    eligible_color_pixels = colorfulness > min_colorfulness_threshold

    # Populate warm plate: pixels must be warm-hued AND sufficiently colorful
    warm_mask_indices = eligible_color_pixels & warm_hue_mask
    warm_plate_data[warm_mask_indices] = colorfulness[warm_mask_indices]

    # Populate cool plate: pixels must be cool-hued AND sufficiently colorful
    cool_mask_indices = eligible_color_pixels & cool_hue_mask
    cool_plate_data[cool_mask_indices] = colorfulness[cool_mask_indices]

    # Convert plates to 0-255 uint8 for display (multiply by 255)
    warm_plate_display = (warm_plate_data * 255).astype(np.uint8)
    cool_plate_display = (cool_plate_data * 255).astype(np.uint8)

    # Display the original, warm, and cool plates
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_array)
    axes[0].set_title(f"Original Image")
    axes[0].axis('off')

    axes[1].imshow(warm_plate_display, cmap='gray')
    axes[1].set_title(f"Warm Colors (Intensity S*V)")
    axes[1].axis('off')

    axes[2].imshow(cool_plate_display, cmap='gray')
    axes[2].set_title(f"Cool Colors (Intensity S*V)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    return warm_plate_display, cool_plate_display


if __name__ == "__main__":
    # --- IMPORTANT: Replace 'path/to/your/image.jpg' with the actual path to your image file ---
    #    Examples:
    #    - Windows: r"C:\Users\YourUser\Pictures\my_photo.jpg" (the 'r' before the string handles backslashes)
    #    - macOS/Linux: "/home/youruser/images/my_photo.png"
    # Make sure the file exists at that path.
    image_file_path = "path/to/your/image.jpg"  # <--- CHANGE THIS LINE

    # --- Optional: Dummy image for quick testing if you don't have an image ready ---
    # This block will create a comprehensive test image if the specified 'image_file_path'
    # doesn't exist or is still set to the default placeholder.
    if not os.path.exists(image_file_path) or image_file_path == "path/to/your/image.jpg":
        print("\n--- No valid image specified or found. Creating a comprehensive dummy image for demonstration. ---")
        dummy_image_name = "warm_cool_test_image_advanced.png"
        if not os.path.exists(dummy_image_name):
            print(f"Creating a dummy image '{dummy_image_name}'...")
            dummy_img_data = np.zeros((150, 200, 3), dtype=np.uint8)

            # Row 1: Pure Warm & Cool (high S, high V)
            dummy_img_data[0:50, 0:50] = [255, 0, 0]  # Red (Warm)
            dummy_img_data[0:50, 50:100] = [255, 165, 0]  # Orange (Warm)
            dummy_img_data[0:50, 100:150] = [255, 255, 0]  # Yellow (Warm)
            dummy_img_data[0:50, 150:200] = [0, 0, 255]  # Blue (Cool)

            # Row 2: Dark Warm & Cool (high S, low V) - will be dimmer in S*V plate
            dummy_img_data[50:100, 0:50] = [128, 0, 0]  # Dark Red (Warm)
            dummy_img_data[50:100, 50:100] = [128, 80, 0]  # Dark Orange (Warm)
            dummy_img_data[50:100, 100:150] = [0, 0, 128]  # Dark Blue (Cool)
            dummy_img_data[50:100, 150:200] = [0, 128, 0]  # Dark Green (Cool)

            # Row 3: Desaturated Warm & Cool (low S, high V) - will be dimmer in S*V plate
            dummy_img_data[100:150, 0:50] = [255, 128, 128]  # Pink (Warm)
            dummy_img_data[100:150, 50:100] = [255, 255, 128]  # Pale Yellow (Warm)
            dummy_img_data[100:150, 100:150] = [128, 128, 255]  # Light Blue (Cool)
            dummy_img_data[100:150, 150:200] = [128, 255, 128]  # Light Green (Cool)

            dummy_img = Image.fromarray(dummy_img_data)
            dummy_img.save(dummy_image_name)
            print("Dummy image created.")
        image_file_path = dummy_image_name  # Use the dummy image for demonstration

    # You can adjust this threshold to fine-tune what's considered "colorful enough"
    # Lower value means more subtle colors are included.
    MIN_COLORFULNESS_THRESHOLD = 0.05

    warm_plate_result, cool_plate_result = create_warm_cool_plates(
        image_file_path,
        min_colorfulness_threshold=MIN_COLORFULNESS_THRESHOLD
    )
    if warm_plate_result is not None:
        print(f"\nWarm and Cool color plates successfully generated and displayed.")

    # You can uncomment and use your generate_temperature_heatmap if needed,
    # but it expects grayscale paths for INDIVIDUAL colors (red, green, blue etc.),
    # not directly the warm/cool plates from this function.
    # If you wanted a heatmap of warm vs cool, you'd combine warm_plate_result
    # and cool_plate_result differently.
    #
    # Example for using generate_temperature_heatmap (requires saving individual color masks first)
    # This part is just for context, not directly run by default with warm/cool plates
    # if warm_plate_result is not None:
    #     temp_red_path = "temp_red_mask.png"
    #     temp_orange_path = "temp_orange_mask.png"
    #     temp_yellow_path = "temp_yellow_mask.png"
    #     temp_green_path = "temp_green_mask.png"
    #     temp_blue_path = "temp_blue_mask.png"
    #     temp_violet_path = "temp_violet_mask.png"
    #     output_heatmap_path = "combined_temperature_heatmap.png"

    #     # You'd need to extract individual color masks similar to previous examples
    #     # and save them to these paths before calling generate_temperature_heatmap.
    #     # This 'create_warm_cool_plates' function doesn't produce those intermediate individual masks.
    #     print("\nNote: generate_temperature_heatmap requires individual color masks (red, orange, etc.),")
    #     print("which are not directly produced by create_warm_cool_plates. This section is commented out.")