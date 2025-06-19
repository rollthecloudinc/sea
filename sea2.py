import threading
import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from skimage.feature import local_binary_pattern
from skimage.segmentation import slic, mark_boundaries
import colorsys
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
import concurrent.futures # Import for ThreadPoolExecutor

# Import all custom utility functions and modules
# Ensure these imports correctly point to your local files
from depth import depth_estimation_heatmap
from util import (
    check_file_exists,
    apply_boolean_mask_by_color,
    identify_negative_superpixel, # This is the function we'll parallelize calls to
    find_overlapping_negative_superpixels,
    visualize_specific_superpixels,
    create_fused_boolean_mask,
    find_closest_negative_pixel_spiral,
    calculate_vector_from_center
)
from shapes import detect_shapes, generate_smooth_obstacle_heatmap, generate_navigable_heatmap, generate_obstacle_heatmap
from color import extract_and_overlay_with_transparency, generate_temperature_heatmap
from texture import analyze_texture

# -----------------------------------------------------------------
# Configuration and Constants
# -----------------------------------------------------------------

INPUT_IMAGE_PATH = "input_image.jpg"  # Replace with your image file
OUTPUT_DIR = "output_results"

# SLIC Parameters
NUM_SUPERPIXELS = 500
COMPACTNESS = 20.0
DESIRED_PIXEL_COVERAGE_PERCENT = 95.0 # For identifying negative superpixels

# Fusion Parameters
MIN_OVERLAP_SOURCES = 2 # At least this many sources must agree on a negative superpixel

# -----------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------

def create_output_paths(base_dir):
    """Generates and returns a dictionary of all output file paths."""
    paths = {
        "depth_heatmap": os.path.join(base_dir, "depth_heatmap.jpg"),
        "shapes_detected": os.path.join(base_dir, "shapes_detected.jpg"),
        "texture_analysis": os.path.join(base_dir, "texture_analysis.jpg"),
        "navigable_heatmap": os.path.join(base_dir, "navigable_heatmap.jpg"),
        "grayscale_red": os.path.join(base_dir, "grayscale_red.jpg"),
        "transparent_red_overlay": os.path.join(base_dir, "transparent_red_overlay.png"),
        "grayscale_blue": os.path.join(base_dir, "grayscale_blue.jpg"),
        "transparent_blue_overlay": os.path.join(base_dir, "transparent_blue_overlay.png"),
        "grayscale_yellow": os.path.join(base_dir, "grayscale_yellow.jpg"),
        "transparent_yellow_overlay": os.path.join(base_dir, "transparent_yellow_overlay.png"),
        "grayscale_orange": os.path.join(base_dir, "grayscale_orange.jpg"),
        "transparent_orange_overlay": os.path.join(base_dir, "transparent_orange_overlay.png"),
        "grayscale_green": os.path.join(base_dir, "grayscale_green.jpg"),
        "transparent_green_overlay": os.path.join(base_dir, "transparent_green_overlay.png"),
        "grayscale_violet": os.path.join(base_dir, "grayscale_violet.jpg"),
        "transparent_violet_overlay": os.path.join(base_dir, "transparent_violet_overlay.png"),
        "color_temp_heatmap": os.path.join(base_dir, "color_temp_heatmap.jpg"),
        "obstacle_heatmap": os.path.join(base_dir, "obstacle_heatmap.jpg"),
        "depth_heatmap_bool": os.path.join(base_dir, "depth_heatmap_bool.jpg"),
        "color_temp_heatmap_bool": os.path.join(base_dir, "color_temp_heatmap_bool.jpg"),
        "obstacle_heatmap_bool": os.path.join(base_dir, "obstacle_heatmap_bool.jpg"),
        "depth_heatmap_superpixel": os.path.join(base_dir, "depth_heatmap_superpixel.jpg"),
        "color_temp_heatmap_superpixel": os.path.join(base_dir, "color_temp_heatmap_superpixel.jpg"),
        "obstacle_heatmap_superpixel": os.path.join(base_dir, "obstacle_heatmap_superpixel.jpg"),
        "superpixel_fusion_map": os.path.join(base_dir, "superpixel_fusion_map.jpg"),
        "superpixel_fusion_map_bool": os.path.join(base_dir, "superpixel_fusion_map_bool.jpg"),
    }
    return paths

def perform_initial_analyses(image_path, paths):
    """Executes initial image processing tasks and saves results."""
    print("--- Performing Initial Image Analyses ---")

    print("Running depth estimation heatmap...")
    depth_estimation_heatmap(image_path, paths["depth_heatmap"])

    print("Running shape detection...")
    detect_shapes(image_path, paths["shapes_detected"])

    print("Running texture analysis...")
    analyze_texture(image_path, paths["texture_analysis"])

    print("Running navigable heatmap generation...")
    generate_navigable_heatmap(image_path, paths["navigable_heatmap"])

    print("Running color overlays...")
    # List of arguments for each thread
    thread_args_list = [
        (image_path, paths["grayscale_red"], paths["transparent_red_overlay"], 0, 30, (255, 0, 0, 128)),
        (image_path, paths["grayscale_blue"], paths["transparent_blue_overlay"], 200, 240, (0, 0, 255, 128)),
        (image_path, paths["grayscale_yellow"], paths["transparent_yellow_overlay"], 45, 75, (255, 255, 0, 128)),
        (image_path, paths["grayscale_orange"], paths["transparent_orange_overlay"], 30, 60, (255, 165, 0, 128)),
        (image_path, paths["grayscale_green"], paths["transparent_green_overlay"], 90, 150, (0, 255, 0, 128)),
        (image_path, paths["grayscale_violet"], paths["transparent_violet_overlay"], 270, 300, (128, 0, 128, 128)),
    ]
    threads = []
    print("Starting color extraction operations in parallel threads...")
    for args in thread_args_list:
        # Create a thread for each operation
        thread = threading.Thread(target=extract_and_overlay_with_transparency, args=args)
        threads.append(thread)
        thread.start() # Start the thread
    # Wait for all threads to complete
    for thread in threads:
        thread.join() # This blocks until the thread finishes

    print("Generating color temperature heatmap...")
    generate_temperature_heatmap(
        paths["grayscale_red"], paths["grayscale_orange"], paths["grayscale_yellow"],
        paths["grayscale_green"], paths["grayscale_blue"], paths["grayscale_violet"],
        paths["color_temp_heatmap"],
    )

    print("Generating smooth obstacle heatmap...")
    generate_smooth_obstacle_heatmap(paths["shapes_detected"], paths["obstacle_heatmap"])


def generate_boolean_masks(paths):
    """Applies boolean masks to generated heatmaps."""
    print("--- Generating Boolean Masks ---")
    apply_boolean_mask_by_color(paths["depth_heatmap"], paths["depth_heatmap_bool"])
    apply_boolean_mask_by_color(paths["color_temp_heatmap"], paths["color_temp_heatmap_bool"])
    apply_boolean_mask_by_color(paths["obstacle_heatmap"], paths["obstacle_heatmap_bool"])


# --- MODIFIED: Parallelized identify_negative_superpixels_from_sources ---
def identify_negative_superpixels_from_sources(original_image_path, segments, paths):
    """
    Identifies negative superpixels from various boolean mask sources in parallel.
    Returns a list of lists, where each inner list contains superpixel labels
    identified as negative by one source.
    """
    all_negative_superpixel_label_lists = []

    # Define a helper function to be executed by each thread
    def _run_identify_and_collect(mask_type, boolean_mask_path, superpixel_viz_path, results_list):
        print(f"\n--- Identifying Negative Space Superpixels (from {mask_type} Mask) ---")
        identified_superpixels = identify_negative_superpixel(
            original_image_path=original_image_path,
            boolean_mask_path=boolean_mask_path,
            output_superpixel_visualization_path=superpixel_viz_path,
            desired_pixel_coverage_percent=DESIRED_PIXEL_COVERAGE_PERCENT,
            visualize=False,
            segments=segments # Pass segments directly
        )
        if identified_superpixels:
            print(f"Found {len(identified_superpixels)} negative superpixels from {mask_type} Mask.")
            results_list.append(identified_superpixels) # Append directly, as list append is mostly atomic
        else:
            print(f"No superpixels found that are predominantly negative space from {mask_type} Mask.")

    # Use ThreadPoolExecutor for parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        # Submit tasks for each source
        futures.append(executor.submit(
            _run_identify_and_collect,
            "Depth", paths["depth_heatmap_bool"], paths["depth_heatmap_superpixel"],
            all_negative_superpixel_label_lists # Pass the shared list
        ))
        futures.append(executor.submit(
            _run_identify_and_collect,
            "Temperature", paths["color_temp_heatmap_bool"], paths["color_temp_heatmap_superpixel"],
            all_negative_superpixel_label_lists # Pass the shared list
        ))
        futures.append(executor.submit(
            _run_identify_and_collect,
            "Obstacle", paths["obstacle_heatmap_bool"], paths["obstacle_heatmap_superpixel"],
            all_negative_superpixel_label_lists # Pass the shared list
        ))

        # Wait for all futures to complete (optional, but good for sequential logic)
        for future in concurrent.futures.as_completed(futures):
            # You can check for exceptions here if needed:
            # try:
            #     future.result()
            # except Exception as exc:
            #     print(f'Generated an exception: {exc}')
            pass # Results are appended directly in _run_identify_and_collect

    print("\n--- All parallel superpixel identification operations completed ---")
    return all_negative_superpixel_label_lists


def perform_superpixel_fusion(original_image_path, segments, all_negative_superpixel_label_lists, paths):
    """Fuses negative superpixels from multiple sources and finds the closest negative pixel."""
    print("\n--- Performing Superpixel-level Overlap/Intersection Fusion ---")

    fused_negative_superpixel_labels = find_overlapping_negative_superpixels(
        list_of_negative_superpixel_labels=all_negative_superpixel_label_lists,
        min_overlap_sources=MIN_OVERLAP_SOURCES
    )

    if fused_negative_superpixel_labels:
        print(
            f"Found {len(fused_negative_superpixel_labels)} fused negative superpixels "
            f"(overlap from >= {MIN_OVERLAP_SOURCES} sources)."
        )
        visualize_specific_superpixels(
            image_path=original_image_path,
            output_path=paths["superpixel_fusion_map"],
            segments=segments,
            superpixel_labels_to_highlight=fused_negative_superpixel_labels,
            highlight_color=(0, 0, 255),  # Red for danger (or green for negative space?)
        )
        fused_boolean_array = create_fused_boolean_mask(
            segments,
            fused_negative_superpixel_labels,
            paths["superpixel_fusion_map_bool"]
        )

        print("\n--- Finding Closest Negative Pixel ---")
        closest_pixel_coords = find_closest_negative_pixel_spiral(fused_boolean_array)

        if closest_pixel_coords:
            print(f"Closest negative space pixel found at (row, col): {closest_pixel_coords}")
            # Optional: Draw the closest pixel on the original image for final visualization
            original_image = cv2.imread(original_image_path)
            if original_image is not None:
                image_height, image_width, _ = original_image.shape
                magnitude, angle = calculate_vector_from_center(image_width, image_height, closest_pixel_coords)
                print(f"  Magnitude: {magnitude:.2f} pixels")
                print(f"  Angle: {angle:.2f} degrees (relative to center, 0=right, counter-clockwise)")
                cv2.circle(original_image, (closest_pixel_coords[1], closest_pixel_coords[0]), 5, (0, 0, 255), -1)
                cv2.imwrite(os.path.join(OUTPUT_DIR, "final_image_with_closest_pixel.png"), original_image)
        else:
            print("No negative space pixels found in the fused boolean mask.")
    else:
        print("No superpixels found that meet the minimum overlap criteria for fusion.")

# -----------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    paths = create_output_paths(OUTPUT_DIR)

    # Load original image for SLIC and subsequent visualizations
    image_for_slic = cv2.imread(INPUT_IMAGE_PATH)
    if image_for_slic is None:
        raise ValueError(f"Error: Unable to load original image for SLIC at {INPUT_IMAGE_PATH}")

    # Generate common superpixel segments ONCE for consistent analysis
    print(f"\nGenerating superpixel segments ({NUM_SUPERPIXELS} segments, compactness {COMPACTNESS})...")
    segments = slic(image_for_slic, n_segments=NUM_SUPERPIXELS, compactness=COMPACTNESS, sigma=1,
                    enforce_connectivity=True, slic_zero=False)
    print(f"Generated {segments.max() + 1} superpixel segments for the image.")

    try:
        # Stage 1: Initial analyses (some parts already parallelized within here)
        perform_initial_analyses(INPUT_IMAGE_PATH, paths)

        # Stage 2: Generate boolean masks (sequential here, but quick)
        generate_boolean_masks(paths)

        # Stage 3: Identify negative superpixels from sources (NOW PARALLELIZED)
        all_negative_superpixel_label_lists = identify_negative_superpixels_from_sources(
            INPUT_IMAGE_PATH, segments, paths
        )

        # Stage 4: Superpixel fusion and closest pixel finding (sequential after parallel parts)
        perform_superpixel_fusion(INPUT_IMAGE_PATH, segments, all_negative_superpixel_label_lists, paths)

    except FileNotFoundError as e:
        print(f"File not found error: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()