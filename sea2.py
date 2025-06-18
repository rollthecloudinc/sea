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

# Import all custom utility functions and modules
from depth import depth_estimation_heatmap
from util import (
    check_file_exists,
    apply_boolean_mask_by_color,
    identify_negative_superpixel,
    find_overlapping_negative_superpixels,
    visualize_specific_superpixels,
    create_fused_boolean_mask,
    find_closest_negative_pixel_spiral,
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
    extract_and_overlay_with_transparency(
        image_path=image_path, grayscale_path=paths["grayscale_red"],
        overlay_path=paths["transparent_red_overlay"], hue_min=0, hue_max=30, highlight_color=(255, 0, 0, 128)
    )
    extract_and_overlay_with_transparency(
        image_path=image_path, grayscale_path=paths["grayscale_blue"],
        overlay_path=paths["transparent_blue_overlay"], hue_min=200, hue_max=240, highlight_color=(0, 0, 255, 128)
    )
    extract_and_overlay_with_transparency(
        image_path=image_path, grayscale_path=paths["grayscale_yellow"],
        overlay_path=paths["transparent_yellow_overlay"], hue_min=45, hue_max=75, highlight_color=(255, 255, 0, 128)
    )
    extract_and_overlay_with_transparency(
        image_path=image_path, grayscale_path=paths["grayscale_orange"],
        overlay_path=paths["transparent_orange_overlay"], hue_min=30, hue_max=60, highlight_color=(255, 165, 0, 128)
    )
    extract_and_overlay_with_transparency(
        image_path=image_path, grayscale_path=paths["grayscale_green"],
        overlay_path=paths["transparent_green_overlay"], hue_min=90, hue_max=150, highlight_color=(0, 255, 0, 128)
    )
    extract_and_overlay_with_transparency(
        image_path=image_path, grayscale_path=paths["grayscale_violet"],
        overlay_path=paths["transparent_violet_overlay"], hue_min=270, hue_max=300, highlight_color=(128, 0, 128, 128)
    )

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

def identify_negative_superpixels_from_sources(original_image_path, segments, paths):
    """Identifies negative superpixels from various boolean mask sources."""
    all_negative_superpixel_label_lists = []

    print("\n--- Identifying Negative Space Superpixels (from Depth Mask) ---")
    identified_superpixels_depth = identify_negative_superpixel(
        original_image_path=original_image_path,
        boolean_mask_path=paths["depth_heatmap_bool"],
        output_superpixel_visualization_path=paths["depth_heatmap_superpixel"],
        desired_pixel_coverage_percent=DESIRED_PIXEL_COVERAGE_PERCENT,
        visualize=True,
        segments=segments
    )
    if identified_superpixels_depth:
        print(f"Found {len(identified_superpixels_depth)} negative superpixels from Depth Mask.")
        all_negative_superpixel_label_lists.append(identified_superpixels_depth)
    else:
        print("No superpixels found that are predominantly negative space from Depth Mask.")

    print("\n--- Identifying Negative Space Superpixels (from Temperature Mask) ---")
    identified_superpixels_temp = identify_negative_superpixel(
        original_image_path=original_image_path,
        boolean_mask_path=paths["color_temp_heatmap_bool"],
        output_superpixel_visualization_path=paths["color_temp_heatmap_superpixel"],
        desired_pixel_coverage_percent=DESIRED_PIXEL_COVERAGE_PERCENT,
        visualize=True,
        segments=segments
    )
    if identified_superpixels_temp:
        print(f"Found {len(identified_superpixels_temp)} negative superpixels from Temperature Mask.")
        all_negative_superpixel_label_lists.append(identified_superpixels_temp)
    else:
        print("No superpixels found that are predominantly negative space from Temperature Mask.")

    print("\n--- Identifying Negative Space Superpixels (from Obstacle Mask) ---")
    identified_superpixels_obstacle = identify_negative_superpixel(
        original_image_path=original_image_path,
        boolean_mask_path=paths["obstacle_heatmap_bool"],
        output_superpixel_visualization_path=paths["obstacle_heatmap_superpixel"],
        desired_pixel_coverage_percent=DESIRED_PIXEL_COVERAGE_PERCENT,
        visualize=True,
        segments=segments
    )
    if identified_superpixels_obstacle:
        print(f"Found {len(identified_superpixels_obstacle)} negative superpixels from Obstacle Mask.")
        all_negative_superpixel_label_lists.append(identified_superpixels_obstacle)
    else:
        print("No superpixels found that are predominantly negative space from Obstacle Mask.")

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
            highlight_color=(0, 0, 255),  # Red for danger
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
        perform_initial_analyses(INPUT_IMAGE_PATH, paths)
        generate_boolean_masks(paths)
        all_negative_superpixel_label_lists = identify_negative_superpixels_from_sources(
            INPUT_IMAGE_PATH, segments, paths
        )
        perform_superpixel_fusion(INPUT_IMAGE_PATH, segments, all_negative_superpixel_label_lists, paths)

    except FileNotFoundError as e:
        print(f"File not found error: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()