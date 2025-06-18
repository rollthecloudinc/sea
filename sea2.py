import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from skimage.feature import local_binary_pattern
from skimage.segmentation import slic # Import SLIC
from skimage.segmentation import mark_boundaries # For visualization
import colorsys
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict # <--- ADD THIS LINE
from depth import depth_estimation_heatmap
from util import check_file_exists, apply_boolean_mask_by_color, identify_negative_superpixel, find_overlapping_negative_superpixels, visualize_specific_superpixels, create_fused_boolean_mask, find_closest_negative_pixel_spiral
from shapes import detect_shapes, generate_smooth_obstacle_heatmap, generate_navigable_heatmap, generate_obstacle_heatmap
from color import extract_and_overlay_with_transparency, generate_temperature_heatmap
from texture import analyze_texture

# Main Function
def main():
    # Input and output paths
    input_image_path = "input_image.jpg"  # Replace with your image file
    output_dir = "output_results"
    os.makedirs(output_dir, exist_ok=True)

    # ... (input/output path definitions) ...

    # --- Create a dummy image for testing if it doesn't exist ---
    # ... (dummy image creation code) ...

    # --- Generate common superpixel segments ONCE for consistent analysis ---
    # This is crucial for comparing superpixels across different masks
    image_for_slic = cv2.imread(input_image_path)  # <--- Image loaded here
    if image_for_slic is None:
        raise ValueError(f"Error: Unable to load original image for SLIC at {input_image_path}")

    # Parameters for SLIC (can be tuned)
    num_superpixels = 500
    compactness = 20.0
    # The `slic` function from `skimage.segmentation` generates the superpixel segments
    segments = slic(image_for_slic, n_segments=num_superpixels, compactness=compactness, sigma=1,
                    enforce_connectivity=True, slic_zero=False)  # <--- 'segments' is created here!
    print(f"\nGenerated {segments.max() + 1} superpixel segments for the image.")

    # Lists to store the identified negative superpixel labels from each source
    all_negative_superpixel_label_lists = [] # <--- It starts here as an empty list

    # Output file paths
    output_image_path_depth = os.path.join(output_dir, "depth_heatmap.jpg")
    output_image_path_shapes = os.path.join(output_dir, "shapes_detected.jpg")
    output_image_path_texture = os.path.join(output_dir, "texture_analysis.jpg")
    output_image_path_navigable = os.path.join(output_dir, "navigable_heatmap.jpg")

    # Color overlay paths
    grayscale_output_path_red = os.path.join(output_dir, "grayscale_red.jpg")
    transparent_overlay_output_path_red = os.path.join(output_dir, "transparent_red_overlay.png")

    grayscale_output_path_blue = os.path.join(output_dir, "grayscale_blue.jpg")
    transparent_overlay_output_path_blue = os.path.join(output_dir, "transparent_blue_overlay.png")

    grayscale_output_path_yellow = os.path.join(output_dir, "grayscale_yellow.jpg")
    transparent_overlay_output_path_yellow = os.path.join(output_dir, "transparent_yellow_overlay.png")

    grayscale_output_path_orange = os.path.join(output_dir, "grayscale_orange.jpg")
    transparent_overlay_output_path_orange = os.path.join(output_dir, "transparent_orange_overlay.png")

    grayscale_output_path_green = os.path.join(output_dir, "grayscale_green.jpg")
    transparent_overlay_output_path_green = os.path.join(output_dir, "transparent_green_overlay.png")

    grayscale_output_path_violet = os.path.join(output_dir, "grayscale_violet.jpg")
    transparent_overlay_output_path_violet = os.path.join(output_dir, "transparent_violet_overlay.png")

    # Run all processing functions
    try:
        # Depth Estimation
        print("Running depth estimation heatmap...")
        depth_estimation_heatmap(input_image_path, output_image_path_depth)

        # Shape Detection
        print("Running shape detection...")
        detect_shapes(input_image_path, output_image_path_shapes)

        # Texture Analysis
        print("Running texture analysis...")
        analyze_texture(input_image_path, output_image_path_texture)

        # Navigable Heatmap Generation
        print("Running navigable heatmap generation...")
        generate_navigable_heatmap(input_image_path, output_image_path_navigable)

        # Color Overlays
        print("Running red color overlay...")
        extract_and_overlay_with_transparency(
            image_path=input_image_path,
            grayscale_path=grayscale_output_path_red,
            overlay_path=transparent_overlay_output_path_red,
            hue_min=0, hue_max=30, highlight_color=(255, 0, 0, 128)
        )

        print("Running blue color overlay...")
        extract_and_overlay_with_transparency(
            image_path=input_image_path,
            grayscale_path=grayscale_output_path_blue,
            overlay_path=transparent_overlay_output_path_blue,
            hue_min=200, hue_max=240, highlight_color=(0, 0, 255, 128)
        )

        print("Running yellow color overlay...")
        extract_and_overlay_with_transparency(
            image_path=input_image_path,
            grayscale_path=grayscale_output_path_yellow,
            overlay_path=transparent_overlay_output_path_yellow,
            hue_min=45, hue_max=75, highlight_color=(255, 255, 0, 128)
        )

        print("Running orange color overlay...")
        extract_and_overlay_with_transparency(
            image_path=input_image_path,
            grayscale_path=grayscale_output_path_orange,
            overlay_path=transparent_overlay_output_path_orange,
            hue_min=30, hue_max=60, highlight_color=(255, 165, 0, 128)
        )

        print("Running green color overlay...")
        extract_and_overlay_with_transparency(
            image_path=input_image_path,
            grayscale_path=grayscale_output_path_green,
            overlay_path=transparent_overlay_output_path_green,
            hue_min=90, hue_max=150, highlight_color=(0, 255, 0, 128)
        )

        print("Running violet color overlay...")
        extract_and_overlay_with_transparency(
            image_path=input_image_path,
            grayscale_path=grayscale_output_path_violet,
            overlay_path=transparent_overlay_output_path_violet,
            hue_min=270,
            hue_max=300, highlight_color=(128, 0, 128, 128)
        )

        red_grayscale_path = grayscale_output_path_red
        orange_grayscale_path = grayscale_output_path_orange
        yellow_grayscale_path = grayscale_output_path_yellow
        green_grayscale_path = grayscale_output_path_green
        blue_grayscale_path = grayscale_output_path_blue
        violet_grayscale_path = grayscale_output_path_violet
        output_temp_heatmap_path = os.path.join(output_dir, "color_temp_heatmap.jpg")

        generate_temperature_heatmap(
            red_grayscale_path,
            orange_grayscale_path,
            yellow_grayscale_path,
            green_grayscale_path,
            blue_grayscale_path,
            violet_grayscale_path,
            output_temp_heatmap_path,
        )

        output_obstacle_heatmap_path = os.path.join(output_dir, "obstacle_heatmap.jpg")
        generate_smooth_obstacle_heatmap(output_image_path_shapes, output_obstacle_heatmap_path)

        #output_navigatable_v2_heatmap_path = os.path.join(output_dir, "navigatable_heatmap_v2.jpg")
        #generate_navigable_heatmap_v2(output_image_path_navigable, output_navigatable_v2_heatmap_path)

        output_image_path_depth_bool = os.path.join(output_dir, "depth_heatmap_bool.jpg")
        output_temp_heatmap_path_bool = os.path.join(output_dir, "color_temp_heatmap_bool.jpg")
        output_obstacle_heatmap_path_bool = os.path.join(output_dir, "obstacle_heatmap_bool.jpg")
        apply_boolean_mask_by_color(output_image_path_depth, output_image_path_depth_bool)
        apply_boolean_mask_by_color(output_temp_heatmap_path, output_temp_heatmap_path_bool)
        apply_boolean_mask_by_color(output_obstacle_heatmap_path, output_obstacle_heatmap_path_bool)

        # 3. Identify Negative Superpixels (now uses the pre-computed boolean mask)
        output_superpixel_visualization_path = os.path.join(output_dir, "depth_heatmap_superpixel.jpg")
        print("\n--- Identifying Negative Space Superpixels ---")
        identified_superpixels = identify_negative_superpixel(
            original_image_path=input_image_path,
            boolean_mask_path=output_image_path_depth_bool, # Pass the path to the already saved boolean mask
            output_superpixel_visualization_path=output_superpixel_visualization_path, # Save the new visualization
            #num_superpixels=500,
            #compactness=20.0,
            desired_pixel_coverage_percent=95.0, # e.g., 95% of pixels must be 0 for this superpixel
            visualize=True,
            segments=segments
        )
        if identified_superpixels:
            print("\nFound the following negative space superpixels:")
            all_negative_superpixel_label_lists.append(identified_superpixels)
        else:
            print("\nNo superpixels found that are predominantly negative space.")

       # 6. Superpixel Analysis from Temperature Boolean Mask
        output_superpixel_visualization_path_temp = os.path.join(output_dir, "color_temp_heatmap_superpixel.jpg")
        print("\n--- Identifying Negative Space Superpixels (from Temperature Mask) ---")
        identified_superpixels_temp = identify_negative_superpixel(
            original_image_path=input_image_path,
            boolean_mask_path=output_temp_heatmap_path_bool, # <--- Using the new boolean mask
            output_superpixel_visualization_path=output_superpixel_visualization_path_temp, # <--- New output path
            #num_superpixels=500,
            #compactness=20.0,
            desired_pixel_coverage_percent=95.0, # Can be adjusted
            visualize=True,
            segments=segments
        )
        if identified_superpixels_temp:
            print(f"\nFound {len(identified_superpixels_temp)} negative superpixels from Temperature Mask.")
            all_negative_superpixel_label_lists.append(identified_superpixels_temp)
        else:
            print("\nNo superpixels found that are predominantly negative space from Temperature Mask.")

        # 9. Superpixel Analysis from Obstacle Boolean Mask
        print("\n--- Identifying Negative Space Superpixels (from Obstacle Mask) ---")
        output_superpixel_visualization_path_obstacle = os.path.join(output_dir, "obstacle_heatmap_superpixel.jpg")
        identified_superpixels_obstacle = identify_negative_superpixel(
            original_image_path=input_image_path,
            boolean_mask_path=output_obstacle_heatmap_path_bool,  # <--- Using the new boolean mask
            output_superpixel_visualization_path=output_superpixel_visualization_path_obstacle,
            # <--- New output path
            #num_superpixels=500,
            #compactness=20.0,
            desired_pixel_coverage_percent=95.0,  # Can be adjusted
            visualize=True,
            segments=segments
        )
        if identified_superpixels_obstacle:
            print(f"\nFound {len(identified_superpixels_obstacle)} negative superpixels from Obstacle Mask.")
            all_negative_superpixel_label_lists.append(identified_superpixels_obstacle)
        else:
            print("\nNo superpixels found that are predominantly negative space from Obstacle Mask.")

        # --- 4. Combined (Union) Negative Superpixel Analysis ---
        print("\n--- Combining Boolean Masks for Union of Negative Space ---")
        # List all the boolean mask paths you want to combine
        boolean_mask_paths_to_combine = [
            output_superpixel_visualization_path,
            output_superpixel_visualization_path_temp,
            output_superpixel_visualization_path_obstacle
        ]

        output_combined_boolean_mask_path = os.path.join(output_dir, "superpixel_fusion_map.jpg")
        #combine_boolean_masks_union(
        #    mask_paths=boolean_mask_paths_to_combine,
        #    output_combined_mask_path=output_combined_boolean_mask_path
        #)

        # --- NEW: Superpixel-level Overlap/Intersection Fusion ---
        print("\n--- Performing Superpixel-level Overlap/Intersection Fusion ---")
        # Define the minimum number of sources that must identify a superpixel as negative
        min_overlap_sources = 2  # e.g., at least 2 out  of 3 sources must agree

        fused_negative_superpixel_labels = find_overlapping_negative_superpixels(
            list_of_negative_superpixel_labels=all_negative_superpixel_label_lists,
            min_overlap_sources=min_overlap_sources
        )

        output_fused_superpixel_visualization_path = os.path.join(output_dir, "superpixel_fusion_map.jpg")
        if fused_negative_superpixel_labels:
            print(
                f"\nFound {len(fused_negative_superpixel_labels)} fused negative superpixels (overlap from >= {min_overlap_sources} sources).")
            # Visualize the fused superpixels
            visualize_specific_superpixels(
                image_path=input_image_path,
                output_path=output_fused_superpixel_visualization_path,
                segments=segments,  # Pass the original segments array
                superpixel_labels_to_highlight=fused_negative_superpixel_labels,
                highlight_color=(0, 0, 255),  # Red for danger
                #show_boundaries=True
            )
        else:
            print("\nNo superpixels found that meet the minimum overlap criteria for fusion.")

        # --- NEW: Create and save the fused boolean mask ---
        output_fused_boolean_mask_path = os.path.join(output_dir, "superpixel_fusion_map_bool.jpg")
        fused_boolean_array = create_fused_boolean_mask(
            segments,
            fused_negative_superpixel_labels,
            output_fused_boolean_mask_path
        )

        # --- NEW: Find the closest negative pixel using the spiral algorithm ---
        closest_pixel_coords = find_closest_negative_pixel_spiral(fused_boolean_array)

        if closest_pixel_coords:
            print(f"Closest negative space pixel found at (row, col): {closest_pixel_coords}")
            # You can visualize this point on the original image if you like
            # e.g., using cv2.circle on image_rgb or a copy of it
        else:
            print("No negative space pixels found in the fused boolean mask.")


    except FileNotFoundError as e:
        print(f"File not found error: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()