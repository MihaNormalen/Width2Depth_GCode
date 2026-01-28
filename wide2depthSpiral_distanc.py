import os
import cv2
import numpy as np
import argparse
import math
import random
from PIL import Image
import traceback

from skimage.morphology import skeletonize, remove_small_objects
import sknw
from scipy.ndimage import distance_transform_edt

class GCodeBaseGenerator:
    """
    Base class for G-code generation with common parameters and dipping logic.
    Includes a global Z-offset for machine coordinate adjustment.
    Dipping now triggered by painting distance traveled, not by fixed number of dips.
    """
    def __init__(self, feed_rate, feed_rate_dip, x_offset, y_offset,
                 dip_location_raw, max_paint_distance_mm,
                 dip_wipe_radius, dip_spiral_radius, z_wipe_travel_raw,
                 dip_entry_radius, total_dip_entries, remove_drops_enabled,
                 dip_shake_distance, z_global_offset_val,
                 z_safe_raw, z_safe_dip_raw):

        self.feed_rate = feed_rate
        self.feed_rate_dip = feed_rate_dip # Slower feed rate for spiral/dipping
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.gcode = []
        self.z_global_offset = z_global_offset_val

        # Z-heights, adjusted by the global offset
        self.z_safe = z_safe_raw + self.z_global_offset
        self.z_safe_dip = z_safe_dip_raw + self.z_global_offset
        self.z_wipe_travel = z_wipe_travel_raw + self.z_global_offset

        # Dipping location, Z-coordinate adjusted by the global Z-offset
        self.dip_location = (dip_location_raw[0], dip_location_raw[1], dip_location_raw[2] + self.z_global_offset)

        # Dipping and wiping parameters - DISTANCE BASED
        self.max_paint_distance_mm = max_paint_distance_mm  # Maximum distance before dipping
        self.dip_wipe_radius = dip_wipe_radius
        self.dip_spiral_radius = dip_spiral_radius
        self.dip_entry_radius = dip_entry_radius # Acts as 'jitter'
        self.total_dip_entries = total_dip_entries
        self.remove_drops_enabled = remove_drops_enabled
        self.dip_shake_distance = dip_shake_distance

        # Internal state
        self._initial_dip_performed = False
        self.dip_count = 0
        self.current_paint_distance = 0.0  # Track painting distance

    def _perform_dip(self, target_x, target_y):
        """
        Performs the dipping sequence:
        1. Travel to dip location (diagonal entry).
        2. Spiral mix in paint.
        3. Lift and wipe towards the target.
        4. Travel to target location (diagonal exit).
        Note: Dwell time removed as requested.
        """
        dip_x, dip_y, dip_z = self.dip_location
        
        # --- 1. DIAGONAL ENTRY INTO DIP STATION ---
        # Random jitter for entry point
        j_x = random.uniform(-self.dip_entry_radius, self.dip_entry_radius)
        j_y = random.uniform(-self.dip_entry_radius, self.dip_entry_radius)
        active_x = dip_x + j_x
        active_y = dip_y + j_y

        # Move rapidly to safe Z above dip location (Diagonal-ish move if supported, otherwise rapid XYZ)
        self.gcode.append(f"G0 X{active_x:.3f} Y{active_y:.3f} Z{self.z_safe_dip:.3f}")

        # --- 2. SPIRAL MIXING ---
        # Plunge into paint
        self.gcode.append(f"G1 Z{dip_z:.3f} F{self.feed_rate_dip}")
        
        # Alternate direction based on dip count
        direction = 1 if (self.dip_count % 2 == 0) else -1
        self.dip_count += 1
        
        # Generate spiral path
        theta = 0
        max_theta = 2.5 * math.pi
        step_theta = 0.1
        
        while theta <= max_theta:
            # Radius grows as theta grows
            r = (theta / max_theta) * self.dip_spiral_radius
            sx = active_x + r * math.cos(theta * direction)
            sy = active_y + r * math.sin(theta * direction)
            self.gcode.append(f"G1 X{sx:.3f} Y{sy:.3f}")
            theta += step_theta
            
        # DIP DURATION REMOVED - no dwell command

        # --- 3. DIAGONAL EXIT AND WIPE ---
        # Lift vertical to safe dip height
        self.gcode.append(f"G0 Z{self.z_safe_dip:.3f}")
        
        # Calculate wipe position: Edge of Petri dish in the direction of the target
        angle = math.atan2(target_y - dip_y, target_x - dip_x)
        wipe_x = dip_x + self.dip_wipe_radius * math.cos(angle)
        wipe_y = dip_y + self.dip_wipe_radius * math.sin(angle)
        
        # Move to wipe position at safe height
        self.gcode.append(f"G0 X{wipe_x:.3f} Y{wipe_y:.3f} Z{self.z_safe_dip:.3f}")
        
        # --- 4. TRAVEL TO TARGET ---
        # Diagonal descent: Move directly to target X,Y at Z_safe (low safe height)
        # Reset feed rate for travel
        self.gcode.append(f"G0 F{self.feed_rate}")
        self.gcode.append(f"G0 X{target_x:.3f} Y{target_y:.3f} Z{self.z_safe:.3f}")
        
        # Reset paint distance counter after dipping
        self.current_paint_distance = 0.0

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write("\n".join(self.gcode))
        print(f"G-code saved to {filename}")

    def _initial_setup(self, start_x=None, start_y=None):
        self.gcode.append("G90 ; Set Absolute Positioning")
        self.gcode.append("G21 ; Set Units to Millimeters")
        self.gcode.append(f"G0 Z{self.z_safe:.2f} ; Lift to general safe height")

        if self.max_paint_distance_mm > 0 and not self._initial_dip_performed:
            if start_x is not None and start_y is not None:
                print("Performing initial brush dip before starting.")
                self._perform_dip(start_x, start_y)
                self._initial_dip_performed = True

class SkeletonGCodeGenerator(GCodeBaseGenerator):
    """
    Generates G-code from an image using skeletonization and variable width strokes.
    Inherits common G-code logic and Z-offset handling from GCodeBaseGenerator.
    Dipping triggered by painting distance instead of fixed number of dips.
    """
    def __init__(self, z_paint_max_raw, z_paint_min_raw, max_width_mm, min_path_length_px, smooth_window_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.z_paint_max = z_paint_max_raw + self.z_global_offset
        self.z_paint_min = z_paint_min_raw + self.z_global_offset
        self.max_width_mm = max(max_width_mm, 0.001)
        self.min_path_length_px = min_path_length_px
        self.smooth_window_size = smooth_window_size

    def _width_to_z(self, width):
        width = min(width, self.max_width_mm)
        return self.z_paint_max + (width / self.max_width_mm) * (self.z_paint_min - self.z_paint_max)

    def _smooth_path(self, path):
        if len(path) < self.smooth_window_size:
            return path
        smoothed_path_new = []
        for i in range(len(path)):
            start_idx = max(0, i - self.smooth_window_size // 2)
            end_idx = min(len(path), i + self.smooth_window_size // 2 + 1)
            window_points = path[start_idx:end_idx]
            if not window_points:
                smoothed_path_new.append(path[i])
                continue
            avg_x = sum(p[0] for p in window_points) / len(window_points)
            avg_y = sum(p[1] for p in window_points) / len(window_points)
            avg_width = sum(p[2] for p in window_points) / len(window_points)
            smoothed_path_new.append((avg_x, avg_y, avg_width))
        return smoothed_path_new

    def _process_image_for_skeleton(self, image_path, target_w_mm=None, target_h_mm=None):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        image = cv2.flip(image, 0)
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        binary_image = remove_small_objects(binary_image.astype(bool), min_size=5).astype(np.uint8) * 255
        skeleton = skeletonize(binary_image // 255).astype(np.uint8)
        distance_map = distance_transform_edt(binary_image)
        graph = sknw.build_sknw(skeleton, multi=False)
        scaled_toolpaths = []
        max_scaled_width = 0
        skel_coords = np.argwhere(skeleton > 0)
        if skel_coords.size == 0:
            print("No skeleton found in the image.")
            return [], 0
        y_coords_skel, x_coords_skel = skel_coords[:, 0], skel_coords[:, 1]
        x_min_skel, y_min_skel = x_coords_skel.min(), y_coords_skel.min()
        x_max_skel, y_max_skel = x_coords_skel.max(), y_coords_skel.max()
        original_w_px = x_max_skel - x_min_skel + 1
        original_h_px = y_max_skel - y_min_skel + 1
        original_w_px = max(original_w_px, 1)
        original_h_px = max(original_h_px, 1)
        scale_factor = 1.0
        if target_w_mm and target_h_mm:
            scale_x = target_w_mm / original_w_px
            scale_y = target_h_mm / original_h_px
            scale_factor = min(scale_x, scale_y)
            print(f"Original px dimensions (skeleton bounds): {original_w_px}w x {original_h_px}h")
            print(f"Target mm dimensions: {target_w_mm}w x {target_h_mm}h")
            print(f"Using scale factor: {scale_factor:.4f}")
        for (s, e) in graph.edges():
            coords = graph[s][e]['pts']
            if len(coords) < self.min_path_length_px:
                continue
            path = []
            for p_idx in range(len(coords)):
                y, x = coords[p_idx]
                pixel_width = distance_map[y, x] * 2
                scaled_x = (x - x_min_skel) * scale_factor
                scaled_y = (y - y_min_skel) * scale_factor
                scaled_width = pixel_width * scale_factor
                if scaled_width > max_scaled_width:
                    max_scaled_width = scaled_width
                path.append((scaled_x, scaled_y, scaled_width))
            scaled_toolpaths.append(path)
        return scaled_toolpaths, max_scaled_width

    def generate_from_image(self, image_path, target_w_mm, target_h_mm):
        print(f"Processing image for skeleton method: {image_path}")
        scaled_toolpaths, max_brush_width = self._process_image_for_skeleton(
            image_path, target_w_mm, target_h_mm
        )
        if not scaled_toolpaths:
            print("No significant strokes found in the image. Exiting.")
            return

        # Prepare initial setup with the first point of the first path
        first_path = scaled_toolpaths[0]
        start_x_initial = first_path[0][0] + self.x_offset
        start_y_initial = first_path[0][1] + self.y_offset
        self._initial_setup(start_x=start_x_initial, start_y=start_y_initial)

        if self.smooth_window_size > 1:
            smoothed_final_toolpaths = []
            for path in scaled_toolpaths:
                smoothed_final_toolpaths.append(self._smooth_path(path))
            scaled_toolpaths = smoothed_final_toolpaths
            print(f"Applied smoothing with window size: {self.smooth_window_size}")
        
        self.max_width_mm = max(max_brush_width, 0.001)
        print(f"Generating G-code with estimated max brush width: {self.max_width_mm:.2f}mm")
        
        total_strokes = len(scaled_toolpaths)
        if self.max_paint_distance_mm > 0:
            print(f"Distance-based dipping enabled: Will dip every {self.max_paint_distance_mm:.1f}mm of painting.")
        else:
            print("Distance-based dipping disabled (max_paint_distance_mm = 0).")
        
        # Track distance and dip state
        need_dip = False
        next_stroke_start = None
        
        for i, path in enumerate(scaled_toolpaths):
            if not path:
                continue

            # Calculate start position of this stroke
            start_x_orig, start_y_orig, start_width = path[0]
            start_x = start_x_orig + self.x_offset
            start_y = start_y_orig + self.y_offset

            # Check if we need to dip BEFORE starting this stroke
            if need_dip and self.max_paint_distance_mm > 0:
                # Perform dip before starting this new stroke
                self._perform_dip(start_x, start_y)
                need_dip = False
            else:
                # Normal travel to start of stroke
                self.gcode.append(f"G0 X{start_x:.2f} Y{start_y:.2f} Z{self.z_safe:.2f}")

            # Painting Sequence - paint the entire stroke
            start_z = self._width_to_z(start_width)
            self.gcode.append(f"G1 Z{start_z:.2f} F{self.feed_rate / 2} ; Lower brush to start painting Z")
            
            # Set initial position for this stroke
            prev_x, prev_y = start_x, start_y
            
            # Calculate total distance for this stroke first
            for j, (x_orig, y_orig, width) in enumerate(path):
                x = x_orig + self.x_offset
                y = y_orig + self.y_offset
                z = self._width_to_z(width)
                
                # Calculate distance for this segment (painting distance only)
                if j > 0:  # Skip first point since we're already there
                    segment_distance = math.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                    self.current_paint_distance += segment_distance
                
                # Paint the segment
                self.gcode.append(f"G1 X{x:.2f} Y{y:.2f} Z{z:.2f} F{self.feed_rate} ; Paint stroke segment")
                prev_x, prev_y = x, y
            
            # After finishing the stroke, check if we exceeded threshold during it
            if self.max_paint_distance_mm > 0 and self.current_paint_distance >= self.max_paint_distance_mm:
                need_dip = True  # Flag that we need to dip before next stroke
            
            self.gcode.append(f"G0 Z{self.z_safe:.2f} ; Lift brush after stroke (to Z_safe)")
            
        final_dip_count = 1 if self._initial_dip_performed else 0
        final_dip_count += self.dip_count - (1 if self._initial_dip_performed else 0)
        total_paint_distance = self.current_paint_distance
        print(f"G-code generation complete. Performed {final_dip_count} dips total (including initial dip).")
        print(f"Total painting distance: {total_paint_distance:.2f}mm")
        self.gcode.append(f"G0 X{self.x_offset:.2f} Y{self.y_offset:.2f} Z{self.z_safe_dip:.2f} ; Return to offset origin at Z_safe_dip")
        self.gcode.append("M2 ; End of program")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate G-code for CNC painting using skeleton method with distance-based dipping.")
    parser.add_argument("input_image", help="Path to the input image file.")
    parser.add_argument("output_gcode", help="Path to the output G-code file.")
    parser.add_argument("--width", type=float, default=130.0, help="Target width of the final artwork in mm.")
    parser.add_argument("--height", type=float, default=None, help="Target height of the final artwork in mm. If not set, equals --width.")
    parser.add_argument("--feed_rate", type=int, default=1500, help="Feed rate for travel moves in mm/min.")
    parser.add_argument("--feed_rate_dip", type=int, default=600, help="Feed rate for dipping spiral moves in mm/min.")
    parser.add_argument("--x_offset", type=float, default=50.0, help="Global X offset for the painting in mm.")
    parser.add_argument("--y_offset", type=float, default=50.0, help="Global Y offset for the painting in mm.")
    parser.add_argument("--z_safe", type=float, default=1.6, help="Safe Z height for rapid moves over the artwork (mm, before offset). Corresponds to z_low.")
    parser.add_argument("--z_paint_max", type=float, default=0.0, help="Z height for the thinnest stroke (mm, before offset).")
    parser.add_argument("--z_paint_min", type=float, default=-3.0, help="Z height for the widest stroke (mm, before offset).")
    parser.add_argument("--z_safe_dip", type=float, default=16.0, help="Higher safe Z for moves to/from the dip location (mm, before offset). Corresponds to z_high.")
    parser.add_argument("--z_global_offset", type=float, default=0, help="Global Z offset to add to all Z coordinates.")
    parser.add_argument("--dip_x", type=float, default=30.0, help="X coordinate of the brush dipping location in mm.")
    parser.add_argument("--dip_y", type=float, default=25.0, help="Y coordinate of the brush dipping location in mm.")
    parser.add_argument("--dip_z", type=float, default=1.2, help="Z coordinate (depth) for brush dipping in mm (before offset).")
    parser.add_argument("--dip_interval_mm", type=float, default=100.0, help="Maximum painting distance (mm) before triggering a dip. Set to 0 to disable distance-based dipping.")
    parser.add_argument("--dip_wipe_radius", type=float, default=27.0, help="Radius of circular wipe (mm).")
    parser.add_argument("--dip_spiral_radius", type=float, default=15.0, help="Radius of the mixing spiral inside the paint container (mm).")
    parser.add_argument("--z_wipe_travel", type=float, default=7.0, help="Unused in new logic, kept for compatibility.")
    parser.add_argument("--dip_entry_radius", type=float, default=5.0, help="Jitter radius for random dip entry points (mm).")
    parser.add_argument("--total_dip_entries", type=int, default=0, help="Unused in new logic.")
    parser.add_argument("--remove_drops_enabled", type=eval, default=True, choices=[True, False], help="Enable brush wiping.")
    parser.add_argument("--dip_shake_distance", type=float, default=0, help="Unused in new logic.")
    parser.add_argument("--max_brush_width", type=float, default=3.0, help="[Skeleton only] Max brush width to map Z-depth from (mm).")
    parser.add_argument("--min_path_length_px", type=int, default=2, help="[Skeleton only] Minimum length of a skeleton segment in pixels to be considered a path.")
    parser.add_argument("--smooth_window_size", type=int, default=2, help="[Skeleton only] Window size for path smoothing.")

    args = parser.parse_args()
    if args.height is None:
        args.height = args.width

    generator_kwargs = {
        'feed_rate': args.feed_rate,
        'feed_rate_dip': args.feed_rate_dip,
        'x_offset': args.x_offset,
        'y_offset': args.y_offset,
        'dip_location_raw': (args.dip_x, args.dip_y, args.dip_z),
        'max_paint_distance_mm': args.dip_interval_mm,
        'dip_wipe_radius': args.dip_wipe_radius,
        'dip_spiral_radius': args.dip_spiral_radius,
        'z_wipe_travel_raw': args.z_wipe_travel,
        'dip_entry_radius': args.dip_entry_radius,
        'total_dip_entries': args.total_dip_entries,
        'remove_drops_enabled': args.remove_drops_enabled,
        'dip_shake_distance': args.dip_shake_distance,
        'z_global_offset_val': args.z_global_offset,
        'z_safe_raw': args.z_safe,
        'z_safe_dip_raw': args.z_safe_dip
    }

    gcode_generator = SkeletonGCodeGenerator(
        z_paint_max_raw=args.z_paint_max,
        z_paint_min_raw=args.z_paint_min,
        max_width_mm=args.max_brush_width,
        min_path_length_px=args.min_path_length_px,
        smooth_window_size=args.smooth_window_size,
        **generator_kwargs
    )

    try:
        gcode_generator.generate_from_image(image_path=args.input_image,
                                            target_w_mm=args.width,
                                            target_h_mm=args.height)
        gcode_generator.save(args.output_gcode)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
