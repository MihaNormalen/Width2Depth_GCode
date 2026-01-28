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
    """
    def __init__(self, feed_rate, x_offset, y_offset,
                 dip_location_raw, total_target_dips, dip_duration_s,
                 dip_wipe_radius, z_wipe_travel_raw,
                 dip_entry_radius, total_dip_entries, remove_drops_enabled,
                 dip_shake_distance, z_global_offset_val,
                 z_safe_raw, z_safe_dip_raw):

        self.feed_rate = feed_rate
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

        # Dipping and wiping parameters
        self.total_target_dips = total_target_dips
        self.dip_duration_s = dip_duration_s
        self.dip_wipe_radius = dip_wipe_radius
        self.dip_entry_radius = dip_entry_radius
        self.total_dip_entries = total_dip_entries
        self.remove_drops_enabled = remove_drops_enabled
        self.dip_shake_distance = dip_shake_distance

        # For remove_drops logic
        self.remove_drops_lift = self.z_wipe_travel
        self.tray_enter_radius = self.dip_entry_radius
        self.remove_drops_radius = self.dip_wipe_radius
        self.offset_x = self.x_offset
        self.offset_y = self.y_offset

        # Flag to ensure only one initial dip
        self._initial_dip_performed = False

    def _get_random_point_in_circle(self, center_x, center_y, radius):
        angle = random.uniform(0, 2 * math.pi)
        r = radius
        x = center_x + r * math.cos(angle)
        y = center_y + r * math.sin(angle)
        return x, y

    def remove_drops(self, tray_x, tray_y, x, y):
        """
        Povzeto iz copicograf.py:
        Premakne čopič na rob posodice za brisanje kapljic.
        """
        dist = math.hypot((tray_x - (x + self.offset_x)), (tray_y - (y + self.offset_y)))
        if dist == 0:
            dist = 0.01  # Prevent division by zero
        ratio_start = self.tray_enter_radius / dist
        ratio_end = self.remove_drops_radius / dist

        delta_x = abs(tray_x - (x + self.offset_x))
        delta_y = abs(tray_y - (y + self.offset_y))
        x_operator = -1 if tray_x > (x + self.offset_x) else 1
        y_operator = -1 if (y + self.offset_y) < tray_y else 1

        x1 = int(tray_x + (x_operator * delta_x * ratio_start))
        y1 = int(tray_y + (y_operator * delta_y * ratio_start))
        x2 = int(tray_x + (x_operator * delta_x * ratio_end))
        y2 = int(tray_y + (y_operator * delta_y * ratio_end))

        # Premik na vstopno točko
        self.gcode.append(f"G0 X{x1:.3f} Y{y1:.3f}")
        # Dvig na višino brisanja
        self.gcode.append(f"G0 Z{self.remove_drops_lift:.3f}")
        # Počasna hitrost za brisanje
        self.gcode.append("G1 F1200")
        # Premik do izhodne točke (rob posodice)
        self.gcode.append(f"G1 X{x2:.3f} Y{y2:.3f}")
        # Povratek na hitro hitrost
        self.gcode.append(f"G0 F{self.feed_rate}")

    def _perform_dip(self):
        # 1. Move to safe height above dip location
        self.gcode.append(f"G0 Z{self.z_safe_dip}")
        self.gcode.append(f"G0 X{self.dip_location[0]:.3f} Y{self.dip_location[1]:.3f}")

        # 2. Dip into paint (center of container)
        self.gcode.append(f"G1 Z{self.dip_location[2]:.3f} F1000")
        if self.dip_duration_s > 0:
            self.gcode.append(f"G4 P{int(self.dip_duration_s * 1000)}")

        # 3. Dvig na višino brisanja
        wipe_z = self.dip_location[2] + 2.0
        self.gcode.append(f"G1 Z{wipe_z:.3f} F1000")

        # 4. Brisanje čopiča na rob posodice (remove_drops logika)
        if self.remove_drops_enabled:
            # Brisanje v smeri +X (lahko dodaš naključni kot)
            self.remove_drops(
                tray_x=self.dip_location[0],
                tray_y=self.dip_location[1],
                x=self.dip_location[0] + self.dip_wipe_radius,
                y=self.dip_location[1]
            )

        # 5. Move up to safe height
        self.gcode.append(f"G1 Z{self.z_safe_dip:.3f} F2000")
        # 6. Reset feed rate to normal
        self.gcode.append(f"G0 F{self.feed_rate}")

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write("\n".join(self.gcode))
        print(f"G-code saved to {filename}")

    def _initial_setup(self):
        self.gcode.append("G90 ; Set Absolute Positioning")
        self.gcode.append("G21 ; Set Units to Millimeters")
        self.gcode.append(f"G0 Z{self.z_safe:.2f} ; Lift to general safe height")

        if self.total_target_dips > 0 and not self._initial_dip_performed:
            print("Performing initial brush dip before starting.")
            self._perform_dip()
            self.total_target_dips = max(0, self.total_target_dips - 1)
            self._initial_dip_performed = True

class SkeletonGCodeGenerator(GCodeBaseGenerator):
    """
    Generates G-code from an image using skeletonization and variable width strokes.
    Inherits common G-code logic and Z-offset handling from GCodeBaseGenerator.
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
        self._initial_setup()
        print(f"Processing image for skeleton method: {image_path}")
        scaled_toolpaths, max_brush_width = self._process_image_for_skeleton(
            image_path, target_w_mm, target_h_mm
        )
        if not scaled_toolpaths:
            print("No significant strokes found in the image. Exiting.")
            return
        if self.smooth_window_size > 1:
            smoothed_final_toolpaths = []
            for path in scaled_toolpaths:
                smoothed_final_toolpaths.append(self._smooth_path(path))
            scaled_toolpaths = smoothed_final_toolpaths
            print(f"Applied smoothing with window size: {self.smooth_window_size}")
        self.max_width_mm = max(max_brush_width, 0.001)
        print(f"Generating G-code with estimated max brush width: {self.max_width_mm:.2f}mm")
        total_strokes = len(scaled_toolpaths)
        dip_interval_strokes = 0
        if self.total_target_dips > 0 and total_strokes > 0:
            dip_interval_strokes = max(1, math.ceil(total_strokes / self.total_target_dips))
            print(f"Total strokes: {total_strokes}, Remaining target dips: {self.total_target_dips}, Dipping every ~{dip_interval_strokes} strokes.")
        else:
            print("No further dips configured after initial setup.")
        dip_counter = 0
        for i, path in enumerate(scaled_toolpaths):
            if not path:
                continue
            if dip_interval_strokes > 0 and (i + 1) % dip_interval_strokes == 0 and dip_counter < self.total_target_dips:
                self._perform_dip()
                dip_counter += 1
                start_x_orig, start_y_orig, _ = path[0]
                start_x = start_x_orig + self.x_offset
                start_y = start_y_orig + self.y_offset
                self.gcode.append(f"G0 X{start_x:.2f} Y{start_y:.2f} Z{self.z_safe_dip:.3f} ; Return to start of next stroke after dip (at Z_safe_dip)")
            start_x_orig, start_y_orig, start_width = path[0]
            start_x = start_x_orig + self.x_offset
            start_y = start_y_orig + self.y_offset
            self.gcode.append(f"G0 X{start_x:.2f} Y{start_y:.2f} Z{self.z_safe:.2f} ; Move to start of stroke (at Z_safe)")
            start_z = self._width_to_z(start_width)
            self.gcode.append(f"G1 Z{start_z:.2f} F{self.feed_rate / 2} ; Lower brush to start painting Z")
            for x_orig, y_orig, width in path:
                x = x_orig + self.x_offset
                y = y_orig + self.y_offset
                z = self._width_to_z(width)
                self.gcode.append(f"G1 X{x:.2f} Y{y:.2f} Z{z:.2f} F{self.feed_rate} ; Paint stroke segment")
            self.gcode.append(f"G0 Z{self.z_safe:.2f} ; Lift brush after stroke (to Z_safe)")
        final_dip_count = 1 if self._initial_dip_performed else 0
        final_dip_count += dip_counter
        print(f"G-code generation complete. Performed {final_dip_count} dips.")
        self.gcode.append(f"G0 X{self.x_offset:.2f} Y{self.y_offset:.2f} Z{self.z_safe_dip:.2f} ; Return to offset origin at Z_safe_dip")
        self.gcode.append("M2 ; End of program")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate G-code for CNC painting using skeleton method with global Z-offset.")
    parser.add_argument("input_image", help="Path to the input image file.")
    parser.add_argument("output_gcode", help="Path to the output G-code file.")
    parser.add_argument("--width", type=float, default=130.0, help="Target width of the final artwork in mm.")
    parser.add_argument("--height", type=float, default=None, help="Target height of the final artwork in mm. If not set, equals --width.")
    parser.add_argument("--feed_rate", type=int, default=100, help="Feed rate for painting moves in mm/min.")
    parser.add_argument("--x_offset", type=float, default=0.0, help="Global X offset for the painting in mm.")
    parser.add_argument("--y_offset", type=float, default=25.0, help="Global Y offset for the painting in mm.")
    parser.add_argument("--z_safe", type=float, default=3.0, help="Safe Z height for rapid moves over the artwork (mm, before offset).")
    parser.add_argument("--z_paint_max", type=float, default=0.0, help="Z height for the thinnest stroke (mm, before offset). This is typically the material surface.")
    parser.add_argument("--z_paint_min", type=float, default=-3.0, help="Z height for the widest stroke (mm, before offset). This is the deepest penetration.")
    parser.add_argument("--z_safe_dip", type=float, default=12.0, help="Higher safe Z for moves to/from the dip location (mm, before offset).")
    parser.add_argument("--z_global_offset", type=float, default=3, help="Global Z offset to add to all Z coordinates.")
    parser.add_argument("--dip_x", type=float, default=51.0, help="X coordinate of the brush dipping location in mm.")
    parser.add_argument("--dip_y", type=float, default=3.0, help="Y coordinate of the brush dipping location in mm.")
    parser.add_argument("--dip_z", type=float, default=3, help="Z coordinate (depth) for brush dipping in mm (before offset).")
    parser.add_argument("--total_dips", type=int, default=20, help="Total number of dips during the process. Set to 0 for no dipping.")
    parser.add_argument("--dip_duration", type=float, default=0.1, help="Duration in seconds the brush dwells in the paint.")
    parser.add_argument("--dip_wipe_radius", type=float, default=29.0, help="Radius of circular wipe (mm). Set to 0 for linear shake-off.")
    parser.add_argument("--z_wipe_travel", type=float, default=7.0, help="Z height for the wiping/shake-off motion (mm, before offset).")
    parser.add_argument("--dip_entry_radius", type=float, default=10.0, help="Radius for random dip entry points (mm).")
    parser.add_argument("--total_dip_entries", type=int, default=0, help="Number of dunks during a single dip cycle.")
    parser.add_argument("--remove_drops_enabled", type=eval, default=True, choices=[True, False], help="Enable brush wiping/shake-off.")
    parser.add_argument("--dip_shake_distance", type=float, default=0, help="Distance for linear shake-off motion (mm). Set to 0 to disable linear shake and use wipe radius if defined.")
    parser.add_argument("--max_brush_width", type=float, default=3.0, help="[Skeleton only] Max brush width to map Z-depth from (mm).")
    parser.add_argument("--min_path_length_px", type=int, default=2, help="[Skeleton only] Minimum length of a skeleton segment in pixels to be considered a path.")
    parser.add_argument("--smooth_window_size", type=int, default=2, help="[Skeleton only] Window size for path smoothing. Set to 1 for no smoothing.")

    args = parser.parse_args()
    if args.height is None:
        args.height = args.width

    generator_kwargs = {
        'feed_rate': args.feed_rate,
        'x_offset': args.x_offset,
        'y_offset': args.y_offset,
        'dip_location_raw': (args.dip_x, args.dip_y, args.dip_z),
        'total_target_dips': args.total_dips,
        'dip_duration_s': args.dip_duration,
        'dip_wipe_radius': args.dip_wipe_radius,
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
