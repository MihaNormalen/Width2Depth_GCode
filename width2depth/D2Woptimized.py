import math
import random
import argparse
import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
import sknw
from scipy.ndimage import distance_transform_edt
import traceback

class GCodeBaseGenerator:
    def __init__(self, feed_rate, x_offset, y_offset,
                 dip_location_raw, dip_duration_s,
                 dip_wipe_radius, z_wipe_travel_raw,
                 dip_entry_radius, remove_drops_enabled,
                 z_global_offset_val, z_safe_raw, z_safe_dip_raw):

        self.feed_rate = feed_rate
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.gcode = []
        self.z_global_offset = z_global_offset_val

        self.z_safe = z_safe_raw + self.z_global_offset
        self.z_safe_dip = z_safe_dip_raw + self.z_global_offset
        self.z_wipe_travel = z_wipe_travel_raw + self.z_global_offset

        self.dip_location = (dip_location_raw[0], dip_location_raw[1], dip_location_raw[2] + self.z_global_offset)
        self.dip_duration_s = dip_duration_s
        self.dip_wipe_radius = dip_wipe_radius
        self.dip_entry_radius = dip_entry_radius
        self.remove_drops_enabled = remove_drops_enabled

        self.remove_drops_lift = self.z_wipe_travel
        self.tray_enter_radius = self.dip_entry_radius
        self.remove_drops_radius = self.dip_wipe_radius
        self.offset_x = self.x_offset
        self.offset_y = self.y_offset
        self.remove_drops_angle_variation_deg = 12.0

        self._initial_dip_performed = False

    def remove_drops(self, tray_x, tray_y, x, y):
        target_x = x
        target_y = y

        dx = target_x - tray_x
        dy = target_y - tray_y
        dist = math.hypot(dx, dy)
        base_angle = math.atan2(dy, dx) if dist != 0 else 0.0

        max_variation_rad = math.radians(self.remove_drops_angle_variation_deg)
        angle = base_angle + random.uniform(-max_variation_rad, max_variation_rad)

        ux, uy = math.cos(angle), math.sin(angle)

        x1, y1 = tray_x + ux * self.tray_enter_radius, tray_y + uy * self.tray_enter_radius
        x2, y2 = tray_x + ux * self.remove_drops_radius, tray_y + uy * self.remove_drops_radius

        self.gcode.append(f"; remove_drops start -> dir_angle={math.degrees(angle):.1f}deg")
        self.gcode.append(f"G0 X{x1:.3f} Y{y1:.3f}")
        self.gcode.append(f"G0 Z{self.remove_drops_lift:.3f}")
        self.gcode.append("G1 F600")
        self.gcode.append(f"G1 X{x2:.3f} Y{y2:.3f}")
        self.gcode.append(f"G0 F{self.feed_rate}")

    def _perform_dip(self, target_x=None, target_y=None):
        """Spiralno namakanje po zgledu dotsSVGkrogci.py."""
        self.gcode.append(f"\n; --- Spiralno Namakanje ---")
        self.gcode.append(f"G0 Z{self.z_safe_dip:.3f}")
        self.gcode.append(f"G0 X{self.dip_location[0]:.3f} Y{self.dip_location[1]:.3f}")
        self.gcode.append(f"G1 Z{self.dip_location[2]:.3f} F800")
        
        d_theta = 0.2
        max_theta = 2 * math.pi
        theta = 0
        while theta <= max_theta:
            r = (theta / max_theta) * self.dip_entry_radius
            mx = self.dip_location[0] + r * math.cos(theta)
            my = self.dip_location[1] + r * math.sin(theta)
            self.gcode.append(f"G1 X{mx:.3f} Y{my:.3f} F1000")
            theta += d_theta

        if self.dip_duration_s > 0:
            self.gcode.append(f"G4 P{int(self.dip_duration_s * 1000)}")

        wipe_z = self.dip_location[2] + 2.0
        self.gcode.append(f"G1 Z{wipe_z:.3f} F800")

        if self.remove_drops_enabled and target_x is not None:
            self.remove_drops(self.dip_location[0], self.dip_location[1], target_x, target_y)

        self.gcode.append(f"G1 Z{self.z_safe_dip:.3f} F600")
        self.gcode.append(f"G0 F{self.feed_rate}")

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write("\n".join(self.gcode))
        print(f"G-code saved to {filename}")

    def _initial_setup(self):
        self.gcode.append("G90 ; Set Absolute Positioning")
        self.gcode.append("G21 ; Set Units to Millimeters")
        self.gcode.append(f"G0 Z{self.z_safe:.3f}")

class SkeletonGCodeGenerator(GCodeBaseGenerator):
    def __init__(self, z_paint_max_raw, z_paint_min_raw, max_width_mm,
                 min_path_length_px, smooth_window_size,
                 dip_distance_threshold, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.z_paint_max = z_paint_max_raw + self.z_global_offset
        self.z_paint_min = z_paint_min_raw + self.z_global_offset
        self.max_width_mm = max(max_width_mm, 0.001)
        self.min_path_length_px = min_path_length_px
        self.smooth_window_size = smooth_window_size
        self.dip_distance_threshold = dip_distance_threshold

    def _width_to_z(self, width):
        width = min(width, self.max_width_mm)
        return self.z_paint_max + (width / self.max_width_mm) * (self.z_paint_min - self.z_paint_max)

    def _smooth_path(self, path):
        if len(path) < self.smooth_window_size: return path
        smoothed = []
        for i in range(len(path)):
            start_idx = max(0, i - self.smooth_window_size // 2)
            end_idx = min(len(path), i + self.smooth_window_size // 2 + 1)
            window = path[start_idx:end_idx]
            avg_x = sum(p[0] for p in window) / len(window)
            avg_y = sum(p[1] for p in window) / len(window)
            avg_w = sum(p[2] for p in window) / len(window)
            smoothed.append((avg_x, avg_y, avg_w))
        return smoothed

    def _process_image_for_skeleton(self, image_path, target_w_mm=None, target_h_mm=None):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None: raise FileNotFoundError(f"Image not found at {image_path}")
        image = cv2.flip(image, 0)
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        binary_image = remove_small_objects(binary_image.astype(bool), min_size=5).astype(np.uint8) * 255
        skeleton = skeletonize(binary_image // 255).astype(np.uint8)
        distance_map = distance_transform_edt(binary_image)
        graph = sknw.build_sknw(skeleton, multi=False)
        
        scaled_toolpaths = []
        max_scaled_width = 0
        skel_coords = np.argwhere(skeleton > 0)
        if skel_coords.size == 0: return [], 0
        
        y_min_skel, x_min_skel = skel_coords.min(axis=0)
        y_max_skel, x_max_skel = skel_coords.max(axis=0)
        original_w_px = max(x_max_skel - x_min_skel + 1, 1)
        original_h_px = max(y_max_skel - y_min_skel + 1, 1)
        
        scale_factor = min(target_w_mm / original_w_px, target_h_mm / original_h_px) if target_w_mm else 1.0

        for (s, e) in graph.edges():
            coords = graph[s][e]['pts']
            if len(coords) < self.min_path_length_px: continue
            path = []
            for (yy, xx) in coords:
                pixel_width = distance_map[yy, xx] * 2
                scaled_x = (xx - x_min_skel) * scale_factor
                scaled_y = (yy - y_min_skel) * scale_factor
                scaled_width = pixel_width * scale_factor
                max_scaled_width = max(max_scaled_width, scaled_width)
                path.append((scaled_x, scaled_y, scaled_width))
            scaled_toolpaths.append(path)
        return scaled_toolpaths, max_scaled_width

    def generate_from_image(self, image_path, target_w_mm, target_h_mm):
        self._initial_setup()
        paths, max_brush_width = self._process_image_for_skeleton(image_path, target_w_mm, target_h_mm)
        if not paths: return
        
        if self.smooth_window_size > 1:
            paths = [self._smooth_path(p) for p in paths]
        
        self.max_width_mm = max(max_brush_width, 0.001)
        
        # --- NAPREDNA OPTIMIZACIJA POTI (Greedy + Reversing) ---
        current_pos = (self.dip_location[0], self.dip_location[1])
        optimized_paths = []
        remaining_paths = paths
        
        print(f"Optimizing {len(remaining_paths)} strokes with path reversal...")

        while remaining_paths:
            best_dist = float('inf')
            best_idx = -1
            reverse_needed = False

            for i, path in enumerate(remaining_paths):
                # Razdalja do začetka poteze
                d_start = math.hypot(path[0][0] + self.x_offset - current_pos[0], 
                                     path[0][1] + self.y_offset - current_pos[1])
                # Razdalja do konca poteze
                d_end = math.hypot(path[-1][0] + self.x_offset - current_pos[0], 
                                   path[-1][1] + self.y_offset - current_pos[1])
                
                if d_start < best_dist:
                    best_dist = d_start
                    best_idx = i
                    reverse_needed = False
                if d_end < best_dist:
                    best_dist = d_end
                    best_idx = i
                    reverse_needed = True

            next_path = remaining_paths.pop(best_idx)
            if reverse_needed:
                next_path = next_path[::-1] # Obrni vrstni red točk v potezi
            
            # Prvo namakanje pred prvo izbrano potezo
            if not optimized_paths:
                self._perform_dip(target_x=next_path[0][0] + self.x_offset, target_y=next_path[0][1] + self.y_offset)

            optimized_paths.append(next_path)
            current_pos = (next_path[-1][0] + self.x_offset, next_path[-1][1] + self.y_offset)

        # --- GENERIRANJE G-KODE ---
        travel_since_last_dip = 0.0
        for path in optimized_paths:
            start_x = path[0][0] + self.x_offset
            start_y = path[0][1] + self.y_offset
            start_width = path[0][2]

            stroke_length = sum(math.hypot(path[i][0]-path[i-1][0], path[i][1]-path[i-1][1]) for i in range(1, len(path)))
            travel_since_last_dip += stroke_length

            if travel_since_last_dip >= self.dip_distance_threshold:
                self._perform_dip(target_x=start_x, target_y=start_y)
                travel_since_last_dip = 0.0

            self.gcode.append(f"G0 X{start_x:.3f} Y{start_y:.3f} Z{self.z_safe:.3f}")
            self.gcode.append(f"G1 Z{self._width_to_z(start_width):.3f} F{self.feed_rate / 2}")
            
            for x_orig, y_orig, width in path:
                self.gcode.append(f"G1 X{x_orig + self.x_offset:.3f} Y{y_orig + self.y_offset:.3f} Z{self._width_to_z(width):.3f} F{self.feed_rate}")
            
            self.gcode.append(f"G0 Z{self.z_safe:.3f}")

        self.gcode.append(f"G0 X{self.x_offset:.3f} Y{self.y_offset:.3f} Z{self.z_safe_dip:.3f}")
        self.gcode.append("M2")

if __name__ == "__main__":
    # Parametri so nastavljeni tako, da so čim bolj varni in optimizirani
    parser = argparse.ArgumentParser(description="Generate G-code for CNC painting (skeleton method).")
    parser.add_argument("input_image", help="Path to input image.")
    parser.add_argument("output_gcode", help="Path to output gcode file.")
    parser.add_argument("--width", type=float, default=130.0)
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument("--feed_rate", type=int, default=1200)
    parser.add_argument("--x_offset", type=float, default=0.0)
    parser.add_argument("--y_offset", type=float, default=25.0)
    parser.add_argument("--z_safe", type=float, default=2.0)
    parser.add_argument("--z_paint_max", type=float, default=0.0)
    parser.add_argument("--z_paint_min", type=float, default=-1.0) # Rahlo globlje za polnejše poteze
    parser.add_argument("--z_safe_dip", type=float, default=10.0)
    parser.add_argument("--z_global_offset", type=float, default=0.0)
    parser.add_argument("--dip_x", type=float, default=63.0)
    parser.add_argument("--dip_y", type=float, default=0.0)
    parser.add_argument("--dip_z", type=float, default=0.0)
    parser.add_argument("--dip_duration", type=float, default=0.2)
    parser.add_argument("--dip_wipe_radius", type=float, default=17.0)
    parser.add_argument("--z_wipe_travel", type=float, default=1.0)
    parser.add_argument("--dip_entry_radius", type=float, default=4.0)
    parser.add_argument("--remove_drops_enabled", type=eval, default=True)
    parser.add_argument("--max_brush_width", type=float, default=4.0)
    parser.add_argument("--min_path_length_px", type=int, default=2)
    parser.add_argument("--smooth_window_size", type=int, default=3)
    parser.add_argument("--dip_distance_threshold", type=float, default=80.0) # Namakanje na vsakih 80cm barvanja

    args = parser.parse_args()
    if args.height is None: args.height = args.width

    generator_kwargs = {
        'feed_rate': args.feed_rate, 'x_offset': args.x_offset, 'y_offset': args.y_offset,
        'dip_location_raw': (args.dip_x, args.dip_y, args.dip_z), 'dip_duration_s': args.dip_duration,
        'dip_wipe_radius': args.dip_wipe_radius, 'z_wipe_travel_raw': args.z_wipe_travel,
        'dip_entry_radius': args.dip_entry_radius, 'remove_drops_enabled': args.remove_drops_enabled,
        'z_global_offset_val': args.z_global_offset, 'z_safe_raw': args.z_safe, 'z_safe_dip_raw': args.z_safe_dip
    }

    gcode_generator = SkeletonGCodeGenerator(
        z_paint_max_raw=args.z_paint_max, z_paint_min_raw=args.z_paint_min,
        max_width_mm=args.max_brush_width, min_path_length_px=args.min_path_length_px,
        smooth_window_size=args.smooth_window_size, dip_distance_threshold=args.dip_distance_threshold,
        **generator_kwargs
    )

    gcode_generator.generate_from_image(args.input_image, args.width, args.height)
    gcode_generator.save(args.output_gcode)