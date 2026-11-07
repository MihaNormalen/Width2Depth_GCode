import math
import random
import argparse
import cv2
import numpy as np
import re
from skimage.morphology import skeletonize, remove_small_objects
import sknw
from scipy.ndimage import distance_transform_edt
import traceback

class GCodeBaseGenerator:
    def __init__(self, feed_rate, x_offset, y_offset,
                 dip_location_raw, dip_duration_s,
                 dip_wipe_radius, z_wipe_travel_raw,
                 dip_entry_radius, remove_drops_enabled,
                 z_global_offset_val, z_safe_raw, z_safe_dip_raw,
                 feed_paint=None, accel_travel=12000, accel_paint=1000,
                 water_dish=None):

        self.feed_rate    = feed_rate                                   # travel feed
        self.feed_paint   = feed_paint if feed_paint is not None else feed_rate
        self.accel_travel = accel_travel
        self.accel_paint  = accel_paint
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.gcode = []
        self.z_global_offset = z_global_offset_val

        self.z_safe = z_safe_raw + self.z_global_offset
        self.z_safe_dip = z_safe_dip_raw + self.z_global_offset
        self.z_wipe_travel = z_wipe_travel_raw + self.z_global_offset

        self.dip_location = (
            dip_location_raw[0],
            dip_location_raw[1],
            dip_location_raw[2] + self.z_global_offset
        )
        self.dip_duration_s = dip_duration_s
        self.dip_wipe_radius = dip_wipe_radius
        self.dip_entry_radius = dip_entry_radius
        self.remove_drops_enabled = remove_drops_enabled

        self.water_dish = water_dish 

        # Spiral dip parameters
        self.dip_jitter = 0.0
        self.dip_spiral_loops = 1.0
        self.dip_spiral_r = 0.0

        self._initial_dip_performed = False

        self.tf_enabled      = False
        self.tf_dishes       = []
        self.tf_min_dist     = 40.0
        self.tf_max_dist     = 80.0
        self.tf_no_repeat    = True
        self._tf_current_idx = -1   

    def _handle_move_g0(self, x, y, z=None):
        """Apply canvas offset and output G0 move."""
        abs_x = x + self.x_offset
        abs_y = y + self.y_offset
        out = f"G0 X{abs_x:.3f} Y{abs_y:.3f}"
        if z is not None:
            out += f" Z{z:.3f}"
        self.gcode.append(out)

    def _handle_move_g1(self, x, y, z=None, feed_rate=None):
        """Apply canvas offset and output G1 move."""
        abs_x = x + self.x_offset
        abs_y = y + self.y_offset
        out = f"G1 X{abs_x:.3f} Y{abs_y:.3f}"
        if z is not None:
            out += f" Z{z:.3f}"
        if feed_rate is not None:
            out += f" F{int(feed_rate)}"
        self.gcode.append(out)

    def _perform_clean(self):
        """
        Cleans the brush in the water dish before moving to a new color.
        Uses identical Safe Y corridor approach/departure and jitter as _perform_dip.
        """
        if not self.water_dish:
            self.gcode.append("; --- WARNING: No water dish defined, skipping clean ---")
            return

        self.gcode.append(f"\n; --- CIKEL CISCENJA (Voda) ---")
        
        dip_abs_x = self.water_dish['x']
        dip_abs_y = self.water_dish['y']
        dip_z     = self.water_dish['z']
        
        ax = dip_abs_x + random.uniform(-self.dip_jitter, self.dip_jitter)
        ay = dip_abs_y + random.uniform(-self.dip_jitter, self.dip_jitter)

        # 1. Lift to z_safe at current position, switch to travel dynamics
        self.gcode.append(f"G0 Z{self.z_safe:.3f} F3000")
        self._set_speed('travel')

        # 2. ORTHOGONAL SAFE APPROACH (Guarantees safe Y before traversing in X)
        self.gcode.append(f"G0 Z{self.z_safe_dip:.3f} F3000")
        # Pull out to safe Y corridor first
        self.gcode.append(f"G0 Y{dip_abs_y + self.dip_wipe_radius:.3f}")
        # Move horizontally to water dish X
        self.gcode.append(f"G0 X{ax:.3f}")
        # Now drop into water dish center in Y
        self.gcode.append(f"G0 Y{ay:.3f}")

        # 3. Plunge into water
        self.gcode.append(f"G1 Z{dip_z:.3f} F3000")
        
        # 4. Stir outward in water to clean
        num_steps = int(self.dip_spiral_loops * 4)
        for i in range(num_steps):
            ang = i * (math.pi / 2)
            r = (i / num_steps) * self.dip_spiral_r if num_steps > 0 else 0.0
            self.gcode.append(
                f"G1 X{ax + r*math.cos(ang):.3f} Y{ay + r*math.sin(ang):.3f} F2500"
            )
            
        # 5. SCRAPE ON EDGE TO REMOVE WATER & REACH SAFE Y
        self.gcode.append(f"G1 Z{self.z_wipe_travel:.3f} F1500")
        self.gcode.append(f"G1 X{dip_abs_x:.3f} Y{dip_abs_y + self.dip_wipe_radius:.3f} F1500")

        # 6. Lift to safe height after cleaning (Now at safe Y corridor)
        self.gcode.append(f"G0 Z{self.z_safe_dip:.3f} F3000")
        
        self.gcode.append("; --- CISCENJE END ---")

    def _perform_dip(self, target_x=None, target_y=None):
        """
        Full dip cycle — handles Safe Y approach and Safe Y departure corridor moves 
        to ensure no glass rims are clipped and no other dishes are crossed.
        """
        # ── Pick dish (absolute machine coords) ──────────────────────────────
        if self.tf_enabled and self.tf_dishes:
            pool = list(range(len(self.tf_dishes)))
            if self.tf_no_repeat and len(pool) > 1 and self._tf_current_idx >= 0:
                pool = [i for i in pool if i != self._tf_current_idx]
            picked_idx = random.choice(pool)
            
            # Trigger Cleaning if color changes
            if self._tf_current_idx >= 0 and picked_idx != self._tf_current_idx:
                self._perform_clean()
                
            self._tf_current_idx = picked_idx
            dish = self.tf_dishes[picked_idx]
            dip_abs_x = dish['x']
            dip_abs_y = dish['y']
            dip_z     = dish['z']
            self.gcode.append(f"\n; --- CIKEL NAMAKANJA -> {dish['name']} (tutti frutti) ---")
        else:
            dip_abs_x = self.dip_location[0]
            dip_abs_y = self.dip_location[1]
            dip_z     = self.dip_location[2]
            self.gcode.append("\n; --- CIKEL NAMAKANJA ---")

        # Target in absolute machine coords (canvas offset baked in)
        if target_x is not None:
            tgt_abs_x = target_x + self.x_offset
            tgt_abs_y = target_y + self.y_offset
        else:
            tgt_abs_x = dip_abs_x
            tgt_abs_y = dip_abs_y + self.dip_wipe_radius

        ax = dip_abs_x + random.uniform(-self.dip_jitter, self.dip_jitter)
        ay = dip_abs_y + random.uniform(-self.dip_jitter, self.dip_jitter)

        # 1. Lift to z_safe at current position, switch to travel dynamics
        self.gcode.append(f"G0 Z{self.z_safe:.3f} F3000")
        self._set_speed('travel')

        # 2. ORTHOGONAL SAFE APPROACH (Guarantees safe Y before traversing in X)
        self.gcode.append(f"G0 Z{self.z_safe_dip:.3f} F3000")
        # Pull out to safe Y corridor first
        self.gcode.append(f"G0 Y{dip_abs_y + self.dip_wipe_radius:.3f}")
        # Move horizontally to dish X
        self.gcode.append(f"G0 X{ax:.3f}")
        # Now drop into dish center in Y
        self.gcode.append(f"G0 Y{ay:.3f}")

        # 3. Plunge to dip depth
        self.gcode.append(f"G1 Z{dip_z:.3f} F3000")

        # 4. Spiral outward while submerged
        num_steps = int(self.dip_spiral_loops * 4)
        for i in range(num_steps):
            ang = i * (math.pi / 2)
            r   = (i / num_steps) * self.dip_spiral_r if num_steps > 0 else 0.0
            self.gcode.append(
                f"G1 X{ax + r*math.cos(ang):.3f} Y{ay + r*math.sin(ang):.3f} F2500"
            )

        # 5. SCRAPE ON EDGE TO REMOVE PAINT & REACH SAFE Y
        self.gcode.append(f"G1 Z{self.z_wipe_travel:.3f} F1500")
        self.gcode.append(f"G1 X{dip_abs_x:.3f} Y{dip_abs_y + self.dip_wipe_radius:.3f} F1500")
        self.gcode.append(f"G0 Z{self.z_safe_dip:.3f} F3000")

        # 6. ORTHOGONAL SAFE DEPARTURE (Travel to canvas via corridor)
        # Move in X to target X while staying on safe Y corridor
        self.gcode.append(f"G0 X{tgt_abs_x:.3f}")
        # Now move down in Y and Z to the Canvas
        self.gcode.append(f"G0 Y{tgt_abs_y:.3f} Z{self.z_safe:.3f}")

        self.gcode.append("; --- DIP END ---")

    def _set_speed(self, mode='travel'):
        accel = self.accel_travel if mode == 'travel' else self.accel_paint
        feed  = self.feed_rate    if mode == 'travel' else self.feed_paint
        self.gcode.append("M400")
        self.gcode.append(f"M204 P{accel} T{accel}")
        if mode == 'travel':
            self.gcode.append(f"G0 F{feed}")
        else:
            self.gcode.append(f"G1 F{feed}")

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write("\n".join(self.gcode))
        print(f"G-code saved to {filename}")

    def _initial_setup(self):
        self.gcode.append(f"G90 ; Set Absolute Positioning")
        self.gcode.append("G21 ; Set Units to Millimeters")
        self.gcode.append(f"G0 X{self.x_offset:.3f} Y{self.y_offset:.3f} Z{self.z_safe:.3f} F3000")

        if not self._initial_dip_performed:
            print("Performing initial brush dip before starting.")
            self._perform_dip()
            self._initial_dip_performed = True


class SkeletonGCodeGenerator(GCodeBaseGenerator):
    def __init__(self, z_paint_max_raw, z_paint_min_raw, max_width_mm,
                 min_path_length_px, smooth_window_size,
                 dip_distance_threshold, path_optimize,
                 tf_enabled=False, tf_dishes=None, tf_min_dist=40.0,  
                 tf_max_dist=80.0, tf_no_repeat=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.z_paint_max = z_paint_max_raw + self.z_global_offset
        self.z_paint_min = z_paint_min_raw + self.z_global_offset
        self.max_width_mm = max(max_width_mm, 0.001)
        self.min_path_length_px = min_path_length_px
        self.smooth_window_size = smooth_window_size
        self.dip_distance_threshold = dip_distance_threshold
        self.path_optimize = path_optimize
        self.tf_enabled      = tf_enabled
        self.tf_dishes       = tf_dishes or []
        self.tf_min_dist     = tf_min_dist
        self.tf_max_dist     = tf_max_dist
        self.tf_no_repeat    = tf_no_repeat

    def _width_to_z(self, width):
        width = min(width, self.max_width_mm)
        return self.z_paint_max + (width / self.max_width_mm) * (self.z_paint_min - self.z_paint_max)

    def _smooth_path(self, path):
        if len(path) < self.smooth_window_size:
            return path
        smoothed = []
        for i in range(len(path)):
            start_idx = max(0, i - self.smooth_window_size // 2)
            end_idx = min(len(path), i + self.smooth_window_size // 2 + 1)
            window = path[start_idx:end_idx]
            if not window:
                smoothed.append(path[i])
                continue
            smoothed.append((
                sum(p[0] for p in window) / len(window),
                sum(p[1] for p in window) / len(window),
                sum(p[2] for p in window) / len(window),
            ))
        return smoothed

    def _optimize_stroke_order(self, paths):
        if not paths:
            return paths
        remaining = list(paths)
        optimized = []
        current_pos = (0.0, 0.0)
        while remaining:
            best_idx = 0
            best_dist = float('inf')
            best_reversed = False
            for i, path in enumerate(remaining):
                sx, sy = path[0][0], path[0][1]
                ex, ey = path[-1][0], path[-1][1]
                d_fwd = math.hypot(sx - current_pos[0], sy - current_pos[1])
                d_rev = math.hypot(ex - current_pos[0], ey - current_pos[1])
                if d_fwd < best_dist:
                    best_dist, best_idx, best_reversed = d_fwd, i, False
                if d_rev < best_dist:
                    best_dist, best_idx, best_reversed = d_rev, i, True
            path = remaining.pop(best_idx)
            if best_reversed:
                path = list(reversed(path))
            optimized.append(path)
            current_pos = (path[-1][0], path[-1][1])
        return optimized

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
        original_w_px = max(x_max_skel - x_min_skel + 1, 1)
        original_h_px = max(y_max_skel - y_min_skel + 1, 1)
        
        # Proportional scale logic
        if target_w_mm and target_h_mm:
            scale_x = target_w_mm / original_w_px
            scale_y = target_h_mm / original_h_px
            scale_factor = min(scale_x, scale_y)
            print(f"Original px dims: {original_w_px}x{original_h_px}, target mm: {target_w_mm}x{target_h_mm}, scale: {scale_factor:.4f}")
        elif target_w_mm:
            scale_factor = target_w_mm / original_w_px
            print(f"Original px dims: {original_w_px}x{original_h_px}, target width: {target_w_mm}mm, scale: {scale_factor:.4f}")
        elif target_h_mm:
            scale_factor = target_h_mm / original_h_px
            print(f"Original px dims: {original_w_px}x{original_h_px}, target height: {target_h_mm}mm, scale: {scale_factor:.4f}")
        else:
            scale_factor = 1.0

        for (s, e) in graph.edges():
            coords = graph[s][e]['pts']
            if len(coords) < self.min_path_length_px:
                continue
            path = []
            for (yy, xx) in coords:
                pixel_width = distance_map[yy, xx] * 4
                scaled_x = (xx - x_min_skel) * scale_factor
                scaled_y = (yy - y_min_skel) * scale_factor
                scaled_width = pixel_width * scale_factor
                if scaled_width > max_scaled_width:
                    max_scaled_width = scaled_width
                path.append((scaled_x, scaled_y, scaled_width))
            scaled_toolpaths.append(path)
        return scaled_toolpaths, max_scaled_width

    def generate_from_image(self, image_path, target_w_mm, target_h_mm):
        self._initial_setup()
        print(f"Processing image for skeleton method: {image_path}")
        scaled_toolpaths, max_brush_width = self._process_image_for_skeleton(image_path, target_w_mm, target_h_mm)
        if not scaled_toolpaths:
            print("No significant strokes found.")
            return

        if self.smooth_window_size > 1:
            scaled_toolpaths = [self._smooth_path(p) for p in scaled_toolpaths]
            print(f"Applied smoothing window: {self.smooth_window_size}")

        if self.path_optimize:
            scaled_toolpaths = self._optimize_stroke_order(scaled_toolpaths)
            print(f"Path optimization applied: {len(scaled_toolpaths)} strokes reordered for minimum travel.")

        self.max_width_mm = max(max_brush_width, 0.001)
        print(f"Estimated max brush width: {self.max_width_mm:.2f}mm")

        travel_since_last_dip = 0.0

        def _next_threshold():
            if self.tf_enabled:
                return random.uniform(self.tf_min_dist, self.tf_max_dist)
            return self.dip_distance_threshold

        current_threshold = _next_threshold()

        for path in scaled_toolpaths:
            if not path:
                continue

            start_x, start_y, start_width = path[0]

            stroke_length = 0.0
            prev_x, prev_y = start_x, start_y
            for x_orig, y_orig, _ in path:
                stroke_length += math.hypot(x_orig - prev_x, y_orig - prev_y)
                prev_x, prev_y = x_orig, y_orig

            if travel_since_last_dip >= current_threshold:
                self._perform_dip(target_x=start_x, target_y=start_y)
                travel_since_last_dip = 0.0
                current_threshold = _next_threshold()

            travel_since_last_dip += stroke_length

            self.gcode.append(
                f"; Stroke len={stroke_length:.2f}mm  travel_since_dip={travel_since_last_dip:.2f}mm"
            )

            self._set_speed('travel')
            self._handle_move_g0(x=start_x, y=start_y, z=self.z_safe)

            self._set_speed('paint')
            start_z = self._width_to_z(start_width)
            self._handle_move_g1(x=start_x, y=start_y, z=start_z, feed_rate=self.feed_paint / 2)

            for x_orig, y_orig, width in path:
                z = self._width_to_z(width)
                self._handle_move_g1(x=x_orig, y=y_orig, z=z, feed_rate=self.feed_paint)

            self._set_speed('travel')
            self._handle_move_g0(x=path[-1][0], y=path[-1][1], z=self.z_safe)

        self._set_speed('travel')
        self.gcode.append(
            f"G0 X{self.dip_location[0]:.3f} Y{self.dip_location[1]:.3f} Z{self.z_safe_dip:.3f} F{self.feed_rate}"
        )
        self.gcode.append("M2 ; End of program")
        print("G-code generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate G-code for CNC painting (skeleton method)."
    )
    parser.add_argument("input_image", help="Path to input image.")
    parser.add_argument("output_gcode", help="Path to output gcode file.")

    # ── Artwork / CNC ─────────────────────────────────────────────────────────
    parser.add_argument("--width",           type=float, default=250.0, help="Target artwork width (mm).")
    parser.add_argument("--height",          type=float, default=None,  help="Target artwork height (mm). Defaults to proportionally fitting width if not provided.")
    parser.add_argument("--feed_rate",       type=int,   default=12000,   help="Travel feed rate (mm/min).")
    parser.add_argument("--feed_paint",      type=int,   default=600,     help="Paint/stroke feed rate (mm/min).")
    parser.add_argument("--accel_travel",    type=int,   default=12000,   help="Acceleration for travel moves (mm/s²).")
    parser.add_argument("--accel_paint",     type=int,   default=100,     help="Acceleration for paint moves (mm/s²).")
    parser.add_argument("--x_offset",        type=float, default=20.0,   help="Global X offset (mm).")
    parser.add_argument("--y_offset",        type=float, default=55.0,  help="Global Y offset (mm).")
    parser.add_argument("--z_safe",          type=float, default=1.0,   help="Safe Z for rapids (mm).")
    parser.add_argument("--z_paint_max",     type=float, default=0.0,   help="Z for thinnest stroke (mm).")
    parser.add_argument("--z_paint_min",     type=float, default=-5.0,  help="Z for widest stroke (mm).")
    parser.add_argument("--z_safe_dip",      type=float, default=8.0,   help="Safe Z for dip moves (mm).")
    parser.add_argument("--z_global_offset", type=float, default=5.0,   help="Global Z offset added to all Z values.")

    # ── Dip Station ───────────────────────────────────────────────────────────
    parser.add_argument("--dip_duration",          type=float, default=0.00,  help="Dip dwell time (s).")
    parser.add_argument("--dip_wipe_radius",        type=float, default=29.0,  help="Wipe radius from tray center (mm).")
    parser.add_argument("--z_wipe_travel",          type=float, default=1.0,   help="Z height for wipe motion (mm).")
    parser.add_argument("--dip_entry_radius",       type=float, default=5.0,   help="Inner wipe start radius (mm).")
    parser.add_argument("--remove_drops_enabled",   type=eval,  default=False, choices=[True, False], help="Enable remove_drops wipe.")
    parser.add_argument("--dip_distance_threshold", type=float, default=120.0,  help="Painted travel (mm) before re-dipping.")
    parser.add_argument("--dip_jitter",        type=float, default=5.0, help="Random XY jitter on dip approach (mm).")
    parser.add_argument("--dip_spiral_loops",  type=float, default=1.0,  help="Number of spiral loops during dip.")
    parser.add_argument("--dip_spiral_r",      type=float, default=5.0, help="Max spiral radius during dip (mm).")

    # ── Color Positions ───────────────────────────────────────────────────────
    parser.add_argument("--color_index", type=int, default=2, choices=[1, 2, 3, 4])

    parser.add_argument("--color1_dip_x", type=float, default=88.0,  help="Color 1 dip X (mm).")
    parser.add_argument("--color1_dip_y", type=float, default=0.0,   help="Color 1 dip Y (mm).")
    parser.add_argument("--color1_dip_z", type=float, default=1.0,   help="Color 1 dip Z depth (mm).")

    parser.add_argument("--color2_dip_x", type=float, default=148.0,  help="Color 2 dip X (mm).")
    parser.add_argument("--color2_dip_y", type=float, default=0.0,   help="Color 2 dip Y (mm).")
    parser.add_argument("--color2_dip_z", type=float, default=1.0,   help="Color 2 dip Z depth (mm).")

    parser.add_argument("--color3_dip_x", type=float, default=209.0, help="Color 3 dip X (mm).")
    parser.add_argument("--color3_dip_y", type=float, default=0.0,   help="Color 3 dip Y (mm).")
    parser.add_argument("--color3_dip_z", type=float, default=1.0,   help="Color 3 dip Z depth (mm).")

    parser.add_argument("--color4_dip_x", type=float, default=269.0, help="Color 4 dip X (mm).")
    parser.add_argument("--color4_dip_y", type=float, default=0.0,   help="Color 4 dip Y (mm).")
    parser.add_argument("--color4_dip_z", type=float, default=1.0,   help="Color 4 dip Z depth (mm).")

    # ── Water Position ───────────────────────────────────────────
    parser.add_argument("--water_dip_x", type=float, default=28.0, help="Water dish X (mm).")
    parser.add_argument("--water_dip_y", type=float, default=0.0,   help="Water dish Y (mm).")
    parser.add_argument("--water_dip_z", type=float, default=1.0,   help="Water dish Z depth (mm).")

    # ── Image Processing ──────────────────────────────────────────────────────
    parser.add_argument("--max_brush_width",    type=float, default=20.0,  help="Max brush width (mm) for Z mapping.")
    parser.add_argument("--min_path_length_px", type=int,   default=15,    help="Minimum skeleton path length in px.")
    parser.add_argument("--smooth_window_size", type=int,   default=7,    help="Smoothing window size.")
    parser.add_argument("--path_optimize",      type=eval,  default=True, choices=[True, False])

    # ── Tutti Frutti ──────────────────────────────────────────────────────────
    parser.add_argument("--tf_enabled",   type=eval,  default=True, choices=[True, False])
    parser.add_argument("--tf_min_dist",  type=float, default=40.0)
    parser.add_argument("--tf_max_dist",  type=float, default=80.0)
    parser.add_argument("--tf_no_repeat", type=eval,  default=True, choices=[True, False])

    args = parser.parse_args()
    
    # Notice that we no longer force height to be identical to width here if it's None.

    color_dip_coords = {
        1: (args.color1_dip_x, args.color1_dip_y, args.color1_dip_z),
        2: (args.color2_dip_x, args.color2_dip_y, args.color2_dip_z),
        3: (args.color3_dip_x, args.color3_dip_y, args.color3_dip_z),
        4: (args.color4_dip_x, args.color4_dip_y, args.color4_dip_z),
    }
    active_dip = color_dip_coords[args.color_index]

    tf_dishes = [
        {"x": args.color1_dip_x, "y": args.color1_dip_y,
         "z": args.color1_dip_z + args.z_global_offset, "name": "Color 1"},
        {"x": args.color2_dip_x, "y": args.color2_dip_y,
         "z": args.color2_dip_z + args.z_global_offset, "name": "Color 2"},
        {"x": args.color3_dip_x, "y": args.color3_dip_y,
         "z": args.color3_dip_z + args.z_global_offset, "name": "Color 3"},
        {"x": args.color4_dip_x, "y": args.color4_dip_y,
         "z": args.color4_dip_z + args.z_global_offset, "name": "Color 4"},
    ]

    water_dish = {
        "x": args.water_dip_x, 
        "y": args.water_dip_y,
        "z": args.water_dip_z + args.z_global_offset, 
        "name": "Water"
    }

    generator_kwargs = {
        'feed_rate':            args.feed_rate,
        'feed_paint':           args.feed_paint,
        'accel_travel':         args.accel_travel,
        'accel_paint':          args.accel_paint,
        'x_offset':             args.x_offset,
        'y_offset':             args.y_offset,
        'dip_location_raw':     active_dip,
        'dip_duration_s':       args.dip_duration,
        'dip_wipe_radius':      args.dip_wipe_radius,
        'z_wipe_travel_raw':    args.z_wipe_travel,
        'dip_entry_radius':     args.dip_entry_radius,
        'remove_drops_enabled': args.remove_drops_enabled,
        'z_global_offset_val':  args.z_global_offset,
        'z_safe_raw':           args.z_safe,
        'z_safe_dip_raw':       args.z_safe_dip,
        'water_dish':           water_dish, 
    }

    try:
        gcode_generator = SkeletonGCodeGenerator(
            z_paint_max_raw=args.z_paint_max,
            z_paint_min_raw=args.z_paint_min,
            max_width_mm=args.max_brush_width,
            min_path_length_px=args.min_path_length_px,
            smooth_window_size=args.smooth_window_size,
            dip_distance_threshold=args.dip_distance_threshold,
            path_optimize=args.path_optimize,
            tf_enabled=args.tf_enabled,
            tf_dishes=tf_dishes,
            tf_min_dist=args.tf_min_dist,
            tf_max_dist=args.tf_max_dist,
            tf_no_repeat=args.tf_no_repeat,
            **generator_kwargs
        )
        gcode_generator.dip_jitter       = args.dip_jitter
        gcode_generator.dip_spiral_loops = args.dip_spiral_loops
        gcode_generator.dip_spiral_r     = args.dip_spiral_r
        gcode_generator.generate_from_image(
            image_path=args.input_image,
            target_w_mm=args.width,
            target_h_mm=args.height
        )
        gcode_generator.save(args.output_gcode)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
