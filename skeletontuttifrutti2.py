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

class VariableBacklashFixMixin:
    """
    Mixin class to add variable backlash compensation logic.
    """
    def __init__(self, bx_start, bx_end, by_start, by_end, max_travel_x, threshold=0.05, safe_feed=200):
        self.bx_start = bx_start
        self.by_start = by_start
        self.bx_end = bx_end
        self.by_end = by_end
        self.max_travel_x = max_travel_x
        self.threshold = threshold
        self.safe_feed = safe_feed

        self.current_x = 0.0  # Logical (Commanded) X
        self.current_y = 0.0  # Logical (Commanded) Y

        # FIX: renamed from offset_x/y to backlash_offset_x/y to avoid aliasing
        # with GCodeBaseGenerator's canvas x_offset/y_offset.
        self.backlash_offset_x = 0.0
        self.backlash_offset_y = 0.0

        self.dir_x = 0  # Current movement direction for X (-1, 0, 1)
        self.dir_y = 0  # Current movement direction for Y (-1, 0, 1)

    def _get_current_backlash(self, position_x):
        """Calculates backlash based on current X position (linear interpolation)."""
        if self.max_travel_x <= 0:
            return self.bx_start, self.by_start
        factor = max(0, min(1, position_x / self.max_travel_x))
        curr_bx = self.bx_start + (self.bx_end - self.bx_start) * factor
        curr_by = self.by_start + (self.by_end - self.by_start) * factor
        return curr_bx, curr_by

    def _apply_backlash_fix(self, cmd_type, target_x, target_y, target_z=None, feed_rate=None):
        """
        Applies variable backlash compensation to a G0/G1 move and appends G-code.
        target_x/y must be the ABSOLUTE logical coordinates (canvas offset already applied by caller).
        Backlash compensation is tracked in backlash_offset_x/y separately.
        """
        active_bx, active_by = self._get_current_backlash(self.current_x)

        # --- X DIRECTION CHANGE ---
        dx = target_x - self.current_x
        if abs(dx) > self.threshold:
            new_dir_x = 1 if dx > 0 else -1
            if self.dir_x != 0 and new_dir_x != self.dir_x:
                change = active_bx if new_dir_x == 1 else -active_bx
                self.backlash_offset_x += change
                phys_x = max(0.0, self.current_x + self.backlash_offset_x)
                phys_y = max(0.0, self.current_y + self.backlash_offset_y)
                self.gcode.append(f"; Fix X (Var: {active_bx:.3f})")
                self.gcode.append(f"G0 X{phys_x:.3f} Y{phys_y:.3f} F{self.safe_feed}")
            self.dir_x = new_dir_x

        # --- Y DIRECTION CHANGE ---
        dy = target_y - self.current_y
        if abs(dy) > self.threshold:
            new_dir_y = 1 if dy > 0 else -1
            if self.dir_y != 0 and new_dir_y != self.dir_y:
                change = active_by if new_dir_y == 1 else -active_by
                self.backlash_offset_y += change
                phys_x = max(0.0, self.current_x + self.backlash_offset_x)
                phys_y = max(0.0, self.current_y + self.backlash_offset_y)
                self.gcode.append(f"; Fix Y (Var: {active_by:.3f})")
                self.gcode.append(f"G0 X{phys_x:.3f} Y{phys_y:.3f} F{self.safe_feed}")
            self.dir_y = new_dir_y

        # --- OUTPUT MOVE ---
        final_x = max(0.0, target_x + self.backlash_offset_x)
        final_y = max(0.0, target_y + self.backlash_offset_y)

        out = f"{cmd_type} X{final_x:.3f} Y{final_y:.3f}"
        if target_z is not None:
            out += f" Z{target_z:.3f}"
        if cmd_type == "G1" and feed_rate is not None:
            out += f" F{int(feed_rate)}"
        self.gcode.append(out)

        self.current_x = target_x
        self.current_y = target_y
        return final_x, final_y

    def _handle_move_g0(self, x, y, z=None):
        self._apply_backlash_fix("G0", x, y, z, feed_rate=self.feed_rate)

    def _handle_move_g1(self, x, y, z=None, feed_rate=None):
        fr = feed_rate if feed_rate is not None else self.feed_rate
        self._apply_backlash_fix("G1", x, y, z, feed_rate=fr)


class GCodeBaseGenerator(VariableBacklashFixMixin):
    def __init__(self, feed_rate, x_offset, y_offset,
                 dip_location_raw, dip_duration_s,
                 dip_wipe_radius, z_wipe_travel_raw,
                 dip_entry_radius, remove_drops_enabled,
                 z_global_offset_val, z_safe_raw, z_safe_dip_raw,
                 bx_start, bx_end, by_start, by_end, max_travel_x):

        super().__init__(bx_start, bx_end, by_start, by_end, max_travel_x)

        self.feed_rate = feed_rate
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

        self.remove_drops_lift = self.z_wipe_travel
        self.tray_enter_radius = self.dip_entry_radius
        self.remove_drops_radius = self.dip_wipe_radius
        self.remove_drops_angle_variation_deg = 12.0

        # Spiral dip parameters (matching lines500 dip cycle)
        self.dip_jitter = 0.0
        self.dip_spiral_loops = 1.0
        self.dip_spiral_r = 0.0

        self._initial_dip_performed = False

        # ── Tutti Frutti state ────────────────────────────────────────────────
        # Set by SkeletonGCodeGenerator when tf_enabled=True.
        # tf_dishes: list of dicts {x, y, z, name}
        #   x/y/z are ABSOLUTE machine coords; z already includes z_global_offset.
        self.tf_enabled      = False
        self.tf_dishes       = []
        self.tf_min_dist     = 40.0
        self.tf_max_dist     = 80.0
        self.tf_no_repeat    = True
        self._tf_current_idx = -1   # last chosen dish index (for no-repeat)

    def get_logical_offset_x(self):
        return self.x_offset

    def get_logical_offset_y(self):
        return self.y_offset

    def _handle_move_g0(self, x, y, z=None):
        """Apply canvas offset once, then forward to Mixin (which adds only backlash)."""
        super()._handle_move_g0(x + self.x_offset, y + self.y_offset, z)

    def _handle_move_g1(self, x, y, z=None, feed_rate=None):
        """Apply canvas offset once, then forward to Mixin (which adds only backlash)."""
        super()._handle_move_g1(x + self.x_offset, y + self.y_offset, z, feed_rate)

    def remove_drops(self, tray_x, tray_y, x, y):
        """
        Wipe/shake-off motion after dipping.
        Coordinates are relative (pre-canvas-offset); _handle_move adds offset internally.
        Sequence:
          1. G0 XY to entry point AT z_safe_dip
          2. G1 lower to z_wipe_travel
          3. G1 slow wipe outward to exit point
          4. G0 lift back to z_safe_dip
        """
        dx = x - tray_x
        dy = y - tray_y
        dist = math.hypot(dx, dy)
        base_angle = math.atan2(dy, dx) if dist > 0 else 0.0

        max_var_rad = math.radians(getattr(self, "remove_drops_angle_variation_deg", 12.0))
        angle = base_angle + random.uniform(-max_var_rad, max_var_rad)
        ux, uy = math.cos(angle), math.sin(angle)

        x1 = tray_x + ux * self.tray_enter_radius
        y1 = tray_y + uy * self.tray_enter_radius
        x2 = tray_x + ux * self.remove_drops_radius
        y2 = tray_y + uy * self.remove_drops_radius

        self.gcode.append(f"; remove_drops start -> dir_angle={math.degrees(angle):.1f}deg")
        self._handle_move_g0(x=x1, y=y1, z=self.z_safe_dip)
        slow_feed = 600
        self._handle_move_g1(x=x1, y=y1, z=self.remove_drops_lift, feed_rate=slow_feed)
        self._handle_move_g1(x=x2, y=y2, z=self.remove_drops_lift, feed_rate=slow_feed)
        self._handle_move_g0(x=x2, y=y2, z=self.z_safe_dip)
        self.gcode.append(f"G0 F{self.feed_rate}")
        self.gcode.append("; remove_drops end")

    def _perform_dip(self, target_x=None, target_y=None):
        """
        Full dip cycle — matches lines500_con_line_cesnja.py UltraPainter._perform_dip_and_travel.

        Normal mode:   dips at self.dip_location (dish chosen by --color_index).
        Tutti Frutti:  randomly picks a dish from self.tf_dishes on every call.
                       tf_no_repeat prevents picking the same dish twice in a row.

        target_x/y: start of the next stroke (relative/pre-offset) — used to aim the wipe.

        Sequence:
          1. G0 lift to z_safe (low clearance)
          2. G0 approach dip position (with jitter) at z_safe_dip (high clearance)
          3. G1 lower to dip_z
          4. G1 spiral outward (dip_spiral_loops * 4 steps, radius up to dip_spiral_r)
          5. G0 lift to z_wipe_travel
          6. G0 move to wipe point (aimed toward target, at dip_wipe_radius distance)
          7. G0 lift to z_safe_dip
          8. G0 travel to target at z_safe
        """
        # ── Pick dish ────────────────────────────────────────────────────────
        if self.tf_enabled and self.tf_dishes:
            pool = list(range(len(self.tf_dishes)))
            if self.tf_no_repeat and len(pool) > 1 and self._tf_current_idx >= 0:
                pool = [i for i in pool if i != self._tf_current_idx]
            picked_idx = random.choice(pool)
            self._tf_current_idx = picked_idx
            dish = self.tf_dishes[picked_idx]
            dip_abs_x = dish['x']
            dip_abs_y = dish['y']
            dip_z     = dish['z']
            self.gcode.append(f"\n; --- DIP START -> {dish['name']} (tutti frutti) ---")
        else:
            dip_abs_x = self.dip_location[0]
            dip_abs_y = self.dip_location[1]
            dip_z     = self.dip_location[2]
            self.gcode.append("\n; --- DIP START ---")

        # Absolute → relative for _handle_move (canvas offset applied inside _handle_move)
        dip_x = dip_abs_x - self.x_offset
        dip_y = dip_abs_y - self.y_offset

        # 1. Lift to z_safe at current position
        self.gcode.append(f"G0 Z{self.z_safe:.3f} F3000")

        # 2. Approach dip with jitter at z_safe_dip
        ax = dip_x + random.uniform(-self.dip_jitter, self.dip_jitter)
        ay = dip_y + random.uniform(-self.dip_jitter, self.dip_jitter)
        self._handle_move_g0(x=ax, y=ay, z=self.z_safe_dip)

        # 3. Lower to dip depth
        self.gcode.append(f"G1 Z{dip_z:.3f} F3000")

        # 4. Spiral outward
        num_steps = int(self.dip_spiral_loops * 4)
        for i in range(num_steps):
            ang = i * (math.pi / 2)
            r = (i / num_steps) * self.dip_spiral_r if num_steps > 0 else 0.0
            self._handle_move_g1(x=ax + r * math.cos(ang),
                                  y=ay + r * math.sin(ang),
                                  feed_rate=2500)

        # 5–8. Wipe toward target then travel
        tx = target_x if target_x is not None else dip_x + self.dip_wipe_radius
        ty = target_y if target_y is not None else dip_y
        ddx, ddy = tx - dip_x, ty - dip_y
        dist = math.hypot(ddx, ddy)
        if dist > 0:
            wx = dip_x + (ddx / dist) * self.dip_wipe_radius
            wy = dip_y + (ddy / dist) * self.dip_wipe_radius
        else:
            wx = dip_x + self.dip_wipe_radius
            wy = dip_y

        self.gcode.append(f"G0 Z{self.z_wipe_travel:.3f} F3000")
        self._handle_move_g0(x=wx, y=wy)
        self.gcode.append(f"G0 Z{self.z_safe_dip:.3f} F3000")
        self._handle_move_g0(x=tx, y=ty, z=self.z_safe)

        self.gcode.append(f"G0 F{self.feed_rate}")
        self.gcode.append("; --- DIP END ---")

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write("\n".join(self.gcode))
        print(f"G-code saved to {filename}")

    def _initial_setup(self):
        self.gcode.append("; --- Variable Backlash Fix Parameters ---")
        self.gcode.append(f"; X Backlash Range: {self.bx_start}mm (X=0) -> {self.bx_end}mm (X={self.max_travel_x})")
        self.gcode.append(f"; Y Backlash Range: {self.by_start}mm (X=0) -> {self.by_end}mm (X={self.max_travel_x})")
        self.gcode.append(f"; Safe Feed for Fix: {self.safe_feed} mm/min")
        self.gcode.append("; ---------------------------------------")
        if self.tf_enabled and self.tf_dishes:
            names = ", ".join(d["name"] for d in self.tf_dishes)
            self.gcode.append(
                f"; TUTTI FRUTTI: ON  stroke {self.tf_min_dist}-{self.tf_max_dist}mm  "
                f"pool: [{names}]  no_repeat: {self.tf_no_repeat}"
            )
        self.gcode.append("G90 ; Set Absolute Positioning")
        self.gcode.append("G21 ; Set Units to Millimeters")

        self._handle_move_g0(x=0.0, y=0.0, z=self.z_safe)

        if not self._initial_dip_performed:
            print("Performing initial brush dip before starting.")
            self._perform_dip()
            self._initial_dip_performed = True


class SkeletonGCodeGenerator(GCodeBaseGenerator):
    def __init__(self, z_paint_max_raw, z_paint_min_raw, max_width_mm,
                 min_path_length_px, smooth_window_size,
                 dip_distance_threshold, path_optimize,
                 # ── Tutti Frutti ──────────────────────────────────────────────
                 tf_enabled=False,
                 tf_dishes=None,    # list of {x,y,z,name} (absolute, z includes global offset)
                 tf_min_dist=40.0,  # min painted mm between dips in TF mode
                 tf_max_dist=80.0,  # max painted mm between dips in TF mode
                 tf_no_repeat=True, # never repeat same dish twice in a row
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.z_paint_max = z_paint_max_raw + self.z_global_offset
        self.z_paint_min = z_paint_min_raw + self.z_global_offset
        self.max_width_mm = max(max_width_mm, 0.001)
        self.min_path_length_px = min_path_length_px
        self.smooth_window_size = smooth_window_size
        self.dip_distance_threshold = dip_distance_threshold
        self.path_optimize = path_optimize
        # Wire TF config into base generator
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
        """Nearest-neighbor stroke ordering, with optional reversal, from canvas origin."""
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
        scale_factor = 1.0
        if target_w_mm and target_h_mm:
            scale_x = target_w_mm / original_w_px
            scale_y = target_h_mm / original_h_px
            scale_factor = min(scale_x, scale_y)
            print(f"Original px dims: {original_w_px}x{original_h_px}, target mm: {target_w_mm}x{target_h_mm}, scale: {scale_factor:.4f}")
        for (s, e) in graph.edges():
            coords = graph[s][e]['pts']
            if len(coords) < self.min_path_length_px:
                continue
            path = []
            for (yy, xx) in coords:
                pixel_width = distance_map[yy, xx] * 2
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

        # ── Dip threshold ─────────────────────────────────────────────────────
        # Normal mode : fixed at self.dip_distance_threshold.
        # Tutti Frutti: randomised after every dip, drawn from [tf_min_dist, tf_max_dist].
        # Matches the HTML script:  this.md = mn + Math.random() * (mx - mn)
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

            # Check BEFORE adding stroke so the stroke after a dip gets fresh paint
            if travel_since_last_dip >= current_threshold:
                self._perform_dip(target_x=start_x, target_y=start_y)
                travel_since_last_dip = 0.0
                current_threshold = _next_threshold()

            travel_since_last_dip += stroke_length

            self.gcode.append(
                f"; Stroke len={stroke_length:.2f}mm  travel_since_dip={travel_since_last_dip:.2f}mm"
            )

            self._handle_move_g0(x=start_x, y=start_y, z=self.z_safe)
            start_z = self._width_to_z(start_width)
            self._handle_move_g1(x=start_x, y=start_y, z=start_z, feed_rate=self.feed_rate / 2)

            for x_orig, y_orig, width in path:
                z = self._width_to_z(width)
                self._handle_move_g1(x=x_orig, y=y_orig, z=z, feed_rate=self.feed_rate)

            self._handle_move_g0(x=path[-1][0], y=path[-1][1], z=self.z_safe)

        dip_x = self.dip_location[0] - self.x_offset
        dip_y = self.dip_location[1] - self.y_offset
        self._handle_move_g0(x=dip_x, y=dip_y, z=self.z_safe_dip)
        self.gcode.append("M2 ; End of program")
        print("G-code generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate G-code for CNC painting (skeleton method) with variable backlash compensation."
    )
    parser.add_argument("input_image", help="Path to input image.")
    parser.add_argument("output_gcode", help="Path to output gcode file.")

    # ── Artwork / CNC ─────────────────────────────────────────────────────────
    parser.add_argument("--width",           type=float, default=130.0, help="Target artwork width (mm).")
    parser.add_argument("--height",          type=float, default=None,  help="Target artwork height (mm). Defaults to --width.")
    parser.add_argument("--feed_rate",       type=int,   default=1300,   help="Feed rate mm/min for painting moves.")
    parser.add_argument("--x_offset",        type=float, default=0.0,   help="Global X offset (mm).")
    parser.add_argument("--y_offset",        type=float, default=29.0,  help="Global Y offset (mm).")
    parser.add_argument("--z_safe",          type=float, default=2.0,   help="Safe Z for rapids (mm, before global offset).")
    parser.add_argument("--z_paint_max",     type=float, default=0.0,   help="Z for thinnest stroke (mm, before global offset).")
    parser.add_argument("--z_paint_min",     type=float, default=-2.0,  help="Z for widest stroke (mm, before global offset).")
    parser.add_argument("--z_safe_dip",      type=float, default=7.0,   help="Safe Z for dip moves (mm, before global offset).")
    parser.add_argument("--z_global_offset", type=float, default=0.0,   help="Global Z offset added to all Z values.")

    # ── Dip Station ───────────────────────────────────────────────────────────
    parser.add_argument("--dip_duration",          type=float, default=0.001,  help="Dip dwell time (s). (unused in spiral-dip mode)")
    parser.add_argument("--dip_wipe_radius",        type=float, default=17.0,  help="Wipe radius from tray center (mm).")
    parser.add_argument("--z_wipe_travel",          type=float, default=1.0,   help="Z height for wipe motion (mm, before global offset).")
    parser.add_argument("--dip_entry_radius",       type=float, default=5.0,   help="Inner wipe start radius (mm). (unused in spiral-dip mode)")
    parser.add_argument("--remove_drops_enabled",   type=eval,  default=True,  choices=[True, False], help="Enable remove_drops wipe. (unused in spiral-dip mode)")
    parser.add_argument("--dip_distance_threshold", type=float, default=90.0,  help="Painted travel (mm) before re-dipping (normal mode).")
    # Spiral dip parameters (matching lines500_con_line_cesnja.py)
    parser.add_argument("--dip_jitter",        type=float, default=7.0, help="Random XY jitter on dip approach (mm).")
    parser.add_argument("--dip_spiral_loops",  type=float, default=1.0,  help="Number of spiral loops during dip.")
    parser.add_argument("--dip_spiral_r",      type=float, default=10.0, help="Max spiral radius during dip (mm).")

    # ── Color Positions ───────────────────────────────────────────────────────
    parser.add_argument("--color_index", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Which color/petri dish to use for this run (1-4). Ignored in TF mode.")

    parser.add_argument("--color1_dip_x", type=float, default=17.0,  help="Color 1 dip X (mm).")
    parser.add_argument("--color1_dip_y", type=float, default=0.0,   help="Color 1 dip Y (mm).")
    parser.add_argument("--color1_dip_z", type=float, default=0.0,   help="Color 1 dip Z depth (mm, before global offset).")

    parser.add_argument("--color2_dip_x", type=float, default=63.0,  help="Color 2 dip X (mm).")
    parser.add_argument("--color2_dip_y", type=float, default=0.0,   help="Color 2 dip Y (mm).")
    parser.add_argument("--color2_dip_z", type=float, default=0.0,   help="Color 2 dip Z depth (mm, before global offset).")

    parser.add_argument("--color3_dip_x", type=float, default=109.0, help="Color 3 dip X (mm).")
    parser.add_argument("--color3_dip_y", type=float, default=0.0,   help="Color 3 dip Y (mm).")
    parser.add_argument("--color3_dip_z", type=float, default=0.0,   help="Color 3 dip Z depth (mm, before global offset).")

    parser.add_argument("--color4_dip_x", type=float, default=155.0, help="Color 4 dip X (mm).")
    parser.add_argument("--color4_dip_y", type=float, default=0.0,   help="Color 4 dip Y (mm).")
    parser.add_argument("--color4_dip_z", type=float, default=0.0,   help="Color 4 dip Z depth (mm, before global offset).")

    # ── Image Processing ──────────────────────────────────────────────────────
    parser.add_argument("--max_brush_width",    type=float, default=3.0,  help="Max brush width (mm) for Z mapping.")
    parser.add_argument("--min_path_length_px", type=int,   default=1,    help="Minimum skeleton path length in px.")
    parser.add_argument("--smooth_window_size", type=int,   default=3,    help="Smoothing window size.")
    parser.add_argument("--path_optimize",      type=eval,  default=True, choices=[True, False],
                        help="Nearest-neighbor stroke reordering to minimize travel distance.")

    # ── Tutti Frutti ──────────────────────────────────────────────────────────
    # Randomly picks a petri dish on every dip from all 4 color positions,
    # mirroring the TF mode in Four-channel-robotic-painting.html.
    parser.add_argument("--tf_enabled",   type=eval,  default=True, choices=[True, False],
                        help="Enable Tutti Frutti: random color picked on every dip.")
    parser.add_argument("--tf_min_dist",  type=float, default=40.0,
                        help="TF mode: min painted distance (mm) between dips.")
    parser.add_argument("--tf_max_dist",  type=float, default=80.0,
                        help="TF mode: max painted distance (mm) between dips.")
    parser.add_argument("--tf_no_repeat", type=eval,  default=True, choices=[True, False],
                        help="TF mode: never dip same color twice in a row.")

    # ── Variable Backlash ─────────────────────────────────────────────────────
    parser.add_argument("--bx_start", type=float, default=0.5,   help="X backlash at X=0 (mm).")
    parser.add_argument("--bx_end",   type=float, default=0.5,   help="X backlash at X=max_x (mm).")
    parser.add_argument("--by_start", type=float, default=1.6,   help="Y backlash at X=0 (mm).")
    parser.add_argument("--by_end",   type=float, default=1.6,   help="Y backlash at X=max_x (mm).")
    parser.add_argument("--max_x",    type=float, default=150.0, help="Maximum X travel (mm).")

    args = parser.parse_args()
    if args.height is None:
        args.height = args.width

    color_dip_coords = {
        1: (args.color1_dip_x, args.color1_dip_y, args.color1_dip_z),
        2: (args.color2_dip_x, args.color2_dip_y, args.color2_dip_z),
        3: (args.color3_dip_x, args.color3_dip_y, args.color3_dip_z),
        4: (args.color4_dip_x, args.color4_dip_y, args.color4_dip_z),
    }
    active_dip = color_dip_coords[args.color_index]
    print(f"Using color {args.color_index} dip position: X={active_dip[0]}, Y={active_dip[1]}, Z={active_dip[2]}")

    # Build the TF dish pool — all 4 colors in absolute machine coords.
    # z has z_global_offset baked in so _perform_dip can use it directly.
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

    if args.tf_enabled:
        print(f"Tutti Frutti ON  stroke={args.tf_min_dist}-{args.tf_max_dist}mm  "
              f"no_repeat={args.tf_no_repeat}")
        print("  Dish pool: " + "  ".join(
            f"{d['name']}({d['x']},{d['y']})" for d in tf_dishes))

    generator_kwargs = {
        'feed_rate':            args.feed_rate,
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
        'bx_start':             args.bx_start,
        'bx_end':               args.bx_end,
        'by_start':             args.by_start,
        'by_end':               args.by_end,
        'max_travel_x':         args.max_x,
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
        # Spiral dip parameters (lines500-style)
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
