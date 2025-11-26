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
        # Backlash at X=0 (start)
        self.bx_start = bx_start
        self.by_start = by_start
        
        # Backlash at X=max_travel_x (end)
        self.bx_end = bx_end
        self.by_end = by_end
        
        self.max_travel_x = max_travel_x 
        
        self.threshold = threshold
        self.safe_feed = safe_feed
        
        self.current_x = 0.0 # Logical (Commanded) X
        self.current_y = 0.0 # Logical (Commanded) Y
        
        self.offset_x = 0.0 # Backlash offset for X
        self.offset_y = 0.0 # Backlash offset for Y
        
        self.dir_x = 0 # Current movement direction for X (-1, 0, 1)
        self.dir_y = 0 # Current movement direction for Y (-1, 0, 1)

    def _get_current_backlash(self, position_x):
        """Calculates backlash based on current X position (Linear interpolation)."""
        if self.max_travel_x <= 0:
            return self.bx_start, self.by_start
        
        # Factor between 0 and 1
        factor = position_x / self.max_travel_x
        factor = max(0, min(1, factor))
        
        # Calculate current required backlash: Start + (Difference * Factor)
        curr_bx = self.bx_start + (self.bx_end - self.bx_start) * factor
        curr_by = self.by_start + (self.by_end - self.by_start) * factor
        
        return curr_bx, curr_by

    def _apply_backlash_fix(self, cmd_type, target_x, target_y, target_z=None, feed_rate=None):
        """
        Applies variable backlash compensation to a G0/G1 move and appends G-code.
        This function is based on the logic from variable_backlash.py's process method.
        It uses self.current_x/y (logical) and updates self.offset_x/y (physical adjustment).
        """
        original_current_x = self.current_x
        original_current_y = self.current_y

        # --- GET ACTIVE BACKLASH (based on logical current position) ---
        active_bx, active_by = self._get_current_backlash(self.current_x)

        # --- X LOGIC ---
        dx = target_x - self.current_x
        if abs(dx) > self.threshold:
            new_dir_x = 1 if dx > 0 else -1
            
            if self.dir_x != 0 and new_dir_x != self.dir_x:
                # Direction change detected, apply compensation
                change = active_bx if new_dir_x == 1 else -active_bx
                self.offset_x += change
                
                # Physical position for the fix move
                phys_x = self.current_x + self.offset_x
                phys_y = self.current_y + self.offset_y
                phys_x = max(0.0, phys_x) # Ensure non-negative
                phys_y = max(0.0, phys_y) # Ensure non-negative
                
                self.gcode.append(f"; Fix X (Var: {active_bx:.3f})")
                self.gcode.append(f"G0 X{phys_x:.3f} Y{phys_y:.3f} F{self.safe_feed}")
                
            self.dir_x = new_dir_x

        # --- Y LOGIC ---
        dy = target_y - self.current_y
        if abs(dy) > self.threshold:
            new_dir_y = 1 if dy > 0 else -1
            
            if self.dir_y != 0 and new_dir_y != self.dir_y:
                # Direction change detected, apply compensation
                change = active_by if new_dir_y == 1 else -active_by
                self.offset_y += change
                
                # Physical position for the fix move
                phys_x = self.current_x + self.offset_x
                phys_y = self.current_y + self.offset_y
                phys_x = max(0.0, phys_x) # Ensure non-negative
                phys_y = max(0.0, phys_y) # Ensure non-negative
                
                self.gcode.append(f"; Fix Y (Var: {active_by:.3f})")
                self.gcode.append(f"G0 X{phys_x:.3f} Y{phys_y:.3f} F{self.safe_feed}")

            self.dir_y = new_dir_y

        # --- OUTPUT G-CODE (Target move with offset) ---
        final_x = target_x + self.offset_x
        final_y = target_y + self.offset_y
        
        final_x = max(0.0, final_x)
        final_y = max(0.0, final_y)

        out = f"{cmd_type} X{final_x:.3f} Y{final_y:.3f}"
        if target_z is not None:
            out += f" Z{target_z:.3f}"
        if cmd_type == "G1" and feed_rate is not None:
            out += f" F{int(feed_rate)}"
        elif cmd_type == "G0":
             # G0 moves are safe_feed in the original script if a fix was applied. 
             # However, since we track feed_rate in the generator, we'll use the provided one 
             # for regular G0, but keep the fix moves at safe_feed.
             pass 

        self.gcode.append(out)
        
        # Update logical (commanded) current position
        self.current_x = target_x
        self.current_y = target_y
        
        return final_x, final_y

    def _handle_move_g0(self, x, y, z=None):
        """Public-facing G0 move that applies backlash fix."""
        self._apply_backlash_fix("G0", x, y, z, feed_rate=self.feed_rate)
    
    def _handle_move_g1(self, x, y, z=None, feed_rate=None):
        """Public-facing G1 move that applies backlash fix."""
        fr = feed_rate if feed_rate is not None else self.feed_rate
        self._apply_backlash_fix("G1", x, y, z, feed_rate=fr)


class GCodeBaseGenerator(VariableBacklashFixMixin):
    def __init__(self, feed_rate, x_offset, y_offset,
                 dip_location_raw, dip_duration_s,
                 dip_wipe_radius, z_wipe_travel_raw,
                 dip_entry_radius, remove_drops_enabled,
                 z_global_offset_val, z_safe_raw, z_safe_dip_raw,
                 # Backlash Fix Parameters
                 bx_start, bx_end, by_start, by_end, max_travel_x):

        # Backlash Mixin Initialization
        super().__init__(bx_start, bx_end, by_start, by_end, max_travel_x)

        # Base Generator Initialization
        self.feed_rate = feed_rate
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.gcode = []
        self.z_global_offset = z_global_offset_val

        # Z heights adjusted by global offset
        self.z_safe = z_safe_raw + self.z_global_offset
        self.z_safe_dip = z_safe_dip_raw + self.z_global_offset
        self.z_wipe_travel = z_wipe_travel_raw + self.z_global_offset

        # dip location (x,y,z) - z adjusted by global offset
        self.dip_location = (dip_location_raw[0], dip_location_raw[1], dip_location_raw[2] + self.z_global_offset)
        self.dip_duration_s = dip_duration_s
        self.dip_wipe_radius = dip_wipe_radius
        self.dip_entry_radius = dip_entry_radius
        self.remove_drops_enabled = remove_drops_enabled

        # parameters used by remove_drops
        self.remove_drops_lift = self.z_wipe_travel
        self.tray_enter_radius = self.dip_entry_radius
        self.remove_drops_radius = self.dip_wipe_radius
        self.offset_x = self.x_offset # X offset from generator args, but also used by Mixin
        self.offset_y = self.y_offset # Y offset from generator args, but also used by Mixin
        # allowed angular variation (degrees) around base direction
        self.remove_drops_angle_variation_deg = 12.0

        self._initial_dip_performed = False

    # Overriding Mixin's offset initialization to match BaseGenerator's needs
    # We must ensure the Mixin's internal offset tracking is correctly initialized
    # with the initial x_offset/y_offset from the arguments.
    # Note: The Mixin's offset_x/y are used to track physical deviations due to backlash.
    # The BaseGenerator's offset_x/y are used to calculate the logical (commanded) target.

    # We will rename the BaseGenerator's arguments to avoid conflict with Mixin's variables
    def get_logical_offset_x(self):
        return self.x_offset
    def get_logical_offset_y(self):
        return self.y_offset

    def _handle_move_g0(self, x, y, z=None):
        """
        Overrides Mixin's move handler to correctly apply the generator's global offset.
        The Mixin expects the target_x/y to be the final logical (commanded) position.
        """
        target_x = x + self.get_logical_offset_x()
        target_y = y + self.get_logical_offset_y()
        super()._handle_move_g0(target_x, target_y, z)

    def _handle_move_g1(self, x, y, z=None, feed_rate=None):
        """
        Overrides Mixin's move handler to correctly apply the generator's global offset.
        """
        target_x = x + self.get_logical_offset_x()
        target_y = y + self.get_logical_offset_y()
        super()._handle_move_g1(target_x, target_y, z, feed_rate)
    
    # --- The original methods below are modified to use the new _handle_move functions. ---

    def remove_drops(self, tray_x, tray_y, x, y):
        """
        Wipe/shake-off motion that leaves the tray in the logical direction of the next stroke.
        'x','y' should be the point representing the direction to continue (e.g. start of next stroke).
        A small angular variation is applied to avoid always wiping the exact same spot.
        """
        # account for offsets if present (already handled by _handle_move)
        target_x_no_offset = x 
        target_y_no_offset = y 
        
        # vector from tray center to target (direction of continuation)
        dx = target_x_no_offset - tray_x
        dy = target_y_no_offset - tray_y
        dist = math.hypot(dx, dy)
        if dist == 0:
            # fallback direction +X if target equals tray center
            base_angle = 0.0
        else:
            base_angle = math.atan2(dy, dx)

        # limited random variation around base angle
        max_variation_deg = getattr(self, "remove_drops_angle_variation_deg", 12.0)
        max_variation_rad = math.radians(max_variation_deg)
        angle = base_angle + random.uniform(-max_variation_rad, max_variation_rad)

        # unit vector for chosen angle
        ux = math.cos(angle)
        uy = math.sin(angle)

        # entry and exit points on the tray circumference (no generator offset yet)
        x1 = tray_x + ux * self.tray_enter_radius
        y1 = tray_y + uy * self.tray_enter_radius
        x2 = tray_x + ux * self.remove_drops_radius
        y2 = tray_y + uy * self.remove_drops_radius

        # perform moves: approach edge, lift to wipe height, slow wipe to outer edge, restore feed
        self.gcode.append(f"; remove_drops start -> dir_angle={math.degrees(angle):.1f}deg")
        self._handle_move_g0(x=x1, y=y1) # Move to x1, y1 (Logical target + offset)
        self._handle_move_g0(x=x1, y=y1, z=self.remove_drops_lift) # Lift to wipe height
        
        # NOTE: G-code F rate must be set *before* G1. We use G1 with F set.
        slow_feed = 600
        self._handle_move_g1(x=x2, y=y2, feed_rate=slow_feed) # Slow wipe to outer edge
        
        # Restore feed rate is no longer necessary as it's included in _handle_move_g1/g0 
        # or subsequent move will use the stored feed rate. We add the old command for clarity.
        self.gcode.append(f"G0 F{self.feed_rate}") 
        self.gcode.append("; remove_drops end")

    def _perform_dip(self, target_x=None, target_y=None):
        """
        Perform dip: move to dip location, lower, dwell, lift to wipe height and perform remove_drops.
        If target_x/target_y provided, pass them to remove_drops so wipe follows the next-stroke direction.
        Note: dip_location coordinates are already offset/global. target_x/y are relative to origin (no offset).
        """
        dip_x_no_offset = self.dip_location[0] - self.get_logical_offset_x()
        dip_y_no_offset = self.dip_location[1] - self.get_logical_offset_y()
        dip_z = self.dip_location[2]

        # move up to safe-dip height then to dip location
        self._handle_move_g0(x=dip_x_no_offset, y=dip_y_no_offset, z=self.z_safe_dip)
        self._handle_move_g0(x=dip_x_no_offset, y=dip_y_no_offset) 
        
        # dip into paint
        dip_feed = 800
        self._handle_move_g1(x=dip_x_no_offset, y=dip_y_no_offset, z=dip_z, feed_rate=dip_feed)
        if self.dip_duration_s > 0:
            self.gcode.append(f"G4 P{int(self.dip_duration_s * 1000)}")

        # lift to wipe height
        wipe_z = dip_z + 2.0
        self._handle_move_g1(x=dip_x_no_offset, y=dip_y_no_offset, z=wipe_z, feed_rate=dip_feed)

        # decide direction target for wipe (target_x/y are relative to origin)
        if target_x is None or target_y is None:
            # Fallback direction, relative to origin
            tx = dip_x_no_offset + self.dip_wipe_radius
            ty = dip_y_no_offset
        else:
            tx = target_x
            ty = target_y

        # perform remove_drops if enabled
        if self.remove_drops_enabled:
            self.remove_drops(
                tray_x=dip_x_no_offset, # Use relative dip coords for internal wipe math
                tray_y=dip_y_no_offset,
                x=tx, # relative target
                y=ty
            )

        # retreat and reset feed (safe_dip move uses G1 F600 in original)
        retreat_feed = 600
        self._handle_move_g1(x=dip_x_no_offset, y=dip_y_no_offset, z=self.z_safe_dip, feed_rate=retreat_feed)
        # Reset feed rate (added for clarity, subsequent moves will use self.feed_rate)
        self.gcode.append(f"G0 F{self.feed_rate}")

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write("\n".join(self.gcode))
        print(f"G-code saved to {filename}")

    def _initial_setup(self):
        # Backlash fix headers
        self.gcode.append("; --- Variable Backlash Fix Parameters ---")
        self.gcode.append(f"; X Backlash Range: {self.bx_start}mm (X=0) -> {self.bx_end}mm (X={self.max_travel_x})")
        self.gcode.append(f"; Y Backlash Range: {self.by_start}mm (X=0) -> {self.by_end}mm (X={self.max_travel_x})")
        self.gcode.append(f"; Safe Feed for Fix: {self.safe_feed} mm/min")
        self.gcode.append("; ---------------------------------------")

        self.gcode.append("G90 ; Set Absolute Positioning")
        self.gcode.append("G21 ; Set Units to Millimeters")
        
        # Initial safe Z lift (logical X/Y at 0, 0)
        self._handle_move_g0(x=0.0, y=0.0, z=self.z_safe) 
        
        if not self._initial_dip_performed:
            # perform one initial dip before painting begins
            print("Performing initial brush dip before starting.")
            self._perform_dip()
            self._initial_dip_performed = True


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
        # Linear map from [0, max_width_mm] to [z_paint_max, z_paint_min]
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
            avg_x = sum(p[0] for p in window) / len(window)
            avg_y = sum(p[1] for p in window) / len(window)
            avg_w = sum(p[2] for p in window) / len(window)
            smoothed.append((avg_x, avg_y, avg_w))
        return smoothed

    def _process_image_for_skeleton(self, image_path, target_w_mm=None, target_h_mm=None):
        # ... (rest of the image processing remains the same)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        image = cv2.flip(image, 0)
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        # Ensure boolean for remove_small_objects and then convert back to uint8
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
        self.max_width_mm = max(max_brush_width, 0.001)
        print(f"Estimated max brush width: {self.max_width_mm:.2f}mm")

        dip_distance_threshold = self.dip_distance_threshold
        travel_since_last_dip = 0.0

        for path in scaled_toolpaths:
            if not path:
                continue
            # compute start of stroke (relative to origin/no generator offset)
            start_x, start_y, start_width = path[0]

            # compute only the painting distance (stroke length)
            stroke_length = 0.0
            prev_x, prev_y = start_x, start_y # relative coords
            for x_orig, y_orig, _ in path:
                stroke_length += math.hypot(x_orig - prev_x, y_orig - prev_y)
                prev_x, prev_y = x_orig, y_orig

            travel_since_last_dip += stroke_length

            # if threshold reached, dip and pass start of this upcoming stroke 
            if travel_since_last_dip >= dip_distance_threshold:
                # target_x/y are relative to origin, which is what _perform_dip expects
                self._perform_dip(target_x=start_x, target_y=start_y)
                travel_since_last_dip = 0.0

            # emit G-code for this stroke
            self.gcode.append(f"; Starting new stroke, length: {stroke_length:.2f}mm")
            # Move to stroke start (Z=safe height)
            self._handle_move_g0(x=start_x, y=start_y, z=self.z_safe) 
            
            # Lower to paint start Z (Z is calculated from width)
            start_z = self._width_to_z(start_width)
            self._handle_move_g1(x=start_x, y=start_y, z=start_z, feed_rate=self.feed_rate / 2) 
            
            # Paint path
            for x_orig, y_orig, width in path:
                z = self._width_to_z(width)
                self._handle_move_g1(x=x_orig, y=y_orig, z=z, feed_rate=self.feed_rate) 
                
            # Lift after stroke (Z=safe height)
            self._handle_move_g0(x=path[-1][0], y=path[-1][1], z=self.z_safe) 

        # final return and program end
        dip_x_no_offset = self.dip_location[0] - self.get_logical_offset_x()
        dip_y_no_offset = self.dip_location[1] - self.get_logical_offset_y()
        self._handle_move_g0(x=dip_x_no_offset, y=dip_y_no_offset, z=self.z_safe_dip)
        self.gcode.append("M2 ; End of program")
        print("G-code generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate G-code for CNC painting (skeleton method) with variable backlash compensation.")
    parser.add_argument("input_image", help="Path to input image.")
    parser.add_argument("output_gcode", help="Path to output gcode file.")
    
    # CNC and Artwork Parameters
    parser.add_argument("--width", type=float, default=130.0, help="Target artwork width (mm).")
    parser.add_argument("--height", type=float, default=None, help="Target artwork height (mm). If omitted equals --width.")
    parser.add_argument("--feed_rate", type=int, default=900, help="Feed rate mm/min for painting moves.")
    parser.add_argument("--x_offset", type=float, default=0.0, help="Global X offset (mm).")
    parser.add_argument("--y_offset", type=float, default=25.0, help="Global Y offset (mm).")
    parser.add_argument("--z_safe", type=float, default=2.0, help="Safe Z for rapids (mm, before offset).")
    parser.add_argument("--z_paint_max", type=float, default=0.0, help="Z for thinnest stroke (mm, before offset).")
    parser.add_argument("--z_paint_min", type=float, default=-0.0, help="Z for widest stroke (mm, before offset).")
    parser.add_argument("--z_safe_dip", type=float, default=7.0, help="Safe Z for dip moves (mm, before offset).")
    parser.add_argument("--z_global_offset", type=float, default=0.0, help="Global Z offset added to all Z coordinates.")
    
    # Dip Station Parameters
    parser.add_argument("--dip_x", type=float, default=63.0, help="Dip location X (mm).")
    parser.add_argument("--dip_y", type=float, default=0.0, help="Dip location Y (mm).")
    parser.add_argument("--dip_z", type=float, default=0.0, help="Dip Z depth (mm, before offset).")
    parser.add_argument("--dip_duration", type=float, default=0.1, help="Dip dwell time (s).")
    parser.add_argument("--dip_wipe_radius", type=float, default=17.0, help="Wipe radius (mm).")
    parser.add_argument("--z_wipe_travel", type=float, default=1.0, help="Z height for wipe motion (mm, before offset).")
    parser.add_argument("--dip_entry_radius", type=float, default=5.0, help="Dip entry radius (mm).")
    parser.add_argument("--remove_drops_enabled", type=eval, default=True, choices=[True, False], help="Enable remove_drops wipe.")
    parser.add_argument("--dip_distance_threshold", type=float, default=500000.0, help="Travel distance (mm of painting) before dipping.")

    # Image Processing Parameters
    parser.add_argument("--max_brush_width", type=float, default=8.0, help="Max brush width (mm) for Z mapping.")
    parser.add_argument("--min_path_length_px", type=int, default=1, help="Minimum skeleton path length in px.")
    parser.add_argument("--smooth_window_size", type=int, default=3, help="Smoothing window size.")
    
    # Variable Backlash Compensation Parameters (from variable_backlash.py)
    parser.add_argument("--bx_start", type=float, default=0.5, help="X Backlash at X=0 (mm).")
    parser.add_argument("--bx_end", type=float, default=0.5, help="X Backlash at X=max_travel_x (mm).")
    parser.add_argument("--by_start", type=float, default=3.2, help="Y Backlash at X=0 (mm).")
    parser.add_argument("--by_end", type=float, default=1.6, help="Y Backlash at X=max_travel_x (mm).")
    parser.add_argument("--max_x", type=float, default=150.0, help="Maximum travel distance of the X axis (for backlash calculation).")

    args = parser.parse_args()
    if args.height is None:
        args.height = args.width

    generator_kwargs = {
        'feed_rate': args.feed_rate,
        'x_offset': args.x_offset,
        'y_offset': args.y_offset,
        'dip_location_raw': (args.dip_x, args.dip_y, args.dip_z),
        'dip_duration_s': args.dip_duration,
        'dip_wipe_radius': args.dip_wipe_radius,
        'z_wipe_travel_raw': args.z_wipe_travel,
        'dip_entry_radius': args.dip_entry_radius,
        'remove_drops_enabled': args.remove_drops_enabled,
        'z_global_offset_val': args.z_global_offset,
        'z_safe_raw': args.z_safe,
        'z_safe_dip_raw': args.z_safe_dip,
        # Backlash parameters
        'bx_start': args.bx_start,
        'bx_end': args.bx_end,
        'by_start': args.by_start,
        'by_end': args.by_end,
        'max_travel_x': args.max_x
    }

    try:
        gcode_generator = SkeletonGCodeGenerator(
            z_paint_max_raw=args.z_paint_max,
            z_paint_min_raw=args.z_paint_min,
            max_width_mm=args.max_brush_width,
            min_path_length_px=args.min_path_length_px,
            smooth_window_size=args.smooth_window_size,
            dip_distance_threshold=args.dip_distance_threshold,
            **generator_kwargs
        )

        gcode_generator.generate_from_image(image_path=args.input_image,
                                            target_w_mm=args.width,
                                            target_h_mm=args.height)
        gcode_generator.save(args.output_gcode)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()