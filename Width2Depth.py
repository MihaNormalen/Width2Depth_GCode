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
        self.offset_x = self.x_offset
        self.offset_y = self.y_offset
        # allowed angular variation (degrees) around base direction
        self.remove_drops_angle_variation_deg = 12.0

        self._initial_dip_performed = False

    def remove_drops(self, tray_x, tray_y, x, y):
        """
        Wipe/shake-off motion that leaves the tray in the logical direction of the next stroke.
        'x','y' should be the point representing the direction to continue (e.g. start of next stroke).
        A small angular variation is applied to avoid always wiping the exact same spot.
        """
        # account for offsets if present
        target_x = x + getattr(self, "offset_x", 0)
        target_y = y + getattr(self, "offset_y", 0)

        # vector from tray center to target (direction of continuation)
        dx = target_x - tray_x
        dy = target_y - tray_y
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

        # entry and exit points on the tray circumference
        x1 = tray_x + ux * self.tray_enter_radius
        y1 = tray_y + uy * self.tray_enter_radius
        x2 = tray_x + ux * self.remove_drops_radius
        y2 = tray_y + uy * self.remove_drops_radius

        # perform moves: approach edge, lift to wipe height, slow wipe to outer edge, restore feed
        self.gcode.append(f"; remove_drops start -> dir_angle={math.degrees(angle):.1f}deg")
        self.gcode.append(f"G0 X{x1:.3f} Y{y1:.3f}")
        self.gcode.append(f"G0 Z{self.remove_drops_lift:.3f}")
        self.gcode.append("G1 F600")
        self.gcode.append(f"G1 X{x2:.3f} Y{y2:.3f}")
        self.gcode.append(f"G0 F{self.feed_rate}")
        self.gcode.append("; remove_drops end")

    def _perform_dip(self, target_x=None, target_y=None):
        """
        Perform dip: move to dip location, lower, dwell, lift to wipe height and perform remove_drops.
        If target_x/target_y provided, pass them to remove_drops so wipe follows the next-stroke direction.
        """
        # move up to safe-dip height then to dip location
        self.gcode.append(f"G0 Z{self.z_safe_dip:.3f}")
        self.gcode.append(f"G0 X{self.dip_location[0]:.3f} Y{self.dip_location[1]:.3f}")
        # dip into paint
        self.gcode.append(f"G1 Z{self.dip_location[2]:.3f} F800")
        if self.dip_duration_s > 0:
            self.gcode.append(f"G4 P{int(self.dip_duration_s * 1000)}")

        # lift to wipe height
        wipe_z = self.dip_location[2] + 2.0
        self.gcode.append(f"G1 Z{wipe_z:.3f} F800")

        # decide direction target for wipe
        if target_x is None or target_y is None:
            tx = self.dip_location[0] + self.dip_wipe_radius
            ty = self.dip_location[1]
        else:
            tx = target_x
            ty = target_y

        # perform remove_drops if enabled
        if self.remove_drops_enabled:
            self.remove_drops(
                tray_x=self.dip_location[0],
                tray_y=self.dip_location[1],
                x=tx,
                y=ty
            )

        # retreat and reset feed
        self.gcode.append(f"G1 Z{self.z_safe_dip:.3f} F600")
        self.gcode.append(f"G0 F{self.feed_rate}")

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write("\n".join(self.gcode))
        print(f"G-code saved to {filename}")

    def _initial_setup(self):
        self.gcode.append("G90 ; Set Absolute Positioning")
        self.gcode.append("G21 ; Set Units to Millimeters")
        self.gcode.append(f"G0 Z{self.z_safe:.3f} ; Lift to general safe height")
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
        self.max_width_mm = max(max_brush_width, 0.001)
        print(f"Estimated max brush width: {self.max_width_mm:.2f}mm")

        dip_distance_threshold = self.dip_distance_threshold
        travel_since_last_dip = 0.0

        for path in scaled_toolpaths:
            if not path:
                continue
            # compute start of stroke (world coords)
            start_x, start_y, start_width = path[0]
            start_x += self.x_offset
            start_y += self.y_offset

            # compute only the painting distance (stroke length), do not count moves to start
            stroke_length = 0.0
            prev_x, prev_y = start_x, start_y
            for x_orig, y_orig, _ in path:
                x = x_orig + self.x_offset
                y = y_orig + self.y_offset
                stroke_length += math.hypot(x - prev_x, y - prev_y)
                prev_x, prev_y = x, y

            travel_since_last_dip += stroke_length

            # if threshold reached, dip and pass start of this upcoming stroke so wipe follows it
            if travel_since_last_dip >= dip_distance_threshold:
                self._perform_dip(target_x=start_x, target_y=start_y)
                travel_since_last_dip = 0.0

            # emit G-code for this stroke
            self.gcode.append(f"G0 X{start_x:.3f} Y{start_y:.3f} Z{self.z_safe:.3f} ; move to stroke start")
            start_z = self._width_to_z(start_width)
            self.gcode.append(f"G1 Z{start_z:.3f} F{self.feed_rate / 2} ; lower to paint")
            prev_x, prev_y = start_x, start_y
            for x_orig, y_orig, width in path:
                x = x_orig + self.x_offset
                y = y_orig + self.y_offset
                z = self._width_to_z(width)
                self.gcode.append(f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} F{self.feed_rate} ; paint segment")
                prev_x, prev_y = x, y
            self.gcode.append(f"G0 Z{self.z_safe:.3f} ; lift after stroke")

        # final return and program end
        self.gcode.append(f"G0 X{self.x_offset:.3f} Y{self.y_offset:.3f} Z{self.z_safe_dip:.3f} ; return to origin")
        self.gcode.append("M2 ; End of program")
        print("G-code generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate G-code for CNC painting (skeleton method).")
    parser.add_argument("input_image", help="Path to input image.")
    parser.add_argument("output_gcode", help="Path to output gcode file.")
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
    parser.add_argument("--dip_x", type=float, default=63.0, help="Dip location X (mm).")
    parser.add_argument("--dip_y", type=float, default=0.0, help="Dip location Y (mm).")
    parser.add_argument("--dip_z", type=float, default=0.0, help="Dip Z depth (mm, before offset).")
    parser.add_argument("--dip_duration", type=float, default=0.1, help="Dip dwell time (s).")
    parser.add_argument("--dip_wipe_radius", type=float, default=17.0, help="Wipe radius (mm).")
    parser.add_argument("--z_wipe_travel", type=float, default=1.0, help="Z height for wipe motion (mm, before offset).")
    parser.add_argument("--dip_entry_radius", type=float, default=5.0, help="Dip entry radius (mm).")
    parser.add_argument("--remove_drops_enabled", type=eval, default=True, choices=[True, False], help="Enable remove_drops wipe.")
    parser.add_argument("--max_brush_width", type=float, default=8.0, help="Max brush width (mm) for Z mapping.")
    parser.add_argument("--min_path_length_px", type=int, default=1, help="Minimum skeleton path length in px.")
    parser.add_argument("--smooth_window_size", type=int, default=3, help="Smoothing window size.")
    parser.add_argument("--dip_distance_threshold", type=float, default=500000.0, help="Travel distance (mm of painting) before dipping.")

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
        'z_safe_dip_raw': args.z_safe_dip
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
