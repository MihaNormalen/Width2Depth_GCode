import math
import argparse
import random
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.cluster.vq import kmeans2
from skimage import measure
import time

class PathOptimizer:
    """Clustered-random path order.

    Strokes are grouped into organic, density-shaped clusters with k-means
    (not a fixed grid, so cluster shapes follow the artwork rather than
    square cells). The clusters themselves are visited in random order -
    so the brush works one blob for a while, then jumps to a random other
    blob - but *within* a cluster, strokes are still chained with a
    nearest-neighbor walk so the brush isn't zig-zagging pointlessly while
    it's in one area.
    """
    @staticmethod
    def optimize(paths, start_pos, cluster_size=30):
        if not paths:
            return []

        num_paths = len(paths)

        # One representative point per stroke (its centroid) used only for
        # grouping nearby strokes together.
        centroids = np.array([
            (sum(pt[0] for pt in p) / len(p), sum(pt[1] for pt in p) / len(p))
            for p in paths
        ])

        k = max(1, round(num_paths / max(1, cluster_size)))
        if k <= 1 or num_paths < 4:
            labels = np.zeros(num_paths, dtype=int)
        else:
            try:
                # minit='++' (k-means++) picks well-spread starting centers,
                # which avoids empty clusters and gives organic, blob-shaped
                # regions that follow the density of the artwork.
                _, labels = kmeans2(centroids, k, minit='++', seed=None)
            except Exception:
                labels = np.zeros(num_paths, dtype=int)

        clusters = {}
        for label, path in zip(labels, paths):
            clusters.setdefault(int(label), []).append(path)

        cluster_list = list(clusters.values())
        random.shuffle(cluster_list)  # visit the blobs themselves in random order

        optimized = []
        curr = np.array(start_pos, dtype=float)
        for cluster in cluster_list:
            remaining = list(cluster)
            while remaining:
                idx = min(range(len(remaining)),
                          key=lambda i: math.hypot(remaining[i][0][0] - curr[0], remaining[i][0][1] - curr[1]))
                path = remaining.pop(idx)
                optimized.append(path)
                curr = np.array(path[-1])
        return optimized

class UltraPainter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.gcode = []
        self.dist_since_dip = 0
        # cfg['petri_positions'] is a list of (x, y) tuples — at least one required.
        # Backward compatible: if only dip_x/dip_y given, wrap them into a single-item list.
        if 'petri_positions' in cfg and cfg['petri_positions']:
            self.petri_positions = list(cfg['petri_positions'])
        else:
            self.petri_positions = [(cfg['dip_x'], cfg['dip_y'])]
        self.current_pos = self.petri_positions[0]
        self.current_max_dist = random.uniform(cfg['min_dist'], cfg['max_dist'])
        # For gradually lowering the dip Z depth based on the paint distance
        # traveled so far (less paint left in the dish over time -> the
        # brush needs to dip deeper).
        self.total_paint_dist = 0.0
        self.cum_paint_dist = 0.0

    def _nearest_petri(self, x, y):
        """Return the (px, py) petri position closest to the given point."""
        return min(self.petri_positions, key=lambda p: math.hypot(p[0] - x, p[1] - y))

    def _set_speed(self, mode='travel'):
        c = self.cfg
        accel = c['accel_travel'] if mode == 'travel' else c['accel_paint']
        feed = c['feed'] if mode == 'travel' else c['feed_paint']
        self.gcode.append("M400")
        self.gcode.append(f"M204 P{accel} T{accel}")
        self.gcode.append(f"G1 F{feed}")

    def _perform_dip_and_travel(self, target_x, target_y):
        c = self.cfg
        # Pick whichever petri dish is closest to where the brush currently is,
        # so we don't always travel back to one fixed spot.
        dip_x, dip_y = self._nearest_petri(self.current_pos[0], self.current_pos[1])

        # Linear interpolation of dip depth based on the fraction of paint
        # distance already covered (frac goes from 0 at the start to 1 at
        # the end): less paint in the dish over time -> the brush gradually
        # dips deeper (lower Z).
        frac = 0.0
        if self.total_paint_dist > 0:
            frac = min(1.0, self.cum_paint_dist / self.total_paint_dist)
        dip_z_start = c.get('dip_z_start', c.get('dip_z', 2.0))
        dip_z_end = c.get('dip_z_end', c.get('dip_z', 2.0))
        current_dip_z = dip_z_start + (dip_z_end - dip_z_start) * frac

        self.gcode.append(f"\n; --- DIP CYCLE (petri @ {dip_x:.1f},{dip_y:.1f}, dip_z={current_dip_z:.3f}, frac={frac:.2f}) ---")
        self.gcode.append(f"G0 Z{c['z_low']:.3f} F3000")
        self._set_speed('travel')
        ax, ay = dip_x + random.uniform(-c['dip_jitter'], c['dip_jitter']), dip_y + random.uniform(-c['dip_jitter'], c['dip_jitter'])
        self.gcode.append(f"G0 X{ax:.3f} Y{ay:.3f} Z{c['z_high']:.3f}")
        self.gcode.append(f"G1 Z{current_dip_z:.3f} F3000")
        
        num_steps = int(c['dip_spiral_loops'] * 4)
        for i in range(num_steps):
            ang = i * (math.pi / 2)
            r = (i / num_steps) * c['dip_spiral_r']
            self.gcode.append(f"G1 X{ax + r*math.cos(ang):.3f} Y{ay + r*math.sin(ang):.3f} F2500")
        
        dx, dy = target_x - dip_x, target_y - dip_y
        dist = math.hypot(dx, dy)
        wx = dip_x + (dx/dist * c['wipe_r']) if dist > 0 else dip_x + c['wipe_r']
        wy = dip_y + (dy/dist * c['wipe_r']) if dist > 0 else dip_y
        
        self.gcode.append(f"G0 Z{c['z_wipe_exit']:.3f} F3000")
        self.gcode.append(f"G0 X{wx:.3f} Y{wy:.3f} F{c['feed']}")
        self.gcode.append(f"G0 Z{c['z_high']:.3f} F3000")
        self.gcode.append(f"G0 X{target_x:.3f} Y{target_y:.3f} Z{c['z_low']:.3f} F{c['feed']}")
        
        self.dist_since_dip = 0
        self.current_max_dist = random.uniform(c['min_dist'], c['max_dist'])
        self.current_pos = (target_x, target_y)

    def generate(self, img_path):
        c = self.cfg
        img = Image.open(img_path).convert('L').transpose(Image.FLIP_TOP_BOTTOM)
        img = img.point(lambda p: 0 if p < 140 else 255)
        
        res = 2.0
        tw, th = int(c['target_width']*res), int(c['target_width']*(img.height/img.width)*res)
        img = img.resize((tw, th), Image.Resampling.NEAREST)
        arr = np.array(img) < 140

        raw_paths = []
        step_px = int((c['brush_w'] * (1 - c['overlap'])) * res)
        
        if c['infill_type'] == 'concentric':
            temp_arr = arr.copy()
            erosion_steps = max(1, step_px)
            while temp_arr.any():
                # Trace actual contours — row scanning missed isolated edge pixels on curves
                contours = measure.find_contours(temp_arr.astype(float), 0.5)
                for contour in contours:
                    # contour coords are (row, col) = (y, x)
                    path = [(pt[1] / res + c['x_off'], pt[0] / res + c['y_off'])
                            for pt in contour]
                    if len(path) > 1:
                        raw_paths.append(path)
                temp_arr = ndimage.binary_erosion(temp_arr, iterations=erosion_steps)
        
        else:
            # LINES INFILL - generate lines with a ROTATED PATTERN
            angle_rad = math.radians(c.get('infill_angle', 0))
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            h, w = arr.shape
            cx, cy = w / 2.0, h / 2.0
            
            # Generate lines in the rotated coordinate system
            diag = int(math.hypot(w, h)) + 10
            
            for y_rot in range(-diag, diag, max(1, step_px)):
                line = []
                for x_rot in range(-diag, diag):
                    # Transform from rotated to original coordinates
                    orig_x = int(cx + x_rot * cos_a - y_rot * sin_a)
                    orig_y = int(cy + x_rot * sin_a + y_rot * cos_a)
                    
                    # Check if inside bounds and active
                    if 0 <= orig_x < w and 0 <= orig_y < h and arr[orig_y, orig_x]:
                        # Transform back to real world coordinates
                        real_x = orig_x / res + c['x_off']
                        real_y = orig_y / res + c['y_off']
                        line.append((real_x, real_y))
                    else:
                        if len(line) > 1:
                            raw_paths.append(line)
                        line = []
                
                if len(line) > 1:
                    raw_paths.append(line)

        # OPTIMIZATION
        paths = PathOptimizer.optimize(raw_paths, (c['dip_x'], c['dip_y']), cluster_size=c.get('cluster_size', 30))

        # Total length of all painting strokes (excluding travel moves) -
        # needed in advance so the dip Z depth can be linearly interpolated
        # based on the fraction of the path already covered.
        total_paint_dist = 0.0
        for path in paths:
            for i in range(1, len(path)):
                total_paint_dist += math.hypot(path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
        self.total_paint_dist = total_paint_dist
        self.cum_paint_dist = 0.0

        self.gcode = ["G90", "G21"]
        if paths:
            self.current_pos = self._nearest_petri(paths[0][0][0], paths[0][0][1])
            self._perform_dip_and_travel(paths[0][0][0], paths[0][0][1])
            for path in paths:
                # BIDIRECTIONAL (in both directions)
                dist_to_start = math.hypot(path[0][0] - self.current_pos[0], path[0][1] - self.current_pos[1])
                dist_to_end = math.hypot(path[-1][0] - self.current_pos[0], path[-1][1] - self.current_pos[1])
                if dist_to_end < dist_to_start: path = path[::-1]
                
                self._set_speed('travel')
                self.gcode.append(f"G0 X{path[0][0]:.3f} Y{path[0][1]:.3f} Z{c['z_low']:.3f}")
                self._set_speed('paint')
                self.gcode.append(f"G1 Z{c['z_paint']:.3f} F2500")
                self.gcode.append(f"G1 F{c['feed_paint']}")  # restore paint feedrate after Z plunge overrode it
                self.current_pos = path[0]
                
                for i in range(1, len(path)):
                    px, py = path[i]
                    dist = math.hypot(px - self.current_pos[0], py - self.current_pos[1])
                    if (self.dist_since_dip + dist) > self.current_max_dist:
                        self.gcode.append(f"G0 Z{c['z_low']:.3f} F3000")
                        self._perform_dip_and_travel(px, py)
                        self._set_speed('paint')
                        self.gcode.append(f"G1 Z{c['z_paint']:.3f} F2500")
                        self.gcode.append(f"G1 F{c['feed_paint']}")  # restore paint feedrate after Z plunge overrode it
                    self.gcode.append(f"G1 X{px:.3f} Y{py:.3f}")
                    self.dist_since_dip += dist
                    self.cum_paint_dist += dist
                    self.current_pos = (px, py)
                self.gcode.append(f"G0 Z{c['z_low']:.3f} F3000")

        self.gcode.append("M400")
        self.gcode.append(f"G0 Z{c['z_high']:.3f} F3000\nM2")
        return "\n".join(self.gcode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input"); parser.add_argument("output")
    args = parser.parse_args()

    config = {
        'target_width': 770,
        'brush_w':      1.0, 
        'overlap':      0.0,
        'infill_type':  'lines',
        'infill_angle': 0.0,    # Arbitrary angle: 0, 45, 90, -45, etc.
        # Roughly how many strokes get painted before jumping to a random
        # new cluster elsewhere on the image. Lower = more frequent jumps
        # (smaller work areas), higher = brush lingers longer per area.
        'cluster_size': 500,
        'min_dist':     80.0,
        'max_dist':     90.0,
        'z_paint':      0.0,
        'z_low':        3.0,
        'z_high':       24.0,
        'z_wipe_exit':  17.4,
        'x_off':        230.0,
        'y_off':        65.0,
        # List of petri/paint-dip positions on the bed. The brush will always
        # dip into whichever one is closest to its current location, instead
        # of always traveling back to a single fixed spot.
        'petri_positions': [
            (77.0, 630.0),
            (94.7, 1906.0),   # add as many as you have, e.g. one near each side of the bed
        ],
        # Kept for backward compatibility (used only if 'petri_positions' is empty/missing)
        'dip_x':        77.0,
        'dip_y':        630.0,
        # The dip depth changes linearly over the whole process, from
        # dip_z_start (at the beginning, dish full of paint) to dip_z_end
        # (towards the end, when there's less paint in the dish -> deeper dip).
        'dip_z_start':  2.0,
        'dip_z_end':    1.0,
        'dip_jitter':   35.0,
        'dip_spiral_loops': 1.0,
        'dip_spiral_r': 25.0,
        'wipe_r':       67.0,
        'feed':         80000,
        'feed_paint':   28000,
        'accel_travel': 1200, 
        'accel_paint':  600,  
    }

    painter = UltraPainter(config)
    with open(args.output, "w") as f:
        f.write(painter.generate(args.input))