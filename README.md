# Width2Depth_GCode

Python tools that turn a black-and-white picture into G-code for a **brush-painting CNC machine**. The main idea:
the centre line of every shape is painted as one stroke, and the brush goes **deeper where the line is wider**. That
gives real, pressure-like brush strokes instead of a constant pen line. The scripts also handle refilling the brush:
a dip in a paint dish (petri dish) every so many mm of painting, with spiral, jitter and wipe moves.

---

## Which script to use

| Folder | Script | What it does |
| :--- | :--- | :--- |
| `infill/` | **`multi_random_path_small_dipZ.py`** ⭐ newest | Fills the shapes with straight lines at any angle, or with rings. Strokes are grouped into blob-shaped clusters (k-means) that are visited in random order. The brush dips in whichever **dish is closest**, and the **dip gets gradually deeper** as the paint in the dishes runs low. Settings in the script are for a large machine (770 mm wide painting, two dishes). |
| `tutti_frutti/` | **`W2D_tf_papa2.py`** ⭐ newest width-to-depth | Width-to-depth strokes for up to **4 colours** (one dish each) and a water dish for rinsing (`--water_enabled`). "Tutti-frutti" mode (`--tf_enabled True`) dips in a random colour dish each time, without repeating the last one. Stroke order optimisation, separate travel / paint speeds and accelerations. |
| | `Width2Depth_tutti-frutti.py` | Earlier version of the above, without the water-rinse option. |
| `width2depth/` | `Width2Depth.py` | The original width-to-depth generator. Dips in the middle of a stroke when the paint distance is used up. |
| | `wide2depthSpiral_distanc.py` | Same, but always finishes the stroke and dips **before** the next one (no dip in the middle of a line). |
| | `D2Woptimized.py` | Width-to-depth with path optimisation, drop removal (`--remove_drops_enabled`) and a dip every `--dip_distance_threshold` mm. |
| | `Width2Depth_with_backlash_fix.py` | Width-to-depth with a variable backlash compensation for X and Y (`--bx_start`, `--bx_end`, `--by_start`, `--by_end`). |
| `web/` | `PaintByNumbers2_GRBL.html` | "Bacterial Painter": a browser tool (open the file, no install) that makes GRBL G-code from a picture. |
| `examples/` | `poletna.png`, `obrazMoski.jpeg` | Pictures to try. |

## Install

Python 3.8 or newer:

```
pip install -r requirements.txt
```

## Usage

### Width-to-depth scripts

```
python width2depth/Width2Depth.py examples/poletna.png out.gcode --width 300 --z_paint_max -0.5 --z_paint_min -0.1 --feed_rate 1500
```

Every option has a default; `python <script> --help` lists them. The most important ones:

| Option | Meaning |
| :--- | :--- |
| `--width`, `--height` | size of the painting in mm (height follows the picture if left out) |
| `--x_offset`, `--y_offset` | where the painting starts on the machine |
| `--z_paint_min`, `--z_paint_max` | Z for the thinnest and for the widest line (the brush depth in between follows the width) |
| `--max_brush_width` | the widest line the brush can make, mm |
| `--z_safe`, `--z_global_offset` | travel height; one offset added to every Z |
| `--dip_x`, `--dip_y`, `--dip_z` | the paint dish (the tutti-frutti scripts use `--color1_dip_x` … `--color4_dip_z` and `--water_dip_*`) |
| `--smooth_window_size`, `--min_path_length_px` | smoothing of the strokes, shortest stroke kept |

### multi_random_path_small_dipZ.py

The settings are in the `config` block at the bottom of the script (painting size, brush width, line angle,
cluster size, dish positions, `dip_z_start` / `dip_z_end`, speeds). Edit them, then:

```
python infill/multi_random_path_small_dipZ.py picture.png out.gcode
```

## How width becomes depth

1. The picture is turned into black and white and skeletonised: every shape becomes its centre line.
2. The distance transform gives the line width at every point of that centre line.
3. Each width is mapped between `--z_paint_min` (thin) and `--z_paint_max` (wide), and every G-code point gets
   its own Z.
4. Strokes are smoothed, ordered to keep travel short, and split by dips in the paint dish.

## License

[Apache 2.0](LICENSE)
