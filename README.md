# Width2Depth_GCode
**Width2Depth_GCode** is a Python tool that maps image line width (via skeletonization) to a variable Z-axis depth for CNC G-code generation. Creates dynamic, pressure-sensitive strokes for artistic CNC plotting/painting.

---

## ✨ Features

* **Adaptive Z-Axis Control:** Converts line intensity/width from an image (derived from skeletonization and distance maps) into a variable Z-axis position (depth or pressure).
* **Dynamic Strokes:** Generates G-code that mimics dynamic brushstrokes, where wider sections of the source image result in deeper or firmer tool contact.
* **Complex Path Generation:** Uses advanced image processing libraries (`scikit-image`, `sknw`) to create efficient, centerline-based toolpaths (skeletonization).
* **Automatic Tool Dipping:** Supports configuration for automatic tool dipping and wiping cycles, crucial for fluid-based applications (e.g., CNC painting).
* **Path Smoothing:** Includes path smoothing options (`smooth_window_size`) to reduce chatter and improve the quality of G-code motion.
* **Safety Offsets:** Allows setting global Z-offsets, safe Z-heights (`z_safe_raw`), and specific Z-heights for dip cycles.

## 🛠️ Prerequisites

The script requires Python 3.x and several external libraries. You can typically install the dependencies using `pip`:

bash
pip install numpy opencv-python scikit-image sknw scipy argparse


python Width2Depth.py <INPUT_IMAGE> <OUTPUT_GCODE> [OPTIONS]

python Width2Depth.py input.png output.gcode --width 300 --height 300 --z_paint_max -0.5 --z_paint_min -0.1 --feed_rate 1500

Argument,Description,Default
INPUT_IMAGE,"Path to the source image file (e.g., PNG, JPEG).",N/A
OUTPUT_GCODE,Path to the final G-code output file.,N/A
--width,Physical width of the drawing area (mm).,Mandatory
--height,Physical height of the drawing area (mm).,Defaults to --width
--z_paint_max,"Maximum painting depth/pressure (e.g., -0.5mm).",0.0
--z_paint_min,"Minimum painting depth/pressure (e.g., -0.1mm).",0.0
--max_brush_width,Maximum width of the brush/tool in millimeters.,1.0
--feed_rate,G1 feed rate in mm/min.,500
--z_safe,Safe Z-height for rapid moves.,5.0
"--dip_x, --dip_y, --dip_z",Coordinates for the tool dipping station (for automatic refills).,0.0

