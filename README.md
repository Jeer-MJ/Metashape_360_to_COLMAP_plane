# Metashape 360° to COLMAP / Licht-Feld Studio Converter

## Overview
Convert Agisoft Metashape equirectangular (spherical) camera exports into:
- **COLMAP text format** (`cameras.txt`, `images.txt`, `points3D.txt`) with rectilinear cubemap crops per frame.
- **Licht-Feld Studio format** (`transforms.json`) with optional cubemap split (PINHOLE crops) or equirectangular passthrough.

Optional PLY point cloud is converted and included in the output (requires Open3D).

Refer to the detail workflow
- [[English] Easy & Fast 3D Gaussian Splatting workflow with 360 Camera](https://zenn.dev/kotohibi/articles/409bc16876b9e0)
- [[日本語] Easy & Fast 3D Gaussian Splatting workflow with 360 Camera](https://zenn.dev/kotohibi/articles/28b137f1873921)

## Features

### COLMAP mode (`metashape_360_to_colmap.py`)
- Equirectangular → Cubemap: 6 rectilinear 90° crops per frame (top/front/right/back/left/bottom), multi-process available
- Writes COLMAP `cameras.txt`, `images.txt`, `points3D.txt`
- Optional PLY point cloud export (requires Open3D)
- Adjustable FoV and crop size
- Optional image-count cap and range selection for quick tests
- YOLO-based mask generation (person, car, and other COCO classes)
- Overexposure (white-blown-out) pixel masking
- Z-axis 180° rotation option for PostShot coordinate system compatibility
- Per-frame yaw offset for improved 3DGS training stability

### Licht-Feld Studio mode (`metashape_360_lfs.py`)
- Outputs `transforms.json` compatible with Licht-Feld Studio
- **Equirectangular passthrough** (default): one entry per source frame, EQUIRECTANGULAR camera model
- **Cubemap split** (`--split-cubemap`): generates 6 PINHOLE crops per frame with correct camera-to-world transforms composed from the source pose
- Adjustable crop size and FoV for split mode
- Selective face skipping (`--skip-directions`) applies only to split output
- YOLO-based mask generation and overexposure masking, same as COLMAP mode

## Requirements
- Agisoft Metashape Standard (https://www.agisoft.com/features/standard-edition/)
- Python 3.9+
- `numpy`, `pillow`, `opencv-python`
- Optional: `ultralytics` (YOLO mask generation), `open3d` (PLY → points3D)

## Usage

### Step 1 — SfM in Metashape
- Set camera type: [Tools] → [Camera Calibration] → [Camera type] → [Spherical]

### Step 2 — Export from Metashape
- Cameras XML: [File] → [Export Cameras...] → select XML type
- Point cloud PLY: [File] → [Export Point Cloud...] → select PLY type

---

### COLMAP mode CLI (`metashape_360_to_colmap.py`)

```bash
python metashape_360_to_colmap.py \
  --images /path/to/equirect_frames \
  --xml /path/to/metashape_cameras.xml \
  --output /path/to/output_colmap \
  --ply /path/to/pointcloud.ply \
  --crop-size 1920 \
  --fov-deg 90 \
  --num-workers 4 \
  --max-images 50 \
  --yaw-offset 30 \
  --generate-masks \
  --yolo-classes 0,2,5 \
  --yolo-conf 0.25 \
  --mask-overexposure \
  --overexposure-threshold 250 \
  --overexposure-dilate 5 \
  --rotate-z180
```

---

### Licht-Feld Studio mode CLI (`metashape_360_lfs.py`)

**Equirectangular passthrough (default):**
```bash
python metashape_360_lfs.py \
  --xml /path/to/metashape_cameras.xml \
  --images /path/to/equirect_frames \
  --output /path/to/output_lfs
```

**Cubemap split — 6 PINHOLE crops per frame:**
```bash
python metashape_360_lfs.py \
  --xml /path/to/metashape_cameras.xml \
  --images /path/to/equirect_frames \
  --output /path/to/output_lfs \
  --split-cubemap \
  --crop-size 1920 \
  --fov-deg 90.0 \
  --skip-directions bottom,top \
  --generate-masks \
  --yolo-classes 0 \
  --mask-overexposure
```

Input: 100 equirectangular images → Output: 600 PNG crops + `transforms.json` (PINHOLE model, 6 frames per source).

---

### GUI app (`metashape_360_gui.py`)
If you prefer interactive operation, launch the GUI:

```bash
python metashape_360_gui.py
```

On Windows, you can also launch the GUI directly with [launch_gui.bat](launch_gui.bat). It starts the app with `env\Scripts\pythonw.exe` when available, and falls back to `env\Scripts\python.exe` or `.venv` equivalents.

Main points:
- `Input/Output Paths`: Select image folder, XML, optional PLY, and output folder.
- `Mode selector`: Switch between COLMAP and Licht-Feld Studio modes at the top of the window.
- `Processing Options` / `Skip Directions` / `Mask Generation`: Available as tabs to reduce vertical space.
- `Advanced` (LFS mode): Collapsible section with `Split into cubemap faces` checkbox. When enabled, Crop Size and FoV spinboxes become active.
- `Dev Options`: Collapsible section (closed by default) for less frequently used options.
- `Run Conversion`: Executes conversion with live progress logs in the GUI output panel.
- `Stop`: Stops the running process and re-enables `Run Conversion`.
- `Save Config` / `Load Config`: Save/load settings as `config.txt`-style files.

Screenshot:


<img src="docs/images/gui_main.png" alt="GUI Screenshot" width="640" />

### Using a Configuration File
You can specify options in `config.txt` file instead of command-line arguments. Create a `config.txt` in the same directory as the script:

```
# config.txt example
images=./equirect/
xml=./cameras.xml
output=./colmap_dataset/
ply=./dense.ply
crop-size=1920
fov-deg=90.0
num-workers=4
max-images=10000
yaw-offset=30
generate-masks=True
```

**For paths with spaces (Windows users):** Enclose paths in quotes (single or double):
```
images="D:\My Documents\equirect frames"
xml="C:\Program Files\project\cameras.xml"
output='D:\Output Folder\colmap'
```

**Priority:** Command-line arguments > config.txt > default values

If you specify an option on the command line, it will override the value in config.txt. See [config.txt.example](config.txt.example) for all available options.

### Key options — COLMAP mode
- `--images` (req): Directory of equirectangular images
- `--xml` (req): Metashape XML export (cameras)
- `--output` (req): Output folder (creates `images/`, `masks/`, `cameras.txt`, `images.txt`, `points3D.txt`)
- `--ply`: Optional PLY to export `points3D.txt` and `points3D.ply`
- `--crop-size`: Crop resolution (square). Default 1920.
- `--fov-deg`: Horizontal FoV of rectilinear crops. Default 90.
- `--max-images`: Limit number of source equirects for quick tests (default 10000)
- `--range-images`: Range of images to process (format: `START-END`, e.g., `10-50`). 0-based index, inclusive.
- `--num-workers`: Number of parallel workers for image reframing (default 4)
- `--skip-directions`: Comma-separated list of directions to skip (top, front, right, back, left, bottom)
- `--generate-masks`: Generate masks for specified objects using YOLO
- `--yolo-classes`: Comma-separated YOLO class IDs (default: `0`). Examples: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
- `--yolo-conf`: Minimum YOLO confidence score `0.0-1.0` (default: `0.25`)
- `--invert-mask`: Invert mask color from BLACK to WHITE
- `--mask-overexposure`: Mask white-blown-out pixels (all RGB channels above threshold)
- `--overexposure-threshold`: Pixel value threshold 0-255 for overexposure detection (default: 250)
- `--overexposure-dilate`: Dilation radius in pixels to cover fringe artifacts (default: 5)
- `--yaw-offset`: Yaw rotation offset in degrees applied per frame to cubemap extraction (default: 0.0)
- `--rotate-z180`: Rotate scene 180° around Z-axis for PostShot compatibility (default: on). Use `--no-rotate-z180` to disable.

### Key options — Licht-Feld Studio mode
- `--images` (req): Directory of equirectangular images
- `--xml` (req): Metashape XML export (cameras)
- `--output` (req): Output folder (creates `images/`, `masks/`, `transforms.json`)
- `--split-cubemap`: Enable cubemap split; generates 6 PINHOLE crops per source frame (default: off)
- `--crop-size`: Size in pixels of each cubemap face when split is enabled (default: 1920)
- `--fov-deg`: Horizontal FoV in degrees for each cubemap face (default: 90.0)
- `--skip-directions`: Comma-separated face names to omit when split is enabled (top, front, right, back, left, bottom)
- `--max-images`: Limit number of source equirects processed
- `--no-copy-images`: Skip copying source images; only write `transforms.json`
- `--generate-masks`: Generate masks using YOLO (same flags as COLMAP mode)
- `--yolo-classes`, `--yolo-conf`, `--invert-mask`, `--mask-overexposure`, `--overexposure-threshold`, `--overexposure-dilate`: Same behavior as COLMAP mode

### Outputs — COLMAP mode
- `output/images/`: Cubemap crops (6 per input frame by default)
- `output/masks/`: Mask images (when `--generate-masks` is specified)
- `output/tmp/`: Temporary files for mask generation; safe to delete after conversion
- `output/cameras.txt`
- `output/images.txt`
- `output/points3D.txt` (+ `points3D.ply` when `--ply` is given)

### Outputs — Licht-Feld Studio mode
- `output/images/`: Equirectangular images (passthrough) or PINHOLE crop PNGs (split mode, 6 per source frame)
- `output/masks/`: Mask images (when `--generate-masks` is specified)
- `output/transforms.json`: Camera poses and intrinsics in Licht-Feld Studio format

### YOLO COCO Classes Reference
Common COCO dataset class IDs for `--yolo-classes`:
- 0: person
- 1: bicycle
- 2: car
- 3: motorcycle
- 5: bus
- 7: truck
- 16: dog
- 17: cat
- [Full COCO class list](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)

## Notes
- I confirmed that it worked with PostShot for 3DGS train.
- When using PostShot for 3DGS training, use `--rotate-z180` to fix coordinate system differences (180° rotation around Z-axis).
- Only spherical sensors are supported; uses the first component transform when multiple are present.
- Intrinsics per crop are PINHOLE with `fx=fy=(w/2)/tan(fov/2)`, `cx=cy=w/2`.
- If orientations look wrong, verify top/front/right/back/left/bottom yaw definitions and FoV.

## License
MIT
