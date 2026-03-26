#!/usr/bin/env python3
"""
Standalone Metashape to LichtFeld Converter

Converts Metashape XML camera poses + PLY point cloud to LichtFeld-compatible
transforms.json format, without requiring nerfstudio.

Dependencies:
    pip install numpy open3d

Optional (required only for --split-cubemap):
    pip install pillow opencv-python

Usage:
    python metashape_to_lichtfeld.py --images ./images/ --xml cameras.xml --ply sparse.ply --output ./output/
    python metashape_to_lichtfeld.py --images ./images/ --xml cameras.xml --split-cubemap --crop-size 1920

Based on nerfstudio's metashape_utils.py (Apache 2.0 License)
"""

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

# Try to import open3d, fall back to plyfile if not available
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    try:
        from plyfile import PlyData, PlyElement
        HAS_PLYFILE = True
    except ImportError:
        HAS_PLYFILE = False

# Optional image-processing libs — only required when --split-cubemap or --generate-masks is used.
try:
    from PIL import Image as _PILImage
    import cv2 as _cv2
    HAS_IMAGE_LIBS = True
except ImportError:
    HAS_IMAGE_LIBS = False

# Optional YOLO — only required when --generate-masks is used.
try:
    from ultralytics import YOLO as _YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

# Native SAM3 predictor (Meta SAM3 model via ultralytics).
# Supports text-concept segmentation with set_image() + predictor(text=...) API.
try:
    from ultralytics.models.sam import SAM3SemanticPredictor as _SAM3SemanticPredictor
    HAS_SAM3_NATIVE = True
except ImportError:
    _SAM3SemanticPredictor = None
    HAS_SAM3_NATIVE = False

# SAM3 engine is available if either ultralytics YOLOE or the native SAM3 predictor is.
HAS_SAM3 = HAS_YOLO or HAS_SAM3_NATIVE


def _lfs_load_open_vocab_model(
    model_path: str,
    conf: float = 0.25,
    half: bool = False,
) -> Any:
    """Load an open-vocabulary segmentation model.

    Tries loaders in this order:
      1. YOLOE / YOLOWorld / YOLO  — lightweight ultralytics models (~67 MB).
      2. SAM3SemanticPredictor     — native Meta SAM3 model (~3.3 GB), which
         uses set_image() + predictor(text=...) inference API.

    The ``conf`` and ``half`` parameters are forwarded to SAM3SemanticPredictor
    via its ``overrides`` dict (they are ignored for YOLOE-family models where
    these are set per-inference call instead).
    """
    import importlib
    ul = importlib.import_module("ultralytics")
    last_exc: Optional[Exception] = None

    # --- Try all YOLOE-family loaders first (fast, ~67 MB models) ---
    for cls_name in ("YOLOE", "YOLOWorld", "YOLO"):
        cls = getattr(ul, cls_name, None)
        if cls is None:
            continue
        try:
            return cls(model_path)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

    # --- Fall back to native SAM3SemanticPredictor (Meta SAM3 checkpoint) ---
    # Meta's SAM3 checkpoint has 'detector'/'tracker' keys that YOLOE cannot
    # load.  SAM3SemanticPredictor is the correct class for that format.
    if _SAM3SemanticPredictor is not None:
        try:
            import torch
            overrides = dict(
                conf=conf,
                task="segment",
                mode="predict",
                model=model_path,
                half=half and torch.cuda.is_available(),
                verbose=False,
                save=False,
            )
            return _SAM3SemanticPredictor(overrides=overrides)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

    raise RuntimeError(
        f"Could not load open-vocabulary model '{model_path}'. "
        f"Last error: {last_exc}. "
        "Ensure ultralytics>=8.3 is installed: pip install -U ultralytics"
    )

# ---------------------------------------------------------------------------
# Cubemap-split helpers (ported from metashape_360_to_colmap.py)
# ---------------------------------------------------------------------------

_ALL_DIRECTIONS = ["top", "front", "right", "back", "left", "bottom"]

_DIRECTION_YAW_DEG: Dict[str, float] = {
    "top": 0.0, "front": 0.0, "right": -90.0,
    "back": 180.0, "left": 90.0, "bottom": 0.0,
}
_DIRECTION_PITCH_DEG: Dict[str, float] = {
    "top": 90.0, "front": 0.0, "right": 0.0,
    "back": 0.0, "left": 0.0, "bottom": -90.0,
}

# Per-process remap-map cache for cubemap splitting.
_lfs_remap_cache: Dict[tuple, Tuple[Any, Any]] = {}


def _lfs_get_face_rotation(direction: str) -> np.ndarray:
    """3x3 rotation matrix that orients the camera toward the given cubemap face."""
    yaw = np.radians(_DIRECTION_YAW_DEG[direction])
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R_yaw = np.array([
        [cos_y, 0.0, sin_y],
        [0.0,   1.0, 0.0  ],
        [-sin_y, 0.0, cos_y],
    ])
    pitch = np.radians(_DIRECTION_PITCH_DEG[direction])
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    R_pitch = np.array([
        [1.0, 0.0,   0.0  ],
        [0.0, cos_p, -sin_p],
        [0.0, sin_p,  cos_p],
    ])
    return R_yaw @ R_pitch


def _lfs_compute_remap_maps(
    direction: str,
    crop_size: int,
    fov_deg: float,
    equirect_w: int,
    equirect_h: int,
) -> Tuple[Any, Any]:
    """Compute (map_x, map_y) sampling arrays for cv2.remap (equirect → perspective)."""
    w_out = h_out = crop_size
    fx = fy = (w_out / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    cx = cy = (w_out - 1) / 2.0

    u, v = np.meshgrid(
        np.arange(w_out, dtype=np.float32),
        np.arange(h_out, dtype=np.float32),
    )
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = np.ones_like(x)
    dirs = np.stack([x, y, z], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    R = _lfs_get_face_rotation(direction).astype(np.float32)
    dirs = dirs @ R.T

    lon = np.arctan2(dirs[..., 0], dirs[..., 2])
    lat = np.arctan2(
        dirs[..., 1],
        np.sqrt(dirs[..., 0] ** 2 + dirs[..., 2] ** 2),
    )

    # flip_vertical=True matches the equirectangular convention used by Metashape
    map_x = (lon / (2 * np.pi) + 0.5) * float(equirect_w)
    map_y = (0.5 + lat / np.pi) * float(equirect_h)
    map_y = np.clip(map_y, 0.0, float(equirect_h - 1))

    return map_x.astype(np.float32), map_y.astype(np.float32)


def _lfs_crop_face(
    equirect_image: Any,
    direction: str,
    crop_size: int,
    fov_deg: float,
) -> Any:
    """Rectilinear crop from equirectangular using cv2.remap. Returns PIL Image."""
    width, height = equirect_image.size
    key = (direction, crop_size, fov_deg, width, height)
    if key not in _lfs_remap_cache:
        _lfs_remap_cache[key] = _lfs_compute_remap_maps(
            direction, crop_size, fov_deg, width, height
        )
    map_x, map_y = _lfs_remap_cache[key]

    equirect_np = np.array(equirect_image.convert("RGB"))
    sampled = _cv2.remap(
        equirect_np,
        map_x,
        map_y,
        interpolation=_cv2.INTER_LINEAR,
        borderMode=_cv2.BORDER_WRAP,
    )
    return _PILImage.fromarray(sampled, mode="RGB")


# ---------------------------------------------------------------------------
# Mask-generation helpers (ported from metashape_360_to_colmap.py)
# ---------------------------------------------------------------------------

def _lfs_create_overexposure_mask(
    image: Any,
    threshold: int = 250,
    dilate_pixels: int = 5,
) -> np.ndarray:
    """Return uint8 (H, W) array where 255 marks overexposed pixels.

    A pixel is overexposed when all three RGB channels are >= threshold.
    The result is dilated by dilate_pixels to cover bloom halation.
    """
    img_np = np.array(image)
    blown = np.all(img_np >= threshold, axis=-1).astype(np.uint8) * 255
    if dilate_pixels > 0:
        kernel_size = dilate_pixels * 2 + 1
        kernel = _cv2.getStructuringElement(
            _cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        blown = _cv2.dilate(blown, kernel, iterations=1)
    return blown


def _lfs_generate_mask(
    image: Any,
    yolo_model: Optional[Any],
    yolo_conf: float = 0.25,
    invert_mask: bool = False,
    class_ids: Optional[List[int]] = None,
    mask_overexposure: bool = False,
    overexposure_threshold: int = 250,
    overexposure_dilate: int = 5,
) -> Any:
    """Generate a combined YOLO + overexposure binary mask for one PIL image.

    Works identically for full equirectangular images and individual cubemap
    face crops — callers are responsible for passing the correct image.

    Returns:
        PIL Image (mode "L") where, by default (invert_mask=False):
            255 = background (keep for training)
              0 = masked region (remove from training)
        When invert_mask=True the polarity is flipped.
    """
    h, w = np.array(image).shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)

    # --- YOLO person/object segmentation ---
    if yolo_model is not None:
        target_classes = class_ids if class_ids else [0]
        results = yolo_model(image, verbose=False, conf=yolo_conf)
        for result in results:
            if result.masks is not None:
                for i, cls in enumerate(result.boxes.cls):
                    if int(cls) in target_classes:
                        mask_data = result.masks.data[i].cpu().numpy().astype(np.float32)
                        mask_resized = _cv2.resize(
                            mask_data, (w, h), interpolation=_cv2.INTER_LINEAR
                        )
                        combined = np.maximum(
                            combined, (mask_resized * 255).astype(np.uint8)
                        )

    # --- Overexposure mask ---
    if mask_overexposure:
        overexp = _lfs_create_overexposure_mask(
            image,
            threshold=overexposure_threshold,
            dilate_pixels=overexposure_dilate,
        )
        combined = np.maximum(combined, overexp)

    # Invert polarity: default is background=white, masked=black
    if not invert_mask:
        combined = 255 - combined

    return _PILImage.fromarray(combined, mode="L")


def _lfs_generate_mask_sam3(
    image: Any,
    model: Any,
    concepts: List[str],
    conf: float = 0.25,
    invert_mask: bool = False,
    mask_overexposure: bool = False,
    overexposure_threshold: int = 250,
    overexposure_dilate: int = 5,
) -> Any:
    """Generate a mask using an open-vocabulary model with text concept prompts.

    Supports two backends:
      • SAM3SemanticPredictor (native Meta SAM3): uses set_image(bgr_array)
        followed by predictor(text=concepts).  Confidence is already baked into
        the predictor's overrides at load time.
      • YOLOE / YOLOWorld / YOLO: uses set_classes(concepts) + model(image).
        Falls back to texts= kwarg or plain inference if set_classes is absent.

    Returns:
        PIL Image (mode "L") with default polarity: 255 = keep, 0 = masked.
    """
    h, w = np.array(image).shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)

    if _SAM3SemanticPredictor is not None and isinstance(model, _SAM3SemanticPredictor):
        # Native Meta SAM3: convert PIL → BGR numpy array then run via predictor API.
        img_bgr = _cv2.cvtColor(np.array(image), _cv2.COLOR_RGB2BGR)
        model.set_image(img_bgr)
        results = model(text=concepts)
    else:
        # YOLOE-family: set text concepts then run standard ultralytics inference.
        try:
            model.set_classes(concepts)
            results = model(image, verbose=False, conf=conf)
        except (AttributeError, TypeError):
            # Model does not support set_classes — try texts= keyword directly.
            try:
                results = model(image, texts=concepts, verbose=False, conf=conf)
            except TypeError:
                results = model(image, verbose=False, conf=conf)

    for result in results:
        if result.masks is not None:
            for mask_data in result.masks.data:
                mask_np = mask_data.cpu().numpy().astype(np.float32)
                mask_r = _cv2.resize(mask_np, (w, h), interpolation=_cv2.INTER_LINEAR)
                combined = np.maximum(combined, (mask_r * 255).astype(np.uint8))
        elif result.boxes is not None:
            # No segmentation masks — rasterise bounding boxes as rectangular regions.
            for bbox in result.boxes.xyxy:
                x1, y1, x2, y2 = (int(v) for v in bbox.cpu().numpy())
                combined[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 255

    if mask_overexposure:
        overexp = _lfs_create_overexposure_mask(
            image,
            threshold=overexposure_threshold,
            dilate_pixels=overexposure_dilate,
        )
        combined = np.maximum(combined, overexp)

    if not invert_mask:
        combined = 255 - combined

    return _PILImage.fromarray(combined, mode="L")


def find_param(calib_xml: ET.Element, param_name: str) -> float:
    """Find a parameter in calibration XML, return 0.0 if not found."""
    param = calib_xml.find(param_name)
    if param is not None and param.text:
        return float(param.text)
    return 0.0


def parse_metashape_xml(xml_path: Path) -> Dict[str, Any]:
    """
    Parse Metashape XML file to extract sensors and camera transforms.
    
    Returns:
        Dictionary containing sensor_dict, camera_model, frames list
    """
    xml_tree = ET.parse(xml_path)
    root = xml_tree.getroot()
    chunk = root[0]
    sensors = chunk.find("sensors")
    
    if sensors is None:
        raise ValueError("No sensors found in Metashape XML")
    
    # Find calibrated sensors
    calibrated_sensors = [
        sensor for sensor in sensors.iter("sensor")
        if sensor.get("type") == "spherical" or sensor.find("calibration")
    ]
    
    if not calibrated_sensors:
        raise ValueError("No calibrated sensor found in Metashape XML")
    
    # Check sensor types are consistent
    sensor_types = [s.get("type") for s in calibrated_sensors]
    if sensor_types.count(sensor_types[0]) != len(sensor_types):
        raise ValueError(
            "All Metashape sensors must have the same type. "
            "Mixed camera types are not supported."
        )
    
    # Map Metashape sensor type to camera model string
    sensor_type = sensor_types[0]
    if sensor_type == "frame":
        camera_model = "PINHOLE"
    elif sensor_type == "fisheye":
        camera_model = "OPENCV_FISHEYE"
    elif sensor_type == "spherical":
        camera_model = "EQUIRECTANGULAR"
    else:
        raise ValueError(f"Unsupported Metashape sensor type: {sensor_type}")
    
    # Parse sensor calibration data
    sensor_dict = {}
    for sensor in calibrated_sensors:
        s = {}
        resolution = sensor.find("resolution")
        if resolution is None:
            raise ValueError("Resolution not found in Metashape XML")
        
        s["w"] = int(resolution.get("width"))
        s["h"] = int(resolution.get("height"))
        
        calib = sensor.find("calibration")
        if calib is None:
            # Spherical sensors may not have calibration
            if sensor_type == "spherical":
                s["fl_x"] = s["w"] / 2.0
                s["fl_y"] = s["h"]
                s["cx"] = s["w"] / 2.0
                s["cy"] = s["h"] / 2.0
            else:
                raise ValueError(f"No calibration found for sensor {sensor.get('id')}")
        else:
            f = calib.find("f")
            if f is None or f.text is None:
                raise ValueError("Focal length not found in Metashape XML")
            s["fl_x"] = s["fl_y"] = float(f.text)
            s["cx"] = find_param(calib, "cx") + s["w"] / 2.0
            s["cy"] = find_param(calib, "cy") + s["h"] / 2.0
            
            # Distortion parameters
            s["k1"] = find_param(calib, "k1")
            s["k2"] = find_param(calib, "k2")
            s["k3"] = find_param(calib, "k3")
            s["k4"] = find_param(calib, "k4")
            s["p1"] = find_param(calib, "p1")
            s["p2"] = find_param(calib, "p2")
        
        sensor_dict[sensor.get("id")] = s
    
    # Parse component transforms (for multi-chunk projects)
    components = chunk.find("components")
    component_dict = {}
    if components is not None:
        for component in components.iter("component"):
            transform = component.find("transform")
            if transform is not None:
                rotation = transform.find("rotation")
                if rotation is None or rotation.text is None:
                    r = np.eye(3)
                else:
                    r = np.array([float(x) for x in rotation.text.split()]).reshape((3, 3))
                
                translation = transform.find("translation")
                if translation is None or translation.text is None:
                    t = np.zeros(3)
                else:
                    t = np.array([float(x) for x in translation.text.split()])
                
                scale = transform.find("scale")
                if scale is None or scale.text is None:
                    s = 1.0
                else:
                    s = float(scale.text)
                
                m = np.eye(4)
                m[:3, :3] = r
                m[:3, 3] = t / s
                component_dict[component.get("id")] = m
    
    # Parse camera frames
    cameras = chunk.find("cameras")
    if cameras is None:
        raise ValueError("No cameras found in Metashape XML")
    
    return {
        "sensor_dict": sensor_dict,
        "component_dict": component_dict,
        "cameras": cameras,
        "camera_model": camera_model
    }


def transform_camera_matrix(transform: np.ndarray, fix_upside_down: bool = True) -> np.ndarray:
    """
    Convert Metashape camera transform to LichtFeld/nerfstudio convention.
    
    Args:
        transform: 4x4 camera-to-world matrix from Metashape
        fix_upside_down: If True, apply additional 180° rotation to fix upside-down scene
    
    Returns:
        Transformed 4x4 matrix
    """
    # Metashape: camera looks at -Z, +X right, +Y up
    # Step 1: Rotate scene according to nerfstudio convention (row swap)
    transform = transform[[2, 0, 1, 3], :]
    
    # Step 2: Convert from OpenCV to OpenGL (flip Y and Z columns)
    transform[:, 1:3] *= -1
    
    # Step 3: Fix orientation with +90° rotation around X-axis
    # This converts from bottom-up view to natural ground-level view
    if fix_upside_down:
        # Rotation matrix around X by +90°: [[1,0,0], [0,0,-1], [0,1,0]]
        cos_90 = 0.0   # cos(90°) = 0
        sin_90 = 1.0   # sin(90°) = 1
        rot_x_pos90 = np.array([
            [1, 0, 0, 0],
            [0, cos_90, -sin_90, 0],  # [0, 0, -1, 0]
            [0, sin_90, cos_90, 0],   # [0, 1, 0, 0]
            [0, 0, 0, 1]
        ], dtype=np.float64)
        transform = rot_x_pos90 @ transform
    
    # Step 4: Pre-compensate for LichtFeld's 180° Y-rotation
    # LichtFeld applies a Y-rotation to convert from OpenGL to COLMAP convention
    # We apply the inverse here so they cancel out
    cos_pi = -1.0  # cos(180°) = -1
    sin_pi = 0.0   # sin(180°) = 0
    y_rot_180 = np.array([
        [cos_pi, 0, sin_pi, 0],
        [0, 1, 0, 0],
        [-sin_pi, 0, cos_pi, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    transform = y_rot_180 @ transform
    
    return transform



def get_applied_transform(fix_upside_down: bool = True) -> np.ndarray:
    """
    Get the 3x4 transformation matrix applied to point cloud.
    
    Note: LichtFeld only applies Y-rotation to cameras, not point clouds.
    So we don't need Y-rotation pre-compensation here.
    """
    # Base transform: row swap [2, 0, 1]
    applied_transform = np.eye(4)[:3, :]
    applied_transform = applied_transform[np.array([2, 0, 1]), :]
    
    # Add orientation fix: +90° rotation around X
    if fix_upside_down:
        cos_90 = 0.0
        sin_90 = 1.0
        rot_x_pos90 = np.array([
            [1, 0, 0],
            [0, cos_90, -sin_90],  # [0, 0, -1]
            [0, sin_90, cos_90]    # [0, 1, 0]
        ], dtype=np.float64)
        applied_transform[:3, :3] = rot_x_pos90 @ applied_transform[:3, :3]
    
    # NOTE: No Y-rotation here - LichtFeld only applies Y-rot to cameras, not point clouds
    
    return applied_transform




def build_component_transform_4x4(component_dict: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """
    Get the combined component transform as a 4x4 matrix.

    For single-component projects (most common), returns that component's transform.
    For multi-component projects, this is more complex - we'd need per-point component IDs.

    Args:
        component_dict: Dictionary of component_id -> 4x4 transform matrix

    Returns:
        4x4 numpy array if single component with transform, None otherwise
    """
    if len(component_dict) == 0:
        return None
    elif len(component_dict) == 1:
        # Single component - use its transform
        return list(component_dict.values())[0]
    else:
        # Multiple components - would need per-point component assignment
        # For now, warn and return None (points stay in component-local coords)
        print("WARNING: Multiple components detected. Point cloud may not be correctly transformed.")
        print("         Consider merging components in Metashape before export.")
        return None


def convert_metashape_to_lichtfeld(
    images_dir: Path,
    xml_path: Path,
    output_dir: Optional[Path] = None,
    ply_path: Optional[Path] = None,
    fix_upside_down: bool = True,
    max_images: Optional[int] = None,
    copy_images: bool = True,
    verbose: bool = True,
    split_cubemap: bool = False,
    crop_size: int = 1920,
    fov_deg: float = 90.0,
    skip_directions: Optional[List[str]] = None,
    generate_masks: bool = False,
    yolo_model_path: str = "yolo11m-seg.pt",
    yolo_classes: Optional[List[int]] = None,
    yolo_conf: float = 0.25,
    invert_mask: bool = False,
    mask_overexposure: bool = False,
    overexposure_threshold: int = 250,
    overexposure_dilate: int = 5,
    mask_engine: str = "yolo",
    sam3_model_path: str = "sam3.pt",
    sam3_concepts: Optional[List[str]] = None,
    sam3_conf: float = 0.25,
    sam3_half: bool = True,
) -> Dict[str, Any]:
    """
    Convert Metashape data to LichtFeld-compatible transforms.json format.

    Args:
        images_dir: Directory containing images
        xml_path: Path to Metashape cameras.xml
        output_dir: Output directory (defaults to same directory as xml_path)
        ply_path: Optional path to point cloud PLY file
        fix_upside_down: If True, fix the upside-down scene orientation
        max_images: Maximum number of camera frames to process (None = all)
        copy_images: If True, copy source images to output_dir/images/ and use
                     relative paths in transforms.json (recommended for portability).
                     Ignored when split_cubemap=True (crops always written to output).
        verbose: Print progress messages
        split_cubemap: If True, split each equirectangular image into perspective
                       cubemap-face crops (same approach as COLMAP mode) and write
                       transforms.json with PINHOLE camera model.  When False (default)
                       the original equirectangular images are used as-is with
                       EQUIRECTANGULAR camera model.
        crop_size: Square pixel size for each cubemap face crop (only used when
                   split_cubemap=True).
        fov_deg: Horizontal field-of-view in degrees for each crop
                 (only used when split_cubemap=True, default 90°).
        skip_directions: List of face directions to omit when split_cubemap=True.
                         Valid values: top, front, right, back, left, bottom.
        generate_masks: If True, run YOLO segmentation on every output image and
                        save binary masks to output_dir/masks/ (requires ultralytics).
        yolo_model_path: Path to the YOLO segmentation model weights file.
        yolo_classes: YOLO class IDs to mask (default: [0] = person).
        yolo_conf: Minimum confidence threshold for YOLO detections (0–1).
        invert_mask: If True, flip mask polarity (masked region = white).
        mask_overexposure: If True, also mask blown-out (overexposed) pixels.
        overexposure_threshold: Per-channel brightness threshold for overexposure
                                detection (0–255, default 250).
        overexposure_dilate: Dilation radius in pixels applied to the overexposure
                             mask to cover bloom halation (default 5).

    Returns:
        Dictionary with conversion statistics
    """
    # Default output to same directory as XML
    if output_dir is None:
        output_dir = xml_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    # In split-cubemap mode the cropped images are always written to output/images/.
    images_output_dir = output_dir / "images"
    if split_cubemap or copy_images:
        images_output_dir.mkdir(parents=True, exist_ok=True)

    if split_cubemap and not HAS_IMAGE_LIBS:
        raise ImportError(
            "--split-cubemap requires Pillow and OpenCV. "
            "Install with: pip install pillow opencv-python"
        )

    # Masking setup — resolved before the frame loop to fail early on missing deps.
    do_masking = generate_masks or mask_overexposure
    masks_output_dir = output_dir / "masks"
    yolo_model_instance = None
    yolo_class_ids = yolo_classes if yolo_classes else [0]

    if do_masking:
        if not HAS_IMAGE_LIBS:
            raise ImportError(
                "Mask generation requires Pillow and OpenCV. "
                "Install with: pip install pillow opencv-python"
            )
        masks_output_dir.mkdir(parents=True, exist_ok=True)

    sam3_model_instance = None

    if generate_masks:
        if mask_engine == "sam3":
            if not HAS_SAM3:
                raise ImportError(
                    "--mask-engine sam3 requires ultralytics. "
                    "Install with: pip install ultralytics"
                )
            if verbose:
                print(f"Loading SAM3 model: {sam3_model_path}")
            sam3_model_instance = _lfs_load_open_vocab_model(
                sam3_model_path, conf=sam3_conf, half=sam3_half
            )
            if verbose:
                is_native = (
                    _SAM3SemanticPredictor is not None
                    and isinstance(sam3_model_instance, _SAM3SemanticPredictor)
                )
                print(f"  Backend: {'SAM3SemanticPredictor (Meta SAM3)' if is_native else 'YOLOE/YOLOWorld'}")
            # SAM3SemanticPredictor handles FP16/CUDA via its overrides at init.
            # For YOLOE-family models, apply FP16 manually after loading.
            if sam3_half and not (
                _SAM3SemanticPredictor is not None
                and isinstance(sam3_model_instance, _SAM3SemanticPredictor)
            ):
                try:
                    import torch
                    if torch.cuda.is_available():
                        sam3_model_instance.to("cuda").half()
                        if verbose:
                            print("  SAM3: FP16 enabled on CUDA")
                except Exception:
                    pass
        else:  # "yolo"
            if not HAS_YOLO:
                raise ImportError(
                    "--generate-masks requires ultralytics. "
                    "Install with: pip install ultralytics"
                )
            if verbose:
                print(f"Loading YOLO model: {yolo_model_path}")
            yolo_model_instance = _YOLO(yolo_model_path)

    if verbose:
        print(f"Parsing Metashape XML: {xml_path}")

    # Parse XML
    xml_data = parse_metashape_xml(xml_path)
    sensor_dict = xml_data["sensor_dict"]
    component_dict = xml_data["component_dict"]
    cameras_xml = xml_data["cameras"]
    camera_model = xml_data["camera_model"]
    
    if verbose:
        if split_cubemap:
            print("Mode: cubemap split (PINHOLE, 6 perspective crops per panorama)")
        else:
            print(f"Camera model: {camera_model}")
        print(f"Found {len(sensor_dict)} sensor(s)")
    
    # Build image filename map
    image_extensions = [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))
        image_files.extend(images_dir.glob(f"*{ext.upper()}"))
    
    image_filename_map = {}
    for img_path in image_files:
        # Map both with and without extension
        image_filename_map[img_path.stem] = img_path
        image_filename_map[img_path.name] = img_path
    
    if verbose:
        print(f"Found {len(image_files)} images in {images_dir}")
    
    # Resolve active face directions for cubemap split mode
    active_directions = _ALL_DIRECTIONS
    if skip_directions:
        active_directions = [d for d in _ALL_DIRECTIONS if d not in skip_directions]
    if verbose and split_cubemap and skip_directions:
        print(f"  Skipping directions: {skip_directions}")
        print(f"  Using directions: {active_directions}")

    # Process frames
    frames = []
    num_skipped = 0
    num_processed = 0
    
    for camera in cameras_xml.iter("camera"):
        if max_images is not None and num_processed >= max_images:
            break

        camera_label = camera.get("label")
        if not camera_label:
            continue
        
        # Find matching image
        if camera_label not in image_filename_map:
            # Try without extension
            camera_label_no_ext = camera_label.split(".")[0]
            if camera_label_no_ext not in image_filename_map:
                if verbose:
                    print(f"  Skipping {camera.get('label')}: no matching image")
                num_skipped += 1
                continue
            camera_label = camera_label_no_ext
        
        # Get sensor data
        sensor_id = camera.get("sensor_id")
        if sensor_id not in sensor_dict:
            if verbose:
                print(f"  Skipping {camera.get('label')}: no sensor calibration")
            num_skipped += 1
            continue
        
        # Get camera transform
        transform_elem = camera.find("transform")
        if transform_elem is None or transform_elem.text is None:
            if verbose:
                print(f"  Skipping {camera.get('label')}: no transform")
            num_skipped += 1
            continue
        
        transform = np.array([float(x) for x in transform_elem.text.split()]).reshape((4, 4))
        
        # Apply component transform if present
        component_id = camera.get("component_id")
        if component_id in component_dict:
            transform = component_dict[component_id] @ transform
        
        src_image = image_filename_map[camera_label]
        base_name = Path(camera_label).stem

        if split_cubemap:
            # --- Cubemap-split mode: generate one PINHOLE frame per face ---
            equirect_img = _PILImage.open(str(src_image))
            if equirect_img.mode != "RGB":
                equirect_img = equirect_img.convert("RGB")

            for direction in active_directions:
                # Compose face direction rotation into the camera pose BEFORE
                # applying the LFS coordinate-system conversion.
                R_face = _lfs_get_face_rotation(direction)
                face_transform = transform.copy()
                face_transform[:3, :3] = transform[:3, :3] @ R_face
                # Translation (camera position) is identical for all faces.

                lfs_transform = transform_camera_matrix(face_transform, fix_upside_down)

                # Crop and save the perspective face image
                output_image_name = f"{base_name}_{direction}.png"
                output_image_path = images_output_dir / output_image_name
                if not output_image_path.exists():
                    cropped = _lfs_crop_face(equirect_img, direction, crop_size, fov_deg)
                    cropped.save(str(output_image_path), compress_level=0)
                elif do_masking:
                    # Need the crop for mask generation even when image already exists.
                    cropped = _lfs_crop_face(equirect_img, direction, crop_size, fov_deg)

                frame = {
                    "file_path": f"images/{output_image_name}",
                    "transform_matrix": lfs_transform.tolist(),
                }

                if do_masking:
                    mask_name = f"{base_name}_{direction}.png"
                    mask_path_file = masks_output_dir / mask_name
                    if not mask_path_file.exists():
                        if mask_engine == "sam3":
                            mask_img = _lfs_generate_mask_sam3(
                                cropped,
                                sam3_model_instance,
                                concepts=sam3_concepts or [],
                                conf=sam3_conf,
                                invert_mask=invert_mask,
                                mask_overexposure=mask_overexposure,
                                overexposure_threshold=overexposure_threshold,
                                overexposure_dilate=overexposure_dilate,
                            )
                        else:
                            mask_img = _lfs_generate_mask(
                                cropped,
                                yolo_model_instance,
                                yolo_conf=yolo_conf,
                                invert_mask=invert_mask,
                                class_ids=yolo_class_ids,
                                mask_overexposure=mask_overexposure,
                                overexposure_threshold=overexposure_threshold,
                                overexposure_dilate=overexposure_dilate,
                            )
                        mask_img.save(str(mask_path_file))
                    frame["mask_path"] = f"masks/{mask_name}"

                frames.append(frame)
        else:
            # --- Original mode: equirectangular image passed directly ---
            transform = transform_camera_matrix(transform, fix_upside_down)

            if copy_images:
                dest = images_output_dir / src_image.name
                if not dest.exists():
                    shutil.copy2(src_image, dest)
                file_path = f"images/{src_image.name}"
            else:
                try:
                    rel_path = src_image.resolve().relative_to(output_dir.resolve())
                    file_path = rel_path.as_posix()
                except ValueError:
                    file_path = src_image.resolve().as_posix()

            frame = {
                "file_path": file_path,
                "transform_matrix": transform.tolist(),
            }
            frame.update(sensor_dict[sensor_id])

            if do_masking:
                mask_name = f"{base_name}.png"
                mask_path_file = masks_output_dir / mask_name
                if not mask_path_file.exists():
                    pil_img = _PILImage.open(str(src_image)).convert("RGB")
                    if mask_engine == "sam3":
                        mask_img = _lfs_generate_mask_sam3(
                            pil_img,
                            sam3_model_instance,
                            concepts=sam3_concepts or [],
                            conf=sam3_conf,
                            invert_mask=invert_mask,
                            mask_overexposure=mask_overexposure,
                            overexposure_threshold=overexposure_threshold,
                            overexposure_dilate=overexposure_dilate,
                        )
                    else:
                        mask_img = _lfs_generate_mask(
                            pil_img,
                            yolo_model_instance,
                            yolo_conf=yolo_conf,
                            invert_mask=invert_mask,
                            class_ids=yolo_class_ids,
                            mask_overexposure=mask_overexposure,
                            overexposure_threshold=overexposure_threshold,
                            overexposure_dilate=overexposure_dilate,
                        )
                    mask_img.save(str(mask_path_file))
                frame["mask_path"] = f"masks/{mask_name}"

            frames.append(frame)

        num_processed += 1
    
    if verbose:
        print(f"Processed {num_processed} source images → {len(frames)} frames")
        if split_cubemap:
            print(f"  Cubemap crops written to: {images_output_dir}")
        elif copy_images:
            print(f"  Images copied to: {images_output_dir}")
        if do_masking:
            print(f"  Masks written to: {masks_output_dir}")
        if max_images is not None and num_processed >= max_images:
            print(f"  (Stopped after {max_images} images due to --max-images)")
        if num_skipped > 0:
            print(f"Skipped {num_skipped} cameras")
    
    # Build output data
    if split_cubemap:
        # PINHOLE intrinsics shared by all face crops — stored at the top level
        fx = fy = (crop_size / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
        data: Dict[str, Any] = {
            "camera_model": "PINHOLE",
            "fl_x": fx,
            "fl_y": fy,
            "cx": crop_size / 2.0,
            "cy": crop_size / 2.0,
            "w": crop_size,
            "h": crop_size,
            "frames": frames,
        }
    else:
        data = {
            "camera_model": camera_model,
            "frames": frames,
        }
    
    # Store applied transform for reference
    applied_transform = get_applied_transform(fix_upside_down)
    data["applied_transform"] = applied_transform.tolist()
    
    # Process point cloud
    if ply_path is not None and ply_path.exists():
        if verbose:
            print(f"Processing point cloud: {ply_path}")

        # NOTE: Metashape's PLY export already applies component transforms to point coordinates,
        # but the XML camera transforms are still in component-local coordinates.
        # So we only apply the LichtFeld coordinate transform to the PLY, NOT the component transform.
        # (Component transform is only needed for cameras from XML)

        if HAS_OPEN3D:
            pc = o3d.io.read_point_cloud(str(ply_path))
            points3D = np.asarray(pc.points)

            # Apply LichtFeld coordinate transform only (row swap + orientation fix)
            # Component transform is NOT applied - PLY export already includes it
            points3D = np.einsum("ij,bj->bi", applied_transform[:3, :3], points3D) + applied_transform[:3, 3]
            pc.points = o3d.utility.Vector3dVector(points3D)

            output_ply = output_dir / "pointcloud.ply"
            o3d.io.write_point_cloud(str(output_ply), pc)
            data["ply_file_path"] = "pointcloud.ply"
            pointcloud_written = True

            if verbose:
                print(f"Wrote point cloud with {len(points3D)} points to {output_ply}")

        elif HAS_PLYFILE:
            # Fallback to plyfile
            plydata = PlyData.read(str(ply_path))
            vertex = plydata['vertex']
            points3D = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T

            # Apply LichtFeld coordinate transform only
            points3D = np.einsum("ij,bj->bi", applied_transform[:3, :3], points3D) + applied_transform[:3, 3]

            # Copy the full structured array to preserve colors, normals, and any other properties
            vertex_data = plydata['vertex'].data.copy()
            vertex_data['x'] = points3D[:, 0].astype(vertex_data['x'].dtype)
            vertex_data['y'] = points3D[:, 1].astype(vertex_data['y'].dtype)
            vertex_data['z'] = points3D[:, 2].astype(vertex_data['z'].dtype)

            output_ply = output_dir / "pointcloud.ply"
            PlyData([PlyElement.describe(vertex_data, 'vertex')]).write(str(output_ply))
            data["ply_file_path"] = "pointcloud.ply"
            pointcloud_written = True

            if verbose:
                print(f"Wrote point cloud with {len(points3D)} points (colors may be lost)")
        else:
            print("WARNING: Neither open3d nor plyfile installed. Skipping point cloud.")
            print("         Install with: pip install open3d")
            pointcloud_written = False
    else:
        pointcloud_written = False
    
    # Write transforms.json
    output_json = output_dir / "transforms.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    
    if verbose:
        print(f"\nWrote {output_json}")
        print(f"Dataset ready: {len(frames)} frames, camera_model={camera_model}")
    
    return {
        "num_frames": len(frames),
        "num_source_images": num_processed,
        "num_skipped": num_skipped,
        "camera_model": "PINHOLE" if split_cubemap else camera_model,
        "split_cubemap": split_cubemap,
        "has_pointcloud": pointcloud_written,
        "images_copied": split_cubemap or copy_images,
        "has_masks": do_masking,
    }


def run_mask_only(
    images_dir: Path,
    output_dir: Path,
    generate_masks: bool = True,
    mask_overexposure: bool = False,
    yolo_model_path: str = "yolo11m-seg.pt",
    yolo_classes: Optional[List[int]] = None,
    yolo_conf: float = 0.25,
    invert_mask: bool = False,
    overexposure_threshold: int = 250,
    overexposure_dilate: int = 5,
    mask_engine: str = "yolo",
    sam3_model_path: str = "sam3.pt",
    sam3_concepts: Optional[List[str]] = None,
    sam3_conf: float = 0.25,
    sam3_half: bool = True,
    max_images: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run only the mask generation pipeline on an image folder.

    No Metashape XML or PLY required. Scans all images in ``images_dir``,
    generates binary masks using YOLO / SAM3 / overexposure detection and
    writes them to ``output_dir/masks/``.

    Returns a dict with processing statistics.
    """
    do_masking = generate_masks or mask_overexposure
    if not do_masking:
        raise ValueError("At least one of generate_masks or mask_overexposure must be enabled.")

    if not HAS_IMAGE_LIBS:
        raise ImportError(
            "Mask generation requires Pillow and OpenCV. "
            "Install with: pip install pillow opencv-python"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    masks_output_dir = output_dir / "masks"
    masks_output_dir.mkdir(parents=True, exist_ok=True)

    # Load model(s) once before the image loop.
    yolo_model_instance = None
    sam3_model_instance = None
    yolo_class_ids = yolo_classes if yolo_classes else [0]

    if generate_masks:
        if mask_engine == "sam3":
            if not HAS_SAM3:
                raise ImportError(
                    "--mask-engine sam3 requires ultralytics. "
                    "Install with: pip install ultralytics"
                )
            if verbose:
                print(f"Loading SAM3 model: {sam3_model_path}")
            sam3_model_instance = _lfs_load_open_vocab_model(
                sam3_model_path, conf=sam3_conf, half=sam3_half
            )
            if verbose:
                is_native = (
                    _SAM3SemanticPredictor is not None
                    and isinstance(sam3_model_instance, _SAM3SemanticPredictor)
                )
                print(f"  Backend: {'SAM3SemanticPredictor (Meta SAM3)' if is_native else 'YOLOE/YOLOWorld'}")
            # SAM3SemanticPredictor handles FP16/CUDA via its overrides at init.
            # For YOLOE-family models, apply FP16 manually after loading.
            if sam3_half and not (
                _SAM3SemanticPredictor is not None
                and isinstance(sam3_model_instance, _SAM3SemanticPredictor)
            ):
                try:
                    import torch
                    if torch.cuda.is_available():
                        sam3_model_instance.to("cuda").half()
                        if verbose:
                            print("  SAM3: FP16 enabled on CUDA")
                except Exception:
                    pass
        else:
            if not HAS_YOLO:
                raise ImportError(
                    "--generate-masks requires ultralytics. "
                    "Install with: pip install ultralytics"
                )
            if verbose:
                print(f"Loading YOLO model: {yolo_model_path}")
            yolo_model_instance = _YOLO(yolo_model_path)

    # Collect images.
    image_extensions = [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp"]
    image_files: List[Path] = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))
        image_files.extend(images_dir.glob(f"*{ext.upper()}"))
    image_files = sorted(set(image_files))

    if max_images is not None:
        image_files = image_files[:max_images]

    if verbose:
        print(f"Found {len(image_files)} images in {images_dir}")
        print(f"Masks will be written to: {masks_output_dir}")

    num_processed = 0
    num_skipped = 0

    for src_image in image_files:
        mask_name = src_image.stem + ".png"
        mask_path_file = masks_output_dir / mask_name

        if mask_path_file.exists():
            if verbose:
                print(f"  Skipping (mask exists): {mask_name}")
            num_skipped += 1
            continue

        try:
            pil_img = _PILImage.open(str(src_image)).convert("RGB")
        except Exception as exc:
            if verbose:
                print(f"  Warning: cannot open {src_image.name}: {exc}")
            num_skipped += 1
            continue

        if mask_engine == "sam3":
            mask_img = _lfs_generate_mask_sam3(
                pil_img,
                sam3_model_instance,
                concepts=sam3_concepts or [],
                conf=sam3_conf,
                invert_mask=invert_mask,
                mask_overexposure=mask_overexposure,
                overexposure_threshold=overexposure_threshold,
                overexposure_dilate=overexposure_dilate,
            )
        else:
            mask_img = _lfs_generate_mask(
                pil_img,
                yolo_model_instance,
                yolo_conf=yolo_conf,
                invert_mask=invert_mask,
                class_ids=yolo_class_ids,
                mask_overexposure=mask_overexposure,
                overexposure_threshold=overexposure_threshold,
                overexposure_dilate=overexposure_dilate,
            )

        mask_img.save(str(mask_path_file))
        num_processed += 1

        if verbose and num_processed % 50 == 0:
            print(f"  {num_processed}/{len(image_files)} masks generated…")

    if verbose:
        print(f"\nMask-only complete: {num_processed} masks generated, {num_skipped} skipped.")
        print(f"Output: {masks_output_dir}")

    return {
        "num_processed": num_processed,
        "num_skipped": num_skipped,
        "masks_dir": str(masks_output_dir),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert Metashape XML + PLY to LichtFeld transforms.json format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Equirectangular mode (default)
    python metashape_360_lfs.py --images ./images/ --xml cameras.xml
    python metashape_360_lfs.py --images ./images/ --xml cameras.xml --ply sparse.ply

    # Cubemap-split mode (PINHOLE, same approach as COLMAP)
    python metashape_360_lfs.py --images ./images/ --xml cameras.xml --split-cubemap
    python metashape_360_lfs.py --images ./images/ --xml cameras.xml --split-cubemap --crop-size 1920 --fov-deg 90
    python metashape_360_lfs.py --images ./images/ --xml cameras.xml --split-cubemap --skip-directions bottom,top
        """
    )
    
    parser.add_argument("--mask-only", action="store_true",
                        help="Run mask generation only (no XML/PLY required). "
                             "Scans --images folder and writes masks to --output/masks/.")
    parser.add_argument("--images", type=Path, required=True,
                        help="Directory containing source images")
    parser.add_argument("--xml", type=Path, default=None,
                        help="Path to Metashape cameras.xml file (not required with --mask-only)")
    parser.add_argument("--ply", type=Path, default=None,
                        help="Optional path to point cloud PLY file")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (defaults to same folder as XML)")
    parser.add_argument("--no-fix-rotation", action="store_true",
                        help="Disable 180° rotation fix (scene may appear upside-down)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Maximum number of camera frames to process (for quick tests)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    parser.add_argument("--no-copy-images", action="store_true",
                        help="Do not copy source images to output/images/; use original paths in transforms.json")
    # --- Cubemap-split options ---
    parser.add_argument("--split-cubemap", action="store_true",
                        help="Split each equirectangular image into 6 perspective cubemap-face crops "
                             "and write transforms.json with PINHOLE camera model (same as COLMAP mode)")
    parser.add_argument("--crop-size", type=int, default=1920,
                        help="Square pixel size for each cubemap face crop (only used with --split-cubemap, default: 1920)")
    parser.add_argument("--fov-deg", type=float, default=90.0,
                        help="Horizontal field-of-view in degrees for each crop (only used with --split-cubemap, default: 90.0)")
    parser.add_argument("--skip-directions", type=str, default="",
                        help="Comma-separated list of face directions to skip with --split-cubemap. "
                             "Valid: top,front,right,back,left,bottom  (e.g. 'bottom,top')")
    # --- Mask-generation options ---
    parser.add_argument("--generate-masks", action="store_true",
                        help="Run YOLO segmentation on every output image and save binary masks "
                             "to output/masks/. Requires: pip install ultralytics")
    parser.add_argument("--yolo-model", type=str, default="yolo11m-seg.pt",
                        help="Path to YOLO segmentation model weights (default: yolo11m-seg.pt)")
    parser.add_argument("--yolo-classes", type=str, default="0",
                        help="Comma-separated YOLO class IDs to mask (default: '0' = person)")
    parser.add_argument("--yolo-conf", type=float, default=0.25,
                        help="Minimum YOLO detection confidence threshold 0–1 (default: 0.25)")
    parser.add_argument("--invert-mask", action="store_true",
                        help="Invert mask polarity: masked region = white instead of black")
    parser.add_argument("--mask-overexposure", action="store_true",
                        help="Also mask blown-out (overexposed) pixels in every image")
    parser.add_argument("--overexposure-threshold", type=int, default=250,
                        help="Per-channel brightness to classify a pixel as overexposed (default: 250)")
    parser.add_argument("--overexposure-dilate", type=int, default=5,
                        help="Dilation radius in pixels for the overexposure mask (default: 5)")
    # --- Mask engine selection ---
    parser.add_argument("--mask-engine", type=str, default="yolo", choices=["yolo", "sam3"],
                        help="Mask generation engine: 'yolo' (class IDs, default) or "
                             "'sam3' (open-vocabulary text concepts, requires 8 GB+ VRAM)")
    parser.add_argument("--sam3-model", type=str, default="sam3.pt",
                        help="Path to the SAM3/open-vocabulary model weights file (default: sam3.pt)")
    parser.add_argument("--sam3-concepts", type=str, default="",
                        help="Comma-separated text concepts for SAM3 to detect and mask "
                             "(e.g. 'person,moving car,tourist'). Used only with --mask-engine sam3")
    parser.add_argument("--sam3-conf", type=float, default=0.25,
                        help="SAM3 detection confidence threshold 0–1 (default: 0.25)")
    parser.add_argument("--no-sam3-half", action="store_true",
                        help="Disable FP16 half-precision for SAM3 (increases VRAM usage)")

    args = parser.parse_args()

    # Validate skip-directions
    valid_directions = set(_ALL_DIRECTIONS)
    skip_directions_list: Optional[List[str]] = None
    if args.skip_directions:
        skip_directions_list = [d.strip().lower() for d in args.skip_directions.split(",") if d.strip()]
        invalid = set(skip_directions_list) - valid_directions
        if invalid:
            print(f"Error: Invalid directions: {invalid}. Valid: {valid_directions}")
            return 1

    # Parse YOLO class IDs (comma-separated ints)
    yolo_classes_list: Optional[List[int]] = None
    if args.yolo_classes:
        try:
            yolo_classes_list = [int(c.strip()) for c in args.yolo_classes.split(",") if c.strip()]
        except ValueError:
            print(f"Error: --yolo-classes must be comma-separated integers, got: {args.yolo_classes}")
            return 1

    # Parse SAM3 text concepts (comma-separated strings)
    sam3_concepts_list: Optional[List[str]] = None
    if args.sam3_concepts:
        sam3_concepts_list = [c.strip() for c in args.sam3_concepts.split(",") if c.strip()]

    if not args.images.is_dir():
        print(f"Error: Images directory not found: {args.images}")
        return 1

    # --- Mask-only mode: no XML/PLY needed ---
    if args.mask_only:
        output_dir = args.output if args.output else args.images.parent / (args.images.name + "_masks")
        try:
            result = run_mask_only(
                images_dir=args.images,
                output_dir=output_dir,
                generate_masks=args.generate_masks,
                mask_overexposure=args.mask_overexposure,
                yolo_model_path=args.yolo_model,
                yolo_classes=yolo_classes_list,
                yolo_conf=args.yolo_conf,
                invert_mask=args.invert_mask,
                overexposure_threshold=args.overexposure_threshold,
                overexposure_dilate=args.overexposure_dilate,
                mask_engine=args.mask_engine,
                sam3_model_path=args.sam3_model,
                sam3_concepts=sam3_concepts_list,
                sam3_conf=args.sam3_conf,
                sam3_half=not args.no_sam3_half,
                max_images=args.max_images,
                verbose=not args.quiet,
            )
            if not args.quiet:
                print(f"  Masks generated: {result['num_processed']}")
                print(f"  Skipped: {result['num_skipped']}")
                print(f"  Output: {result['masks_dir']}")
            return 0
        except Exception as e:
            print(f"Error: {e}")
            return 1

    if args.xml is None:
        print("Error: --xml is required unless --mask-only is specified.")
        return 1

    if not args.xml.is_file():
        print(f"Error: XML file not found: {args.xml}")
        return 1
    
    if args.ply and not args.ply.is_file():
        print(f"Error: PLY file not found: {args.ply}")
        return 1
    
    try:
        result = convert_metashape_to_lichtfeld(
            images_dir=args.images,
            xml_path=args.xml,
            output_dir=args.output,
            ply_path=args.ply,
            fix_upside_down=not args.no_fix_rotation,
            max_images=args.max_images,
            copy_images=not args.no_copy_images,
            verbose=not args.quiet,
            split_cubemap=args.split_cubemap,
            crop_size=args.crop_size,
            fov_deg=args.fov_deg,
            skip_directions=skip_directions_list,
            generate_masks=args.generate_masks,
            yolo_model_path=args.yolo_model,
            yolo_classes=yolo_classes_list,
            yolo_conf=args.yolo_conf,
            invert_mask=args.invert_mask,
            mask_overexposure=args.mask_overexposure,
            overexposure_threshold=args.overexposure_threshold,
            overexposure_dilate=args.overexposure_dilate,
            mask_engine=args.mask_engine,
            sam3_model_path=args.sam3_model,
            sam3_concepts=sam3_concepts_list,
            sam3_conf=args.sam3_conf,
            sam3_half=not args.no_sam3_half,
        )
        
        if not args.quiet:
            print("\nConversion complete!")
            print(f"  Frames: {result['num_frames']}")
            print(f"  Source images: {result['num_source_images']}")
            print(f"  Skipped: {result['num_skipped']}")
            print(f"  Camera model: {result['camera_model']}")
            if result['split_cubemap']:
                print(f"  Mode: cubemap split (PINHOLE crops)")
            print(f"  Point cloud: {'Yes' if result['has_pointcloud'] else 'No'}")
            print(f"  Images in output: {'Yes' if result['images_copied'] else 'No (original paths used)'}")
            print(f"  Masks: {'Yes' if result['has_masks'] else 'No'}")
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
