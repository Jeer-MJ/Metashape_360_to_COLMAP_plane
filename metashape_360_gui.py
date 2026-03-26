#!/usr/bin/env python3
"""
Metashape 360° to COLMAP Converter - GUI Application

A graphical user interface for the metashape_360_to_colmap.py script.
Provides easy configuration and execution of the conversion process.
"""

import os
import sys
import subprocess
import threading
import queue
import multiprocessing
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path

# Default values matching config.txt.example
DEFAULTS = {
    "images": "./equirect/",
    "xml": "./cameras.xml",
    "ply": "./sparse_point_cloud.ply",
    "output": "./colmap_dataset/",
    "output_mode": "COLMAP",
    "crop_size": 1920,
    "fov_deg": 90.0,
    "max_images": 10000,
    "range_start": 0,
    "range_end": 10000,
    "num_workers": min(multiprocessing.cpu_count(), 8),
    "generate_masks": True,
    "mask_engine": "yolo",
    "invert_mask": False,
    "yaw_offset": 0.0,
    "quiet": False,
    "yolo_classes": "0",
    "yolo_conf": 0.25,
    "mask_overexposure": False,
    "overexposure_threshold": 250,
    "overexposure_dilate": 5,
    "yolo_model": "yolo11m-seg.pt",
    "sam3_model": "sam3.pt",
    "sam3_concepts": "person,people,tourist,selfie stick,moving car,motorbike,bicycle",
    "sam3_conf": 0.25,
    "sam3_half": True,
    "apply_component_transform": False,
    "flip_vertical": True,
    "rotate_z180": True,
    "fix_upside_down": True,
    "lfs_copy_images": True,
    "lfs_split_cubemap": False,
    "enable_cubemap": True,
    "language": "EN",
    "sharp_frame_enabled": False,
    "sharp_frame_method": "best-n",
    "sharp_frame_num_frames": 300,
    "sharp_frame_min_buffer": 3,
    "sharp_frame_batch_size": 5,
    "sharp_frame_batch_buffer": 2,
    "sharp_frame_window_size": 15,
    "sharp_frame_sensitivity": 50,
    "sharp_frame_output": "",
    "video_enabled": False,
    "video_path": "",
    "video_mode": "every-n",
    "video_every_n": 10,
    "video_fps": 2.0,
    "video_start_time": 0.0,
    "video_end_time": 0.0,
    "video_output": "",
}

VALID_DIRECTIONS = ["top", "front", "right", "back", "left", "bottom"]
YOLO_SEG_MODELS = [
    "yolo11n-seg.pt",
    "yolo11s-seg.pt",
    "yolo11m-seg.pt",
    "yolo11l-seg.pt",
    "yolo11x-seg.pt",
]

UI_TEXT = {
    "EN": {
        "app_title": "Metashape 360° to COLMAP Converter",
        "language": "Language:",
        "paths_section": "Input/Output Paths",
        "images_label": "Input Images Folder:",
        "xml_label": "Metashape XML:",
        "ply_label": "PLY File (Optional):",
        "output_label": "Output Folder:",
        "browse": "Browse...",
        "proc_section": "Processing Options",
        "crop_size": "Crop Size:",
        "fov": "FoV:",
        "max_images": "Max Images:",
        "workers": "Workers:",
        "range": "Image Range:",
        "range_start": "Start:",
        "range_end": "End:",
        "yaw_offset": "Yaw Offset:",
        "skip_section": "Skip Directions",
        "skip_note": "(Skip cubemap generation for selected directions)",
        "mask_section": "Mask Generation",
        "enable_yolo": "Enable mask generation",
        "mask_engine_label": "Engine:",
        "mask_engine_yolo": "YOLO (fast, class IDs)",
        "mask_engine_sam3": "SAM3 (text prompts, 8GB+ VRAM)",
        "yolo_classes": "YOLO Class IDs:",
        "yolo_conf": "YOLO Confidence:",
        "yolo_model": "YOLO Model:",
        "invert_mask": "Invert mask (object=white)",
        "sam3_model": "SAM3 Model:",
        "sam3_concepts": "Concepts (comma-separated):",
        "sam3_conf": "SAM3 Confidence:",
        "sam3_half": "FP16 (reduces VRAM to ~1.7 GB)",
        "sam3_vram_warn": "SAM3 requires 8 GB+ VRAM and runs sequentially (1 worker).",
        "enable_overexp": "Enable overexposure mask",
        "threshold": "Threshold (0-255):",
        "dilate": "Dilation Radius:",
        "advanced_section": "Dev Options(Don't change)",
        "advanced_show": "Show Dev Options",
        "advanced_hide": "Hide Dev Options",
        "flip_vertical": "Flip vertical (for equirect sampling)",
        "rotate_z180": "Rotate 180° around Z-axis (PostShot compatibility)",
        "apply_component": "Apply component transform for PLY",
        "quiet": "Quiet mode (suppress progress output)",
        "save_cfg": "Save Config",
        "load_cfg": "Load Config",
        "reset_defaults": "Reset Defaults",
        "run": "▶ Run Conversion",
        "stop": "■ Stop",
        "console_section": "Output Log",
        "clear_log": "Clear Log",
        "err_title": "Input Error",
        "err_images_required": "Please specify input images folder",
        "err_images_missing": "Input images folder not found: {path}",
        "err_xml_required": "Please specify Metashape XML file",
        "err_xml_missing": "XML file not found: {path}",
        "err_output_required": "Please specify output folder",
        "err_ply_missing": "PLY file not found: {path}",
        "cmd": "Command: {cmd}",
        "done_ok": "✓ Conversion completed successfully.",
        "done_ng": "✗ Conversion exited with error code {code}.",
        "error": "Error: {error}",
        "stopped": "Processing stopped.",
        "exit_confirm_title": "Confirm Exit",
        "exit_confirm_running": "A conversion is still running. Exit and force-stop the CLI process?",
        "reset_done": "Settings reset to defaults.",
        "saved": "Config saved: {path}",
        "auto_loaded": "Auto-loaded config.txt.",
        "load_error": "Config load error: {error}",
        "loaded": "Config loaded: {path}",
        "tip_images": "Folder containing equirectangular images",
        "tip_xml": "cameras.xml exported from Metashape",
        "tip_ply": "Point cloud file exported from Metashape (optional)",
        "tip_output": "Destination folder for COLMAP dataset output",
        "tip_crop_size": "Output image size (pixels)",
        "tip_fov": "Horizontal field of view for rectilinear crops (degrees)",
        "tip_max_images": "Maximum number of images to process (for testing)",
        "tip_workers": "Number of worker processes for parallel processing",
        "tip_range": "Index range of images to process (0-based)",
        "tip_yaw_offset": "Per-frame yaw rotation offset (degrees)",
        "tip_yolo_classes": "Comma-separated class IDs (0=person, 2=car, 3=motorcycle, 5=bus, 7=truck)",
        "tip_yolo_conf": "Minimum YOLO confidence score to keep detections (0.0-1.0)",
        "tip_overexp_threshold": "Treat pixels as overexposed when all RGB channels exceed this value",
        "tip_overexp_dilate": "Dilation amount to cover fringe artifacts around masks (pixels)",
        "tip_flip_vertical": "Flip Y-axis during equirectangular image sampling",
        "output_mode": "Output Mode:",
        "mode_colmap": "COLMAP / PostShot (cubemap crops)",
        "mode_lfs": "Licht-Feld Studio (transforms.json)",
        "mode_sharp_frame": "Sharp Frame Only (no XML/3D)",
        "mode_mask_only": "Mask Only (no XML/3D)",
        "tip_mode_sharp_frame": "Scan an image folder and select only the sharpest frames. No Metashape XML or PLY file required. Requires: pip install sharp-frames",
        "tip_mode_mask_only": "Run YOLO / SAM3 / overexposure masking on an image folder without any Metashape XML or PLY. Masks are saved to output/masks/.",
        "fix_upside_down": "Fix upside-down orientation (+90° X rotation)",
        "tip_fix_upside_down": "Apply +90° rotation around X-axis to correct upside-down scenes in LFS",
        "lfs_copy_images": "Copy images to output folder",
        "tip_lfs_copy_images": "Copy source images into output/images/ and use relative paths in transforms.json (recommended for portability)",
        "tip_mode": "COLMAP mode: generates 6 cubemap crops + cameras.txt/images.txt.  LFS mode: generates transforms.json for Licht-Feld Studio directly.",
        "lfs_split_cubemap": "Split into cubemap faces (PINHOLE crops)",
        "tip_lfs_split_cubemap": "Split each equirectangular image into 6 perspective crops and use PINHOLE camera model in transforms.json — same approach as COLMAP mode",
        "lfs_split_crop_fov": "Crop Size / FoV:",
        "cubemap_section": "Cubemap Options",
        "enable_cubemap": "Enable Cubemap Split",
        "tip_enable_cubemap": "Split equirectangular images into 6 perspective (PINHOLE) crops. Always active in COLMAP mode; optional in LFS mode.",
        "lfs_mask_note": "Mask generation is not supported in LFS mode.",
        "skip_inline": "Skip directions:",
        "sf_section": "Sharp Frame Pre-filter",
        "sf_enable": "Enable sharp frame extraction before conversion",
        "sf_method_label": "Method:",
        "sf_method_best_n": "best-n (target count)",
        "sf_method_batched": "batched (window blocks)",
        "sf_method_outlier": "outlier-removal (remove blurry)",
        "sf_num_frames": "Target frames:",
        "sf_min_buffer": "Min buffer:",
        "sf_batch_size": "Batch size:",
        "sf_batch_buffer": "Batch buffer:",
        "sf_window_size": "Window size:",
        "sf_sensitivity": "Sensitivity (0-100):",
        "sf_output": "Output folder (empty = auto):",
        "tip_sf_enable": "Run sharp-frames (Reflct) to select only the sharpest images from the input folder before conversion. Requires: pip install sharp-frames",
        "tip_sf_num_frames": "Target number of sharpest frames to select (best-n method)",
        "tip_sf_sensitivity": "0 = keep more frames, 100 = remove aggressively (outlier-removal method)",
        "tip_sf_output": "Folder where selected sharp frames are written. Leave empty to auto-derive as <images_folder>_sharp/",
        "sf_note": "Requires: pip install sharp-frames",
        "vid_section": "Video Frame Extraction",
        "vid_enable": "Extract frames from video before conversion",
        "vid_path": "Video File:",
        "vid_mode_label": "Mode:",
        "vid_mode_every_n": "every-N (frame interval)",
        "vid_mode_fps": "at FPS (target rate)",
        "vid_every_n": "Every N frames:",
        "vid_fps": "Target FPS:",
        "vid_start_time": "Start (s):",
        "vid_end_time": "End (s):",
        "vid_time_note": "(0 = beginning / end of video)",
        "vid_output": "Output folder (empty = auto):",
        "tip_vid_enable": "Extract image frames from a video file before conversion. Requires: opencv-python (already installed)",
        "tip_vid_path": "Path to the source video file (MP4, AVI, MOV, MKV, WebM)",
        "tip_vid_every_n": "Extract one frame every N frames (e.g. 10 = keep 1 in 10 frames)",
        "tip_vid_fps": "Target extraction rate in frames per second (e.g. 2.0 = 2 frames/s)",
        "tip_vid_output": "Folder where extracted frames are saved. Leave empty to auto-derive as <video_name>_frames/",
        "err_video_required": "Please specify a video file",
        "err_video_missing": "Video file not found: {path}",
        "sf_run_extract": "▶ Run Phase 1: Extract Frames",
        "sf_run_sharpen": "▶ Run Phase 2: Select Sharp Frames",
        "sf_only_images_note": "Source: Images Folder (or extracted from video above)",
    },
    "JP": {
        "app_title": "Metashape 360° to COLMAP コンバーター",
        "language": "言語:",
        "paths_section": "入力/出力パス",
        "images_label": "入力画像フォルダ:",
        "xml_label": "Metashape XML:",
        "ply_label": "PLYファイル (任意):",
        "output_label": "出力フォルダ:",
        "browse": "参照...",
        "proc_section": "処理オプション",
        "crop_size": "クロップサイズ:",
        "fov": "視野角 (FoV):",
        "max_images": "最大画像数:",
        "workers": "ワーカー数:",
        "range": "画像範囲指定:",
        "range_start": "開始:",
        "range_end": "終了:",
        "yaw_offset": "Yawオフセット:",
        "skip_section": "スキップ方向",
        "skip_note": "(選択した方向のCubemap生成をスキップします)",
        "mask_section": "マスク生成",
        "enable_yolo": "マスク生成を有効化",
        "mask_engine_label": "エンジン:",
        "mask_engine_yolo": "YOLO (高速, クラスID)",
        "mask_engine_sam3": "SAM3 (テキストプロンプト, 8GB+ VRAM)",
        "yolo_classes": "YOLOクラスID:",
        "yolo_conf": "YOLO信頼度閾値:",
        "yolo_model": "YOLOモデル:",
        "invert_mask": "マスク反転 (物体=白)",
        "sam3_model": "SAM3モデル:",
        "sam3_concepts": "概念 (カンマ区切り):",
        "sam3_conf": "SAM3信頼度閾値:",
        "sam3_half": "FP16 (VRAM約1.7GBに削減)",
        "sam3_vram_warn": "SAM3は8GB以上のVRAMが必要で順次処理 (ワーカー1)。",
        "enable_overexp": "露出オーバーマスクを有効化",
        "threshold": "閾値 (0-255):",
        "dilate": "膨張半径:",
        "advanced_section": "開発者向けオプション (変更不要)",
        "advanced_show": "開発者向けオプションを表示",
        "advanced_hide": "開発者向けオプションを隠す",
        "flip_vertical": "垂直フリップ (Equirect用)",
        "rotate_z180": "Z軸180°回転 (PostShot互換)",
        "apply_component": "PLYにコンポーネント変換を適用",
        "quiet": "静音モード (進捗非表示)",
        "save_cfg": "設定を保存",
        "load_cfg": "設定を読み込み",
        "reset_defaults": "デフォルトに戻す",
        "run": "▶ 変換を実行",
        "stop": "■ 停止",
        "console_section": "出力ログ",
        "clear_log": "ログをクリア",
        "err_title": "入力エラー",
        "err_images_required": "入力画像フォルダを指定してください",
        "err_images_missing": "入力画像フォルダが見つかりません: {path}",
        "err_xml_required": "Metashape XMLファイルを指定してください",
        "err_xml_missing": "XMLファイルが見つかりません: {path}",
        "err_output_required": "出力フォルダを指定してください",
        "err_ply_missing": "PLYファイルが見つかりません: {path}",
        "cmd": "実行コマンド: {cmd}",
        "done_ok": "✓ 変換が正常に完了しました。",
        "done_ng": "✗ 変換がエラーコード {code} で終了しました。",
        "error": "エラー: {error}",
        "stopped": "処理を停止しました。",
        "exit_confirm_title": "終了確認",
        "exit_confirm_running": "変換処理が実行中です。終了してCLIプロセスを強制停止しますか？",
        "reset_done": "設定をデフォルトに戻しました。",
        "saved": "設定を保存しました: {path}",
        "auto_loaded": "config.txt を自動読み込みしました。",
        "load_error": "設定ファイル読み込みエラー: {error}",
        "loaded": "設定を読み込みました: {path}",
        "tip_images": "Equirectangular画像が含まれるフォルダ",
        "tip_xml": "Metashapeからエクスポートしたcameras.xml",
        "tip_ply": "Metashapeからエクスポートした点群ファイル (任意)",
        "tip_output": "COLMAP形式のデータセット出力先",
        "tip_crop_size": "出力画像のサイズ (pixels)",
        "tip_fov": "Rectilinear cropsの水平視野角 (degrees)",
        "tip_max_images": "処理する最大画像数 (テスト用)",
        "tip_workers": "並列処理のワーカープロセス数",
        "tip_range": "処理する画像のインデックス範囲 (0ベース)",
        "tip_yaw_offset": "フレームごとのYaw回転オフセット (degrees)",
        "tip_yolo_classes": "カンマ区切りのクラスID (0=person, 2=car, 3=motorcycle, 5=bus, 7=truck)",
        "tip_yolo_conf": "検出を採用する最小YOLO信頼度スコア (0.0-1.0)",
        "tip_overexp_threshold": "全RGBチャンネルがこの値を超えるピクセルを露出オーバーとみなす",
        "tip_overexp_dilate": "マスク周辺のフリンジアーティファクトをカバーする膨張量 (pixels)",
        "tip_flip_vertical": "Equirectangular画像サンプリング時のY軸反転",
        "output_mode": "出力モード:",
        "mode_colmap": "COLMAP / PostShot (Cubemapクロップ)",
        "mode_lfs": "Licht-Feld Studio (transforms.json)",
        "mode_sharp_frame": "シャープフレームのみ (XML/3D不要)",
        "tip_mode_sharp_frame": "画像フォルダをスキャンして最もシャープなフレームのみを選択します。Metashape XMLもPLYも不要です。要件: pip install sharp-frames",
        "mode_mask_only": "マスクのみ (XML/3D不要)",
        "tip_mode_mask_only": "MetashapeのXMLやPLYなしで、画像フォルダにYOLO / SAM3 / 過露出マスクを適用します。マスクは output/masks/ に保存されます。",
        "fix_upside_down": "上下反転を修正 (+90° X軸回転)",
        "tip_fix_upside_down": "LFSで逆さまのシーンを補正するX軸+90°回転を適用",
        "lfs_copy_images": "画像を出力フォルダにコピー",
        "tip_lfs_copy_images": "ソース画像をoutput/images/にコピーしてtransforms.jsonで相対パスを使用 (可搬性のため推奨)",
        "tip_mode": "COLMAPモード: 6面Cubemapクロップ＋cameras.txt/images.txtを生成。LFSモード: Licht-Feld Studio用transforms.jsonを直接生成。",
        "lfs_split_cubemap": "Cubemap面分割 (PINHOLEクロップ)",
        "tip_lfs_split_cubemap": "各Equirectangular画像を6面の透視投影クロップに分割し、transforms.jsonでPINHOLEカメラモデルを使用 — COLMAPモードと同じアプローチ",
        "lfs_split_crop_fov": "クロップサイズ / FoV:",
        "cubemap_section": "Cubemapオプション",
        "enable_cubemap": "Cubemap分割を有効化",
        "tip_enable_cubemap": "Equirectangular画像を6面のPINHOLEクロップに分割します。COLMAPモードでは常に有効；LFSモードでは任意。",
        "lfs_mask_note": "マスク生成はLFSモードでは使用できません。",
        "skip_inline": "スキップ方向:",
        "sf_section": "シャープフレーム前処理",
        "sf_enable": "変換前にシャープフレーム抽出を有効化",
        "sf_method_label": "手法:",
        "sf_method_best_n": "best-n (目標フレーム数)",
        "sf_method_batched": "batched (ウィンドウブロック)",
        "sf_method_outlier": "outlier-removal (ブレフレーム除去)",
        "sf_num_frames": "目標フレーム数:",
        "sf_min_buffer": "最小バッファ:",
        "sf_batch_size": "バッチサイズ:",
        "sf_batch_buffer": "バッチバッファ:",
        "sf_window_size": "ウィンドウサイズ:",
        "sf_sensitivity": "感度 (0-100):",
        "sf_output": "出力フォルダ (空=自動):",
        "tip_sf_enable": "変換前にsharp-frames (Reflct)を使用して最もシャープな画像を選択します。要件: pip install sharp-frames",
        "tip_sf_num_frames": "選択するシャープフレームの目標数 (best-n手法)",
        "tip_sf_sensitivity": "0=多めに保持、100=積極的に除去 (outlier-removal手法)",
        "tip_sf_output": "選択されたシャープフレームの書き出し先。空の場合は<images_folder>_sharp/に自動設定。",
        "sf_note": "要件: pip install sharp-frames",
        "vid_section": "動画フレーム抽出",
        "vid_enable": "変換前に動画からフレームを抽出",
        "vid_path": "動画ファイル:",
        "vid_mode_label": "モード:",
        "vid_mode_every_n": "every-N (フレーム間隔)",
        "vid_mode_fps": "at FPS (目標レート)",
        "vid_every_n": "Nフレームごと:",
        "vid_fps": "目標FPS:",
        "vid_start_time": "開始 (秒):",
        "vid_end_time": "終了 (秒):",
        "vid_time_note": "(0 = 動画の先頭 / 末尾)",
        "vid_output": "出力フォルダ (空=自動):",
        "tip_vid_enable": "変換前に動画ファイルから画像フレームを抽出します。要件: opencv-python (インストール済み)",
        "tip_vid_path": "ソース動画ファイルのパス (MP4, AVI, MOV, MKV, WebM)",
        "tip_vid_every_n": "動画からNフレームごとに1枚抽出 (例: 10 = 10枚に1枚)",
        "tip_vid_fps": "1秒あたりの抽出フレーム数 (例: 2.0 = 毎秒2フレーム)",
        "tip_vid_output": "抽出フレームの保存先。空の場合は<動画名>_frames/に自動設定。",
        "err_video_required": "動画ファイルを指定してください",
        "err_video_missing": "動画ファイルが見つかりません: {path}",
        "sf_run_extract": "▶ フェーズ1実行: フレーム抽出",
        "sf_run_sharpen": "▶ フェーズ2実行: シャープフレーム選択",
        "sf_only_images_note": "入力: 画像フォルダ (または上の動画から抽出)",
    },
}


class ToolTip:
    """Simple tooltip implementation for Tkinter widgets."""
    
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)
    
    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        x, y, _, cy = self.widget.bbox("insert") if hasattr(self.widget, 'bbox') else (0, 0, 0, 0)
        x = x + self.widget.winfo_rootx() + 25
        y = y + cy + self.widget.winfo_rooty() + 25
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                        background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                        font=("Segoe UI", 9))
        label.pack(ipadx=4)
    
    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class Metashape360GUI:
    """Main GUI Application class."""
    
    def __init__(self, root):
        self.root = root
        self.root.geometry("850x900")
        self.root.minsize(750, 700)
        self.is_closing = False
        self.main_canvas = None
        self.main_scrollbar = None
        self.console = None
        self.options_notebook = None
        
        # Output queue for async process output
        self.output_queue = queue.Queue()
        self.process = None
        
        # Variables for all settings
        self.init_variables()
        
        # Setup UI
        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close_requested)
        
        # Load config if exists
        self.load_config_file()
        
        # Start output polling
        self.poll_output()
    
    def init_variables(self):
        """Initialize all tkinter variables."""
        # Path variables
        self.var_images = tk.StringVar(value=DEFAULTS["images"])
        self.var_xml = tk.StringVar(value=DEFAULTS["xml"])
        self.var_ply = tk.StringVar(value=DEFAULTS["ply"])
        self.var_output = tk.StringVar(value=DEFAULTS["output"])
        
        # Processing options
        self.var_crop_size = tk.IntVar(value=DEFAULTS["crop_size"])
        self.var_fov_deg = tk.DoubleVar(value=DEFAULTS["fov_deg"])
        self.var_max_images = tk.IntVar(value=DEFAULTS["max_images"])
        self.var_range_enabled = tk.BooleanVar(value=False)
        self.var_range_start = tk.IntVar(value=DEFAULTS["range_start"])
        self.var_range_end = tk.IntVar(value=DEFAULTS["range_end"])
        self.var_num_workers = tk.IntVar(value=DEFAULTS["num_workers"])
        self.var_yaw_offset = tk.DoubleVar(value=DEFAULTS["yaw_offset"])
        
        # Direction skip checkboxes
        self.var_skip_directions = {d: tk.BooleanVar(value=False) for d in VALID_DIRECTIONS}
        
        # Mask options
        self.var_generate_masks = tk.BooleanVar(value=DEFAULTS["generate_masks"])
        self.var_mask_engine = tk.StringVar(value=DEFAULTS["mask_engine"])
        self.var_invert_mask = tk.BooleanVar(value=DEFAULTS["invert_mask"])
        self.var_yolo_classes = tk.StringVar(value=DEFAULTS["yolo_classes"])
        self.var_yolo_conf = tk.DoubleVar(value=DEFAULTS["yolo_conf"])
        self.var_yolo_model = tk.StringVar(value=DEFAULTS["yolo_model"])
        self.var_sam3_model = tk.StringVar(value=DEFAULTS["sam3_model"])
        self.var_sam3_concepts = tk.StringVar(value=DEFAULTS["sam3_concepts"])
        self.var_sam3_conf = tk.DoubleVar(value=DEFAULTS["sam3_conf"])
        self.var_sam3_half = tk.BooleanVar(value=DEFAULTS["sam3_half"])
        
        # Overexposure mask
        self.var_mask_overexposure = tk.BooleanVar(value=DEFAULTS["mask_overexposure"])
        self.var_overexposure_threshold = tk.IntVar(value=DEFAULTS["overexposure_threshold"])
        self.var_overexposure_dilate = tk.IntVar(value=DEFAULTS["overexposure_dilate"])
        
        # Advanced options
        self.var_flip_vertical = tk.BooleanVar(value=DEFAULTS["flip_vertical"])
        self.var_rotate_z180 = tk.BooleanVar(value=DEFAULTS["rotate_z180"])
        self.var_fix_upside_down = tk.BooleanVar(value=DEFAULTS["fix_upside_down"])
        self.var_lfs_copy_images = tk.BooleanVar(value=DEFAULTS["lfs_copy_images"])
        self.var_lfs_split_cubemap = tk.BooleanVar(value=DEFAULTS["lfs_split_cubemap"])
        self.var_apply_component = tk.BooleanVar(value=DEFAULTS["apply_component_transform"])
        self.var_quiet = tk.BooleanVar(value=DEFAULTS["quiet"])
        self.var_language = tk.StringVar(value=DEFAULTS["language"])
        self.var_advanced_expanded = tk.BooleanVar(value=False)
        self.var_output_mode = tk.StringVar(value=DEFAULTS["output_mode"])
        self.var_enable_cubemap = tk.BooleanVar(value=DEFAULTS["enable_cubemap"])

        # Sharp frame pre-filter
        self.var_sharp_frame_enabled = tk.BooleanVar(value=DEFAULTS["sharp_frame_enabled"])
        self.var_sharp_frame_method = tk.StringVar(value=DEFAULTS["sharp_frame_method"])
        self.var_sharp_frame_num_frames = tk.IntVar(value=DEFAULTS["sharp_frame_num_frames"])
        self.var_sharp_frame_min_buffer = tk.IntVar(value=DEFAULTS["sharp_frame_min_buffer"])
        self.var_sharp_frame_batch_size = tk.IntVar(value=DEFAULTS["sharp_frame_batch_size"])
        self.var_sharp_frame_batch_buffer = tk.IntVar(value=DEFAULTS["sharp_frame_batch_buffer"])
        self.var_sharp_frame_window_size = tk.IntVar(value=DEFAULTS["sharp_frame_window_size"])
        self.var_sharp_frame_sensitivity = tk.IntVar(value=DEFAULTS["sharp_frame_sensitivity"])
        self.var_sharp_frame_output = tk.StringVar(value=DEFAULTS["sharp_frame_output"])

        # Video frame extraction
        self.var_video_enabled = tk.BooleanVar(value=DEFAULTS["video_enabled"])
        self.var_video_path = tk.StringVar(value=DEFAULTS["video_path"])
        self.var_video_mode = tk.StringVar(value=DEFAULTS["video_mode"])
        self.var_video_every_n = tk.IntVar(value=DEFAULTS["video_every_n"])
        self.var_video_fps = tk.DoubleVar(value=DEFAULTS["video_fps"])
        self.var_video_start_time = tk.DoubleVar(value=DEFAULTS["video_start_time"])
        self.var_video_end_time = tk.DoubleVar(value=DEFAULTS["video_end_time"])
        self.var_video_output = tk.StringVar(value=DEFAULTS["video_output"])

    def t(self, key, **kwargs):
        """Get localized UI text."""
        lang = self.var_language.get() if hasattr(self, "var_language") else "EN"
        table = UI_TEXT.get(lang, UI_TEXT["EN"])
        value = table.get(key, UI_TEXT["EN"].get(key, key))
        return value.format(**kwargs) if kwargs else value

    def on_language_changed(self, event=None):
        """Rebuild UI when language changes."""
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the main UI layout."""
        old_log = ""
        old_options_tab_idx = 0
        if self.console is not None:
            old_log = self.console.get("1.0", tk.END)
        if self.options_notebook is not None:
            try:
                old_options_tab_idx = self.options_notebook.index(self.options_notebook.select())
            except Exception:
                old_options_tab_idx = 0
        if self.main_canvas is not None:
            self.main_canvas.destroy()
        if self.main_scrollbar is not None:
            self.main_scrollbar.destroy()
        self.root.unbind_all("<MouseWheel>")
        self.root.title(self.t("app_title"))

        # Main container with scrollbar
        self.main_canvas = tk.Canvas(self.root)
        self.main_scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.main_canvas.yview)
        scrollable_frame = ttk.Frame(self.main_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )
        
        self.main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        self.main_canvas.configure(yscrollcommand=self.main_scrollbar.set)
        
        # Enable mouse wheel scrolling
        def on_mousewheel(event):
            self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.main_canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        self.main_canvas.pack(side="left", fill="both", expand=True)
        self.main_scrollbar.pack(side="right", fill="y")
        
        # Padding
        padx = 10
        pady = 5

        language_frame = ttk.Frame(scrollable_frame)
        language_frame.pack(fill="x", padx=padx, pady=(10, 5))
        ttk.Label(language_frame, text=self.t("language")).pack(side="left", padx=5)
        lang_combo = ttk.Combobox(
            language_frame,
            textvariable=self.var_language,
            values=["EN", "JP"],
            state="readonly",
            width=6,
        )
        lang_combo.pack(side="left")
        lang_combo.bind("<<ComboboxSelected>>", self.on_language_changed)
        
        # ===== Output Mode Selector =====
        self.mode_frame = mode_frame = ttk.LabelFrame(scrollable_frame, text=self.t("output_mode"), padding=8)
        mode_frame.pack(fill="x", padx=padx, pady=(0, pady))

        rb_colmap = ttk.Radiobutton(
            mode_frame,
            text=self.t("mode_colmap"),
            variable=self.var_output_mode,
            value="COLMAP",
            command=self.toggle_mode_ui,
        )
        rb_colmap.pack(side="left", padx=10)
        ToolTip(rb_colmap, self.t("tip_mode"))

        rb_lfs = ttk.Radiobutton(
            mode_frame,
            text=self.t("mode_lfs"),
            variable=self.var_output_mode,
            value="LFS",
            command=self.toggle_mode_ui,
        )
        rb_lfs.pack(side="left", padx=10)
        ToolTip(rb_lfs, self.t("tip_mode"))

        rb_sf = ttk.Radiobutton(
            mode_frame,
            text=self.t("mode_sharp_frame"),
            variable=self.var_output_mode,
            value="SHARP_FRAME",
            command=self.toggle_mode_ui,
        )
        rb_sf.pack(side="left", padx=10)
        ToolTip(rb_sf, self.t("tip_mode_sharp_frame"))

        rb_mask_only = ttk.Radiobutton(
            mode_frame,
            text=self.t("mode_mask_only"),
            variable=self.var_output_mode,
            value="MASK_ONLY",
            command=self.toggle_mode_ui,
        )
        rb_mask_only.pack(side="left", padx=10)
        ToolTip(rb_mask_only, self.t("tip_mode_mask_only"))

        # ===== Video Frame Extraction Section =====
        self.vid_lf = vid_lf = ttk.LabelFrame(scrollable_frame, text=self.t("vid_section"), padding=8)
        vid_lf.pack(fill="x", padx=padx, pady=(0, pady))

        vid_top_row = ttk.Frame(vid_lf)
        vid_top_row.pack(fill="x")

        vid_enable_cb = ttk.Checkbutton(
            vid_top_row,
            text=self.t("vid_enable"),
            variable=self.var_video_enabled,
            command=self.toggle_video_options,
        )
        vid_enable_cb.pack(side="left", padx=5, pady=(2, 4))
        ToolTip(vid_enable_cb, self.t("tip_vid_enable"))

        self.vid_inner = ttk.Frame(vid_lf)
        self.vid_inner.pack(fill="x", padx=5, pady=(0, 4))

        vid_path_row = ttk.Frame(self.vid_inner)
        vid_path_row.pack(fill="x", pady=(0, 4))
        ttk.Label(vid_path_row, text=self.t("vid_path")).pack(side="left", padx=5)
        self.vid_path_entry = ttk.Entry(
            vid_path_row, textvariable=self.var_video_path, width=42
        )
        self.vid_path_entry.pack(side="left", padx=5, fill="x", expand=True)
        ToolTip(self.vid_path_entry, self.t("tip_vid_path"))
        ttk.Button(
            vid_path_row, text=self.t("browse"), width=8,
            command=lambda: self.browse_file(
                self.var_video_path,
                [("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")],
            ),
        ).pack(side="left", padx=5)

        vid_mode_row = ttk.Frame(self.vid_inner)
        vid_mode_row.pack(fill="x", pady=(0, 4))
        ttk.Label(vid_mode_row, text=self.t("vid_mode_label")).pack(side="left", padx=(0, 8))
        for _vval, _vkey in [("every-n", "vid_mode_every_n"), ("fps", "vid_mode_fps")]:
            ttk.Radiobutton(
                vid_mode_row,
                text=self.t(_vkey),
                variable=self.var_video_mode,
                value=_vval,
                command=self.toggle_video_mode,
            ).pack(side="left", padx=6)

        self.vid_subopts_container = ttk.Frame(self.vid_inner)
        self.vid_subopts_container.pack(fill="x", pady=(0, 4))

        self.vid_every_n_frame = ttk.Frame(self.vid_subopts_container)
        ttk.Label(self.vid_every_n_frame, text=self.t("vid_every_n")).pack(side="left", padx=5)
        vid_n_spin = ttk.Spinbox(
            self.vid_every_n_frame, from_=1, to=1000, increment=1,
            textvariable=self.var_video_every_n, width=8,
        )
        vid_n_spin.pack(side="left", padx=5)
        ToolTip(vid_n_spin, self.t("tip_vid_every_n"))

        self.vid_fps_frame = ttk.Frame(self.vid_subopts_container)
        ttk.Label(self.vid_fps_frame, text=self.t("vid_fps")).pack(side="left", padx=5)
        vid_fps_spin = ttk.Spinbox(
            self.vid_fps_frame, from_=0.1, to=60.0, increment=0.5,
            textvariable=self.var_video_fps, width=8, format="%.1f",
        )
        vid_fps_spin.pack(side="left", padx=5)
        ToolTip(vid_fps_spin, self.t("tip_vid_fps"))

        vid_time_row = ttk.Frame(self.vid_inner)
        vid_time_row.pack(fill="x", pady=(0, 4))
        ttk.Label(vid_time_row, text=self.t("vid_start_time")).pack(side="left", padx=5)
        ttk.Spinbox(
            vid_time_row, from_=0.0, to=86400.0, increment=1.0,
            textvariable=self.var_video_start_time, width=8, format="%.1f",
        ).pack(side="left", padx=5)
        ttk.Label(vid_time_row, text=self.t("vid_end_time")).pack(side="left", padx=(10, 5))
        ttk.Spinbox(
            vid_time_row, from_=0.0, to=86400.0, increment=1.0,
            textvariable=self.var_video_end_time, width=8, format="%.1f",
        ).pack(side="left", padx=5)
        ttk.Label(vid_time_row, text=self.t("vid_time_note"), foreground="gray").pack(
            side="left", padx=10
        )

        vid_out_row = ttk.Frame(self.vid_inner)
        vid_out_row.pack(fill="x")
        ttk.Label(vid_out_row, text=self.t("vid_output")).pack(side="left", padx=5)
        self.vid_output_entry = ttk.Entry(
            vid_out_row, textvariable=self.var_video_output, width=42
        )
        self.vid_output_entry.pack(side="left", padx=5, fill="x", expand=True)
        ToolTip(self.vid_output_entry, self.t("tip_vid_output"))
        ttk.Button(
            vid_out_row, text=self.t("browse"), width=8,
            command=lambda: self.browse_folder(self.var_video_output),
        ).pack(side="left", padx=5)

        self.toggle_video_mode()
        self.toggle_video_options()

        # Per-phase run button (visible only in Sharp Frame Only mode)
        self.btn_sf_extract = ttk.Button(
            vid_lf, text=self.t("sf_run_extract"), command=self.run_extract_only
        )
        self.btn_sf_extract.pack(fill="x", padx=5, pady=(4, 2))
        self.btn_sf_extract.pack_forget()  # Hidden until SHARP_FRAME mode

        # ===== Sharp Frame Pre-filter Section =====
        self.sf_lf = sf_lf = ttk.LabelFrame(scrollable_frame, text=self.t("sf_section"), padding=8)
        sf_lf.pack(fill="x", padx=padx, pady=(0, pady))

        sf_top_row = ttk.Frame(sf_lf)
        sf_top_row.pack(fill="x")

        self.sf_enable_cb = ttk.Checkbutton(
            sf_top_row,
            text=self.t("sf_enable"),
            variable=self.var_sharp_frame_enabled,
            command=self.toggle_sharp_frame_options,
        )
        self.sf_enable_cb.pack(side="left", padx=5, pady=(2, 4))
        ToolTip(self.sf_enable_cb, self.t("tip_sf_enable"))

        ttk.Label(sf_top_row, text=self.t("sf_note"), foreground="gray").pack(
            side="right", padx=10
        )

        self.sf_inner = ttk.Frame(sf_lf)
        self.sf_inner.pack(fill="x", padx=5, pady=(0, 4))

        # Method selector row
        method_row = ttk.Frame(self.sf_inner)
        method_row.pack(fill="x", pady=(0, 4))
        ttk.Label(method_row, text=self.t("sf_method_label")).pack(side="left", padx=(0, 8))
        for val, key in [
            ("best-n", "sf_method_best_n"),
            ("batched", "sf_method_batched"),
            ("outlier-removal", "sf_method_outlier"),
        ]:
            ttk.Radiobutton(
                method_row,
                text=self.t(key),
                variable=self.var_sharp_frame_method,
                value=val,
                command=self.toggle_sharp_frame_method,
            ).pack(side="left", padx=6)

        # Method sub-option container — only one sub-frame visible at a time
        self.sf_subopts_container = ttk.Frame(self.sf_inner)
        self.sf_subopts_container.pack(fill="x", pady=(0, 4))

        self.sf_best_n_frame = ttk.Frame(self.sf_subopts_container)
        ttk.Label(self.sf_best_n_frame, text=self.t("sf_num_frames")).pack(side="left", padx=5)
        sf_num_frames_spin = ttk.Spinbox(
            self.sf_best_n_frame, from_=10, to=10000, increment=50,
            textvariable=self.var_sharp_frame_num_frames, width=8,
        )
        sf_num_frames_spin.pack(side="left", padx=5)
        ToolTip(sf_num_frames_spin, self.t("tip_sf_num_frames"))
        ttk.Label(self.sf_best_n_frame, text=self.t("sf_min_buffer")).pack(side="left", padx=(12, 5))
        ttk.Spinbox(
            self.sf_best_n_frame, from_=1, to=50,
            textvariable=self.var_sharp_frame_min_buffer, width=6,
        ).pack(side="left", padx=5)

        self.sf_batched_frame = ttk.Frame(self.sf_subopts_container)
        ttk.Label(self.sf_batched_frame, text=self.t("sf_batch_size")).pack(side="left", padx=5)
        ttk.Spinbox(
            self.sf_batched_frame, from_=2, to=100,
            textvariable=self.var_sharp_frame_batch_size, width=6,
        ).pack(side="left", padx=5)
        ttk.Label(self.sf_batched_frame, text=self.t("sf_batch_buffer")).pack(side="left", padx=(12, 5))
        ttk.Spinbox(
            self.sf_batched_frame, from_=0, to=20,
            textvariable=self.var_sharp_frame_batch_buffer, width=6,
        ).pack(side="left", padx=5)

        self.sf_outlier_frame = ttk.Frame(self.sf_subopts_container)
        ttk.Label(self.sf_outlier_frame, text=self.t("sf_window_size")).pack(side="left", padx=5)
        ttk.Spinbox(
            self.sf_outlier_frame, from_=3, to=100,
            textvariable=self.var_sharp_frame_window_size, width=6,
        ).pack(side="left", padx=5)
        ttk.Label(self.sf_outlier_frame, text=self.t("sf_sensitivity")).pack(side="left", padx=(12, 5))
        sf_sens_spin = ttk.Spinbox(
            self.sf_outlier_frame, from_=0, to=100,
            textvariable=self.var_sharp_frame_sensitivity, width=6,
        )
        sf_sens_spin.pack(side="left", padx=5)
        ToolTip(sf_sens_spin, self.t("tip_sf_sensitivity"))

        # Output folder row
        sf_out_row = ttk.Frame(self.sf_inner)
        sf_out_row.pack(fill="x")
        ttk.Label(sf_out_row, text=self.t("sf_output")).pack(side="left", padx=5)
        self.sf_output_entry = ttk.Entry(sf_out_row, textvariable=self.var_sharp_frame_output, width=42)
        self.sf_output_entry.pack(side="left", padx=5, fill="x", expand=True)
        ToolTip(self.sf_output_entry, self.t("tip_sf_output"))
        ttk.Button(
            sf_out_row, text=self.t("browse"), width=8,
            command=lambda: self.browse_folder(self.var_sharp_frame_output),
        ).pack(side="left", padx=5)

        self.toggle_sharp_frame_method()
        self.toggle_sharp_frame_options()

        # Per-phase run button (visible only in Sharp Frame Only mode)
        self.btn_sf_sharpen = ttk.Button(
            sf_lf, text=self.t("sf_run_sharpen"), command=self.run_sharpen_only
        )
        self.btn_sf_sharpen.pack(fill="x", padx=5, pady=(4, 2))
        self.btn_sf_sharpen.pack_forget()  # Hidden until SHARP_FRAME mode

        # ===== File Paths Section =====
        self.paths_frame = ttk.LabelFrame(scrollable_frame, text=self.t("paths_section"), padding=10)
        self.paths_frame.pack(fill="x", padx=padx, pady=pady)
        
        # Images folder
        self.add_path_entry(self.paths_frame, self.t("images_label"), self.var_images,
                           is_folder=True, row=0,
                           tooltip=self.t("tip_images"))
        
        # XML file
        self.add_path_entry(self.paths_frame, self.t("xml_label"), self.var_xml,
                           is_folder=False, row=1, filetypes=[("XML files", "*.xml")],
                           tooltip=self.t("tip_xml"))
        
        # PLY file
        self.add_path_entry(self.paths_frame, self.t("ply_label"), self.var_ply,
                           is_folder=False, row=2, filetypes=[("PLY files", "*.ply")],
                           tooltip=self.t("tip_ply"))
        
        # Output folder
        self.add_path_entry(self.paths_frame, self.t("output_label"), self.var_output,
                           is_folder=True, row=3,
                           tooltip=self.t("tip_output"))

        # Save references to rows that are hidden in Sharp Frame Only mode
        self.paths_xml_widgets = list(self.paths_frame.grid_slaves(row=1))
        self.paths_ply_widgets = list(self.paths_frame.grid_slaves(row=2))
        self.paths_out_widgets = list(self.paths_frame.grid_slaves(row=3))

        # ===== Tabbed Options Section (visible in BOTH modes) =====
        self.options_notebook = ttk.Notebook(scrollable_frame)
        self.options_notebook.pack(fill="x", padx=padx, pady=pady)

        proc_tab = self.proc_tab = ttk.Frame(self.options_notebook, padding=10)
        cubemap_tab = self.cubemap_tab = ttk.Frame(self.options_notebook, padding=10)
        mask_tab = self.mask_tab = ttk.Frame(self.options_notebook, padding=10)
        self.options_notebook.add(proc_tab, text=self.t("proc_section"))
        self.options_notebook.add(cubemap_tab, text=self.t("cubemap_section"))
        self.options_notebook.add(mask_tab, text=self.t("mask_section"))

        # --- Processing tab (shared, some widgets COLMAP-only) ---
        ttk.Label(proc_tab, text=self.t("max_images")).grid(row=0, column=0, sticky="w", padx=5)
        max_spin = ttk.Spinbox(proc_tab, from_=1, to=100000, increment=100,
                               textvariable=self.var_max_images, width=10)
        max_spin.grid(row=0, column=1, sticky="w", padx=5, pady=3)
        ToolTip(max_spin, self.t("tip_max_images"))

        ttk.Label(proc_tab, text=self.t("workers")).grid(row=0, column=2, sticky="w", padx=5)
        self.worker_spin = ttk.Spinbox(proc_tab, from_=1, to=32, increment=1,
                                       textvariable=self.var_num_workers, width=10)
        self.worker_spin.grid(row=0, column=3, sticky="w", padx=5, pady=3)
        ToolTip(self.worker_spin, self.t("tip_workers"))

        self.range_check = ttk.Checkbutton(proc_tab, text=self.t("range"),
                                           variable=self.var_range_enabled,
                                           command=self.toggle_range)
        self.range_check.grid(row=1, column=0, sticky="w", padx=5, pady=3)

        self.range_frame = ttk.Frame(proc_tab)
        self.range_frame.grid(row=1, column=1, columnspan=3, sticky="w", padx=5)

        ttk.Label(self.range_frame, text=self.t("range_start")).pack(side="left")
        self.range_start_spin = ttk.Spinbox(self.range_frame, from_=0, to=100000,
                                            textvariable=self.var_range_start, width=8)
        self.range_start_spin.pack(side="left", padx=2)
        ttk.Label(self.range_frame, text=self.t("range_end")).pack(side="left", padx=(10, 0))
        self.range_end_spin = ttk.Spinbox(self.range_frame, from_=1, to=100000,
                                          textvariable=self.var_range_end, width=8)
        self.range_end_spin.pack(side="left", padx=2)
        ToolTip(self.range_frame, self.t("tip_range"))
        self.toggle_range()

        ttk.Label(proc_tab, text=self.t("yaw_offset")).grid(row=2, column=0, sticky="w", padx=5)
        self.yaw_spin = ttk.Spinbox(proc_tab, from_=-180, to=180, increment=5,
                                    textvariable=self.var_yaw_offset, width=10)
        self.yaw_spin.grid(row=2, column=1, sticky="w", padx=5, pady=3)
        ToolTip(self.yaw_spin, self.t("tip_yaw_offset"))

        # --- Cubemap tab ---
        # In COLMAP mode the checkbox is locked ON; in LFS mode the user can toggle it.
        self.cubemap_enable_cb = ttk.Checkbutton(
            cubemap_tab,
            text=self.t("enable_cubemap"),
            variable=self.var_enable_cubemap,
            command=self.toggle_cubemap_options,
        )
        self.cubemap_enable_cb.grid(row=0, column=0, columnspan=4, sticky="w", padx=5, pady=(0, 6))
        ToolTip(self.cubemap_enable_cb, self.t("tip_enable_cubemap"))

        self.cubemap_inner_frame = ttk.Frame(cubemap_tab)
        self.cubemap_inner_frame.grid(row=1, column=0, columnspan=4, sticky="ew")

        ttk.Label(self.cubemap_inner_frame, text=self.t("crop_size")).grid(
            row=0, column=0, sticky="w", padx=5)
        self.crop_spin = ttk.Spinbox(self.cubemap_inner_frame, from_=256, to=4096, increment=64,
                                     textvariable=self.var_crop_size, width=10)
        self.crop_spin.grid(row=0, column=1, sticky="w", padx=5, pady=3)
        ToolTip(self.crop_spin, self.t("tip_crop_size"))

        ttk.Label(self.cubemap_inner_frame, text=self.t("fov")).grid(
            row=0, column=2, sticky="w", padx=5)
        self.fov_spin = ttk.Spinbox(self.cubemap_inner_frame, from_=60, to=120, increment=5,
                                    textvariable=self.var_fov_deg, width=10)
        self.fov_spin.grid(row=0, column=3, sticky="w", padx=5, pady=3)
        ToolTip(self.fov_spin, self.t("tip_fov"))

        ttk.Label(self.cubemap_inner_frame, text=self.t("skip_inline")).grid(
            row=1, column=0, sticky="w", padx=5, pady=(8, 0))
        self.skip_dir_checkbuttons = {}
        for i, direction in enumerate(VALID_DIRECTIONS):
            cb = ttk.Checkbutton(self.cubemap_inner_frame, text=direction.capitalize(),
                                 variable=self.var_skip_directions[direction])
            cb.grid(row=1, column=i + 1, padx=6, pady=(8, 0))
            self.skip_dir_checkbuttons[direction] = cb

        ttk.Label(self.cubemap_inner_frame, text=self.t("skip_note"),
                  foreground="gray").grid(row=2, column=0, columnspan=7, sticky="w",
                                         padx=5, pady=(2, 0))

        # --- Mask generation tab (COLMAP-only; informational note shown in LFS) ---
        self.lbl_mask_lfs_note = ttk.Label(
            mask_tab, text=self.t("lfs_mask_note"), foreground="gray"
        )

        self.yolo_enable_cb = ttk.Checkbutton(mask_tab, text=self.t("enable_yolo"),
                                              variable=self.var_generate_masks,
                                              command=self.toggle_mask_options)
        self.yolo_enable_cb.grid(row=1, column=0, columnspan=2, sticky="w")

        # Engine selector row
        engine_frame = ttk.Frame(mask_tab)
        engine_frame.grid(row=2, column=0, columnspan=4, sticky="w", pady=(4, 2))
        ttk.Label(engine_frame, text=self.t("mask_engine_label")).pack(side="left", padx=5)
        self.rb_engine_yolo = ttk.Radiobutton(
            engine_frame,
            text=self.t("mask_engine_yolo"),
            variable=self.var_mask_engine,
            value="yolo",
            command=self.toggle_mask_engine,
        )
        self.rb_engine_yolo.pack(side="left", padx=5)
        self.rb_engine_sam3 = ttk.Radiobutton(
            engine_frame,
            text=self.t("mask_engine_sam3"),
            variable=self.var_mask_engine,
            value="sam3",
            command=self.toggle_mask_engine,
        )
        self.rb_engine_sam3.pack(side="left", padx=5)

        # YOLO options frame
        self.yolo_frame = ttk.LabelFrame(mask_tab, text="YOLO", padding=6)
        self.yolo_frame.grid(row=3, column=0, columnspan=4, sticky="ew", pady=4)

        ttk.Label(self.yolo_frame, text=self.t("yolo_classes")).grid(row=0, column=0, sticky="w", padx=5)
        yolo_entry = ttk.Entry(self.yolo_frame, textvariable=self.var_yolo_classes, width=20)
        yolo_entry.grid(row=0, column=1, sticky="w", padx=5)
        ToolTip(yolo_entry, self.t("tip_yolo_classes"))

        ttk.Label(self.yolo_frame, text=self.t("yolo_conf")).grid(row=0, column=2, sticky="w", padx=5)
        conf_spin = ttk.Spinbox(self.yolo_frame, from_=0.0, to=1.0, increment=0.05,
                                textvariable=self.var_yolo_conf, width=8)
        conf_spin.grid(row=0, column=3, sticky="w", padx=5)
        ToolTip(conf_spin, self.t("tip_yolo_conf"))

        ttk.Label(self.yolo_frame, text=self.t("yolo_model")).grid(row=1, column=0, sticky="w", padx=5)
        model_combo = ttk.Combobox(
            self.yolo_frame,
            textvariable=self.var_yolo_model,
            values=YOLO_SEG_MODELS,
            state="readonly",
            width=24,
        )
        model_combo.grid(row=1, column=1, columnspan=3, sticky="w", padx=5)

        ttk.Checkbutton(self.yolo_frame, text=self.t("invert_mask"),
                        variable=self.var_invert_mask).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=5, pady=3)

        # SAM3 options frame
        self.sam3_frame = ttk.LabelFrame(mask_tab, text="SAM3", padding=6)
        self.sam3_frame.grid(row=4, column=0, columnspan=4, sticky="ew", pady=4)

        ttk.Label(self.sam3_frame, text=self.t("sam3_model")).grid(row=0, column=0, sticky="w", padx=5)
        sam3_model_entry = ttk.Entry(self.sam3_frame, textvariable=self.var_sam3_model, width=24)
        sam3_model_entry.grid(row=0, column=1, sticky="w", padx=5)
        ttk.Button(
            self.sam3_frame, text=self.t("browse"), width=8,
            command=lambda: self.browse_file(self.var_sam3_model, [("PT files", "*.pt"), ("All files", "*.*")])
        ).grid(row=0, column=2, padx=5)

        ttk.Label(self.sam3_frame, text=self.t("sam3_concepts")).grid(row=1, column=0, sticky="w", padx=5)
        sam3_concepts_entry = ttk.Entry(self.sam3_frame, textvariable=self.var_sam3_concepts, width=48)
        sam3_concepts_entry.grid(row=1, column=1, columnspan=2, sticky="ew", padx=5, pady=2)

        ttk.Label(self.sam3_frame, text=self.t("sam3_conf")).grid(row=2, column=0, sticky="w", padx=5)
        sam3_conf_spin = ttk.Spinbox(self.sam3_frame, from_=0.0, to=1.0, increment=0.05,
                                     textvariable=self.var_sam3_conf, width=8)
        sam3_conf_spin.grid(row=2, column=1, sticky="w", padx=5)

        ttk.Checkbutton(self.sam3_frame, text=self.t("sam3_half"),
                        variable=self.var_sam3_half).grid(
            row=2, column=2, sticky="w", padx=5)

        ttk.Label(
            self.sam3_frame, text=self.t("sam3_vram_warn"), foreground="orange"
        ).grid(row=3, column=0, columnspan=3, sticky="w", padx=5, pady=(4, 0))

        ttk.Checkbutton(self.sam3_frame, text=self.t("invert_mask"),
                        variable=self.var_invert_mask).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=5, pady=3)

        ttk.Separator(mask_tab, orient="horizontal").grid(row=5, column=0, columnspan=4, sticky="ew", pady=10)

        self.overexp_enable_cb = ttk.Checkbutton(mask_tab, text=self.t("enable_overexp"),
                                                 variable=self.var_mask_overexposure,
                                                 command=self.toggle_overexposure_options)
        self.overexp_enable_cb.grid(row=6, column=0, columnspan=2, sticky="w")

        self.overexposure_frame = ttk.Frame(mask_tab)
        self.overexposure_frame.grid(row=7, column=0, columnspan=4, sticky="ew", pady=5)

        ttk.Label(self.overexposure_frame, text=self.t("threshold")).grid(row=0, column=0, sticky="w", padx=5)
        thresh_spin = ttk.Spinbox(self.overexposure_frame, from_=200, to=255, increment=1,
                                  textvariable=self.var_overexposure_threshold, width=8)
        thresh_spin.grid(row=0, column=1, sticky="w", padx=5)
        ToolTip(thresh_spin, self.t("tip_overexp_threshold"))

        ttk.Label(self.overexposure_frame, text=self.t("dilate")).grid(row=0, column=2, sticky="w", padx=5)
        dilate_spin = ttk.Spinbox(self.overexposure_frame, from_=0, to=50, increment=1,
                                  textvariable=self.var_overexposure_dilate, width=8)
        dilate_spin.grid(row=0, column=3, sticky="w", padx=5)
        ToolTip(dilate_spin, self.t("tip_overexp_dilate"))

        self.toggle_mask_options()
        self.toggle_mask_engine()
        self.toggle_overexposure_options()
        tab_count = self.options_notebook.index("end")
        if tab_count > 0:
            self.options_notebook.select(min(old_options_tab_idx, tab_count - 1))
        
        # ===== Advanced Options Section (Accordion) =====
        adv_container = ttk.LabelFrame(scrollable_frame, text=self.t("advanced_section"), padding=10)
        adv_container.pack(fill="x", padx=padx, pady=pady)

        self.adv_toggle_btn = ttk.Button(
            adv_container,
            text="",
            command=self.toggle_advanced_section,
        )
        self.adv_toggle_btn.pack(anchor="w", padx=5, pady=(0, 5))

        self.adv_content_frame = ttk.Frame(adv_container)

        flip_cb = ttk.Checkbutton(
            self.adv_content_frame,
            text=self.t("flip_vertical"),
            variable=self.var_flip_vertical,
        )
        flip_cb.grid(row=0, column=0, sticky="w", padx=5)
        ToolTip(flip_cb, self.t("tip_flip_vertical"))

        self.rotate_z180_cb = ttk.Checkbutton(
            self.adv_content_frame,
            text=self.t("rotate_z180"),
            variable=self.var_rotate_z180,
        )
        self.rotate_z180_cb.grid(row=0, column=1, sticky="w", padx=5)

        self.apply_component_cb = ttk.Checkbutton(
            self.adv_content_frame,
            text=self.t("apply_component"),
            variable=self.var_apply_component,
        )
        self.apply_component_cb.grid(row=1, column=0, sticky="w", padx=5)

        ttk.Checkbutton(
            self.adv_content_frame,
            text=self.t("quiet"),
            variable=self.var_quiet,
        ).grid(row=1, column=1, sticky="w", padx=5)

        self.fix_upside_down_cb = ttk.Checkbutton(
            self.adv_content_frame,
            text=self.t("fix_upside_down"),
            variable=self.var_fix_upside_down,
        )
        self.fix_upside_down_cb.grid(row=2, column=0, sticky="w", padx=5)
        ToolTip(self.fix_upside_down_cb, self.t("tip_fix_upside_down"))

        self.lfs_copy_images_cb = ttk.Checkbutton(
            self.adv_content_frame,
            text=self.t("lfs_copy_images"),
            variable=self.var_lfs_copy_images,
        )
        self.lfs_copy_images_cb.grid(row=2, column=1, sticky="w", padx=5)
        ToolTip(self.lfs_copy_images_cb, self.t("tip_lfs_copy_images"))

        self.update_advanced_section_visibility()
        
        # ===== Action Buttons =====
        btn_frame = ttk.Frame(scrollable_frame)
        btn_frame.pack(fill="x", padx=padx, pady=10)
        
        ttk.Button(btn_frame, text=self.t("save_cfg"), command=self.save_config).pack(side="left", padx=5)
        ttk.Button(btn_frame, text=self.t("load_cfg"), command=self.load_config).pack(side="left", padx=5)
        ttk.Button(btn_frame, text=self.t("reset_defaults"), command=self.reset_defaults).pack(side="left", padx=5)
        
        self.run_btn = ttk.Button(btn_frame, text=self.t("run"), command=self.run_conversion, style="Accent.TButton")
        self.run_btn.pack(side="right", padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text=self.t("stop"), command=self.stop_conversion, state="disabled")
        self.stop_btn.pack(side="right", padx=5)
        
        # ===== Output Console =====
        console_frame = ttk.LabelFrame(scrollable_frame, text=self.t("console_section"), padding=10)
        console_frame.pack(fill="both", expand=True, padx=padx, pady=pady)
        
        self.console = scrolledtext.ScrolledText(console_frame, height=12, wrap=tk.WORD,
                                                  font=("Consolas", 9))
        self.console.pack(fill="both", expand=True)
        
        # Clear console button
        ttk.Button(console_frame, text=self.t("clear_log"),
                  command=lambda: self.console.delete(1.0, tk.END)).pack(anchor="e", pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(scrollable_frame, mode="indeterminate")
        self.progress.pack(fill="x", padx=padx, pady=5)

        if old_log:
            self.console.insert(tk.END, old_log)
            self.console.see(tk.END)
        
        # Apply mode-dependent visibility for the current mode selection
        self.toggle_mode_ui()
    
    def add_path_entry(self, parent, label, variable, is_folder=True, row=0, 
                       filetypes=None, tooltip=None):
        """Add a path entry with browse button."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=3)
        
        entry = ttk.Entry(parent, textvariable=variable, width=50)
        entry.grid(row=row, column=1, sticky="ew", padx=5, pady=3)
        parent.columnconfigure(1, weight=1)
        
        if tooltip:
            ToolTip(entry, tooltip)
        
        cmd = lambda: self.browse_folder(variable) if is_folder else self.browse_file(variable, filetypes)
        ttk.Button(parent, text=self.t("browse"), command=cmd, width=8).grid(row=row, column=2, padx=5, pady=3)
    
    def browse_folder(self, variable):
        """Open folder browser dialog."""
        path = filedialog.askdirectory(initialdir=variable.get() or ".")
        if path:
            variable.set(path)
    
    def browse_file(self, variable, filetypes=None):
        """Open file browser dialog."""
        filetypes = filetypes or [("All files", "*.*")]
        path = filedialog.askopenfilename(initialdir=os.path.dirname(variable.get()) or ".",
                                          filetypes=filetypes)
        if path:
            variable.set(path)
    
    def toggle_range(self):
        """Toggle range input fields state; always disabled in LFS mode."""
        is_colmap = self.var_output_mode.get() == "COLMAP"
        enabled = is_colmap and self.var_range_enabled.get()
        state = "normal" if enabled else "disabled"
        self.range_start_spin.config(state=state)
        self.range_end_spin.config(state=state)
    
    def toggle_mask_options(self):
        """Toggle mask engine selector and engine-specific frames based on enable checkbox."""
        enabled = self.var_generate_masks.get()
        for widget in (self.rb_engine_yolo, self.rb_engine_sam3):
            try:
                widget.config(state="normal" if enabled else "disabled")
            except tk.TclError:
                pass
        self.toggle_mask_engine()

    def toggle_mask_engine(self):
        """Show/hide YOLO or SAM3 frame depending on the selected engine."""
        enabled = self.var_generate_masks.get()
        engine = self.var_mask_engine.get()

        yolo_state = "normal" if enabled and engine == "yolo" else "disabled"
        for child in self.yolo_frame.winfo_children():
            try:
                child.config(state=yolo_state)
            except tk.TclError:
                pass

        sam3_state = "normal" if enabled and engine == "sam3" else "disabled"
        for child in self.sam3_frame.winfo_children():
            try:
                child.config(state=sam3_state)
            except tk.TclError:
                pass
    
    def toggle_overexposure_options(self):
        """Toggle overexposure mask options state."""
        state = "normal" if self.var_mask_overexposure.get() else "disabled"
        for child in self.overexposure_frame.winfo_children():
            try:
                child.config(state=state)
            except tk.TclError:
                pass

    def update_advanced_section_visibility(self):
        """Update accordion state for advanced options section."""
        expanded = self.var_advanced_expanded.get()
        if expanded:
            self.adv_content_frame.pack(fill="x", padx=0, pady=0)
            self.adv_toggle_btn.config(text=f"▼ {self.t('advanced_hide')}")
        else:
            self.adv_content_frame.pack_forget()
            self.adv_toggle_btn.config(text=f"▶ {self.t('advanced_show')}")

    def toggle_advanced_section(self):
        """Toggle expanded/collapsed state for advanced options."""
        self.var_advanced_expanded.set(not self.var_advanced_expanded.get())
        self.update_advanced_section_visibility()

    def toggle_cubemap_options(self):
        """Enable/disable cubemap inner controls based on the enable_cubemap toggle."""
        enabled = self.var_enable_cubemap.get()
        state = "normal" if enabled else "disabled"
        self.crop_spin.config(state=state)
        self.fov_spin.config(state=state)
        for cb in self.skip_dir_checkbuttons.values():
            cb.config(state=state)

    def toggle_mode_ui(self):
        """Enable/disable widgets based on the selected output mode."""
        mode = self.var_output_mode.get()
        is_colmap = mode == "COLMAP"
        is_sharp_only = mode == "SHARP_FRAME"
        is_mask_only = mode == "MASK_ONLY"

        # ── Sharp Frame Only mode: simplified two-phase UI.
        #    Hide XML/PLY/Output path rows and the processing/cubemap/mask tabs.
        #    Show the per-phase run buttons inside each section.
        if is_sharp_only:
            self.var_sharp_frame_enabled.set(True)
            if hasattr(self, "sf_enable_cb"):
                self.sf_enable_cb.config(state="disabled")
            # Hide XML (row 1), PLY (row 2), Output (row 3) — keep Images (row 0)
            for w in (getattr(self, "paths_xml_widgets", []) +
                      getattr(self, "paths_ply_widgets", []) +
                      getattr(self, "paths_out_widgets", [])):
                w.grid_remove()
            # Hide the tabbed options notebook (Processing / Cubemap / Mask)
            if self.options_notebook.winfo_ismapped():
                self.options_notebook.pack_forget()
            # Show the video and sharp-frame sections (hidden in COLMAP/LFS modes)
            if hasattr(self, "vid_lf") and not self.vid_lf.winfo_ismapped():
                self.vid_lf.pack(fill="x", padx=10, pady=(0, 5), after=self.mode_frame)
            if hasattr(self, "sf_lf") and not self.sf_lf.winfo_ismapped():
                self.sf_lf.pack(fill="x", padx=10, pady=(0, 5), after=self.vid_lf)
            # Label the sections as phase steps
            if hasattr(self, "vid_lf"):
                self.vid_lf.config(text=f"\u2460 {self.t('vid_section')}")
            if hasattr(self, "sf_lf"):
                self.sf_lf.config(text=f"\u2461 {self.t('sf_section')}")
            # Show the individual phase run buttons
            if hasattr(self, "btn_sf_extract"):
                self.btn_sf_extract.pack(fill="x", padx=5, pady=(4, 2))
            if hasattr(self, "btn_sf_sharpen"):
                self.btn_sf_sharpen.pack(fill="x", padx=5, pady=(4, 2))
            if hasattr(self, "adv_toggle_btn"):
                self.adv_toggle_btn.config(state="disabled")
            self._set_widget_tree_state(self.adv_content_frame, "disabled")
            self.toggle_sharp_frame_options()
            return

        # ── Restore widgets hidden by Sharp Frame Only mode ──────────────────
        if hasattr(self, "sf_enable_cb"):
            self.sf_enable_cb.config(state="normal")
        # Restore XML, PLY, Output rows
        for w in (getattr(self, "paths_xml_widgets", []) +
                  getattr(self, "paths_ply_widgets", []) +
                  getattr(self, "paths_out_widgets", [])):
            w.grid()
        # Restore the options notebook right after paths_frame
        if not self.options_notebook.winfo_ismapped():
            self.options_notebook.pack(fill="x", padx=10, pady=5, after=self.paths_frame)
        # Restore section titles
        if hasattr(self, "vid_lf"):
            self.vid_lf.config(text=self.t("vid_section"))
        if hasattr(self, "sf_lf"):
            self.sf_lf.config(text=self.t("sf_section"))
        # Hide the per-phase run buttons
        if hasattr(self, "btn_sf_extract"):
            self.btn_sf_extract.pack_forget()
        if hasattr(self, "btn_sf_sharpen"):
            self.btn_sf_sharpen.pack_forget()
        # Hide video extraction and sharp-frame sections in COLMAP/LFS modes
        if hasattr(self, "vid_lf") and self.vid_lf.winfo_ismapped():
            self.vid_lf.pack_forget()
        if hasattr(self, "sf_lf") and self.sf_lf.winfo_ismapped():
            self.sf_lf.pack_forget()
        if hasattr(self, "adv_toggle_btn"):
            self.adv_toggle_btn.config(state="normal")
        self._set_widget_tree_state(self.adv_content_frame, "normal")

        # Re-enable all notebook tabs (may have been disabled by MASK_ONLY mode).
        for _tab in (self.cubemap_tab, self.proc_tab, self.mask_tab):
            self.options_notebook.tab(_tab, state="normal")

        # ── Mask Only mode: no XML/PLY needed; only the Mask tab is relevant. ──
        if is_mask_only:
            for w in (getattr(self, "paths_xml_widgets", []) +
                      getattr(self, "paths_ply_widgets", [])):
                w.grid_remove()
            for w in getattr(self, "paths_out_widgets", []):
                w.grid()
            self.options_notebook.tab(self.cubemap_tab, state="disabled")
            self.options_notebook.tab(self.proc_tab, state="disabled")
            self.options_notebook.select(self.mask_tab)
            self.lbl_mask_lfs_note.grid_remove()
            self.yolo_enable_cb.config(state="normal")
            self.overexp_enable_cb.config(state="normal")
            self.toggle_mask_options()
            self.toggle_mask_engine()
            self.toggle_overexposure_options()
            if hasattr(self, "adv_toggle_btn"):
                self.adv_toggle_btn.config(state="disabled")
            self._set_widget_tree_state(self.adv_content_frame, "disabled")
            return

        # Notebook is visible; adjust per-widget state instead.

        # Cubemap tab: COLMAP always forces cubemap ON and locks the checkbox.
        if is_colmap:
            self.var_enable_cubemap.set(True)
            self.cubemap_enable_cb.config(state="disabled")
        else:
            self.cubemap_enable_cb.config(state="normal")

        # Processing tab: workers, range and yaw are COLMAP-only.
        colmap_only_state = "normal" if is_colmap else "disabled"
        self.worker_spin.config(state=colmap_only_state)
        self.range_check.config(state=colmap_only_state)
        self.yaw_spin.config(state=colmap_only_state)
        self.toggle_range()  # evaluates both mode and checkbox state

        # Advanced: COLMAP-only options.
        self.rotate_z180_cb.config(state=colmap_only_state)
        self.apply_component_cb.config(state=colmap_only_state)

        # Advanced: LFS-only options.
        lfs_only_state = "normal" if not is_colmap else "disabled"
        self.fix_upside_down_cb.config(state=lfs_only_state)
        self.lfs_copy_images_cb.config(state=lfs_only_state)

        # Mask tab: enabled in both COLMAP and LFS modes.
        self.lbl_mask_lfs_note.grid_remove()
        self.yolo_enable_cb.config(state="normal")
        self.overexp_enable_cb.config(state="normal")
        self.toggle_mask_options()
        self.toggle_mask_engine()
        self.toggle_overexposure_options()

        # Sync cubemap inner controls.
        self.toggle_cubemap_options()
        self.toggle_sharp_frame_options()
    
    def _set_widget_tree_state(self, widget, state):
        """Recursively set the state on a widget and all its descendants."""
        try:
            widget.config(state=state)
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._set_widget_tree_state(child, state)

    def toggle_sharp_frame_options(self):
        """Enable/disable the sharp frame sub-panel and sync method visibility."""
        if not hasattr(self, "sf_inner"):
            return
        enabled = self.var_sharp_frame_enabled.get()
        state = "normal" if enabled else "disabled"
        self._set_widget_tree_state(self.sf_inner, state)
        self.toggle_sharp_frame_method()

    def toggle_sharp_frame_method(self):
        """Show the sub-frame matching the selected method; hide the others."""
        if not hasattr(self, "sf_best_n_frame"):
            return
        method = self.var_sharp_frame_method.get()
        for frame, val in [
            (self.sf_best_n_frame, "best-n"),
            (self.sf_batched_frame, "batched"),
            (self.sf_outlier_frame, "outlier-removal"),
        ]:
            if val == method:
                frame.pack(fill="x")
            else:
                frame.pack_forget()

    def toggle_video_options(self):
        """Enable/disable the video sub-panel and sync mode visibility."""
        if not hasattr(self, "vid_inner"):
            return
        enabled = self.var_video_enabled.get()
        state = "normal" if enabled else "disabled"
        self._set_widget_tree_state(self.vid_inner, state)
        self.toggle_video_mode()

    def toggle_video_mode(self):
        """Show the sub-frame matching the selected video extraction mode."""
        if not hasattr(self, "vid_every_n_frame"):
            return
        mode = self.var_video_mode.get()
        for frame, val in [
            (self.vid_every_n_frame, "every-n"),
            (self.vid_fps_frame, "fps"),
        ]:
            if val == mode:
                frame.pack(fill="x")
            else:
                frame.pack_forget()

    def reset_defaults(self):
        """Reset all settings to defaults."""
        self.var_images.set(DEFAULTS["images"])
        self.var_xml.set(DEFAULTS["xml"])
        self.var_ply.set(DEFAULTS["ply"])
        self.var_output.set(DEFAULTS["output"])
        self.var_crop_size.set(DEFAULTS["crop_size"])
        self.var_fov_deg.set(DEFAULTS["fov_deg"])
        self.var_max_images.set(DEFAULTS["max_images"])
        self.var_range_enabled.set(False)
        self.var_range_start.set(DEFAULTS["range_start"])
        self.var_range_end.set(DEFAULTS["range_end"])
        self.var_num_workers.set(DEFAULTS["num_workers"])
        self.var_yaw_offset.set(DEFAULTS["yaw_offset"])
        self.var_generate_masks.set(DEFAULTS["generate_masks"])
        self.var_mask_engine.set(DEFAULTS["mask_engine"])
        self.var_invert_mask.set(DEFAULTS["invert_mask"])
        self.var_yolo_classes.set(DEFAULTS["yolo_classes"])
        self.var_yolo_conf.set(DEFAULTS["yolo_conf"])
        self.var_yolo_model.set(DEFAULTS["yolo_model"])
        self.var_sam3_model.set(DEFAULTS["sam3_model"])
        self.var_sam3_concepts.set(DEFAULTS["sam3_concepts"])
        self.var_sam3_conf.set(DEFAULTS["sam3_conf"])
        self.var_sam3_half.set(DEFAULTS["sam3_half"])
        self.var_mask_overexposure.set(DEFAULTS["mask_overexposure"])
        self.var_overexposure_threshold.set(DEFAULTS["overexposure_threshold"])
        self.var_overexposure_dilate.set(DEFAULTS["overexposure_dilate"])
        self.var_flip_vertical.set(DEFAULTS["flip_vertical"])
        self.var_rotate_z180.set(DEFAULTS["rotate_z180"])
        self.var_fix_upside_down.set(DEFAULTS["fix_upside_down"])
        self.var_lfs_copy_images.set(DEFAULTS["lfs_copy_images"])
        self.var_enable_cubemap.set(DEFAULTS["enable_cubemap"])
        self.var_apply_component.set(DEFAULTS["apply_component_transform"])
        self.var_quiet.set(DEFAULTS["quiet"])
        self.var_output_mode.set(DEFAULTS["output_mode"])

        for d in VALID_DIRECTIONS:
            self.var_skip_directions[d].set(False)

        self.var_sharp_frame_enabled.set(DEFAULTS["sharp_frame_enabled"])
        self.var_sharp_frame_method.set(DEFAULTS["sharp_frame_method"])
        self.var_sharp_frame_num_frames.set(DEFAULTS["sharp_frame_num_frames"])
        self.var_sharp_frame_min_buffer.set(DEFAULTS["sharp_frame_min_buffer"])
        self.var_sharp_frame_batch_size.set(DEFAULTS["sharp_frame_batch_size"])
        self.var_sharp_frame_batch_buffer.set(DEFAULTS["sharp_frame_batch_buffer"])
        self.var_sharp_frame_window_size.set(DEFAULTS["sharp_frame_window_size"])
        self.var_sharp_frame_sensitivity.set(DEFAULTS["sharp_frame_sensitivity"])
        self.var_sharp_frame_output.set(DEFAULTS["sharp_frame_output"])

        self.var_video_enabled.set(DEFAULTS["video_enabled"])
        self.var_video_path.set(DEFAULTS["video_path"])
        self.var_video_mode.set(DEFAULTS["video_mode"])
        self.var_video_every_n.set(DEFAULTS["video_every_n"])
        self.var_video_fps.set(DEFAULTS["video_fps"])
        self.var_video_start_time.set(DEFAULTS["video_start_time"])
        self.var_video_end_time.set(DEFAULTS["video_end_time"])
        self.var_video_output.set(DEFAULTS["video_output"])

        self.toggle_mode_ui()
        
        self.log(f"{self.t('reset_done')}\n")
    
    def _build_mask_only_cmd(self, images_override=None) -> list:
        """Build command arguments for metashape_360_lfs.py --mask-only."""
        cmd = self._get_base_python_cmd("metashape_360_lfs.py")
        images = images_override if images_override else self.var_images.get()
        cmd.extend(["--mask-only"])
        cmd.extend(["--images", images])
        cmd.extend(["--output", self.var_output.get()])
        cmd.extend(["--max-images", str(self.var_max_images.get())])
        if self.var_generate_masks.get():
            cmd.append("--generate-masks")
            cmd.extend(["--mask-engine", self.var_mask_engine.get()])
            if self.var_mask_engine.get() == "sam3":
                cmd.extend(["--sam3-model", self.var_sam3_model.get()])
                cmd.extend(["--sam3-concepts", self.var_sam3_concepts.get()])
                cmd.extend(["--sam3-conf", str(self.var_sam3_conf.get())])
                if not self.var_sam3_half.get():
                    cmd.append("--no-sam3-half")
            else:
                cmd.extend(["--yolo-model", self.var_yolo_model.get()])
                yolo_cls = self.var_yolo_classes.get().strip()
                if yolo_cls:
                    cmd.extend(["--yolo-classes", yolo_cls])
                cmd.extend(["--yolo-conf", str(self.var_yolo_conf.get())])
            if self.var_invert_mask.get():
                cmd.append("--invert-mask")
        if self.var_mask_overexposure.get():
            cmd.append("--mask-overexposure")
            cmd.extend(["--overexposure-threshold", str(self.var_overexposure_threshold.get())])
            cmd.extend(["--overexposure-dilate", str(self.var_overexposure_dilate.get())])
        if self.var_quiet.get():
            cmd.append("--quiet")
        return cmd

    def build_command(self, images_override=None):
        """Build the command line arguments based on the selected output mode."""
        mode = self.var_output_mode.get()
        if mode == "LFS":
            return self._build_lfs_cmd(images_override=images_override)
        if mode == "SHARP_FRAME":
            return self._build_sharp_frame_standalone_cmd(images_override=images_override)
        if mode == "MASK_ONLY":
            return self._build_mask_only_cmd(images_override=images_override)
        return self._build_colmap_cmd(images_override=images_override)

    def _get_base_python_cmd(self, script_name: str) -> list:
        """Return the base interpreter + script invocation for a given script name."""
        if getattr(sys, "frozen", False):
            exe = self.get_app_base_dir() / script_name.replace(".py", ".exe")
            if not exe.exists():
                raise FileNotFoundError(
                    f"Required converter executable was not found: {exe}"
                )
            return [str(exe)]

        script_path = self.get_app_base_dir() / script_name
        venv_python = self.get_app_base_dir() / ".venv" / "Scripts" / "python.exe"
        if not venv_python.exists():
            venv_python = self.get_app_base_dir() / ".venv" / "bin" / "python"
        python_exe = str(venv_python) if venv_python.exists() else sys.executable
        return [python_exe, "-u", str(script_path)]

    def _build_colmap_cmd(self, images_override=None) -> list:
        """Build command arguments for metashape_360_to_colmap.py."""
        cmd = self._get_base_python_cmd("metashape_360_to_colmap.py")

        images = images_override if images_override else self.var_images.get()
        cmd.extend(["--images", images])
        cmd.extend(["--xml", self.var_xml.get()])
        cmd.extend(["--output", self.var_output.get()])

        if self.var_ply.get().strip():
            cmd.extend(["--ply", self.var_ply.get()])

        cmd.extend(["--crop-size", str(self.var_crop_size.get())])
        cmd.extend(["--fov-deg", str(self.var_fov_deg.get())])
        cmd.extend(["--max-images", str(self.var_max_images.get())])
        cmd.extend(["--num-workers", str(self.var_num_workers.get())])
        cmd.extend(["--yaw-offset", str(self.var_yaw_offset.get())])

        if self.var_range_enabled.get():
            cmd.extend(["--range-images", f"{self.var_range_start.get()}-{self.var_range_end.get()}"])

        skip_dirs = [d for d in VALID_DIRECTIONS if self.var_skip_directions[d].get()]
        if skip_dirs:
            cmd.extend(["--skip-directions", ",".join(skip_dirs)])

        if self.var_generate_masks.get():
            cmd.append("--generate-masks")
            cmd.extend(["--mask-engine", self.var_mask_engine.get()])
            if self.var_mask_engine.get() == "sam3":
                cmd.extend(["--sam3-model", self.var_sam3_model.get()])
                cmd.extend(["--sam3-concepts", self.var_sam3_concepts.get()])
                cmd.extend(["--sam3-conf", str(self.var_sam3_conf.get())])
                if not self.var_sam3_half.get():
                    cmd.append("--no-sam3-half")
            else:
                cmd.extend(["--yolo-classes", self.var_yolo_classes.get()])
                cmd.extend(["--yolo-conf", str(self.var_yolo_conf.get())])
                cmd.extend(["--yolo-model", self.var_yolo_model.get()])
            if self.var_invert_mask.get():
                cmd.append("--invert-mask")

        if self.var_mask_overexposure.get():
            cmd.append("--mask-overexposure")
            cmd.extend(["--overexposure-threshold", str(self.var_overexposure_threshold.get())])
            cmd.extend(["--overexposure-dilate", str(self.var_overexposure_dilate.get())])

        if self.var_flip_vertical.get():
            cmd.append("--flip-vertical")
        else:
            cmd.append("--no-flip-vertical")

        if self.var_rotate_z180.get():
            cmd.append("--rotate-z180")
        else:
            cmd.append("--no-rotate-z180")

        if self.var_apply_component.get():
            cmd.append("--apply-component-transform-for-ply")

        if self.var_quiet.get():
            cmd.append("--quiet")

        return cmd

    def _build_lfs_cmd(self, images_override=None) -> list:
        """Build command arguments for metashape_360_lfs.py."""
        cmd = self._get_base_python_cmd("metashape_360_lfs.py")

        images = images_override if images_override else self.var_images.get()
        cmd.extend(["--images", images])
        cmd.extend(["--xml", self.var_xml.get()])
        cmd.extend(["--output", self.var_output.get()])

        if self.var_ply.get().strip():
            cmd.extend(["--ply", self.var_ply.get()])

        cmd.extend(["--max-images", str(self.var_max_images.get())])

        if not self.var_fix_upside_down.get():
            cmd.append("--no-fix-rotation")

        if not self.var_lfs_copy_images.get() and not self.var_enable_cubemap.get():
            # --no-copy-images is ignored in split mode (crops are always written)
            cmd.append("--no-copy-images")

        # Cubemap split options
        if self.var_enable_cubemap.get():
            cmd.append("--split-cubemap")
            cmd.extend(["--crop-size", str(self.var_crop_size.get())])
            cmd.extend(["--fov-deg", str(self.var_fov_deg.get())])
            skip_dirs = [d for d in VALID_DIRECTIONS if self.var_skip_directions[d].get()]
            if skip_dirs:
                cmd.extend(["--skip-directions", ",".join(skip_dirs)])

        if self.var_quiet.get():
            cmd.append("--quiet")

        # Mask generation options (same args available in both modes)
        if self.var_generate_masks.get():
            cmd.append("--generate-masks")
            cmd.extend(["--mask-engine", self.var_mask_engine.get()])
            if self.var_mask_engine.get() == "sam3":
                cmd.extend(["--sam3-model", self.var_sam3_model.get()])
                cmd.extend(["--sam3-concepts", self.var_sam3_concepts.get()])
                cmd.extend(["--sam3-conf", str(self.var_sam3_conf.get())])
                if not self.var_sam3_half.get():
                    cmd.append("--no-sam3-half")
            else:
                cmd.extend(["--yolo-model", self.var_yolo_model.get()])
                yolo_cls = self.var_yolo_classes.get().strip()
                if yolo_cls:
                    cmd.extend(["--yolo-classes", yolo_cls])
                cmd.extend(["--yolo-conf", str(self.var_yolo_conf.get())])
            if self.var_invert_mask.get():
                cmd.append("--invert-mask")
        if self.var_mask_overexposure.get():
            cmd.append("--mask-overexposure")
            cmd.extend(["--overexposure-threshold", str(self.var_overexposure_threshold.get())])
            cmd.extend(["--overexposure-dilate", str(self.var_overexposure_dilate.get())])

        return cmd

    def _build_sharp_frame_cmd(self, images_override=None) -> tuple:
        """Build the sharp-frames CLI command for the pre-filter step.

        Returns (cmd_list, output_dir_str).
        """
        images = images_override if images_override else self.var_images.get()
        sf_out = self.var_sharp_frame_output.get().strip()
        if not sf_out:
            images_path = Path(images)
            sf_out = str(images_path.parent / (images_path.name + "_sharp"))

        base = self.get_app_base_dir()
        sf_exe = base / ".venv" / "Scripts" / "sharp-frames.exe"
        if not sf_exe.exists():
            sf_exe = base / ".venv" / "bin" / "sharp-frames"
        if sf_exe.exists():
            cmd = [str(sf_exe)]
        else:
            # Fallback: run via the venv python as a module (some install layouts)
            venv_python = base / ".venv" / "Scripts" / "python.exe"
            if not venv_python.exists():
                venv_python = base / ".venv" / "bin" / "python"
            python_exe = str(venv_python) if venv_python.exists() else sys.executable
            cmd = [python_exe, "-m", "sharp_frames"]

        cmd.extend([images, sf_out, "--force-overwrite"])

        method = self.var_sharp_frame_method.get()
        cmd.extend(["--selection-method", method])

        if method == "best-n":
            cmd.extend(["--num-frames", str(self.var_sharp_frame_num_frames.get())])
            cmd.extend(["--min-buffer", str(self.var_sharp_frame_min_buffer.get())])
        elif method == "batched":
            cmd.extend(["--batch-size", str(self.var_sharp_frame_batch_size.get())])
            cmd.extend(["--batch-buffer", str(self.var_sharp_frame_batch_buffer.get())])
        else:
            cmd.extend(["--outlier-window-size", str(self.var_sharp_frame_window_size.get())])
            cmd.extend(["--outlier-sensitivity", str(self.var_sharp_frame_sensitivity.get())])

        return cmd, sf_out

    def _build_sharp_frame_standalone_cmd(self, images_override=None) -> list:
        """Build the sharp-frames CLI command for Sharp Frame Only mode.

        Uses the output folder (var_output) as the destination.
        If the output folder is empty, auto-derives it from the images folder name.
        """
        images = images_override if images_override else self.var_images.get()
        out = self.var_sharp_frame_output.get().strip()
        if not out:
            images_path = Path(images)
            out = str(images_path.parent / (images_path.name + "_sharp"))

        base = self.get_app_base_dir()
        sf_exe = base / ".venv" / "Scripts" / "sharp-frames.exe"
        if not sf_exe.exists():
            sf_exe = base / ".venv" / "bin" / "sharp-frames"
        if sf_exe.exists():
            cmd = [str(sf_exe)]
        else:
            # Fallback: run via the venv python as a module.
            venv_python = base / ".venv" / "Scripts" / "python.exe"
            if not venv_python.exists():
                venv_python = base / ".venv" / "bin" / "python"
            python_exe = str(venv_python) if venv_python.exists() else sys.executable
            cmd = [python_exe, "-m", "sharp_frames"]

        cmd.extend([images, out, "--force-overwrite"])

        method = self.var_sharp_frame_method.get()
        cmd.extend(["--selection-method", method])

        if method == "best-n":
            cmd.extend(["--num-frames", str(self.var_sharp_frame_num_frames.get())])
            cmd.extend(["--min-buffer", str(self.var_sharp_frame_min_buffer.get())])
        elif method == "batched":
            cmd.extend(["--batch-size", str(self.var_sharp_frame_batch_size.get())])
            cmd.extend(["--batch-buffer", str(self.var_sharp_frame_batch_buffer.get())])
        else:
            cmd.extend(["--outlier-window-size", str(self.var_sharp_frame_window_size.get())])
            cmd.extend(["--outlier-sensitivity", str(self.var_sharp_frame_sensitivity.get())])

        return cmd

    def _run_video_extraction(self) -> str:
        """Extract frames from a video file using cv2.

        Streams progress to the output queue.
        Returns the path to the folder containing extracted frames.
        """
        import cv2 as _cv2

        video_path = self.var_video_path.get().strip()
        vid_out = self.var_video_output.get().strip()
        if not vid_out:
            vid_out = str(Path(video_path).parent / (Path(video_path).stem + "_frames"))

        out_dir = Path(vid_out)
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = _cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
        native_fps = cap.get(_cv2.CAP_PROP_FPS) or 30.0
        duration = total_frames / native_fps

        self.output_queue.put(
            f"  File: {Path(video_path).name}\n"
            f"  Duration: {duration:.1f}s  |  Total frames: {total_frames}"
            f"  |  Native FPS: {native_fps:.3f}\n"
            f"  Output: {vid_out}\n\n"
        )

        start_t = self.var_video_start_time.get()
        end_t = self.var_video_end_time.get()
        start_frame = max(0, int(round(start_t * native_fps)))
        end_frame = min(
            total_frames,
            int(round(end_t * native_fps)) if end_t > 0 else total_frames,
        )

        mode = self.var_video_mode.get()
        if mode == "every-n":
            step = max(1, self.var_video_every_n.get())
            frames_to_extract = set(range(start_frame, end_frame, step))
        else:  # fps
            target_fps = max(0.001, self.var_video_fps.get())
            interval = max(1, round(native_fps / target_fps))
            frames_to_extract = set(range(start_frame, end_frame, interval))

        total_selected = len(frames_to_extract)
        self.output_queue.put(f"  Extracting {total_selected} frames...\n")

        n_digits = len(str(total_selected))
        if start_frame > 0:
            cap.set(_cv2.CAP_PROP_POS_FRAMES, start_frame)

        current_pos = start_frame
        extracted = 0
        while current_pos < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if current_pos in frames_to_extract:
                extracted += 1
                fname = out_dir / f"frame_{extracted:0{n_digits}d}.jpg"
                _cv2.imwrite(str(fname), frame, [_cv2.IMWRITE_JPEG_QUALITY, 95])
                if extracted % 100 == 0:
                    self.output_queue.put(
                        f"  Progress: {extracted}/{total_selected} frames extracted\n"
                    )
            current_pos += 1

        cap.release()
        self.output_queue.put(
            f"  Done: {extracted} frames saved to:\n  {vid_out}\n\n"
        )
        return vid_out

    def get_app_base_dir(self):
        """Return directory that contains app runtime files.

        - dev: folder containing this .py file
        - frozen: folder containing the .exe
        """
        if getattr(sys, "frozen", False):
            return Path(sys.executable).parent
        return Path(__file__).parent
    
    def validate_inputs(self):
        """Validate required inputs before running."""
        errors = []

        # Video extraction is only active in SHARP_FRAME mode; the section is hidden
        # in COLMAP/LFS mode. Guard against stale True values from a previous mode.
        mode = self.var_output_mode.get()
        is_sharp_only = mode == "SHARP_FRAME"
        is_mask_only = mode == "MASK_ONLY"
        video_active = is_sharp_only and self.var_video_enabled.get()
        if video_active:
            if not self.var_video_path.get().strip():
                errors.append(self.t("err_video_required"))
            elif not Path(self.var_video_path.get()).exists():
                errors.append(self.t("err_video_missing", path=self.var_video_path.get()))
        else:
            if not self.var_images.get().strip():
                errors.append(self.t("err_images_required"))
            elif not Path(self.var_images.get()).exists():
                errors.append(self.t("err_images_missing", path=self.var_images.get()))

        if not is_sharp_only and not is_mask_only:
            if not self.var_xml.get().strip():
                errors.append(self.t("err_xml_required"))
            elif not Path(self.var_xml.get()).exists():
                errors.append(self.t("err_xml_missing", path=self.var_xml.get()))

        if not is_sharp_only and not self.var_output.get().strip():
            errors.append(self.t("err_output_required"))

        if not is_sharp_only and not is_mask_only and self.var_ply.get().strip() and not Path(self.var_ply.get()).exists():
            errors.append(self.t("err_ply_missing", path=self.var_ply.get()))

        if errors:
            messagebox.showerror(self.t("err_title"), "\n".join(errors))
            return False
        return True
    
    def run_conversion(self):
        """Validate inputs and start the conversion pipeline in a background thread."""
        if not self.validate_inputs():
            return

        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress.start()
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def run_extract_only(self):
        """Validate and run only Phase 1: video frame extraction."""
        if not self.var_video_enabled.get():
            messagebox.showwarning(
                self.t("err_title"),
                "Video extraction is not enabled. Enable it first.",
            )
            return
        if not self.var_video_path.get().strip():
            messagebox.showerror(self.t("err_title"), self.t("err_video_required"))
            return
        if not Path(self.var_video_path.get()).exists():
            messagebox.showerror(
                self.t("err_title"),
                self.t("err_video_missing", path=self.var_video_path.get()),
            )
            return

        self.run_btn.config(state="disabled")
        if hasattr(self, "btn_sf_extract"):
            self.btn_sf_extract.config(state="disabled")
        if hasattr(self, "btn_sf_sharpen"):
            self.btn_sf_sharpen.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress.start()
        threading.Thread(target=self._run_extract_only_task, daemon=True).start()

    def _run_extract_only_task(self):
        """Background thread for Phase 1 (frame extraction only)."""
        sep = "=" * 50
        try:
            self.output_queue.put(f"{sep}\n[\u2460 Phase 1] Video Frame Extraction\n{sep}\n\n")
            result_dir = self._run_video_extraction()
            self.output_queue.put(
                f"\nExtraction complete.\nFrames saved to: {result_dir}\n{sep}\n"
            )
        except Exception as e:
            self.output_queue.put(f"Error during frame extraction: {e}\n")
        finally:
            self.output_queue.put("__DONE__")

    def run_sharpen_only(self):
        """Validate and run only Phase 2: sharp frame selection."""
        # Source images: either the video output folder or the explicit images folder
        if self.var_video_enabled.get():
            vid_out = self.var_video_output.get().strip()
            if not vid_out and self.var_video_path.get().strip():
                vid_out = str(
                    Path(self.var_video_path.get()).parent
                    / (Path(self.var_video_path.get()).stem + "_frames")
                )
            if not vid_out or not Path(vid_out).exists():
                messagebox.showerror(
                    self.t("err_title"),
                    f"Video output folder not found:\n{vid_out}\n\nRun Phase 1 first.",
                )
                return
            images_override = vid_out
        else:
            if not self.var_images.get().strip():
                messagebox.showerror(self.t("err_title"), self.t("err_images_required"))
                return
            if not Path(self.var_images.get()).exists():
                messagebox.showerror(
                    self.t("err_title"),
                    self.t("err_images_missing", path=self.var_images.get()),
                )
                return
            images_override = None

        self.run_btn.config(state="disabled")
        if hasattr(self, "btn_sf_extract"):
            self.btn_sf_extract.config(state="disabled")
        if hasattr(self, "btn_sf_sharpen"):
            self.btn_sf_sharpen.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress.start()
        threading.Thread(
            target=self._run_sharpen_only_task, args=(images_override,), daemon=True
        ).start()

    def _run_sharpen_only_task(self, images_override):
        """Background thread for Phase 2 (sharp frame selection only)."""
        sep = "=" * 50
        try:
            cmd = self._build_sharp_frame_standalone_cmd(images_override=images_override)
            self.output_queue.put(
                f"{sep}\n[\u2461 Phase 2] Sharp Frame Selection\n{sep}\n"
                f"{self.t('cmd', cmd=' '.join(cmd))}\n\n"
            )
            rc = self._run_subprocess(cmd)
            self.output_queue.put(f"\n{sep}\n")
            if rc == 0:
                self.output_queue.put("Sharp frame selection complete.\n")
            else:
                self.output_queue.put(f"Sharp frame selection failed (exit code {rc}).\n")
        except Exception as e:
            self.output_queue.put(f"Error during sharp frame selection: {e}\n")
        finally:
            self.output_queue.put("__DONE__")

    def _run_subprocess(self, cmd) -> int:
        """Launch a subprocess, stream its stdout to the output queue, and return the exit code."""
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            universal_newlines=True,
            cwd=str(self.get_app_base_dir()),
            env=self.get_subprocess_env(),
            creationflags=self.get_subprocess_creationflags(),
        )

        line_buf = ""
        while True:
            ch = self.process.stdout.read(1)
            if ch == "":
                if line_buf:
                    self.output_queue.put(line_buf + "\n")
                    line_buf = ""
                if self.process.poll() is not None:
                    break
                continue
            if ch in ("\n", "\r"):
                if line_buf:
                    self.output_queue.put(line_buf + "\n")
                    line_buf = ""
                continue
            line_buf += ch

        self.process.wait()
        return self.process.returncode

    def _run_pipeline(self):
        """Background thread: video extraction → sharp frame → main conversion."""
        try:
            images_override = None
            sep = "=" * 50

            is_sharp_only = self.var_output_mode.get() == "SHARP_FRAME"
            # Video extraction and sharp-frame pre-filtering are only applicable in
            # SHARP_FRAME mode. Their UI sections are hidden in COLMAP/LFS modes, so
            # stale True values from a previous mode must not trigger these steps.
            video_active = is_sharp_only and self.var_video_enabled.get()
            # In SHARP_FRAME mode the sharp-frame command IS the main conversion step
            # (built by build_command), so it must not also run as a pre-filter.
            # In COLMAP/LFS modes the section is hidden — sf_active is always False.
            sf_active = False
            total_steps = int(video_active) + int(sf_active) + 1
            current_step = 1

            # Step: Video frame extraction
            if video_active:
                self.output_queue.put(
                    f"[Step {current_step}/{total_steps}] Video frame extraction\n\n"
                )
                try:
                    images_override = self._run_video_extraction()
                except Exception as e:
                    self.output_queue.put(f"Video extraction failed: {e}\n")
                    return
                self.output_queue.put(f"{sep}\n")
                current_step += 1

            # Step: Sharp frame pre-filter
            if sf_active:
                sf_cmd, sf_out = self._build_sharp_frame_cmd(images_override=images_override)
                self.output_queue.put(
                    f"[Step {current_step}/{total_steps}] Sharp frame extraction\n"
                    f"{self.t('cmd', cmd=' '.join(sf_cmd))}\n\n"
                )
                rc = self._run_subprocess(sf_cmd)
                self.output_queue.put(f"\n{sep}\n")
                if rc != 0:
                    self.output_queue.put(
                        f"Sharp frame extraction failed (exit code {rc}).\n"
                    )
                    return
                self.output_queue.put("Sharp frame extraction completed.\n\n")
                images_override = sf_out
                current_step += 1

            # Step: Main conversion
            try:
                main_cmd = self.build_command(images_override=images_override)
            except Exception as e:
                self.output_queue.put(f"{self.t('error', error=e)}\n")
                return

            self.output_queue.put(
                f"[Step {current_step}/{total_steps}] {self.t('cmd', cmd=' '.join(main_cmd))}\n\n"
            )
            rc = self._run_subprocess(main_cmd)
            self.output_queue.put(f"\n{sep}\n")
            if rc == 0:
                self.output_queue.put(f"{self.t('done_ok')}\n")
            else:
                self.output_queue.put(f"{self.t('done_ng', code=rc)}\n")

        except Exception as e:
            self.output_queue.put(f"{self.t('error', error=e)}\n")
        finally:
            self.output_queue.put("__DONE__")

    def get_subprocess_env(self):
        """Return environment for subprocess execution.

        Force unbuffered stdio so CLI logs are streamed into the GUI promptly,
        especially when the GUI itself is running as a windowed executable.
        """
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        return env

    def get_subprocess_creationflags(self):
        """Return subprocess creation flags.

        On Windows, suppress creating an extra console window when the GUI
        launches the CLI executable.
        """
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            return subprocess.CREATE_NO_WINDOW
        return 0
    
    def stop_conversion(self):
        """Stop the running conversion process."""
        if self.process:
            self.stop_process(force=False)

            # Restore UI state immediately so the user can run again
            # even if the worker thread takes time to observe process exit.
            self.run_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.progress.stop()

            self.log(f"\n{self.t('stopped')}\n")
            self.process = None

    def stop_process(self, force=False):
        """Stop the current CLI process.

        If force=True on Windows, kill the full process tree to ensure the
        bundled CLI process is terminated immediately.
        """
        if not self.process:
            return

        proc = self.process
        try:
            if force and os.name == "nt":
                # taskkill /T /F kills the target process and its child tree.
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    creationflags=self.get_subprocess_creationflags(),
                )
            else:
                proc.terminate()
        except Exception:
            pass

        try:
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def on_close_requested(self):
        """Handle window close with confirmation while conversion is running."""
        if self.process and self.process.poll() is None:
            should_exit = messagebox.askyesno(
                self.t("exit_confirm_title"),
                self.t("exit_confirm_running"),
            )
            if not should_exit:
                return
            self.stop_process(force=True)

        self.is_closing = True
        self.root.destroy()
    
    def poll_output(self):
        """Poll the output queue for process output."""
        if self.is_closing:
            return

        try:
            while True:
                msg = self.output_queue.get_nowait()
                if msg == "__DONE__":
                    self.run_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    self.progress.stop()
                    self.process = None
                    # Re-enable per-phase SF buttons if they exist
                    for btn in ("btn_sf_extract", "btn_sf_sharpen"):
                        try:
                            getattr(self, btn).config(state="normal")
                        except (AttributeError, tk.TclError):
                            pass
                else:
                    self.log(msg)
        except queue.Empty:
            pass
        try:
            self.root.after(100, self.poll_output)
        except tk.TclError:
            # Window is already closing/closed.
            return
    
    def log(self, message):
        """Log message to console."""
        self.console.insert(tk.END, message)
        self.console.see(tk.END)
    
    def save_config(self):
        """Save current settings to config.txt."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Config files", "*.txt"), ("All files", "*.*")],
            initialfile="config.txt"
        )
        if not filepath:
            return
        
        q = lambda p: f'"{p}"' if " " in p else p
        lines = [
            "# Configuration file for metashape_360_to_colmap.py",
            "# Generated by GUI application",
            "",
            f"images={q(self.var_images.get())}",
            f"xml={q(self.var_xml.get())}",
            f"ply={q(self.var_ply.get())}",
            f"output={q(self.var_output.get())}",
            "",
            f"crop-size={self.var_crop_size.get()}",
            f"fov-deg={self.var_fov_deg.get()}",
            f"max-images={self.var_max_images.get()}",
        ]
        
        if self.var_range_enabled.get():
            lines.append(f"range-images={self.var_range_start.get()}-{self.var_range_end.get()}")
        
        lines.extend([
            f"num-workers={self.var_num_workers.get()}",
            f"yaw-offset={self.var_yaw_offset.get()}",
            "",
        ])
        
        skip_dirs = [d for d in VALID_DIRECTIONS if self.var_skip_directions[d].get()]
        lines.append(f"skip-directions={','.join(skip_dirs)}")
        
        lines.extend([
            "",
            f"generate-masks={self.var_generate_masks.get()}",
            f"mask-engine={self.var_mask_engine.get()}",
            f"invert-mask={self.var_invert_mask.get()}",
            f"yolo-classes={self.var_yolo_classes.get()}",
            f"yolo-conf={self.var_yolo_conf.get()}",
            f"yolo-model={self.var_yolo_model.get()}",
            f"sam3-model={self.var_sam3_model.get()}",
            f"sam3-concepts={self.var_sam3_concepts.get()}",
            f"sam3-conf={self.var_sam3_conf.get()}",
            f"sam3-half={self.var_sam3_half.get()}",
            "",
            f"mask-overexposure={self.var_mask_overexposure.get()}",
            f"overexposure-threshold={self.var_overexposure_threshold.get()}",
            f"overexposure-dilate={self.var_overexposure_dilate.get()}",
            "",
            f"flip-vertical={self.var_flip_vertical.get()}",
            f"rotate-z180={self.var_rotate_z180.get()}",
            f"fix-upside-down={self.var_fix_upside_down.get()}",
            f"lfs-copy-images={self.var_lfs_copy_images.get()}",
            f"enable-cubemap={self.var_enable_cubemap.get()}",
            f"apply-component-transform-for-ply={self.var_apply_component.get()}",
            f"quiet={self.var_quiet.get()}",
            f"output-mode={self.var_output_mode.get()}",
            f"language={self.var_language.get()}",
            "",
            f"sharp-frame-enabled={self.var_sharp_frame_enabled.get()}",
            f"sharp-frame-method={self.var_sharp_frame_method.get()}",
            f"sharp-frame-num-frames={self.var_sharp_frame_num_frames.get()}",
            f"sharp-frame-min-buffer={self.var_sharp_frame_min_buffer.get()}",
            f"sharp-frame-batch-size={self.var_sharp_frame_batch_size.get()}",
            f"sharp-frame-batch-buffer={self.var_sharp_frame_batch_buffer.get()}",
            f"sharp-frame-window-size={self.var_sharp_frame_window_size.get()}",
            f"sharp-frame-sensitivity={self.var_sharp_frame_sensitivity.get()}",
            f"sharp-frame-output={self.var_sharp_frame_output.get()}",
            "",
            f"video-enabled={self.var_video_enabled.get()}",
            f"video-path={q(self.var_video_path.get())}",
            f"video-mode={self.var_video_mode.get()}",
            f"video-every-n={self.var_video_every_n.get()}",
            f"video-fps={self.var_video_fps.get()}",
            f"video-start-time={self.var_video_start_time.get()}",
            f"video-end-time={self.var_video_end_time.get()}",
            f"video-output={q(self.var_video_output.get())}",
        ])
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        self.log(f"{self.t('saved', path=filepath)}\n")
    
    def load_config(self):
        """Load settings from config.txt."""
        filepath = filedialog.askopenfilename(
            filetypes=[("Config files", "*.txt"), ("All files", "*.*")]
        )
        if not filepath:
            return
        self.load_config_from_path(filepath)
    
    def load_config_file(self):
        """Load config.txt if it exists in the script directory."""
        config_path = self.get_app_base_dir() / "config.txt"
        if config_path.exists():
            self.load_config_from_path(str(config_path))
            self.log(f"{self.t('auto_loaded')}\n")
    
    def load_config_from_path(self, filepath):
        """Load configuration from specified path."""
        config = {}
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        config[key] = value
        except Exception as e:
            self.log(f"{self.t('load_error', error=e)}\n")
            return
        
        # Apply loaded config
        if "images" in config:
            self.var_images.set(config["images"])
        if "xml" in config:
            self.var_xml.set(config["xml"])
        if "ply" in config:
            self.var_ply.set(config["ply"])
        if "output" in config:
            self.var_output.set(config["output"])
        
        if "crop-size" in config:
            self.var_crop_size.set(int(config["crop-size"]))
        if "fov-deg" in config:
            self.var_fov_deg.set(float(config["fov-deg"]))
        if "max-images" in config:
            self.var_max_images.set(int(config["max-images"]))
        if "num-workers" in config:
            self.var_num_workers.set(int(config["num-workers"]))
        if "yaw-offset" in config:
            self.var_yaw_offset.set(float(config["yaw-offset"]))
        
        if "range-images" in config and config["range-images"]:
            parts = config["range-images"].split("-")
            if len(parts) == 2:
                self.var_range_enabled.set(True)
                self.var_range_start.set(int(parts[0]))
                self.var_range_end.set(int(parts[1]))
                self.toggle_range()
        
        if "skip-directions" in config and config["skip-directions"]:
            skip_list = [d.strip().lower() for d in config["skip-directions"].split(",") if d.strip()]
            for d in VALID_DIRECTIONS:
                self.var_skip_directions[d].set(d in skip_list)
        
        # Boolean values
        def parse_bool(val):
            return val.lower() in ("true", "1", "yes")
        
        if "generate-masks" in config:
            self.var_generate_masks.set(parse_bool(config["generate-masks"]))
        if "mask-engine" in config:
            self.var_mask_engine.set(config["mask-engine"])
        if "invert-mask" in config:
            self.var_invert_mask.set(parse_bool(config["invert-mask"]))
        if "yolo-classes" in config:
            self.var_yolo_classes.set(config["yolo-classes"])
        if "yolo-conf" in config:
            self.var_yolo_conf.set(float(config["yolo-conf"]))
        if "yolo-model" in config:
            self.var_yolo_model.set(config["yolo-model"])
        if "sam3-model" in config:
            self.var_sam3_model.set(config["sam3-model"])
        if "sam3-concepts" in config:
            self.var_sam3_concepts.set(config["sam3-concepts"])
        if "sam3-conf" in config:
            self.var_sam3_conf.set(float(config["sam3-conf"]))
        if "sam3-half" in config:
            self.var_sam3_half.set(parse_bool(config["sam3-half"]))
        
        if "mask-overexposure" in config:
            self.var_mask_overexposure.set(parse_bool(config["mask-overexposure"]))
        if "overexposure-threshold" in config:
            self.var_overexposure_threshold.set(int(config["overexposure-threshold"]))
        if "overexposure-dilate" in config:
            self.var_overexposure_dilate.set(int(config["overexposure-dilate"]))
        
        if "flip-vertical" in config:
            self.var_flip_vertical.set(parse_bool(config["flip-vertical"]))
        if "rotate-z180" in config:
            self.var_rotate_z180.set(parse_bool(config["rotate-z180"]))
        if "fix-upside-down" in config:
            self.var_fix_upside_down.set(parse_bool(config["fix-upside-down"]))
        if "lfs-copy-images" in config:
            self.var_lfs_copy_images.set(parse_bool(config["lfs-copy-images"]))
        # Support both new key and legacy key from older config files.
        if "enable-cubemap" in config:
            self.var_enable_cubemap.set(parse_bool(config["enable-cubemap"]))
        elif "lfs-split-cubemap" in config:
            self.var_enable_cubemap.set(parse_bool(config["lfs-split-cubemap"]))
        if "apply-component-transform-for-ply" in config:
            self.var_apply_component.set(parse_bool(config["apply-component-transform-for-ply"]))
        if "quiet" in config:
            self.var_quiet.set(parse_bool(config["quiet"]))
        if "output-mode" in config and config["output-mode"] in ("COLMAP", "LFS", "SHARP_FRAME"):
            self.var_output_mode.set(config["output-mode"])
        if "language" in config and config["language"] in UI_TEXT:
            self.var_language.set(config["language"])

        if "sharp-frame-enabled" in config:
            self.var_sharp_frame_enabled.set(parse_bool(config["sharp-frame-enabled"]))
        if "sharp-frame-method" in config and config["sharp-frame-method"] in ("best-n", "batched", "outlier-removal"):
            self.var_sharp_frame_method.set(config["sharp-frame-method"])
        if "sharp-frame-num-frames" in config:
            self.var_sharp_frame_num_frames.set(int(config["sharp-frame-num-frames"]))
        if "sharp-frame-min-buffer" in config:
            self.var_sharp_frame_min_buffer.set(int(config["sharp-frame-min-buffer"]))
        if "sharp-frame-batch-size" in config:
            self.var_sharp_frame_batch_size.set(int(config["sharp-frame-batch-size"]))
        if "sharp-frame-batch-buffer" in config:
            self.var_sharp_frame_batch_buffer.set(int(config["sharp-frame-batch-buffer"]))
        if "sharp-frame-window-size" in config:
            self.var_sharp_frame_window_size.set(int(config["sharp-frame-window-size"]))
        if "sharp-frame-sensitivity" in config:
            self.var_sharp_frame_sensitivity.set(int(config["sharp-frame-sensitivity"]))
        if "sharp-frame-output" in config:
            self.var_sharp_frame_output.set(config["sharp-frame-output"])

        if "video-enabled" in config:
            self.var_video_enabled.set(parse_bool(config["video-enabled"]))
        if "video-path" in config:
            self.var_video_path.set(config["video-path"])
        if "video-mode" in config and config["video-mode"] in ("every-n", "fps"):
            self.var_video_mode.set(config["video-mode"])
        if "video-every-n" in config:
            self.var_video_every_n.set(int(config["video-every-n"]))
        if "video-fps" in config:
            self.var_video_fps.set(float(config["video-fps"]))
        if "video-start-time" in config:
            self.var_video_start_time.set(float(config["video-start-time"]))
        if "video-end-time" in config:
            self.var_video_end_time.set(float(config["video-end-time"]))
        if "video-output" in config:
            self.var_video_output.set(config["video-output"])

        self.toggle_mode_ui()

        self.log(f"{self.t('loaded', path=filepath)}\n")

        if "language" in config and config["language"] in UI_TEXT:
            self.setup_ui()


def main():
    multiprocessing.freeze_support()

    root = tk.Tk()
    
    # Set DPI awareness on Windows
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    
    # Configure ttk style
    style = ttk.Style()
    if "vista" in style.theme_names():
        style.theme_use("vista")
    elif "clam" in style.theme_names():
        style.theme_use("clam")
    
    app = Metashape360GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
