"""train.py — Download & train the ``ed-bw6q4/mongolia-plates`` dataset locally.

A self-contained, single-file training script dedicated to the Roboflow dataset:

    https://universe.roboflow.com/ed-bw6q4/mongolia-plates

Dataset facts
-------------
    * ~9,304 images of Mongolian license plates
    * 36 classes (digits 0-9 + Cyrillic letters; each character is its own box)
    * Character-level annotations -> a YOLO model trained here acts as an OCR
      engine (detect each character, read left-to-right to rebuild the plate)
    * Ships with a trained YOLOv11 model (reported mAP@50 ~99.5%)
    * License: CC BY 4.0

What this script does
---------------------
    1. Downloads the dataset from Roboflow in a YOLO label format.
    2. Trains an Ultralytics YOLO model on it.
    3. Validates the best checkpoint and prints mAP.
    4. Saves ``best.pt`` next to this script (and optionally into the project
       ``models/`` folder) as ``mn_plate_ocr_yolo.pt``.

Quick start
-----------
    pip install ultralytics roboflow
    export ROBOFLOW_API_KEY="xxxxxxxx"     # https://app.roboflow.com -> Settings

    python train.py                        # sensible defaults (100 epochs)
    python train.py --epochs 150 --model yolo11n.pt --device 0
    python train.py --skip-download --data ./mongolia-plates-1/data.yaml
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Hard-coded target dataset (override via CLI flags if needed).
# --------------------------------------------------------------------------- #
WORKSPACE = "ed-bw6q4"
PROJECT = "mongolia-plates"
VERSION = 1

HERE = Path(__file__).resolve().parent
# Try to locate the project models/ folder (…/training/mongolia_plates -> root).
PROJECT_ROOT = HERE.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_NAME = "mn_plate_ocr_yolo.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download & train ed-bw6q4/mongolia-plates locally.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Roboflow
    p.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY"),
                   help="Roboflow API key (or set ROBOFLOW_API_KEY).")
    p.add_argument("--workspace", default=WORKSPACE, help="Roboflow workspace.")
    p.add_argument("--project", default=PROJECT, help="Roboflow project.")
    p.add_argument("--version", type=int, default=VERSION, help="Dataset version.")
    p.add_argument("--export-format", default="yolov8",
                   choices=["yolov8", "yolov11", "yolov9", "yolov5"],
                   help="Roboflow export format (shared YOLO label layout).")
    p.add_argument("--skip-download", action="store_true",
                   help="Use an existing --data data.yaml instead of downloading.")
    p.add_argument("--data", default=None, help="Path to existing data.yaml.")
    # Training
    p.add_argument("--model", default="yolov8n.pt",
                   help="Base weights (e.g. yolov8n.pt, yolo11n.pt).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16, help="Use -1 for AutoBatch.")
    p.add_argument("--device", default=None, help="'0', '0,1' or 'cpu'. Auto if omitted.")
    p.add_argument("--patience", type=int, default=30, help="Early-stop patience.")
    p.add_argument("--name", default="mn_plate_ocr_yolo", help="Run name.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-deploy", action="store_true",
                   help="Do not copy best.pt into the project models/ folder.")
    return p.parse_args()


def download_dataset(a: argparse.Namespace) -> str:
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("Install roboflow first: pip install roboflow\n"
                 "(or use --skip-download --data /path/to/data.yaml)")
    if not a.api_key:
        sys.exit("Missing Roboflow API key. Set ROBOFLOW_API_KEY or pass --api-key.\n"
                 "Get it at https://app.roboflow.com -> Settings -> Roboflow API.")

    print(f"[1/4] Downloading {a.workspace}/{a.project} v{a.version} "
          f"({a.export_format}) ...")
    rf = Roboflow(api_key=a.api_key)
    project = rf.workspace(a.workspace).project(a.project)
    dataset = project.version(a.version).download(a.export_format)

    data_yaml = Path(dataset.location) / "data.yaml"
    if not data_yaml.exists():
        sys.exit(f"data.yaml not found at {data_yaml}")
    print(f"      Downloaded to: {dataset.location}")
    _summarize_classes(data_yaml)
    return str(data_yaml)


def _summarize_classes(data_yaml: Path) -> None:
    try:
        import yaml
        cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        names = cfg.get("names")
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        if names:
            print(f"      Classes ({len(names)}): {names}")
    except Exception:
        pass


def train(a: argparse.Namespace, data_yaml: str) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("Install ultralytics first: pip install ultralytics")

    # Fix: Roboflow datasets often have inconsistent relative paths in data.yaml.
    # We normalize everything to absolute paths with forward slashes for cross-platform compatibility.
    try:
        import yaml
        yaml_path = Path(data_yaml).resolve()
        dataset_root = yaml_path.parent
        with open(yaml_path, "r", encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f)
        
        # 1. Set the base path to absolute (using forward slashes for Windows compatibility in YOLO)
        data_cfg["path"] = str(dataset_root).replace("\\", "/")
        
        # 2. Clean up train/val/test paths and handle missing folders
        existing_subdirs = [d.name for d in dataset_root.iterdir() if d.is_dir()]
        print(f"      Found subdirectories: {existing_subdirs}")
        
        # Priority mapping: try to find these folders in order
        for key in ["train", "val", "test"]:
            if key in data_cfg:
                p = str(data_cfg[key]).replace("\\", "/")
                while p.startswith("../") or p.startswith("./"):
                    p = p.split("/", 1)[1] if "/" in p else ""
                
                # Check if the defined path exists
                if not (dataset_root / p).exists():
                    # Fallback logic:
                    # 1. Try to find any directory that matches the key (e.g. 'valid' instead of 'valid/images')
                    # 2. If it's 'val' or 'test' and missing, use 'train' as fallback to prevent crash
                    potential_match = next((d for d in existing_subdirs if key in d or d.startswith(key[:3])), None)
                    if potential_match:
                        p = potential_match
                    elif key in ["val", "test"] and "train" in existing_subdirs:
                        print(f"      Warning: {key} folder not found, using 'train' as fallback.")
                        p = "train"
                    elif "train" not in existing_subdirs and existing_subdirs:
                        # If even 'train' is missing, use the first available directory
                        p = existing_subdirs[0]
                
                data_cfg[key] = p
        
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data_cfg, f)
        
        print(f"      Normalized data.yaml for Windows/Linux compatibility:")
        print(f"        path:  {data_cfg['path']}")
        print(f"        train: {data_cfg.get('train')}")
        print(f"        val:   {data_cfg.get('val')}")
    except Exception as e:
        print(f"      Warning: Could not update data.yaml paths: {e}")

    print(f"[2/4] Training {a.model} for {a.epochs} epochs ...")
    model = YOLO(a.model)
    results = model.train(
        data=str(yaml_path), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        device=a.device, patience=a.patience, name=a.name, seed=a.seed, plots=True,
    )
    save_dir = Path(getattr(results, "save_dir", model.trainer.save_dir))
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        sys.exit(f"best.pt not found at {best}")

    print(f"[3/4] Validating: {best}")
    metrics = YOLO(str(best)).val(data=data_yaml, imgsz=a.imgsz, device=a.device)
    try:
        print(f"      mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")
    except Exception:
        pass
    return best


def deploy(best: Path, no_deploy: bool) -> None:
    # Always keep a copy next to this script.
    local_copy = HERE / OUTPUT_NAME
    shutil.copy2(best, local_copy)
    print(f"[4/4] Saved weights -> {local_copy}")

    if not no_deploy and MODELS_DIR.exists() or (not no_deploy and MODELS_DIR.parent.exists()):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        target = MODELS_DIR / OUTPUT_NAME
        shutil.copy2(best, target)
        print(f"      Deployed to project models/ -> {target}")
        print("\nTo use as the OCR stage, set in your .env:\n"
              f"    PLATE_OCR_MODEL_PATH=/app/models/{OUTPUT_NAME}\n")


def main() -> None:
    a = parse_args()
    if a.skip_download:
        if not a.data:
            sys.exit("--skip-download requires --data /path/to/data.yaml")
        print(f"[1/4] Using existing data.yaml: {a.data}")
        _summarize_classes(Path(a.data))
        data_yaml = a.data
    else:
        data_yaml = download_dataset(a)

    best = train(a, data_yaml)
    deploy(best, a.no_deploy)
    print("\nDone. These weights detect each character; reconstruct the plate "
          "string with ../ocr_postprocess.py (read_plate()).")


if __name__ == "__main__":
    main()
