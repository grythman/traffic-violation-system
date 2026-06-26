"""Download and train the ``ed-bw6q4/mongolia-plates`` dataset locally.

This dataset (https://universe.roboflow.com/ed-bw6q4/mongolia-plates) is a
**character-level** Mongolian license-plate recognition dataset:

    * ~9,304 images
    * 36 classes (digits 0-9 + Cyrillic letters, each character is its own box)
    * Comes with a trained YOLOv11 model (mAP@50 ~99.5%)
    * License: CC BY 4.0

Because every character is annotated as a separate object, a YOLO detector
trained on it acts as an **OCR engine**: it detects each character's box + class,
and reading them left-to-right reconstructs the plate string. This can replace
EasyOCR for Mongolian plates (where EasyOCR struggles with Cyrillic).

This script:
    1. Downloads the dataset from Roboflow (YOLOv8/YOLOv11 format).
    2. Trains an Ultralytics YOLO model on it.
    3. Validates the best checkpoint and prints mAP.
    4. Copies ``best.pt`` into the project ``models/`` folder as
       ``mn_plate_ocr_yolo.pt`` for use as the OCR stage.

Usage
-----
    pip install -r training/requirements-train.txt
    export ROBOFLOW_API_KEY="xxxxxxxx"

    # Default targets ed-bw6q4/mongolia-plates with a YOLOv8 base
    python training/train_ocr.py --epochs 100 --imgsz 640 --model yolov8n.pt

    # To match the original (YOLOv11) base model:
    python training/train_ocr.py --model yolo11n.pt --epochs 120

    # Offline: train from an existing data.yaml
    python training/train_ocr.py --skip-download --data /path/to/data.yaml

After training, the helper ``ocr_postprocess.py`` shows how to turn raw
character detections into a plate string. To wire it into the API as an OCR
engine, set in your .env:
    PLATE_OCR_MODEL_PATH=/app/models/mn_plate_ocr_yolo.pt
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Defaults: the character-level Mongolian plate dataset on Roboflow Universe.
# https://universe.roboflow.com/ed-bw6q4/mongolia-plates
# --------------------------------------------------------------------------- #
DEFAULT_WORKSPACE = "ed-bw6q4"
DEFAULT_PROJECT = "mongolia-plates"
DEFAULT_VERSION = 1  # the only dataset version published at time of writing

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
TARGET_WEIGHTS = MODELS_DIR / "mn_plate_ocr_yolo.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and train the ed-bw6q4/mongolia-plates character-level "
            "OCR dataset locally."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Roboflow download options ---
    parser.add_argument(
        "--api-key",
        default=os.getenv("ROBOFLOW_API_KEY"),
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var).",
    )
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE,
                        help="Roboflow workspace slug.")
    parser.add_argument("--project", default=DEFAULT_PROJECT,
                        help="Roboflow project slug.")
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION,
                        help="Roboflow dataset version number.")
    parser.add_argument("--export-format", default="yolov8",
                        choices=["yolov8", "yolov11", "yolov9", "yolov5"],
                        help="Roboflow export format (YOLO label format is shared).")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip Roboflow download and use --data directly.")
    parser.add_argument("--data", default=None,
                        help="Path to an existing data.yaml (with --skip-download).")

    # --- Training options ---
    parser.add_argument("--model", default="yolov8n.pt",
                        help="Base weights to fine-tune (e.g. yolov8n.pt, yolo11n.pt).")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=320, help="Training image size.")
    parser.add_argument("--batch", type=int, default=64,
                        help="Batch size (use -1 for AutoBatch).")
    parser.add_argument("--device", default=None,
                        help="CUDA device id(s) e.g. '0' or 'cpu'. Auto if omitted.")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early-stopping patience (epochs without improvement).")
    parser.add_argument("--name", default="mn_plate_ocr_yolo",
                        help="Run name under runs/detect/.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    return parser.parse_args()


def download_dataset(args: argparse.Namespace) -> str:
    """Download the dataset from Roboflow and return the path to data.yaml."""
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit(
            "The 'roboflow' package is required to download the dataset.\n"
            "Install it with: pip install roboflow\n"
            "Or pass --skip-download --data /path/to/data.yaml to train offline."
        )

    if not args.api_key:
        sys.exit(
            "No Roboflow API key provided.\n"
            "Get one at https://app.roboflow.com (Settings -> Roboflow API),\n"
            "then either export ROBOFLOW_API_KEY=... or pass --api-key."
        )

    print(f"[1/4] Downloading {args.workspace}/{args.project} "
          f"v{args.version} ({args.export_format}) from Roboflow...")
    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download(args.export_format)

    data_yaml = Path(dataset.location) / "data.yaml"
    if not data_yaml.exists():
        sys.exit(f"Expected data.yaml not found at {data_yaml}")

    print(f"      Dataset downloaded to: {dataset.location}")
    _print_class_summary(data_yaml)
    return str(data_yaml)


def _print_class_summary(data_yaml: Path) -> None:
    """Print the class list so the user can confirm character classes."""
    try:
        import yaml  # PyYAML ships with ultralytics
        with open(data_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        names = cfg.get("names")
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        if names:
            print(f"      Classes ({len(names)}): {names}")
    except Exception:
        pass  # purely informational


def train(args: argparse.Namespace, data_yaml: str) -> Path:
    """Train YOLO and return the path to the best checkpoint."""
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "The 'ultralytics' package is required for training.\n"
            "Install it with: pip install ultralytics"
        )

    print(f"[2/4] Training {args.model} for {args.epochs} epochs "
          f"(character-level OCR)...")
    model = YOLO(args.model)

    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        name=args.name,
        seed=args.seed,
        plots=True,
        cache=True,        # Нэмэх: Дискнээс зураг унших гацалтыг арилгана
        workers=2,         # Нэмэх: Colab CPU-ийн ачааллыг оновчтой болгоно
    )

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else \
        Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"

    if not best.exists():
        sys.exit(f"Training finished but best.pt was not found at {best}")

    print(f"[3/4] Validating best checkpoint: {best}")
    val_model = YOLO(str(best))
    metrics = val_model.val(data=data_yaml, imgsz=args.imgsz, device=args.device)
    try:
        print(f"      mAP50:    {metrics.box.map50:.4f}")
        print(f"      mAP50-95: {metrics.box.map:.4f}")
    except Exception:  # pragma: no cover - metrics layout varies by version
        pass

    return best


def deploy_weights(best: Path) -> None:
    """Copy the trained weights into the project models/ folder."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, TARGET_WEIGHTS)
    print(f"[4/4] Deployed OCR weights -> {TARGET_WEIGHTS}")
    print(
        "\nDone! These weights detect each character as a box.\n"
        "Use training/ocr_postprocess.py to turn detections into a plate string.\n"
        "To wire into the API, set in your .env:\n"
        f"    PLATE_OCR_MODEL_PATH=/app/models/{TARGET_WEIGHTS.name}\n"
    )


def main() -> None:
    args = parse_args()

    if args.skip_download:
        if not args.data:
            sys.exit("--skip-download requires --data /path/to/data.yaml")
        data_yaml = args.data
        print(f"[1/4] Skipping download, using existing data.yaml: {data_yaml}")
        _print_class_summary(Path(data_yaml))
    else:
        data_yaml = download_dataset(args)

    best = train(args, data_yaml)
    deploy_weights(best)


if __name__ == "__main__":
    main()
