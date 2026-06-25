"""Train a custom YOLOv8 license-plate detector on a Roboflow dataset.

This script automates the full workflow:

    1. Download a license-plate detection dataset from Roboflow (YOLOv8 format).
    2. Train an Ultralytics YOLOv8 model on it.
    3. Validate the best checkpoint.
    4. Copy the resulting ``best.pt`` into the project ``models/`` folder as
       ``mn_plate_yolov8.pt`` so it can be plugged straight into the API via
       ``PLATE_MODEL_PATH``.

Designed for the Mongolian plate dataset
``computer-vision-m1xzb/mongolian-plate`` but works with any Roboflow
license-plate detection project (just change the env vars / CLI flags).

Usage
-----
    # 1. Install deps
    pip install -r training/requirements-train.txt

    # 2. Provide your Roboflow API key (https://app.roboflow.com -> Settings)
    export ROBOFLOW_API_KEY="xxxxxxxx"

    # 3. Run (defaults target the Mongolian plate dataset)
    python training/train.py --epochs 100 --imgsz 640 --model yolov8n.pt

    # Offline mode: if you already have a data.yaml, skip the download
    python training/train.py --data /path/to/data.yaml --skip-download

After training, set in your .env:
    PLATE_MODEL_PATH=/app/models/mn_plate_yolov8.pt
    OCR_LANGUAGES=mn,en
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Defaults: the Mongolian plate dataset found on Roboflow Universe.
# https://universe.roboflow.com/computer-vision-m1xzb/mongolian-plate
# --------------------------------------------------------------------------- #
DEFAULT_WORKSPACE = "computer-vision-m1xzb"
DEFAULT_PROJECT = "mongolian-plate"
DEFAULT_VERSION = 2  # latest version visible on Universe at time of writing

# Where the trained weights should end up so the API can use them.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
TARGET_WEIGHTS = MODELS_DIR / "mn_plate_yolov8.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Roboflow dataset and train a YOLOv8 plate detector.",
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
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip Roboflow download and use --data directly.")
    parser.add_argument("--data", default=None,
                        help="Path to an existing data.yaml (used with --skip-download).")

    # --- Training options ---
    parser.add_argument("--model", default="yolov8n.pt",
                        help="Base YOLOv8 weights to fine-tune (n/s/m/l/x).")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (use -1 for AutoBatch).")
    parser.add_argument("--device", default=None,
                        help="CUDA device id(s) e.g. '0' or 'cpu'. Auto if omitted.")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early-stopping patience (epochs without improvement).")
    parser.add_argument("--name", default="mn_plate_yolov8",
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
          f"v{args.version} from Roboflow...")
    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download("yolov8")

    data_yaml = Path(dataset.location) / "data.yaml"
    if not data_yaml.exists():
        sys.exit(f"Expected data.yaml not found at {data_yaml}")

    print(f"      Dataset downloaded to: {dataset.location}")
    return str(data_yaml)


def train(args: argparse.Namespace, data_yaml: str) -> Path:
    """Train YOLOv8 and return the path to the best checkpoint."""
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "The 'ultralytics' package is required for training.\n"
            "Install it with: pip install ultralytics"
        )

    print(f"[2/4] Training YOLOv8 ({args.model}) for {args.epochs} epochs...")
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
    )

    # Ultralytics exposes the run directory via the trainer.
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
    except Exception:  # pragma: no cover - metrics layout can vary by version
        pass

    return best


def deploy_weights(best: Path) -> None:
    """Copy the trained weights into the project models/ folder."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, TARGET_WEIGHTS)
    print(f"[4/4] Deployed weights -> {TARGET_WEIGHTS}")
    print(
        "\nDone! To use them, set in your .env:\n"
        f"    PLATE_MODEL_PATH=/app/models/{TARGET_WEIGHTS.name}\n"
        "    OCR_LANGUAGES=mn,en\n"
        "Then run: docker-compose up --build\n"
    )


def main() -> None:
    args = parse_args()

    if args.skip_download:
        if not args.data:
            sys.exit("--skip-download requires --data /path/to/data.yaml")
        data_yaml = args.data
        print(f"[1/4] Skipping download, using existing data.yaml: {data_yaml}")
    else:
        data_yaml = download_dataset(args)

    best = train(args, data_yaml)
    deploy_weights(best)


if __name__ == "__main__":
    main()
