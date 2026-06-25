"""Quick sanity-check for a trained plate detector.

Runs the trained weights on a single image (or a folder of images) and prints
the detected plate bounding boxes + confidences, saving an annotated copy.

Usage
-----
    python training/predict.py --weights models/mn_plate_yolov8.pt --source car.jpg
    python training/predict.py --weights models/mn_plate_yolov8.pt --source ./samples/
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained YOLOv8 plate detector on an image or folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", default="models/mn_plate_yolov8.pt",
                        help="Path to trained .pt weights.")
    parser.add_argument("--source", required=True,
                        help="Image file or directory of images.")
    parser.add_argument("--conf", type=float, default=0.30,
                        help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size.")
    parser.add_argument("--device", default=None, help="CUDA id or 'cpu'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Install ultralytics first: pip install ultralytics")

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(
            f"Weights not found: {weights}\n"
            "Train them first with: python training/train.py"
        )

    model = YOLO(str(weights))
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,  # writes annotated images to runs/detect/predict/
    )

    total = 0
    for r in results:
        n = len(r.boxes) if r.boxes is not None else 0
        total += n
        print(f"\n{Path(r.path).name}: {n} plate(s) detected")
        if r.boxes is not None:
            for i, box in enumerate(r.boxes):
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                xyxy_fmt = [round(v, 1) for v in xyxy]
                print(f"  #{i + 1}  conf={conf:.3f}  bbox(xyxy)={xyxy_fmt}")

    print(f"\nTotal plates detected: {total}")
    print("Annotated images saved under: runs/detect/predict/")


if __name__ == "__main__":
    main()
