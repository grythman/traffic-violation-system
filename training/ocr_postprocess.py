"""Turn character-level YOLO detections into a license-plate string.

The ``ed-bw6q4/mongolia-plates`` model detects each character as a separate box
with a class (digit or letter). To read the plate we must:

    1. Run detection on a plate crop.
    2. Sort the detected characters left-to-right (and group rows if needed).
    3. Concatenate their class labels into a string.

This module provides a reusable ``read_plate()`` function plus a CLI so you can
test the trained OCR weights on an image directly:

    python training/ocr_postprocess.py \
        --weights models/mn_plate_ocr_yolo.pt --source plate_crop.jpg

It is intentionally dependency-light (only ultralytics + numpy) so it can be
imported by the API service later if you decide to replace EasyOCR.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple


def _group_into_rows(boxes: List[dict], row_tol_ratio: float = 0.5) -> List[List[dict]]:
    """Group character boxes into rows based on vertical overlap.

    Mongolian plates are usually single-row, but grouping makes the reader
    robust to two-line plates (e.g. some trailer/motorcycle plates).
    """
    if not boxes:
        return []

    # Sort by vertical centre.
    boxes = sorted(boxes, key=lambda b: b["cy"])
    rows: List[List[dict]] = []
    current: List[dict] = [boxes[0]]
    # Use median character height as the row tolerance reference.
    heights = sorted(b["h"] for b in boxes)
    median_h = heights[len(heights) // 2]
    tol = median_h * row_tol_ratio

    for b in boxes[1:]:
        ref_cy = sum(x["cy"] for x in current) / len(current)
        if abs(b["cy"] - ref_cy) <= tol:
            current.append(b)
        else:
            rows.append(current)
            current = [b]
    rows.append(current)
    return rows


def decode_detections(
    xyxy: List[Tuple[float, float, float, float]],
    classes: List[str],
    confs: List[float],
) -> str:
    """Reconstruct the plate string from raw detections.

    Parameters
    ----------
    xyxy:    list of (x1, y1, x2, y2) boxes.
    classes: list of class label strings (e.g. "1", "2", "A").
    confs:   list of confidence scores (unused for ordering, kept for API parity).
    """
    boxes = []
    for (x1, y1, x2, y2), cls in zip(xyxy, classes):
        boxes.append({
            "cx": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
            "h": (y2 - y1),
            "label": str(cls),
        })

    rows = _group_into_rows(boxes)
    # Sort rows top-to-bottom, characters left-to-right within each row.
    rows.sort(key=lambda r: sum(b["cy"] for b in r) / len(r))
    parts: List[str] = []
    for row in rows:
        row.sort(key=lambda b: b["cx"])
        parts.append("".join(b["label"] for b in row))
    return "".join(parts)


def read_plate(model, image, conf: float = 0.25, imgsz: int = 640) -> str:
    """Run the OCR model on an image (path or ndarray) and return the string."""
    results = model.predict(source=image, conf=conf, imgsz=imgsz, verbose=False)
    if not results:
        return ""
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return ""

    names = r.names  # {class_id: label}
    xyxy = [tuple(map(float, b.xyxy[0].tolist())) for b in r.boxes]
    classes = [names[int(b.cls[0])] for b in r.boxes]
    confs = [float(b.conf[0]) for b in r.boxes]
    return decode_detections(xyxy, classes, confs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read a plate string from a crop using the OCR YOLO model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", default="models/mn_plate_ocr_yolo.pt",
                        help="Path to trained character-OCR weights.")
    parser.add_argument("--source", required=True,
                        help="Image file (ideally a tight plate crop).")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size.")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Install ultralytics first: pip install ultralytics")

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(
            f"Weights not found: {weights}\n"
            "Train them first: python training/train_ocr.py"
        )

    model = YOLO(str(weights))
    plate = read_plate(model, args.source, conf=args.conf, imgsz=args.imgsz)
    print(f"Detected plate string: {plate!r}")


if __name__ == "__main__":
    main()
