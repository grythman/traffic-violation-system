"""License plate recognition service.

Two-stage OCR:

  1. **Custom YOLO character detector** (preferred for Mongolian plates).
     If a custom-trained model (``mn_plate_ocr_yolo.pt``) is available, it
     detects each plate CHARACTER as its own box. Reading the characters
     left-to-right (and grouping rows for two-line plates) reconstructs the
     plate string. This is far more accurate on Cyrillic than EasyOCR.

  2. **EasyOCR fallback.** If the custom model is not present (or produces no
     usable result), recognition falls back to the EasyOCR reader.

Both engines are loaded lazily and cached as singletons because initialising
them (and downloading weights) is slow.
"""
from dataclasses import dataclass
from threading import Lock

import numpy as np

from app.core.config import settings
from app.core.logging_config import get_logger
from app.utils.plate_utils import (
    is_plausible_plate,
    is_valid_mn_plate,
    normalize_mn_plate,
    normalize_plate,
)

logger = get_logger(__name__)


@dataclass
class PlateResult:
    """Recognised plate text and its confidence."""

    text: str
    confidence: float
    # Which engine produced this result: "yolo_ocr" or "easyocr".
    engine: str = "easyocr"


def _resolve_ocr_model_path() -> str | None:
    """Resolve the custom OCR model path.

    Order: explicit ``PLATE_OCR_MODEL_PATH`` -> auto-detected
    ``<MODELS_DIR>/mn_plate_ocr_yolo.pt`` -> None.
    """
    import os

    explicit = settings.PLATE_OCR_MODEL_PATH
    if explicit and os.path.exists(explicit):
        return explicit

    auto = os.path.join(settings.MODELS_DIR, "mn_plate_ocr_yolo.pt")
    if os.path.exists(auto):
        return auto

    if explicit:
        logger.warning(
            "PLATE_OCR_MODEL_PATH=%s set but file not found; "
            "and no model at %s. Falling back to EasyOCR.",
            explicit,
            auto,
        )
    return None


def _group_into_rows(boxes: list[dict], row_tol_ratio: float = 0.5) -> list[list[dict]]:
    """Group character boxes into rows based on vertical proximity.

    Mongolian plates are usually single-row, but grouping makes the reader
    robust to two-line plates (e.g. some trailer/motorcycle plates).
    """
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b["cy"])
    heights = sorted(b["h"] for b in boxes)
    median_h = heights[len(heights) // 2] or 1.0
    tol = median_h * row_tol_ratio

    rows: list[list[dict]] = []
    current: list[dict] = [boxes[0]]
    for b in boxes[1:]:
        ref_cy = sum(x["cy"] for x in current) / len(current)
        if abs(b["cy"] - ref_cy) <= tol:
            current.append(b)
        else:
            rows.append(current)
            current = [b]
    rows.append(current)
    return rows


class OCRService:
    """Thread-safe singleton supporting custom YOLO-OCR with EasyOCR fallback."""

    _instance: "OCRService | None" = None
    _lock: Lock = Lock()

    def __init__(self) -> None:
        self._reader = None
        self._reader_lock = Lock()
        self._yolo_ocr = None
        self._yolo_ocr_lock = Lock()
        self._yolo_ocr_path = _resolve_ocr_model_path()
        if self._yolo_ocr_path:
            logger.info(
                "Custom YOLO OCR model detected at %s; it will be used as the "
                "primary OCR engine (EasyOCR remains the fallback).",
                self._yolo_ocr_path,
            )
        else:
            logger.info("No custom YOLO OCR model found; using EasyOCR only.")

    @classmethod
    def get_instance(cls) -> "OCRService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # -- lazy loaders -------------------------------------------------------
    def _get_reader(self):
        if self._reader is None:
            with self._reader_lock:
                if self._reader is None:
                    import easyocr

                    logger.info(
                        "Initialising EasyOCR reader (langs=%s, gpu=%s)",
                        settings.OCR_LANGUAGES,
                        settings.USE_GPU,
                    )
                    self._reader = easyocr.Reader(
                        settings.OCR_LANGUAGES, gpu=settings.USE_GPU
                    )
        return self._reader

    def _get_yolo_ocr(self):
        if self._yolo_ocr is None:
            with self._yolo_ocr_lock:
                if self._yolo_ocr is None:
                    from ultralytics import YOLO

                    logger.info(
                        "Loading custom YOLO OCR model from %s", self._yolo_ocr_path
                    )
                    self._yolo_ocr = YOLO(self._yolo_ocr_path)
        return self._yolo_ocr

    @property
    def custom_ocr_enabled(self) -> bool:
        return self._yolo_ocr_path is not None

    # -- public API ---------------------------------------------------------
    def read_plate(self, image_region: np.ndarray) -> PlateResult | None:
        """Extract the most plausible plate string from an image region.

        Tries the custom YOLO character detector first (if available). If it
        yields a result that is NOT a valid Mongolian plate, EasyOCR is tried
        as a fallback and the better result is returned.
        """
        yolo_result: PlateResult | None = None
        if self.custom_ocr_enabled:
            try:
                yolo_result = self._read_with_yolo(image_region)
            except Exception:  # noqa: BLE001 - never let OCR crash the pipeline
                logger.exception(
                    "Custom YOLO OCR failed; falling back to EasyOCR."
                )
                yolo_result = None

            # A valid Mongolian plate from the custom model is trusted directly.
            if yolo_result and is_valid_mn_plate(yolo_result.text):
                logger.info(
                    "OCR (yolo_ocr) plate: %s (conf=%.3f)",
                    yolo_result.text,
                    yolo_result.confidence,
                )
                return yolo_result

        easy_result = self._read_with_easyocr(image_region)

        # Prefer a result that validates as a real Mongolian plate; otherwise
        # prefer the higher-confidence candidate.
        candidates = [r for r in (yolo_result, easy_result) if r is not None]
        if not candidates:
            logger.info("No plausible plate text recognised.")
            return None

        valid = [r for r in candidates if is_valid_mn_plate(r.text)]
        best = max(valid or candidates, key=lambda r: r.confidence)
        logger.info(
            "OCR (%s) plate: %s (conf=%.3f)", best.engine, best.text, best.confidence
        )
        return best

    # -- engine implementations --------------------------------------------
    def _read_with_yolo(self, image_region: np.ndarray) -> PlateResult | None:
        """Detect characters with the custom model and reconstruct the plate."""
        model = self._get_yolo_ocr()
        device = 0 if settings.USE_GPU else "cpu"
        results = model.predict(
            source=image_region,
            conf=settings.PLATE_OCR_CONFIDENCE_THRESHOLD,
            imgsz=settings.PLATE_OCR_IMGSZ,
            device=device,
            verbose=False,
        )
        if not results:
            return None
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None

        names = result.names  # {class_id: label}
        boxes: list[dict] = []
        confs: list[float] = []
        for box in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            label = str(names[int(box.cls[0])])
            conf = float(box.conf[0])
            boxes.append(
                {
                    "cx": (x1 + x2) / 2.0,
                    "cy": (y1 + y2) / 2.0,
                    "h": (y2 - y1),
                    "label": label,
                }
            )
            confs.append(conf)

        if not boxes:
            return None

        # Reconstruct: rows top-to-bottom, characters left-to-right within rows.
        rows = _group_into_rows(boxes)
        rows.sort(key=lambda r: sum(b["cy"] for b in r) / len(r))
        parts: list[str] = []
        for row in rows:
            row.sort(key=lambda b: b["cx"])
            parts.append("".join(b["label"] for b in row))
        raw_text = "".join(parts)

        text = normalize_mn_plate(raw_text)
        if not text:
            return None
        avg_conf = float(np.mean(confs)) if confs else 0.0
        return PlateResult(text=text, confidence=avg_conf, engine="yolo_ocr")

    def _read_with_easyocr(self, image_region: np.ndarray) -> PlateResult | None:
        reader = self._get_reader()
        # detail=1 returns (bbox, text, confidence) tuples. An optional
        # allow-list constrains recognition to expected characters (helps a
        # lot for Mongolian Cyrillic + digit plates).
        ocr_kwargs = {"detail": 1}
        if settings.OCR_ALLOWLIST:
            ocr_kwargs["allowlist"] = settings.OCR_ALLOWLIST
        raw_results = reader.readtext(image_region, **ocr_kwargs)

        best: PlateResult | None = None
        for _bbox, text, conf in raw_results:
            if not is_plausible_plate(text):
                continue
            # Normalise with Mongolian-aware transliteration so Latin homoglyphs
            # collapse onto Cyrillic before downstream validation.
            normalized = normalize_mn_plate(text)
            if not normalized:
                normalized = normalize_plate(text)
            candidate = PlateResult(
                text=normalized, confidence=float(conf), engine="easyocr"
            )
            if best is None or candidate.confidence > best.confidence:
                best = candidate
        return best


def get_ocr_service() -> OCRService:
    """FastAPI-friendly accessor for the OCR singleton."""
    return OCRService.get_instance()
