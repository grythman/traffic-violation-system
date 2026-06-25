"""License-plate localisation service (optional stage-2 detector).

This wraps a CUSTOM YOLOv8 model trained to find the license-plate region
inside a vehicle crop. For Mongolian plates you typically train (or download)
such a model and point ``settings.PLATE_MODEL_PATH`` at the resulting ``.pt``.

If no plate model is configured, this service is a no-op and the pipeline runs
OCR directly on the full vehicle crop.
"""
from dataclasses import dataclass
from threading import Lock

import numpy as np

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PlateBox:
    confidence: float
    bbox: list[int]  # [x1, y1, x2, y2] relative to the input image/crop


class PlateDetectionService:
    """Thread-safe singleton wrapper around a custom plate-detection model."""

    _instance: "PlateDetectionService | None" = None
    _lock: Lock = Lock()

    def __init__(self) -> None:
        self._model = None
        self._model_lock = Lock()
        self._enabled = bool(settings.PLATE_MODEL_PATH)

    @classmethod
    def get_instance(cls) -> "PlateDetectionService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _get_model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    from ultralytics import YOLO

                    logger.info(
                        "Loading custom plate-detection model from %s",
                        settings.PLATE_MODEL_PATH,
                    )
                    self._model = YOLO(settings.PLATE_MODEL_PATH)
        return self._model

    def detect_plate_regions(self, image: np.ndarray) -> list[PlateBox]:
        """Return candidate plate bounding boxes, highest confidence first."""
        if not self._enabled:
            return []
        model = self._get_model()
        device = 0 if settings.USE_GPU else "cpu"
        results = model.predict(
            source=image,
            conf=settings.PLATE_CONFIDENCE_THRESHOLD,
            device=device,
            verbose=False,
        )
        boxes: list[PlateBox] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                boxes.append(PlateBox(confidence=round(conf, 4), bbox=[x1, y1, x2, y2]))
        boxes.sort(key=lambda b: b.confidence, reverse=True)
        return boxes


def get_plate_detection_service() -> PlateDetectionService:
    return PlateDetectionService.get_instance()
