"""Vehicle detection service powered by YOLOv8 (Ultralytics).

The model is loaded lazily and cached as a singleton so that the (expensive)
weight loading happens only once per process. This keeps the service modular
and testable: callers depend only on ``detect_vehicles``.
"""
from dataclasses import dataclass
from threading import Lock

import numpy as np

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# COCO class ids that represent vehicles in the default YOLOv8 model.
# 2: car, 3: motorcycle, 5: bus, 7: truck
_VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class VehicleDetection:
    """A single detected vehicle bounding box."""

    vehicle_type: str
    confidence: float
    bbox: list[int]  # [x1, y1, x2, y2]


class DetectionService:
    """Thread-safe singleton wrapper around a YOLOv8 model."""

    _instance: "DetectionService | None" = None
    _lock: Lock = Lock()

    def __init__(self) -> None:
        self._model = None
        self._model_lock = Lock()

    @classmethod
    def get_instance(cls) -> "DetectionService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _get_model(self):
        """Lazily import and load the Ultralytics model (thread-safe)."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    # Imported here so the heavy dependency is only required
                    # when detection is actually used.
                    from ultralytics import YOLO

                    logger.info(
                        "Loading YOLOv8 model from %s", settings.YOLO_MODEL_PATH
                    )
                    self._model = YOLO(settings.YOLO_MODEL_PATH)
        return self._model

    def detect_vehicles(self, image: np.ndarray) -> list[VehicleDetection]:
        """Run inference and return vehicle detections above the threshold."""
        model = self._get_model()
        device = 0 if settings.USE_GPU else "cpu"
        results = model.predict(
            source=image,
            conf=settings.YOLO_CONFIDENCE_THRESHOLD,
            device=device,
            verbose=False,
        )

        detections: list[VehicleDetection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in _VEHICLE_CLASS_IDS:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                detections.append(
                    VehicleDetection(
                        vehicle_type=_VEHICLE_CLASS_IDS[cls_id],
                        confidence=round(conf, 4),
                        bbox=[x1, y1, x2, y2],
                    )
                )
        logger.info("Detected %d vehicle(s).", len(detections))
        return detections


def get_detection_service() -> DetectionService:
    """FastAPI-friendly accessor for the detection singleton."""
    return DetectionService.get_instance()
