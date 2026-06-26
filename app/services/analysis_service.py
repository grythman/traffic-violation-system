"""Analysis pipeline orchestration.

Ties together image loading, YOLOv8 detection, EasyOCR plate recognition, the
rule engine, and persistence. The pipeline NEVER issues fines; it only records
pending violations for human review.
"""
import numpy as np
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.schemas.analysis import (
    AnalyzeRequest,
    AnalyzeResponse,
    DetectedVehicle,
)
from app.services import crud
from app.services.detection_service import DetectionService, VehicleDetection
from app.services.ocr_service import OCRService
from app.services.plate_detection_service import PlateDetectionService
from app.services.rule_engine import RuleEngine
from app.utils.image_loader import (
    ImageLoadError,
    load_image_from_base64,
    load_image_from_url,
)
from app.utils.plate_utils import is_valid_mn_plate

logger = get_logger(__name__)


class AnalysisService:
    """High-level service that runs the full LPR + rule pipeline."""

    def __init__(
        self,
        detection_service: DetectionService,
        ocr_service: OCRService,
        rule_engine: RuleEngine,
        plate_detection_service: PlateDetectionService | None = None,
    ) -> None:
        self._detector = detection_service
        self._ocr = ocr_service
        self._rules = rule_engine
        # Optional stage-2 plate localiser (custom weights, e.g. Mongolian).
        self._plate_detector = plate_detection_service

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _load_image(request: AnalyzeRequest) -> np.ndarray:
        if request.image_base64:
            return load_image_from_base64(request.image_base64)
        return load_image_from_url(request.image_url)  # type: ignore[arg-type]

    @staticmethod
    def _crop(image: np.ndarray, det: VehicleDetection) -> np.ndarray:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = det.bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return image
        return image[y1:y2, x1:x2]

    @staticmethod
    def _crop_bbox(image: np.ndarray, bbox: list[int]) -> np.ndarray:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return image
        return image[y1:y2, x1:x2]

    def _read_plate_for_region(self, vehicle_crop: np.ndarray):
        """Run OCR on a vehicle crop.

        If a custom plate detector is enabled, first localise the plate region
        and OCR only that tight crop (much more accurate for Mongolian plates);
        otherwise OCR the whole vehicle crop.
        """
        if self._plate_detector is not None and self._plate_detector.enabled:
            plate_boxes = self._plate_detector.detect_plate_regions(vehicle_crop)
            if plate_boxes:
                plate_crop = self._crop_bbox(vehicle_crop, plate_boxes[0].bbox)
                return self._ocr.read_plate(plate_crop)
        return self._ocr.read_plate(vehicle_crop)

    # -- main entry ---------------------------------------------------------
    def analyze(self, db: Session, request: AnalyzeRequest) -> AnalyzeResponse:
        """Run detection, OCR, rule evaluation, and persistence."""
        try:
            image = self._load_image(request)
        except ImageLoadError as exc:
            raise ValueError(str(exc)) from exc

        detections = self._detector.detect_vehicles(image)

        detected_vehicles: list[DetectedVehicle] = []
        best_plate: str | None = None
        best_plate_conf: float | None = None
        best_plate_valid = False

        def _consider(plate_text: str | None, plate_conf: float | None) -> None:
            """Track the "primary" plate, preferring format-valid Mongolian
            plates over higher-confidence but malformed reads."""
            nonlocal best_plate, best_plate_conf, best_plate_valid
            if not plate_text:
                return
            valid = is_valid_mn_plate(plate_text)
            conf = plate_conf or 0.0
            # Selection priority: (is_valid, confidence).
            better = (valid, conf) > (best_plate_valid, best_plate_conf or 0.0)
            if best_plate is None or better:
                best_plate = plate_text
                best_plate_conf = plate_conf
                best_plate_valid = valid

        for det in detections:
            region = self._crop(image, det)
            plate_result = self._read_plate_for_region(region)
            plate_text = plate_result.text if plate_result else None
            plate_conf = plate_result.confidence if plate_result else None

            detected_vehicles.append(
                DetectedVehicle(
                    vehicle_type=det.vehicle_type,
                    detection_confidence=det.confidence,
                    bounding_box=det.bbox,
                    license_plate=plate_text,
                    plate_confidence=plate_conf,
                )
            )

            _consider(plate_text, plate_conf)

        # If no vehicle was detected, still attempt OCR on the full frame so a
        # plate-only crop image can be analysed.
        if not detections:
            plate_result = self._read_plate_for_region(image)
            if plate_result:
                _consider(plate_result.text, plate_result.confidence)

        # Evaluate the rule engine against the supplied metadata.
        rule_result = self._rules.evaluate(request.metadata)

        if rule_result is None:
            return AnalyzeResponse(
                detected_vehicles=detected_vehicles,
                primary_license_plate=best_plate,
                violation_detected=False,
                violation=None,
                message="No violation detected. Vehicle appears compliant.",
            )

        # A violation was flagged. We MUST have a plate to attribute it to.
        if not best_plate:
            return AnalyzeResponse(
                detected_vehicles=detected_vehicles,
                primary_license_plate=None,
                violation_detected=True,
                violation=None,
                message=(
                    "Violation conditions met, but no license plate could be "
                    "recognised; no record was created."
                ),
            )

        vehicle = crud.get_or_create_vehicle(
            db,
            license_plate=best_plate,
            vehicle_type=(detected_vehicles[0].vehicle_type if detected_vehicles else None),
            plate_confidence=best_plate_conf,
        )

        violation = crud.create_pending_violation(
            db,
            vehicle=vehicle,
            violation_type=rule_result.violation_type,
            description=rule_result.description,
            detected_speed_kmh=rule_result.detected_speed_kmh,
            speed_limit_kmh=rule_result.speed_limit_kmh,
            location=request.metadata.location,
            evidence_image_ref=request.image_url,
        )

        from app.schemas.violation import ViolationRead

        return AnalyzeResponse(
            detected_vehicles=detected_vehicles,
            primary_license_plate=best_plate,
            violation_detected=True,
            violation=ViolationRead.model_validate(violation),
            message=(
                "Violation detected and recorded with status "
                "'pending_human_review'. A human operator must review it; "
                "the system will not issue a fine automatically."
            ),
        )
