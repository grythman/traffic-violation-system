"""License plate recognition service powered by EasyOCR.

Like the detection service, the EasyOCR reader is loaded lazily and cached as
a singleton because initialising the reader (and downloading weights) is slow.
"""
from dataclasses import dataclass
from threading import Lock

import numpy as np

from app.core.config import settings
from app.core.logging_config import get_logger
from app.utils.plate_utils import is_plausible_plate, normalize_plate

logger = get_logger(__name__)


@dataclass
class PlateResult:
    """Recognised plate text and its confidence."""

    text: str
    confidence: float


class OCRService:
    """Thread-safe singleton wrapper around an EasyOCR reader."""

    _instance: "OCRService | None" = None
    _lock: Lock = Lock()

    def __init__(self) -> None:
        self._reader = None
        self._reader_lock = Lock()

    @classmethod
    def get_instance(cls) -> "OCRService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

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

    def read_plate(self, image_region: np.ndarray) -> PlateResult | None:
        """Extract the most plausible plate string from an image region.

        Returns the highest-confidence plausible candidate, or ``None`` if no
        plate-like text is found.
        """
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
            candidate = PlateResult(text=normalize_plate(text), confidence=float(conf))
            if best is None or candidate.confidence > best.confidence:
                best = candidate

        if best:
            logger.info(
                "OCR plate candidate: %s (conf=%.3f)", best.text, best.confidence
            )
        else:
            logger.info("No plausible plate text recognised.")
        return best


def get_ocr_service() -> OCRService:
    """FastAPI-friendly accessor for the OCR singleton."""
    return OCRService.get_instance()
