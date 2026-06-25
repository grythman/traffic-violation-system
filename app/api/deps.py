"""FastAPI dependency providers.

These keep the endpoint signatures clean and make the services easy to mock in
unit tests.
"""
from app.services.analysis_service import AnalysisService
from app.services.detection_service import get_detection_service
from app.services.ocr_service import get_ocr_service
from app.services.plate_detection_service import get_plate_detection_service
from app.services.rule_engine import get_rule_engine


def get_analysis_service() -> AnalysisService:
    """Construct the AnalysisService with its collaborating singletons."""
    return AnalysisService(
        detection_service=get_detection_service(),
        ocr_service=get_ocr_service(),
        rule_engine=get_rule_engine(),
        plate_detection_service=get_plate_detection_service(),
    )
