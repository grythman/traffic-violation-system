"""/analyze endpoint: LPR + rule evaluation pipeline."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_analysis_service
from app.core.logging_config import get_logger
from app.db.session import get_db
from app.schemas.analysis import AnalyzeRequest, AnalyzeResponse
from app.services.analysis_service import AnalysisService

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect vehicles, read plates, and flag potential violations",
)
def analyze(
    payload: AnalyzeRequest,
    db: Session = Depends(get_db),
    service: AnalysisService = Depends(get_analysis_service),
) -> AnalyzeResponse:
    """Analyse an image (base64 or URL) plus metadata.

    The pipeline detects vehicles with YOLOv8, extracts the license plate with
    EasyOCR, and evaluates the rule engine. If a violation is detected, a record
    is inserted with status ``pending_human_review`` — the system never issues a
    fine automatically.
    """
    try:
        return service.analyze(db=db, request=payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surface unexpected errors cleanly
        logger.exception("Unexpected error during analysis")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error while analysing the image.",
        ) from exc
