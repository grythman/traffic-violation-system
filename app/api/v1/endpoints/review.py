"""/review endpoints: the human-in-the-loop control plane."""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.enums import ViolationStatus
from app.core.logging_config import get_logger
from app.db.session import get_db
from app.schemas.violation import ViolationRead, ViolationReviewRequest
from app.services import crud

logger = get_logger(__name__)
router = APIRouter()

# Only these decisions are valid for a human review.
_ALLOWED_DECISIONS = {ViolationStatus.APPROVED, ViolationStatus.REJECTED}


@router.get(
    "/review",
    response_model=list[ViolationRead],
    summary="List violations (optionally filter by status)",
)
def list_violations(
    db: Session = Depends(get_db),
    status_filter: ViolationStatus | None = Query(
        default=None, alias="status", description="Filter by violation status."
    ),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[ViolationRead]:
    """Return violations for an operator's review queue."""
    violations = crud.list_violations(
        db, status=status_filter, skip=skip, limit=limit
    )
    return [ViolationRead.model_validate(v) for v in violations]


@router.get(
    "/review/{violation_id}",
    response_model=ViolationRead,
    summary="Fetch a single violation for review",
)
def get_violation(
    violation_id: int,
    db: Session = Depends(get_db),
) -> ViolationRead:
    violation = crud.get_violation(db, violation_id)
    if violation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Violation not found."
        )
    return ViolationRead.model_validate(violation)


@router.post(
    "/review/{violation_id}",
    response_model=ViolationRead,
    summary="Human operator approves or rejects a violation",
)
def review_violation(
    violation_id: int,
    payload: ViolationReviewRequest,
    db: Session = Depends(get_db),
) -> ViolationRead:
    """Apply a human review decision.

    This is the ONLY way a violation can move out of ``pending_human_review``
    and the ONLY way a fine can ever be created (on approval).
    """
    if payload.decision not in _ALLOWED_DECISIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Decision must be either 'approved' or 'rejected'.",
        )

    violation = crud.get_violation(db, violation_id)
    if violation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Violation not found."
        )

    if violation.status != ViolationStatus.PENDING_HUMAN_REVIEW:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Violation has already been reviewed "
                f"(current status: {violation.status.value})."
            ),
        )

    updated = crud.apply_review_decision(
        db,
        violation=violation,
        decision=payload.decision,
        reviewed_by=payload.reviewed_by,
        review_notes=payload.review_notes,
        # When omitted, crud applies the configured default amount/currency
        # for the violation type — approval ALWAYS issues a linked fine.
        fine_amount=payload.fine_amount,
        fine_currency=payload.fine_currency,
    )
    logger.info(
        "Violation %s reviewed by %s -> %s",
        violation_id,
        payload.reviewed_by,
        payload.decision.value,
    )
    return ViolationRead.model_validate(updated)
