"""Pydantic schemas for the Violation resource."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import ViolationStatus, ViolationType
from app.schemas.fine import FineRead
from app.schemas.vehicle import VehicleRead


class ViolationBase(BaseModel):
    violation_type: ViolationType = ViolationType.OVER_SPEEDING
    detected_speed_kmh: float | None = None
    speed_limit_kmh: float | None = None
    location: str | None = None
    description: str | None = None
    evidence_image_ref: str | None = None


class ViolationRead(ViolationBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    vehicle_id: int
    status: ViolationStatus
    reviewed_by: str | None = None
    review_notes: str | None = None
    reviewed_at: datetime | None = None
    detected_at: datetime
    created_at: datetime

    vehicle: VehicleRead | None = None
    fine: FineRead | None = None


class ReviewDecision(str):
    """Allowed human review decisions (approved / rejected)."""


class ViolationReviewRequest(BaseModel):
    """Payload submitted by a human operator to review a violation."""

    decision: ViolationStatus = Field(
        ...,
        description="Must be 'approved' or 'rejected'.",
        examples=[ViolationStatus.APPROVED],
    )
    reviewed_by: str = Field(..., max_length=128, examples=["operator_jane"])
    review_notes: str | None = Field(default=None, examples=["Clear evidence."])

    # Optional fine details applied only when decision == approved.
    fine_amount: float | None = Field(default=None, ge=0, examples=[150.0])
    fine_currency: str | None = Field(default="USD", max_length=3)
