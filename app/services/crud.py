"""Persistence layer (CRUD) for the domain models.

Keeping database access here decouples the API layer from SQLAlchemy details
and centralises the business invariants (e.g. violations always start as
``pending_human_review``).
"""
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.enums import FineStatus, ViolationStatus, ViolationType
from app.models.fine import Fine
from app.models.vehicle import Vehicle
from app.models.violation import Violation


# --- Vehicles --------------------------------------------------------------
def get_or_create_vehicle(
    db: Session,
    license_plate: str,
    vehicle_type: str | None = None,
    plate_confidence: float | None = None,
) -> Vehicle:
    """Return the existing vehicle for a plate, or create a new one."""
    vehicle = db.scalar(
        select(Vehicle).where(Vehicle.license_plate == license_plate)
    )
    if vehicle is None:
        vehicle = Vehicle(
            license_plate=license_plate,
            vehicle_type=vehicle_type,
            plate_confidence=plate_confidence,
        )
        db.add(vehicle)
        db.flush()  # populate vehicle.id without committing yet
    else:
        # Update metadata if we have fresher information.
        if vehicle_type:
            vehicle.vehicle_type = vehicle_type
        if plate_confidence is not None:
            vehicle.plate_confidence = plate_confidence
    return vehicle


# --- Violations ------------------------------------------------------------
def create_pending_violation(
    db: Session,
    vehicle: Vehicle,
    violation_type: ViolationType,
    description: str,
    detected_speed_kmh: float | None = None,
    speed_limit_kmh: float | None = None,
    location: str | None = None,
    evidence_image_ref: str | None = None,
) -> Violation:
    """Create a violation that ALWAYS starts as pending human review.

    This function is the single entry point the AI pipeline uses; it can never
    produce an approved violation or a fine.
    """
    violation = Violation(
        vehicle=vehicle,
        violation_type=violation_type,
        status=ViolationStatus.PENDING_HUMAN_REVIEW,
        description=description,
        detected_speed_kmh=detected_speed_kmh,
        speed_limit_kmh=speed_limit_kmh,
        location=location,
        evidence_image_ref=evidence_image_ref,
    )
    db.add(violation)
    db.commit()
    db.refresh(violation)
    return violation


def get_violation(db: Session, violation_id: int) -> Violation | None:
    """Fetch a violation with its vehicle and fine eagerly loaded."""
    return db.scalar(
        select(Violation)
        .options(selectinload(Violation.vehicle), selectinload(Violation.fine))
        .where(Violation.id == violation_id)
    )


def list_violations(
    db: Session,
    status: ViolationStatus | None = None,
    skip: int = 0,
    limit: int = 50,
) -> list[Violation]:
    """List violations, optionally filtered by status."""
    stmt = (
        select(Violation)
        .options(selectinload(Violation.vehicle), selectinload(Violation.fine))
        .order_by(Violation.detected_at.desc())
        .offset(skip)
        .limit(limit)
    )
    if status is not None:
        stmt = stmt.where(Violation.status == status)
    return list(db.scalars(stmt).all())


def apply_review_decision(
    db: Session,
    violation: Violation,
    decision: ViolationStatus,
    reviewed_by: str,
    review_notes: str | None = None,
    fine_amount: float | None = None,
    fine_currency: str = "USD",
) -> Violation:
    """Apply a human operator's review decision.

    On approval, optionally create the associated fine. This is the ONLY path
    through which a fine can ever be created.
    """
    violation.status = decision
    violation.reviewed_by = reviewed_by
    violation.review_notes = review_notes
    violation.reviewed_at = datetime.now(timezone.utc)

    if decision == ViolationStatus.APPROVED and fine_amount is not None:
        fine = Fine(
            violation=violation,
            amount=fine_amount,
            currency=fine_currency,
            status=FineStatus.ISSUED,
            issued_by=reviewed_by,
        )
        db.add(fine)

    db.commit()
    db.refresh(violation)
    return violation
