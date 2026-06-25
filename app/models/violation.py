"""Violation ORM model."""
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import ViolationStatus, ViolationType
from app.db.session import Base


class Violation(Base):
    """A detected traffic violation pending human review.

    IMPORTANT: Every violation is created with status
    ``PENDING_HUMAN_REVIEW``. The AI is forbidden from issuing fines; the
    transition to ``APPROVED``/``REJECTED`` is performed exclusively by a human
    operator through the review endpoint.
    """

    __tablename__ = "violations"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    vehicle_id: Mapped[int] = mapped_column(
        ForeignKey("vehicles.id", ondelete="CASCADE"), index=True, nullable=False
    )

    violation_type: Mapped[ViolationType] = mapped_column(
        Enum(ViolationType, name="violation_type_enum"),
        default=ViolationType.OVER_SPEEDING,
        nullable=False,
    )
    status: Mapped[ViolationStatus] = mapped_column(
        Enum(ViolationStatus, name="violation_status_enum"),
        default=ViolationStatus.PENDING_HUMAN_REVIEW,
        nullable=False,
        index=True,
    )

    # Evidence / measured metrics.
    detected_speed_kmh: Mapped[float | None] = mapped_column(Float, nullable=True)
    speed_limit_kmh: Mapped[float | None] = mapped_column(Float, nullable=True)
    location: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Optional reference (URL / storage key) to the source evidence image.
    evidence_image_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Human review audit trail.
    reviewed_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    review_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    vehicle: Mapped["Vehicle"] = relationship(  # noqa: F821
        back_populates="violations"
    )
    fine: Mapped["Fine | None"] = relationship(  # noqa: F821
        back_populates="violation",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Violation id={self.id} type={self.violation_type.value} "
            f"status={self.status.value}>"
        )
