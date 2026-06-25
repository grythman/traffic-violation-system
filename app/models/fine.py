"""Fine ORM model."""
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import FineStatus
from app.db.session import Base


class Fine(Base):
    """A monetary fine.

    A fine is ONLY created after a human operator approves the related
    violation. It has a strict one-to-one relationship with a violation.
    """

    __tablename__ = "fines"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    violation_id: Mapped[int] = mapped_column(
        ForeignKey("violations.id", ondelete="CASCADE"),
        unique=True,
        index=True,
        nullable=False,
    )

    amount: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="USD", nullable=False)
    status: Mapped[FineStatus] = mapped_column(
        Enum(FineStatus, name="fine_status_enum"),
        default=FineStatus.ISSUED,
        nullable=False,
    )
    # The human operator who authorised issuing this fine.
    issued_by: Mapped[str | None] = mapped_column(String(128), nullable=True)

    issued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    violation: Mapped["Violation"] = relationship(  # noqa: F821
        back_populates="fine"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Fine id={self.id} violation_id={self.violation_id} "
            f"amount={self.amount} {self.currency}>"
        )
