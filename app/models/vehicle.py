"""Vehicle ORM model."""
from datetime import datetime

from sqlalchemy import DateTime, Float, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class Vehicle(Base):
    """A vehicle identified by its license plate.

    A vehicle is created (or re-used) whenever a plate is recognised during
    analysis. Multiple violations can reference the same vehicle.
    """

    __tablename__ = "vehicles"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    license_plate: Mapped[str] = mapped_column(
        String(32), unique=True, index=True, nullable=False
    )
    vehicle_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Confidence with which the plate text was extracted by OCR.
    plate_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    violations: Mapped[list["Violation"]] = relationship(  # noqa: F821
        back_populates="vehicle",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Vehicle id={self.id} plate={self.license_plate!r}>"
