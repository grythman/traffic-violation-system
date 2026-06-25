"""Pydantic schemas for the Fine resource."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import FineStatus


class FineBase(BaseModel):
    amount: float = Field(..., ge=0)
    currency: str = Field(default="USD", max_length=3)


class FineRead(FineBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    violation_id: int
    status: FineStatus
    issued_by: str | None = None
    issued_at: datetime
