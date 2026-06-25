"""Pydantic schemas for the Vehicle resource."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class VehicleBase(BaseModel):
    license_plate: str = Field(..., max_length=32, examples=["ABC1234"])
    vehicle_type: str | None = Field(default=None, examples=["car"])
    plate_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class VehicleCreate(VehicleBase):
    pass


class VehicleRead(VehicleBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: datetime
