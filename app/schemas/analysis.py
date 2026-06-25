"""Schemas for the /analyze endpoint."""
from pydantic import BaseModel, Field, model_validator

from app.schemas.violation import ViolationRead


class AnalysisMetadata(BaseModel):
    """Contextual metadata used by the rule engine."""

    speed: float | None = Field(
        default=None, ge=0, description="Measured speed in km/h.", examples=[80]
    )
    speed_limit: float | None = Field(
        default=None, ge=0, description="Legal speed limit in km/h.", examples=[60]
    )
    location: str | None = Field(default=None, examples=["Main St & 5th Ave"])


class AnalyzeRequest(BaseModel):
    """Request body for /analyze. Provide EITHER ``image_base64`` OR ``image_url``."""

    image_base64: str | None = Field(
        default=None, description="Base64-encoded image bytes (optionally data-URI)."
    )
    image_url: str | None = Field(
        default=None, description="Publicly reachable URL of the image."
    )
    metadata: AnalysisMetadata = Field(default_factory=AnalysisMetadata)

    @model_validator(mode="after")
    def ensure_one_source(self) -> "AnalyzeRequest":
        if not self.image_base64 and not self.image_url:
            raise ValueError("Provide either 'image_base64' or 'image_url'.")
        if self.image_base64 and self.image_url:
            raise ValueError(
                "Provide only one of 'image_base64' or 'image_url', not both."
            )
        return self


class DetectedVehicle(BaseModel):
    """A single detected vehicle plus its recognised plate."""

    vehicle_type: str
    detection_confidence: float
    bounding_box: list[int] = Field(
        ..., description="[x1, y1, x2, y2] in pixel coordinates."
    )
    license_plate: str | None = None
    plate_confidence: float | None = None


class AnalyzeResponse(BaseModel):
    """Response returned by /analyze."""

    detected_vehicles: list[DetectedVehicle]
    primary_license_plate: str | None = None
    violation_detected: bool = False
    violation: ViolationRead | None = None
    message: str
