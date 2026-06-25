"""Application configuration.

All settings are loaded from environment variables (12-factor app style),
with sane defaults so the project also runs locally without Docker.
"""
from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised, type-safe application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- General -----------------------------------------------------------
    PROJECT_NAME: str = "Traffic Violation Detection System"
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # --- Database ----------------------------------------------------------
    POSTGRES_USER: str = "tvds_user"
    POSTGRES_PASSWORD: str = "tvds_password"
    POSTGRES_HOST: str = "db"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "tvds_db"
    DATABASE_URL: str | None = None

    # --- ML / Inference ----------------------------------------------------
    # --- Vehicle detector (stage 1) ---------------------------------------
    # General vehicle detector. "yolov8n.pt" (COCO) is auto-downloaded and is
    # fine for finding cars/trucks; it does NOT need Mongolian training.
    YOLO_MODEL_PATH: str = "yolov8n.pt"
    # Minimum confidence threshold for accepting a YOLO detection.
    YOLO_CONFIDENCE_THRESHOLD: float = 0.35

    # --- License-plate detector (stage 2, OPTIONAL but recommended) --------
    # A CUSTOM YOLOv8 .pt trained to localise the plate region. For Mongolian
    # plates, train/obtain a plate-detection model and point this at the .pt
    # file (mounted into the container at /app/models/). When empty, the
    # system falls back to running OCR on the whole vehicle crop.
    PLATE_MODEL_PATH: str = ""  # e.g. "/app/models/mn_plate_yolov8.pt"
    PLATE_CONFIDENCE_THRESHOLD: float = 0.30

    # --- OCR (stage 3) -----------------------------------------------------
    # EasyOCR languages. Mongolian plates use the Cyrillic alphabet, so use
    # ["mn"] or ["mn", "en"]. EasyOCR supports a "mn" Cyrillic model.
    # For best accuracy on Mongolian plates, consider a custom-trained
    # recognition model (see README "Mongolian plates" section).
    OCR_LANGUAGES: List[str] = ["en"]
    # Optional EasyOCR character allow-list to constrain recognition output,
    # e.g. Latin + Cyrillic + digits. Empty = no restriction.
    OCR_ALLOWLIST: str = ""

    # Run OCR/YOLO on GPU if available.
    USE_GPU: bool = False

    # --- Rule Engine -------------------------------------------------------
    # Tolerance (km/h) applied before flagging an over-speed violation.
    SPEED_TOLERANCE_KMH: float = 0.0

    # --- HTTP fetch --------------------------------------------------------
    IMAGE_FETCH_TIMEOUT: int = 15  # seconds

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_url(cls, v: str | None, info) -> str:
        """Build a SQLAlchemy URL from discrete parts when not provided."""
        if isinstance(v, str) and v:
            return v
        data = info.data
        user = data.get("POSTGRES_USER")
        password = data.get("POSTGRES_PASSWORD")
        host = data.get("POSTGRES_HOST")
        port = data.get("POSTGRES_PORT")
        db = data.get("POSTGRES_DB")
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()


settings = get_settings()
