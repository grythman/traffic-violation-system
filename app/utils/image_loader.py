"""Utilities to load images from base64 strings or remote URLs into arrays."""
import base64
import binascii
from io import BytesIO

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class ImageLoadError(ValueError):
    """Raised when an image cannot be decoded or fetched."""


def _pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an RGB numpy array."""
    return np.array(image.convert("RGB"))


def load_image_from_base64(data: str) -> np.ndarray:
    """Decode a (possibly data-URI prefixed) base64 string into an RGB array."""
    if "," in data and data.strip().startswith("data:"):
        # Strip a data URI prefix like "data:image/png;base64,".
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageLoadError("Invalid base64 image data.") from exc
    try:
        image = Image.open(BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageLoadError("Could not decode the provided image bytes.") from exc
    return _pil_to_rgb_array(image)


def load_image_from_url(url: str) -> np.ndarray:
    """Fetch an image from a URL and decode it into an RGB array."""
    try:
        resp = requests.get(url, timeout=settings.IMAGE_FETCH_TIMEOUT, stream=True)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise ImageLoadError(f"Failed to fetch image from URL: {exc}") from exc
    try:
        image = Image.open(BytesIO(resp.content))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageLoadError("Fetched resource is not a valid image.") from exc
    return _pil_to_rgb_array(image)
