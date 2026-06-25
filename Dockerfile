# syntax=docker/dockerfile:1

##############################################################################
# Traffic Violation Detection System - API image
#
# Uses a slim Python base and installs the system libraries required by
# OpenCV / EasyOCR / Torch. Designed for CPU inference out of the box.
##############################################################################
FROM python:3.11-slim AS runtime

# Prevent Python from writing .pyc files and buffering stdout/stderr.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies needed by OpenCV, Torch and EasyOCR at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first to leverage Docker layer caching.
COPY requirements.txt .
# Install CPU-only torch wheels to keep the image small and avoid CUDA.
RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source.
COPY app ./app

# Create a non-root user for safety.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /home/appuser/.EasyOCR \
    && chown -R appuser:appuser /app /home/appuser
USER appuser

# EasyOCR / Ultralytics cache locations.
ENV HOME=/home/appuser

EXPOSE 8000

# Basic container healthcheck hitting the liveness probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
