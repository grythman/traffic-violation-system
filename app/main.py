"""FastAPI application factory and entrypoint."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging_config import configure_logging, get_logger
from app.db.init_db import init_db

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Run startup/shutdown logic."""
    logger.info("Starting %s ...", settings.PROJECT_NAME)
    init_db()
    yield
    logger.info("Shutting down %s ...", settings.PROJECT_NAME)


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version="1.0.0",
        description=(
            "Modular Traffic Violation Detection System. The AI detects "
            "vehicles (YOLOv8) and reads plates (EasyOCR), but NEVER issues "
            "fines: every violation is recorded as 'pending_human_review'."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    @app.get("/", tags=["root"])
    def root() -> dict:
        return {
            "service": settings.PROJECT_NAME,
            "docs": "/docs",
            "api_prefix": settings.API_V1_PREFIX,
        }

    return app


app = create_app()
