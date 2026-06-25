"""Database initialisation utilities.

In production you would manage schema migrations with Alembic. For the scope
of this project we expose a simple ``init_db`` that creates all tables. It is
idempotent and safe to call on every startup.
"""
from app.core.logging_config import get_logger
from app.db.session import Base, engine

# Importing the models module registers every model on ``Base.metadata``.
from app import models  # noqa: F401  (side-effect import)

logger = get_logger(__name__)


def init_db() -> None:
    """Create all tables that do not yet exist."""
    logger.info("Creating database tables (if not present)...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables are ready.")
