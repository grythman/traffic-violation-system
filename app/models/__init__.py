"""ORM models package.

Importing this package registers every model on ``Base.metadata`` so that
``create_all`` (and Alembic autogenerate) can discover them.
"""
from app.models.fine import Fine
from app.models.vehicle import Vehicle
from app.models.violation import Violation

__all__ = ["Vehicle", "Violation", "Fine"]
