"""The Rule Engine evaluates contextual metadata to flag violations.

It is intentionally simple and pluggable: each rule is a small function that
inspects the metadata and returns an optional ``RuleResult``. New rule types
(red-light, wrong-way, etc.) can be added by registering more rule callables.
"""
from dataclasses import dataclass
from typing import Callable

from app.core.config import settings
from app.core.enums import ViolationType
from app.core.logging_config import get_logger
from app.schemas.analysis import AnalysisMetadata

logger = get_logger(__name__)


@dataclass
class RuleResult:
    """The outcome of evaluating a single rule."""

    violation_type: ViolationType
    description: str
    detected_speed_kmh: float | None = None
    speed_limit_kmh: float | None = None


def _speed_rule(metadata: AnalysisMetadata) -> RuleResult | None:
    """Flag an over-speeding violation when speed exceeds the limit."""
    speed = metadata.speed
    limit = metadata.speed_limit
    if speed is None or limit is None:
        return None
    if speed > limit + settings.SPEED_TOLERANCE_KMH:
        over_by = round(speed - limit, 2)
        return RuleResult(
            violation_type=ViolationType.OVER_SPEEDING,
            description=(
                f"Vehicle travelled at {speed} km/h in a {limit} km/h zone "
                f"(exceeded by {over_by} km/h)."
            ),
            detected_speed_kmh=speed,
            speed_limit_kmh=limit,
        )
    return None


# Registry of active rules. Append new rule callables here to extend logic.
_RULES: list[Callable[[AnalysisMetadata], "RuleResult | None"]] = [
    _speed_rule,
]


class RuleEngine:
    """Evaluates all registered rules against the provided metadata."""

    def evaluate(self, metadata: AnalysisMetadata) -> RuleResult | None:
        """Return the first triggered rule result, or ``None`` if compliant."""
        for rule in _RULES:
            result = rule(metadata)
            if result is not None:
                logger.info(
                    "Rule triggered: %s - %s",
                    result.violation_type.value,
                    result.description,
                )
                return result
        logger.info("No rule triggered; metadata indicates compliance.")
        return None


def get_rule_engine() -> RuleEngine:
    """FastAPI-friendly accessor for the rule engine."""
    return RuleEngine()
