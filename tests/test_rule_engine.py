"""Unit tests for the rule engine (no ML dependencies required)."""
from app.core.enums import ViolationType
from app.schemas.analysis import AnalysisMetadata
from app.services.rule_engine import RuleEngine


def test_overspeeding_triggers_violation():
    engine = RuleEngine()
    result = engine.evaluate(AnalysisMetadata(speed=80, speed_limit=60))
    assert result is not None
    assert result.violation_type == ViolationType.OVER_SPEEDING
    assert result.detected_speed_kmh == 80
    assert result.speed_limit_kmh == 60


def test_compliant_speed_no_violation():
    engine = RuleEngine()
    assert engine.evaluate(AnalysisMetadata(speed=55, speed_limit=60)) is None


def test_missing_metadata_no_violation():
    engine = RuleEngine()
    assert engine.evaluate(AnalysisMetadata()) is None
