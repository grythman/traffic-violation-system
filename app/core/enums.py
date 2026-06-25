"""Shared enumerations for the domain model."""
import enum


class ViolationType(str, enum.Enum):
    """Supported categories of traffic violations."""

    OVER_SPEEDING = "over_speeding"
    RED_LIGHT = "red_light"
    NO_HELMET = "no_helmet"
    WRONG_WAY = "wrong_way"
    OTHER = "other"


class ViolationStatus(str, enum.Enum):
    """Lifecycle states of a violation record.

    A violation is ALWAYS created as ``PENDING_HUMAN_REVIEW``. The AI never
    issues a fine directly; only a human operator can approve or reject.
    """

    PENDING_HUMAN_REVIEW = "pending_human_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class FineStatus(str, enum.Enum):
    """Lifecycle states of a fine, issued only after human approval."""

    ISSUED = "issued"
    PAID = "paid"
    CANCELLED = "cancelled"
