"""Helpers for cleaning and normalising license plate text.

Supports both Latin (A-Z, 0-9) and Cyrillic characters so that Mongolian
plates (e.g. "1234 УБА") are preserved rather than stripped away.
"""
import re

# Keep Latin letters, Cyrillic letters and digits; drop everything else
# (spaces, dashes, punctuation, OCR noise). \u0400-\u04FF is the Cyrillic block.
_KEEP = re.compile(r"[^A-Z0-9\u0400-\u04FF]")


def normalize_plate(text: str) -> str:
    """Upper-case and strip characters that never appear in a plate.

    Latin and Cyrillic letters are both upper-cased and kept; digits are kept.
    """
    return _KEEP.sub("", text.upper().strip())


def _is_letter(ch: str) -> bool:
    """True for Latin or Cyrillic letters."""
    return ch.isalpha()


def is_plausible_plate(text: str, min_len: int = 4, max_len: int = 12) -> bool:
    """Heuristic check that a recognised string looks like a plate.

    A plausible plate has a reasonable length and contains at least one digit
    or letter. This filters out spurious OCR noise while accepting both Latin
    and Cyrillic (Mongolian) plate formats.
    """
    cleaned = normalize_plate(text)
    if not (min_len <= len(cleaned) <= max_len):
        return False
    has_digit = any(c.isdigit() for c in cleaned)
    has_alpha = any(_is_letter(c) for c in cleaned)
    return has_digit or has_alpha
