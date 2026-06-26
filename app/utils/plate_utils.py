"""Helpers for cleaning and normalising license plate text.

Supports both Latin (A-Z, 0-9) and Cyrillic characters so that Mongolian
plates (e.g. "1234 УБА") are preserved rather than stripped away.

Mongolian civilian plates follow a strict 4-digit + 3-Cyrillic-letter format,
e.g. "1234 УБА" -> "1234УБА". OCR engines frequently emit Latin homoglyphs
(A, B, E, K, M, H, O, P, C, T, X, Y) for the visually identical Cyrillic
letters, so we transliterate those before validating.
"""
import re
import unicodedata

# Keep Latin letters, Cyrillic letters and digits; drop everything else
# (spaces, dashes, punctuation, OCR noise). \u0400-\u04FF is the Cyrillic block
# and covers the Mongolian-specific letters Ү (\u04AE/\u04AF) and Ө
# (\u04E8/\u04E9).
_KEEP = re.compile(r"[^A-Z0-9\u0400-\u04FF]")

# Latin -> Cyrillic homoglyph map. These Latin letters look identical to their
# Cyrillic counterparts and are a very common OCR confusion on Cyrillic plates.
_LATIN_TO_CYRILLIC = {
    "A": "А", "B": "В", "E": "Е", "K": "К", "M": "М",
    "H": "Н", "O": "О", "P": "Р", "C": "С", "T": "Т",
    "X": "Х", "Y": "У",
}

# The Cyrillic letters that may legitimately appear on a Mongolian plate.
# (Full Cyrillic alphabet incl. Mongolian-specific Ө and Ү.)
_CYRILLIC_LETTERS = (
    "АБВГДЕЁЖЗИЙКЛМНОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯ"
)

# Canonical Mongolian civilian plate: 4 digits followed by exactly 3 Cyrillic
# letters.
_MN_PLATE_PATTERN = re.compile(
    r"^(\d{4})([" + _CYRILLIC_LETTERS + r"]{3})$"
)


def _transliterate_latin(text: str) -> str:
    """Map Latin homoglyphs to their Cyrillic counterparts."""
    return "".join(_LATIN_TO_CYRILLIC.get(ch, ch) for ch in text)


def normalize_plate(text: str) -> str:
    """Upper-case and strip characters that never appear in a plate.

    Steps: Unicode NFC normalisation -> upper-case -> strip separators and OCR
    noise (anything that is not a Latin letter, Cyrillic letter or digit).

    Example: "1234 уба" -> "1234УБА"
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text).upper().strip()
    return _KEEP.sub("", text)


def normalize_mn_plate(text: str) -> str:
    """Normalise specifically for Mongolian plates.

    Same as :func:`normalize_plate` but additionally transliterates Latin
    homoglyphs to Cyrillic, so "1234 YBA" (Latin) -> "1234УВА" (Cyrillic).
    """
    return _transliterate_latin(normalize_plate(text))


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


def is_valid_mn_plate(text: str) -> bool:
    """Strict validation of the Mongolian 4-digit + 3-Cyrillic-letter format.

    The input is normalised (incl. Latin->Cyrillic transliteration) before
    matching, so "1234 уба" and "1234 YBA" both validate as True.
    """
    return bool(_MN_PLATE_PATTERN.match(normalize_mn_plate(text)))


def parse_mn_plate(text: str) -> dict:
    """Return normalised value, validity and components for a raw plate string."""
    normalized = normalize_mn_plate(text)
    match = _MN_PLATE_PATTERN.match(normalized)
    return {
        "raw": text,
        "normalized": normalized,
        "is_valid": bool(match),
        "digits": match.group(1) if match else None,
        "letters": match.group(2) if match else None,
    }
