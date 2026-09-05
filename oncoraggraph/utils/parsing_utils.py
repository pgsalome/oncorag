"""Utility helpers for parsing model output, normalizing dates, and cleaning text."""

from __future__ import annotations

import json
import re


def parse_json_loose(text: str):
    """Attempt to parse JSON even when the string contains surrounding noise."""
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(cleaned)


def extract_date_from_note(note_text: str) -> str | None:
    """Extract the most likely encounter date string from a note."""
    patterns = [
        r"(?:encounter|visit|surgery|procedure|date) (?:date )?(?:is|was|on) (\d{1,2}/\d{1,2}/\d{4})",
        r"date of service is (\d{1,2}/\d{1,2}/\d{4})",
        r"occurred on (\d{1,2}/\d{1,2}/\d{4})",
        r"the date is (\d{1,2}/\d{1,2}/\d{4})",
        r"date of operation[: ]+(\d{1,2}/\d{1,2}/\d{2,4})",
        r"date obtained[: ]+(\d{1,2}/\d{1,2}/\d{2,4})",
        r"date received[: ]+(\d{1,2}/\d{1,2}/\d{2,4})",
        r"specimen (?:was )?(?:collected|obtained)[: ]+(\d{1,2}/\d{1,2}/\d{2,4})",
        r"collected on (\d{1,2}/\d{1,2}/\d{4})",
        r"accession #:?\s*[A-Za-z0-9\-]+\s*.*?\b(\d{1,2}/\d{1,2}/\d{2,4})",
        r"chart time[: ]+(\d{4}-\d{2}-\d{2})",
        r"store time[: ]+(\d{4}-\d{2}-\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, note_text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _two_digit_year_to_four(year: int) -> int:
    """Convert 2-digit years to 4-digit with a pivot of 50 (00-49 -> 2000+, 50-99 -> 1900+)."""
    if year < 100:
        return 2000 + year if year <= 49 else 1900 + year
    return year


def normalize_date_to_mmddyyyy(value: str) -> str:
    """Normalize diverse date formats to MM/DD/YYYY; assume day=01 when missing."""
    if not isinstance(value, str) or not value.strip():
        return value

    text = value.strip()
    # Clean artifacts such as 'e.g.' or 'e/' and fix scientific notation typos like '2e22'
    text = re.sub(r"e\.g\.,?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"e/", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\d)e(?=\d{2})", "0", text)

    # Handle strings like '06/08 at 1137'
    match = re.match(r"^\s*(\d{1,2})/(\d{2})\s+at\s+\d", text, flags=re.IGNORECASE)
    if match:
        month = int(match.group(1))
        year = _two_digit_year_to_four(int(match.group(2)))
        return f"{month:02d}/01/{year:04d}"
    month_map = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "sept": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }

    def fmt(month: int, day: int, year: int) -> str:
        return f"{month:02d}/{day:02d}/{year:04d}"

    match = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", text)
    if match:
        month, day, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        year = _two_digit_year_to_four(year)
        if 1 <= month <= 12 and 1 <= day <= 31:
            return fmt(month, day, year)

    match = re.search(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b", text)
    if match:
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return fmt(month, day, year)

    match = re.search(r"\b([A-Za-z]+)\s+(\d{1,2}),\s*(\d{2,4})\b", text)
    if match:
        month = month_map.get(match.group(1).lower())
        day = int(match.group(2))
        year = _two_digit_year_to_four(int(match.group(3)))
        if month and 1 <= day <= 31:
            return fmt(month, day, year)

    match = re.search(r"\b([A-Za-z]+)\s+(\d{1,2})\s+(\d{2,4})\b", text)
    if match:
        month = month_map.get(match.group(1).lower())
        day = int(match.group(2))
        year = _two_digit_year_to_four(int(match.group(3)))
        if month and 1 <= day <= 31:
            return fmt(month, day, year)

    match = re.search(r"\b([A-Za-z]+)\s+(\d{4})\b", text)
    if match:
        month = month_map.get(match.group(1).lower())
        year = int(match.group(2))
        if month:
            return fmt(month, 1, year)

    match = re.search(r"\b(\d{4})[/-](\d{1,2})\b", text)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12:
            return fmt(month, 1, year)

    match = re.search(r"\b(\d{1,2})[/-](\d{4})\b", text)
    if match:
        month, year = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12:
            return fmt(month, 1, year)

    return value


def normalize_entity_text(text: str) -> str:
    """Normalize entity strings to improve downstream matching and deduplication."""
    normalized = text.lower()
    normalized = re.sub(r"\s+(surgery|procedure|operation)$", "", normalized)
    normalized = re.sub(r"^(left|right|bilateral)\s+", "", normalized)
    normalized = " ".join(normalized.split())
    normalized = normalized.rstrip(".,;:")
    return normalized
