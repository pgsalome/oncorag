"""Load clinical notes with authoritative patient, report, and date metadata."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date as calendar_date
import json
from pathlib import Path
import re
from typing import Iterable


_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")
_NOTE_FILENAME = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})(?:__(?P<suffix>.+))?\.txt")
_REQUIRED_FIELDS = {"patient_id", "report_type", "date", "path"}
_LANGUAGE_ALIASES = {
    "en": "en", "eng": "en", "english": "en",
    "de": "de", "deu": "de", "ger": "de", "german": "de", "deutsch": "de",
    "unknown": "unknown", "und": "unknown", "mixed": "mixed",
}


@dataclass(frozen=True)
class NoteRecord:
    patient_id: str
    note_id: str
    report_type: str
    date: str
    path: Path
    text: str
    language: str = "unknown"

    def __post_init__(self) -> None:
        for field in ("patient_id", "note_id", "report_type"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"{field} must be a nonempty string without surrounding whitespace")
            if any(ord(char) < 32 for char in value):
                raise ValueError(f"{field} must not contain control characters")
            if field != "note_id" and (value in {".", ".."} or "/" in value or "\\" in value):
                raise ValueError(f"{field} must be a single path component")
        _validate_date(self.date)
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError(f"Note {self.note_id!r} has blank text")
        object.__setattr__(self, "path", Path(self.path).resolve())
        object.__setattr__(self, "language", _normalize_language(self.language))


def _validate_date(value: str) -> str:
    if not isinstance(value, str) or not _DATE_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid note date {value!r}; expected YYYY-MM-DD")
    try:
        calendar_date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid calendar date {value!r}") from exc
    return value


def _normalize_language(value: str) -> str:
    if not isinstance(value, str) or value.strip().lower() not in _LANGUAGE_ALIASES:
        raise ValueError("Note language must be en, de, mixed, or unknown")
    return _LANGUAGE_ALIASES[value.strip().lower()]


def _read_note(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"Note file does not exist or is not a file: {path}")
    try:
        return path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read UTF-8 note: {path}") from exc


def _folder_notes(root: Path, default_language: str) -> Iterable[NoteRecord]:
    if not root.is_dir():
        raise ValueError(f"Notes root is not a directory: {root}")
    for path in sorted(root.rglob("*.txt")):
        relative = path.relative_to(root)
        if not path.is_file() or len(relative.parts) != 3:
            raise ValueError(f"Expected patient_id/report_type/YYYY-MM-DD[__note_id].txt: {relative}")
        patient_id, report_type, filename = relative.parts
        match = _NOTE_FILENAME.fullmatch(filename)
        if match is None:
            raise ValueError(f"Invalid note filename: {relative}")
        yield NoteRecord(
            patient_id=patient_id,
            note_id=f"{report_type}/{path.stem}",
            report_type=report_type,
            date=match.group("date"),
            path=path,
            text=_read_note(path),
            language=default_language,
        )


def _registry_rows(registry: Path) -> list[dict]:
    if not registry.is_file():
        raise ValueError(f"Registry does not exist or is not a file: {registry}")
    suffix = registry.suffix.lower()
    try:
        with registry.open(encoding="utf-8-sig", newline="") as handle:
            if suffix == ".csv":
                reader = csv.DictReader(handle)
                fields = reader.fieldnames or []
                if len(fields) != len(set(fields)):
                    raise ValueError("Registry CSV has duplicate column names")
                if not _REQUIRED_FIELDS.issubset(fields):
                    raise ValueError(f"Registry requires columns: {', '.join(sorted(_REQUIRED_FIELDS))}")
                rows = list(reader)
                if any(None in row for row in rows):
                    raise ValueError("Registry CSV has a row with more values than columns")
            elif suffix == ".json":
                rows = json.load(handle)
                if isinstance(rows, dict):
                    rows = rows.get("notes")
                if not isinstance(rows, list):
                    raise ValueError("Registry JSON must be a list of notes or an object containing 'notes'")
            elif suffix == ".jsonl":
                rows = []
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid registry JSON on line {line_number}") from exc
            else:
                raise ValueError("Registry must be CSV, JSON, or JSONL")
    except (OSError, UnicodeError, csv.Error, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read registry: {registry}: {exc}") from exc
    return rows


def _registry_notes(registry: Path, default_language: str) -> Iterable[NoteRecord]:
    for row_number, row in enumerate(_registry_rows(registry), start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Registry record {row_number} must be an object")
        for field in _REQUIRED_FIELDS:
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Registry record {row_number} requires a nonempty string '{field}'")
        path = Path(row["path"])
        if not path.is_absolute():
            path = registry.parent / path
        note_id = row.get("note_id") or f"{row['report_type']}/{path.stem}"
        yield NoteRecord(
            patient_id=row["patient_id"],
            note_id=note_id,
            report_type=row["report_type"],
            date=row["date"],
            path=path,
            text=_read_note(path),
            language=row.get("language") or default_language,
        )


def group_notes_by_patient(records: Iterable[NoteRecord]) -> dict[str, list[NoteRecord]]:
    """Group notes deterministically, rejecting duplicate identities or source files."""
    patients: dict[str, list[NoteRecord]] = {}
    seen_ids: set[tuple[str, str]] = set()
    seen_paths: set[Path] = set()
    for record in sorted(records, key=lambda note: (note.patient_id, note.date, note.report_type, note.note_id)):
        identity = (record.patient_id, record.note_id)
        if identity in seen_ids:
            raise ValueError(f"Duplicate note_id {record.note_id!r} for patient {record.patient_id!r}")
        if record.path in seen_paths:
            raise ValueError(f"Duplicate note source file: {record.path}")
        seen_ids.add(identity)
        seen_paths.add(record.path)
        patients.setdefault(record.patient_id, []).append(record)
    return patients


def load_notes(
    *,
    notes_root: str | Path | None = None,
    registry_path: str | Path | None = None,
    default_language: str = "unknown",
) -> list[NoteRecord]:
    """Load either a classified note folder or a registry with explicit metadata.

    A registry requires patient_id, report_type, date, and path. Its optional
    note_id and language fields override deterministic IDs and the language
    default. Relative note paths are interpreted from the registry directory.
    """
    if bool(notes_root) == bool(registry_path):
        raise ValueError("Specify exactly one of notes_root or registry_path")
    language = _normalize_language(default_language)
    if notes_root:
        records = list(_folder_notes(Path(notes_root).resolve(), language))
    else:
        records = list(_registry_notes(Path(registry_path).resolve(), language))
    if not records:
        raise ValueError("No clinical notes found in the configured input")
    grouped = group_notes_by_patient(records)
    return [note for notes in grouped.values() for note in notes]


__all__ = ["NoteRecord", "load_notes", "group_notes_by_patient"]
