#!/usr/bin/env python3
"""Export reviewed synthetic inputs into portable, metadata-driven note datasets.

Only explicitly selected label fields enter the output. The clinical text is
preserved, so source provenance and a separate text review still matter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any


REGISTRY_FIELDS = ["patient_id", "note_id", "report_type", "date", "language", "path"]
EVENT_STRING_FIELDS = ("ctcae_term", "grade_status", "temporality", "event_status")
CONTACT_MARKERS = re.compile(
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|https?://|"
    r"\b(?:MRN|medical record number)\s*[:#]?\s*\d{5,}",
    re.IGNORECASE,
)
LOCAL_PATH_MARKERS = re.compile(r"/(?:home|Users|mnt)/|[A-Za-z]:\\Users\\")
SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
SOURCE_PATIENT_PATTERNS = {
    "en": re.compile(r"SYN-TNBC-\d+\Z"),
    "de": re.compile(r"SYN-RICCI-\d+\Z"),
}


def safe_child(root: Path, relative: str) -> Path:
    candidate = root / relative
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("A dataset path escapes its root") from exc
    return candidate


def component(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SAFE_COMPONENT.fullmatch(value):
        raise ValueError(f"Invalid {field}: expected a nonempty safe path component")
    return value


def screen_text(text: str, identifiers: re.Pattern[str] | None = None) -> None:
    if identifiers is not None and identifiers.search(text):
        raise ValueError("Known source identifier found in public text; review required")
    if CONTACT_MARKERS.search(text):
        raise ValueError("Contact or record-number marker found in public text; review required")
    if LOCAL_PATH_MARKERS.search(text):
        raise ValueError("Local filesystem path found in public text; review required")


def identifier_pattern(records: list[dict[str, Any]]) -> re.Pattern[str] | None:
    identifiers = {
        str(record["source_style_patient_id"]).strip()
        for record in records
        if record.get("source_style_patient_id") not in (None, "", "style_unknown")
    }
    if not identifiers:
        return None
    return re.compile(
        r"(?<![A-Za-z0-9])(?:"
        + "|".join(re.escape(value) for value in sorted(identifiers))
        + r")(?![A-Za-z0-9])"
    )


def compact_events(events: Any, identifiers: re.Pattern[str] | None) -> list[dict[str, Any]]:
    if not isinstance(events, list):
        raise ValueError("Expected a list of synthetic events")
    result = []
    for index, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            raise ValueError("Expected a synthetic event object")
        item: dict[str, Any] = {"event_id": f"event_{index:03d}"}
        for key in EVENT_STRING_FIELDS:
            if key in event and event[key] is not None:
                value = event[key]
                if not isinstance(value, str) or len(value) > 300:
                    raise ValueError(f"Invalid synthetic event {key}")
                screen_text(value, identifiers)
                item[key] = value
        if not item.get("ctcae_term"):
            raise ValueError("Missing CTCAE term in synthetic event")
        grade = event.get("grade")
        if grade is not None and (type(grade) is not int or grade not in range(0, 6)):
            raise ValueError("Synthetic grade must be null or an integer from 0 to 5")
        item["grade"] = grade
        if "negated" in event:
            if type(event["negated"]) is not bool:
                raise ValueError("Synthetic negated value must be boolean")
            item["negated"] = event["negated"]
        result.append(item)
    return result


def patient_splits(patient_ids: list[str], seed: int = 42) -> dict[str, str]:
    patients = sorted(set(patient_ids))
    random.Random(seed).shuffle(patients)
    train_end = int(len(patients) * 0.7)
    dev_end = train_end + int(len(patients) * 0.15)
    return {
        patient_id: "train" if i < train_end else "dev" if i < dev_end else "test"
        for i, patient_id in enumerate(patients)
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_registry(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REGISTRY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(root: Path, metadata: dict[str, Any]) -> None:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    metadata["files"] = {
        path.relative_to(root).as_posix(): {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for path in files
    }
    metadata["payload_bytes"] = sum(entry["bytes"] for entry in metadata["files"].values())
    write_json(root / "manifest.json", metadata)


def export_cohort(source: Path, destination: Path, language: str, seed: int = 42) -> dict[str, Any]:
    """Export one full cohort; refuse overwrite and paths outside its roots."""
    if language not in SOURCE_PATIENT_PATTERNS:
        raise ValueError("Full synthetic export supports en or de")
    source = source.resolve()
    destination = destination.absolute()
    if destination.exists():
        raise FileExistsError("Output dataset exists; use a new versioned output directory")
    if source == destination.resolve() or source in destination.resolve().parents:
        raise ValueError("Output dataset cannot be inside its source")
    labels_dir = safe_child(source, "labels")
    notes_dir = safe_child(source, "synthetic_notes")
    if not labels_dir.is_dir() or not notes_dir.is_dir():
        raise ValueError("Source requires labels/ and synthetic_notes/ directories")
    records = []
    for path in sorted(labels_dir.glob("*.json")):
        safe_child(source, path.relative_to(source).as_posix())
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict) or obj.get("note_id") != path.stem:
            raise ValueError("Source note ID does not match its label filename")
        records.append(obj)
    if not records:
        raise ValueError("Source contains no synthetic labels")
    identifiers = identifier_pattern(records)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".synthetic-export-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        registry = []
        labels = []
        seen_notes: set[str] = set()
        patient_ids = []
        for record in records:
            patient_id = component(record.get("patient_id"), "patient_id")
            if not SOURCE_PATIENT_PATTERNS[language].fullmatch(patient_id):
                raise ValueError("Source patient identifier does not use the expected synthetic prefix")
            note_id = component(record.get("note_id"), "note_id")
            if note_id in seen_notes:
                raise ValueError("Duplicate source note ID")
            seen_notes.add(note_id)
            report_type = component(record.get("note_type"), "report_type")
            note_date = str(record.get("note_date", ""))
            if date.fromisoformat(note_date).isoformat() != note_date:
                raise ValueError("Synthetic note date must be YYYY-MM-DD")
            note_path = safe_child(notes_dir, note_id + ".txt")
            note_text = note_path.read_text(encoding="utf-8")
            if not note_text.strip():
                raise ValueError("Empty synthetic note")
            screen_text(note_text, identifiers)
            relative = f"notes/{patient_id}/{report_type}/{note_date}__{note_id}.txt"
            output_note = safe_child(staging, relative)
            output_note.parent.mkdir(parents=True, exist_ok=True)
            output_note.write_text(note_text, encoding="utf-8")
            metadata = {
                "patient_id": patient_id,
                "note_id": note_id,
                "report_type": report_type,
                "date": note_date,
                "language": language,
            }
            registry.append({**metadata, "path": relative})
            labels.append({**metadata, "events": compact_events(record.get("events"), identifiers)})
            patient_ids.append(patient_id)
        write_registry(staging / "registry.csv", registry)
        write_jsonl(staging / "labels.jsonl", labels)
        splits = patient_splits(patient_ids, seed)
        write_json(staging / "splits.json", {
            "unit": "patient", "seed": seed, "ratios": {"train": 0.7, "dev": 0.15, "test": 0.15},
            "patients": dict(sorted(splits.items())),
        })
        metadata = {
            "schema_version": 1,
            "dataset_id": "synthetic_english" if language == "en" else "synthetic_german",
            "language": language,
            "patient_count": len(splits),
            "note_count": len(registry),
            "event_count": sum(len(row["events"]) for row in labels),
            "split_patient_counts": dict(sorted(Counter(splits.values()).items())),
            "provenance": {
                "generation": "CTCAE template-derived English notes with added Synthea encounter metadata"
                if language == "en" else "Purpose-generated German recurrent high-grade glioma template notes",
                "source_dataset": "hybrid_synthea_ctcae_phase2" if language == "en"
                else "ricci_rhgg_termgrade_longitudinal",
                "text_changed": False,
                "labels": "Allowlisted note metadata and event labels; evidence and original source metadata omitted",
                "license_status": "Release license and source-text redistribution review pending",
            },
            "screening": {
                "notes_checked": len(registry),
                "labels_checked": len(labels),
                "known_source_identifier_matches": 0,
                "contact_or_record_marker_matches": 0,
                "absolute_path_marker_matches": 0,
                "scope": "Known source identifiers and explicit markers only; not a complete privacy review",
            },
            "gold_scope": "Note-level source CTCAE event labels only; no general patient-feature gold is inferred",
        }
        write_manifest(staging, metadata)
        staging.rename(destination)
    return {key: metadata[key] for key in ("dataset_id", "patient_count", "note_count", "event_count", "payload_bytes")}


# These cases are authored here, with no source-cohort or patient-derived text.
FIXTURE_CASES = [
    {"id": "SYN-DEMO-001", "diagnosis_date": "2020-03-01", "age": 52,
     "treatment": "temozolomide", "diagnosis": {"en": "glioblastoma", "de": "Glioblastom"},
     "therapy_date": "2020-03-15", "lab_date": "2020-04-01", "early_hb": 12.4, "latest_hb": 11.2},
    {"id": "SYN-DEMO-002", "diagnosis_date": "2021-05-10", "age": 67,
     "treatment": "bevacizumab", "diagnosis": {"en": "recurrent glioblastoma", "de": "rezidiviertes Glioblastom"},
     "therapy_date": "2021-05-24", "lab_date": "2021-06-03", "early_hb": 10.3, "latest_hb": 12.1},
    {"id": "SYN-DEMO-003", "diagnosis_date": "2022-01-08", "age": 44,
     "treatment": "radiotherapy", "diagnosis": {"en": "astrocytoma", "de": "Astrozytom"},
     "therapy_date": "2022-01-22", "lab_date": "2022-02-07", "early_hb": 14.8, "latest_hb": 13.5},
]


def fixture_text(case: dict[str, Any], kind: str, language: str) -> str:
    diagnosis = case["diagnosis"][language]
    treatment = case["treatment"] if language == "en" else {
        "temozolomide": "Temozolomid", "bevacizumab": "Bevacizumab", "radiotherapy": "Strahlentherapie"
    }[case["treatment"]]
    if language == "en":
        bodies = {
            "oncology": f"Initial diagnosis: {diagnosis}. Diagnosis date: {case['diagnosis_date']}. "
            f"Age at diagnosis: {case['age']} years. Hemoglobin: {case['early_hb']} g/dL. Treatment has not started.",
            "treatment": f"Treatment with {treatment} started today. This is the only cancer-directed treatment "
            "documented in this synthetic timeline.",
            "laboratory": f"Hemoglobin measured today: {case['latest_hb']} g/dL. This is the latest hemoglobin measurement.",
        }
        title = "Synthetic clinical report"
    else:
        bodies = {
            "oncology": f"Erstdiagnose: {diagnosis}. Diagnosedatum: {case['diagnosis_date']}. "
            f"Alter bei Erstdiagnose: {case['age']} Jahre. Hämoglobin: {str(case['early_hb']).replace('.', ',')} g/dL. "
            "Die Behandlung hat noch nicht begonnen.",
            "treatment": f"Heute wurde die Behandlung mit {treatment} begonnen. Dies ist die einzige "
            "tumorgerichtete Behandlung in diesem synthetischen Verlauf.",
            "laboratory": f"Hämoglobin heute: {str(case['latest_hb']).replace('.', ',')} g/dL. "
            "Dies ist die letzte dokumentierte Hämoglobinmessung.",
        }
        title = "Synthetischer klinischer Bericht"
    return f"{title}\n\n{bodies[kind]}\n"


def export_fixtures(destination: Path) -> None:
    if destination.exists():
        raise FileExistsError("Fixture output exists; use a new output directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".synthetic-fixtures-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "fixtures"
        staging.mkdir()
        for variant in ("english", "german", "mixed"):
            root = staging / variant
            root.mkdir()
            registry = []
            gold = []
            for index, case in enumerate(FIXTURE_CASES):
                note_ids = {}
                for position, kind in enumerate(("oncology", "treatment", "laboratory")):
                    language = "en" if variant == "english" else "de"
                    if variant == "mixed":
                        language = ("de", "en", "en")[position] if index % 2 == 0 else ("en", "de", "de")[position]
                    note_id = f"{case['id']}-{kind}"
                    note_ids[kind] = note_id
                    note_date = case[{"oncology": "diagnosis_date", "treatment": "therapy_date", "laboratory": "lab_date"}[kind]]
                    relative = f"notes/{case['id']}/{kind}/{note_date}__{note_id}.txt"
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(fixture_text(case, kind, language), encoding="utf-8")
                    registry.append({"patient_id": case["id"], "note_id": note_id, "report_type": kind,
                                     "date": note_date, "language": language, "path": relative})
                for feature, value, supporting_note in (
                    ("diagnosis_date", case["diagnosis_date"], "oncology"),
                    ("age_at_diagnosis", case["age"], "oncology"),
                    ("treatment_name", case["treatment"], "treatment"),
                    ("latest_hemoglobin", case["latest_hb"], "laboratory"),
                ):
                    gold.append({"patient_id": case["id"], "feature": feature, "value": value,
                                 "evidence_note_ids": [note_ids[supporting_note]]})
            write_registry(root / "registry.csv", registry)
            write_jsonl(root / "gold.jsonl", gold)
            write_manifest(root, {
                "schema_version": 1, "dataset_id": f"fixture_{variant}", "language": variant,
                "patient_count": len(FIXTURE_CASES), "note_count": len(registry), "gold_count": len(gold),
                "provenance": "Purpose-authored synthetic regression cases; no source clinical text or patient data",
                "paired_variants": True,
                "scope": "Regression fixtures, not a held-out clinical performance benchmark",
            })
        staging.rename(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--english-source", type=Path)
    parser.add_argument("--german-source", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixtures-only", action="store_true")
    args = parser.parse_args()
    if not args.fixtures_only and (args.english_source is None or args.german_source is None):
        parser.error("Both --english-source and --german-source are required unless --fixtures-only is set")
    targets = [args.output_root / "fixtures"]
    if not args.fixtures_only:
        targets += [args.output_root / "english", args.output_root / "german"]
    if any(target.exists() for target in targets):
        parser.error("Output datasets already exist; use a new versioned output root")
    summaries = []
    if not args.fixtures_only:
        summaries.append(export_cohort(args.english_source, args.output_root / "english", "en", args.seed))
        summaries.append(export_cohort(args.german_source, args.output_root / "german", "de", args.seed))
    export_fixtures(args.output_root / "fixtures")
    print(json.dumps({"cohorts": summaries, "fixture_variants": ["english", "german", "mixed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
