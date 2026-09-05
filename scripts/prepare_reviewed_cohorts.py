#!/usr/bin/env python3
"""Recreate the reviewed public cohorts from the exact audited local inputs.

The fingerprints cover projected notes, relative registry, allowlisted labels,
and patient splits, never the private upstream label metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from export_synthetic_datasets import export_cohort, write_manifest


REVIEWED_INPUTS = {
    "english": "be2737652e94d01e37f6d0d4c318cb35900216fe3579b5674faedc5d1c7e044a",
    "german": "def4936409b0b7aaf74c4046b920797f915a22f67350d149f22c12ab2b2106da",
}
REVIEW_ID = "oncorag-cohort-provenance-v1"
REVIEWED_HEADER_SHA256 = "cd98e541ff63e377b598c6322c26ca48972ff4eee52899f06d01f9a03146bb84"


def projection_fingerprint(files: dict) -> str:
    return hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def mark_reviewed(root: Path, variant: str) -> dict:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if projection_fingerprint(manifest["files"]) != REVIEWED_INPUTS[variant]:
        raise ValueError("Input differs from the audited cohort; a new provenance review is required")
    for relative, expected in manifest["files"].items():
        path = root / relative
        contents = path.read_bytes()
        if len(contents) != expected["bytes"] or hashlib.sha256(contents).hexdigest() != expected["sha256"]:
            raise ValueError("Projected file differs from its reviewed manifest")
    for note in sorted((root / "notes").rglob("*.txt")):
        text = note.read_text(encoding="utf-8")
        if variant == "german":
            header, separator, body = text.partition("\n")
            if not separator or hashlib.sha256(header.encode("utf-8")).hexdigest() != REVIEWED_HEADER_SHA256:
                raise ValueError("Unexpected German note header")
            text = body
            notice = "SYNTHETISCHER BERICHT - kein realer Patient, keine reale Klinik.\n\n"
        else:
            notice = "SYNTHETIC REPORT - not a real patient or clinical record.\n\n"
        note.write_text(notice + text, encoding="utf-8")
    manifest["provenance"].update({
        "text_changed": True,
        "text_changes": ["Explicit synthetic notice added"] + (
            ["Real institution header removed"] if variant == "german" else []
        ),
        "license_status": "See repository LICENSE and the dataset provenance notice for terminology attribution",
    })
    manifest["review"] = {
        "id": REVIEW_ID,
        "scope": "Technical provenance and projected-payload review; not clinical validation or legal certification",
        "input_projection_sha256": REVIEWED_INPUTS[variant],
        "source_patient_metadata_included": False,
        "upstream_evidence_snippets_included": False,
        "clinical_grade_validation": False,
    }
    manifest["known_limitations"] = [
        "Template-based cases are not representative clinical populations",
        "Note-level CTCAE labels are not gold answers for arbitrary patient features",
        "No claim of expert-validated grades or fully consistent longitudinal trajectories",
    ]
    if variant == "german":
        manifest["known_limitations"].append(
            "1498 upstream events had supplementary evidence absent from the note; all upstream evidence snippets are omitted"
        )
    manifest.pop("files")
    manifest.pop("payload_bytes")
    manifest_path.unlink()
    write_manifest(root, manifest)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def prepare_reviewed_cohorts(english_source: Path, german_source: Path, output_root: Path) -> dict:
    output_root = output_root.absolute()
    variants = {"english": (english_source, "en"), "german": (german_source, "de")}
    if any((output_root / variant).exists() for variant in variants):
        raise FileExistsError("Cohort output exists; use a new versioned output directory")
    for source, _ in variants.values():
        if output_root.resolve() == source.resolve() or source.resolve() in output_root.resolve().parents:
            raise ValueError("Output datasets cannot be inside an input source")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reviewed-cohorts-", dir=output_root.parent) as temporary:
        staging = Path(temporary)
        summaries = {}
        for variant, (source, language) in variants.items():
            export_cohort(source, staging / variant, language, seed=42)
            manifest = mark_reviewed(staging / variant, variant)
            summaries[variant] = {
                key: manifest[key] for key in ("patient_count", "note_count", "event_count", "payload_bytes")
            }
        # Both cohorts must pass review before either becomes a release artifact.
        output_root.mkdir(parents=True, exist_ok=True)
        for variant in variants:
            (staging / variant).rename(output_root / variant)
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--english-source", type=Path, required=True)
    parser.add_argument("--german-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_reviewed_cohorts(args.english_source, args.german_source, args.output_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
