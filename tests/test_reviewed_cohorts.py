import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GERMAN_INSTITUTION = "Universitaetsklinikum Beispielstadt\n"
COHORT_NAMES = {"english": "oncorag-e", "german": "oncorag-d"}


@pytest.fixture
def reviewed(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "reviewed_cohorts_test", ROOT / "scripts/prepare_reviewed_cohorts.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fictional_sources(tmp_path, monkeypatch, reviewed):
    sources = {}
    fingerprints = {}
    for variant, language, patient in (
        ("english", "en", "SYN-TNBC-0001"),
        ("german", "de", "SYN-RICCI-0001"),
    ):
        source = tmp_path / "fictional-inputs" / variant
        (source / "synthetic_notes").mkdir(parents=True)
        (source / "labels").mkdir()
        note_id = f"fictional_{language}_0001"
        note = f"Patient {patient}. Document {note_id}. A fictional report describes fatigue.\n"
        if variant == "german":
            note = GERMAN_INSTITUTION + f"Patient {patient}. Dokument {note_id}. Ein erfundener Bericht beschreibt Fatigue.\n"
        (source / "synthetic_notes" / f"{note_id}.txt").write_text(note, encoding="utf-8")
        payload = {
            "patient_id": patient,
            "note_id": note_id,
            "note_type": "oncology",
            "note_date": "2020-01-01",
            "source_style_patient_id": "FICTIONAL-SOURCE-24681357",
            "history": {"private": "FICTIONAL-HISTORY-EXCLUDE"},
            "events": [{
                "ctcae_term": "Fatigue", "grade": 2, "temporality": "current",
                "source_grade_evidence": ["FICTIONAL-EVIDENCE-EXCLUDE"],
            }],
        }
        (source / "labels" / f"{note_id}.json").write_text(json.dumps(payload), encoding="utf-8")
        projected = tmp_path / "fictional-projections" / variant
        reviewed.export_cohort(source, projected, language, _preserve_source_ids=True)
        manifest = json.loads((projected / "manifest.json").read_text(encoding="utf-8"))
        fingerprints[variant] = reviewed.projection_fingerprint(manifest["files"])
        sources[variant] = source
    monkeypatch.setattr(reviewed, "REVIEWED_INPUTS", fingerprints)
    monkeypatch.setattr(reviewed, "REVIEWED_HEADER_SHA256", hashlib.sha256(GERMAN_INSTITUTION.rstrip("\n").encode()).hexdigest())
    return sources


def prepare(reviewed, sources, output):
    return reviewed.prepare_reviewed_cohorts(sources["english"], sources["german"], output)


@pytest.mark.parametrize("variant", ["english", "german"])
def test_changed_input_projection_blocks_both_outputs(reviewed, fictional_sources, tmp_path, variant):
    note = next((fictional_sources[variant] / "synthetic_notes").glob("*.txt"))
    note.write_text(note.read_text(encoding="utf-8") + "Changed fictional text.\n", encoding="utf-8")
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="Input differs from the audited cohort"):
        prepare(reviewed, fictional_sources, output)
    assert not output.exists()


def test_unexpected_source_header_blocks_both_outputs(reviewed, fictional_sources, tmp_path, monkeypatch):
    monkeypatch.setattr(reviewed, "REVIEWED_HEADER_SHA256", hashlib.sha256(b"Different fictional header").hexdigest())
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="Unexpected German note header"):
        prepare(reviewed, fictional_sources, output)
    assert not output.exists()


@pytest.mark.parametrize("language", ["en", "de"])
def test_projected_payload_tampering_blocks_both_outputs(
    reviewed, fictional_sources, tmp_path, monkeypatch, language,
):
    export = reviewed.export_cohort

    def tampered_export(source, destination, selected_language, seed=42, **kwargs):
        result = export(source, destination, selected_language, seed, **kwargs)
        if selected_language == language:
            note = next((destination / "notes").rglob("*.txt"))
            note.write_text(note.read_text(encoding="utf-8") + "Unreviewed fictional text.\n", encoding="utf-8")
        return result

    monkeypatch.setattr(reviewed, "export_cohort", tampered_export)
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="Projected file differs from its reviewed manifest"):
        prepare(reviewed, fictional_sources, output)
    assert not output.exists()


def test_reviewed_outputs_have_notices_and_only_public_labels(reviewed, fictional_sources, tmp_path):
    source_contents = {
        path: path.read_bytes()
        for source in fictional_sources.values()
        for path in source.rglob("*") if path.is_file()
    }
    output = tmp_path / "release"
    summaries = prepare(reviewed, fictional_sources, output)
    assert set(summaries) == {"oncorag-e", "oncorag-d"}
    for variant, cohort in COHORT_NAMES.items():
        root = output / cohort
        assert {key: summaries[cohort][key] for key in ("patient_count", "note_count", "event_count")} == {
            "patient_count": 1, "note_count": 1, "event_count": 1,
        }
        note = next((root / "notes").rglob("*.txt")).read_text(encoding="utf-8")
        original = next((fictional_sources[variant] / "synthetic_notes").glob("*.txt")).read_text(encoding="utf-8")
        source_label = json.loads(next((fictional_sources[variant] / "labels").glob("*.json")).read_text(encoding="utf-8"))
        original = original.replace(source_label["patient_id"], f"{cohort}-0001").replace(
            source_label["note_id"], f"{cohort}-note-00001",
        )
        if variant == "english":
            assert note == "SYNTHETIC REPORT - not a real patient or clinical record.\n\n" + original
        else:
            assert note == (
                "SYNTHETISCHER BERICHT - kein realer Patient, keine reale Klinik.\n\n"
                + original.removeprefix(GERMAN_INSTITUTION)
            )
            assert GERMAN_INSTITUTION.strip() not in note
        labels = [json.loads(line) for line in (root / "labels.jsonl").read_text().splitlines()]
        assert len(labels) == 1
        assert set(labels[0]) == {"patient_id", "note_id", "report_type", "date", "language", "events"}
        assert labels[0]["events"] == [{
            "event_id": "event_001", "ctcae_term": "Fatigue", "temporality": "current", "grade": 2,
        }]
        public_text = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*") if path.is_file())
        for forbidden in (
            "FICTIONAL-SOURCE-24681357", "FICTIONAL-HISTORY-EXCLUDE", "FICTIONAL-EVIDENCE-EXCLUDE",
            "source_style_patient_id", "source_grade_evidence",
        ):
            assert forbidden not in public_text
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["dataset_id"] == cohort
        assert labels[0]["patient_id"] == f"{cohort}-0001"
        assert labels[0]["note_id"] == f"{cohort}-note-00001"
        assert manifest["provenance"]["text_changed"] is True
        assert manifest["review"]["input_projection_sha256"] == reviewed.REVIEWED_INPUTS[variant]
        for key in ("source_patient_metadata_included", "upstream_evidence_snippets_included", "clinical_grade_validation"):
            assert manifest["review"][key] is False
    assert all(path.read_bytes() == contents for path, contents in source_contents.items())


def test_reviewed_manifests_are_deterministic_complete_and_exclude_self(reviewed, fictional_sources, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    assert prepare(reviewed, fictional_sources, first) == prepare(reviewed, fictional_sources, second)
    for cohort in COHORT_NAMES.values():
        root = first / cohort
        manifest_path = root / "manifest.json"
        assert manifest_path.read_bytes() == (second / cohort / "manifest.json").read_bytes()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
        assert "manifest.json" not in manifest["files"]
        assert files == set(manifest["files"]) | {"manifest.json"}
        assert manifest["payload_bytes"] == sum(item["bytes"] for item in manifest["files"].values())
        for relative, expected in manifest["files"].items():
            contents = (root / relative).read_bytes()
            assert expected == {"bytes": len(contents), "sha256": hashlib.sha256(contents).hexdigest()}
            assert contents == (second / cohort / relative).read_bytes()


@pytest.mark.parametrize("variant", ["english", "german"])
def test_existing_cohort_output_is_preserved(reviewed, fictional_sources, tmp_path, variant):
    output = tmp_path / "release"
    existing = output / COHORT_NAMES[variant]
    existing.mkdir(parents=True)
    marker = existing / "keep.txt"
    marker.write_text("Existing fictional output.\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Cohort output exists"):
        prepare(reviewed, fictional_sources, output)
    assert marker.read_text(encoding="utf-8") == "Existing fictional output.\n"
    assert set(output.iterdir()) == {existing}


@pytest.mark.parametrize("variant,nested", [("english", False), ("german", True)])
def test_output_cannot_be_inside_either_source(reviewed, fictional_sources, variant, nested):
    source = fictional_sources[variant]
    output = source / "reviewed" if nested else source
    with pytest.raises(ValueError, match="cannot be inside an input source"):
        prepare(reviewed, fictional_sources, output)
    if nested:
        assert not output.exists()
