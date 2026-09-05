import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("synthetic_export", ROOT / "scripts/export_synthetic_datasets.py")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def write_source(root, note_text="Synthetic report. Hemoglobin 12.1 g/dL.", **overrides):
    (root / "synthetic_notes").mkdir(parents=True)
    (root / "labels").mkdir()
    payload = {
        "patient_id": "SYN-TNBC-0001", "note_id": "syn_0001", "note_type": "oncology",
        "note_date": "2020-01-01", "source_style_patient_id": "987654321",
        "history": {"private": "must never be exported"},
        "events": [{"ctcae_term": "Fatigue", "grade": 2, "negated": False,
                    "temporality": "current", "source_grade_evidence": ["private payload"]}],
        **overrides,
    }
    (root / "synthetic_notes" / f"{payload['note_id']}.txt").write_text(note_text, encoding="utf-8")
    (root / "labels" / f"{payload['note_id']}.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def test_export_allowlists_labels_and_preserves_note_metadata(tmp_path):
    source = write_source(tmp_path / "source")
    destination = tmp_path / "public"
    exporter.export_cohort(source, destination, "en")
    labels = [json.loads(line) for line in (destination / "labels.jsonl").read_text().splitlines()]
    assert labels == [{
        "patient_id": "SYN-TNBC-0001", "note_id": "syn_0001", "report_type": "oncology",
        "date": "2020-01-01", "language": "en",
        "events": [{"event_id": "event_001", "ctcae_term": "Fatigue", "temporality": "current",
                    "grade": 2, "negated": False}],
    }]
    public_text = "\n".join(path.read_text() for path in destination.rglob("*") if path.is_file())
    for private_value in ("987654321", "must never be exported", "private payload", "source_style_patient_id"):
        assert private_value not in public_text
    with (destination / "registry.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["path"] == "notes/SYN-TNBC-0001/oncology/2020-01-01__syn_0001.txt"
    assert (destination / row["path"]).read_text() == (source / "synthetic_notes/syn_0001.txt").read_text()


@pytest.mark.parametrize("text", [
    "Record 987654321 contains synthetic content.",
    "Record prefix987654321suffix contains synthetic content.",
    "Contact person@example.org.",
    "MRN: 12345678.",
    "Source file /home/person/notes.txt.",
])
def test_export_rejects_source_identifiers_and_contact_markers(tmp_path, text):
    source = write_source(tmp_path / "source", note_text=text)
    with pytest.raises(ValueError, match="review required"):
        exporter.export_cohort(source, tmp_path / "public", "en")
    assert not (tmp_path / "public").exists()


@pytest.mark.parametrize("overrides", [
    {"patient_id": "SYN-TNBC-987654321"},
    {"note_id": "syn987654321suffix"},
    {"note_type": "oncology987654321suffix"},
    {"source_style_patient_id": "syn_0001.txt"},
])
def test_export_rejects_source_identifiers_in_metadata_and_paths(tmp_path, overrides):
    source = write_source(tmp_path / "source", **overrides)
    with pytest.raises(ValueError, match="Known source identifier"):
        exporter.export_cohort(source, tmp_path / "public", "en")
    assert not (tmp_path / "public").exists()


@pytest.mark.parametrize("field,value,identifier", [
    ("source_style_patient_id", "FICTIONAL-ID-24681357", "FICTIONAL-ID-24681357"),
    ("source_style_patient_ids", ["FICTIONAL-ID-24681357"], "FICTIONAL-ID-24681357"),
    ("style_source_patient_ids", ["FICTIONAL-ID-24681357"], "FICTIONAL-ID-24681357"),
    ("sampled_source_patient_ids", [246813579], "246813579"),
    ("style_patient_ids", [None, "", "style_unknown", "FICTIONAL-ID-24681357"], "FICTIONAL-ID-24681357"),
])
def test_export_rejects_nested_source_identifier_fields(tmp_path, field, value, identifier):
    source = write_source(
        tmp_path / "source", note_text=f"Synthetic note contains prefix{identifier}suffix.",
        history={"sources": [{field: value}]},
    )
    with pytest.raises(ValueError, match="Known source identifier"):
        exporter.export_cohort(source, tmp_path / "public", "en")
    assert not (tmp_path / "public").exists()


@pytest.mark.parametrize("source_identifier", [None, "", "style_unknown", "  ", " style_unknown "])
def test_export_does_not_treat_synthetic_metadata_or_empty_ids_as_source_ids(tmp_path, source_identifier):
    source = write_source(
        tmp_path / "source", note_text="Synthetic patient SYN-TNBC-0001, note syn_0001.",
        source_style_patient_id=source_identifier,
        history={"patient_id": "SYN-TNBC-0001", "note_id": "syn_0001"},
    )
    destination = tmp_path / "public"
    exporter.export_cohort(source, destination, "en")
    assert (destination / "registry.csv").is_file()


def test_export_rejects_symlink_escape(tmp_path):
    source = write_source(tmp_path / "source")
    note = source / "synthetic_notes/syn_0001.txt"
    note.unlink()
    outside = tmp_path / "outside.txt"
    outside.write_text("Do not read outside the dataset")
    note.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes its root"):
        exporter.export_cohort(source, tmp_path / "public", "en")


def test_export_rejects_path_in_metadata(tmp_path):
    source = write_source(tmp_path / "source", note_type="../../outside")
    with pytest.raises(ValueError, match="report_type"):
        exporter.export_cohort(source, tmp_path / "public", "en")


def test_export_rejects_non_synthetic_patient_identifier(tmp_path):
    source = write_source(tmp_path / "source", patient_id="1234567")
    with pytest.raises(ValueError, match="synthetic prefix"):
        exporter.export_cohort(source, tmp_path / "public", "en")


def test_same_day_reports_are_not_overwritten(tmp_path):
    source = write_source(tmp_path / "source")
    second = json.loads((source / "labels/syn_0001.json").read_text())
    second["note_id"] = "syn_0002"
    (source / "labels/syn_0002.json").write_text(json.dumps(second))
    (source / "synthetic_notes/syn_0002.txt").write_text("A second synthetic report on the same day.")
    destination = tmp_path / "public"
    exporter.export_cohort(source, destination, "en")
    assert len(list((destination / "notes").rglob("*.txt"))) == 2


def test_export_refuses_overwrite(tmp_path):
    source = write_source(tmp_path / "source")
    destination = tmp_path / "public"
    exporter.export_cohort(source, destination, "en")
    original = (destination / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        exporter.export_cohort(source, destination, "en")
    assert (destination / "manifest.json").read_bytes() == original


def test_patient_splits_are_reproducible_and_patient_disjoint():
    ids = [f"SYN-TNBC-{index:04d}" for index in range(489)]
    splits = exporter.patient_splits(ids)
    assert splits == exporter.patient_splits(list(reversed(ids)) + ids)
    assert set(splits) == set(ids)
    assert {name: list(splits.values()).count(name) for name in ("train", "dev", "test")} == {
        "train": 342, "dev": 73, "test": 74,
    }


def test_mixed_demo_has_both_languages_with_matching_gold(tmp_path):
    destination = tmp_path / "demo"
    exporter.export_demo(destination)
    reference_gold = None
    for variant in ("english", "german", "mixed"):
        root = destination / variant
        gold = [json.loads(line) for line in (root / "gold.jsonl").read_text().splitlines()]
        if reference_gold is None:
            reference_gold = gold
        assert gold == reference_gold
        with (root / "registry.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 9
        for patient_id in {row["patient_id"] for row in rows}:
            patient_rows = [row for row in rows if row["patient_id"] == patient_id]
            languages = {row["language"] for row in patient_rows}
            assert languages == ({"en", "de"} if variant == "mixed" else {"en" if variant == "english" else "de"})
            latest = max(patient_rows, key=lambda row: row["date"])
            if variant == "mixed":
                earliest = min(patient_rows, key=lambda row: row["date"])
                assert earliest["language"] != latest["language"]
            lab_gold = next(row for row in gold if row["patient_id"] == patient_id and row["feature"] == "latest_hemoglobin")
            assert lab_gold["evidence_note_ids"] == [latest["note_id"]]
        for row in gold:
            assert set(row["evidence_note_ids"]).issubset({note["note_id"] for note in rows if note["patient_id"] == row["patient_id"]})


@pytest.mark.parametrize("variant", ["english", "german", "mixed"])
def test_committed_demo_manifests_are_complete_and_match_hashes(variant):
    root = ROOT / "examples/datasets/demo" / variant
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["dataset_id"] == f"demo_{variant}"
    files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    assert files == set(manifest["files"]) | {"manifest.json"}
    for relative, expected in manifest["files"].items():
        path = exporter.safe_child(root, relative)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"]
        assert path.stat().st_size == expected["bytes"]


@pytest.mark.parametrize("option", ["--demo-only", "--fixtures-only"])
def test_demo_only_cli_reproduces_bundled_examples_without_source_data(tmp_path, option):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/export_synthetic_datasets.py"),
         option, "--output-root", str(tmp_path)],
        check=True, capture_output=True, text=True,
    )
    assert json.loads(result.stdout) == {
        "cohorts": [], "demo_variants": ["english", "german", "mixed"],
    }
    assert {path.name for path in tmp_path.iterdir()} == {"demo"}
    for variant in ("english", "german", "mixed"):
        bundled = ROOT / "examples/datasets/demo" / variant
        generated = tmp_path / "demo" / variant
        expected_files = {path.relative_to(bundled) for path in bundled.rglob("*") if path.is_file()}
        actual_files = {path.relative_to(generated) for path in generated.rglob("*") if path.is_file()}
        assert actual_files == expected_files
        for relative in expected_files:
            assert (generated / relative).read_bytes() == (bundled / relative).read_bytes()


@pytest.mark.parametrize("variant,events", [("english", 5761), ("german", 5987)])
def test_full_export_manifest_counts_and_patient_split_integrity(variant, events):
    root = ROOT / "examples/datasets" / variant
    if not root.exists():
        pytest.skip("Full synthetic cohort is an optional release artifact")
    manifest = json.loads((root / "manifest.json").read_text())
    with (root / "registry.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    patients = {row["patient_id"] for row in rows}
    splits = json.loads((root / "splits.json").read_text())
    assert len(rows) == manifest["note_count"] == 2930
    assert len(patients) == manifest["patient_count"] == 489
    assert manifest["event_count"] == events
    assert set(splits["patients"]) == patients
    assert len({row["path"] for row in rows}) == len(rows)
    assert all(exporter.safe_child(root, row["path"]).is_file() for row in rows)
    labels = [json.loads(line) for line in (root / "labels.jsonl").read_text().splitlines()]
    assert len(labels) == len(rows)
    assert sum(len(row["events"]) for row in labels) == events
    for label in labels:
        assert set(label) == {"patient_id", "note_id", "report_type", "date", "language", "events"}
        for event in label["events"]:
            assert set(event) <= {"event_id", "ctcae_term", "grade_status", "temporality", "event_status", "grade", "negated"}
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    assert actual_files == set(manifest["files"]) | {"manifest.json"}
    for relative, expected in manifest["files"].items():
        path = exporter.safe_child(root, relative)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"]
        assert path.stat().st_size == expected["bytes"]
