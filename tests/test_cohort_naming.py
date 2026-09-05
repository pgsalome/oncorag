import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cohort_naming_export", ROOT / "scripts/export_synthetic_datasets.py")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def write_source(root, language, note_ids=("source_00012", "source_00027"), reverse=False):
    (root / "synthetic_notes").mkdir(parents=True)
    (root / "labels").mkdir()
    prefix = "SYN-TNBC" if language == "en" else "SYN-RICCI"
    patient_ids = [f"{prefix}-{index:03d}" for index in (1, 14)]
    records = []
    for index, (patient, note) in enumerate(zip(patient_ids, note_ids)):
        body = (
            f"Patient {patient}. Document-ID: {note}.\n"
            f"Hemoglobin: 12.{index} g/dL. Treatment started on 2020-01-01.\n"
            f"Unrelated tokens: prefix{patient}suffix and {note}_extension.\n"
        )
        payload = {
            "patient_id": patient, "note_id": note, "note_type": "oncology",
            "note_date": "2020-01-01", "events": [{
                "ctcae_term": "Fatigue", "grade": index + 1,
                "temporality": "current", "negated": False,
            }],
        }
        records.append((note, body, payload))
    for note, body, payload in reversed(records) if reverse else records:
        (root / "synthetic_notes" / f"{note}.txt").write_text(body, encoding="utf-8")
        (root / "labels" / f"{note}.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def snapshot(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def registry_rows(root):
    with (root / "registry.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("language,cohort", [("en", "oncorag-e"), ("de", "oncorag-d")])
def test_public_naming_preserves_clinical_content_labels_dates_and_splits(tmp_path, language, cohort):
    source = write_source(tmp_path / "source", language)
    original_source = snapshot(source)
    root = tmp_path / "projected"
    exporter.export_cohort(source, root, language, _preserve_source_ids=True)
    original = snapshot(root)
    old_rows = registry_rows(root)
    old_labels = [json.loads(line) for line in original["labels.jsonl"].decode().splitlines()]
    old_splits = json.loads(original["splits.json"])
    manifest = exporter.rename_public_cohort(root, language)
    rows = registry_rows(root)
    labels = [json.loads(line) for line in (root / "labels.jsonl").read_text().splitlines()]
    splits = json.loads((root / "splits.json").read_text())

    assert manifest["dataset_id"] == cohort
    assert manifest["public_naming"]["version"] == 1
    assert manifest["patient_count"] == len(rows) == 2
    assert manifest["note_count"] == manifest["event_count"] == 2
    for old, row, old_label, label in zip(old_rows, rows, old_labels, labels):
        patient = f"{cohort}-{int(old['patient_id'].rsplit('-', 1)[1]):04d}"
        note = f"{cohort}-note-{int(old['note_id'].rsplit('_', 1)[1]):05d}"
        assert row == {**old, "patient_id": patient, "note_id": note,
                       "path": f"notes/{patient}/oncology/{old['date']}__{note}.txt"}
        assert label == {**old_label, "patient_id": patient, "note_id": note}
        assert splits["patients"][patient] == old_splits["patients"][old["patient_id"]]
        expected = original[old["path"]].decode().replace(
            f"Patient {old['patient_id']}.", f"Patient {patient}.",
        ).replace(f"Document-ID: {old['note_id']}.", f"Document-ID: {note}.")
        assert (root / row["path"]).read_text() == expected
        assert not (root / old["path"]).exists()
    assert {key: value for key, value in splits.items() if key != "patients"} == {
        key: value for key, value in old_splits.items() if key != "patients"
    }
    assert snapshot(source) == original_source
    for relative, expected in manifest["files"].items():
        contents = (root / relative).read_bytes()
        assert expected == {"bytes": len(contents), "sha256": hashlib.sha256(contents).hexdigest()}
    before_repeat = snapshot(root)
    assert exporter.rename_public_cohort(root, language) == manifest
    assert snapshot(root) == before_repeat


@pytest.mark.parametrize("note_ids", [("zulu", "alpha"), ("source_0001", "another_1")])
def test_non_numeric_or_colliding_note_suffixes_use_stable_sorted_ordinals(tmp_path, note_ids):
    for name, reverse in (("first", False), ("second", True)):
        source = write_source(tmp_path / f"source-{name}", "en", note_ids, reverse)
        exporter.export_cohort(source, tmp_path / name, "en")
    assert snapshot(tmp_path / "first") == snapshot(tmp_path / "second")
    assert [row["note_id"] for row in registry_rows(tmp_path / "first")] == [
        "oncorag-e-note-00001", "oncorag-e-note-00002",
    ]


def test_naming_rejects_changed_payload_without_editing_anything(tmp_path):
    source = write_source(tmp_path / "source", "en")
    root = tmp_path / "projected"
    exporter.export_cohort(source, root, "en", _preserve_source_ids=True)
    note = next((root / "notes").rglob("*.txt"))
    note.write_text(note.read_text() + "Changed text.\n")
    before = snapshot(root)
    with pytest.raises(ValueError, match="Dataset file differs from its manifest"):
        exporter.rename_public_cohort(root, "en")
    assert snapshot(root) == before


def test_wrong_language_rejected_before_naming(tmp_path):
    source = write_source(tmp_path / "source", "en")
    root = tmp_path / "projected"
    exporter.export_cohort(source, root, "en", _preserve_source_ids=True)
    before = snapshot(root)
    with pytest.raises(ValueError, match="Dataset language"):
        exporter.rename_public_cohort(root, "de")
    assert snapshot(root) == before


def test_patient_and_note_identifier_collision_is_rejected(tmp_path):
    source = write_source(tmp_path / "source", "en", ("SYN-TNBC-001", "source_00027"))
    root = tmp_path / "projected"
    exporter.export_cohort(source, root, "en", _preserve_source_ids=True)
    before = snapshot(root)
    with pytest.raises(ValueError, match="Patient and note identifiers must be distinct"):
        exporter.rename_public_cohort(root, "en")
    assert snapshot(root) == before


def test_export_cli_uses_public_cohort_directory_names(tmp_path, monkeypatch, capsys):
    english = write_source(tmp_path / "english-source", "en")
    german = write_source(tmp_path / "german-source", "de")
    output = tmp_path / "datasets"
    monkeypatch.setattr("sys.argv", [
        "export_synthetic_datasets.py", "--english-source", str(english),
        "--german-source", str(german), "--output-root", str(output),
    ])
    assert exporter.main() == 0
    summaries = json.loads(capsys.readouterr().out)
    assert [summary["dataset_id"] for summary in summaries["cohorts"]] == ["oncorag-e", "oncorag-d"]
    assert {path.name for path in output.iterdir()} == {"oncorag-e", "oncorag-d", "demo"}
