"""Offline integration checks for the bundled full synthetic cohorts."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import pytest

from oncoraggraph.config.feature_schema import load_feature_specs, validate_feature_specs
from oncoraggraph.config.pipeline_config import load_pipeline_config
from oncoraggraph.ingestion import load_notes
from oncoraggraph.pipeline import run_pipeline


ROOT = Path(__file__).resolve().parents[1]
COHORT_NAMES = {"english": "oncorag-e", "german": "oncorag-d"}
COHORT_FEATURES = {
    "english": {
        "latest_explicit_visit_date": "date",
        "highest_documented_treatment_week": "integer",
        "greatest_documented_functional_limitation": "ordinal",
    },
    "german": {
        "diagnosis_date": "date",
        "prior_radiotherapy_dose": "numeric",
        "recurrent_radiotherapy_dose": "numeric",
        "tumor_laterality": "categorical",
    },
}


def full_config(language):
    return load_pipeline_config(ROOT / "configs" / f"{COHORT_NAMES[language]}.json")


def require_bundled_cohort(config):
    if not Path(config["inputs"]["registry_path"]).parent.exists():
        pytest.skip("Full synthetic cohorts are not included in demo-only releases")


@pytest.mark.parametrize("language", COHORT_FEATURES)
def test_bundled_full_cohort_validation(language):
    config = full_config(language)
    require_bundled_cohort(config)
    specs = load_feature_specs(config["features"]["specifications"])

    assert {spec["name"]: spec["type"] for spec in specs} == COHORT_FEATURES[language]
    assert run_pipeline(config, stage="validate") == {
        "patients": 489,
        "notes": 2930,
        "features": len(COHORT_FEATURES[language]),
    }
    assert config["features"]["configuration_mode"] == "manual"
    assert config["features"]["language"] == language
    assert config["cohort"]["name"] == COHORT_NAMES[language]
    assert config["vector_store"]["collection_namespace"] == COHORT_NAMES[language]
    assert "gold_path" not in config.get("evaluation", {})


@pytest.mark.parametrize("language", COHORT_FEATURES)
def test_full_cohort_folder_and_registry_have_identical_note_inventory(language):
    config = full_config(language)
    require_bundled_cohort(config)
    dataset_root = Path(config["inputs"]["registry_path"]).parent
    registered = load_notes(registry_path=config["inputs"]["registry_path"])
    folder_notes = load_notes(notes_root=dataset_root / "notes")

    def inventory(notes):
        # IDs and language defaults differ between the two supported loaders.
        return Counter(
            (
                note.patient_id,
                note.report_type,
                note.date,
                note.path.relative_to(dataset_root).as_posix(),
                hashlib.sha256(note.text.encode("utf-8")).hexdigest(),
            )
            for note in notes
        )

    assert len(registered) == len(folder_notes) == 2930
    assert len({note.patient_id for note in registered}) == 489
    assert inventory(registered) == inventory(folder_notes)
    assert {note.language for note in registered} == {config["cohort"]["language"]}


@pytest.mark.parametrize("language", COHORT_NAMES)
def test_public_cohort_identifiers_match_paths_labels_and_report_headers(language):
    config = full_config(language)
    require_bundled_cohort(config)
    cohort = COHORT_NAMES[language]
    root = Path(config["inputs"]["registry_path"]).parent
    notes = load_notes(registry_path=config["inputs"]["registry_path"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    labels = {row["note_id"]: row for row in (
        json.loads(line) for line in (root / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    )}
    assert root.name == manifest["dataset_id"] == cohort
    assert set(labels) == {note.note_id for note in notes}
    for note in notes:
        assert re.fullmatch(rf"{cohort}-\d{{4}}", note.patient_id)
        assert re.fullmatch(rf"{cohort}-note-\d{{5}}", note.note_id)
        assert labels[note.note_id]["patient_id"] == note.patient_id
        assert labels[note.note_id]["date"] == note.date
        assert note.path.name == f"{note.date}__{note.note_id}.txt"
        assert not re.search(r"SYN-(?:TNBC|RICCI)-|ricci_syn_", note.text)
        if language == "german":
            assert f"Dokument-ID: {note.note_id}" in note.text
            assert f"Patient {note.patient_id}" in note.text


def test_full_cohorts_and_demo_use_distinct_namespaces():
    configs = [full_config(language) for language in COHORT_FEATURES]
    configs.extend(
        load_pipeline_config(ROOT / "configs" / f"oncorag_synthetic_{language}.json")
        for language in ("english", "german", "mixed")
    )

    for section, key in (
        ("cohort", "name"),
        ("vector_store", "collection_namespace"),
        ("features", "generated_config_dir"),
        ("outputs", "root"),
    ):
        assert len({config[section][key] for config in configs}) == len(configs), (section, key)

    for cache_key in ("graph_cache_dir", "chroma_cache_dir", "prompt_cache_dir"):
        assert len({
            Path(config["outputs"]["root"]) / config["outputs"][cache_key]
            for config in configs
        }) == len(configs), cache_key


@pytest.mark.parametrize("language", COHORT_FEATURES)
def test_full_cohort_feature_configs_generate_locally(tmp_path, language):
    config = full_config(language)
    require_bundled_cohort(config)
    generated_dir = tmp_path / "generated"
    config["features"]["generated_config_dir"] = str(generated_dir)
    config["outputs"]["root"] = str(tmp_path / "outputs")
    specs = load_feature_specs(config["features"]["specifications"])

    result = run_pipeline(config, stage="config")

    assert set(result["features"]) == set(COHORT_FEATURES[language])
    assert {path.name for path in generated_dir.iterdir()} == (
        {f"{spec['name']}.json" for spec in specs} | {"generation_manifest.json"}
    )
    for spec in specs:
        generated = json.loads((generated_dir / f"{spec['name']}.json").read_text(encoding="utf-8"))
        assert validate_feature_specs([generated["feature"]]) == [spec]
        assert generated["language"] == language
        assert generated["config_generation"] == {"mode": "manual", "ontology_enriched": False}
        assert generated["top_cuis"] == []
        assert generated["rules"]["questions"]
    assert not Path(config["outputs"]["root"]).exists()
