"""Synthetic scoring keeps failed cases and preserves patient-level uncertainty."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from oncoraggraph.config.feature_schema import load_feature_specs
from oncoraggraph.config.pipeline_config import load_pipeline_config
from oncoraggraph.evaluation import evaluate_results, load_records, main, write_experiment_configs


ROOT = Path(__file__).resolve().parents[1]
CATEGORY = {"name": "category", "type": "categorical", "expected_range": ["a", "b"], "complexity": "simple"}


def row(patient, value, **extra):
    return {"patient_id": patient, "feature": "category", "value": value, **extra}


def test_missing_predictions_and_wrong_values_penalize_f1():
    gold = [row("p1", "a"), row("p2", "b"), row("p3", "a")]
    predictions = [row("p1", "a", status="ok", confidence="High"), row("p2", "a", status="ok", confidence="Low")]
    report = evaluate_results({"results": predictions}, gold, feature_specs=[CATEGORY], resamples=20)
    assert report["expected_predictions"] == 3
    assert report["received_predictions"] == 2
    assert report["exact_match"] == pytest.approx(1 / 3)
    assert report["categorical_macro_f1"] == pytest.approx(.25)
    assert report["status_counts"]["missing_prediction"] == 1
    assert report["confidence_groups"]["High"]["accuracy"] == 1
    assert report["confidence_groups"]["Low"]["accuracy"] == 0
    assert report["complexity_strata"]["simple"]["count"] == 3


def test_errors_and_invalid_values_cannot_score_as_correct_nulls():
    gold = [row("p1", None), row("p2", None), row("p3", None)]
    predictions = [row("p1", None, status="error"), row("p2", None, status="invalid"), row("p3", None, status="missing")]
    report = evaluate_results(predictions, gold, feature_specs=[CATEGORY], resamples=10)
    assert report["exact_match"] == pytest.approx(1 / 3)
    assert report["status_rates"]["error"] == pytest.approx(1 / 3)
    assert report["status_rates"]["invalid"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("prediction", ["52", 52.0, True])
def test_evaluation_does_not_coerce_wrong_output_types(prediction):
    spec = {"name": "category", "type": "integer", "expected_range": [0, 120]}
    report = evaluate_results([row("p1", prediction, status="ok")], [row("p1", 52)], feature_specs=[spec], resamples=5)
    assert report["exact_match"] == 0
    assert report["status_counts"]["invalid"] == 1


def test_boolean_integer_equality_does_not_hide_type_mismatch_without_specs():
    report = evaluate_results([row("p1", 1, status="ok")], [row("p1", True)], resamples=5)
    assert report["exact_match"] == 0
    assert report["categorical_macro_f1"] is None


@pytest.mark.parametrize("predictions,gold", [
    ([row("p1", "a", status="ok")] * 2, [row("p1", "a")]),
    ([], [row("p1", "a")] * 2),
    ([row("p2", "a", status="ok")], [row("p1", "a")]),
    ([row(1, "a", status="ok")], [row("1", "a")]),
    ([], [row(" p1", "a")]),
])
def test_bad_join_keys_are_rejected(predictions, gold):
    with pytest.raises(ValueError):
        evaluate_results(predictions, gold, resamples=5)


def test_bootstrap_samples_whole_patients_and_is_seeded():
    gold, predictions = [], []
    for patient, correct in (("p1", True), ("p2", False), ("p3", True)):
        for feature in ("one", "two"):
            gold.append({"patient_id": patient, "feature": feature, "value": "yes"})
            predictions.append({"patient_id": patient, "feature": feature, "value": "yes" if correct else "no", "status": "ok"})
    report = evaluate_results(predictions, gold, seed=12, resamples=100)
    repeated = evaluate_results(predictions, gold, seed=12, resamples=100)
    assert report == repeated
    rng = np.random.default_rng(12)
    values = [np.mean(np.array([1, 0, 1])[rng.integers(0, 3, size=3)]) for _ in range(100)]
    interval = report["confidence_intervals"]["exact_match"]
    assert [interval["low"], interval["high"]] == pytest.approx(np.quantile(values, [.025, .975]))
    assert report["bootstrap"]["unit"] == "patient"


@pytest.mark.parametrize("language", ["english", "german", "mixed"])
def test_demo_gold_has_perfect_score_when_predicted_exactly(language):
    gold = load_records(ROOT / "examples" / "datasets" / "demo" / language / "gold.jsonl")
    predictions = [{**record, "status": "ok", "confidence": "High"} for record in gold]
    report = evaluate_results(predictions, gold, feature_specs=load_feature_specs(ROOT / "examples" / "features.synthetic.yaml"), resamples=15)
    assert report["exact_match"] == 1
    assert report["categorical_macro_f1"] == 1
    assert report["confidence_intervals"]["categorical_macro_f1"]["low"] == 1


def test_rare_class_rule_recomputes_metrics_and_reports_denominator():
    gold = [row(f"p{i}", "a" if i < 9 else "b") for i in range(10)]
    predictions = [row(f"p{i}", "a", status="ok") for i in range(10)]
    report = evaluate_results(predictions, gold, feature_specs=[CATEGORY], resamples=5, rare_class={"max_minority_count": 1, "max_prevalence": .1})
    sensitivity = report["rare_class_sensitivity"]
    assert sensitivity["excluded_rows"] == 1
    assert sensitivity["rare_classes"] == [{"feature": "category", "value": "b", "count": 1, "prevalence": .1}]
    assert sensitivity["excluding_rare_classes"]["count"] == 9
    assert sensitivity["excluding_rare_classes"]["categorical_macro_f1"] == 1


@pytest.fixture
def run_config(tmp_path):
    config = json.loads((ROOT / "configs" / "oncorag_full_pipeline.example.json").read_text())
    config["inputs"] = {"notes_root": "notes"}
    config["features"].update(specifications="features.json", generated_config_dir="generated", configuration_mode="manual")
    config["outputs"]["root"] = "output"
    config["runtime"]["random_seed"] = 37
    config["evaluation"] = {
        "gold_path": "gold.jsonl", "bootstrap": {"resamples": 7, "confidence_level": .9},
        "top_k_ablation": {"values": [3, 5, 10]},
        "retrieval_weight_sensitivity": {"relative_perturbations": [.5, 1.5], "one_at_a_time": True},
        "model_comparison": {"models": ["phi3:medium", "phi3:mini"], "context_windows": [4096, 131072]},
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "features.json").write_text(json.dumps({"features": [CATEGORY]}), encoding="utf-8")
    (tmp_path / "gold.jsonl").write_text(json.dumps(row("p1", "a")) + "\n", encoding="utf-8")
    return path


def test_experiment_configs_are_runnable_isolated_and_one_at_a_time(run_config, tmp_path):
    original = run_config.read_bytes()
    directory = tmp_path / "experiments with spaces"
    manifest = write_experiment_configs(run_config, directory)
    assert len(manifest["experiments"]) == 18
    assert manifest["executed"] is False
    baseline = load_pipeline_config(run_config)
    outputs, namespaces, chroma_paths = set(), set(), set()
    for entry in manifest["experiments"]:
        config = load_pipeline_config(entry["config_path"])
        outputs.add(config["outputs"]["root"])
        namespaces.add(config["vector_store"]["collection_namespace"])
        chroma_paths.add(config["vector_store"]["chroma"]["path"])
        assert Path(config["inputs"]["notes_root"]).is_absolute()
        assert Path(config["features"]["specifications"]).is_absolute()
        assert Path(config["features"]["generated_config_dir"]).is_relative_to(directory)
        assert Path(config["evaluation"]["gold_path"]).is_absolute()
        assert entry["run_argv"][entry["run_argv"].index("--config") + 1] == entry["config_path"]
        assert entry["run_argv"][entry["run_argv"].index("--ollama-model") + 1] == config["runtime"]["ollama"]["model"]
        assert config["features"]["generated_config_dir"] == manifest["feature_config_snapshot"]
        assert config["features"]["generate_if_missing"] is False
        assert "evaluate_argv" in entry
        if entry["name"].startswith("weight_"):
            changed = [key for key, value in config["retrieval"]["weights"].items() if value != baseline["retrieval"]["weights"][key]]
            assert len(changed) == 1
            assert config["retrieval"]["top_k"] == baseline["retrieval"]["top_k"]
    assert len(outputs) == len(namespaces) == len(chroma_paths) == 18
    assert (directory / "run_experiments.sh").stat().st_mode & 0o111
    assert run_config.read_bytes() == original


def test_cli_uses_config_bootstrap_and_explicit_overrides(run_config, tmp_path):
    predictions = tmp_path / "results.json"
    predictions.write_text(json.dumps({"results": [row("p1", "a", status="ok", confidence="High")]}), encoding="utf-8")
    output = tmp_path / "evaluation.json"
    assert main(["--results", str(predictions), "--config", str(run_config), "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["bootstrap"] == {"resamples": 7, "seed": 37, "confidence_level": .9, "unit": "patient"}
    assert main(["--results", str(predictions), "--config", str(run_config), "--output", str(output), "--resamples", "3", "--seed", "2"]) == 0
    assert json.loads(output.read_text())["bootstrap"]["resamples"] == 3
    assert json.loads(output.read_text())["bootstrap"]["seed"] == 2
