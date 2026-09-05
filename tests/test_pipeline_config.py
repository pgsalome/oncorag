import json
from pathlib import Path

import pytest

from oncoraggraph.config.pipeline_config import load_pipeline_config, validate_pipeline_config


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "oncorag_full_pipeline.example.json"


def test_example_pipeline_config_is_valid():
    config = load_pipeline_config(CONFIG_PATH)

    assert config["retrieval"]["top_k"] == 5
    assert config["evaluation"]["bootstrap"]["resamples"] == 1000


def test_pipeline_config_requires_a_note_source():
    config = json.loads(CONFIG_PATH.read_text())
    config["inputs"]["notes_root"] = None
    config["inputs"]["registry_path"] = None

    with pytest.raises(ValueError, match="notes_root or registry_path"):
        validate_pipeline_config(config)


def test_pipeline_config_requires_reviewer_reproducibility_settings():
    config = json.loads(CONFIG_PATH.read_text())
    config["retrieval"]["weights"].pop("graph_weight")

    with pytest.raises(ValueError, match="graph_weight"):
        validate_pipeline_config(config)


def replace_setting(config, path, value):
    parent = config
    parts = path.split(".")
    for key in parts[:-1]:
        parent = parent.setdefault(key, {})
    parent[parts[-1]] = value


@pytest.mark.parametrize("path,value", [
    ("cohort", []), ("features.ontology_enrichment", None),
    ("graph", []), ("graph.context_filters", True), ("graph.deduplication", "enabled"),
    ("evaluation", []), ("evaluation.bootstrap", None),
    ("evaluation.top_k_ablation", []), ("evaluation.model_comparison", "phi3"),
    ("temporal_anchoring", []), ("temporal_anchoring.baseline", None),
    ("features.configuration_mode", []), ("features.language", {}),
])
def test_malformed_sections_raise_validation_errors_instead_of_runtime_type_errors(path, value):
    config = json.loads(CONFIG_PATH.read_text())
    replace_setting(config, path, value)
    with pytest.raises(ValueError):
        validate_pipeline_config(config)


@pytest.mark.parametrize("path", [
    "features.generate_if_missing", "runtime.local_processing_only",
    "graph.include_report_sentences", "graph.context_filters.allow_negated",
    "graph.context_filters.allow_hypothetical", "graph.context_filters.allow_family",
    "graph.context_filters.allow_historical", "graph.deduplication.enabled",
    "retrieval.graph_diffusion.enabled", "evaluation.feature_complexity_stratification",
    "evaluation.top_k_ablation.keep_all_other_parameters_fixed",
])
def test_boolean_parameters_reject_string_values(path):
    config = json.loads(CONFIG_PATH.read_text())
    replace_setting(config, path, "false")
    with pytest.raises(ValueError, match="boolean"):
        validate_pipeline_config(config)
    replace_setting(config, path, False)
    validate_pipeline_config(config)


@pytest.mark.parametrize("path,value", [
    ("retrieval.top_k", True), ("retrieval.top_k", 0), ("retrieval.top_k", 2.5),
    ("retrieval.weights.semantic_weight", float("nan")),
    ("retrieval.weights.graph_weight", -1),
    ("retrieval.graph_diffusion.residual_alpha", 1.01),
    ("retrieval.graph_diffusion.max_neighbors", 0),
    ("features.ontology_enrichment.minimum_relevance_score", -0.1),
    ("features.ontology_enrichment.max_concepts_per_feature", False),
    ("graph.deduplication.similarity_threshold", 1.1),
    ("graph.model_configs", [{"name": True}]),
    ("graph.model_configs", [{"name": "model", "priority": "first"}]),
    ("runtime.workers", 2), ("runtime.random_seed", -1),
    ("runtime.ollama.num_ctx", 0), ("runtime.ollama.timeout_seconds", float("inf")),
    ("runtime.ollama.temperature", -0.1), ("runtime.ollama.max_tokens", True),
    ("runtime.ollama.validation_retries", -1), ("runtime.ollama.validation_retries", 4),
    ("runtime.ollama.validation_retries", True), ("runtime.ollama.validation_retries", 1.5),
    ("evaluation.bootstrap.resamples", 0), ("evaluation.bootstrap.confidence_level", 1),
    ("evaluation.top_k_ablation.values", None), ("evaluation.top_k_ablation.values", [True]),
    ("temporal_anchoring.baseline.window_months", -1),
])
def test_invalid_numeric_parameters_fail_before_execution(path, value):
    config = json.loads(CONFIG_PATH.read_text())
    replace_setting(config, path, value)
    with pytest.raises(ValueError):
        validate_pipeline_config(config)


@pytest.mark.parametrize("retries", [0, 1, 2, 3])
def test_validation_retry_bounds_allow_zero_through_three(retries):
    config = json.loads(CONFIG_PATH.read_text())
    config["runtime"]["ollama"]["validation_retries"] = retries
    validate_pipeline_config(config)


@pytest.mark.parametrize("host", [
    "ftp://127.0.0.1", "http://name:secret@127.0.0.1", "http://127.0.0.1:invalid",
    "http://127.0.0.1:0", "http://127.0.0.1:65536", "http://[invalid", "",
])
def test_invalid_ollama_urls_fail_before_network_requests(host):
    config = json.loads(CONFIG_PATH.read_text())
    config["runtime"]["ollama"]["host"] = host
    with pytest.raises(ValueError):
        validate_pipeline_config(config)


def test_paper_values_are_defaults_not_frozen_deployment_requirements():
    config = json.loads(CONFIG_PATH.read_text())
    config["features"]["ontology_enrichment"].update(max_concepts_per_feature=8, minimum_relevance_score=.8)
    config["retrieval"]["top_k"] = 7
    config["retrieval"]["graph_diffusion"].update(iterations=4, residual_alpha=.3, max_neighbors=6)
    config["evaluation"]["bootstrap"].update(resamples=25, confidence_level=.9)
    config["evaluation"]["top_k_ablation"]["values"] = [1, 7]
    config["evaluation"]["model_comparison"]["context_windows"] = [8192]
    validate_pipeline_config(config)
    config.pop("evaluation")
    validate_pipeline_config(config)


def test_config_paths_resolve_from_config_directory_not_current_directory(tmp_path, monkeypatch):
    config = json.loads(CONFIG_PATH.read_text())
    config["inputs"] = {"registry_path": "notes/registry.csv"}
    config["features"].update(specifications="features.yaml", generated_config_dir="generated")
    config["outputs"]["root"] = "outputs"
    config["vector_store"]["chroma"] = {"path": "vector_cache"}
    directory = tmp_path / "config"
    directory.mkdir()
    path = directory / "run.json"
    path.write_text(json.dumps(config))
    monkeypatch.chdir(tmp_path)
    resolved = load_pipeline_config(path)
    assert resolved["inputs"]["registry_path"] == str(directory / "notes/registry.csv")
    assert resolved["features"]["specifications"] == str(directory / "features.yaml")
    assert resolved["features"]["generated_config_dir"] == str(directory / "generated")
    assert resolved["outputs"]["root"] == str(directory / "outputs")
    assert resolved["vector_store"]["chroma"]["path"] == str(directory / "vector_cache")
