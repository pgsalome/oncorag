"""Portable deployment parameters; paper evaluation settings are optional."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from urllib.parse import urlparse

RETRIEVAL_WEIGHT_NAMES = (
    "semantic_weight", "lexical_weight", "name_weight", "graph_weight",
    "boost_alpha", "penalty_beta",
)


def _number(value, name, minimum=0, maximum=None, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if (isinstance(value, float) and not math.isfinite(value)) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its allowed range")


def _mapping(parent, key):
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _optional_mapping(parent, key):
    return _mapping(parent, key) if key in parent else {}


def _boolean_fields(parent, fields, section):
    for key in fields:
        if key in parent and not isinstance(parent[key], bool):
            raise ValueError(f"{section}.{key} must be boolean")


def _choice(value, name, choices):
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{name} must be one of: {', '.join(sorted(choices))}")


def _path(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path")


def validate_runtime_config(runtime):
    """Apply the same generation limits and local-host policy to every interface."""
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be an object")
    _number(runtime.get("workers", 1), "runtime.workers", 1, 1, integer=True)
    _number(runtime.get("random_seed", 2025), "runtime.random_seed", 0, integer=True)
    _boolean_fields(runtime, ("local_processing_only",), "runtime")
    ollama = _mapping(runtime, "ollama")
    _path(ollama.get("host"), "runtime.ollama.host")
    url = urlparse(ollama["host"])
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
        raise ValueError("runtime.ollama.host must be an HTTP(S) URL without credentials")
    try:
        if url.port is not None and url.port < 1:
            raise ValueError("Port must be positive")
    except ValueError as exc:
        raise ValueError("runtime.ollama.host has an invalid port") from exc
    if runtime.get("local_processing_only", True) and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("local_processing_only requires a loopback Ollama host")
    if not isinstance(ollama.get("model"), str) or not ollama["model"].strip():
        raise ValueError("runtime.ollama.model is required")
    _number(ollama.get("temperature", 0), "runtime.ollama.temperature", 0, 2)
    _number(ollama.get("num_ctx", 4096), "runtime.ollama.num_ctx", 256, integer=True)
    _number(ollama.get("timeout_seconds", 120), "runtime.ollama.timeout_seconds", .001)
    _number(ollama.get("max_tokens", 1024), "runtime.ollama.max_tokens", 1, integer=True)
    _number(ollama.get("validation_retries", 1), "runtime.ollama.validation_retries", 0, 3, integer=True)


def validate_chat_config(chat):
    if not isinstance(chat, dict):
        raise ValueError("chat must be an object")
    _number(chat.get("history_turns", 5), "chat.history_turns", 0, 20, integer=True)
    _number(chat.get("max_question_chars", 4000), "chat.max_question_chars", 1, 32000, integer=True)
    _number(chat.get("max_history_chars", 12000), "chat.max_history_chars", 1, 128000, integer=True)
    _number(chat.get("feature_match_threshold", .45), "chat.feature_match_threshold", 0, 1)


def validate_pipeline_config(config):
    if not isinstance(config, dict):
        raise ValueError("Pipeline config must be an object")
    inputs = _mapping(config, "inputs")
    features = _mapping(config, "features")
    runtime = _mapping(config, "runtime")
    retrieval = _mapping(config, "retrieval")
    outputs = _mapping(config, "outputs")
    cohort = _optional_mapping(config, "cohort")
    if "name" in cohort:
        _path(cohort["name"], "cohort.name")
    if "language" in cohort:
        _choice(cohort["language"], "cohort.language", {"en", "de", "english", "german", "mixed", "unknown"})
    if bool(inputs.get("notes_root")) == bool(inputs.get("registry_path")):
        raise ValueError("Specify exactly one of notes_root or registry_path")
    for key in ("notes_root", "registry_path", "patient_ids_file"):
        if inputs.get(key) is not None:
            _path(inputs[key], f"inputs.{key}")
    for key in ("specifications", "generated_config_dir"):
        _path(features.get(key), f"features.{key}")
    _choice(features.get("configuration_mode", "manual"), "features.configuration_mode", {"manual", "automatic"})
    _choice(features.get("language", "english"), "features.language", {"english", "german", "mixed"})
    _boolean_fields(features, ("generate_if_missing",), "features")
    ontology = _optional_mapping(features, "ontology_enrichment")
    _number(ontology.get("max_concepts_per_feature", 5), "max_concepts_per_feature", 1, integer=True)
    _number(ontology.get("minimum_relevance_score", .6), "minimum_relevance_score", 0, 1)
    validate_runtime_config(runtime)
    validate_chat_config(_optional_mapping(config, "chat"))
    _number(retrieval.get("top_k"), "retrieval.top_k", 1, integer=True)
    _number(retrieval.get("candidate_entity_limit", 30), "candidate_entity_limit", 1, integer=True)
    _number(retrieval.get("graph_depth", 2), "graph_depth", 0, integer=True)
    weights = _mapping(retrieval, "weights")
    for name in RETRIEVAL_WEIGHT_NAMES:
        _number(weights.get(name), f"retrieval.weights.{name}")
    diffusion = _mapping(retrieval, "graph_diffusion")
    _boolean_fields(diffusion, ("enabled",), "graph_diffusion")
    _number(diffusion.get("iterations", 2), "graph_diffusion.iterations", 1, integer=True)
    _number(diffusion.get("residual_alpha", .6), "graph_diffusion.residual_alpha", 0, 1)
    _number(diffusion.get("similarity_threshold", .55), "similarity_threshold", 0, 1)
    _number(diffusion.get("max_neighbors", 12), "max_neighbors", 1, integer=True)
    _path(outputs.get("root"), "outputs.root")
    for key in ("graph_cache_dir", "chroma_cache_dir", "prompt_cache_dir", "results_file"):
        if key in outputs:
            _path(outputs[key], f"outputs.{key}")
            if Path(outputs[key]).is_absolute() or ".." in Path(outputs[key]).parts:
                raise ValueError(f"outputs.{key} must be relative to outputs.root")
    if "vector_store" in config:
        from ..vector_store.config import validate_vector_store_config
        validate_vector_store_config(config["vector_store"])
    graph = _optional_mapping(config, "graph")
    _boolean_fields(graph, ("include_report_sentences",), "graph")
    context = _optional_mapping(graph, "context_filters")
    _boolean_fields(context, ("allow_negated", "allow_hypothetical", "allow_family", "allow_historical"), "graph.context_filters")
    deduplication = _optional_mapping(graph, "deduplication")
    _boolean_fields(deduplication, ("enabled",), "graph.deduplication")
    _number(deduplication.get("similarity_threshold", .85), "graph.deduplication.similarity_threshold", 0, 1)
    models = graph.get("model_configs", [{"name": "en_ner_bc5cdr_md", "entity_types": []}])
    if not isinstance(models, list) or not models or any(
        not isinstance(model, dict) or not isinstance(model.get("name"), str) or not model["name"].strip()
        for model in models
    ):
        raise ValueError("graph.model_configs must contain named spaCy models")
    for model in models:
        if "priority" in model:
            _number(model["priority"], "graph.model_configs.priority", 0, integer=True)
    temporal = _optional_mapping(config, "temporal_anchoring")
    for key in ("baseline", "outcome", "mimic"):
        policy = _optional_mapping(temporal, key)
        if "window_months" in policy:
            _number(policy["window_months"], f"temporal_anchoring.{key}.window_months")
    evaluation = _optional_mapping(config, "evaluation")
    _boolean_fields(evaluation, ("feature_complexity_stratification",), "evaluation")
    if "gold_path" in evaluation:
        _path(evaluation["gold_path"], "evaluation.gold_path")
    bootstrap = _optional_mapping(evaluation, "bootstrap")
    _number(bootstrap.get("resamples", 1000), "bootstrap.resamples", 1, integer=True)
    _number(bootstrap.get("confidence_level", .95), "bootstrap.confidence_level", .001, .999)
    ablation = _optional_mapping(evaluation, "top_k_ablation")
    values = ablation.get("values", [])
    if not isinstance(values, list):
        raise ValueError("top_k_ablation.values must be a list")
    _boolean_fields(ablation, ("keep_all_other_parameters_fixed",), "top_k_ablation")
    for value in values:
        _number(value, "top_k_ablation.values", 1, integer=True)
    for key in ("confidence_calibration", "rare_class_sensitivity", "retrieval_weight_sensitivity", "inter_rater_agreement", "model_comparison"):
        _optional_mapping(evaluation, key)


def load_pipeline_config(path):
    """Resolve input paths relative to the JSON file, never the working dir."""
    config_path = Path(path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_pipeline_config(config)
    config = copy.deepcopy(config)
    for section, keys in {
        "inputs": ("notes_root", "registry_path", "patient_ids_file"),
        "features": ("specifications", "generated_config_dir"),
        "outputs": ("root",),
        "evaluation": ("gold_path",),
    }.items():
        for key in keys:
            value = config.get(section, {}).get(key)
            if value:
                config[section][key] = str((config_path.parent / Path(value).expanduser()).resolve())
    chroma = config.get("vector_store", {}).get("chroma", {})
    if chroma.get("path"):
        chroma["path"] = str((config_path.parent / Path(chroma["path"]).expanduser()).resolve())
    return config
