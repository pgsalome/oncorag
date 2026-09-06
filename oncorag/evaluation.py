"""Evaluate structured synthetic extractions and prepare controlled experiments."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import sys

import numpy as np
from sklearn.metrics import f1_score

from .config.feature_schema import generate_feature_configs, load_feature_specs, validate_feature_specs, validate_feature_value
from .config.pipeline_config import RETRIEVAL_WEIGHT_NAMES, load_pipeline_config, validate_pipeline_config


_STATUSES = {"ok", "missing", "invalid", "error"}
_CONFIDENCE = ("High", "Medium", "Low", "Unreported")
_CLASSIFICATION = {"categorical", "ordinal", "boolean"}
_UNAVAILABLE = "unavailable-prediction"


def load_records(path: str | Path) -> list[dict]:
    """Load JSONL gold or a pipeline JSON object containing results."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        records = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(records, dict):
            records = records.get("results")
    if not isinstance(records, list):
        raise ValueError("Input must contain a list of patient-feature records")
    return records


def _indexed(records, name):
    if isinstance(records, dict):
        records = records.get("results")
    if not isinstance(records, list):
        raise ValueError(f"{name} must contain a list of records")
    indexed = {}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError(f"Each {name} row must be an object")
        for field in ("patient_id", "feature"):
            if not isinstance(row.get(field), str) or not row[field] or row[field] != row[field].strip():
                raise ValueError(f"{name}.{field} must be a nonempty string without surrounding whitespace")
        key = (row["patient_id"], row["feature"])
        if key in indexed:
            raise ValueError(f"Duplicate {name} patient-feature key: {key!r}")
        if "value" not in row:
            raise ValueError(f"{name} record requires a value, including explicit null for missing")
        indexed[key] = row
    return indexed


def _typed_value(value, spec):
    if value is None:
        return None
    if type(value) not in {str, int, float, bool} or (type(value) is float and not math.isfinite(value)):
        raise ValueError("Values must be finite JSON scalars or null")
    if spec is None:
        return value
    feature_type = spec["type"]
    required = {"integer": {int}, "numeric": {int, float}, "boolean": {bool}}
    if type(value) not in required.get(feature_type, {str}):
        raise ValueError(f"Value does not have the declared {feature_type} type")
    validated = validate_feature_value(value, spec)
    if validated is None:
        raise ValueError("Missing values must be represented by JSON null")
    return validated


def _token(value):
    return json.dumps([type(value).__name__, value], ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _exact(left, right):
    return type(left) is type(right) and left == right


def _metrics(rows, feature_specs, labels):
    per_feature = {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["feature"]].append(row)
    for feature, group in sorted(grouped.items()):
        entry = {"count": len(group), "exact_match": sum(row["correct"] for row in group) / len(group)}
        if feature_specs.get(feature, {}).get("type") in _CLASSIFICATION:
            truth_tokens = [row["gold_token"] for row in group]
            predicted_tokens = [row["prediction_token"] for row in group]
            active_labels = [label for label in labels[feature] if label in truth_tokens or label in predicted_tokens]
            entry["macro_f1"] = float(f1_score(
                truth_tokens, predicted_tokens,
                labels=active_labels, average="macro", zero_division=0,
            ))
        per_feature[feature] = entry
    categorical_f1 = [entry["macro_f1"] for entry in per_feature.values() if "macro_f1" in entry]
    return {
        "count": len(rows),
        "exact_match": sum(row["correct"] for row in rows) / len(rows) if rows else None,
        "categorical_macro_f1": sum(categorical_f1) / len(categorical_f1) if categorical_f1 else None,
        "per_feature": per_feature,
    }


def _interval(values, confidence_level):
    if not values:
        return None
    alpha = (1 - confidence_level) / 2
    low, high = np.quantile(values, [alpha, 1 - alpha])
    return {"low": float(low), "high": float(high), "effective_resamples": len(values)}


def evaluate_results(
    results, gold, *, feature_specs=None, resamples=1000, seed=2025,
    confidence_level=.95, rare_class=None,
):
    """Evaluate every gold row, retaining absent, invalid, and failed predictions.

    Confidence intervals resample whole patients with replacement. F1 uses gold
    classes, omitting classes absent from both truth and predictions in a sample.
    """
    if type(resamples) is not int or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(confidence_level, bool) or not isinstance(confidence_level, (int, float)) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    predictions, expected = _indexed(results, "results"), _indexed(gold, "gold")
    if not expected:
        raise ValueError("Gold records must not be empty")
    extras = predictions.keys() - expected.keys()
    if extras:
        raise ValueError(f"Predictions contain {len(extras)} patient-feature keys absent from gold")
    specs = {} if feature_specs is None else {spec["name"]: spec for spec in validate_feature_specs(feature_specs)}
    if specs and any(feature not in specs for _, feature in expected):
        raise ValueError("Feature specifications must cover every feature in gold")
    rows = []
    for (patient_id, feature), truth in sorted(expected.items()):
        spec = specs.get(feature)
        try:
            target = _typed_value(truth["value"], spec)
        except ValueError as exc:
            raise ValueError(f"Invalid gold value for {patient_id}/{feature}: {exc}") from exc
        prediction = predictions.get((patient_id, feature))
        status = "missing_prediction"
        confidence = "Unreported"
        value = None
        if prediction is not None:
            status = prediction.get("status")
            if status not in _STATUSES:
                raise ValueError("Prediction status must be ok, missing, invalid, or error")
            confidence = prediction.get("confidence") or "Unreported"
            if confidence not in _CONFIDENCE:
                raise ValueError("Confidence must be High, Medium, Low, or omitted")
            if status in {"ok", "missing"}:
                try:
                    value = _typed_value(prediction["value"], spec)
                    if (status == "ok" and value is None) or (status == "missing" and value is not None):
                        raise ValueError("Status does not agree with the predicted value")
                except ValueError:
                    status = "invalid"
        available = status in {"ok", "missing"}
        correct = available and _exact(target, value)
        rows.append({
            "patient_id": patient_id, "feature": feature, "gold_value": target,
            "predicted_value": value, "status": status, "confidence": confidence,
            "correct": correct, "gold_token": _token(target),
            "prediction_token": _token(value) if available else _UNAVAILABLE,
        })
    labels = {feature: sorted({row["gold_token"] for row in rows if row["feature"] == feature}) for feature in {row["feature"] for row in rows}}
    report = _metrics(rows, specs, labels)
    patients = sorted({row["patient_id"] for row in rows})
    counts = Counter(row["status"] for row in rows)
    report.update({
        "patients": len(patients), "features": len(labels), "expected_predictions": len(rows),
        "received_predictions": len(predictions),
        "status_counts": {status: counts[status] for status in (*sorted(_STATUSES), "missing_prediction")},
        "status_rates": {status: counts[status] / len(rows) for status in (*sorted(_STATUSES), "missing_prediction")},
        "bootstrap": {"resamples": resamples, "seed": seed, "confidence_level": confidence_level, "unit": "patient"},
        "f1_classes": "observed gold classes; classes absent from both truth and predictions in a bootstrap sample are omitted; unavailable predictions count as false negatives",
        "feature_types_supplied": bool(specs),
    })
    report["confidence_groups"] = {
        group: {"count": sum(row["confidence"] == group for row in rows),
                "accuracy": _metrics([row for row in rows if row["confidence"] == group], specs, labels)["exact_match"]}
        for group in _CONFIDENCE
    }
    strata = defaultdict(list)
    for row in rows:
        complexity = specs.get(row["feature"], {}).get("complexity", "unspecified")
        if not isinstance(complexity, str) or not complexity.strip():
            raise ValueError("Feature complexity must be a nonempty string when supplied")
        strata[complexity].append(row)
    report["complexity_strata"] = {name: _metrics(group, specs, labels) for name, group in sorted(strata.items())}

    if rare_class is not None and not isinstance(rare_class, dict):
        raise ValueError("rare_class must be an object")
    rare = {"max_minority_count": 5, "max_prevalence": .1, **(rare_class or {})}
    if type(rare["max_minority_count"]) is not int or rare["max_minority_count"] < 0:
        raise ValueError("max_minority_count must be a nonnegative integer")
    if isinstance(rare["max_prevalence"], bool) or not isinstance(rare["max_prevalence"], (int, float)) or not 0 <= rare["max_prevalence"] <= 1:
        raise ValueError("max_prevalence must be between zero and one")
    rare_keys, rare_details = set(), []
    for feature in sorted(labels):
        if specs.get(feature, {}).get("type") not in _CLASSIFICATION:
            continue
        group = [row for row in rows if row["feature"] == feature and row["gold_value"] is not None]
        support = Counter(row["gold_token"] for row in group)
        for token, count in sorted(support.items()):
            prevalence = count / len(group)
            if count <= rare["max_minority_count"] or prevalence < rare["max_prevalence"]:
                rare_keys.add((feature, token))
                rare_details.append({"feature": feature, "value": json.loads(token)[1], "count": count, "prevalence": prevalence})
    retained = [row for row in rows if (row["feature"], row["gold_token"]) not in rare_keys]
    retained_labels = {feature: sorted({row["gold_token"] for row in retained if row["feature"] == feature}) for feature in labels}
    report["rare_class_sensitivity"] = {
        "rule": "count <= max_minority_count OR prevalence < max_prevalence",
        **rare, "rare_classes": rare_details, "excluded_rows": len(rows) - len(retained),
        "excluding_rare_classes": _metrics(retained, specs, retained_labels),
    }

    by_patient = {patient: [row for row in rows if row["patient_id"] == patient] for patient in patients}
    samples = defaultdict(list)
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        sampled = [row for index in rng.integers(0, len(patients), size=len(patients)) for row in by_patient[patients[int(index)]]]
        summary = _metrics(sampled, specs, labels)
        for metric in ("exact_match", "categorical_macro_f1"):
            if summary[metric] is not None:
                samples[("overall", metric)].append(summary[metric])
        for feature, metrics in summary["per_feature"].items():
            for metric in ("exact_match", "macro_f1"):
                if metric in metrics:
                    samples[(feature, metric)].append(metrics[metric])
        for confidence in _CONFIDENCE:
            group = [row for row in sampled if row["confidence"] == confidence]
            if group:
                samples[("confidence", confidence)].append(sum(row["correct"] for row in group) / len(group))
    report["confidence_intervals"] = {metric: _interval(samples[("overall", metric)], confidence_level) for metric in ("exact_match", "categorical_macro_f1")}
    for feature, entry in report["per_feature"].items():
        entry["confidence_intervals"] = {metric: _interval(samples[(feature, metric)], confidence_level) for metric in ("exact_match", "macro_f1") if metric in entry}
    for confidence, entry in report["confidence_groups"].items():
        entry["confidence_interval"] = _interval(samples[("confidence", confidence)], confidence_level)
    report["cases"] = [{key: value for key, value in row.items() if not key.endswith("_token")} for row in rows]
    return report


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_experiment_configs(config_path: str | Path, directory: str | Path) -> dict:
    """Write runnable one-factor retrieval and model/context configurations."""
    baseline = load_pipeline_config(config_path)
    directory = Path(directory).expanduser().resolve()
    specs = load_feature_specs(baseline["features"]["specifications"])
    source_features = Path(baseline["features"]["generated_config_dir"])
    source_paths = {spec["name"]: source_features / f"{spec['name']}.json" for spec in specs}
    frozen_features = directory / "feature_configs"
    if all(path.is_file() for path in source_paths.values()):
        frozen_data = {}
        for spec in specs:
            data = json.loads(source_paths[spec["name"]].read_text(encoding="utf-8"))
            declared = validate_feature_specs([data.get("feature")])[0]
            if any(declared.get(field) != spec.get(field) for field in ("name", "type", "expected_range", "units")):
                raise ValueError("Existing feature configs do not match current specifications; regenerate baseline configs first")
            frozen_data[spec["name"]] = data
        for name, data in frozen_data.items():
            _write_json(frozen_features / f"{name}.json", data)
    elif baseline["features"].get("configuration_mode", "manual") == "manual":
        generate_feature_configs(specs, frozen_features, language=baseline["features"].get("language", "english"))
    else:
        raise ValueError("Automatic experiments require baseline feature configs first; run the pipeline with --stage config")
    if (frozen_features / "generation_manifest.json").exists():
        raise ValueError("Experiment feature snapshot contains a generation manifest; choose a new experiment directory")
    baseline["features"].update(generated_config_dir=str(frozen_features), generate_if_missing=False)
    evaluation = baseline.get("evaluation", {})
    variants = [("baseline", baseline)]
    for value in evaluation.get("top_k_ablation", {}).get("values", []):
        variant = copy.deepcopy(baseline)
        variant["retrieval"]["top_k"] = value
        variants.append((f"top_k_{value}", variant))
    model_settings = evaluation.get("model_comparison", {})
    for model in model_settings.get("models") or [baseline["runtime"]["ollama"]["model"]]:
        for context in model_settings.get("context_windows") or [baseline["runtime"]["ollama"]["num_ctx"]]:
            variant = copy.deepcopy(baseline)
            variant["runtime"]["ollama"].update(model=model, num_ctx=context)
            variants.append((f"model_{model}_context_{context}", variant))
    weight_settings = evaluation.get("retrieval_weight_sensitivity", {})
    if weight_settings.get("one_at_a_time", True) is not True:
        raise ValueError("Experiment generation supports one-at-a-time weight sensitivity")
    for weight in RETRIEVAL_WEIGHT_NAMES:
        for factor in weight_settings.get("relative_perturbations", []):
            if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor < 0:
                raise ValueError("Weight perturbations must be finite nonnegative numbers")
            variant = copy.deepcopy(baseline)
            variant["retrieval"]["weights"][weight] *= factor
            variants.append((f"weight_{weight}_times_{factor}", variant))
    seen, experiments = set(), []
    for label, variant in variants:
        digest = hashlib.sha256(json.dumps(variant, sort_keys=True).encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", label).strip("_") + "_" + digest[:8]
        output = directory / "outputs" / slug
        variant["outputs"].update(root=str(output), graph_cache_dir="graphs", chroma_cache_dir="chroma", prompt_cache_dir="prompt_cache")
        vector = variant.setdefault("vector_store", {"backend": "chroma"})
        vector["collection_namespace"] = f"{vector.get('collection_namespace', 'experiments')}:{slug}"
        vector.setdefault("chroma", {})["path"] = str(output / "chroma")
        validate_pipeline_config(variant)
        destination = directory / "configs" / f"{slug}.json"
        result_path = output / variant["outputs"].get("results_file", "structured_features.json")
        _write_json(destination, variant)
        run_argv = [sys.executable, "-m", "oncorag.pipeline", "--config", str(destination),
                    "--ollama-model", variant["runtime"]["ollama"]["model"],
                    "--ollama-host", variant["runtime"]["ollama"]["host"]]
        entry = {"name": label, "config_path": str(destination), "output_root": str(output), "results_path": str(result_path), "run_argv": run_argv}
        gold_path = variant.get("evaluation", {}).get("gold_path")
        if gold_path:
            entry["evaluate_argv"] = [sys.executable, "-m", "oncorag.evaluation", "--config", str(destination), "--results", str(result_path), "--gold", gold_path, "--output", str(output / "evaluation.json")]
        experiments.append(entry)
    manifest = {"source_config": str(Path(config_path).resolve()), "feature_config_snapshot": str(frozen_features),
                "feature_configs_fixed": True, "experiments": experiments, "executed": False}
    _write_json(directory / "experiments.json", manifest)
    commands = ["#!/bin/sh", "set -eu", "failed=0", ""]
    for entry in experiments:
        commands.append(f"if ! {shlex.join(entry['run_argv'])}; then failed=1; fi")
        if "evaluate_argv" in entry:
            commands.append(f"if ! {shlex.join(entry['evaluate_argv'])}; then failed=1; fi")
    commands.append('exit "$failed"')
    script = directory / "run_experiments.sh"
    script.write_text("\n".join(commands) + "\n", encoding="utf-8")
    script.chmod(0o755)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results")
    parser.add_argument("--gold")
    parser.add_argument("--output")
    parser.add_argument("--config")
    parser.add_argument("--resamples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--write-experiments", metavar="DIR")
    args = parser.parse_args(argv)
    try:
        if args.write_experiments:
            if not args.config:
                parser.error("--write-experiments requires --config")
            manifest = write_experiment_configs(args.config, args.write_experiments)
            print(f"Wrote {len(manifest['experiments'])} experiment configs; no experiments executed")
            return 0
        config = load_pipeline_config(args.config) if args.config else {}
        evaluation = config.get("evaluation", {})
        gold_path = args.gold or evaluation.get("gold_path")
        if not args.results or not gold_path or not args.output:
            parser.error("Evaluation requires --results, --gold (or evaluation.gold_path), and --output")
        bootstrap = evaluation.get("bootstrap", {})
        report = evaluate_results(
            load_records(args.results), load_records(gold_path),
            feature_specs=load_feature_specs(config["features"]["specifications"]) if config else None,
            resamples=args.resamples if args.resamples is not None else bootstrap.get("resamples", 1000),
            seed=args.seed if args.seed is not None else bootstrap.get("seed", config.get("runtime", {}).get("random_seed", 2025)),
            confidence_level=bootstrap.get("confidence_level", .95),
            rare_class=evaluation.get("rare_class_sensitivity"),
        )
        _write_json(args.output, report)
        print(json.dumps({key: report[key] for key in ("expected_predictions", "exact_match", "categorical_macro_f1", "status_counts")}, indent=2))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
