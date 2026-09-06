"""Portable feature specifications, deterministic configs, and typed values."""

from __future__ import annotations

import csv
from datetime import date, datetime
from decimal import Decimal
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any

import yaml


_TYPES = {
    "numeric": "numeric", "number": "numeric", "float": "numeric", "decimal": "numeric",
    "integer": "integer", "int": "integer", "categorical": "categorical",
    "category": "categorical", "ordinal": "ordinal", "boolean": "boolean", "bool": "boolean",
    "date": "date", "string": "string", "text": "string",
}
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_RANGE = re.compile(rf"^\s*({_NUMBER})\s*(?:-|to|:|\.\.)\s*({_NUMBER})\s*$", re.IGNORECASE)
_MISSING = {"", "missing", "null", "none", "n/a", "not reported"}


def _number(value: Any) -> int | float:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not numeric feature values")
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[-+]?\d+,\d+", text):
            text = text.replace(",", ".")
        if not re.fullmatch(_NUMBER, text):
            raise ValueError("Expected a complete numeric value without units or surrounding text")
        decimal = Decimal(text)
        result = int(decimal) if decimal == decimal.to_integral_value() else float(decimal)
        if result == 0 and decimal != 0:
            raise ValueError("Numeric value is too small to represent")
    elif isinstance(value, int):
        result = value
    elif isinstance(value, Real):
        result = float(value)
    else:
        raise ValueError("Expected a numeric value")
    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError("Numeric values must be finite")
    return result


def _numeric_range(value: Any, feature_type: str) -> dict | None:
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "n/a"}):
        return None
    if isinstance(value, str):
        match = _RANGE.fullmatch(value)
        if match is None:
            raise ValueError("Numeric expected_range must be min-max, [min, max], or {min, max}")
        lower, upper = match.groups()
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        lower, upper = value
    elif isinstance(value, dict) and value and set(value) <= {"min", "max"}:
        lower, upper = value.get("min"), value.get("max")
    else:
        raise ValueError("Numeric expected_range must contain lower and upper bounds")
    bounds = [_number(bound) if bound is not None else None for bound in (lower, upper)]
    if feature_type == "integer":
        if any(bound is not None and int(bound) != bound for bound in bounds):
            raise ValueError("Integer feature bounds must be integers")
        bounds = [int(bound) if bound is not None else None for bound in bounds]
    if all(bound is not None for bound in bounds) and bounds[0] > bounds[1]:
        raise ValueError("expected_range.min must not exceed expected_range.max")
    return dict(zip(("min", "max"), bounds))


def _strings(value: Any, field: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field} must be a list of nonempty strings")
    return list(dict.fromkeys(item.strip() for item in value))


def validate_feature_specs(data: Any) -> list[dict]:
    """Normalize a features list or a YAML/JSON object containing that list.

    Numeric ranges are inclusive dictionaries; categorical ranges are exact
    string labels. Existing names using hyphens and underscores remain intact.
    """
    features = data.get("features") if isinstance(data, dict) else data
    if not isinstance(features, list) or not features:
        raise ValueError("Feature specifications must contain a nonempty features list")
    normalized = []
    seen = set()
    for position, raw in enumerate(features, 1):
        if not isinstance(raw, dict):
            raise ValueError(f"Feature {position} must be an object")
        name = raw.get("name", raw.get("feature_name"))
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Feature {position} requires a name")
        name = re.sub(r"\s+", "_", name.strip())
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", name):
            raise ValueError(f"Invalid feature name: {name!r}; use letters, digits, underscores, or hyphens")
        if name.casefold() in seen:
            raise ValueError(f"Duplicate feature name: {name}")
        seen.add(name.casefold())
        raw_type = raw.get("type", raw.get("data_type", raw.get("expected_output_type")))
        feature_type = _TYPES.get(raw_type.strip().lower()) if isinstance(raw_type, str) else None
        if feature_type is None:
            raise ValueError(f"Feature {name} requires a supported type")
        spec = dict(raw)
        spec.update(name=name, type=feature_type)
        for field in ("display_name", "description"):
            text = spec.get(field, name.replace("_", " ").replace("-", " "))
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Feature {name}.{field} must be a nonempty string")
            spec[field] = text.strip()
        units = spec.get("units", spec.get("unit"))
        if units is not None and (not isinstance(units, str) or not units.strip()):
            raise ValueError(f"Feature {name}.units must be a nonempty string or null")
        spec["units"] = units.strip() if units else None
        expected = spec.get("expected_range", spec.get("allowed_values"))
        if feature_type in {"numeric", "integer"}:
            spec["expected_range"] = _numeric_range(expected, feature_type)
        elif feature_type in {"categorical", "ordinal"}:
            if isinstance(expected, str):
                expected = next(csv.reader([expected], skipinitialspace=True))
            if not isinstance(expected, (list, tuple)) or not expected:
                raise ValueError(f"Feature {name} requires categorical expected_range values")
            if any(not isinstance(item, (str, int, float, bool)) or (isinstance(item, float) and not math.isfinite(item)) for item in expected):
                raise ValueError(f"Feature {name} categories must be finite scalar labels")
            labels = [str(item).strip() for item in expected]
            if any(not label for label in labels) or len(set(labels)) != len(labels):
                raise ValueError(f"Feature {name} categories must be nonempty and unique")
            spec["expected_range"] = labels
        else:
            if expected is not None and expected != "" and expected != "N/A":
                if feature_type != "boolean" or expected not in ([False, True], [True, False], "true, false", "false, true"):
                    raise ValueError(f"Feature {name}: expected_range is only supported for numeric or categorical features")
            spec["expected_range"] = None
        for field in ("keywords", "synonyms", "common_queries"):
            if field in spec:
                spec[field] = _strings(spec[field], f"Feature {name}.{field}")
        normalized.append(spec)
    return normalized


def load_feature_specs(path: str | Path) -> list[dict]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    return validate_feature_specs(data)


def validate_feature_value(value: Any, spec: dict) -> str | int | float | bool | None:
    """Return a typed scalar; reject invalid output instead of guessing a value."""
    spec = validate_feature_specs([spec])[0]
    feature_type = spec["type"]
    if value is None:
        return None
    if feature_type in {"categorical", "ordinal"}:
        if isinstance(value, (str, int, float, bool)) and str(value).strip() in spec["expected_range"]:
            return str(value).strip()
    if isinstance(value, str) and value.strip().lower() in _MISSING:
        return None
    if feature_type in {"numeric", "integer"}:
        number = _number(value)
        if feature_type == "integer" and int(number) != number:
            raise ValueError(f"Feature {spec['name']} requires an integer")
        bounds = spec["expected_range"] or {}
        if bounds.get("min") is not None and number < bounds["min"]:
            raise ValueError(f"Feature {spec['name']} is below its minimum")
        if bounds.get("max") is not None and number > bounds["max"]:
            raise ValueError(f"Feature {spec['name']} exceeds its maximum")
        if feature_type == "integer":
            return int(number)
        try:
            numeric = float(number)
        except OverflowError:
            raise ValueError("Numeric value is too large to represent") from None
        if not math.isfinite(numeric):
            raise ValueError("Numeric values must be finite")
        return numeric
    if feature_type in {"categorical", "ordinal"}:
        raise ValueError(f"Feature {spec['name']} must exactly match an allowed category")
    if feature_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"true", "yes", "ja"}:
                return True
            if token in {"false", "no", "nein"}:
                return False
        raise ValueError(f"Feature {spec['name']} requires a boolean")
    if feature_type == "date":
        if isinstance(value, date) and not isinstance(value, datetime):
            return value.isoformat()
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
            raise ValueError(f"Feature {spec['name']} requires an ISO YYYY-MM-DD date")
        return date.fromisoformat(value.strip()).isoformat()
    if not isinstance(value, str):
        raise ValueError(f"Feature {spec['name']} requires a string")
    return value.strip()


def generate_feature_configs(specs: list[dict], output_dir: str | Path, language: str = "english") -> dict[str, Path]:
    """Write deterministic manual configs without model or ontology services."""
    specs = validate_feature_specs(specs)
    languages = {"en": "english", "english": "english", "de": "german", "german": "german", "mixed": "mixed"}
    if language not in languages:
        raise ValueError("language must be english, german, or mixed")
    language = languages[language]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for spec in specs:
        label = spec["display_name"]
        questions = spec.get("common_queries") or (
            [f"What is the {label}?", f"Welcher Wert ist fuer {label} dokumentiert?"] if language == "mixed"
            else [f"Welcher Wert ist fuer {label} dokumentiert?"] if language == "german"
            else [f"What is the {label}?"]
        )
        keywords = list(dict.fromkeys([*spec.get("keywords", []), label, spec["description"], *spec.get("synonyms", [])]))
        config = {
            "feature": spec, "feature_name": spec["name"], "display_name": label,
            "data_type": spec["type"], "language": language,
            "config_generation": {"mode": "manual", "ontology_enriched": False},
            "enrichment": {"normalized_name": label, "synonyms": spec.get("synonyms", []), "semantic_keywords": keywords},
            "common_queries": questions, "top_cuis": [], "related_features": [],
            "rules": {"keywords": keywords, "questions": questions},
        }
        if spec["type"] in {"categorical", "ordinal"}:
            options = {str(index + 1): value for index, value in enumerate(spec["expected_range"])}
            options["missing"] = "Missing"
            config["output_format"] = {"type": "categorical", "options": options}
        path = output_dir / f"{spec['name']}.json"
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paths[spec["name"]] = path
    return paths
