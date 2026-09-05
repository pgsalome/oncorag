"""Feature contracts cover real validation failures without models or services."""

from datetime import date, datetime
import importlib
import json

import pytest

from oncoraggraph.config.feature_schema import (
    generate_feature_configs,
    load_feature_specs,
    validate_feature_specs,
    validate_feature_value,
)


def spec(feature_type, expected_range=None):
    return {"name": "measurement", "type": feature_type, "expected_range": expected_range}


@pytest.mark.parametrize("bounds", ["-5-10", "-5 to 10", [-5, 10], {"min": -5, "max": 10}])
def test_legacy_numeric_ranges_normalize_and_are_enforced(bounds):
    normalized = validate_feature_specs([spec("float", bounds)])[0]
    assert normalized["type"] == "numeric"
    assert normalized["expected_range"] == {"min": -5, "max": 10}
    assert validate_feature_value("-5", normalized) == -5.0
    assert validate_feature_value("10", normalized) == 10.0
    assert validate_feature_value("2,5", normalized) == 2.5
    assert type(validate_feature_value("2", normalized)) is float
    for value in (-6, 11, "2.5 cm", "about 2", True, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            validate_feature_value(value, normalized)


def test_integer_is_not_truncated_or_rounded():
    definition = spec("integer", [0, 10])
    assert validate_feature_value("4.0", definition) == 4
    assert type(validate_feature_value(4, definition)) is int
    for value in (4.1, "4.1", "4 years", False):
        with pytest.raises(ValueError):
            validate_feature_value(value, definition)
    exact = "9007199254740993"
    assert validate_feature_value(exact, spec("integer")) == int(exact)
    with pytest.raises(ValueError):
        validate_feature_value("1e-4000", spec("integer"))


def test_categories_match_complete_labels_and_preserve_unknown():
    definition = spec("categorical", "positive, negative, Unknown")
    assert validate_feature_value("negative", definition) == "negative"
    assert validate_feature_value("Unknown", definition) == "Unknown"
    for value in ("not positive", "positive or negative", "Positive", ["positive"]):
        with pytest.raises(ValueError):
            validate_feature_value(value, definition)
    assert validate_feature_value("Missing", definition) is None
    assert validate_feature_value(None, definition) is None


def test_date_calendar_validation_and_precision():
    definition = spec("date")
    assert validate_feature_value("2024-02-29", definition) == "2024-02-29"
    assert validate_feature_value(date(2024, 2, 29), definition) == "2024-02-29"
    for value in ("2023-02-29", "2024-13-01", "01.03.2024", "2024", "2024-02", datetime(2024, 2, 29)):
        with pytest.raises(ValueError):
            validate_feature_value(value, definition)


def test_boolean_is_typed_and_not_truthiness():
    for value in (True, "true", "Yes", "ja"):
        assert validate_feature_value(value, spec("boolean")) is True
    for value in (False, "false", "No", "nein"):
        assert validate_feature_value(value, spec("boolean")) is False
    for value in (1, 0, "possibly", "not yes", []):
        with pytest.raises(ValueError):
            validate_feature_value(value, spec("boolean"))


@pytest.mark.parametrize("features", [
    [], {}, [{"name": "x"}], [spec("unsupported")],
    [{"name": "../outside", "type": "string"}],
    [spec("string"), spec("string")],
    [spec("numeric", [10, 0])], [spec("numeric", "usually low")],
    [spec("integer", [0.5, 5])], [spec("numeric", [0, float("nan")])],
    [spec("categorical", [])], [spec("categorical", ["a", "a"])],
])
def test_invalid_specs_rejected(features):
    with pytest.raises(ValueError):
        validate_feature_specs(features)


def test_files_preserve_declared_fields_and_generate_deterministically(tmp_path):
    source = tmp_path / "features.yaml"
    source.write_text(
        "features:\n- name: lab_atrx-loss_status\n  type: categorical\n"
        "  description: ATRX status\n  expected_range: [retained, lost]\n"
        "  keywords: [ATRX, Verlust]\n- name: tumor_size\n  type: numeric\n"
        "  expected_range: [0, 100]\n  unit: mm\n",
        encoding="utf-8",
    )
    specs = load_feature_specs(source)
    assert specs[1]["units"] == "mm"
    paths = generate_feature_configs(specs, tmp_path / "configs", "mixed")
    assert set(paths) == {"lab_atrx-loss_status", "tumor_size"}
    before = {name: path.read_bytes() for name, path in paths.items()}
    config = json.loads(before["lab_atrx-loss_status"])
    assert config["config_generation"] == {"mode": "manual", "ontology_enriched": False}
    assert "Verlust" in config["rules"]["keywords"]
    assert len(config["common_queries"]) == 2
    assert config["feature"]["expected_range"] == ["retained", "lost"]
    generate_feature_configs(specs, tmp_path / "configs", "mixed")
    assert before == {name: path.read_bytes() for name, path in paths.items()}


def test_create_config_import_and_manual_cli_never_download(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Manual config generation must not use network services")

    monkeypatch.setattr("requests.get", no_network)
    monkeypatch.setattr("requests.post", no_network)
    generator = importlib.import_module("oncoraggraph.create_config")
    monkeypatch.setattr(generator, "_load_wordnet", no_network)
    monkeypatch.delenv("UMLS_API_KEY", raising=False)
    source = tmp_path / "features.json"
    source.write_text(json.dumps({"features": [spec("integer", [0, 5])]}), encoding="utf-8")
    assert generator.main([
        "--features-file", str(source), "--output-dir", str(tmp_path / "configs"),
        "--output-file", str(tmp_path / "mappings.json"), "--mode", "manual", "--language", "english",
    ]) == 0
    assert (tmp_path / "configs" / "measurement.json").is_file()


def test_ollama_runtime_parameters_reach_request(monkeypatch):
    generator = importlib.import_module("oncoraggraph.create_config")
    monkeypatch.setattr(generator, "OLLAMA_SETTINGS", {"temperature": 0, "num_ctx": 4096, "seed": 7, "timeout_seconds": 19})
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11435")
    monkeypatch.setenv("OLLAMA_MODEL", "phi3:mini")
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "result"}

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr(generator.requests, "post", post)
    assert generator.run_ollama("prompt") == "result"
    url, request = calls[0]
    assert url == "http://127.0.0.1:11435/api/generate"
    assert request["json"]["model"] == "phi3:mini"
    assert request["json"]["options"] == {"temperature": 0, "num_ctx": 4096, "seed": 7}
    assert request["timeout"] == 19
