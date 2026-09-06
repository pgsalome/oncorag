"""Enforce the local processing policy before reading notes or initializing services."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from oncorag import chat_runtime, pipeline
from oncorag.config.pipeline_config import load_pipeline_config, validate_pipeline_config


ROOT = Path(__file__).resolve().parents[1]


def local_config(tmp_path):
    config = load_pipeline_config(ROOT / "configs" / "oncorag_synthetic_mixed.json")
    config["features"]["generated_config_dir"] = str(tmp_path / "generated")
    config["outputs"]["root"] = str(tmp_path / "outputs")
    config["vector_store"].update(backend="iris", iris={"host": "127.0.0.1"})
    return config


def set_service_host(config, service, host):
    if service == "IRIS":
        config["vector_store"]["iris"]["host"] = host
    else:
        authority = f"[{host}]" if ":" in host else host
        config["runtime"]["ollama"]["host"] = f"http://{authority}:11434"


@pytest.mark.parametrize("service", ["IRIS", "Ollama"])
@pytest.mark.parametrize("host", [
    "remote.example.org", "192.168.1.5", "10.0.0.5", "localhost.example.org",
    "127.0.0.1.example.org", "0.0.0.0", "::", "2001:db8::1",
])
def test_local_policy_rejects_nonloopback_services(tmp_path, service, host):
    config = local_config(tmp_path)
    set_service_host(config, service, host)
    with pytest.raises(ValueError, match=f"loopback {service} host"):
        validate_pipeline_config(config)


@pytest.mark.parametrize("service", ["IRIS", "Ollama"])
@pytest.mark.parametrize("host", ["localhost", "LOCALHOST", "127.0.0.1", "127.0.0.2", "::1"])
def test_local_policy_accepts_loopback_services(tmp_path, service, host):
    config = local_config(tmp_path)
    set_service_host(config, service, host)
    validate_pipeline_config(config)


def test_default_policy_applies_to_default_iris_host_and_rejects_remote_host(tmp_path):
    config = local_config(tmp_path)
    config["runtime"].pop("local_processing_only", None)
    config["vector_store"]["iris"] = {}
    validate_pipeline_config(config)
    config["vector_store"]["iris"]["host"] = "remote.example.org"
    with pytest.raises(ValueError, match="loopback IRIS host"):
        validate_pipeline_config(config)


def test_explicit_nonlocal_opt_in_allows_remote_services(tmp_path):
    config = local_config(tmp_path)
    config["runtime"]["local_processing_only"] = False
    for service in ("IRIS", "Ollama"):
        set_service_host(config, service, "remote.example.org")
    validate_pipeline_config(config)


def test_inactive_remote_iris_settings_do_not_change_chroma_locality(tmp_path):
    config = local_config(tmp_path)
    config["vector_store"].update(backend="chroma", iris={"host": "remote.example.org"})
    validate_pipeline_config(config)


@pytest.mark.parametrize("stage", ["validate", "config", "graph", "extract"])
def test_pipeline_rejects_remote_iris_before_reading_notes_or_creating_outputs(tmp_path, monkeypatch, stage):
    config = local_config(tmp_path)
    config["vector_store"]["iris"]["host"] = "remote.example.org"
    side_effects = [Mock() for _ in range(5)]
    for name, spy in zip(("prepare_inputs", "prepare_features", "prepare_patient_graph",
                          "prepare_patient_index", "OllamaExtractor"), side_effects):
        monkeypatch.setattr(pipeline, name, spy)
    with pytest.raises(ValueError, match="loopback IRIS host"):
        pipeline.run_pipeline(config, stage=stage)
    for spy in side_effects:
        spy.assert_not_called()
    assert not Path(config["outputs"]["root"]).exists()
    assert not Path(config["features"]["generated_config_dir"]).exists()


def test_chat_rejects_remote_iris_before_reading_notes(tmp_path, monkeypatch):
    config = local_config(tmp_path)
    config["vector_store"]["iris"]["host"] = "remote.example.org"
    prepare_inputs = Mock()
    monkeypatch.setattr(chat_runtime, "prepare_inputs", prepare_inputs)
    with pytest.raises(ValueError, match="loopback IRIS host"):
        chat_runtime.ChatSession(config)
    prepare_inputs.assert_not_called()


@pytest.mark.parametrize("service", ["IRIS", "Ollama"])
def test_chat_revalidates_changed_service_before_patient_selection(tmp_path, monkeypatch, service):
    session = chat_runtime.ChatSession(local_config(tmp_path))
    session.patient_id = "previous-patient"
    session.graph = object()
    session.collection = object()
    session._history = [{"role": "user", "content": "Earlier question"}]
    set_service_host(session.config, service, "remote.example.org")
    side_effects = [Mock() for _ in range(4)]
    for name, spy in zip(("prepare_inputs", "prepare_features", "prepare_patient_graph",
                          "prepare_patient_index"), side_effects):
        monkeypatch.setattr(chat_runtime, name, spy)
    with pytest.raises(ValueError, match=f"loopback {service} host"):
        session.select_patient("SYN-DEMO-001")
    for spy in side_effects:
        spy.assert_not_called()
    assert session.patient_id is None and session.graph is None and session.collection is None
    assert session.history == []


@pytest.mark.parametrize("interface", ["pipeline", "chat"])
def test_cli_backend_override_cannot_activate_remote_iris_under_local_policy(tmp_path, monkeypatch, capsys, interface):
    config = local_config(tmp_path)
    config["vector_store"].update(backend="chroma", iris={"host": "remote.example.org"})
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    module = pipeline if interface == "pipeline" else chat_runtime
    prepare_inputs = Mock()
    prepare_index = Mock()
    monkeypatch.setattr(module, "prepare_inputs", prepare_inputs)
    monkeypatch.setattr(module, "prepare_patient_index", prepare_index)
    args = ["--config", str(config_path), "--vector-backend", "iris"]
    if interface == "pipeline":
        with pytest.raises(ValueError, match="loopback IRIS host"):
            pipeline.main(args)
    else:
        assert chat_runtime.main(args + ["--patient-id", "SYN-DEMO-001", "-q", "Treatment?"]) == 1
        assert "loopback IRIS host" in capsys.readouterr().err
    prepare_inputs.assert_not_called()
    prepare_index.assert_not_called()


def test_remote_iris_opt_in_reaches_configured_backend(tmp_path, monkeypatch):
    import networkx as nx

    config = local_config(tmp_path)
    config["runtime"]["local_processing_only"] = False
    config["vector_store"]["iris"]["host"] = "remote.example.org"
    monkeypatch.setattr(pipeline, "seed_runtime", lambda runtime: None)
    collection_factory = Mock(return_value=object())
    result = pipeline.run_pipeline(
        config, graph_builder=lambda notes, **kwargs: nx.Graph(),
        collection_factory=collection_factory,
        indexer=lambda graph, collection, filters, **kwargs: collection,
        retriever=lambda *args: ([], {}), extractor=Mock(),
    )
    assert result["patients"] == 3 and result["failures"] == 0
    assert collection_factory.call_count == 3
    for call in collection_factory.call_args_list:
        assert call.kwargs["config"]["backend"] == "iris"
        assert call.kwargs["config"]["iris"]["host"] == "remote.example.org"
