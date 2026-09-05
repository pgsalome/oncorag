"""Shared graph/index and patient-isolation contracts for public chat sessions."""

from dataclasses import dataclass, field
import json
from pathlib import Path

import pytest

from oncoraggraph import chat_runtime, pipeline
from oncoraggraph.config.pipeline_config import validate_pipeline_config
from test_pipeline import fixture_config, local_models


@dataclass
class Answer:
    question: str
    answer: str
    status: str = "ok"
    citations: list = field(default_factory=list)
    reasoning: str = ""


@pytest.mark.parametrize("status", ["error", "invalid"])
def test_cli_failure_displays_reason(status, capsys):
    chat_runtime._show_response(Answer("Treatment?", "", status=status,
                                       reasoning="No verified evidence was returned."))
    output = capsys.readouterr().out
    assert status in output
    assert "No verified evidence was returned." in output


class Service:
    calls = []

    def __init__(self, feature_config_dir, **settings):
        self.settings = settings

    def answer_question(self, patient_id, graph, collection, question, *, history):
        self.calls.append((patient_id, question, history))
        assert graph.graph["patient_id"] == patient_id
        return Answer(question, f"Answer for {patient_id}")


def new_session(tmp_path, **overrides):
    config = fixture_config(tmp_path)
    config.update(overrides)
    Service.calls = []
    return chat_runtime.ChatSession(config, service_factory=Service)


def test_list_patients_does_not_load_models_or_generate_features(tmp_path):
    session = new_session(tmp_path)
    assert session.patient_ids == ["SYN-DEMO-001", "SYN-DEMO-002", "SYN-DEMO-003"]
    assert session.patient_id is None
    assert not Path(session.config["features"]["generated_config_dir"]).exists()
    with pytest.raises(ValueError, match="Select a patient"):
        session.ask("What treatment started?")


@pytest.mark.parametrize("language", ["english", "german", "mixed"])
def test_chat_reuses_extraction_graph_without_rebuilding(tmp_path, local_models, language):
    config = fixture_config(tmp_path, language)
    pipeline.run_pipeline(config, stage="graph")

    def no_rebuild(*args, **kwargs):
        raise AssertionError("Existing extraction graph must be reused")

    session = chat_runtime.ChatSession(config, graph_builder=no_rebuild, service_factory=Service)
    session.select_patient("SYN-DEMO-001")
    assert session.graph_path.exists()
    assert session.graph.graph["patient_id"] == "SYN-DEMO-001"
    assert session.collection.count() > 0
    assert session.ask("What treatment started?").status == "ok"
    session.ask("When did it start?")
    assert Service.calls[-1][2][-2]["content"] == "What treatment started?"
    assert len(session.history) == 4


def test_switch_patient_and_failed_switch_clear_everything(tmp_path, local_models):
    session = new_session(tmp_path)
    session.select_patient("SYN-DEMO-001")
    session.ask("Treatment?")
    session.select_patient("SYN-DEMO-002")
    assert session.history == []
    session.ask("Age?")
    assert Service.calls[-1][2] == []
    with pytest.raises(ValueError, match="not present"):
        session.select_patient("unknown")
    assert session.patient_id is None
    assert session.graph is None and session.collection is None and session._service is None
    assert session.history == []
    with pytest.raises(ValueError, match="Select a patient"):
        session.ask("When?")


def test_failed_backend_switch_cannot_retain_previous_patient(tmp_path, local_models):
    session = new_session(tmp_path)
    session.select_patient("SYN-DEMO-001")
    session.ask("Treatment?")
    session._indexer = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Backend unavailable"))
    with pytest.raises(RuntimeError, match="Backend unavailable"):
        session.select_patient("SYN-DEMO-002")
    assert session.patient_id is None and session.history == [] and session.graph is None


def test_history_is_bounded_copied_and_reset_without_unloading(tmp_path, local_models):
    session = new_session(tmp_path, chat={"history_turns": 1})
    session.select_patient("SYN-DEMO-001")
    session.ask("First question")
    session.ask("Second question")
    assert len(session.history) == 2
    exposed = session.history
    exposed[0]["content"] = "tampered"
    assert session.history[0]["content"] == "Second question"
    session.reset()
    assert session.history == [] and session.patient_id == "SYN-DEMO-001"
    session.close()
    assert session.patient_id is None and session.graph is None


def test_invalid_graph_cache_patient_fails_before_index(tmp_path, local_models):
    session = new_session(tmp_path)
    session.select_patient("SYN-DEMO-001")
    cache = session.graph_path
    data = json.loads(cache.read_text())
    data["graph"]["patient_id"] = "someone-else"
    cache.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="different patient"):
        session.select_patient("SYN-DEMO-001")
    assert session.patient_id is None and session.graph is None


def test_changed_notes_invalidate_shared_graph_cache(tmp_path, local_models):
    from test_pipeline import tiny_config
    config, note = tiny_config(tmp_path)
    session = chat_runtime.ChatSession(config, service_factory=Service)
    session.select_patient("patient1")
    original = session.graph_path
    note.write_text("Weight is 80 kg.")
    session.select_patient("patient1")
    assert session.graph_path != original


@pytest.mark.parametrize("backend", ["chroma", "iris"])
def test_configured_backend_and_namespace_are_forwarded(tmp_path, local_models, backend):
    config = fixture_config(tmp_path)
    config["vector_store"]["backend"] = backend
    captured = {}
    collection = object()

    def factory(patient_id, config):
        captured.update(patient_id=patient_id, config=config)
        return collection

    def indexer(graph, selected, entity_type_filter, *, replace):
        assert selected is collection and replace
        assert "Sentence" in entity_type_filter["required"]
        return selected

    session = chat_runtime.ChatSession(config, collection_factory=factory, indexer=indexer, service_factory=Service)
    session.select_patient("SYN-DEMO-001")
    assert captured["patient_id"] == "SYN-DEMO-001"
    assert captured["config"]["backend"] == backend
    assert captured["config"]["collection_namespace"] == config["vector_store"]["collection_namespace"]
    assert session._service.settings["runtime_config"] == config["runtime"]


@pytest.mark.parametrize("chat", [None, {"history_turns": True}, {"history_turns": -1},
                                 {"max_question_chars": 0}, {"max_history_chars": 0},
                                 {"feature_match_threshold": 2}])
def test_invalid_chat_limits_rejected_before_side_effects(tmp_path, chat):
    config = fixture_config(tmp_path)
    config["chat"] = chat
    with pytest.raises(ValueError):
        validate_pipeline_config(config)


def test_json_cli_model_backend_override(tmp_path, monkeypatch, capsys):
    config = fixture_config(tmp_path)
    path = tmp_path / "params.json"
    path.write_text(json.dumps(config))
    captured = {}

    class FakeSession:
        def __init__(self, params):
            captured.update(params)
        def select_patient(self, pid, **kwargs):
            assert pid == "SYN-DEMO-001"
        def ask(self, question):
            return Answer(question, "Temozolomide")
        def close(self):
            pass

    monkeypatch.setattr(chat_runtime, "ChatSession", FakeSession)
    monkeypatch.setenv("OLLAMA_MODEL", "environment-model")
    status = chat_runtime.main(["--config", str(path), "--patient-id", "SYN-DEMO-001",
                               "--question", "Treatment?", "--json", "--ollama-model", "explicit-model",
                               "--ollama-host", "http://127.0.0.1:11435", "--vector-backend", "iris"])
    assert status == 0
    assert json.loads(capsys.readouterr().out)["answer"] == "Temozolomide"
    assert captured["runtime"]["ollama"]["model"] == "explicit-model"
    assert captured["runtime"]["ollama"]["host"].endswith(":11435")
    assert captured["vector_store"]["backend"] == "iris"


@pytest.mark.parametrize("verbose", [False, True])
def test_json_cli_routes_loading_and_question_diagnostics_to_stderr(tmp_path, monkeypatch, capsys, verbose):
    from oncoraggraph.utils.logging_utils import log

    path = tmp_path / "params.json"
    path.write_text(json.dumps(fixture_config(tmp_path)))

    class NoisySession:
        def __init__(self, config):
            pass

        def select_patient(self, patient_id, **kwargs):
            log("Selection warning", level="WARNING")
            log("Selection details", level="INFO")
            print("Model initialization diagnostic")

        def ask(self, question):
            log("Retrieval diagnostic", level="ERROR")
            log("Retrieval details", level="INFO")
            return Answer(question, "A supported answer")

        def close(self):
            pass

    monkeypatch.setattr(chat_runtime, "ChatSession", NoisySession)
    arguments = ["--config", str(path), "--patient-id", "SYN-DEMO-001",
                 "--question", "Treatment?", "--json"]
    if verbose:
        arguments.append("--verbose")

    assert chat_runtime.main(arguments) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out)["answer"] == "A supported answer"
    assert "Selection warning" in captured.err
    assert "Retrieval diagnostic" in captured.err
    assert "Model initialization diagnostic" in captured.err
    assert ("Selection details" in captured.err) == verbose
    assert ("Retrieval details" in captured.err) == verbose
