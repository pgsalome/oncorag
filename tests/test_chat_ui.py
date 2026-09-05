"""Optional Streamlit UI coverage without model or network requests."""

from pathlib import Path
from types import SimpleNamespace
import json

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from oncoraggraph import chat_app


ROOT = Path(__file__).resolve().parents[1]


class FakeChatSession:
    instances = []

    def __init__(self, config):
        self.config = config
        self.patient_ids = ["synthetic-en", "synthetic-de", "unavailable"]
        self.patient_id = None
        self.history = []
        self.closed = False
        self.force_rebuild = False
        self.instances.append(self)

    def select_patient(self, patient_id, force_rebuild=False):
        self.patient_id = None
        self.history.clear()
        if patient_id == "unavailable":
            raise ValueError("Synthetic loading failure")
        self.patient_id = patient_id
        self.force_rebuild = force_rebuild

    def ask(self, question):
        self.history.extend([{"role": "user", "content": question},
                             {"role": "assistant", "content": "Weight is 70 kg."}])
        return SimpleNamespace(
            answer="Weight is 70 kg.", reasoning="A documented measurement.", status="ok",
            citations=[{"note_id": "synthetic-note", "date": "2025-01-02", "report_type": "oncology",
                        "quote": "Weight is 70 kg.", "language": "en"}], retrieval_info={},
            temporal_data={"series": [{"name": "weight", "unit": "kg", "data": [
                {"date": "2025-01-02", "value": 70, "source": "synthetic-note", "context": "Weight is 70 kg."},
                {"date": "2025-01-03", "value": 71},
            ]}]}, medical_definitions={"weight": "Body weight"},
            ontology_citations=[{"name": "Synthetic concept"}],
        )

    def reset(self):
        self.history.clear()

    def close(self):
        self.closed = True
        self.patient_id = None


@pytest.fixture(autouse=True)
def isolated_ollama_environment(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)


@pytest.fixture
def app(monkeypatch):
    FakeChatSession.instances = []
    monkeypatch.setattr(chat_app, "_create_session", FakeChatSession)
    result = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=10).run()
    assert not result.exception
    return result


def choose_and_ask(app):
    app.selectbox[0].select("synthetic-en").run()
    assert not app.chat_input[0].disabled
    app.chat_input[0].set_value("What is the weight?").run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert not next(button for button in app.button if button.label == "Clear conversation").disabled


def test_chat_ui_requires_patient_then_renders_answer_evidence_and_timeline(app):
    assert app.chat_input[0].disabled
    assert app.title[0].value == "OncoRAG Chat"
    choose_and_ask(app)
    assert any(item.value == "Weight is 70 kg." for item in app.markdown)
    assert any(item.value == "Weight is 70 kg." for item in app.text)
    assert any(item.value == "synthetic-note | 2025-01-02 | oncology | en" for item in app.caption)
    assert {item.label for item in app.expander} >= {"Evidence", "Reasoning", "Timeline", "Medical definitions", "Ontology sources"}
    assert len(app.dataframe) == 1
    assert app.dataframe[0].value.loc[0, "source"] == "synthetic-note"
    assert app.dataframe[0].value.loc[0, "context"] == "Weight is 70 kg."
    assert len(app.get("arrow_vega_lite_chart")) == 1
    assert len(app.session_state["oncorag_chat_state"].session.history) == 2


def test_model_abstention_keeps_source_backed_timeline_without_successful_narrative(app, monkeypatch):
    original_ask = FakeChatSession.ask
    explanation = "The model did not produce a supported narrative answer."

    def abstain(self, question):
        response = original_ask(self, question)
        response.status = "missing"
        response.answer = explanation
        response.reasoning = "Source-backed measurements were independently parsed."
        response.citations = []
        return response

    monkeypatch.setattr(FakeChatSession, "ask", abstain)
    choose_and_ask(app)

    assert any(item.value == explanation for item in app.warning)
    assert not any(item.value in {explanation, "Weight is 70 kg."} for item in app.markdown)
    assert not app.error
    assert len(app.get("arrow_vega_lite_chart")) == 1
    assert len(app.dataframe) == 1
    row = app.dataframe[0].value.iloc[0]
    assert row["value"] == 70
    assert row["source"] == "synthetic-note"
    assert row["context"] == "Weight is 70 kg."


def test_patient_switch_clears_transcript_and_runtime_history(app):
    choose_and_ask(app)
    app.selectbox[0].select("synthetic-de").run()
    assert not app.exception
    assert not app.chat_message
    assert app.session_state["oncorag_chat_state"].session.history == []
    assert app.session_state["oncorag_chat_state"].session.patient_id == "synthetic-de"


def test_failed_patient_load_clears_old_state_and_disables_questions(app):
    choose_and_ask(app)
    app.selectbox[0].select("unavailable").run()
    assert not app.exception
    assert not app.chat_message
    assert app.chat_input[0].disabled
    state = app.session_state["oncorag_chat_state"]
    assert state.session.patient_id is None
    assert state.session.history == []
    assert "Synthetic loading failure" in app.error[0].value


def test_failed_configuration_change_discards_patient_and_conversation(app, tmp_path):
    choose_and_ask(app)
    previous = app.session_state["oncorag_chat_state"].session
    app.text_input[0].set_value(str(tmp_path / "missing.json")).run()
    assert not app.exception
    assert previous.closed
    assert previous.history == []
    assert not app.chat_message
    assert app.chat_input[0].disabled
    assert app.session_state["oncorag_chat_state"].session is None


def test_clear_conversation_preserves_selected_patient(app):
    choose_and_ask(app)
    next(button for button in app.button if button.label == "Clear conversation").click().run()
    assert not app.exception
    assert not app.chat_message
    assert not app.chat_input[0].disabled
    assert next(button for button in app.button if button.label == "Clear conversation").disabled
    assert app.session_state["oncorag_chat_state"].session.patient_id == "synthetic-en"
    assert app.session_state["oncorag_chat_state"].session.history == []


def test_question_failure_persists_after_rerun_and_can_be_cleared(app, monkeypatch):
    def fail(self, question):
        raise RuntimeError("Synthetic model failure")

    monkeypatch.setattr(FakeChatSession, "ask", fail)
    app.selectbox[0].select("synthetic-en").run()
    app.chat_input[0].set_value("What is the weight?").run()

    assert not app.exception
    assert len(app.chat_message) == 2
    assert "Synthetic model failure" in app.error[0].value
    assert not app.chat_input[0].disabled
    clear_button = next(button for button in app.button if button.label == "Clear conversation")
    assert not clear_button.disabled

    app.run()
    assert not app.exception
    assert "Synthetic model failure" in app.error[0].value
    next(button for button in app.button if button.label == "Clear conversation").click().run()
    assert not app.chat_message
    assert not app.error


def test_followup_rerun_keeps_patient_selector_and_session_stable(app):
    choose_and_ask(app)
    previous = app.session_state["oncorag_chat_state"].session
    app.chat_input[0].set_value("On what date?").run()

    assert not app.exception
    assert len(app.chat_message) == 4
    assert app.selectbox[0].label == "Patient"
    assert app.selectbox[0].value == "synthetic-en"
    assert app.session_state["oncorag_chat_state"].session is previous
    assert len(previous.history) == 4


def test_reloading_patient_rebuilds_and_clears_conversation(app):
    choose_and_ask(app)
    next(button for button in app.button if button.label == "Reload patient").click().run()
    assert not app.exception
    assert not app.chat_message
    assert not app.chat_input[0].disabled
    assert app.session_state["oncorag_chat_state"].session.force_rebuild
    assert app.session_state["oncorag_chat_state"].session.history == []


def test_valid_configuration_change_does_not_reselect_the_previous_patient(app, tmp_path):
    choose_and_ask(app)
    previous = app.session_state["oncorag_chat_state"].session
    config = json.loads((ROOT / "configs/oncorag_synthetic_mixed.json").read_text())
    config["cohort"]["name"] = "another-synthetic-cohort"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    app.text_input[0].set_value(str(path)).run()

    assert not app.exception
    assert previous.closed
    assert previous.history == []
    assert not app.chat_message
    assert app.selectbox[0].value is None
    assert app.chat_input[0].disabled
    assert app.session_state["oncorag_chat_state"].session is not previous


def test_app_sessions_do_not_share_patient_or_transcript(app):
    choose_and_ask(app)
    other = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=10).run()
    assert not other.exception
    assert not other.chat_message
    assert other.chat_input[0].disabled
    assert other.session_state["oncorag_chat_state"].session is not app.session_state["oncorag_chat_state"].session


def test_config_signature_changes_when_file_contents_change(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"value": 1}')
    previous = chat_app._config_signature(path)
    path.write_text('{"value": 2}')
    assert chat_app._config_signature(path) != previous


def test_invalid_path_signature_is_reported_without_preserving_old_session():
    assert chat_app._config_signature("\x00")[1] == "ValueError"


def test_ui_cli_overrides_reach_session_config(monkeypatch):
    monkeypatch.setattr(chat_app, "_create_session", FakeChatSession)
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11436")
    monkeypatch.setenv("OLLAMA_MODEL", "environment-model")
    args = ["--config", str(ROOT / "configs/oncorag_synthetic_mixed.json"),
            "--ollama-host", "http://127.0.0.1:11435", "--ollama-model", "synthetic-model"]
    app = AppTest.from_string(f"from oncoraggraph.chat_app import main\nmain({args!r})").run()

    assert not app.exception
    state = app.session_state["oncorag_chat_state"]
    assert state.session.config["runtime"]["ollama"]["host"] == "http://127.0.0.1:11435"
    assert state.session.config["runtime"]["ollama"]["model"] == "synthetic-model"
    assert state.signature[1:] == ("http://127.0.0.1:11435", "synthetic-model")


def test_ui_environment_overrides_reach_session_and_change_signature(monkeypatch):
    monkeypatch.setattr(chat_app, "_create_session", FakeChatSession)
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11435")
    monkeypatch.setenv("OLLAMA_MODEL", "environment-model")
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=10).run()
    assert not app.exception
    previous = app.session_state["oncorag_chat_state"].session
    assert previous.config["runtime"]["ollama"]["host"] == "http://127.0.0.1:11435"
    assert previous.config["runtime"]["ollama"]["model"] == "environment-model"

    monkeypatch.setenv("OLLAMA_MODEL", "new-environment-model")
    app.run()

    assert not app.exception
    state = app.session_state["oncorag_chat_state"]
    assert previous.closed
    assert state.session is not previous
    assert state.signature[1:] == ("http://127.0.0.1:11435", "new-environment-model")
    assert state.session.config["runtime"]["ollama"]["model"] == "new-environment-model"


@pytest.mark.parametrize("status", ["invalid", "error"])
@pytest.mark.parametrize("reasoning", ["Evidence validation failed.", ""])
def test_invalid_and_error_responses_render_visible_failure_not_unvalidated_payload(status, reasoning):
    payload = {"status": status, "answer": "", "reasoning": reasoning,
               "citations": [{"quote": "Rejected quote must not be shown."}]}
    app = AppTest.from_string(
        "from types import SimpleNamespace\nfrom oncoraggraph.chat_app import _render_response\n"
        f"_render_response(SimpleNamespace(**{payload!r}))"
    ).run()

    assert not app.exception
    assert len(app.error) == 1
    assert app.error[0].value == (reasoning or "No answer was produced.")
    assert not app.expander
    assert not app.text


def test_ui_cli_overrides_are_validated_before_session_creation(monkeypatch):
    monkeypatch.setattr(chat_app, "_create_session", FakeChatSession)
    state = chat_app.ChatUIState()
    state.configure(ROOT / "configs/oncorag_synthetic_mixed.json", "signature",
                    ollama_host="http://external.invalid:11434")

    assert state.session is None
    assert state.error


def test_ui_starts_with_real_synthetic_registry_without_loading_patient_models():
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=10).run()

    assert not app.exception
    assert not app.error
    assert len(app.selectbox[0].options) == 3
    assert app.chat_input[0].disabled
    session = app.session_state["oncorag_chat_state"].session
    assert session.patient_id is None
    assert session.graph is None
    assert session.collection is None
    assert session.history == []


@pytest.mark.parametrize("data", [None, [], {"series": None}, {"series": [None, {"data": None}]}])
def test_malformed_optional_timeline_data_has_no_rows(data):
    assert chat_app._temporal_rows(data) == []
