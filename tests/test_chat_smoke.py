import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("chat_smoke", ROOT / "scripts/run_chat_smoke.py")
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class FakeSession:
    configurations = []

    def __init__(self, config):
        self.configurations.append(copy.deepcopy(config))
        self.notes = smoke.load_notes(registry_path=config["inputs"]["registry_path"])
        self.patient_ids = sorted({note.patient_id for note in self.notes})
        self.patient_id = None
        self.history = []
        self.treatments = {"SYN-DEMO-001": "temozolomide", "SYN-DEMO-002": "bevacizumab"}

    def select_patient(self, patient_id):
        self.patient_id = patient_id
        self.history = []

    def ask(self, question):
        note = next(note for note in self.notes if note.patient_id == self.patient_id and note.report_type == "treatment")
        answer = note.date if "YYYY-MM-DD" in question else self.treatments[self.patient_id]
        self.history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
        return SimpleNamespace(answer=answer, reasoning="", status="ok", citations=[{
            "note_id": note.note_id, "quote": note.text.strip(), "date": note.date,
            "report_type": note.report_type, "language": note.language,
        }])


def config(variant="english"):
    return smoke.load_pipeline_config(ROOT / f"configs/oncorag_synthetic_{variant}.json")


@pytest.mark.parametrize("variant", ["english", "german", "mixed"])
def test_chat_sequence_covers_follow_up_and_patient_switch(variant):
    result = smoke.run_variant(config(variant), session_factory=FakeSession)
    assert result["passed"] is True
    assert result["turn_count"] == 3
    assert [turn["patient_id"] for turn in result["turns"]] == ["SYN-DEMO-001", "SYN-DEMO-001", "SYN-DEMO-002"]
    assert result["turns"][1]["response"]["answer"] == "2020-03-15"


def test_patient_switch_must_clear_history():
    class StickySession(FakeSession):
        def select_patient(self, patient_id):
            self.patient_id = patient_id

    result = smoke.run_variant(config(), session_factory=StickySession)
    assert result["passed"] is False
    assert "Selecting a patient did not reset conversation state" in result["failures"]


@pytest.mark.parametrize("problem", ["cross_patient", "memory_only", "fabricated_quote", "changed_quote_case", "wrong_date", "wrong_language"])
def test_invalid_citation_provenance_fails_smoke(problem):
    class InvalidCitationSession(FakeSession):
        def ask(self, question):
            response = super().ask(question)
            citation = response.citations[0]
            if problem == "cross_patient":
                citation["note_id"] = "SYN-DEMO-003-treatment"
            elif problem == "memory_only":
                citation["note_id"] = "conversation_history"
            elif problem == "fabricated_quote":
                citation["quote"] = "The prior assistant stated this fact."
            elif problem == "changed_quote_case":
                citation["quote"] = citation["quote"].upper()
            elif problem == "wrong_date":
                citation["date"] = "1999-01-01"
            else:
                citation["language"] = "unknown"
            return response

    result = smoke.run_variant(config(), session_factory=InvalidCitationSession)
    assert result["passed"] is False
    assert all(not turn["checks_passed"] for turn in result["turns"])


@pytest.mark.parametrize("status", ["", "missing", "unknown", "error"])
def test_expected_evidence_fixture_requires_success_status(status):
    class WrongStatusSession(FakeSession):
        def ask(self, question):
            response = super().ask(question)
            response.status = status
            return response

    result = smoke.run_variant(config(), session_factory=WrongStatusSession)
    assert result["passed"] is False
    assert "Response did not contain a supported answer" in result["failures"]


def test_wrong_treatment_and_followup_date_fail_even_with_correct_citation():
    class WrongAnswerSession(FakeSession):
        def ask(self, question):
            response = super().ask(question)
            response.answer = "2020-03-01" if "YYYY-MM-DD" in question else "radiotherapy"
            return response

    result = smoke.run_variant(config(), session_factory=WrongAnswerSession)
    assert result["passed"] is False
    assert any("start date" in failure for failure in result["failures"])
    assert any("expected started treatment" in failure for failure in result["failures"])


def test_smoke_propagates_runtime_and_keeps_patient_output_scopes_separate(tmp_path):
    FakeSession.configurations = []
    summary = smoke.run_smoke("http://127.0.0.1:11435", "local-test-model", tmp_path, session_factory=FakeSession)
    assert summary["passed"] is True
    assert set(summary["variants"]) == {"english", "german", "mixed"}
    assert sum(result["turn_count"] for result in summary["variants"].values()) == 9
    assert len({cfg["outputs"]["root"] for cfg in FakeSession.configurations}) == 3
    for cfg in FakeSession.configurations:
        assert cfg["runtime"]["ollama"]["host"] == "http://127.0.0.1:11435"
        assert cfg["runtime"]["ollama"]["model"] == "local-test-model"
    assert json.loads((tmp_path / "summary.json").read_text())["passed"] is True


def test_model_failure_is_recorded_for_all_variants_and_fails_run(tmp_path):
    class BrokenSession(FakeSession):
        def ask(self, question):
            raise RuntimeError("Model unavailable")

    summary = smoke.run_smoke("http://127.0.0.1:11434", "missing-model", tmp_path, session_factory=BrokenSession)
    assert summary["passed"] is False
    assert all(not row["passed"] for row in summary["variants"].values())
    assert all("Model unavailable" in row["failures"][0] for row in summary["variants"].values())


def test_remote_model_host_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="loopback"):
        smoke.run_smoke("https://external.invalid", "model", tmp_path, session_factory=FakeSession)


@pytest.mark.parametrize("passed,code", [(True, 0), (False, 1)])
def test_cli_exit_code_matches_checks(monkeypatch, tmp_path, passed, code):
    monkeypatch.setattr(smoke, "run_smoke", lambda *args: {"passed": passed})
    assert smoke.main(["--output-dir", str(tmp_path)]) == code
