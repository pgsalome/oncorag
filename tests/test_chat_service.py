"""Chat integration uses real fixture graphs/retrieval and explicit model doubles."""

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import spacy

from oncoraggraph import pipeline
from oncoraggraph.chat import ChatGraphService
from oncoraggraph.chat import service as service_module
from oncoraggraph.chat.medical_definitions import get_medical_definition
from oncoraggraph.chat.temporal_extraction import _extract_measurement, extract_temporal_data
from oncoraggraph.config.pipeline_config import load_pipeline_config
from oncoraggraph.graph import graph_builder
from oncoraggraph.models import model_init


ROOT = Path(__file__).resolve().parents[1]


class FakeEmbedding:
    def encode(self, texts, **kwargs):
        return np.asarray([[1.0, float("treatment" in text.lower()),
                            float("hemoglobin" in text.lower())] for text in texts])


class FakeCollection:
    def __init__(self, graph):
        self.ids = [node for node, attrs in graph.nodes(data=True) if attrs["label"] == "Sentence"]
        self.queries = []

    def count(self):
        return len(self.ids)

    def query(self, *, query_texts, n_results):
        self.queries.append(query_texts[0])
        return {"ids": [self.ids[:n_results]]}


class FakeClient:
    def __init__(self, response=None):
        self.response = response
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        if callable(self.response):
            value = self.response(kwargs)
        else:
            value = self.response
        return {"response": value if isinstance(value, str) else json.dumps(value)}


def evidence_from_call(call):
    return json.loads(call["prompt"].split("\nEVIDENCE: ", 1)[1].split("\nThe previous output", 1)[0])


def grounded_fixture_answer(call):
    context = evidence_from_call(call)
    entry = next((item for item in context if item["report_type"] == "treatment"), context[0])
    return {"answer": entry["text"], "reasoning": "Supported by this source note.",
            "evidence": [{"note_id": entry["note_id"], "quote": entry["text"]}]}


@pytest.fixture
def chat_setup(tmp_path, monkeypatch):
    nlp = spacy.blank("xx")
    nlp.add_pipe("sentencizer")
    monkeypatch.setattr(graph_builder, "get_scispacy_model", lambda name: nlp)
    monkeypatch.setattr(graph_builder, "extract_and_deduplicate_entities", lambda *args: [])
    monkeypatch.setattr(model_init, "initialize_models", lambda: None)
    monkeypatch.setattr(model_init, "CLINICAL_EMBEDDER", FakeEmbedding())
    monkeypatch.setattr(model_init, "get_combined_reranker_scores", lambda pairs, **kwargs: [0.5] * len(pairs))

    def build(variant="mixed", **chat_options):
        config = load_pipeline_config(ROOT / "configs" / f"oncorag_synthetic_{variant}.json")
        config["features"]["generated_config_dir"] = str(tmp_path / variant)
        config["retrieval"]["top_k"] = 30
        config["retrieval"]["candidate_entity_limit"] = 50
        config["runtime"]["ollama"]["validation_retries"] = 0
        specs, patients = pipeline.prepare_inputs(config)
        pipeline.prepare_features(config, specs)
        patient_id, notes = next(iter(patients.items()))
        graph = graph_builder.build_patient_graph(notes, model_configs=[{"name": "fixture", "entity_types": []}])
        collection = FakeCollection(graph)
        service = ChatGraphService(config["features"]["generated_config_dir"],
                                   runtime_config=config["runtime"], retrieval_config=config["retrieval"],
                                   chat_config=chat_options)
        service._client = FakeClient(grounded_fixture_answer)
        return service, patient_id, graph, collection, config
    return build


@pytest.mark.parametrize("variant", ["english", "german", "mixed"])
def test_fixture_chat_uses_shared_retrieval_and_real_source_citations(chat_setup, variant):
    service, patient, graph, collection, config = chat_setup(variant)
    response = service.answer_question(patient, graph, collection, "What treatment actually started?")
    assert response.status == "ok", response.reasoning
    assert collection.queries
    assert response.retrieval_info["query"] == collection.queries[-1]
    assert service.retrieval_config == config["retrieval"]
    notes = {attrs["note_id"]: attrs for _, attrs in graph.nodes(data=True) if attrs["label"] == "Note"}
    for citation in response.citations:
        note = notes[citation["note_id"]]
        assert citation["quote"] in note["text"]
        assert citation["date"] == note["note_date"] == citation["note_date"]
        assert citation["passage"] == citation["quote"]
        assert citation["language"] == note["language"]
    if variant == "mixed":
        assert {entry["language"] for entry in evidence_from_call(service._client.calls[0])} == {"en", "de"}


def test_true_followup_retrieves_again_and_uses_history_only_as_reference(chat_setup):
    service, patient, graph, collection, _ = chat_setup()
    first_question = "What treatment actually started?"
    first = service.answer_question(patient, graph, collection, first_question)
    history = [{"role": "user", "content": first_question},
               {"role": "assistant", "content": first.answer}]
    def date_answer(call):
        context = evidence_from_call(call)
        note = next(entry for entry in context if entry["report_type"] == "treatment")
        return {"answer": note["date"], "reasoning": "Date of the documented treatment note.",
                "evidence": [{"note_id": note["note_id"], "quote": note["text"]}]}
    service._client.response = date_answer
    followup = service.answer_question(patient, graph, collection, "When did it start? Use YYYY-MM-DD.", history=history)
    assert followup.status == "ok", followup.reasoning
    assert len(collection.queries) == 2
    assert first_question in collection.queries[-1]
    assert first_question in service._client.calls[-1]["prompt"]
    assert followup.answer == followup.citations[0]["date"]
    assert "not evidence" in service._client.calls[-1]["prompt"].lower()


@pytest.mark.parametrize("wrong", ["quote", "note_id", "uncited", "history_quote", "extra_field"])
def test_hallucinated_and_unsupported_responses_are_invalid(chat_setup, wrong):
    service, patient, graph, collection, _ = chat_setup()
    def invalid(call):
        result = grounded_fixture_answer(call)
        if wrong == "quote":
            result["evidence"][0]["quote"] = "Invented finding not in the source."
        elif wrong == "note_id":
            result["evidence"][0]["note_id"] = "different-patient-note"
        elif wrong == "uncited":
            result["evidence"] = []
        elif wrong == "extra_field":
            result["unapproved"] = "content"
        else:
            result["evidence"][0]["quote"] = "A historical assistant claimed a new diagnosis."
        return result
    service._client.response = invalid
    result = service.answer_question(patient, graph, collection, "What was documented?", history=[
        {"role": "assistant", "content": "A historical assistant claimed a new diagnosis."}])
    assert result.status == "invalid"
    assert not result.answer and not result.citations


def test_patient_mismatch_and_cross_patient_history_rejected_before_retrieval(chat_setup):
    service, patient, graph, collection, _ = chat_setup()
    assert service.answer_question("other-patient", graph, collection, "Question").status == "invalid"
    result = service.answer_question(patient, graph, collection, "Question", history=[
        {"patient_id": "other-patient", "role": "user", "content": "Secret"}])
    assert result.status == "invalid"
    assert not collection.queries and not service._client.calls
    graph.add_node("other-patient", label="Patient")
    assert service.answer_question(patient, graph, collection, "Question").status == "invalid"


def test_selected_citation_must_match_original_note(chat_setup, monkeypatch):
    service, patient, graph, collection, _ = chat_setup()
    note = next(attrs for _, attrs in graph.nodes(data=True) if attrs["label"] == "Note")
    monkeypatch.setattr(service_module, "retrieve_context", lambda *args: (
        [{"note_id": note["note_id"], "text": "Text not in the original source"}], {}))
    result = service.answer_question(patient, graph, collection, "Question")
    assert result.status == "invalid" and not result.citations
    assert not service._client.calls


@pytest.mark.parametrize("context", [None, [None], [{"note_id": []}], [{"note_id": "missing", "text": "quote"}]])
def test_malformed_retrieval_provenance_is_invalid(chat_setup, monkeypatch, context):
    service, patient, graph, collection, _ = chat_setup()
    monkeypatch.setattr(service_module, "retrieve_context", lambda *args: (context, {}))
    result = service.answer_question(patient, graph, collection, "Question")
    assert result.status == "invalid" and not service._client.calls


@pytest.mark.parametrize("raw", ["[]", '{"answer": NaN, "reasoning": "", "evidence": []}'])
def test_model_nonobject_and_nonfinite_json_are_invalid(chat_setup, raw):
    service, patient, graph, collection, _ = chat_setup()
    service._client.response = raw
    result = service.answer_question(patient, graph, collection, "Question")
    assert result.status == "invalid" and not result.citations


@pytest.mark.parametrize("exception", [RuntimeError("backend failure"), ValueError("backend failure")])
def test_backend_errors_remain_errors_not_missing(chat_setup, monkeypatch, exception):
    service, patient, graph, collection, _ = chat_setup()
    def fail(*args):
        raise exception
    monkeypatch.setattr(service_module, "retrieve_context", fail)
    response = service.answer_question(patient, graph, collection, "Question")
    assert response.status == "error"
    assert response.retrieval_info["error_type"] == type(exception).__name__
    assert not service._client.calls


def test_ollama_failure_does_not_leak_url_or_secret(chat_setup):
    service, patient, graph, collection, _ = chat_setup()
    service._client.response = ConnectionError("http://private-host?apiKey=secret")
    response = service.answer_question(patient, graph, collection, "Question")
    assert response.status == "error"
    assert "secret" not in str(response)
    assert "private-host" not in str(response)


def test_no_context_and_model_null_are_explicit_missing(chat_setup):
    service, patient, graph, collection, _ = chat_setup()
    service._client.response = {"answer": None, "reasoning": "No supported answer", "evidence": []}
    assert service.answer_question(patient, graph, collection, "Question").status == "missing"
    calls = len(service._client.calls)
    collection.ids = []
    assert service.answer_question(patient, graph, collection, "Question").status == "missing"
    assert len(service._client.calls) == calls


def test_native_ollama_receives_configured_options_and_schema(chat_setup, monkeypatch):
    import ollama
    service, patient, graph, collection, _ = chat_setup()
    client = service._client
    settings = service.runtime_config["ollama"]
    settings.update(host="http://localhost:11439", timeout_seconds=17, model="fixture-model",
                    temperature=0.1, num_ctx=8192, max_tokens=123)
    service.runtime_config["random_seed"] = 77
    constructed = []
    monkeypatch.setattr(ollama, "Client", lambda **kwargs: constructed.append(kwargs) or client)
    service._client = None
    assert service.answer_question(patient, graph, collection, "Question").status == "ok"
    assert constructed == [{"host": "http://localhost:11439", "timeout": 17}]
    call = client.calls[0]
    assert call["model"] == "fixture-model"
    assert call["options"] == {"temperature": 0.1, "num_ctx": 8192, "num_predict": 123, "seed": 77}
    assert call["format"]["required"] == ["answer", "reasoning", "evidence"]
    assert call["format"]["properties"]["evidence"]["items"]["properties"]["note_id"]["enum"]


def test_validation_retry_is_bounded_and_preserves_evidence(chat_setup):
    service, patient, graph, collection, _ = chat_setup()
    service.runtime_config["ollama"]["validation_retries"] = 1
    service._client.response = lambda call: "not json" if len(service._client.calls) == 1 else grounded_fixture_answer(call)
    response = service.answer_question(patient, graph, collection, "Question")
    assert response.status == "ok"
    assert len(service._client.calls) == 2
    assert evidence_from_call(service._client.calls[0]) == evidence_from_call(service._client.calls[1])


def test_history_and_questions_have_explicit_limits(chat_setup):
    service, patient, graph, collection, _ = chat_setup(history_turns=1, max_history_chars=8, max_question_chars=20)
    history = [{"role": "user", "content": "old-message"}, {"role": "assistant", "content": "0123456789"}]
    assert service._bounded_history(patient, history) == [{"role": "assistant", "content": "23456789"}]
    assert service.answer_question(patient, graph, collection, "x" * 21).status == "invalid"
    service.chat_config["history_turns"] = 0
    assert service._bounded_history(patient, history) == []


@pytest.mark.parametrize("variant", ["english", "german", "mixed"])
def test_temporal_chart_uses_selected_dated_source_evidence(chat_setup, variant):
    service, patient, graph, collection, _ = chat_setup(variant)
    question = "Show hemoglobin values over time" if variant == "english" else "H\u00e4moglobin Werte im Verlauf"
    result = service.answer_question(patient, graph, collection, question)
    assert result.status == "ok", result.reasoning
    assert result.temporal_data and result.temporal_data["series"]
    selected = evidence_from_call(service._client.calls[0])
    for series in result.temporal_data["series"]:
        assert series["name"] == "hemoglobin"
        for point in series["data"]:
            assert isinstance(point["value"], (int, float))
            assert any(point["source"] == entry["note_id"] and point["date"] == entry["date"]
                       for entry in selected)


def test_feature_matching_and_definitions_do_not_shadow_patient_citations(chat_setup):
    service, patient, graph, collection, _ = chat_setup()
    assert service._match_feature("latest hemoglobin") is not None
    service._feature_configs["definition"] = {"enrichment": {
        "normalized_name": "chemotherapy", "description": "A cached medical definition",
        "ontology_mappings": {"umls": [{"cui": "C123", "name": "chemotherapy"}]}}}
    result = service.answer_question(patient, graph, collection, "What chemotherapy was documented?")
    assert result.status == "ok"
    assert result.ontology_citations[0]["source"] == "UMLS"
    assert all("note_id" in citation and "quote" in citation for citation in result.citations)
    assert get_medical_definition("chemotherapy", {"empty": {"enrichment": {}}}) is None


@pytest.mark.parametrize("text,measurement,expected", [
    ("Height 170 cm, weight 70 kg.", "weight", (70, "kg")),
    ("Weight 70 kg, height 170 cm.", "height", (170, "cm")),
    ("Height 170 cm.", "weight", (None, None)),
    ("Weight 70 kg.", "height", (None, None)),
    ("Gewicht 70,5 kg.", "weight", (70.5, "kg")),
    ("Groesse 1,70 m, Gewicht 70,5 kg.", "height", (1.7, "m")),
    ("Height 68 inches; weight 154.5 lbs.", "height", (68, "inches")),
    ("Height 68 inches; weight 154.5 lbs.", "weight", (154.5, "lbs")),
    ("Height 5.8 ft; weight 154 lb.", "height", (5.8, "ft")),
    ("Height 68 in.", "height", (68, "in")),
    ("BMI 24,7 kg/m2. Weight 70,5 kg.", "bmi", (24.7, "kg/m2")),
    ("BMI 24,7 kg/m2.", "weight", (None, None)),
    ("BMI 24.", "bmi", (24, "kg/m2")),
    ("BMI unavailable; hemoglobin 12.4 g/dL.", "bmi", (None, None)),
    ("Weight 70 kg. Hemoglobin 12,4 g/dL.", "hemoglobin", (12.4, "g/dL")),
    ("Haemoglobin: 124 g/L.", "hemoglobin", (124, "g/L")),
    ("Haemoglobin 7,7 mmol/L.", "hemoglobin", (7.7, "mmol/L")),
    ("Hemoglobin measured today: 11.2 g/dL.", "hemoglobin", (11.2, "g/dL")),
    ("H\u00e4moglobin heute: 12,1 g/dL.", "hemoglobin", (12.1, "g/dL")),
    ("Hemoglobin measured today: unavailable; sodium 136 mmol/L.", "hemoglobin", (None, None)),
    ("H\u00e4moglobin heute: unbestimmt; Natrium 136 mmol/L.", "hemoglobin", (None, None)),
    ("Haemoglobin missing; sodium 136 mmol/L.", "hemoglobin", (None, None)),
    ("Temperature 37,5 \u00b0C; weight 70 kg.", "temperature", (37.5, "\u00b0C")),
    ("Temp 98.6 \u00b0F.", "temperature", (98.6, "\u00b0F")),
    ("Pulse 72 bpm. Weight 70 kg.", "heart rate", (72, "bpm")),
    ("Herzfrequenz 72 /min.", "heart rate", (72, "/min")),
    ("Blood pressure 120/80 mmHg; weight 70 kg.", "blood pressure", ("120/80", "mmHg")),
    ("Blutdruck 120/80.", "blood pressure", ("120/80", None)),
    ("Blood pressure not recorded. Ratio 120/80.", "blood pressure", (None, None)),
    ("Dose 80,5 mg/m2.", "dose", (80.5, "mg/m2")),
])
def test_temporal_measurement_value_and_unit_are_correctly_associated(text, measurement, expected):
    assert _extract_measurement(text, measurement) == expected


def test_temporal_chart_keeps_decimal_comma_and_metric_unit_together():
    graph = nx.Graph()
    graph.add_node("source", label="Sentence", text="Height 170 cm, Gewicht 70,5 kg.",
                   note_id="source-note", note_date="2020-01-01")
    series = extract_temporal_data(graph, ["source"], "weight")
    assert len(series) == 1 and len(series[0].data_points) == 1
    point = series[0].data_points[0]
    assert (point.value, point.unit, point.source, point.date) == (70.5, "kg", "source-note", "2020-01-01")


@pytest.mark.parametrize("variant", ["english", "german", "mixed"])
@pytest.mark.parametrize("patient_id,expected", [
    ("SYN-DEMO-001", [("2020-03-01", 12.4), ("2020-04-01", 11.2)]),
    ("SYN-DEMO-002", [("2021-05-10", 10.3), ("2021-06-03", 12.1)]),
    ("SYN-DEMO-003", [("2022-01-08", 14.8), ("2022-02-07", 13.5)]),
])
def test_fixture_timeline_preserves_both_baseline_and_latest_measurements(chat_setup, variant, patient_id, expected):
    service, _, _, _, config = chat_setup(variant)
    _, patients = pipeline.prepare_inputs(config)
    graph = graph_builder.build_patient_graph(patients[patient_id], model_configs=[{"name": "fixture", "entity_types": []}])
    response = service.answer_question(patient_id, graph, FakeCollection(graph), "Show hemoglobin values over time")
    assert response.status == "ok", response.reasoning
    series = response.temporal_data["series"]
    assert len(series) == 1 and series[0]["name"] == "hemoglobin"
    assert [(point["date"], point["value"]) for point in series[0]["data"]] == expected
    assert {point["unit"] for point in series[0]["data"]} == {"g/dL"}
    assert {point["source"] for point in series[0]["data"]} == {
        f"{patient_id}-oncology", f"{patient_id}-laboratory"}


def test_model_abstention_preserves_independently_sourced_fixture_timeline(chat_setup):
    service, patient, graph, collection, _ = chat_setup("mixed")
    service._client.response = {"answer": None, "reasoning": "The model abstained.", "evidence": []}
    result = service.answer_question(patient, graph, collection, "Plot hemoglobin values over time.")
    assert result.status == "missing" and not result.citations
    assert result.answer == "The model did not produce a supported answer from the retrieved notes."
    assert result.reasoning == "The model abstained."
    points = result.temporal_data["series"][0]["data"]
    assert [(point["date"], point["value"], point["unit"]) for point in points] == [
        ("2020-03-01", 12.4, "g/dL"), ("2020-04-01", 11.2, "g/dL")]
    selected = evidence_from_call(service._client.calls[0])
    assert all(any(point["source"] == entry["note_id"] and point["date"] == entry["date"]
                   for entry in selected) for point in points)


@pytest.mark.parametrize("model_result,status", [("not JSON", "invalid"), (ConnectionError("offline"), "error")])
def test_invalid_or_error_model_responses_do_not_expose_timeline(chat_setup, model_result, status):
    service, patient, graph, collection, _ = chat_setup("mixed")
    service._client.response = model_result
    result = service.answer_question(patient, graph, collection, "Plot hemoglobin values over time.")
    assert result.status == status and result.temporal_data is None and not result.citations


def test_temporal_chemotherapy_decimal_comma_has_its_matched_dose_unit():
    graph = nx.Graph()
    graph.add_node("source", label="Sentence", text="Paclitaxel 80,5 mg/m2 administered.",
                   note_id="source-note", note_date="2020-01-01")
    series = extract_temporal_data(graph, ["source"], "chemotherapy")
    point = series[0].data_points[0]
    assert (point.value, point.unit) == (80.5, "mg/m2")


@pytest.mark.parametrize("key,value", [
    ("host", "https://remote.example.com"),
    ("host", "http://user:password@localhost:11434"),
    ("host", "file:///tmp/model"),
    ("validation_retries", 4),
    ("validation_retries", -1),
    ("timeout_seconds", 0),
    ("num_ctx", 1),
])
def test_direct_service_enforces_shared_runtime_policy(chat_setup, key, value):
    service, _, _, _, _ = chat_setup()
    runtime = service.runtime_config
    runtime["ollama"][key] = value
    with pytest.raises(ValueError):
        ChatGraphService(service.feature_config_dir, runtime_config=runtime,
                         retrieval_config=service.retrieval_config)


@pytest.mark.parametrize("key,value", [("history_turns", 21), ("max_question_chars", 32001),
                                       ("max_history_chars", 128001), ("feature_match_threshold", 1.1)])
def test_direct_service_enforces_shared_chat_limits(chat_setup, key, value):
    service, _, _, _, _ = chat_setup()
    with pytest.raises(ValueError):
        ChatGraphService(service.feature_config_dir, runtime_config=service.runtime_config,
                         retrieval_config=service.retrieval_config, chat_config={key: value})


def test_direct_service_remote_host_requires_explicit_nonlocal_opt_in(chat_setup):
    service, _, _, _, _ = chat_setup()
    runtime = service.runtime_config
    runtime["ollama"]["host"] = "https://remote.example.com"
    runtime["local_processing_only"] = False
    configured = ChatGraphService(service.feature_config_dir, runtime_config=runtime,
                                  retrieval_config=service.retrieval_config)
    assert configured._client is None
