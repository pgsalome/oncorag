"""Pipeline integration tests with real graphs/storage and explicit model doubles."""

import importlib
import json
from pathlib import Path
from unittest.mock import Mock

from chromadb.api.types import EmbeddingFunction
import networkx as nx
import numpy as np
import pytest
import spacy

from oncorag import pipeline
from oncorag.config.pipeline_config import load_pipeline_config
from oncorag.graph import graph_builder
from oncorag.llm import prompt_builder
from oncorag.models import model_init
from oncorag.vector_store.backend import get_vector_collection


ROOT = Path(__file__).resolve().parents[1]


class DeterministicEmbedding(EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, input):
        return self.encode(input).tolist()

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append([
                1.0,
                float("diagnos" in lower),
                float("hemoglobin" in lower or "h\u00e4moglobin" in lower),
                float("treatment" in lower or "behandlung" in lower),
                float("age" in lower or "alter" in lower),
            ])
        return np.asarray(vectors, dtype=np.float32)

    @staticmethod
    def name():
        return "oncorag_pipeline_test_embedding"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return DeterministicEmbedding()


@pytest.fixture
def local_models(monkeypatch):
    nlp = spacy.blank("xx")
    nlp.add_pipe("sentencizer")
    embedding = DeterministicEmbedding()
    monkeypatch.setattr(graph_builder, "get_scispacy_model", lambda name: nlp)
    monkeypatch.setattr(graph_builder, "extract_and_deduplicate_entities", lambda *args: [])
    monkeypatch.setattr(model_init, "get_chroma_embedding_function", lambda: embedding)
    chroma_module = importlib.import_module("oncorag.chroma.chroma_index")
    monkeypatch.setattr(chroma_module, "get_chroma_embedding_function", lambda: embedding)
    monkeypatch.setattr(model_init, "initialize_models", lambda: None)
    monkeypatch.setattr(model_init, "CLINICAL_EMBEDDER", embedding)
    monkeypatch.setattr(model_init, "get_combined_reranker_scores", lambda pairs, **kwargs: [0.5] * len(pairs))
    return embedding


def fixture_config(tmp_path, variant="mixed"):
    config = load_pipeline_config(ROOT / "configs" / f"oncorag_synthetic_{variant}.json")
    config["features"]["generated_config_dir"] = str(tmp_path / "generated")
    config["outputs"]["root"] = str(tmp_path / "outputs")
    config["vector_store"]["chroma"] = {"path": str(tmp_path / "chroma")}
    config["retrieval"]["top_k"] = 20
    config["retrieval"]["candidate_entity_limit"] = 50
    return config


def prompt_parts(prompt):
    feature = json.loads(prompt.split("\nFEATURE: ", 1)[1].split("\nTEMPORAL POLICY: ", 1)[0])
    evidence = json.loads(prompt.split("\nEVIDENCE: ", 1)[1])
    return feature, evidence


@pytest.mark.parametrize("variant", ["english", "german", "mixed"])
def test_bundled_fixture_pipeline_uses_real_graphs_chroma_and_typed_validation(tmp_path, local_models, variant):
    config = fixture_config(tmp_path, variant)
    gold_rows = [json.loads(line) for line in Path(config["evaluation"]["gold_path"]).read_text().splitlines()]
    gold = {(row["patient_id"], row["feature"]): row for row in gold_rows}
    prompts = []

    def fixture_model(prompt):
        # This is a model double, not an extraction-accuracy benchmark.
        feature, context = prompt_parts(prompt)
        prompts.append(context)
        patients = {entry["note_id"].rsplit("-", 1)[0] for entry in context}
        assert len(patients) == 1
        expected = gold[(patients.pop(), feature["name"])]
        candidates = [entry for entry in context if entry["note_id"] in expected["evidence_note_ids"]]
        assert candidates, "The actual retriever must retain the expected source note"
        raw_value = expected["value"]
        if feature["type"] in {"date", "integer", "numeric"}:
            candidates = [entry for entry in candidates if (
                str(raw_value).replace(".", ",") if entry["language"] == "de" and feature["type"] == "numeric"
                else str(raw_value)
            ) in entry["text"]]
            assert candidates, "The source value must survive sentence selection intact"
        evidence = candidates[0]
        if feature["type"] == "numeric":
            raw_value = str(raw_value).replace(".", ",") if evidence["language"] == "de" else str(raw_value)
        return {"value": raw_value, "confidence": "High", "reasoning": "Fixture model response",
                "evidence": [{"note_id": evidence["note_id"], "quote": evidence["text"]}]}

    result = pipeline.run_pipeline(config, extractor=fixture_model)

    assert result["failures"] == 0
    assert (result["patients"], result["notes"], result["features"]) == (3, 9, 4)
    assert len(result["graphs"]) == 3
    assert len(prompts) == 12
    assert {(row["patient_id"], row["feature"]): row["value"] for row in result["results"]} == {
        key: row["value"] for key, row in gold.items()
    }
    for row in result["results"]:
        assert row["status"] == "ok"
        if row["feature"] == "age_at_diagnosis":
            assert type(row["value"]) is int
        elif row["feature"] == "latest_hemoglobin":
            assert type(row["value"]) is float
    for graph_path in result["graphs"]:
        graph = nx.node_link_graph(json.loads(Path(graph_path).read_text()), edges="links")
        assert graph.graph["note_count"] == 3
        assert graph.graph["languages"] == ({"english": ["en"], "german": ["de"], "mixed": ["de", "en"]}[variant])
        notes = [attrs for _, attrs in graph.nodes(data=True) if attrs["label"] == "Note"]
        assert {attrs["report_type"] for attrs in notes} == {"oncology", "treatment", "laboratory"}
        assert len({attrs["note_id"] for attrs in notes}) == 3
        collection = get_vector_collection(graph.graph["patient_id"], config["vector_store"])
        assert collection.count() == sum(attrs["label"] == "Sentence" for _, attrs in graph.nodes(data=True))
        assert collection.query(query_texts=["hemoglobin"], n_results=1)["ids"][0]
    generated = json.loads((Path(config["features"]["generated_config_dir"]) / "latest_hemoglobin.json").read_text())
    assert generated["feature"]["expected_range"] == {"min": 0, "max": 30}
    assert generated["config_generation"]["mode"] == "manual"
    persisted = json.loads((Path(config["outputs"]["root"]) / "structured_features.json").read_text())
    assert persisted == result


def tiny_config(tmp_path, **feature_overrides):
    note = tmp_path / "notes/patient1/oncology/2025-01-02.txt"
    note.parent.mkdir(parents=True)
    note.write_text("Weight is 70 kg. Fatigue is documented.", encoding="utf-8")
    spec = {"name": "weight", "type": "numeric", "expected_range": [0, 300],
            "description": "Documented weight in kg", **feature_overrides}
    specs_path = tmp_path / "features.json"
    specs_path.write_text(json.dumps({"features": [spec]}))
    config = fixture_config(tmp_path, "english")
    config["inputs"] = {"notes_root": str(tmp_path / "notes")}
    config["features"]["specifications"] = str(specs_path)
    config["retrieval"]["graph_diffusion"]["enabled"] = False
    return config, note


def supported_response(value=70, **overrides):
    return {"value": value, "confidence": "High", "evidence": [
        {"note_id": "oncology/2025-01-02", "quote": "Weight is 70 kg."},
    ], **overrides}


@pytest.mark.parametrize("example_location", ["examples", "enrichment"])
def test_generated_guidance_reaches_extraction_and_retries(tmp_path, local_models, monkeypatch, example_location):
    config, _ = tiny_config(tmp_path)
    prepare_features = pipeline.prepare_features
    guidance = {
        "rules": {"extraction_guidelines": ["Use the most recent measured weight."]},
        "output_format": {"type": "numeric", "unit": "kg"},
    }
    examples = [{"context": "Example weight is 90 kg.", "value": 90}]
    if example_location == "enrichment":
        guidance["ehr_examples"] = ["Example weight is 90 kg."]
    else:
        guidance["examples"] = examples

    def enriched_features(config, specs):
        features = prepare_features(config, specs)
        features["weight"].update({key: value for key, value in guidance.items() if key != "ehr_examples"})
        if example_location == "enrichment":
            features["weight"]["enrichment"]["ehr_examples"] = guidance["ehr_examples"]
        return features

    monkeypatch.setattr(pipeline, "prepare_features", enriched_features)
    extractor = Mock(side_effect=[
        supported_response(value=90, evidence=[{
            "note_id": "oncology/2025-01-02", "quote": "Example weight is 90 kg.",
        }]),
        supported_response(),
    ])

    result = pipeline.run_pipeline(config, extractor=extractor)

    assert result["failures"] == 0
    assert result["results"][0]["value"] == 70
    assert result["results"][0]["attempts"] == 2
    for call in extractor.call_args_list:
        prompt = call.args[0]
        actual = json.loads(prompt.split("\nFEATURE CONFIGURATION: ", 1)[1].split("\nFEATURE: ", 1)[0])
        assert actual == guidance
        assert "cite only EVIDENCE" in prompt
    assert "Evidence quote or note ID is not in retrieved context" in extractor.call_args.args[0]


def test_extraction_prompt_keeps_declared_labels_and_generated_option_mapping():
    spec = {"name": "treatment", "type": "categorical",
            "expected_range": ["chemotherapy", "radiotherapy"]}
    config = {"output_format": {"options": {
        "A": "chemotherapy", "B": "radiotherapy", "C": "Missing",
    }}, "top_cuis": ["unused-ontology-payload"]}
    context = [{"note_id": "n1", "text": "Radiotherapy started."}]

    prompt = pipeline.extraction_prompt(spec, context, {}, config)

    assert prompt_parts(prompt) == (spec, context)
    assert json.dumps(config["output_format"]) in prompt
    assert "unused-ontology-payload" not in prompt
    assert "FEATURE defines the authoritative output type" in prompt
    assert "JSON null for missing values" in prompt
    with pytest.raises(ValueError, match="allowed category"):
        pipeline.validate_extraction({"value": "B"}, spec, context)


def test_extraction_prompt_accepts_missing_feature_configuration():
    spec = {"name": "weight", "type": "numeric"}
    prompt = pipeline.extraction_prompt(spec, [], {})
    assert "\nFEATURE CONFIGURATION: {}\nFEATURE: " in prompt
    assert prompt_parts(prompt) == (spec, [])


@pytest.mark.parametrize("response,status", [
    (supported_response(), "ok"),
    (supported_response(value=301), "invalid"),
    (supported_response(value="70 kg"), "invalid"),
    (supported_response(evidence=[{"note_id": "unknown", "quote": "Weight is 70 kg."}]), "invalid"),
    (supported_response(evidence=[{"note_id": "oncology/2025-01-02", "quote": "Invented supporting text."}]), "invalid"),
    (supported_response(evidence=[]), "invalid"),
    (supported_response(confidence="certain"), "invalid"),
    (supported_response(confidence=[]), "invalid"),
    ({"value": None, "confidence": "Low", "evidence": []}, "missing"),
    ({"confidence": "Low"}, "invalid"),
    ([], "invalid"),
])
def test_pipeline_distinguishes_invalid_outputs_missing_and_valid_results(tmp_path, local_models, response, status):
    config, _ = tiny_config(tmp_path)
    result = pipeline.run_pipeline(config, extractor=lambda prompt: response)
    assert result["results"][0]["status"] == status
    assert result["failures"] == int(status == "invalid")
    if status in {"missing", "invalid"}:
        assert result["results"][0]["value"] is None


def test_pipeline_reports_service_errors_and_does_not_call_model_without_evidence(tmp_path, local_models):
    config, _ = tiny_config(tmp_path)
    config["runtime"]["ollama"]["validation_retries"] = 1
    unavailable = Mock(side_effect=ConnectionError("model unavailable"))
    result = pipeline.run_pipeline(config, extractor=unavailable)
    assert result["results"][0]["status"] == "error"
    assert result["failures"] == 1
    unavailable.assert_called_once()
    model = Mock(side_effect=AssertionError("No model request should be made"))
    result = pipeline.run_pipeline(config, extractor=model, retriever=lambda *args: ([], {}))
    assert result["results"][0]["status"] == "missing"
    assert result["failures"] == 0
    model.assert_not_called()


@pytest.mark.parametrize("values,retries,expected_status,expected_attempts", [
    ([301, 70], 1, "ok", 2),
    ([70], 1, "ok", 1),
    ([301, 302], 1, "invalid", 2),
    ([301], 0, "invalid", 1),
])
def test_validation_retries_preserve_original_evidence_and_record_all_attempts(
    tmp_path, local_models, values, retries, expected_status, expected_attempts,
):
    config, _ = tiny_config(tmp_path)
    config["runtime"]["ollama"]["validation_retries"] = retries
    responses = [supported_response(value=value) for value in values]
    extractor = Mock(side_effect=responses)

    result = pipeline.run_pipeline(config, extractor=extractor)

    assert extractor.call_count == expected_attempts
    row = result["results"][0]
    assert row["status"] == expected_status
    assert row["attempts"] == expected_attempts
    assert row["value"] == (70 if expected_status == "ok" else None)
    cache_path = next((Path(config["outputs"]["root"]) / "prompt_cache").rglob("weight.json"))
    cached = json.loads(cache_path.read_text())
    assert cached["response"] == responses[-1]
    assert cached["result"] == row
    assert len(cached["attempts"]) == expected_attempts
    original_prompt = extractor.call_args_list[0].args[0]
    _, evidence = prompt_parts(original_prompt)
    assert evidence
    assert cached["prompt"] == original_prompt
    for index, attempt in enumerate(cached["attempts"]):
        assert attempt["response"] == responses[index]
        assert attempt["prompt"] == extractor.call_args_list[index].args[0]
        if values[index] > 300:
            assert attempt["validation_error"] == "Feature weight exceeds its maximum"
        else:
            assert "validation_error" not in attempt
        if index:
            assert attempt["prompt"].startswith(original_prompt)
            assert cached["attempts"][index - 1]["validation_error"] in attempt["prompt"]
            assert str(config["evaluation"]["gold_path"]) not in attempt["prompt"]


def test_nonfinite_raw_model_response_is_invalid_and_persisted_as_strict_json(tmp_path, local_models):
    config, _ = tiny_config(tmp_path)
    config["runtime"]["ollama"]["validation_retries"] = 1
    extractor = Mock(return_value=supported_response(value=float("nan")))

    result = pipeline.run_pipeline(config, extractor=extractor)

    assert result["results"][0]["status"] == "invalid"
    assert result["results"][0]["error"] == "Numeric values must be finite"
    assert result["results"][0]["attempts"] == 2
    assert result["failures"] == 1
    extractor.assert_called()
    assert extractor.call_count == 2
    output = Path(config["outputs"]["root"])
    for path in output.rglob("*.json"):
        json.loads(path.read_text(), parse_constant=lambda value: pytest.fail(f"Non-JSON constant {value} in {path}"))
    cache_path = next((output / "prompt_cache").rglob("weight.json"))
    cached = json.loads(cache_path.read_text())
    assert "invalid_raw_response" in cached["response"]
    assert all("invalid_raw_response" in item["response"] for item in cached["attempts"])


def test_invalid_stage_is_rejected_before_input_access_or_writes(tmp_path, monkeypatch):
    prepare = Mock(side_effect=AssertionError("Inputs must not be touched"))
    write = Mock(side_effect=AssertionError("Outputs must not be written"))
    monkeypatch.setattr(pipeline, "prepare_inputs", prepare)
    monkeypatch.setattr(pipeline, "write_json", write)

    with pytest.raises(ValueError, match="stage must be"):
        pipeline.run_pipeline({"outputs": {"root": str(tmp_path / "outputs")}}, stage="unknown")

    prepare.assert_not_called()
    write.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_automatic_feature_generation_cache_changes_with_seed(tmp_path, monkeypatch):
    config, _ = tiny_config(tmp_path)
    config["features"]["configuration_mode"] = "automatic"
    creator = importlib.import_module("oncorag.create_config")

    def generate(**kwargs):
        specs = pipeline.load_feature_specs(kwargs["features_file"])
        pipeline.generate_feature_configs(specs, kwargs["output_dir"], language=kwargs["language"])

    generator = Mock(side_effect=generate)
    monkeypatch.setattr(creator, "process_features_with_ontology_mapping", generator)
    first = pipeline.run_pipeline(config, stage="config")
    manifest_path = Path(config["features"]["generated_config_dir"]) / "generation_manifest.json"
    first_manifest = json.loads(manifest_path.read_text())
    assert first == {"features": ["weight"]}
    assert generator.call_count == 1
    assert generator.call_args.kwargs["seed"] == config["runtime"]["random_seed"]

    assert pipeline.run_pipeline(config, stage="config") == first
    assert generator.call_count == 1
    assert json.loads(manifest_path.read_text()) == first_manifest

    config["runtime"]["random_seed"] += 1
    assert pipeline.run_pipeline(config, stage="config") == first
    assert generator.call_count == 2
    assert generator.call_args.kwargs["seed"] == config["runtime"]["random_seed"]
    assert json.loads(manifest_path.read_text())["fingerprint"] != first_manifest["fingerprint"]


def test_graph_context_and_all_provenance_are_stable_across_insertion_and_seed_order():
    graph = nx.Graph()
    for note_id, date in [("note:z", "2025-01-02"), ("note:a", "2025-01-01")]:
        graph.add_node(note_id, label="Note", note_date=date,
                       note_file=f"{note_id}.txt", note_path=f"/notes/{note_id}.txt")
        for index, text in enumerate(["Weight is 70 kg.", "Fatigue is documented."]):
            sentence_id = f"{note_id}_sent_{index}"
            graph.add_node(sentence_id, label="Sentence")
            graph.add_edge(note_id, sentence_id, source_sentence=text,
                           source_sentence_id=sentence_id, source_note_id=note_id)
    reverse = nx.Graph()
    reverse.add_nodes_from(reversed(list(graph.nodes(data=True))))
    reverse.add_edges_from(reversed(list(graph.edges(data=True))))

    actual = prompt_builder.get_context_from_graph_with_metadata(graph, ["note:z", "missing", "note:a"])
    reordered = prompt_builder.get_context_from_graph_with_metadata(reverse, ["note:a", "note:z", "missing"])

    assert actual == reordered
    context, metadata = actual
    assert context == "Fatigue is documented.\nWeight is 70 kg."
    assert len(metadata) == 2
    for index, item in enumerate(metadata):
        assert item == {
            "sentence": context.splitlines()[index],
            "sentence_ids": [f"note:a_sent_{1 - index}", f"note:z_sent_{1 - index}"],
            "note_ids": ["note:a", "note:z"],
            "note_dates": ["2025-01-01", "2025-01-02"],
            "note_files": ["note:a.txt", "note:z.txt"],
            "note_paths": ["/notes/note:a.txt", "/notes/note:z.txt"],
        }


def test_graph_cache_changes_with_note_content_and_settings_and_forced_rebuild(tmp_path, local_models):
    config, note_path = tiny_config(tmp_path)
    builder = Mock(wraps=graph_builder.build_patient_graph)
    first = pipeline.run_pipeline(config, graph_builder=builder, stage="graph")
    second = pipeline.run_pipeline(config, graph_builder=builder, stage="graph")
    assert first["graphs"] == second["graphs"]
    assert builder.call_count == 1
    note_path.write_text("Weight is 72 kg. A new note fact.")
    changed = pipeline.run_pipeline(config, graph_builder=builder, stage="graph")
    assert changed["graphs"] != first["graphs"]
    assert builder.call_count == 2
    config["graph"]["include_report_sentences"] = False
    new_settings = pipeline.run_pipeline(config, graph_builder=builder, stage="graph")
    assert new_settings["graphs"] != changed["graphs"]
    assert builder.call_count == 3
    forced = pipeline.run_pipeline(config, graph_builder=builder, force_rebuild=True, stage="graph")
    assert forced["graphs"] == new_settings["graphs"]
    assert builder.call_count == 4


def test_pipeline_version_invalidates_cached_graphs_and_run_fingerprint(tmp_path, local_models, monkeypatch):
    config, _ = tiny_config(tmp_path)
    builder = Mock(wraps=graph_builder.build_patient_graph)
    current_version = pipeline.PIPELINE_VERSION
    with monkeypatch.context() as old_runtime:
        old_runtime.setattr(pipeline, "PIPELINE_VERSION", "portable-v1.1")
        old = pipeline.run_pipeline(config, graph_builder=builder, stage="graph")
    current = pipeline.run_pipeline(config, graph_builder=builder, stage="graph")
    assert current["pipeline_version"] == current_version
    assert current["run_fingerprint"] != old["run_fingerprint"]
    assert current["graphs"] != old["graphs"]
    assert builder.call_count == 2


def test_retrieval_parameters_and_feature_config_reach_the_pipeline(tmp_path, local_models):
    config, _ = tiny_config(tmp_path)
    config["retrieval"]["top_k"] = 1
    config["retrieval"]["weights"]["semantic_weight"] = 0.3
    retriever = Mock(wraps=pipeline.retrieve_context)
    pipeline.run_pipeline(config, retriever=retriever, extractor=lambda prompt: {"value": None})
    args = retriever.call_args.args
    assert isinstance(args[0], nx.Graph)
    assert args[1].count() == 2
    assert args[2]["name"] == "weight"
    assert args[3]["feature"]["expected_range"] == {"min": 0, "max": 300}
    assert args[4] == config["retrieval"]
    caches = list((Path(config["outputs"]["root"]) / "prompt_cache").rglob("weight.json"))
    cached = json.loads(caches[0].read_text())
    assert cached["retrieval"]["top_k"] == 1
    assert cached["retrieval"]["semantic_weight"] == 0.3


def test_multiline_source_sentence_keeps_provenance_through_retrieval(tmp_path, local_models):
    config, note_path = tiny_config(tmp_path)
    note_path.write_text("Synthetic clinical report\n\nWeight is 70 kg.\nFatigue is documented.")
    result = pipeline.run_pipeline(config, extractor=lambda prompt: supported_response())
    assert result["results"][0]["status"] == "ok"
    cache_path = next((Path(config["outputs"]["root"]) / "prompt_cache").rglob("weight.json"))
    cached = json.loads(cache_path.read_text())
    _, context = prompt_parts(cached["prompt"])
    weight = next(item for item in context if "Weight is 70 kg." in item["text"])
    assert weight == {"text": "Weight is 70 kg.", "note_id": "oncology/2025-01-02",
                      "date": "2025-01-02", "report_type": "oncology", "language": "en"}


def test_changed_note_replaces_stale_chroma_entries(tmp_path, local_models):
    config, note_path = tiny_config(tmp_path)
    first = pipeline.run_pipeline(config, extractor=lambda prompt: {"value": None})
    collection = get_vector_collection("patient1", config["vector_store"])
    assert collection.count() == 2
    note_path.write_text("A newly measured weight is 72 kg.")
    changed = pipeline.run_pipeline(config, extractor=lambda prompt: {"value": None})
    documents = collection.get()["documents"]
    assert collection.count() == 1
    assert "72 kg" in documents[0]
    assert all("70 kg" not in document and "Fatigue" not in document for document in documents)
    assert changed["results"][0]["graph_fingerprint"] != first["results"][0]["graph_fingerprint"]


def test_ollama_host_model_and_generation_parameters_reach_client(monkeypatch):
    import ollama

    runtime = {"random_seed": 17, "ollama": {
        "host": "http://127.0.0.1:11435", "model": "fixture-model", "timeout_seconds": 91,
        "temperature": 0.2, "num_ctx": 8192, "max_tokens": 321,
    }}
    client = Mock()
    client.chat.return_value = {"message": {"content": '{"value": null}'}}
    constructor = Mock(return_value=client)
    monkeypatch.setattr(ollama, "Client", constructor)
    assert pipeline.OllamaExtractor(runtime)("fixture prompt") == {"value": None}
    constructor.assert_called_once_with(host="http://127.0.0.1:11435", timeout=91)
    arguments = client.chat.call_args.kwargs
    assert arguments["model"] == "fixture-model"
    assert arguments["messages"] == [{"role": "user", "content": "fixture prompt"}]
    assert arguments["options"] == {"temperature": 0.2, "num_ctx": 8192, "num_predict": 321, "seed": 17}
    assert arguments["format"]["type"] == "object"
    assert set(arguments["format"]["required"]) == {"value", "confidence", "reasoning", "evidence"}


@pytest.mark.parametrize("feature_type,expected_range,value_schema", [
    ("boolean", None, {"type": ["boolean", "null"]}),
    ("date", None, {"type": ["string", "null"]}),
    ("integer", {"min": 0, "max": 120}, {"type": ["integer", "null"]}),
    ("numeric", {"min": 0, "max": 30}, {"type": ["number", "null"]}),
    ("text", None, {"type": ["string", "null"]}),
    ("categorical", ["Positive", "Negative"], {"type": ["string", "null"], "enum": ["Positive", "Negative", None]}),
    ("ordinal", ["0", "1", "2"], {"type": ["string", "null"], "enum": ["0", "1", "2", None]}),
])
def test_ollama_response_schema_uses_feature_type_and_exact_retrieved_evidence(
    monkeypatch, feature_type, expected_range, value_schema,
):
    import ollama

    client = Mock()
    client.chat.return_value = {"message": {"content": '{"value": null}'}}
    monkeypatch.setattr(ollama, "Client", Mock(return_value=client))
    extractor = pipeline.OllamaExtractor({"ollama": {"host": "http://127.0.0.1:11435", "model": "fixture"}})
    spec = {"name": "measurement", "type": feature_type, "expected_range": expected_range}
    context = [
        {"note_id": "note-z", "text": 'Source says "Positive".'},
        {"note_id": "note-a", "text": "\nH\u00e4moglobin: 11,3 g/dL.\n"},
        {"note_id": "note-z", "text": 'Source says "Positive".'},
    ]

    extractor.configure_response(spec, context)
    assert extractor("fixture prompt") == {"value": None}

    schema = client.chat.call_args.kwargs["format"]
    assert schema["properties"]["value"] == value_schema
    evidence = schema["properties"]["evidence"]
    assert evidence["type"] == "array"
    assert set(evidence["items"]["required"]) == {"note_id", "quote"}
    assert evidence["items"]["properties"] == {
        "note_id": {"type": "string", "enum": ["note-a", "note-z"]},
        "quote": {"type": "string", "enum": ["\nH\u00e4moglobin: 11,3 g/dL.\n", 'Source says "Positive".']},
    }
    extractor.configure_response({"type": "boolean"}, [{"note_id": "new-note", "text": "New source."}])
    extractor("second feature prompt")
    updated_schema = client.chat.call_args.kwargs["format"]["properties"]
    assert updated_schema["value"] == {"type": ["boolean", "null"]}
    assert updated_schema["evidence"]["items"]["properties"]["note_id"]["enum"] == ["new-note"]
    assert updated_schema["evidence"]["items"]["properties"]["quote"]["enum"] == ["New source."]


def test_known_note_and_quote_from_different_sources_still_fail_validation():
    spec = {"name": "weight", "type": "numeric", "expected_range": {"min": 0, "max": 300}}
    context = [{"note_id": "note-a", "text": "Weight is 70 kg."},
               {"note_id": "note-b", "text": "Fatigue is documented."}]
    response = supported_response(evidence=[{"note_id": "note-b", "quote": "Weight is 70 kg."}])

    with pytest.raises(ValueError, match="Evidence quote or note ID is not in retrieved context"):
        pipeline.validate_extraction(response, spec, context)


def test_pipeline_configures_real_extractor_with_current_feature_and_context(tmp_path, local_models, monkeypatch):
    import ollama

    config, _ = tiny_config(tmp_path)
    client = Mock()
    client.chat.return_value = {"message": {"content": json.dumps(supported_response())}}
    monkeypatch.setattr(ollama, "Client", Mock(return_value=client))

    result = pipeline.run_pipeline(config)

    assert result["results"][0]["status"] == "ok"
    client.chat.assert_called_once()
    schema = client.chat.call_args.kwargs["format"]["properties"]
    assert schema["value"] == {"type": ["number", "null"]}
    assert schema["evidence"]["items"]["properties"] == {
        "note_id": {"type": "string", "enum": ["oncology/2025-01-02"]},
        "quote": {"type": "string", "enum": ["Fatigue is documented.", "Weight is 70 kg."]},
    }


def rerank_options(**weights):
    return {"weights": {"semantic_weight": 1, "lexical_weight": 0, "name_weight": 0,
                        "graph_weight": 0, "boost_alpha": 0, "penalty_beta": 0, **weights},
            "graph_diffusion": {"enabled": False}}


@pytest.mark.parametrize("top_k", [1, 3, 5, 10])
def test_reranker_uses_exact_configured_top_k_despite_legacy_environment(monkeypatch, top_k):
    monkeypatch.setenv("ONCORAG_RERANK_TOP_FINAL", "2")
    monkeypatch.setenv("ONCORAG_KEYWORD_FALLBACK", "5")
    monkeypatch.setattr(model_init, "get_combined_reranker_scores", lambda pairs, **kwargs: [0.5] * len(pairs))
    sentences = [f"Measurement sample {index} is documented." for index in range(12)]
    _, _, details = prompt_builder.rerank_context(
        "\n".join(sentences), "Measurement", top_k=top_k, runtime_options=rerank_options(),
    )
    assert details["total_sentences_after"] == top_k
    assert len(details["top_sentences"]) == top_k


def test_reranker_lexical_and_semantic_weights_change_order(monkeypatch):
    monkeypatch.setattr(model_init, "get_combined_reranker_scores", lambda pairs, **kwargs: [
        1.0 if "first" in sentence else 0.0 for _, sentence in pairs
    ])
    context = "Measurement first report.\nMeasurement measurement measurement second report."
    _, _, semantic = prompt_builder.rerank_context(
        context, "measurement second", top_k=1, runtime_options=rerank_options(),
    )
    _, _, lexical = prompt_builder.rerank_context(
        context, "measurement second", top_k=1,
        runtime_options=rerank_options(semantic_weight=0, lexical_weight=1),
    )
    assert semantic["top_sentences"] == ["Measurement first report."]
    assert lexical["top_sentences"] == ["Measurement measurement measurement second report."]


def test_reranker_keeps_german_decimal_commas_as_part_of_values(monkeypatch):
    monkeypatch.setattr(model_init, "get_combined_reranker_scores", lambda pairs, **kwargs: [0.5] * len(pairs))
    sentence = "Haemoglobin heute: 11,2 g/dL."
    _, _, details = prompt_builder.rerank_context(
        sentence, "Haemoglobin", top_k=1, runtime_options=rerank_options(),
    )
    assert details["top_sentences"] == [sentence]


@pytest.mark.parametrize("cpu_fallback", [False, True])
def test_graph_scores_remain_aligned_after_candidate_limits(monkeypatch, cpu_fallback):
    from oncorag.rerank.graphrag_reranker import GraphReranker

    sentences = ["Unrelated report.", "Measurement low report.", "Measurement high report.", "Another unrelated report."]
    monkeypatch.setattr(GraphReranker, "score", lambda self, question, texts: (
        [10.0, 0.0, 1.0, 10.0], {"total_candidates": 4},
    ))
    monkeypatch.setattr(prompt_builder, "_runtime_int", lambda key, env, default: {
        "rerank_candidates": 2, "rerank_cpu_candidates": 1,
    }.get(key, default))
    calls = []

    def semantic_scores(pairs, **kwargs):
        calls.append(kwargs)
        if cpu_fallback and len(calls) == 1:
            raise RuntimeError("CUDA out of memory")
        return [0.0] * len(pairs)

    monkeypatch.setattr(model_init, "get_combined_reranker_scores", semantic_scores)
    options = rerank_options(semantic_weight=0, graph_weight=1)
    options["graph_diffusion"] = {"enabled": True}
    _, _, details = prompt_builder.rerank_context(
        "\n".join(sentences), "Measurement", normalized_name="Measurement",
        top_k=1, runtime_options=options,
        sentence_meta=[{"sentence": sentence, "note_ids": [str(index)]} for index, sentence in enumerate(sentences)],
    )

    expected_index = 1 if cpu_fallback else 2
    assert details["top_sentences"] == [sentences[expected_index]]
    assert details["top_note_ids"] == [str(expected_index)]
    assert details["graph_reranker_scores"] == ([0.0] if cpu_fallback else [0.0, 1.0])
    if cpu_fallback:
        assert calls[1]["device_override"] == "cpu"
