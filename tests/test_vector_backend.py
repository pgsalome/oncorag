import importlib
import json
from unittest.mock import Mock

import networkx as nx
import pytest
from chromadb.api.types import EmbeddingFunction

from oncoraggraph.vector_store.backend import get_vector_collection, index_graph_nodes
from oncoraggraph.vector_store.config import load_vector_store_config, validate_vector_store_config


class ToyEmbedding(EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, input):
        return [[0.0, 1.0, 0.0] if "fatigue" in text else [1.0, 0.0, 0.0] for text in input]

    @staticmethod
    def name():
        return "oncorag_test_embedding"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return ToyEmbedding()


@pytest.fixture
def graph():
    graph = nx.Graph()
    graph.add_node("SYN-001", label="Patient")
    graph.add_node("report", label="Note")
    graph.add_node("nausea", label="Condition", original_text="nausea", is_negated=True)
    graph.add_node("fatigue", label="Condition", original_text="fatigue", cluster_size=2)
    graph.add_node("drug", label="Treatment")
    return graph


def test_config_loading_precedence_and_relative_cache_path(tmp_path, monkeypatch):
    env_config = tmp_path / "env.json"
    env_config.write_text(json.dumps({"backend": "iris"}))
    chosen_config = tmp_path / "chosen.yaml"
    chosen_config.write_text("vector_store:\n  backend: iris\n  chroma:\n    path: cache\n")
    monkeypatch.setenv("ONCORAGGRAPH_VECTOR_STORE_CONFIG", str(env_config))
    monkeypatch.setenv("ONCORAGGRAPH_VECTOR_BACKEND", "iris")
    settings = load_vector_store_config(chosen_config, backend="chroma")
    assert settings["backend"] == "chroma"
    assert settings["chroma"]["path"] == str(tmp_path / "cache")
    assert load_vector_store_config()["backend"] == "iris"


@pytest.mark.parametrize("settings", [
    {"backend": "unsupported"}, {"collection_namespace": ""}, {"iris": []},
    {"chroma": {"path": ""}}, {"backend": "iris", "iris": {"password": "do-not-store"}},
    {"backend": "iris", "iris": {"table": "SQLUser.Bad; DROP TABLE Other"}},
])
def test_bad_backend_settings_fail_before_database_access(settings):
    with pytest.raises(ValueError):
        validate_vector_store_config(settings)


def test_explicit_config_without_vector_settings_is_rejected(tmp_path):
    path = tmp_path / "missing.json"
    path.write_text('{"retrieval": {"top_k": 5}}')
    with pytest.raises(ValueError, match="vector_store"):
        load_vector_store_config(path)


def test_chroma_factory_isolates_exact_patient_ids_and_cohorts(monkeypatch):
    chroma_module = importlib.import_module("oncoraggraph.chroma.chroma_index")
    factory = Mock()
    monkeypatch.setattr(chroma_module, "get_chroma_collection", factory)
    for patient_id, namespace in [("patient-1", "en"), ("patient_1", "en"), ("patient-1", "de")]:
        get_vector_collection(patient_id, {"backend": "chroma", "collection_namespace": namespace})
    names = [call.kwargs["collection_name"] for call in factory.call_args_list]
    assert len(set(names)) == 3
    get_vector_collection("patient-1", {"backend": "chroma", "collection_namespace": "en"})
    assert factory.call_args.kwargs["collection_name"] == names[0]


def test_iris_factory_passes_shared_embedding_and_config(monkeypatch):
    iris_module = importlib.import_module("oncoraggraph.vector_store.iris")
    models = importlib.import_module("oncoraggraph.models.model_init")
    factory = Mock()
    embedding = ToyEmbedding()
    monkeypatch.setattr(iris_module, "IRISCollection", factory)
    monkeypatch.setattr(models, "get_chroma_embedding_function", lambda: embedding)
    result = get_vector_collection("SYN-001", {
        "backend": "iris", "collection_namespace": "mixed", "iris": {"vector_dimension": 3},
    })
    assert result is factory.return_value
    assert factory.call_args.args[0] == "SYN-001"
    assert factory.call_args.args[1]["vector_dimension"] == 3
    assert factory.call_args.args[2] is embedding
    assert factory.call_args.kwargs["collection_namespace"] == "mixed"


def test_graph_payload_is_identical_for_both_backends(graph):
    filters = {"required": ["Condition", "Treatment"], "exclude": ["Treatment"]}
    chroma, iris = Mock(), Mock()
    iris.backend = "iris"
    chroma.get.return_value = {"ids": ["obsolete"]}
    index_graph_nodes(graph, chroma, filters, replace=True)
    index_graph_nodes(graph, iris, filters, replace=True)
    assert chroma.add.call_args.kwargs == iris.replace.call_args.kwargs
    assert chroma.add.call_args.kwargs["ids"] == ["nausea", "fatigue"]
    assert chroma.add.call_args.kwargs["metadatas"][0]["is_negated"] is True
    chroma.delete.assert_called_once_with(ids=["obsolete"])
    iris.add.assert_not_called()


def test_empty_rebuild_removes_stale_index_for_both_backends():
    chroma, iris = Mock(), Mock()
    iris.backend = "iris"
    chroma.get.return_value = {"ids": ["obsolete"]}
    index_graph_nodes(nx.Graph(), chroma, replace=True)
    index_graph_nodes(nx.Graph(), iris, replace=True)
    chroma.delete.assert_called_once_with(ids=["obsolete"])
    chroma.add.assert_not_called()
    iris.replace.assert_called_once_with(ids=[], documents=[], metadatas=[])


def test_real_chroma_indexes_queries_and_rebuilds_in_isolation(tmp_path, monkeypatch, graph):
    chroma_module = importlib.import_module("oncoraggraph.chroma.chroma_index")
    monkeypatch.setattr(chroma_module, "get_chroma_embedding_function", ToyEmbedding)
    config = {"backend": "chroma", "chroma": {"path": str(tmp_path / "chroma")}}
    collection = get_vector_collection("SYN-001", config)
    index_graph_nodes(graph, collection, {"required": ["Condition"]})
    result = collection.query(query_texts=["fatigue"], n_results=2)
    assert result["ids"][0] == ["fatigue", "nausea"]
    assert result["distances"][0] == pytest.approx([0.0, 1.0])
    reopened = get_vector_collection("SYN-001", config)
    assert reopened.query(query_texts=["fatigue"], n_results=1)["ids"][0] == ["fatigue"]
    assert get_vector_collection("SYN-002", config).count() == 0
    graph.remove_node("nausea")
    index_graph_nodes(graph, collection, {"required": ["Condition"]}, replace=True)
    assert collection.get(include=[])["ids"] == ["fatigue"]


def test_chroma_dimension_recovery_uses_same_configured_database(tmp_path, monkeypatch, graph):
    chroma_module = importlib.import_module("oncoraggraph.chroma.chroma_index")
    monkeypatch.setattr(chroma_module, "get_chroma_embedding_function", ToyEmbedding)
    config = {"backend": "chroma", "chroma": {"path": str(tmp_path / "custom_chroma")}}
    collection = get_vector_collection("SYN-001", config)
    index_graph_nodes(graph, collection)
    other = get_vector_collection("SYN-002", config)
    index_graph_nodes(graph, other)

    class TwoDimensionalEmbedding(ToyEmbedding):
        def __call__(self, input):
            return [[1.0, 0.0] for text in input]

    monkeypatch.setattr(chroma_module, "get_chroma_embedding_function", TwoDimensionalEmbedding)
    collection = get_vector_collection("SYN-001", config)
    rebuilt = index_graph_nodes(graph, collection, replace=True)
    assert rebuilt.count() == 3
    assert rebuilt.query(query_texts=["nausea"], n_results=1)["distances"][0] == pytest.approx([0.0])
    assert other.count() == 3


def test_extraction_cli_forwards_iris_settings(tmp_path, monkeypatch):
    main = importlib.import_module("oncoraggraph.main")
    config_path = tmp_path / "vector.json"
    config_path.write_text(json.dumps({"vector_store": {"backend": "iris", "collection_namespace": "mixed"}}))
    runner = Mock(return_value={"value": "Missing"})
    monkeypatch.delenv("ONCORAGGRAPH_VECTOR_BACKEND", raising=False)
    monkeypatch.setattr(main, "run_feature_extraction", runner)
    monkeypatch.setattr("sys.argv", [
        "oncoraggraph", "notes/SYN-001", "nausea", "--vector-store-config", str(config_path),
        "--cache-dir", str(tmp_path / "prompts"),
    ])
    assert main.main() == {"value": "Missing"}
    assert runner.call_args.kwargs["vector_store_config"]["backend"] == "iris"
    assert runner.call_args.kwargs["vector_store_config"]["collection_namespace"] == "mixed"
    assert runner.call_args.kwargs["prompt_cache_dir"] == str(tmp_path / "prompts")


@pytest.mark.parametrize("configured_backend,cli_backend", [
    ("chroma", None), ("iris", None), ("chroma", "iris"), ("iris", "chroma"),
])
def test_pipeline_cli_propagates_backend_to_patient_index_and_extraction(
    tmp_path, monkeypatch, configured_backend, cli_backend,
):
    from pathlib import Path
    from oncoraggraph import pipeline
    from oncoraggraph.config.pipeline_config import load_pipeline_config
    from oncoraggraph.graph import graph_builder
    from oncoraggraph.vector_store import backend

    root = Path(__file__).resolve().parents[1]
    config = load_pipeline_config(root / "configs" / "oncorag_synthetic_english.json")
    config["features"]["generated_config_dir"] = str(tmp_path / "features")
    config["outputs"]["root"] = str(tmp_path / "outputs")
    config["vector_store"].update(
        backend=configured_backend,
        chroma={"path": str(tmp_path / "vectors")},
        iris={"host": "127.0.0.1", "port": 1973, "table": "SQLUser.TestVectors"},
    )
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(json.dumps(config))
    collection = Mock()
    collection_factory = Mock(return_value=collection)
    indexer = Mock(return_value=collection)
    retriever = Mock(return_value=([], {}))
    extractor = Mock(side_effect=AssertionError("Empty evidence must not call the model"))
    monkeypatch.setattr(graph_builder, "build_patient_graph", lambda *args, **kwargs: nx.Graph())
    monkeypatch.setattr(backend, "get_vector_collection", collection_factory)
    monkeypatch.setattr(backend, "index_graph_nodes", indexer)
    monkeypatch.setattr(pipeline, "retrieve_context", retriever)
    monkeypatch.setattr(pipeline, "OllamaExtractor", lambda runtime: extractor)
    args = ["--config", str(config_path)]
    if cli_backend:
        args.extend(["--vector-backend", cli_backend])

    assert pipeline.main(args) == 0

    expected_backend = cli_backend or configured_backend
    assert collection_factory.call_count == indexer.call_count == 3
    assert len({call.args[0] for call in collection_factory.call_args_list}) == 3
    for factory_call, index_call in zip(collection_factory.call_args_list, indexer.call_args_list):
        settings = factory_call.kwargs["config"]
        assert settings["backend"] == expected_backend
        assert settings["collection_namespace"] == "synthetic_english"
        assert settings["chroma"]["path"] == str(tmp_path / "vectors")
        assert settings["iris"]["host"] == "127.0.0.1"
        assert settings["iris"]["port"] == 1973
        assert settings["iris"]["table"] == "SQLUser.TestVectors"
        assert index_call.args[0].graph["patient_id"] == factory_call.args[0]
        assert index_call.args[1] is collection
        assert index_call.kwargs["replace"] is True
    assert retriever.call_count == 12
    assert all(call.args[1] is collection for call in retriever.call_args_list)
    extractor.assert_not_called()
    parameters = json.loads((tmp_path / "outputs" / "parameters.json").read_text())
    assert parameters["vector_store"]["backend"] == expected_backend
    results = json.loads((tmp_path / "outputs" / "structured_features.json").read_text())
    assert len(results["results"]) == 12
    assert all(row["status"] == "missing" for row in results["results"])


def test_pipeline_config_validates_vector_backend():
    from pathlib import Path
    from oncoraggraph.config.pipeline_config import load_pipeline_config, validate_pipeline_config

    path = Path(__file__).resolve().parents[1] / "configs" / "oncorag_full_pipeline.example.json"
    config = load_pipeline_config(path)
    config["vector_store"]["backend"] = "iris"
    validate_pipeline_config(config)
    config["vector_store"]["iris"]["port"] = True
    with pytest.raises(ValueError, match="port"):
        validate_pipeline_config(config)
