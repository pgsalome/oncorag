"""IRIS collection contract tests without a server or embedding model."""

from copy import deepcopy
import json
import math
import os
import re
import uuid

import pytest

from oncorag.vector_store.iris import IRISCollection, validate_iris_config


class MemoryDatabase:
    def __init__(self):
        self.rows = {}
        self.table_exists = False
        self.calls = []
        self.connections = []
        self.fail_node = None

    def connect(self, *args, **kwargs):
        connection = MemoryConnection(self)
        self.connections.append(connection)
        self.connection_args = args, kwargs
        return connection


class MemoryConnection:
    def __init__(self, database):
        self.database = database
        self.rows = deepcopy(database.rows)
        self.closed = False
        self.committed = False
        self.rolled_back = False
        self.autocommit = True
        self.cursors = []

    def cursor(self):
        cursor = MemoryCursor(self)
        self.cursors.append(cursor)
        return cursor

    def setAutoCommit(self, value):
        self.autocommit = value

    def commit(self):
        self.database.rows = deepcopy(self.rows)
        self.committed = True

    def rollback(self):
        self.rows = deepcopy(self.database.rows)
        self.rolled_back = True

    def close(self):
        self.closed = True


class MemoryCursor:
    def __init__(self, connection):
        self.connection = connection
        self.closed = False
        self.results = []

    def execute(self, sql, params=()):
        database = self.connection.database
        database.calls.append((sql, params))
        rows = self.connection.rows
        if "INFORMATION_SCHEMA.TABLES" in sql:
            self.results = [(int(database.table_exists),)]
        elif sql.startswith("CREATE TABLE"):
            database.table_exists = True
        elif sql.startswith("DELETE"):
            assert "WHERE collection_key = ?" in sql
            keys = [key for key in rows if key[0] == params[0] and (len(params) == 1 or key[1] == params[1])]
            for key in keys:
                del rows[key]
        elif sql.startswith("INSERT"):
            if params[2] == database.fail_node:
                raise RuntimeError("driver error containing secret-password and patient text")
            rows[tuple(params[:2])] = tuple(params[2:])
        elif sql.startswith("SELECT COUNT(*)"):
            self.results = [(sum(key[0] == params[0] for key in rows),)]
        elif sql.startswith("SELECT TOP"):
            query = json.loads(params[0])
            ranked = []
            for key, row in rows.items():
                if key[0] != params[1]:
                    continue
                embedding = json.loads(row[3])
                cosine = sum(a * b for a, b in zip(query, embedding)) / (
                    math.sqrt(sum(a * a for a in query)) * math.sqrt(sum(a * a for a in embedding))
                )
                ranked.append((cosine, key[1], row[:3]))
            ranked.sort(key=lambda match: (-match[0], match[1]))
            limit = int(re.match(r"SELECT TOP (\d+)", sql)[1])
            self.results = [(*row, cosine) for cosine, _, row in ranked[:limit]]
        else:
            raise AssertionError(f"Unexpected SQL: {sql}")

    def fetchone(self):
        return self.results[0]

    def fetchall(self):
        return self.results

    def close(self):
        self.closed = True


@pytest.fixture
def database(monkeypatch):
    monkeypatch.setenv("IRIS_USERNAME", "test-user")
    monkeypatch.setenv("IRIS_PASSWORD", "secret-password")
    return MemoryDatabase()


def collection(database, patient="patient-1", namespace="english", embed=None, **options):
    return IRISCollection(
        patient,
        {"vector_dimension": 3, "initialize_schema": False, **options},
        embed or (lambda texts: [[1.0, 0.0, 0.0] for text in texts]),
        collection_namespace=namespace,
        connect_factory=database.connect,
    )


def test_bound_values_preserve_unicode_and_isolate_patients_and_cohorts(database):
    primary = collection(database, patient="a-b")
    others = [collection(database, patient="a_b"), collection(database, patient="A-B"), collection(database, patient="a-b", namespace="german")]
    node = "finding'; DROP TABLE SQLUser.OncoRAGVectors; --"
    document = "\u00dcbelkeit and fatigue (Condition)"
    metadata = {"label": "Condition", "negated": False, "original_text": "\u00dcbelkeit", "count": 2}
    primary.add([node], [document], [metadata])
    for other in others:
        other.add(["other"], ["unrelated"], [{}])
    assert primary.count() == 1
    assert all(other.count() == 1 for other in others)
    result = primary.query(["nausea", "\u00dcbelkeit"], n_results=10)
    assert result["ids"] == [[node], [node]]
    assert result["documents"] == [[document], [document]]
    assert result["metadatas"] == [[metadata], [metadata]]
    for sql, params in database.calls:
        assert node not in sql and document not in sql
    assert all(connection.closed for connection in database.connections)
    assert all(not connection.autocommit for connection in database.connections)
    assert all(cursor.closed for connection in database.connections for cursor in connection.cursors)


def test_cosine_distance_rank_limit_and_repeatable_ties(database):
    vectors = {"aligned": [2.0, 0, 0], "orthogonal": [0, 3.0, 0], "opposite": [-1, 0, 0], "query": [1, 0, 0]}
    store = collection(database, embed=lambda texts: [vectors[text] for text in texts])
    store.add(["a", "b", "c", "d"], ["aligned", "orthogonal", "opposite", "aligned"], [{}] * 4)
    result = store.query(["query"], n_results=10)
    assert set(result["ids"][0][:2]) == {"a", "d"}
    assert result["ids"][0][2:] == ["b", "c"]
    assert result["distances"] == [[0.0, 0.0, 1.0, 2.0]]
    assert result == store.query(["query"], n_results=10)
    assert len(store.query(["query"], n_results=1)["ids"][0]) == 1


def test_add_upserts_replace_removes_stale_and_clear_is_patient_scoped(database):
    store = collection(database)
    unrelated = collection(database, patient="patient-2")
    store.add(["old", "same"], ["old text", "first text"], [{}, {}])
    store.add(["same"], ["updated text"], [{"updated": True}])
    assert store.count() == 2
    unrelated.add(["other"], ["patient two"], [{}])
    store.replace(["new"], ["new text"], [{}])
    assert store.query(["q"])["ids"] == [["new"]]
    assert unrelated.count() == 1
    store.replace([], [], [])
    assert store.count() == 0
    assert store.query(["q"])["ids"] == [[]]
    assert unrelated.count() == 1


@pytest.mark.parametrize("replace", [False, True])
def test_failed_write_rolls_back_whole_batch_and_redacts_driver_error(database, replace):
    store = collection(database)
    store.add(["original"], ["old text"], [{}])
    saved = deepcopy(database.rows)
    database.fail_node = "broken"
    method = store.replace if replace else store.add
    with pytest.raises(RuntimeError, match="IRIS vector operation failed") as caught:
        method(["original", "broken"], ["changed text", "new text"], [{}, {}])
    assert database.rows == saved
    assert database.connections[-1].rolled_back
    assert database.connections[-1].closed
    assert "secret-password" not in str(caught.value)
    assert caught.value.__suppress_context__


@pytest.mark.parametrize("vectors", [[], [[1, 2]], [[0, 0, 0]], [[float("nan"), 0, 0]], [[float("inf"), 0, 0]], [["1", 0, 0]], [[True, 0, 0]]])
def test_invalid_embeddings_never_write(database, vectors):
    store = collection(database, embed=lambda texts: vectors)
    with pytest.raises(ValueError):
        store.replace(["id"], ["text"], [{}])
    assert database.connections == []
    assert database.rows == {}


@pytest.mark.parametrize("ids,documents,metadata", [(["a", "a"], ["one", "two"], [{}, {}]), (["a"], [], [{}]), (["a"], ["text"], [{"invalid": float("nan")}]), (["a"], ["text"], [None])])
def test_invalid_batches_never_write(database, ids, documents, metadata):
    store = collection(database)
    with pytest.raises(ValueError):
        store.replace(ids, documents, metadata)
    assert database.connections == []


@pytest.mark.parametrize("options", [{"table": "SQLUser.Vectors;DROP TABLE x"}, {"table": "Vectors"}, {"port": True}, {"port": 65536}, {"vector_dimension": 0}, {"timeout_seconds": -1}, {"initialize_schema": "false"}, {"password": "secret"}, {"username": 42}])
def test_invalid_config_fails_without_connection(database, options):
    with pytest.raises(ValueError):
        collection(database, **options)
    assert database.connections == []


def test_validation_is_driver_and_credentials_free(monkeypatch):
    monkeypatch.delenv("IRIS_USERNAME", raising=False)
    monkeypatch.delenv("IRIS_PASSWORD", raising=False)
    assert validate_iris_config({})["vector_dimension"] == 768


def test_optional_driver_is_loaded_only_when_needed(monkeypatch):
    def missing_driver(name):
        raise ImportError("not installed")

    monkeypatch.setattr("oncorag.vector_store.iris.importlib.import_module", missing_driver)
    store = IRISCollection("p", {"initialize_schema": False}, lambda texts: [])
    with pytest.raises(ImportError, match="intersystems-irispython"):
        store.count()


def test_missing_credentials_and_failed_connections_are_actionable(database, monkeypatch):
    store = collection(database)
    monkeypatch.delenv("IRIS_PASSWORD")
    with pytest.raises(ValueError, match="password environment variable"):
        store.count()
    assert database.connections == []
    monkeypatch.setenv("IRIS_PASSWORD", "secret-password")

    def fail_connect(*args, **kwargs):
        raise RuntimeError("could not connect with secret-password")

    store._connect_factory = fail_connect
    with pytest.raises(RuntimeError, match="IRIS connection failed") as caught:
        store.count()
    assert "secret-password" not in str(caught.value)
    assert "secret-password" not in repr(store)


def test_schema_creation_and_connection_settings(database, monkeypatch):
    monkeypatch.setenv("DEMO_IRIS_PASSWORD", "alternative-password")
    collection(
        database, initialize_schema=True, username="configured-user",
        password_env="DEMO_IRIS_PASSWORD", timeout_seconds=17, ssl_configuration="tls-demo",
    )
    assert database.table_exists
    create_calls = [sql for sql, _ in database.calls if sql.startswith("CREATE")]
    assert len(create_calls) == 1
    assert "VECTOR(DOUBLE, 3)" in create_calls[0]
    assert "PRIMARY KEY (collection_key, node_key)" in create_calls[0]
    assert database.connection_args == (
        ("127.0.0.1", 1972, "USER", "configured-user", "alternative-password"),
        {"timeout": 17, "sharedmemory": False, "sslconfig": "tls-demo"},
    )
    collection(database, initialize_schema=True)
    assert len([sql for sql, _ in database.calls if sql.startswith("CREATE")]) == 1


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "1;DROP TABLE x"])
def test_query_limit_validated_without_connecting(database, limit):
    with pytest.raises(ValueError, match="positive integer"):
        collection(database).query(["q"], n_results=limit)
    assert database.connections == []


def test_graph_index_and_retrieval_return_original_graph_ids(database):
    import networkx as nx

    from oncorag.chroma.chroma_index import find_start_nodes
    from oncorag.vector_store.backend import index_graph_nodes

    graph = nx.Graph()
    graph.add_node("entity:nausea", label="Condition", original_text="\u00dcbelkeit", is_negated=False)
    graph.add_node("entity:treatment", label="Treatment", original_text="Radiotherapy")
    graph.add_node("report:1", label="Document", original_text="synthetic report")
    store = collection(database)
    index_graph_nodes(graph, store, replace=True)
    ids, details = find_start_nodes(store, ["nausea", "\u00dcbelkeit"], n_results=10)
    assert set(ids) == {"entity:nausea", "entity:treatment"}
    assert all(node in graph for node in ids)
    assert all(search["best_distance"] == 0.0 for search in details["searches_performed"])
    result = store.query(["nausea"])
    assert set(metadata["label"] for metadata in result["metadatas"][0]) == {"Condition", "Treatment"}
    graph.remove_node("entity:treatment")
    index_graph_nodes(graph, store, replace=True)
    assert store.query(["nausea"])["ids"] == [["entity:nausea"]]


@pytest.mark.skipif(
    os.getenv("ONCORAG_TEST_IRIS") != "1",
    reason="Set ONCORAG_TEST_IRIS=1 and explicit IRIS credentials to test a real server",
)
def test_live_iris_collection_round_trip():
    assert os.getenv("IRIS_USERNAME") and os.getenv("IRIS_PASSWORD"), "Set IRIS_USERNAME and IRIS_PASSWORD explicitly"
    options = {
        "host": os.getenv("IRIS_HOST", "127.0.0.1"),
        "port": int(os.getenv("IRIS_PORT", "1972")),
        "namespace": os.getenv("IRIS_NAMESPACE", "USER"),
        "table": os.getenv("ONCORAG_TEST_IRIS_TABLE", "SQLUser.OncoRAGTestVectors"),
        "vector_dimension": 3,
        "ssl_configuration": os.getenv("IRIS_SSL_CONFIGURATION"),
    }
    namespace = "oncorag-test-" + uuid.uuid4().hex
    vectors = {"\u00dcbelkeit": [1.0, 0, 0], "fatigue": [0, 1.0, 0], "nausea": [1.0, 0, 0]}
    embed = lambda texts: [vectors[text] for text in texts]
    first = IRISCollection("patient", options, embed, collection_namespace=namespace)
    second = IRISCollection("patient", options, embed, collection_namespace=namespace + "-other")
    try:
        first.replace(["nausea", "fatigue"], ["\u00dcbelkeit", "fatigue"], [{"language": "german"}, {}])
        second.add(["other"], ["fatigue"], [{}])
        assert first.count() == 2
        result = first.query(["nausea"], n_results=5)
        assert result["ids"] == [["nausea", "fatigue"]]
        assert result["distances"][0] == pytest.approx([0.0, 1.0])
        assert result["documents"][0][0] == "\u00dcbelkeit"
        assert result["metadatas"][0][0] == {"language": "german"}
        first.replace(["updated"], ["fatigue"], [{}])
        assert first.count() == 1
        assert first.query(["nausea"])["ids"] == [["updated"]]
        assert second.query(["nausea"])["ids"] == [["other"]]
        first.replace([], [], [])
        assert first.count() == 0 and second.count() == 1
    finally:
        first.replace([], [], [])
        second.replace([], [], [])
