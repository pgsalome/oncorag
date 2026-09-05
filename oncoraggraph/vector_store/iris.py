"""Patient-scoped IRIS vector collections using the optional native DB-API."""

from __future__ import annotations

from contextlib import contextmanager, suppress
import hashlib
import importlib
import json
import math
from numbers import Real
import os
import re
from typing import Callable


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\Z")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_DEFAULTS = {
    "host": "127.0.0.1",
    "port": 1972,
    "namespace": "USER",
    "username": None,
    "password_env": "IRIS_PASSWORD",
    "table": "SQLUser.OncoRAGVectors",
    "vector_dimension": 768,
    "timeout_seconds": 30,
    "ssl_configuration": None,
    "initialize_schema": True,
}
_NODE_ID_LIMIT = 32768
_TEXT_LIMIT = 1048576


def validate_iris_config(config: dict) -> dict:
    """Validate options and return defaults without loading a driver or secrets."""
    if not isinstance(config, dict):
        raise ValueError("vector_store.iris must be an object")
    if any(key not in _DEFAULTS for key in config):
        raise ValueError("Unknown vector_store.iris option; use password_env for credentials")
    options = {**_DEFAULTS, **config}
    for key in ("host", "namespace", "password_env", "table"):
        if not isinstance(options[key], str) or not options[key].strip():
            raise ValueError(f"vector_store.iris.{key} must be a nonempty string")
    for key in ("username", "ssl_configuration"):
        value = options[key]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"vector_store.iris.{key} must be a nonempty string or null")
    if not _IDENTIFIER.fullmatch(options["table"]):
        raise ValueError("vector_store.iris.table must have the form Schema.Table using ASCII identifiers")
    if not _ENV_NAME.fullmatch(options["password_env"]):
        raise ValueError("vector_store.iris.password_env must be an environment variable name")
    for key in ("port", "vector_dimension", "timeout_seconds"):
        value = options[key]
        if type(value) is not int or value < 1:
            raise ValueError(f"vector_store.iris.{key} must be a positive integer")
    if options["port"] > 65535:
        raise ValueError("vector_store.iris.port must be at most 65535")
    if type(options["initialize_schema"]) is not bool:
        raise ValueError("vector_store.iris.initialize_schema must be boolean")
    return options


class IRISCollection:
    """Implement the Chroma collection subset used by graph retrieval.

    Each operation owns its connection so collections can be used by worker
    processes without sharing a live database session. Reindexing with replace()
    commits all rows for one patient together.
    """

    backend = "iris"

    def __init__(
        self,
        patient_id: str,
        config: dict,
        embedding_function: Callable,
        *,
        collection_namespace: str = "default",
        connect_factory=None,
    ):
        for name, value in (("patient_id", patient_id), ("collection_namespace", collection_namespace)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        if not callable(embedding_function):
            raise ValueError("embedding_function must be callable")
        self._options = validate_iris_config(config)
        self._embedding_function = embedding_function
        self._connect_factory = connect_factory
        identity = json.dumps([collection_namespace, patient_id], ensure_ascii=False, separators=(",", ":"))
        self._collection_key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        self.name = f"iris_{self._collection_key}"
        if self._options["initialize_schema"]:
            self._initialize_schema()

    def _connect(self):
        factory = self._connect_factory
        if factory is None:
            try:
                factory = importlib.import_module("iris.dbapi").connect
            except (ImportError, AttributeError):
                raise ImportError(
                    "IRIS requires the optional intersystems-irispython driver; install the project's iris extra"
                ) from None
        username = self._options["username"] or os.getenv("IRIS_USERNAME")
        password = os.getenv(self._options["password_env"])
        if not username or not password:
            raise ValueError("IRIS requires a username (config or IRIS_USERNAME) and the configured password environment variable")
        kwargs = {"timeout": self._options["timeout_seconds"], "sharedmemory": False}
        if self._options["ssl_configuration"] is not None:
            kwargs["sslconfig"] = self._options["ssl_configuration"]
        try:
            return factory(
                self._options["host"], self._options["port"], self._options["namespace"],
                username, password, **kwargs,
            )
        except Exception:
            raise RuntimeError("IRIS connection failed; check the server, credentials, and TLS settings") from None

    @contextmanager
    def _session(self, *, write=False):
        connection = self._connect()
        cursor = None
        try:
            connection.setAutoCommit(False)
            cursor = connection.cursor()
            yield cursor
            if write:
                connection.commit()
        except BaseException as exc:
            with suppress(Exception):
                connection.rollback()
            if isinstance(exc, Exception):
                # Driver diagnostics may include parameters or credentials.
                raise RuntimeError(
                    "IRIS vector operation failed; verify the table schema, vector dimension, and database permissions"
                ) from None
            raise
        finally:
            if cursor is not None:
                with suppress(Exception):
                    cursor.close()
            with suppress(Exception):
                connection.close()

    def _table_exists(self, cursor):
        schema, table = self._options["table"].split(".")
        cursor.execute(
            "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?",
            [schema, table],
        )
        return bool(cursor.fetchone()[0])

    def _initialize_schema(self):
        table = self._options["table"]
        dimension = self._options["vector_dimension"]
        with self._session(write=True) as cursor:
            if self._table_exists(cursor):
                return
            try:
                cursor.execute(
                    f"CREATE TABLE {table} ("
                    "collection_key VARCHAR(64) NOT NULL, "
                    "node_key VARCHAR(64) NOT NULL, "
                    f"node_id VARCHAR({_NODE_ID_LIMIT}) NOT NULL, "
                    f"document VARCHAR({_TEXT_LIMIT}) NOT NULL, "
                    f"metadata_json VARCHAR({_TEXT_LIMIT}) NOT NULL, "
                    f"embedding VECTOR(DOUBLE, {dimension}) NOT NULL, "
                    "PRIMARY KEY (collection_key, node_key))"
                )
            except Exception as exc:
                duplicate = getattr(exc, "sqlcode", None) == -201 or re.search(
                    r"SQLCODE\s*[:=]?\s*<?-201\b", str(exc), re.IGNORECASE
                )
                if not duplicate or not self._table_exists(cursor):
                    raise

    def _embed(self, documents):
        if not documents:
            return []
        try:
            embeddings = list(self._embedding_function(documents))
        except Exception:
            raise ValueError("The embedding function failed to produce vectors") from None
        if len(embeddings) != len(documents):
            raise ValueError("The embedding function must return one vector per document")
        vectors = []
        dimension = self._options["vector_dimension"]
        for embedding in embeddings:
            try:
                values = list(embedding)
            except TypeError:
                raise ValueError("Each embedding must be a numeric vector") from None
            if len(values) != dimension:
                raise ValueError(f"Embedding dimension must equal vector_store.iris.vector_dimension ({dimension})")
            if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
                raise ValueError("Embedding values must be finite numbers")
            try:
                vector = [float(value) for value in values]
            except (ValueError, OverflowError):
                raise ValueError("Embedding values must be finite numbers") from None
            if not all(math.isfinite(value) for value in vector):
                raise ValueError("Embedding values must be finite numbers")
            if not any(vector):
                raise ValueError("Cosine search requires a nonzero embedding vector")
            vectors.append(json.dumps(vector, allow_nan=False, separators=(",", ":")))
        return vectors

    def _prepare_rows(self, ids, documents, metadatas):
        if any(isinstance(values, (str, bytes)) for values in (ids, documents, metadatas)):
            raise ValueError("ids, documents, and metadatas must be sequences")
        try:
            ids, documents, metadatas = list(ids), list(documents), list(metadatas)
        except TypeError:
            raise ValueError("ids, documents, and metadatas must be sequences") from None
        if len(ids) != len(documents) or len(ids) != len(metadatas):
            raise ValueError("ids, documents, and metadatas must have equal lengths")
        for node_id in ids:
            if not isinstance(node_id, str) or not node_id or len(node_id) > _NODE_ID_LIMIT:
                raise ValueError(f"Node IDs must be nonempty strings of at most {_NODE_ID_LIMIT} characters")
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate node IDs in one indexing batch")
        if any(not isinstance(doc, str) or not doc or len(doc) > _TEXT_LIMIT for doc in documents):
            raise ValueError(f"Documents must be nonempty strings of at most {_TEXT_LIMIT} characters")
        metadata_json = []
        for metadata in metadatas:
            if not isinstance(metadata, dict):
                raise ValueError("Each metadata value must be a JSON object")
            try:
                serialized = json.dumps(metadata, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            except (TypeError, ValueError):
                raise ValueError("Metadata must be JSON serializable and contain only finite numbers") from None
            if len(serialized) > _TEXT_LIMIT:
                raise ValueError(f"Serialized metadata must be at most {_TEXT_LIMIT} characters")
            metadata_json.append(serialized)
        vectors = self._embed(documents)
        return [
            [self._collection_key, hashlib.sha256(node_id.encode("utf-8")).hexdigest(), node_id, doc, metadata, vector]
            for node_id, doc, metadata, vector in zip(ids, documents, metadata_json, vectors)
        ]

    def _write(self, rows, *, replace):
        table = self._options["table"]
        with self._session(write=True) as cursor:
            if replace:
                cursor.execute(f"DELETE FROM {table} WHERE collection_key = ?", [self._collection_key])
            for row in rows:
                if not replace:
                    cursor.execute(
                        f"DELETE FROM {table} WHERE collection_key = ? AND node_key = ?", row[:2]
                    )
                cursor.execute(
                    f"INSERT INTO {table} (collection_key, node_key, node_id, document, metadata_json, embedding) "
                    "VALUES (?, ?, ?, ?, ?, TO_VECTOR(?, DOUBLE))", row,
                )

    def add(self, ids, documents, metadatas):
        """Insert or update IDs atomically, retaining other nodes in this collection."""
        rows = self._prepare_rows(ids, documents, metadatas)
        if rows:
            self._write(rows, replace=False)

    def replace(self, ids, documents, metadatas):
        """Atomically replace this patient's index, including clearing an empty graph."""
        self._write(self._prepare_rows(ids, documents, metadatas), replace=True)

    def count(self):
        with self._session() as cursor:
            cursor.execute(
                f"SELECT COUNT(*) FROM {self._options['table']} WHERE collection_key = ?",
                [self._collection_key],
            )
            return int(cursor.fetchone()[0])

    def query(self, query_texts, n_results=10):
        if type(n_results) is not int or n_results < 1:
            raise ValueError("n_results must be a positive integer")
        if isinstance(query_texts, (str, bytes)):
            raise ValueError("query_texts must be a sequence of strings")
        try:
            texts = list(query_texts)
        except TypeError:
            raise ValueError("query_texts must be a sequence of strings") from None
        if any(not isinstance(text, str) or not text for text in texts):
            raise ValueError("query_texts must contain nonempty strings")
        vectors = self._embed(texts)
        result = {"ids": [], "distances": [], "documents": [], "metadatas": []}
        if not vectors:
            return result
        with self._session() as cursor:
            for vector in vectors:
                cursor.execute(
                    f"SELECT TOP {n_results} node_id, document, metadata_json, "
                    "VECTOR_COSINE(embedding, TO_VECTOR(?, DOUBLE)) AS similarity "
                    f"FROM {self._options['table']} WHERE collection_key = ? "
                    "ORDER BY similarity DESC, node_key ASC",
                    [vector, self._collection_key],
                )
                ids, distances, documents, metadatas = [], [], [], []
                for row in cursor.fetchall():
                    similarity = float(row[3])
                    if not math.isfinite(similarity):
                        raise ValueError("IRIS returned a nonfinite cosine similarity")
                    ids.append(str(row[0]))
                    distances.append(1.0 - min(1.0, max(-1.0, similarity)))
                    documents.append(str(row[1]))
                    metadatas.append(json.loads(row[2]))
                result["ids"].append(ids)
                result["distances"].append(distances)
                result["documents"].append(documents)
                result["metadatas"].append(metadatas)
        return result
