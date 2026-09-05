"""Route graph indexing to ChromaDB or InterSystems IRIS."""

from __future__ import annotations

import hashlib
import json

from .config import load_vector_store_config, validate_vector_store_config
from .records import graph_index_records
from ..utils.logging_utils import log


def get_vector_collection(patient_id: str, config: dict | None = None):
    settings = load_vector_store_config() if config is None else validate_vector_store_config(config)
    if not isinstance(patient_id, str) or not patient_id:
        raise ValueError("patient_id must be a nonempty string")
    if settings["backend"] == "iris":
        from .iris import IRISCollection
        from ..models.model_init import get_chroma_embedding_function

        return IRISCollection(
            patient_id,
            settings["iris"],
            get_chroma_embedding_function(),
            collection_namespace=settings["collection_namespace"],
        )

    from ..chroma.chroma_index import get_chroma_collection

    # Namespace and exact patient ID prevent collisions across cohorts and IDs.
    identity = json.dumps([settings["collection_namespace"], patient_id], ensure_ascii=False)
    collection_name = "oncorag_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:48]
    return get_chroma_collection(
        patient_id,
        cache_dir=settings["chroma"].get("path"),
        collection_name=collection_name,
    )


def index_graph_nodes(graph, collection, entity_type_filter=None, *, replace: bool = False):
    if getattr(collection, "backend", None) == "iris":
        ids, documents, metadatas, breakdown = graph_index_records(graph, entity_type_filter)
        write = collection.replace if replace else collection.add
        write(ids=ids, documents=documents, metadatas=metadatas)
        log(f"Indexed {len(ids)} entities in IRIS: {breakdown}", level="SUCCESS")
        return collection

    from ..chroma.chroma_index import index_graph_nodes_in_chroma

    return index_graph_nodes_in_chroma(graph, collection, entity_type_filter, replace=replace)
