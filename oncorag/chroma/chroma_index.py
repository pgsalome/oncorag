"""ChromaDB indexing and retrieval utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
import networkx as nx

from ..utils.logging_utils import log
from ..models.model_init import get_chroma_embedding_function
from ..vector_store.records import DEFAULT_ENTITY_LABELS, graph_index_records

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CACHE = PACKAGE_ROOT / "chroma_db_cache"
CHROMA_CACHE_DIR = Path(os.getenv("ONCORAG_CHROMA_CACHE_DIR", os.getenv("CHROMA_CACHE_DIR", str(_DEFAULT_CACHE))))
CHROMA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
def get_chroma_collection(
    patient_id: str, *, cache_dir: str | Path | None = None, collection_name: str | None = None
):
    """Create or retrieve a ChromaDB collection for a patient."""
    cache_path = Path(cache_dir) if cache_dir is not None else CHROMA_CACHE_DIR
    cache_path.mkdir(parents=True, exist_ok=True)
    # A storage failure must not erase other patients or silently lose persistence.
    client = chromadb.PersistentClient(path=str(cache_path))
    collection_name = collection_name or f"patient_{patient_id.replace('-', '_')}"

    embedding_function = get_chroma_embedding_function()

    try:
        collection = client.get_collection(name=collection_name)
        collection._embedding_function = embedding_function  # type: ignore[attr-defined]
        return collection
    except Exception:
        log("Creating new ChromaDB collection with ClinicalBERT embeddings", level="STEP")
        return client.create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )


def index_graph_nodes_in_chroma(
    graph: nx.Graph,
    collection,
    entity_type_filter: Dict | None = None,
    *,
    replace: bool = False,
) -> chromadb.api.models.Collection.Collection:
    """Index clinical entities in ChromaDB with optional type filtering."""
    log("Indexing entities in ChromaDB with biomedical embeddings...", level="STEP")

    node_ids, documents, metadatas, breakdown = graph_index_records(graph, entity_type_filter)
    if replace:
        old_ids = collection.get(include=[])["ids"]
        if old_ids:
            collection.delete(ids=old_ids)
    if not node_ids:
        log("No relevant nodes to index after filtering", level="WARNING")
        return collection

    new_collection = collection

    try:
        new_collection.add(documents=documents, ids=node_ids, metadatas=metadatas)
    except chromadb.errors.InvalidArgumentError as exc:
        message = str(exc).lower()
        if "dimension" not in message:
            raise

        log(
            "Chroma collection embedding dimension mismatch detected; recreating collection",
            level="WARNING",
        )
        collection_name = getattr(collection, "name", None)
        try:
            server_api = getattr(collection, "_client", None)
            if server_api is not None:
                # Collection._client is ServerAPI; rebuilding requires public ClientAPI.
                client = chromadb.Client(
                    settings=server_api.get_settings(),
                    tenant=getattr(collection, "tenant", "default_tenant"),
                    database=getattr(collection, "database", "default_database"),
                )
            else:
                client = chromadb.PersistentClient(path=str(CHROMA_CACHE_DIR))
            if collection_name:
                client.delete_collection(name=collection_name)
                log(
                    f"Deleted stale Chroma collection '{collection_name}'",
                    level="INFO",
                )
            new_collection = client.create_collection(
                name=collection_name or f"patient_{os.urandom(4).hex()}",
                embedding_function=get_chroma_embedding_function(),
                metadata={"hnsw:space": "cosine"},
            )
            new_collection.add(documents=documents, ids=node_ids, metadatas=metadatas)
        except Exception as recreate_exc:
            log(
                f"Failed to rebuild Chroma collection: {recreate_exc}",
                level="ERROR",
            )
            raise exc

    log(f"Indexed {len(node_ids)} entities: {breakdown}", level="SUCCESS")
    return new_collection


def find_start_nodes(
    collection,
    keywords: List[str],
    n_results: int | None = None,
) -> Tuple[List[str], Dict]:
    """Search a vector collection for relevant graph nodes using biomedical embeddings."""
    log(
        f"Searching for entities matching keywords: {keywords[:3]}{'...' if len(keywords) > 3 else ''}",
        level="STEP",
    )

    limit = n_results or int(os.getenv("ONCORAG_RETRIEVAL_RESULTS", "20") or 20)
    retrieval_details: Dict = {
        "keywords_used": keywords,
        "n_results_requested": limit,
        "searches_performed": [],
    }

    try:
        if not keywords:
            return [], retrieval_details

        all_candidate_nodes: List[str] = []

        for keyword in keywords:
            results = collection.query(query_texts=[keyword], n_results=limit)
            nodes = results["ids"][0] if results["ids"] else []
            distances = results.get("distances", [[]])[0]

            all_candidate_nodes.extend(nodes)

            retrieval_details["searches_performed"].append(
                {
                    "keyword": keyword,
                    "results_count": len(nodes),
                    "avg_distance": (sum(distances) / len(distances)) if distances else None,
                    "best_distance": min(distances) if distances else None,
                    "top_matches": [
                        {"entity": nodes[i], "distance": distances[i]}
                        for i in range(min(3, len(nodes)))
                    ]
                    if nodes and distances
                    else [],
                }
            )

        combined_query = " ".join(keywords)
        results = collection.query(query_texts=[combined_query], n_results=limit)
        combined_nodes = results["ids"][0] if results["ids"] else []
        combined_distances = results.get("distances", [[]])[0]

        all_candidate_nodes.extend(combined_nodes)

        retrieval_details["searches_performed"].append(
            {
                "keyword": f"COMBINED: {combined_query}",
                "results_count": len(combined_nodes),
                "avg_distance": (sum(combined_distances) / len(combined_distances))
                if combined_distances
                else None,
                "best_distance": min(combined_distances) if combined_distances else None,
            }
        )

        unique_nodes = list(dict.fromkeys(all_candidate_nodes))

        retrieval_details["total_candidates_before_dedup"] = len(all_candidate_nodes)
        retrieval_details["total_unique_entities"] = len(unique_nodes)
        retrieval_details["duplicates_removed"] = len(all_candidate_nodes) - len(unique_nodes)
        retrieval_details["final_nodes_returned"] = min(len(unique_nodes), limit)

        if unique_nodes:
            log(f"Found {len(unique_nodes)} candidate entities", level="SUCCESS")
        else:
            log("No matching entities found", level="WARNING")

        return unique_nodes[:limit], retrieval_details
    except Exception as exc:
        log(f"Error querying vector index: {exc}", level="ERROR")
        retrieval_details["error"] = str(exc)
        return [], retrieval_details


__all__ = [
    "get_chroma_collection",
    "index_graph_nodes_in_chroma",
    "find_start_nodes",
]
