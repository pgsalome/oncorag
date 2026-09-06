"""Build the same graph entity records for every vector backend."""

from __future__ import annotations

from collections import Counter

import networkx as nx


DEFAULT_ENTITY_LABELS = [
    "Condition", "Treatment", "Procedure", "Anatomy", "GeneProtein", "Organism", "Other",
]


def graph_index_records(graph: nx.Graph, entity_type_filter: dict | None = None):
    if entity_type_filter:
        labels = set(entity_type_filter.get("required", []) + entity_type_filter.get("optional", []))
        labels.difference_update(entity_type_filter.get("exclude", []))
    else:
        labels = set(DEFAULT_ENTITY_LABELS)

    ids, documents, metadatas = [], [], []
    breakdown = Counter()
    for node_id, attrs in graph.nodes(data=True):
        label = attrs.get("label")
        if label not in labels:
            continue
        original = attrs.get("original_text", node_id)
        ids.append(str(node_id))
        documents.append(f"{original} ({label})")
        metadatas.append({
            "label": str(label),
            "original_text": str(attrs.get("original_text", "")),
            "source_model": str(attrs.get("source_model", "unknown")),
            "cluster_size": int(attrs.get("cluster_size") or 1),
            "is_negated": bool(attrs.get("is_negated", False)),
            "is_historical": bool(attrs.get("is_historical", False)),
            "is_family": bool(attrs.get("is_family", False)),
            "is_hypothetical": bool(attrs.get("is_hypothetical", False)),
        })
        breakdown[label] += 1
    return ids, documents, metadatas, dict(breakdown)
