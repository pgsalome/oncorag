"""Graph helper utilities for querying and summarizing clinical graphs."""

from __future__ import annotations

from typing import List

import networkx as nx
import pandas as pd


def get_entity_details_from_graph(graph: nx.Graph, start_nodes: List[str]) -> List[dict]:
    """Extract detailed information about entities for downstream analysis."""
    entity_details: List[dict] = []

    for node in start_nodes:
        if graph.has_node(node):
            node_data = graph.nodes[node]
            entity_info = {
                "entity": node,
                "original_text": node_data.get("original_text", node),
                "label": node_data.get("label", "unknown"),
                "source_model": node_data.get("source_model", "unknown"),
                "cluster_size": node_data.get("cluster_size", 1),
                "is_negated": node_data.get("is_negated", False),
                "is_historical": node_data.get("is_historical", False),
                "is_family": node_data.get("is_family", False),
                "is_hypothetical": node_data.get("is_hypothetical", False),
            }

            if node_data.get("cluster_size", 1) > 1:
                neighbors = list(graph.neighbors(node))
                entity_info["neighbors"] = neighbors[:5]

            entity_details.append(entity_info)

    return entity_details


def get_clinical_entity_stats(graph: nx.Graph) -> dict:
    """Compute summary statistics for graph nodes."""
    node_data = dict(graph.nodes(data=True))
    node_count = len(node_data)

    stats = {
        "total_nodes": node_count,
        "total_edges": graph.number_of_edges(),
        "nodes_by_type": {},
        "clinical_entities": 0,
        "avg_degree": (
            sum(dict(graph.degree()).values()) / node_count if node_count > 0 else 0
        ),
    }

    if node_count == 0:
        stats["error"] = "Graph contains no nodes"
        return stats

    nodes_df = pd.DataFrame.from_dict(node_data, orient="index")

    if "label" not in nodes_df.columns:
        stats["error"] = "No labeled nodes found"
        return stats

    clinical_types = [
        "Condition",
        "Treatment",
        "Procedure",
        "Anatomy",
        "GeneProtein",
        "Organism",
        "Other",
    ]

    stats["nodes_by_type"] = nodes_df["label"].value_counts().to_dict()
    stats["clinical_entities"] = len(nodes_df[nodes_df["label"].isin(clinical_types)])
    return stats


__all__ = [
    "get_entity_details_from_graph",
    "get_clinical_entity_stats",
]
