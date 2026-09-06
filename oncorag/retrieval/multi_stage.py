"""Hybrid multi-stage graph retrieval inspired by GraphRAG survey best practices.

Stages:
 1. Query enhancement (expansion + decomposition) to enrich retrieval terms.
 2. Vector retrieval over Chroma to identify seed clinical entities.
 3. Structural expansion on the patient graph to assemble a focused subgraph.
 4. Knowledge pruning using lexical/semantic heuristics to reduce noise.

The implementation intentionally keeps model-free heuristics lightweight so it can
run inside the existing pipeline without additional training.
"""

from __future__ import annotations

import itertools
import math
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import networkx as nx

from ..utils.logging_utils import log


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class RetrievalStageDetails:
    """Capture diagnostics for each retrieval stage."""

    stage_name: str
    notes: Dict[str, object] = field(default_factory=dict)


@dataclass
class MultiStageRetrievalResult:
    """Aggregate the result and per-stage metadata."""

    start_nodes: List[str]
    expanded_nodes: Set[str]
    pruned_nodes: List[str]
    stages: List[RetrievalStageDetails]


# ---------------------------------------------------------------------------
# Query enhancement helpers
# ---------------------------------------------------------------------------


_STOPWORDS = {
    "the",
    "and",
    "with",
    "from",
    "that",
    "this",
    "these",
    "those",
    "are",
    "was",
    "were",
    "for",
    "have",
    "has",
    "had",
    "does",
    "did",
    "into",
    "onto",
    "per",
    "about",
    "patient",
    "clinical",
    "note",
    "notes",
    "history",
    "current",
    "past",
    "any",
    "case",
    "cases",
    "show",
    "tell",
    "list",
    "value",
    "status",
    "result",
    "results",
    "question",
    "answer",
}

_MAX_QUERY_TERMS = 80
_MAX_TERM_LEN = 48
_MAX_TERM_TOKENS = 5


def _normalize_term(term: str) -> str:
    clean = term.strip().lower()
    clean = re.sub(r"[^a-z0-9+/ ]+", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _morph_variants(term: str) -> Set[str]:
    """Return simple singular/plural and hyphen variants."""

    variants = {term}
    if term.endswith("s") and len(term) > 3:
        variants.add(term[:-1])
    else:
        variants.add(f"{term}s")

    variants.update({term.replace("-", " "), term.replace(" ", "-")})
    return {v for v in variants if v}


def expand_query_terms(
    question: str,
    base_keywords: Sequence[str] | None = None,
    synonyms: Sequence[str] | None = None,
    additional_terms: Sequence[str] | None = None,
) -> List[str]:
    """Generate diversified retrieval cues for the question.

    Inspired by GraphRAG survey Section 6.4 (query enhancement) where query
    expansion and decomposition are recommended to better cover relevant graph
    neighborhoods.
    """

    tokens = re.findall(r"[\w+/]+", question.lower())
    filtered = [_normalize_term(tok) for tok in tokens if tok not in _STOPWORDS and len(tok) > 2]

    def _yield_variants(term: str) -> Iterable[str]:
        yield term
        for variant in _morph_variants(term):
            yield variant
        if " " in term:
            for part in term.split():
                yield part
        for part in term.split("/"):
            if len(part) > 2:
                yield part

    seed_terms: List[str] = []
    seed_terms.extend(filtered)
    seed_terms.extend(_normalize_term(str(src)) for src in (base_keywords or []))
    seed_terms.extend(_normalize_term(str(syn)) for syn in (synonyms or []))
    seed_terms.extend(_normalize_term(str(extra)) for extra in (additional_terms or []))

    base_normalized = [
        term
        for term in (_normalize_term(str(src)) for src in (base_keywords or []))
        if term
    ]

    seen: Set[str] = set()
    ranked: List[str] = []

    def _should_keep(term: str, force: bool = False) -> bool:
        if not term or term in seen:
            return False
        if force:
            return True
        if term in _STOPWORDS:
            return False
        if not re.search(r"[a-z]", term):
            return False
        if len(term) > _MAX_TERM_LEN:
            return False
        if term.count(" ") >= _MAX_TERM_TOKENS:
            return False
        return True

    # Always prioritize explicit keywords from config
    for term in base_normalized:
        for variant in _yield_variants(term):
            normalized_variant = _normalize_term(variant)
            if _should_keep(normalized_variant, force=True):
                ranked.append(normalized_variant)
                seen.add(normalized_variant)
                if len(ranked) >= _MAX_QUERY_TERMS:
                    break
        if len(ranked) >= _MAX_QUERY_TERMS:
            break

    if len(ranked) < _MAX_QUERY_TERMS:
        for term in seed_terms:
            normalized = _normalize_term(term)
            for variant in _yield_variants(normalized):
                normalized_variant = _normalize_term(variant)
                if not _should_keep(normalized_variant):
                    continue
                ranked.append(normalized_variant)
                seen.add(normalized_variant)
                if len(ranked) >= _MAX_QUERY_TERMS:
                    break
            if len(ranked) >= _MAX_QUERY_TERMS:
                break

    log(
        f"Query expansion generated {len(ranked)} terms from question '{question[:60]}...'",
        level="INFO",
        debug=True,
    )
    return ranked


# ---------------------------------------------------------------------------
# Structural expansion and pruning
# ---------------------------------------------------------------------------


def _edge_importance(relation: str | None) -> float:
    """Map relation types to heuristic weights (stage 2 in survey)."""

    if not relation:
        return 0.2
    relation = relation.upper()
    if relation == "MENTIONS":
        return 1.0
    if relation.startswith("HAS_"):
        return 0.6
    if relation == "OCCURRED_ON":
        return 0.4
    if relation == "CO_OCCURS":
        return 0.35
    return 0.25


def _score_neighbor(current_score: float, relation: str | None, weight_attr: float | None) -> float:
    base = _edge_importance(relation)
    if weight_attr:
        base *= 1.0 + 0.1 * math.log1p(weight_attr)
    return current_score * base


def structural_expand_subgraph(
    graph: nx.Graph,
    seed_nodes: Iterable[str],
    max_depth: int = 2,
    max_nodes: int = 80,
) -> Tuple[Set[str], Dict[str, float]]:
    """Multi-hop expansion guided by relation heuristics.

    Aligns with non-parametric multi-stage retrieval approaches surveyed (e.g.,
    BFS/Steiner tree variants) but keeps implementation lightweight.
    """

    seeds = [node for node in seed_nodes if graph.has_node(node)]
    visited: Set[str] = set(seeds)
    scores: Dict[str, float] = {node: 1.0 for node in seeds}

    queue: deque[Tuple[str, int]] = deque((node, 0) for node in seeds)

    while queue and len(visited) < max_nodes:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue

        base_score = scores.get(node, 1.0)
        for neighbor in graph.neighbors(node):
            edge_data = graph.get_edge_data(node, neighbor) or {}
            relation = edge_data.get("relation")
            weight_attr = edge_data.get("weight")
            candidate_score = _score_neighbor(base_score, relation, weight_attr)

            if neighbor not in scores or candidate_score > scores[neighbor]:
                scores[neighbor] = candidate_score

            if neighbor not in visited and len(visited) < max_nodes:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    return visited, scores


def _token_frequency(tokens: Sequence[str]) -> Counter:
    return Counter(tok for tok in tokens if tok)


def _node_text_candidates(node: str, data: Dict[str, object]) -> List[str]:
    candidates: List[str] = [str(node)]
    for key in ("original_text", "label", "text"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value)
    return candidates


def prune_nodes_by_relevance(
    graph: nx.Graph,
    candidate_nodes: Iterable[str],
    question_terms: Sequence[str],
    max_nodes: int = 40,
    structural_scores: Dict[str, float] | None = None,
) -> List[str]:
    """Filter nodes via lexical overlap scoring.

    Mirrors survey Section 6.4.2 (knowledge pruning) where retrieved elements are
    re-evaluated to drop low-relevance information.
    """

    question_counter = _token_frequency(question_terms)
    scored: List[Tuple[float, str]] = []

    for node in candidate_nodes:
        if not graph.has_node(node):
            continue
        data = graph.nodes[node]
        label = str(data.get("label", ""))
        if label in {"Patient", "Note"}:
            continue
        if not any(ch.isalpha() for ch in str(node)):
            continue
        node_terms: Set[str] = set()
        for text in _node_text_candidates(node, data):
            tokens = re.findall(r"[a-z0-9+/]+", text.lower())
            node_terms.update(tokens)

        overlap = sum(
            question_counter.get(term, 0) * (1.0 + 0.1 * len(term))
            for term in node_terms
        )

        if overlap <= 0:
            continue

        label_weight = {
            "Condition": 1.6,
            "Procedure": 1.3,
            "Treatment": 1.25,
            "Anatomy": 1.1,
            "GeneProtein": 1.05,
            "Organism": 0.9,
        }.get(label, 0.8)

        structural_bonus = 1.0
        if structural_scores is not None:
            structural_bonus += min(structural_scores.get(node, 0.0), 1.0)

        total_score = overlap * label_weight * structural_bonus
        scored.append((total_score, node))

    top_nodes = [node for _, node in sorted(scored, key=lambda item: (-item[0], item[1]))[:max_nodes]]
    return top_nodes


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def multi_stage_graph_retrieval(
    question: str,
    graph: nx.Graph,
    vector_collection,
    base_keywords: Sequence[str] | None,
    synonyms: Sequence[str] | None,
    *,
    additional_terms: Sequence[str] | None = None,
    vector_top_k: int = 30,
    expansion_depth: int = 2,
    expansion_max_nodes: int = 120,
    prune_max_nodes: int = 40,
) -> MultiStageRetrievalResult:
    """Execute multi-stage retrieval and return pruned nodes for context assembly."""

    stages: List[RetrievalStageDetails] = []

    expanded_terms = expand_query_terms(question, base_keywords, synonyms, additional_terms)
    detail = RetrievalStageDetails(
        stage_name="query_expansion",
        notes={"terms": expanded_terms},
    )
    stages.append(detail)

    from ..chroma.chroma_index import find_start_nodes  # local import to avoid cycle

    start_nodes, stage1_info = find_start_nodes(
        vector_collection,
        expanded_terms,
        n_results=vector_top_k,
    )
    stages.append(
        RetrievalStageDetails(stage_name="vector_retrieval", notes=stage1_info),
    )

    if not start_nodes:
        return MultiStageRetrievalResult(
            start_nodes=[],
            expanded_nodes=set(),
            pruned_nodes=[],
            stages=stages,
        )

    expanded_nodes, score_map = structural_expand_subgraph(
        graph,
        start_nodes,
        max_depth=expansion_depth,
        max_nodes=expansion_max_nodes,
    )

    stages.append(
        RetrievalStageDetails(
            stage_name="structural_expansion",
            notes={
                "seed_count": len(start_nodes),
                "expanded_count": len(expanded_nodes),
                "score_summary": {
                    "max": max(score_map.values()) if score_map else 0.0,
                    "min": min(score_map.values()) if score_map else 0.0,
                    "mean": sum(score_map.values()) / len(score_map) if score_map else 0.0,
                },
            },
        )
    )

    pruned_nodes = prune_nodes_by_relevance(
        graph,
        expanded_nodes,
        expanded_terms,
        max_nodes=prune_max_nodes,
        structural_scores=score_map,
    )

    stages.append(
        RetrievalStageDetails(
            stage_name="knowledge_pruning",
            notes={
                "retained_nodes": pruned_nodes,
                "retained_count": len(pruned_nodes),
            },
        )
    )

    return MultiStageRetrievalResult(
        start_nodes=start_nodes,
        expanded_nodes=expanded_nodes,
        pruned_nodes=pruned_nodes,
        stages=stages,
    )


__all__ = [
    "expand_query_terms",
    "multi_stage_graph_retrieval",
    "structural_expand_subgraph",
    "prune_nodes_by_relevance",
    "MultiStageRetrievalResult",
    "RetrievalStageDetails",
]
