"""Lightweight graph-based reranker built from retrieved context sentences."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from ..models import model_init
from ..utils.logging_utils import log


@dataclass
class _ContextSentence:
    index: int
    text: str


class GraphReranker:
    """Construct a small graph from reranked sentences and diffuse relevance scores."""

    def __init__(
        self,
        max_sentences: int = 128,
        similarity_threshold: float = 0.55,
        max_neighbors: int = 12,
        num_layers: int = 2,
        alpha: float = 0.6,
    ) -> None:
        self.max_sentences = max_sentences
        self.similarity_threshold = similarity_threshold
        self.max_neighbors = max_neighbors
        self.num_layers = max(1, num_layers)
        self.alpha = alpha
        self._embedder: Optional[SentenceTransformer] = None

    def score(
        self,
        question: str,
        sentences: Sequence[str],
    ) -> Tuple[Optional[List[float]], Optional[Dict[str, int]]]:
        if not question or not sentences:
            return None, None

        subset = [
            _ContextSentence(index=i, text=sent)
            for i, sent in enumerate(sentences[: self.max_sentences])
            if sent and sent.strip()
        ]
        if not subset:
            return None, None

        features = self._encode_sentences([item.text for item in subset])
        if features is None:
            return None, None

        adjacency = self._build_adjacency(features)
        propagated = self._propagate(features, adjacency)
        question_vec = self._encode_sentences([question])
        if question_vec is None:
            return None, None
        question_vec = question_vec[0]

        subset_scores = propagated @ question_vec
        subset_scores = self._normalize(subset_scores)

        scores = np.zeros(len(sentences), dtype=np.float32)
        for item, score in zip(subset, subset_scores):
            scores[item.index] = float(score)

        metadata = {
            "processed": len(subset),
            "graph_nodes": adjacency.shape[0],
            "total_candidates": len(sentences),
        }
        return scores.tolist(), metadata

    def _ensure_embedder(self) -> SentenceTransformer:
        if self._embedder is not None:
            return self._embedder
        model_init.initialize_models()
        embedder = model_init.CLINICAL_EMBEDDER
        if embedder is None:
            log(
                "SapBERT embedder unavailable; falling back to all-MiniLM-L6-v2.",
                level="WARNING",
            )
            embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self._embedder = embedder
        return embedder

    def _encode_sentences(self, texts: Sequence[str]) -> Optional[NDArray[np.float32]]:
        if not texts:
            return None
        embedder = self._ensure_embedder()
        try:
            emb = embedder.encode(
                list(texts),
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            log(f"Sentence embedding failed: {exc}", level="WARNING")
            return None
        return np.array(emb, dtype=np.float32)

    def _build_adjacency(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        if features.size == 0:
            return np.zeros((0, 0), dtype=np.float32)

        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        normalized = features / norms
        similarity = normalized @ normalized.T
        np.fill_diagonal(similarity, 0.0)

        if self.similarity_threshold > 0.0:
            similarity[similarity < self.similarity_threshold] = 0.0

        n = similarity.shape[0]
        if self.max_neighbors and self.max_neighbors < n:
            for i in range(n):
                row = similarity[i]
                if np.count_nonzero(row) <= self.max_neighbors:
                    continue
                keep_idx = np.argpartition(row, -self.max_neighbors)[-self.max_neighbors :]
                mask = np.ones_like(row, dtype=bool)
                mask[keep_idx] = False
                similarity[i][mask] = 0.0

        # Symmetrize to encourage mutual influence.
        similarity = np.maximum(similarity, similarity.T)
        np.fill_diagonal(similarity, 1.0)

        row_sum = similarity.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return similarity / row_sum

    def _propagate(self, features: NDArray[np.float32], adjacency: NDArray[np.float32]) -> NDArray[np.float32]:
        propagated = features.copy()
        base = features.copy()
        for _ in range(self.num_layers):
            propagated = adjacency @ propagated
            propagated = self.alpha * propagated + (1.0 - self.alpha) * base
            propagated = np.maximum(propagated, 0.0)
        return propagated

    @staticmethod
    def _normalize(scores: NDArray[np.float32]) -> NDArray[np.float32]:
        if scores.size == 0:
            return scores
        min_val = float(scores.min())
        max_val = float(scores.max())
        if max_val - min_val < 1e-6:
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)


_GRAPH_RERANKER: Optional[GraphReranker] = None


def get_graph_reranker() -> GraphReranker:
    global _GRAPH_RERANKER
    if _GRAPH_RERANKER is None:
        max_sentences = int(os.getenv("ONCORAGGRAPH_GRAPH_RERANK_MAX_DOCS", "128") or 128)
        similarity_threshold = float(os.getenv("ONCORAGGRAPH_GRAPH_RERANK_SIM_THRESHOLD", "0.55") or 0.55)
        neighbors = int(os.getenv("ONCORAGGRAPH_GRAPH_RERANK_MAX_NEIGHBORS", "12") or 12)
        layers = int(os.getenv("ONCORAGGRAPH_GRAPH_RERANK_LAYERS", "2") or 2)
        alpha = float(os.getenv("ONCORAGGRAPH_GRAPH_RERANK_ALPHA", "0.6") or 0.6)
        _GRAPH_RERANKER = GraphReranker(
            max_sentences=max_sentences,
            similarity_threshold=similarity_threshold,
            max_neighbors=neighbors,
            num_layers=layers,
            alpha=alpha,
        )
    return _GRAPH_RERANKER


__all__ = ["GraphReranker", "get_graph_reranker"]
