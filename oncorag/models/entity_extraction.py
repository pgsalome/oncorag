"""Entity extraction and deduplication helpers."""

from __future__ import annotations

from typing import List, Dict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.logging_utils import log
from . import model_init


def extract_entities_from_single_model(
    text: str,
    model_name: str,
    context_filters: Dict,
) -> List[Dict]:
    """Extract entities using a single scispaCy model with medspaCy context."""
    model_init.initialize_models()
    nlp_sci = model_init.get_scispacy_model(model_name)
    doc_sci = nlp_sci(text)
    doc_med = model_init.NLP_MED(text)

    # Transfer scispaCy entities to medspaCy doc for context analysis
    new_ents = []
    for ent_sci in doc_sci.ents:
        try:
            span = doc_med.char_span(
                ent_sci.start_char,
                ent_sci.end_char,
                label=ent_sci.label_,
                alignment_mode="expand",
            )
            if span is not None:
                new_ents.append(span)
        except Exception:
            continue

    try:
        doc_med.ents = new_ents
    except Exception:
        pass

    entities: List[Dict] = []
    for ent in doc_med.ents:
        entity_info = {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
            "source_model": model_name,
            "is_negated": getattr(ent._, "is_negated", False),
            "is_historical": getattr(ent._, "is_historical", False),
            "is_family": getattr(ent._, "is_family", False),
            "is_hypothetical": getattr(ent._, "is_hypothetical", False),
        }

        if (
            (not context_filters.get("allow_negated", True) and entity_info["is_negated"])
            or (
                not context_filters.get("allow_hypothetical", True)
                and entity_info["is_hypothetical"]
            )
            or (
                not context_filters.get("allow_family", True)
                and entity_info["is_family"]
            )
            or (
                not context_filters.get("allow_historical", True)
                and entity_info["is_historical"]
            )
        ):
            continue

        entities.append(entity_info)

    return entities


def embed_entities(entities: List[Dict]) -> List[Dict]:
    """Attach embeddings to entity entries."""
    if not entities:
        return entities

    texts = [e["text"] for e in entities]
    model_init.initialize_models()
    try:
        embeddings = model_init.CLINICAL_EMBEDDER.encode(
            texts, show_progress_bar=False
        )
    except Exception as exc:
        log(
            f"SapBERT embedding failed on GPU ({exc}); retrying on CPU...",
            level="WARNING",
        )
        cpu_embedder = model_init.move_clinical_embedder_to_cpu(force_reload=True)
        embeddings = cpu_embedder.encode(texts, show_progress_bar=False)

    for entity, embedding in zip(entities, embeddings):
        entity["embedding"] = embedding

    return entities


def cluster_similar_entities(
    entities: List[Dict],
    similarity_threshold: float = 0.85,
) -> List[List[Dict]]:
    """Cluster entities by cosine similarity over embeddings."""
    if not entities:
        return []

    embeddings = np.array([e["embedding"] for e in entities])
    sim_matrix = cosine_similarity(embeddings)

    clusters: List[List[Dict]] = []
    used = set()

    for i in range(len(entities)):
        if i in used:
            continue

        similar_indices = np.where(sim_matrix[i] >= similarity_threshold)[0]
        cluster = [entities[j] for j in similar_indices]

        clusters.append(cluster)
        used.update(similar_indices)

    return clusters


def select_best_representative(cluster: List[Dict], model_priority: Dict) -> Dict:
    """Choose a representative entity from a cluster."""
    if len(cluster) == 1:
        return cluster[0]

    best = max(
        cluster,
        key=lambda e: (
            model_priority.get(e["source_model"], 0),
            len(e["text"]),
        ),
    )

    best["cluster_size"] = len(cluster)
    best["alternative_mentions"] = [
        e["text"] for e in cluster if e["text"] != best["text"]
    ]

    return best


def resolve_overlapping_spans(entities: List[Dict]) -> List[Dict]:
    """Remove overlapping spans while keeping best coverage."""
    if not entities:
        return []

    sorted_entities = sorted(entities, key=lambda e: (e["start"], -len(e["text"])))

    non_overlapping: List[Dict] = []
    last_end = -1

    for entity in sorted_entities:
        if entity["start"] >= last_end:
            non_overlapping.append(entity)
            last_end = entity["end"]
        else:
            if non_overlapping and entity["end"] > non_overlapping[-1]["end"]:
                if len(entity["text"]) > len(non_overlapping[-1]["text"]):
                    non_overlapping[-1] = entity
                    last_end = entity["end"]

    return non_overlapping


def extract_and_deduplicate_entities(
    text: str,
    model_configs: List[Dict],
    context_filters: Dict,
    dedup_config: Dict,
) -> List[Dict]:
    """Extract entities from multiple models and deduplicate them."""
    log(
        f"Extracting entities from {len(model_configs)} models...",
        level="STEP",
        debug=True,
    )

    all_entities: List[Dict] = []
    model_priority: Dict[str, int] = {}

    for idx, model_config in enumerate(model_configs):
        model_name = model_config["name"]
        priority = model_config.get("priority", idx + 1)
        model_priority[model_name] = priority

        entities = extract_entities_from_single_model(text, model_name, context_filters)
        all_entities.extend(entities)

        log(
            f"  Model {model_name}: {len(entities)} entities",
            level="INFO",
            debug=True,
        )

    if not all_entities:
        log("No entities found by any model", level="WARNING", debug=True)
        return []

    log(
        f"Total entities before deduplication: {len(all_entities)}",
        level="INFO",
        debug=True,
    )

    if not dedup_config.get("enabled", True):
        return all_entities

    all_entities = embed_entities(all_entities)

    similarity_threshold = dedup_config.get("similarity_threshold", 0.85)
    clusters = cluster_similar_entities(all_entities, similarity_threshold)

    log(
        f"Clustered into {len(clusters)} groups (threshold={similarity_threshold})",
        level="INFO",
        debug=True,
    )

    representatives = [
        select_best_representative(cluster, model_priority) for cluster in clusters
    ]

    final_entities = resolve_overlapping_spans(representatives)

    log(
        f"Final entities after deduplication: {len(final_entities)}",
        level="SUCCESS",
        debug=True,
    )

    return final_entities


__all__ = [
    "extract_entities_from_single_model",
    "extract_and_deduplicate_entities",
    "embed_entities",
    "cluster_similar_entities",
    "select_best_representative",
    "resolve_overlapping_spans",
]
