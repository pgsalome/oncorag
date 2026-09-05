"""Utilities for handling filesystem caching and file-level graph processing."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable

import networkx as nx
from datetime import datetime
import traceback

from .logging_utils import log


def cache_slugify(feature: str | None) -> str:
    """Normalize feature names for filesystem-safe cache paths."""
    if not feature:
        return "unknown_feature"

    slug = str(feature).strip()
    if not slug:
        return "unknown_feature"

    slug = slug.replace(os.sep, "_")
    slug = slug.replace("/", "_")
    slug = re.sub(r"\s+", "_", slug)
    slug = re.sub(r"_+", "_", slug)
    return slug


def save_prompt_to_cache(
    prompt,
    context,
    feature,
    pid,
    response,
    cache_dir: Path,
    raw_context: str | None = None,
    retrieved_entities: list | None = None,
    graph_stats: dict | None = None,
    retrieval_info: dict | None = None,
    reranking_details: dict | None = None,
    config_info: dict | None = None,
    timing_info: dict | None = None,
    validation_info: dict | None = None,
) -> None:
    """Persist prompt, context, and metadata for auditing."""
    feature_slug = cache_slugify(feature)
    feature_dir = cache_dir / feature_slug
    feature_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = feature_dir / f"{pid}_{feature_slug}_{timestamp}.json"

    # Extract per-query reranked contexts if available
    per_query_reranked_contexts = []
    if isinstance(response, dict) and "per_query_results" in response:
        per_query_reranked_contexts = [
            {
                "question": qr.get("question", ""),
                "reranked_context": qr.get("reranked_context", ""),
                "prompt": qr.get("prompt", ""),
            }
            for qr in response.get("per_query_results", [])
        ]

    data = {
        "timestamp": timestamp,
        "patient_id": pid,
        "feature": feature,
        "final_result": response,
        "prompt_sent_to_llm": prompt,
        "reranked_context_sent_to_llm": context,
        "raw_context_before_reranking": raw_context,
        "raw_context_stats": (
            {
                "sentence_count": len([s for s in raw_context.split("\n") if s.strip()]),
                "char_count": len(raw_context),
                "avg_sentence_length": len(raw_context)
                / max(len([s for s in raw_context.split("\n") if s.strip()]), 1),
            }
            if raw_context
            else None
        ),
        "reranked_context_stats": {
            "sentence_count": len([s for s in context.split("\n") if s.strip()]),
            "char_count": len(context),
            "sentences_filtered_out": (
                len([s for s in raw_context.split("\n") if s.strip()])
                - len([s for s in context.split("\n") if s.strip()])
            )
            if raw_context
            else 0,
        },
    }
    
    # Add per-query reranked contexts if available
    if per_query_reranked_contexts:
        data["per_query_reranked_contexts"] = per_query_reranked_contexts

    if isinstance(response, dict) and response.get("gt_value") is not None:
        data["ground_truth"] = response["gt_value"]

    if retrieved_entities:
        entity_summary = {
            "entities": retrieved_entities,
            "total_count": len(retrieved_entities),
            "by_model": {},
            "by_label": {},
            "negated_count": sum(1 for e in retrieved_entities if e.get("is_negated")),
            "historical_count": sum(
                1 for e in retrieved_entities if e.get("is_historical")
            ),
            "hypothetical_count": sum(
                1 for e in retrieved_entities if e.get("is_hypothetical")
            ),
            "family_count": sum(1 for e in retrieved_entities if e.get("is_family")),
            "total_cluster_size": sum(
                e.get("cluster_size", 1) for e in retrieved_entities
            ),
            "entities_that_were_deduplicated": sum(
                1 for e in retrieved_entities if e.get("cluster_size", 1) > 1
            ),
        }
        for ent in retrieved_entities:
            model = ent.get("source_model", "unknown")
            entity_summary["by_model"][model] = (
                entity_summary["by_model"].get(model, 0) + 1
            )
            label = ent.get("label", "unknown")
            entity_summary["by_label"][label] = (
                entity_summary["by_label"].get(label, 0) + 1
            )
        data["retrieved_entities"] = entity_summary

    if graph_stats:
        data["graph_statistics"] = graph_stats
    if retrieval_info:
        data["retrieval_info"] = retrieval_info
    if reranking_details:
        data["reranking_details"] = reranking_details
    if config_info:
        data["configuration"] = config_info
    if timing_info:
        data["timing_performance"] = timing_info
    if validation_info:
        data["validation"] = validation_info

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)

    log(f"Comprehensive cache saved: {output_path}", level="INFO", debug=True)


def process_single_file(
    file_path: Path,
    patient_id: str,
    model_configs: list[dict],
    context_filters: dict,
    dedup_config: dict,
    *,
    split_fn: Callable[[str], list[str]],
    process_notes_fn: Callable[
        [list[str], str, str, list[dict], dict, dict, str | None], nx.Graph
    ],
) -> nx.Graph:
    """Read a single text file and convert it into a graph via supplied processors."""
    try:
        log(f"Reading file: {file_path.name}", level="STEP", debug=True)
        with open(file_path, "r", encoding="utf-8") as handle:
            content = handle.read()
        notes = split_fn(content)
        if not notes:
            log(f"No documents found in {file_path.name}", level="WARNING", debug=True)
            return nx.Graph()
        return process_notes_fn(
            notes,
            patient_id,
            file_path.name,
            model_configs,
            context_filters,
            dedup_config,
            str(file_path),
        )
    except Exception as exc:
        tb = traceback.format_exc()
        log(
            f"Error processing {file_path.name}: {exc}\n{tb}",
            level="ERROR",
            debug=True,
        )
        return nx.Graph()
