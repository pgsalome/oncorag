"""Prompt construction and context reranking helpers."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import re

from ..utils.logging_utils import log
from ..config.system_config import get_system_config
from ..models import model_init

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


GRAPH_RERANKER_ENABLED = os.getenv("ONCORAG_ENABLE_GRAPH_RERANKER", "false").lower() in {
    "1",
    "true",
    "yes",
}
GRAPH_RERANK_WEIGHT = float(os.getenv("ONCORAG_GRAPH_RERANK_WEIGHT", "0.3") or 0.3)


def _runtime_defaults() -> Dict:
    try:
        cfg = get_system_config().config
    except Exception:
        return {}
    if isinstance(cfg, dict):
        return cfg.get("runtime_defaults", {}) or {}
    return {}


def _runtime_int(key: str, env_key: str, default: int) -> int:
    val = _runtime_defaults().get(key)
    if val is None or val == "":
        env_val = os.getenv(env_key)
        if env_val is not None and env_val != "":
            val = env_val
    try:
        return int(val)
    except Exception:
        return default


def resolve_missing_option(config: Dict) -> Tuple[Optional[str], str]:
    """Return (option_key, label) representing the Missing choice."""
    output_format = config.get("output_format") or {}
    options = output_format.get("options") if isinstance(output_format, dict) else None

    missing_key: Optional[str] = None
    missing_label = "Missing"

    if isinstance(options, dict) and options:
        for key, value in options.items():
            if isinstance(value, str) and value.strip().lower() == "missing":
                missing_key = key
                missing_label = value
                break
        if missing_key is None:
            keys = list(options.keys())
            if keys:
                missing_key = keys[-1]
                missing_label = options[missing_key]

    return missing_key, missing_label


def get_context_from_graph_with_metadata(
    graph: nx.Graph,
    start_nodes: List[str],
    max_depth: int = 2,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract context sentences from a graph neighborhood with trace metadata."""
    log(f"Extracting context from graph (depth={max_depth})...", level="STEP")

    context_items: List[Dict[str, Any]] = []
    index_by_sentence: Dict[str, int] = {}

    for start_node in sorted((n for n in start_nodes if graph.has_node(n)), key=str):
        ego_nodes = sorted(nx.ego_graph(graph, start_node, radius=max_depth).nodes(), key=str)
        for node in ego_nodes:
            for neighbor in sorted(graph.neighbors(node), key=str):
                edge_data = graph.get_edge_data(node, neighbor)
                if not edge_data or "source_sentence" not in edge_data:
                    continue
                sentence = str(edge_data.get("source_sentence") or "").strip()
                if not sentence:
                    continue
                idx = index_by_sentence.get(sentence)
                if idx is None:
                    idx = len(context_items)
                    index_by_sentence[sentence] = idx
                    context_items.append(
                        {
                            "sentence": sentence,
                            "sentence_ids": set(),
                            "note_ids": set(),
                            "note_dates": set(),
                            "note_files": set(),
                            "note_paths": set(),
                        }
                    )

                item = context_items[idx]
                sentence_id = edge_data.get("source_sentence_id")
                note_id = edge_data.get("source_note_id")
                if not note_id:
                    node_label = graph.nodes[node].get("label") if graph.has_node(node) else None
                    neighbor_label = graph.nodes[neighbor].get("label") if graph.has_node(neighbor) else None
                    if node_label == "Note":
                        note_id = node
                    elif neighbor_label == "Note":
                        note_id = neighbor
                if sentence_id:
                    item["sentence_ids"].add(str(sentence_id))
                if note_id:
                    item["note_ids"].add(str(note_id))

                if note_id and graph.has_node(note_id):
                    note_attrs = graph.nodes[note_id]
                    note_date = note_attrs.get("note_date")
                    note_file = note_attrs.get("note_file")
                    note_path = note_attrs.get("note_path")
                    if note_date:
                        item["note_dates"].add(str(note_date))
                    if note_file:
                        item["note_files"].add(str(note_file))
                    if note_path:
                        item["note_paths"].add(str(note_path))
                if note_id and not item["note_files"]:
                    note_id_str = str(note_id)
                    if "_note_" in note_id_str:
                        item["note_files"].add(note_id_str.split("_note_", 1)[0])

    for item in context_items:
        item["sentence_ids"] = sorted(item["sentence_ids"])
        item["note_ids"] = sorted(item["note_ids"])
        item["note_dates"] = sorted(item["note_dates"])
        item["note_files"] = sorted(item["note_files"])
        item["note_paths"] = sorted(item["note_paths"])
    context_items.sort(key=lambda item: (
        item["sentence"], tuple(item["sentence_ids"]), tuple(item["note_ids"]),
    ))

    log(f"Retrieved {len(context_items)} unique context sentences", level="SUCCESS")
    return "\n".join(item["sentence"] for item in context_items), context_items


def get_context_from_graph(
    graph: nx.Graph,
    start_nodes: List[str],
    max_depth: int = 2,
) -> str:
    """Extract context sentences from a graph neighborhood."""
    context, _ = get_context_from_graph_with_metadata(
        graph,
        start_nodes,
        max_depth=max_depth,
    )
    return context


def build_prompt(config: Dict, context: str) -> str:
    """Build an extraction prompt with negation handling."""
    med_filter = (
        f"\n**Medical Context Filter:**\n{config['medical_context']}"
        if config.get("medical_context")
        else ""
    )

    negation_instructions = """
**CRITICAL - Negation Handling:**
- Pay CLOSE attention to negative language: "no", "denies", "without", "ruled out", "negative for", "absence of"
- "No evidence of surgery" → Surgery = No
- "Patient did not undergo mastectomy" → Surgery = No  
- "Ruled out for surgery" → Surgery = No
- Only respond "Yes" if there is CLEAR POSITIVE evidence of surgery being performed
- Distinguish "history of" (past event, still counts as Yes) from "no history of" (No)
"""

    guidelines_block = ""
    extraction_guidelines = (
        config.get("rules", {}).get("extraction_guidelines")
        if isinstance(config.get("rules"), dict)
        else None
    )
    if extraction_guidelines:
        guidelines_block = "\n**Extraction Guidelines:**\n" + "\n".join(
            f"- {guideline}" for guideline in extraction_guidelines if guideline
        )
        if not guidelines_block.endswith("\n"):
            guidelines_block += "\n"

    base = f"""You are a specialized clinical data analyst extracting information from breast cancer medical records.

**Context:**
---
{context}
---

{negation_instructions}

**Task:** Based ONLY on the context above, determine the value for "{config['feature_name']}".

**Description:** {config['description']}{med_filter}
{guidelines_block}
"""

    if config.get("data_type") in ["boolean", "categorical", "ordinal"]:
        options = (config.get("output_format") or {}).get("options")
        if options:
            opts_lines = []
            lang = os.getenv("ONCORAG_LANGUAGE", "").lower()
            for k, v in options.items():
                label = v
                if lang.startswith("ger"):
                    # Add common German mappings inline for clarity
                    if str(v).lower() == "left":
                        label = f"{v} (links)"
                    elif str(v).lower() == "right":
                        label = f"{v} (rechts)"
                    elif str(v).lower() == "bilateral":
                        label = f"{v} (beidseitig)"
                    elif str(v).lower() == "central":
                        label = f"{v} (zentral)"
                opts_lines.append(f"  {k}: {label}")

            opts = "\n".join(opts_lines)
            missing_key, missing_default = resolve_missing_option(config)
            option_keys = list(options.keys())
            allowed_keys_str = ", ".join(option_keys)
            fallback_key = missing_key or (option_keys[-1] if option_keys else "Missing")

            examples = ""
            if config.get("examples"):
                examples = "\n**Examples:**"
                for example in config["examples"]:
                    examples += (
                        f'\nContext: "{example["context"]}"\nOutput: {example["output"]}\n'
                    )

            strict_rules = """
**STRICT OUTPUT RULES:**
- Return a single JSON object containing ONLY the keys `reasoning`, `confidence`, `value`, and `evidence`.
- `value` MUST be set to exactly one of the option keys ({allowed_keys_str}). Do not spell out the option text.
- If no option is supported by the context, set `value` to "{fallback_key}".
- Do NOT introduce additional fields such as "answer", "diagnosis", or any other keys.
"""

            type_inst = f"""
**Options:**
{opts}

{examples}

**If the context does not contain explicit evidence supporting any option, respond with option {fallback_key} ({missing_default}).**

{strict_rules}

**Response Format (JSON):**
{{
  "reasoning": "Brief explanation citing specific phrases from context",
  "confidence": "High/Medium/Low",
  "value": "one of: {allowed_keys_str}",
  "evidence": "Direct quote from context supporting your answer"
}}

Respond ONLY with valid JSON. Do not include any text outside the JSON object.
"""
        else:
            log(
                f"No output options defined for feature '{config.get('feature_name')}'. "
                "Falling back to free-text extraction instructions.",
                level="WARNING",
            )
            type_inst = """
**Response Format (JSON):**
{
  "reasoning": "Brief explanation citing specific phrases from context",
  "confidence": "High/Medium/Low",
  "value": "extracted value",
  "evidence": "Direct quote from context supporting your answer"
}
"""
    else:
        type_inst = """
**Response Format (JSON):**
{
  "reasoning": "Brief explanation",
  "confidence": "High/Medium/Low",
  "value": "extracted value",
  "evidence": "Direct quote from context"
}
"""

    return base + type_inst


def rerank_context(
    context: str,
    query: str,
    top_k: int = 5,
    keywords: Optional[List[str]] = None,
    normalized_name: Optional[str] = None,
    synonyms: Optional[List[str]] = None,
    expected_values: Optional[List[str]] = None,
    sentence_meta: Optional[List[Dict[str, Any]]] = None,
    runtime_options: Optional[Dict[str, Any]] = None,
) -> Tuple[str, float, Dict]:
    """Rerank context sentences by relevance using cross-encoder with keyword boosting."""
    if not context or not context.strip():
        return "", 0.0, {}
    runtime_options = runtime_options or {}
    weights = runtime_options.get("weights", {})
    diffusion = runtime_options.get("graph_diffusion", {})

    raw_lines = [raw.strip() for raw in context.split("\n") if raw.strip()]
    meta_by_sentence: Dict[str, Dict[str, Any]] = {}
    if sentence_meta:
        for meta in sentence_meta:
            if not isinstance(meta, dict):
                continue
            sent = str(meta.get("sentence") or "").strip()
            if sent and sent not in meta_by_sentence:
                meta_by_sentence[sent] = meta
            # Model sentence segmentation may span report lines; each line keeps provenance.
            for line in sent.splitlines():
                if line.strip():
                    meta_by_sentence.setdefault(line.strip(), meta)

    seen_sentences: set[str] = set()
    sentences: List[str] = []
    aligned_meta: List[Dict[str, Any]] = []
    for idx, raw in enumerate(raw_lines):
        sentence = raw.strip()
        if not sentence or sentence in seen_sentences:
            continue
        seen_sentences.add(sentence)
        sentences.append(sentence)
        if sentence_meta and len(sentence_meta) == len(raw_lines):
            meta = sentence_meta[idx]
        else:
            meta = meta_by_sentence.get(sentence) if sentence_meta else None
        if not isinstance(meta, dict):
            meta = {"sentence": sentence}
        aligned_meta.append(meta)

    # Further split very long sentences at natural clause boundaries so key phrases
    # (e.g., semicolon-separated demographic facts) stay visible to the reranker.
    max_clause_len = int(os.getenv("ONCORAG_RERANK_CLAUSE_MAX_LEN", "160") or 160)
    refined_sentences: List[str] = []
    refined_meta: List[Dict[str, Any]] = []
    for sentence, meta in zip(sentences, aligned_meta):
        if len(sentence) <= max_clause_len and ";" not in sentence and "," not in sentence:
            refined_sentences.append(sentence)
            refined_meta.append(meta)
            continue

        # Split on semicolons first, then commas, keeping delimiters out of the output.
        clause_candidates: List[str] = []
        for delimiter in ((";",) if runtime_options else (";", ",")):
            parts = [part.strip() for part in sentence.split(delimiter)]
            if len(parts) > 1:
                clause_candidates = parts
                break

        if clause_candidates:
            for clause in clause_candidates:
                if clause:
                    refined_sentences.append(clause)
                    refined_meta.append(meta)
        else:
            refined_sentences.append(sentence)
            refined_meta.append(meta)

    sentences = refined_sentences
    aligned_meta = refined_meta
    
    # Dynamically build medical abbreviation normalization map from config
    # Extract abbreviations from synonyms and feature context
    # e.g., "ETOH Use" synonym + "Alcohol Use" normalized_name → ETOH maps to "alcohol"
    medical_abbrev_map: Dict[str, str] = {}
    
    if normalized_name:
        # Extract the main concept word from normalized_name (e.g., "Alcohol Use" → "alcohol")
        main_concept = normalized_name.lower().split()[0] if normalized_name else None
        
        # Common medical abbreviations that should map to common concepts
        common_abbrevs = {
            "etoh": "alcohol",
            "alc": "alcohol", 
            "hx": "history",
            "s/p": "status post",
            "pt": "patient",
            "dx": "diagnosis",
        }
        
        # Add common abbreviations if they match the concept
        if main_concept in ["alcohol", "drinking"]:
            medical_abbrev_map.update({
                "etoh": "alcohol",
                "+etoh": "alcohol",
                "etoh/": "alcohol",
                "etoh ": "alcohol ",
                "alc ": "alcohol ",
                "alc.": "alcohol",
            })
        elif main_concept in ["history"]:
            medical_abbrev_map.update({
                "hx ": "history ",
                "hx.": "history",
            })
        
        # Extract abbreviations from synonyms
        # Look for patterns like "ETOH Use" where ETOH is an abbreviation
        if synonyms:
            for synonym in synonyms:
                syn_lower = synonym.lower()
                # Pattern: "ABBREV Word" or "Word ABBREV" where ABBREV is all caps
                abbrev_matches = re.findall(r'\b([A-Z]{2,})\b', synonym)
                for abbrev in abbrev_matches:
                    abbrev_lower = abbrev.lower()
                    # If we have a main concept, map abbreviation to it
                    if main_concept and abbrev_lower not in medical_abbrev_map:
                        # Check if the synonym contains the main concept (e.g., "ETOH Use" contains "Use" from "Alcohol Use")
                        remaining_words = syn_lower.replace(abbrev_lower, "").strip()
                        if main_concept in remaining_words or any(word in main_concept for word in remaining_words.split() if len(word) > 3):
                            medical_abbrev_map[abbrev_lower] = main_concept
                            # Also add common variants
                            medical_abbrev_map[f"+{abbrev_lower}"] = main_concept
                            medical_abbrev_map[f"{abbrev_lower}/"] = main_concept
                            medical_abbrev_map[f"{abbrev_lower} "] = f"{main_concept} "
    
    # Normalize medical abbreviations in sentences before reranking
    # This helps the cross-encoder understand the semantic connection
    normalized_sentences = []
    for sentence in sentences:
        normalized = sentence.lower()
        # Apply abbreviations in order (longer first to avoid partial matches)
        for abbrev, full_term in sorted(medical_abbrev_map.items(), key=lambda x: -len(x[0])):
            normalized = normalized.replace(abbrev, full_term)
        normalized_sentences.append(normalized)
    # Keep original sentences for output, but use normalized for reranking
    sentences_for_reranking = normalized_sentences
    
    stopwords = {
        "the",
        "and",
        "with",
        "from",
        "that",
        "this",
        "these",
        "those",
        "patient",
        "patients",
        "medical",
        "record",
        "records",
        "number",
        "mrn",
        "dob",
        "date",
        "birth",
        "age",
        "years",
        "clinic",
        "hospital",
        "visit",
        "notes",
        "note",
        "progress",
        "the patient",
    }
    if runtime_options:
        stopwords.difference_update({"date", "age", "years", "birth"})
    important_short_terms = {"no", "yes", "rare", "mild", "high", "low", "none", "etoh", "age"}

    lexical_terms: Dict[str, float] = {}
    core_patterns: set[str] = set()

    concept_tokens = {
        tok
        for tok in (normalized_name or "").lower().split()
        if tok and tok not in stopwords and len(tok) >= 4
    }
    single_word_synonyms = {
        syn.strip().lower()
        for syn in (synonyms or [])
        if isinstance(syn, str)
        and syn.strip()
        and " " not in syn.strip()
        and syn.strip().lower() not in stopwords
    }

    feature_terms = [normalized_name or "", *(synonyms or [])]
    short_feature_terms = {
        term.strip().lower()
        for term in feature_terms
        if isinstance(term, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9]{1,2}", term.strip())
    }
    for term in feature_terms:
        # Preserve configured abbreviations such as ER in "ER status".
        short_feature_terms.update(
            token.lower() for token in re.findall(r"\b[A-Z][A-Z0-9]{1,2}\b", str(term))
        )
    short_feature_terms.difference_update(stopwords)

    def _extract_tokens(text: str) -> list[str]:
        return [
            tok for tok in re.findall(r"[a-z0-9]+", text.lower())
            if len(tok) >= 4 or tok in short_feature_terms
        ]

    if normalized_name:
        core_patterns.update(_extract_tokens(normalized_name))
    if synonyms:
        for syn in synonyms:
            core_patterns.update(_extract_tokens(str(syn)))

    lowered_name = (normalized_name or "").lower()
    if "alcohol" in lowered_name or "etoh" in lowered_name:
        core_patterns.update({"alcohol", "drink", "drinks", "drinking", "etoh", "wine", "beer", "liquor"})
    if not core_patterns:
        core_patterns.update({"alcohol", "etoh"})

    def _contains_keyword(lower_sentence: str, term: str) -> bool:
        term = term.lower()
        if not term:
            return False
        if (len(term) <= 3 and term.isalpha()) or (
            len(term) <= 4 and term.isalnum() and any(char.isdigit() for char in term)
        ):
            return re.search(rf"\b{re.escape(term)}\b", lower_sentence) is not None
        return term in lower_sentence

    def _is_relevant(term: str) -> bool:
        lower = term.lower()
        return any(_contains_keyword(lower, pattern) for pattern in core_patterns)

    boost_terms: List[str] = []

    def _append_boost_term(term: str) -> None:
        term = term.strip().lower()
        if not term:
            return
        if term in stopwords:
            return
        if not _is_relevant(term):
            return
        boost_terms.append(term)

    if keywords:
        # Add full keywords that actually contain the core concept tokens.
        for keyword in keywords:
            if not keyword:
                continue
            lowered = keyword.strip().lower()
            _append_boost_term(lowered)
            # Also extract individual tokens from multi-word keywords for better matching.
            for token in lowered.split():
                if len(token) > 2 and (
                    token in concept_tokens
                    or token in single_word_synonyms
                    or token in important_short_terms
                ):
                    _append_boost_term(token)
            # Add common abbreviations/variants for known concepts.
            if "alcohol" in lowered:
                for variant in ("etoh", "+etoh"):
                    _append_boost_term(variant)
    query_lower = (query or "").lower()

    if "surgery" in query_lower:
        boost_terms.extend(
            [
                "s/p",
                "status post",
                "lumpectomy",
                "mastectomy",
                "slnb",
                "snb",
                "sln",
                "sentinel lymph node",
                "sentinel node biopsy",
                "axillary dissection",
                "alnd",
                "wire localized",
                "partial mastectomy",
                "re-excision",
                "breast excision",
                "port removal",
                "operative note",
                "was performed",
                "date of surgery",
                "lumpectomy/snb",
                "initial surgery",
                "first surgery",
                "primary surgery",
                "neoadjuvant",
                "date of operation",
                "date obtained",
                "date collected",
                "specimen collected",
                "specimen obtained",
                "accession date",
                "date received",
                "operating room",
                "surgical pathology report",
                "operative report",
            ]
        )
    if "radiotherapy" in query_lower or "radiation" in query_lower or "radiotherapy performed" in query_lower:
        boost_terms.extend([
            "radiation",
            "radiotherapy",
            "radiation therapy",
            "radiation treatment",
            "imrt",
            "protons",
            "radiation completed",
            "radiation dermatitis",
            "radiation fields",
            "radiation oncology",
            "gray",
            "cgy",
            "boost",
            "irradiation",
            "post-radiation",
        ])

    boost_terms = [term for term in dict.fromkeys(boost_terms) if len(term) > 1]

    important_short_terms = {"no", "yes", "rare", "mild", "high", "low", "none", "etoh"}

    def _register_term(term: str, weight: float = 1.0) -> None:
        if not term:
            return
        cleaned = term.strip().lower()
        if not cleaned:
            return
        if len(cleaned) == 1:
            return
        if cleaned in stopwords:
            return
        if (
            len(cleaned) < 4
            and cleaned not in important_short_terms
            and cleaned not in short_feature_terms
            and " " not in cleaned
        ):
            return
        lexical_terms[cleaned] = max(weight, lexical_terms.get(cleaned, 0.0))

    base_terms: List[str] = []
    if keywords:
        base_terms.extend(str(t) for t in keywords if str(t).strip())
    if synonyms:
        base_terms.extend(str(t) for t in synonyms if str(t).strip())
    if normalized_name:
        base_terms.append(str(normalized_name))
        base_terms.extend(part for part in str(normalized_name).split() if part)
    if expected_values:
        expected_terms = [str(v).strip() for v in expected_values if str(v).strip()]
    else:
        expected_terms = []

    for term in base_terms:
        if not _is_relevant(term):
            continue
        _register_term(term, weight=1.0)
        if " " in term:
            for token in term.split():
                if _is_relevant(token):
                    _register_term(token, weight=0.6)

    for term in expected_terms:
        if _is_relevant(term):
            _register_term(term, weight=0.8)
        if " " in term:
            for token in term.split():
                if _is_relevant(token):
                    _register_term(token, weight=0.5)

    if boost_terms:
        filtered_boost = [term for term in boost_terms if _is_relevant(term)]
        if filtered_boost:
            boost_terms = filtered_boost
        else:
            boost_terms = list(core_patterns)
    else:
        boost_terms = list(core_patterns)

    boost_terms = [term for term in dict.fromkeys(boost_terms)]

    query_tokens = [tok for tok in re.split(r"[^a-z0-9]+", query.lower()) if tok]
    for token in query_tokens:
        _register_term(token, weight=0.4)

    sent_lowers = [s.lower() for s in sentences_for_reranking]

    graph_scores: Optional[List[float]] = None
    graph_metadata: Optional[Dict[str, int]] = None
    if diffusion.get("enabled", GRAPH_RERANKER_ENABLED) and sentences:
        try:
            from ..rerank.graphrag_reranker import get_graph_reranker

            if diffusion:
                from ..rerank.graphrag_reranker import GraphReranker
                graph_reranker = GraphReranker(
                    max_sentences=len(sentences),
                    similarity_threshold=diffusion.get("similarity_threshold", .55),
                    max_neighbors=diffusion.get("max_neighbors", 12),
                    num_layers=diffusion.get("iterations", 2),
                    alpha=diffusion.get("residual_alpha", .6),
                )
            else:
                graph_reranker = get_graph_reranker()
            if graph_reranker is not None:
                graph_scores, graph_metadata = graph_reranker.score(query, sentences)
        except Exception as exc:  # pragma: no cover - runtime safety
            if runtime_options:
                raise RuntimeError("Configured graph reranking failed") from exc
            log(f"Graph reranker failed: {exc}", level="WARNING")
            graph_scores = None
            graph_metadata = None
        if graph_scores is not None and len(graph_scores) != len(sentences):
            graph_scores = None

    max_candidates = _runtime_int("rerank_candidates", "ONCORAG_RERANK_CANDIDATES", 512)
    if len(sentences) > max_candidates:
        priority_indices = []
        fallback_indices = []
        for idx, lower in enumerate(sent_lowers):
            if any(_contains_keyword(lower, term) for term in boost_terms):
                priority_indices.append(idx)
            else:
                fallback_indices.append(idx)

        selected_idx = priority_indices[:max_candidates]
        if len(selected_idx) < max_candidates:
            selected_idx.extend(fallback_indices[: max_candidates - len(selected_idx)])
        selected_idx = sorted(selected_idx)
        sentences = [sentences[i] for i in selected_idx]  # Keep original sentences
        aligned_meta = [aligned_meta[i] for i in selected_idx] if aligned_meta else []
        sentences_for_reranking = [sentences_for_reranking[i] for i in selected_idx]
        sent_lowers = [sent_lowers[i] for i in selected_idx]
        if graph_scores is not None:
            graph_scores = [graph_scores[i] for i in selected_idx]
    if not sentences:
        return "", 0.0, {}

    log(f"Re-ranking {len(sentences)} sentences for relevance...", level="STEP")

    # Use normalized sentences for reranking (better semantic matching)
    # But keep original sentences for output
    pairs = [[query, sentence] for sentence in sentences_for_reranking]
    try:
        scores = model_init.get_combined_reranker_scores(pairs)
    except Exception as exc:
        message = str(exc).lower()
        is_cuda_issue = "cuda" in message or "cublas" in message or "hip" in message
        if not is_cuda_issue:
            raise

        log("CUDA execution failed during reranking; retrying on CPU...", level="WARNING")
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        fallback_limit = _runtime_int("rerank_cpu_candidates", "ONCORAG_RERANK_CPU_CANDIDATES", 256)
        if len(sentences) > fallback_limit:
            log(
                f"Reducing rerank candidates to {fallback_limit} for CPU fallback",
                level="WARNING",
            )
            sentences = sentences[:fallback_limit]
            aligned_meta = aligned_meta[:fallback_limit] if aligned_meta else []
            sentences_for_reranking = sentences_for_reranking[:fallback_limit]
            sent_lowers = sent_lowers[:fallback_limit]
            if graph_scores is not None:
                graph_scores = graph_scores[:fallback_limit]

        pairs = [[query, sentence] for sentence in sentences_for_reranking]
        scores = model_init.get_combined_reranker_scores(
            pairs,
            device_override="cpu",
            force_reload=True,
            prefer_lightweight=True,
        )

    semantic_scores = [float(x) for x in scores]

    # Additional semantic pass using normalized name/synonyms to spotlight direct matches
    name_queries: List[str] = []
    if normalized_name and isinstance(normalized_name, str) and normalized_name.strip():
        name_queries.append(normalized_name.strip())
    if synonyms:
        for syn in synonyms[:5]:
            if syn and isinstance(syn, str) and syn.strip():
                name_queries.append(syn.strip())
    if expected_values:
        for val in expected_values[:3]:
            if val and isinstance(val, str) and val.strip():
                name_queries.append(val.strip())
    name_queries = list(dict.fromkeys(name_queries))[:5]

    name_scores_norm = [0.0] * len(sentences)
    if name_queries and sentences:
        name_pairs = []
        for query in name_queries:
            name_pairs.extend([[query, sentence] for sentence in sentences_for_reranking])
        try:
            raw_name_scores = model_init.get_combined_reranker_scores(name_pairs)
        except Exception:
            raw_name_scores = []
        if getattr(raw_name_scores, "tolist", None):
            raw_name_scores = raw_name_scores.tolist()
        if isinstance(raw_name_scores, (list, tuple)) and len(raw_name_scores):
            per_sentence = len(sentences)
            best_scores = [float("-inf")] * per_sentence
            offset = 0
            for _ in name_queries:
                segment = raw_name_scores[offset: offset + per_sentence]
                for idx, score in enumerate(segment):
                    if score > best_scores[idx]:
                        best_scores[idx] = float(score)
                offset += per_sentence
            best_scores = [0.0 if val == float("-inf") else val for val in best_scores]
            max_name_score = max(best_scores) if best_scores else 0.0
            if max_name_score > 0:
                name_scores_norm = [score / max_name_score for score in best_scores]

    lexical_scores: List[float] = []
    if lexical_terms:
        for lower in sent_lowers:
            score = 0.0
            has_alcohol_term = any(_contains_keyword(lower, tok) for tok in core_patterns)
            for term, weight in lexical_terms.items():
                if _contains_keyword(lower, term):
                    score += weight
            if re.search(r"\b\d+(\.\d+)?\s*(standard\s+)?drinks?\b", lower):
                score += 1.0
            if "per week" in lower and has_alcohol_term:
                score += 0.6
            if has_alcohol_term and any(word in lower for word in ["occasional", "occasionally", "rarely", "social"]):
                score += 0.6
            negative_phrases = [
                "not on file",
                "not currently",
                "not documented",
                "no alcohol",
                "denies alcohol",
                "unknown",
                "not provided",
            ]
            if any(phrase in lower for phrase in negative_phrases):
                score -= 0.5
            lexical_scores.append(score)
    else:
        lexical_scores = [0.0] * len(sentences)

    max_lexical = max(lexical_scores) if lexical_scores else 0.0
    lexical_weight = weights.get("lexical_weight", float(os.getenv("ONCORAG_LEXICAL_WEIGHT", "0.25") or 0.25))
    lexical_components = [
        lexical_weight * (lex / max_lexical) if max_lexical > 0 else 0.0
        for lex in lexical_scores
    ]

    name_weight = weights.get("name_weight", float(os.getenv("ONCORAG_NAME_WEIGHT", "0.3") or 0.3))
    semantic_weight = weights.get("semantic_weight", 1.0)
    combined_scores = [
        semantic_weight * semantic_scores[i] + lexical_components[i] + name_weight * name_scores_norm[i]
        for i in range(len(sentences))
    ]
    if graph_scores is not None:
        combined_scores = [
            combined_scores[i] + weights.get("graph_weight", GRAPH_RERANK_WEIGHT) * graph_scores[i]
            for i in range(len(sentences))
        ]

    base_ranked = list(zip(combined_scores, sentences))

    has_keyword = (
        [
            any(_contains_keyword(lower, term) for term in boost_terms)
            for lower in sent_lowers
        ]
        if boost_terms
        else [False] * len(sentences)
    )

    alpha = weights.get("boost_alpha", 0.4)
    boosted_scores = [
        base + (alpha if flag else 0.0)
        for base, flag in zip([score for score, _ in base_ranked], has_keyword)
    ]

    penalty_terms: List[str] = []
    if "recurrence" in query_lower:
        penalty_terms.extend([
            "no evidence of recurrence",
            "no recurrence",
            "without recurrence",
            "ned",
            "disease free",
            "free of recurrent",
            "negative for recurrence",
        ])
    if "surgery date" in query_lower or "surgery" in query_lower:
        penalty_terms.extend([
            "revision",
            "revision reconstruction",
            "reconstruction",
            "revision reconstruction",
            "revision breast",
            "reconstruction",
            "repeat",
            "re-exploration",
            "second stage",
            "revision procedure",
            "reconstruction breast",
            "re-do",
        ])
    toxicity_related = False
    normalized_feature = (normalized_name or "").lower()
    if any(keyword in normalized_feature for keyword in ["toxicity", "side effect", "adverse reaction"]):
        toxicity_related = True
    if "toxicity" in query_lower or "side effect" in query_lower:
        toxicity_related = True
    if toxicity_related:
        penalty_terms.extend([
            "anticipated side effects",
            "potential side effects",
            "discussed potential",
            "discussing potential",
            "discussing anticipated",
            "education on side effects",
            "anticipatory guidance",
            "counseled about side effects",
        ])
    penalty_terms = [term for term in dict.fromkeys(penalty_terms)]

    beta = weights.get("penalty_beta", 0.4) if penalty_terms else 0.0
    if penalty_terms:
        penalty_flags = [any(term in lower for term in penalty_terms) for lower in sent_lowers]
        boosted_scores = [
            score - (beta if flag else 0.0)
            for score, flag in zip(boosted_scores, penalty_flags)
        ]
    else:
        penalty_flags = [False] * len(sentences)

    ranked_idx = sorted(
        range(len(sentences)), key=lambda i: boosted_scores[i], reverse=True
    )

    max_final_sentences = int(os.getenv("ONCORAG_RERANK_TOP_FINAL", "20") or 20)
    target_top_k = max(1, top_k if runtime_options else min(top_k, max_final_sentences))

    min_keyword = min(5, target_top_k) if boost_terms else 0
    top_idx = ranked_idx[:target_top_k]
    keyword_hits = sum(1 for i in top_idx if has_keyword[i])

    if boost_terms and keyword_hits < min_keyword:
        keyword_candidates = [i for i in ranked_idx[target_top_k:] if has_keyword[i]]
        j = target_top_k - 1
        while keyword_candidates and keyword_hits < min_keyword and j >= 0:
            if not has_keyword[top_idx[j]]:
                top_idx[j] = keyword_candidates.pop(0)
                keyword_hits += 1
            j -= 1

    top_idx = sorted(top_idx, key=lambda i: boosted_scores[i], reverse=True)
    top_sentences = [sentences[i] for i in top_idx]
    def _collect_meta_values(items: List[Dict[str, Any]], key: str) -> List[str]:
        values: List[str] = []
        for item in items:
            for val in item.get(key) or []:
                sval = str(val)
                if sval and sval not in values:
                    values.append(sval)
        return values
    keyword_fallback_limit = 0 if runtime_options else int(os.getenv("ONCORAG_KEYWORD_FALLBACK", "5") or 5)
    if boost_terms and keyword_fallback_limit > 0:
        keyword_candidates = [
            i for i in ranked_idx if i not in top_idx and has_keyword[i]
        ][:keyword_fallback_limit]
        for idx in keyword_candidates:
            sentence = sentences[idx]
            if sentence not in top_sentences:
                top_sentences.append(sentence)
                top_idx.append(idx)
    top_meta = [aligned_meta[i] for i in top_idx] if aligned_meta else []
    top_score = max(([boosted_scores[i] for i in top_idx]), default=0.0)

    ranked_for_details = sorted(
        [
            {
                "sentence": (sentences[i][:100] + "...") if len(sentences[i]) > 100 else sentences[i],
                "semantic_score": float(semantic_scores[i]),
                "lexical_score": float(lexical_scores[i]),
                "combined_score": float(combined_scores[i]),
                "score": float(boosted_scores[i]),
                "boosted_score": float(boosted_scores[i]),
                "has_keyword": bool(has_keyword[i]),
                "penalized": bool(penalty_flags[i]) if penalty_terms else False,
                "kept": i in top_idx,
            }
            for i in range(len(sentences))
        ],
        key=lambda item: item["boosted_score"],
        reverse=True,
    )

    remaining_scores = [
        boosted_scores[i] for i in range(len(sentences)) if i not in top_idx
    ]

    reranking_details: Dict = {
        "total_sentences_before": len(sentences),
        "total_sentences_after": len(top_sentences),
        "sentences_dropped": len(sentences) - len(top_sentences),
        "top_k": target_top_k,
        "best_score": float(max(boosted_scores)) if sentences else 0.0,
        "worst_score_kept": (
            float(min([boosted_scores[i] for i in top_idx])) if top_idx else None
        ),
        "best_score_dropped": float(max(remaining_scores)) if remaining_scores else None,
        "semantic_scores": [float(score) for score in semantic_scores],
        "lexical_scores": [float(score) for score in lexical_scores],
        "combined_scores": [float(score) for score in combined_scores],
        "all_scores": [float(score) for score in boosted_scores],
        "score_distribution": {
            "min": float(min(boosted_scores)),
            "max": float(max(boosted_scores)),
            "mean": float(sum(boosted_scores) / len(boosted_scores)),
            "median": float(sorted([float(score) for score in boosted_scores])[len(sentences) // 2]),
        },
        "boost_alpha": alpha if boost_terms else 0.0,
        "penalty_beta": beta if penalty_terms else 0.0,
        "lexical_weight": lexical_weight,
        "semantic_weight": semantic_weight,
        "name_weight": name_weight,
        "graph_weight": weights.get("graph_weight", GRAPH_RERANK_WEIGHT),
        "min_keyword_enforced": min_keyword,
        "keywords_used_for_boost": boost_terms,
        "keywords_used_for_penalty": penalty_terms,
        "sentences_with_scores": ranked_for_details,
        "graph_reranker_scores": graph_scores,
        "graph_reranker_metadata": graph_metadata,
        "top_sentences": top_sentences,
        "top_sentence_meta": top_meta,
        "top_sentence_ids": _collect_meta_values(top_meta, "sentence_ids"),
        "top_note_ids": _collect_meta_values(top_meta, "note_ids"),
        "top_note_dates": _collect_meta_values(top_meta, "note_dates"),
        "top_note_files": _collect_meta_values(top_meta, "note_files"),
        "top_note_paths": _collect_meta_values(top_meta, "note_paths"),
    }

    log(
        f"Selected top {len(top_sentences)} sentences (keyword hits in top: {keyword_hits}/{len(top_sentences)})",
        level="SUCCESS",
    )

    return "\n".join(top_sentences), float(top_score), reranking_details


__all__ = [
    "get_context_from_graph",
    "get_context_from_graph_with_metadata",
    "build_prompt",
    "rerank_context",
]
