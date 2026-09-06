"""Graph construction utilities for the clinical extraction pipeline."""

from __future__ import annotations

import os
import re
from datetime import date as calendar_date
from pathlib import Path
from typing import List, Optional, Pattern, Tuple
from urllib.parse import quote

import networkx as nx
import yaml

from ..utils.logging_utils import log
from ..utils.parsing_utils import extract_date_from_note, normalize_entity_text
from ..ingestion import NoteRecord, group_notes_by_patient
from ..models.entity_extraction import extract_and_deduplicate_entities
from ..models.model_init import get_scispacy_model


def _load_dataset_profile() -> str:
    """Fetch dataset profile from environment or system configuration."""
    env_value = os.getenv("ONCORAG_DATASET_PROFILE")
    if env_value:
        return env_value.strip().lower()

    config_path = Path(__file__).resolve().parents[1] / "system_config.yaml"
    if config_path.exists():
        try:
            config = yaml.safe_load(config_path.read_text()) or {}
            profile = config.get("dataset_profile")
            if isinstance(profile, str) and profile.strip():
                return profile.strip().lower()
        except Exception:
            log(
                "Unable to read dataset_profile from system_config.yaml; defaulting to 'default'",
                level="WARNING",
                debug=True,
            )

    return "default"


DATASET_PROFILE = _load_dataset_profile()
KNOWN_DATASET_PROFILES = {"default", "mimic", "ricci", "custom"}


def _load_report_end_pattern() -> Tuple[Optional[str], Optional[Pattern[str]]]:
    raw_pattern = os.getenv("ONCORAG_REPORT_END_PATTERN")
    if not raw_pattern:
        return None, None
    text = raw_pattern.strip()
    if not text:
        return None, None
    try:
        compiled = re.compile(text, re.IGNORECASE)
        return text, compiled
    except re.error as exc:
        log(
            f"Invalid ONCORAG_REPORT_END_PATTERN regex '{raw_pattern}': {exc}",
            level="WARNING",
            debug=True,
        )
        return None, None


REPORT_END_PATTERN, REPORT_END_REGEX = _load_report_end_pattern()
_WARNED_UNKNOWN_DATASET = False
_WARNED_CUSTOM_PATTERN = False


def _split_mimic_style_notes(file_content: str) -> List[str]:
    """Split notes formatted with MIMIC-style metadata headers."""
    sections: List[str] = []
    current: List[str] = []

    for raw_line in file_content.splitlines():
        line = raw_line.rstrip("\n")

        if re.match(r"^={5,}$", line):
            if current:
                sections.append("\n".join(current).strip())
                current = []
            continue

        if re.match(r"^Note ID:\s+", line):
            if current:
                sections.append("\n".join(current).strip())
                current = []
            current.append(line)
            continue

        if re.match(r"^===", line):
            if current:
                sections.append("\n".join(current).strip())
                current = []
            continue

        current.append(line)

    if current:
        sections.append("\n".join(current).strip())

    sections = [s for s in sections if s]
    return sections


def _split_notes_by_end_pattern(file_content: str, pattern: Pattern[str]) -> List[str]:
    """Split notes when a regex indicates the end of each report."""
    if not file_content.strip():
        return []
    sections: List[str] = []
    current: List[str] = []
    for raw_line in file_content.splitlines():
        current.append(raw_line)
        if pattern.search(raw_line):
            section = "\n".join(current).strip()
            if section:
                sections.append(section)
            current = []
    if current:
        section = "\n".join(current).strip()
        if section:
            sections.append(section)
    return sections


def _clean_ricci_document(document: str) -> str:
    """Trim Ricci cohort letters to drop boilerplate headers/footers."""
    if not document.strip():
        return ""

    lines = document.splitlines()
    content_pattern = re.compile(
        r"^(patient|diagnosen?|diagnose|therapie|chronologie|aktuell|sehr|liebe|herrn?|frau|zur|an\s+die|bei|wir)\b",
        re.IGNORECASE,
    )

    start_idx = 0
    for idx, line in enumerate(lines):
        normalized = line.strip().lstrip("|").strip()
        if not normalized:
            continue
        if content_pattern.match(normalized):
            start_idx = idx
            break
    else:
        header_keywords = (
            "universitätsklinikum",
            "universitaetsklinikum",
            "universitatsklinikum",
            "klinikum ",
            "krankenhaus ",
            "radioonkologie",
            "strahlentherapie",
            "terminvereinbarungen",
            "ambulanz",
            "privat-sprechstunde",
            "tel",
            "fax",
            "anschrift",
            "postfach",
        )
        for idx, line in enumerate(lines):
            normalized = line.strip().lstrip("|").strip().lower()
            if not normalized:
                continue
            if normalized.startswith(header_keywords):
                continue
            start_idx = idx
            break

    trimmed_lines = lines[start_idx:]
    while trimmed_lines and not trimmed_lines[0].strip():
        trimmed_lines = trimmed_lines[1:]

    footer_markers = ("elektronisches dokument", "*999", "-----------------------")
    end_idx = len(trimmed_lines)
    for idx in range(len(trimmed_lines) - 1, -1, -1):
        normalized = trimmed_lines[idx].strip()
        lowered = normalized.lower()
        if not normalized:
            end_idx = idx
            continue
        if lowered.startswith(footer_markers) or lowered == "*999999*":
            end_idx = idx
            continue
        break

    trimmed_lines = trimmed_lines[:end_idx]
    return "\n".join(trimmed_lines).strip()


def _split_ricci_style_notes(file_content: str) -> List[str]:
    """Split Ricci cohort files, which are letter-style documents."""
    if not file_content.strip():
        return []

    header_pattern = re.compile(
        r"^(?:\|)?\s*(?:universit(?:ä|ae|a)tsklinikum|klinikum|krankenhaus)\b",
        re.IGNORECASE,
    )
    sections: List[str] = []
    current: List[str] = []

    for raw_line in file_content.splitlines():
        if header_pattern.match(raw_line.strip()):
            if current:
                sections.append("\n".join(current).strip())
                current = []
        current.append(raw_line)

    if current:
        sections.append("\n".join(current).strip())

    if not sections:
        sections = [file_content]

    cleaned_sections = []
    for section in sections:
        cleaned = _clean_ricci_document(section)
        if cleaned:
            cleaned_sections.append(cleaned)

    return cleaned_sections


def split_into_documents(file_content: str) -> List[str]:
    """Split a single raw file into individual note documents."""
    global _WARNED_UNKNOWN_DATASET, _WARNED_CUSTOM_PATTERN

    if DATASET_PROFILE not in KNOWN_DATASET_PROFILES:
        if not _WARNED_UNKNOWN_DATASET:
            log(
                f"Dataset profile '{DATASET_PROFILE}' not recognized; falling back to default heuristics.",
                level="WARNING",
            )
            _WARNED_UNKNOWN_DATASET = True

    if DATASET_PROFILE == "custom":
        if REPORT_END_REGEX:
            custom_docs = _split_notes_by_end_pattern(file_content, REPORT_END_REGEX)
            if custom_docs:
                return custom_docs
            if not _WARNED_CUSTOM_PATTERN:
                log(
                    "report_end_pattern did not match any sections; using default heuristics.",
                    level="WARNING",
                )
                _WARNED_CUSTOM_PATTERN = True
        else:
            if not _WARNED_CUSTOM_PATTERN:
                log(
                    "dataset_profile 'custom' requires report_end_pattern; using default heuristics.",
                    level="WARNING",
                )
                _WARNED_CUSTOM_PATTERN = True

    if DATASET_PROFILE == "ricci":
        ricci_docs = _split_ricci_style_notes(file_content)
        if ricci_docs:
            return ricci_docs

    prefer_mimic = DATASET_PROFILE == "mimic"

    if prefer_mimic:
        mimic_docs = _split_mimic_style_notes(file_content)
        if mimic_docs:
            return mimic_docs

    split_pattern = (
        r"("
        r"The PATIENT is .*?, (?:with|and the) M(?:RN|edical Record Number).*? (?:DOB|Date of Birth).*?\."
        r"|The PATIENT is .*?, (?:with|and the) (?:M(?:RN|edical Record Number)|(?:DOB|Date of Birth)).*?\."
        r"|The PATIENT is .*?, and the MRN is \d+\."
        r"|The PATIENT is .*? and the MRN is \d+\."
        r"|The PATIENT is .*?, and her medical record number is \d+\."
        r"|PATIENT is [A-Z][a-z]+, [A-Z][a-z]+ L"
        r"|The PATIENT is [A-Z][a-z]+, [A-Z][a-z]+ L"
        r"|The PATIENT is [A-Za-z, \.]+\."
        r")"
    )

    raw_documents = re.split(f"(?={split_pattern})", file_content, flags=re.IGNORECASE)
    clean_documents = [
        doc.strip()
        for doc in raw_documents
        if doc.strip()
        and re.match(split_pattern.strip("()"), doc.strip(), flags=re.IGNORECASE)
    ]

    if not clean_documents:
        clean_documents = _split_mimic_style_notes(file_content)

    if not clean_documents:
        log(
            "No header patterns found. Using entire file as single document.",
            level="WARNING",
            debug=True,
        )
        if file_content.strip():
            clean_documents = [file_content.strip()]
        else:
            log("File is empty or contains only whitespace.", level="ERROR")

    return clean_documents


def add_or_get_node(graph: nx.Graph, name: str, **attrs) -> str:
    """Ensure a node exists and return its identifier."""
    if name and not graph.has_node(name):
        graph.add_node(name, **attrs)
    return name


def process_notes_to_graph(
    notes: List[str],
    patient_id: str,
    file_name: str,
    model_configs: List[dict],
    context_filters: dict,
    dedup_config: dict,
    file_path: str | None = None,
    *,
    note_id: str | None = None,
    note_date: str | None = None,
    report_type: str | None = None,
    language: str | None = None,
    include_report_sentences: bool = False,
) -> nx.Graph:
    """Convert notes into a graph, preferring explicitly supplied note metadata."""
    if not model_configs:
        raise ValueError("At least one entity extraction model is required")
    if note_id is not None and (not note_id.strip() or len(notes) != 1):
        raise ValueError("An explicit note_id requires exactly one note")
    if note_date is not None:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", note_date):
            raise ValueError("Explicit note_date must be YYYY-MM-DD")
        calendar_date.fromisoformat(note_date)
    graph = nx.Graph()
    patient_node = add_or_get_node(graph, patient_id, label="Patient")

    log(
        f"Processing {len(notes)} clinical notes from {file_name}...",
        level="STEP",
        debug=True,
    )

    first_model = get_scispacy_model(model_configs[0]["name"])

    for i, note_text in enumerate(notes):
        source_note_id = note_id or f"{file_name}_note_{i}"
        graph_note_id = (
            f"note:{quote(patient_id, safe='')}:{quote(source_note_id, safe='')}"
            if note_id is not None else source_note_id
        )
        note_node = add_or_get_node(
            graph,
            graph_note_id,
            label="Note",
            text=note_text if note_id is not None else note_text[:500],
            note_id=source_note_id,
            note_file=file_name,
            note_path=file_path or file_name,
        )
        if report_type is not None:
            graph.nodes[note_node]["report_type"] = report_type
        if language is not None:
            graph.nodes[note_node]["language"] = language
        graph.add_edge(patient_node, note_node, relation="HAS_NOTE")

        resolved_note_date = note_date if note_date is not None else extract_date_from_note(note_text)
        if resolved_note_date:
            date_node = add_or_get_node(graph, resolved_note_date, label="Date")
            graph.add_edge(note_node, date_node, relation="HAS_DATE")
            # Store canonical note date on the note node for downstream lookups.
            graph.nodes[note_node]["note_date"] = resolved_note_date

        entities = extract_and_deduplicate_entities(
            note_text,
            model_configs,
            context_filters,
            dedup_config,
        )

        doc = first_model(note_text)
        sent_texts = [sent.text for sent in doc.sents]
        if include_report_sentences:
            # Preserve facts that biomedical entity recognition does not identify.
            for sentence_index, sentence in enumerate(sent_texts):
                if not sentence.strip():
                    continue
                sentence_id = f"{graph_note_id}_sent_{sentence_index}"
                graph.add_node(
                    sentence_id,
                    label="Sentence",
                    text=sentence,
                    original_text=sentence,
                    source_model="report_sentence",
                    note_id=source_note_id,
                    note_date=resolved_note_date,
                    report_type=report_type,
                    language=language,
                    note_file=file_name,
                    note_path=file_path or file_name,
                )
                graph.add_edge(
                    note_node,
                    sentence_id,
                    relation="CONTAINS_SENTENCE",
                    source_sentence=sentence,
                    source_sentence_id=sentence_id,
                    source_sentence_index=sentence_index,
                    source_note_id=graph_note_id,
                )
        note_entities: List[str] = []

        for entity_info in entities:
            entity_text = entity_info["text"].strip()
            original_label = entity_info["label"]

            if len(entity_text) < 3:
                continue
            if any(
                re.match(pattern, entity_text, re.IGNORECASE)
                for pattern in [
                    r"^\d+$",
                    r"^[A-Z]$",
                    r"^(the|and|or|of|to|in|for|on|at|by)$",
                ]
            ):
                continue

            standardized_label = "Other"

            if original_label in ["CANCER", "PATHOLOGICAL_FORMATION"]:
                standardized_label = "Condition"
            elif original_label in ["SIMPLE_CHEMICAL", "AMINO_ACID", "ORGANISM_SUBSTANCE"]:
                standardized_label = "Treatment"
            elif original_label in [
                "ANATOMICAL_SYSTEM",
                "ORGAN",
                "TISSUE",
                "CELL",
                "MULTI-TISSUE_STRUCTURE",
                "ORGANISM_SUBDIVISION",
                "DEVELOPING_ANATOMICAL_STRUCTURE",
                "IMMATERIAL_ANATOMICAL_ENTITY",
                "CELLULAR_COMPONENT",
            ]:
                standardized_label = "Anatomy"
            elif original_label in ["GENE_OR_GENE_PRODUCT"]:
                standardized_label = "GeneProtein"
            elif original_label in ["ORGANISM"]:
                standardized_label = "Organism"
            elif original_label in ["DISEASE"]:
                standardized_label = "Condition"
            elif original_label in ["CHEMICAL"]:
                standardized_label = "Treatment"
            elif original_label in ["GGP"]:
                standardized_label = "GeneProtein"
            elif original_label in ["SO"]:
                standardized_label = "GeneProtein"
            elif original_label in ["TAXON"]:
                standardized_label = "Organism"
            elif original_label in ["CHEBI"]:
                standardized_label = "Treatment"
            elif original_label in ["GO", "GO_BP", "GO_CC", "GO_MF"]:
                standardized_label = "Other"
            elif original_label in ["CL", "CELL_LINE", "CELL_TYPE"]:
                standardized_label = "Anatomy"
            elif original_label in ["DNA", "RNA", "PROTEIN"]:
                standardized_label = "GeneProtein"
            elif original_label in ["ENTITY"]:
                standardized_label = "Other"

            procedure_keywords = [
                "resection",
                "excision",
                "lumpectomy",
                "mastectomy",
                "dissection",
                "biopsy",
                "surgery",
                "operation",
            ]
            if any(kw in entity_text.lower() for kw in procedure_keywords):
                standardized_label = "Procedure"

            normalized_entity = normalize_entity_text(entity_text)
            if len(normalized_entity) < 3:
                continue

            source_sentence = None
            source_sentence_idx = None
            for idx, sent in enumerate(sent_texts):
                if entity_text.lower() in sent.lower():
                    source_sentence = sent
                    source_sentence_idx = idx
                    break

            add_or_get_node(
                graph,
                normalized_entity,
                label=standardized_label,
                original_label=original_label,
                original_text=entity_text,
                source_model=entity_info.get("source_model", "unknown"),
                cluster_size=entity_info.get("cluster_size", 1),
                is_negated=entity_info.get("is_negated", False),
                is_historical=entity_info.get("is_historical", False),
                is_family=entity_info.get("is_family", False),
                is_hypothetical=entity_info.get("is_hypothetical", False),
            )

            if source_sentence:
                sentence_id = None
                if source_sentence_idx is not None:
                    sentence_id = f"{graph_note_id}_sent_{source_sentence_idx}"
                graph.add_edge(
                    note_node,
                    normalized_entity,
                    relation="MENTIONS",
                    source_sentence=source_sentence,
                    source_sentence_id=sentence_id,
                    source_sentence_index=source_sentence_idx,
                    source_note_id=graph_note_id,
                )

            if resolved_note_date and graph.has_node(resolved_note_date):
                graph.add_edge(normalized_entity, resolved_note_date, relation="OCCURRED_ON")

            if standardized_label in ["Procedure", "Condition", "Anatomy"]:
                graph.add_edge(
                    patient_node,
                    normalized_entity,
                    relation=f"HAS_{standardized_label.upper()}",
                )

            note_entities.append(normalized_entity)

        for j in range(len(note_entities)):
            for k in range(j + 1, len(note_entities)):
                ent1, ent2 = note_entities[j], note_entities[k]
                if ent1 != ent2:
                    if graph.has_edge(ent1, ent2):
                        graph[ent1][ent2]["weight"] = (
                            graph[ent1][ent2].get("weight", 1) + 1
                        )
                    else:
                        graph.add_edge(ent1, ent2, relation="CO_OCCURS", weight=1)

    return graph


def build_patient_graph(
    records: List[NoteRecord],
    *,
    model_configs: List[dict],
    context_filters: dict | None = None,
    dedup_config: dict | None = None,
    include_report_sentences: bool = True,
) -> nx.Graph:
    """Build one patient's graph without re-inferring report metadata from text."""
    grouped = group_notes_by_patient(records)
    if len(grouped) != 1:
        raise ValueError("build_patient_graph requires notes for exactly one patient")
    patient_id, notes = next(iter(grouped.items()))
    filters = dict(context_filters) if context_filters is not None else {
        "allow_negated": True,
        "allow_hypothetical": False,
        "allow_family": True,
        "allow_historical": True,
    }
    deduplication = dict(dedup_config) if dedup_config is not None else {
        "enabled": True,
        "similarity_threshold": 0.85,
        "selection_strategy": "longest_from_best_model",
    }
    graphs = [
        process_notes_to_graph(
            [note.text], patient_id, note.path.name, model_configs,
            filters, deduplication, str(note.path),
            note_id=note.note_id, note_date=note.date,
            report_type=note.report_type, language=note.language,
            include_report_sentences=include_report_sentences,
        )
        for note in notes
    ]
    graph = nx.compose_all(graphs)
    graph.graph.update(
        patient_id=patient_id,
        note_count=len(notes),
        languages=sorted({note.language for note in notes}),
        includes_report_sentences=include_report_sentences,
    )
    return graph


__all__ = [
    "split_into_documents",
    "add_or_get_node",
    "process_notes_to_graph",
    "build_patient_graph",
]
