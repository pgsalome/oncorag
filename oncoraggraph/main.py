import re
import argparse
import pickle
import json
from pathlib import Path
from typing import Optional, Dict, Tuple
import hashlib
import networkx as nx
import yaml
import os
from types import SimpleNamespace

GERMAN_LATERALITY_MAP = {
    'links': 'left', 'linksseitig': 'left',
    'rechts': 'right', 'rechtsseitig': 'right',
    'bilateral': 'bilateral', 'beidseitig': 'bilateral', 'beidseitige': 'bilateral',
    'zentral': 'central'
}


def map_german_laterality(text: str):
    low = text.lower() if isinstance(text, str) else ''
    for token, mapped in GERMAN_LATERALITY_MAP.items():
        if token in low:
            return mapped
    return None
from .utils.logging_utils import log, set_debug_mode, is_debug_mode
from .utils.parsing_utils import normalize_date_to_mmddyyyy, extract_date_from_note
from .utils.file_utils import save_prompt_to_cache, process_single_file
from .models.model_init import initialize_models
from .graph.graph_builder import split_into_documents, process_notes_to_graph
from .graph.graph_utils import get_entity_details_from_graph, get_clinical_entity_stats
from .vector_store.backend import get_vector_collection, index_graph_nodes
from .vector_store.config import load_vector_store_config, validate_vector_store_config
from .retrieval import multi_stage_graph_retrieval
from .llm.prompt_builder import (
    build_prompt,
    rerank_context,
    get_context_from_graph,
    get_context_from_graph_with_metadata,
)
from .llm.llm_query import query_llm_with_prompt, validate_extraction

PACKAGE_ROOT = Path(__file__).resolve().parent
SYSTEM_CONFIG_PATH = PACKAGE_ROOT / "system_config.yaml"


def _load_dataset_profile() -> str:
    env_profile = os.getenv("ONCORAGGRAPH_DATASET_PROFILE")
    if env_profile:
        return env_profile.strip().lower()
    if SYSTEM_CONFIG_PATH.exists():
        try:
            data = yaml.safe_load(SYSTEM_CONFIG_PATH.read_text()) or {}
            profile = data.get("dataset_profile")
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

DEFAULT_CONTEXT_FILTERS = {
    # Keep negated/family/historical evidence so the LLM can decide; drop hypothetical/planning.
    "allow_negated": True,
    "allow_hypothetical": False,
    "allow_family": True,
    "allow_historical": True,
}
DEFAULT_RERANK_TOP_K = 30
DEFAULT_GRAPH_SEARCH_DEPTH = 2
DEFAULT_DEDUP_CONFIG = {
    "enabled": True,
    "similarity_threshold": 0.85,
    "selection_strategy": "longest_from_best_model",
}
DEFAULT_SCISPACY_MODELS = [
    {"name": "en_ner_bionlp13cg_md", "priority": 1},
    {"name": "en_ner_bc5cdr_md", "priority": 2},
]

PLANNING_EVIDENCE_CUES = [
    "potential side effect",
    "potential side effects",
    "anticipated side effect",
    "anticipated side effects",
    "discussing potential",
    "discussed potential",
    "discussing anticipated",
    "education on side effects",
    "counseled about side effects",
    "anticipatory guidance",
]

# Cache heavy per-patient artifacts in memory so multiple feature extractions can reuse them.
IN_MEMORY_GRAPH_CACHE: Dict[str, Tuple[nx.Graph, dict]] = {}


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    return value not in {"0", "false", "off", "no"}


USE_ENRICHMENT_TERMS = _env_flag("ONCORAGGRAPH_USE_ENRICHMENT_TERMS", True)
EARLY_STOP_ON_VALUE = _env_flag("ONCORAGGRAPH_EARLY_STOP_ON_VALUE", False)


def _graph_cache_key(patient_dir: Path) -> str:
    try:
        return str(patient_dir.resolve())
    except Exception:
        return str(patient_dir)


def _dedupe_strings(values):
    seen = set()
    result = []
    for value in values or []:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _evidence_suggests_planning(text: Optional[str]) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(cue in lower for cue in PLANNING_EVIDENCE_CUES)


def _derive_output_format(feature_block: dict) -> dict | None:
    feature_type = str(feature_block.get("type", "")).lower()
    expected_range = feature_block.get("expected_range", "")
    if feature_type != "categorical" or not expected_range:
        return None
    values = [v.strip() for v in expected_range.split(",") if v.strip()]
    if not values:
        return None
    options = {}
    for idx, value in enumerate(values):
        key = chr(ord("A") + idx)
        options[key] = value
    missing_key = "C"
    while missing_key in options:
        missing_key = chr(ord(missing_key) + 1)
    options[missing_key] = "Missing"
    return {"type": "categorical", "options": options}


def normalize_feature_config(raw_config: dict) -> dict:
    """Bridge newer config schema to the runtime format."""
    feature_block = raw_config.get("feature", {}) or {}
    enrichment = raw_config.get("enrichment", {}) or {}

    canonical_name = (
        feature_block.get("name")
        or raw_config.get("feature_name")
        or feature_block.get("source_feature_name")
        or enrichment.get("normalized_name")
        or raw_config.get("name")
    )
    display_name = enrichment.get("normalized_name") or raw_config.get("display_name") or canonical_name

    raw_config["feature_name"] = canonical_name or display_name or ""
    raw_config["display_name"] = display_name or canonical_name or ""
    if enrichment.get("description"):
        raw_config["description"] = enrichment["description"]
    else:
        raw_config.setdefault("description", "")
    raw_config.setdefault("data_type", feature_block.get("type"))
    raw_config.setdefault(
        "medical_context",
        feature_block.get("medical_context") or enrichment.get("clinical_context") or raw_config.get("medical_context"),
    )

    # Ensure keywords exist
    keywords = []
    existing_keywords = raw_config.get("rules", {}).get("keywords", []) if raw_config.get("rules") else []
    keywords.extend(existing_keywords or [])
    keywords.extend(enrichment.get("synonyms", []))
    keywords.extend(enrichment.get("semantic_keywords", []))
    keywords.extend(enrichment.get("search_terms", []))
    keywords.extend(raw_config.get("common_queries", []))

    ontology = enrichment.get("ontology_mappings", {}) or {}
    for entries in ontology.values():
        for entry in entries:
            keywords.append(entry.get("name", ""))
            keywords.append(entry.get("search_term", ""))

    keywords = _dedupe_strings(keywords)
    if not keywords and raw_config["feature_name"]:
        keywords = [raw_config["feature_name"]]

    rules = raw_config.setdefault("rules", {})
    rules["keywords"] = keywords

    # Examples
    if not raw_config.get("examples"):
        examples = []
        for example in enrichment.get("ehr_examples", []):
            text = str(example).strip()
            if text:
                examples.append({"context": text, "output": "A"})
        if examples:
            raw_config["examples"] = examples

    # Output format
    if "output_format" not in raw_config or not raw_config["output_format"]:
        derived = _derive_output_format(feature_block)
        if derived:
            raw_config["output_format"] = derived
    else:
        options = raw_config["output_format"].get("options")
        if isinstance(options, dict) and "Missing" not in options.values():
            missing_key = "C"
            while missing_key in options:
                missing_key = chr(ord(missing_key) + 1)
            raw_config["output_format"]["options"][missing_key] = "Missing"

    return raw_config

# ---------------------------------------------------------------------------
# Pipeline configuration helpers
# ---------------------------------------------------------------------------
PIPELINE_CONFIG_PATH = Path(__file__).resolve().parent / "system_config.yaml"
if PIPELINE_CONFIG_PATH.exists():
    try:
        PIPELINE_CONFIG = yaml.safe_load(PIPELINE_CONFIG_PATH.read_text()) or {}
    except Exception:
        PIPELINE_CONFIG = {}
else:
    PIPELINE_CONFIG = {}

UNIT_CONVERSION = PIPELINE_CONFIG.get("unit_conversion", {})
TEMPORAL_POLICY = PIPELINE_CONFIG.get("temporal_policy", {})
TEMPORAL_INDICATORS = TEMPORAL_POLICY.get("indicators", {})
PAST_TERMS = [term.lower() for term in TEMPORAL_INDICATORS.get("past", [])]
PLANNED_TERMS = [term.lower() for term in TEMPORAL_INDICATORS.get("planned", [])]


def apply_temporal_filters(raw_context: str) -> str:
    """Remove sentences that match temporal policy exclusions."""
    if not raw_context.strip():
        return raw_context

    keep_sentences = []
    for sentence in [s for s in raw_context.split("\n") if s.strip()]:
        lower = sentence.lower()
        if TEMPORAL_POLICY.get("history_only") == "reject" and any(term in lower for term in PAST_TERMS):
            continue
        if TEMPORAL_POLICY.get("planned") == "reject" and any(term in lower for term in PLANNED_TERMS):
            continue
        keep_sentences.append(sentence)

    return "\n".join(keep_sentences)


def convert_units(feature_name: str, value):
    """Normalize units for numeric features when possible."""
    if not isinstance(value, str):
        return value

    text = value.strip()
    lower = text.lower()
    if not lower:
        return value

    feature_lower = feature_name.lower()

    def extract_number(s: str) -> Optional[float]:
        match = re.search(r"[-+]?\d*\.?\d+", s)
        if not match:
            return None
        try:
            return float(match.group())
        except ValueError:
            return None

    if feature_lower == "weight":
        lbs_factor = UNIT_CONVERSION.get("weight", {}).get("lbs_to_kg")
        if lbs_factor and ("lb" in lower or "pound" in lower):
            num = extract_number(lower)
            if num is not None:
                kg = num * lbs_factor
                return f"{kg:.1f} kg"
    if feature_lower == "height":
        cm_factor = UNIT_CONVERSION.get("height", {}).get("in_to_cm")
        if cm_factor:
            # Feet and inches pattern 5'6" etc
            match = re.match(r"\s*(\d)\s*'\s*(\d{1,2})", lower)
            if match:
                feet = float(match.group(1))
                inches = float(match.group(2))
                total_in = feet * 12 + inches
                cm = total_in * cm_factor
                return f"{cm:.1f} cm"
            if "inch" in lower or " in" in lower:
                num = extract_number(lower)
                if num is not None:
                    cm = num * cm_factor
                    return f"{cm:.1f} cm"
        if "cm" in lower:
            num = extract_number(lower)
            if num is not None:
                return f"{num:.1f} cm"
    if feature_lower == "body mass index":
        num = extract_number(lower)
        if num is not None:
            return f"{num:.1f} kg/m^2"
    return value


_DEFAULT_MISSING_STRINGS = {
    "missing",
    "not documented",
    "not provided",
    "not found",
    "unknown",
    "no data",
    "n/a",
}


def _is_missing_value(value, missing_label: Optional[str] = None) -> bool:
    """Return True if the LLM output should be treated as missing."""
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return True
        lower = stripped.lower()
        if missing_label and stripped == missing_label:
            return True
        if missing_label and lower == missing_label.lower():
            return True
        if lower in _DEFAULT_MISSING_STRINGS:
            return True
    return False


def _extract_dates_iso(text: str) -> list[tuple[str, str]]:
    """
    Extract date-like strings from text and return list of (iso_date, raw_match).
    Supports common formats: YYYY-MM-DD, DD.MM.YYYY, DD/MM/YYYY, DD-MM-YYYY, YYYY/MM/DD.
    """
    import datetime
    if not text:
        return []

    # Patterns: (regex, builder) where builder returns iso string or raises
    patterns = [
        # YYYY-MM-DD or YYYY/MM/DD
        (r"(?<!\d)(\d{4})[/-](\d{2})[/-](\d{2})", lambda g: f"{g[0]}-{g[1]}-{g[2]}"),
        # DD.MM.YYYY or DD/MM/YYYY or DD-MM-YYYY
        (r"(?<!\d)(\d{2})[./-](\d{2})[./-](\d{4})", lambda g: f"{g[2]}-{g[1]}-{g[0]}"),
        # YYYY.MM.DD
        (r"(?<!\d)(\d{4})[.](\d{2})[.](\d{2})", lambda g: f"{g[0]}-{g[1]}-{g[2]}"),
        # DD.MM.YY (assume 20YY)
        (r"(?<!\d)(\d{2})[.](\d{2})[.](\d{2})", lambda g: f"20{g[2]}-{g[1]}-{g[0]}"),
        # MM/YYYY or MM-YYYY -> assume day = 01
        (r"(?<!\d)(\d{2})[/-](\d{4})", lambda g: f"{g[1]}-{g[0]}-01"),
    ]

    results: list[tuple[str, str]] = []
    for regex, builder in patterns:
        for match in re.finditer(regex, text):
            raw = match.group(0)
            try:
                iso = builder(match.groups())
                dt = datetime.datetime.strptime(iso, "%Y-%m-%d")
                results.append((dt.strftime("%Y-%m-%d"), raw))
            except Exception:
                continue
    return results


def _resolve_missing_option(config: dict) -> Tuple[Optional[str], str]:
    """Return (option_key, label) that represents Missing for categorical features."""
    options = config.get("output_format", {}).get("options") if isinstance(config.get("output_format"), dict) else None
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


# --- Main ---

def run_feature_extraction(
    patient_dir: str | Path,
    feature: str,
    *,
    gt_value: Optional[str] = None,
    force_rebuild: bool = False,
    debug: bool = False,
    prompt_cache_dir: Optional[str | Path] = None,
    vector_store_config: Optional[dict] = None,
):
    import time  # Make sure this is imported at the top of the file

    vector_settings = (
        load_vector_store_config()
        if vector_store_config is None
        else validate_vector_store_config(vector_store_config)
    )
    vector_backend = vector_settings["backend"]
    args = SimpleNamespace(
        patient_dir=str(patient_dir),
        feature=feature,
        gt_value=gt_value,
        force_rebuild=force_rebuild,
        debug=debug,
    )

    set_debug_mode(args.debug)

    # Track timing
    timing_info = {
        "start_time": time.time(),
        "model_init_time": None,
        "graph_build_time": None,
        "graph_loaded_from_cache": False,
        "chromadb_index_time": None,
        "vector_index_time": None,
        "vector_backend": vector_backend,
        "retrieval_time": None,
        "reranking_time": None,
        "llm_query_time": None,
        "validation_time": None,
        "total_time": None
    }

    log("CLINICAL DATA EXTRACTION PIPELINE", level="HEADER")
    log(f"Feature: {args.feature}", level="INFO")
    log(f"Patient: {Path(args.patient_dir).name}", level="INFO")

    package_root = Path(__file__).resolve().parent

    # Load config
    log("LOADING CONFIGURATION", level="SUBHEADER")
    features_env = os.getenv("ONCORAGGRAPH_FEATURES_DIR")
    if features_env:
        config_root = Path(features_env)
    else:
        config_root = package_root / "config"
        if not config_root.is_dir():
            config_root = package_root / "feature_configs"

    config_path = config_root / f"{args.feature}.json"
    if not config_path.exists():
        log(f"Config file not found: {config_path}", level="ERROR")
        return
    
    with open(config_path) as f:
        config = normalize_feature_config(json.load(f))
    feature_label = config.get("display_name") or config.get("feature_name") or args.feature
    log(f"Config loaded: {feature_label}", level="SUCCESS")

    def _attach_display_name(record: dict) -> None:
        if isinstance(record, dict) and feature_label:
            record.setdefault("feature_normlise", feature_label)
    
    # Get model configurations
    model_configs = config.get("scispacy_models") or DEFAULT_SCISPACY_MODELS
    log(f"Models to use: {[m['name'] for m in model_configs]}", level="INFO")
    
    # Get deduplication config
    dedup_config = DEFAULT_DEDUP_CONFIG
    
    if dedup_config.get("enabled"):
        log(f"Entity deduplication enabled (threshold={dedup_config.get('similarity_threshold')})", level="INFO")
    else:
        log("Entity deduplication disabled", level="INFO")

    # Validate patient directory
    patient_dir = Path(args.patient_dir)
    if not patient_dir.is_dir():
        log(f"Patient directory not found: {patient_dir}", level="ERROR")
        return

    patient_id = patient_dir.name
    graph_cache_key = _graph_cache_key(patient_dir)

    # Set cache directories relative to package root
    graph_cache_dir = package_root / "graph_cache"
    prompt_cache_selection: Optional[str | Path] = prompt_cache_dir
    if prompt_cache_selection is None:
        prompt_cache_selection = os.getenv("ONCORAGGRAPH_PROMPT_CACHE_DIR")

    if prompt_cache_selection is not None:
        resolved_prompt_cache_dir = Path(prompt_cache_selection)
    else:
        prompt_cache_root = package_root / "prompt_cache"
        if DATASET_PROFILE == "mimic":
            prompt_cache_root = package_root / "prompt_cache_mimic"
        resolved_prompt_cache_dir = prompt_cache_root

    graph_cache_dir.mkdir(exist_ok=True)
    resolved_prompt_cache_dir.mkdir(exist_ok=True)
    prompt_cache_dir = resolved_prompt_cache_dir

    # Initialize models
    t_start = time.time()
    initialize_models()
    timing_info["model_init_time"] = time.time() - t_start

    # Build or load graph
    log("BUILDING CLINICAL KNOWLEDGE GRAPH", level="SUBHEADER")
    
    # Generate cache filename
    dir_hash = hashlib.md5(str(patient_dir.resolve()).encode()).hexdigest()
    cache_file = graph_cache_dir / f"{patient_id}_{dir_hash[:8]}.gpickle"
    
    graph = None
    stats = None
    context_filters = DEFAULT_CONTEXT_FILTERS.copy()

    if not args.force_rebuild:
        cached_entry = IN_MEMORY_GRAPH_CACHE.get(graph_cache_key)
        if cached_entry:
            graph, stats = cached_entry
            timing_info["graph_loaded_from_cache"] = True
            timing_info["graph_build_time"] = 0
            log("Using in-memory cached graph", level="INFO")

    # Try loading from cache first
    if graph is None and cache_file.exists() and not args.force_rebuild:
        log(f"Loading cached graph: {cache_file.name}", level="INFO")
        try:
            with open(cache_file, 'rb') as f:
                graph = pickle.load(f)
            stats = get_clinical_entity_stats(graph)
            timing_info["graph_loaded_from_cache"] = True
            timing_info["graph_build_time"] = 0  # No build time since loaded from cache
            log("Graph loaded from cache", level="SUCCESS")
            log(f"  • Total nodes: {stats.get('total_nodes', 0)}", level="INFO")
            log(f"  • Total edges: {stats.get('total_edges', 0)}", level="INFO")
            log(f"  • Clinical entities: {stats.get('clinical_entities', 0)}", level="INFO")
            if stats.get("error"):
                log(f"  • Warning: {stats['error']}", level="WARNING")
            IN_MEMORY_GRAPH_CACHE[graph_cache_key] = (graph, stats)
        except Exception as e:
            log(f"Error loading cache: {e}. Rebuilding...", level="WARNING")
            graph = None  # Force rebuild
    
    # Build graph if not loaded from cache
    if graph is None:
        log(f"Context filters: negated={context_filters.get('allow_negated')}, "
            f"hypothetical={context_filters.get('allow_hypothetical')}, "
            f"family={context_filters.get('allow_family')}, "
            f"historical={context_filters.get('allow_historical')}", level="INFO", debug=True)
        
        txt_files = sorted(
            [f for f in patient_dir.glob("*.txt") if f.is_file() and not f.name.startswith('.')]
        )
        log(f"Found {len(txt_files)} clinical note files", level="INFO")
        
        t_start = time.time()
        list_of_graphs = [
            process_single_file(
                f,
                patient_id,
                model_configs,
                context_filters,
                dedup_config,
                split_fn=split_into_documents,
                process_notes_fn=process_notes_to_graph,
            )
            for f in txt_files
        ]
        graph = nx.compose_all(list_of_graphs)
        timing_info["graph_build_time"] = time.time() - t_start
        
        stats = get_clinical_entity_stats(graph)
        log(f"Graph statistics:", level="SUCCESS")
        log(f"  • Total nodes: {stats['total_nodes']}", level="INFO")
        log(f"  • Total edges: {stats['total_edges']}", level="INFO")
        log(f"  • Clinical entities: {stats['clinical_entities']}", level="INFO")
        if is_debug_mode():
            log(f"  • Entity breakdown: {stats.get('nodes_by_type', {})}", level="INFO")
        
        # Cache graph
        with open(cache_file, 'wb') as f:
            pickle.dump(graph, f)
        log(f"Graph cached: {cache_file.name}", level="INFO", debug=True)
        IN_MEMORY_GRAPH_CACHE[graph_cache_key] = (graph, stats)

    # Index graph entities in the selected vector store.
    if stats is None:
        stats = get_clinical_entity_stats(graph)

    log(f"INDEXING IN {vector_backend.upper()}", level="SUBHEADER")
    collection = get_vector_collection(patient_id, vector_settings)
    
    # Get entity type filters from config
    t_start = time.time()
    if args.force_rebuild or collection.count() == 0:
        collection = index_graph_nodes(graph, collection, None, replace=args.force_rebuild)
    else:
        log(f"Using existing {vector_backend} index ({collection.count()} entities)", level="INFO")
    timing_info["vector_index_time"] = time.time() - t_start
    if vector_backend == "chroma":
        timing_info["chromadb_index_time"] = timing_info["vector_index_time"]

    # Retrieval
    feature_prompt_label = config.get("display_name") or config.get("feature_name") or args.feature
    log("RETRIEVAL & EXTRACTION", level="SUBHEADER")
    log(f"Patient/feature: {patient_id} | {feature_prompt_label}", level="INFO")

    keywords = [
        k.strip()
        for k in config.get("rules", {}).get("keywords", [config.get("feature_name", args.feature)])
        if str(k).strip()
    ]
    log(
        f"Base search keywords: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}",
        level="INFO",
    )

    enrichment_block = config.get("enrichment", {}) or {}
    synonyms = enrichment_block.get("synonyms", []) if isinstance(enrichment_block, dict) else []
    additional_terms: list[str] = []
    if USE_ENRICHMENT_TERMS and isinstance(enrichment_block, dict):
        semantic_keywords = enrichment_block.get("semantic_keywords", []) or []
        ontology_terms: list[str] = []
        for entries in (enrichment_block.get("ontology_mappings") or {}).values():
            for entry in entries:
                if isinstance(entry, dict):
                    name = entry.get("name")
                    search_term = entry.get("search_term")
                    if name:
                        ontology_terms.append(str(name))
                    if search_term:
                        ontology_terms.append(str(search_term))
        common_queries = config.get("common_queries") or []
        additional_terms = list(
            dict.fromkeys(
                term
                for term in (semantic_keywords + ontology_terms + list(common_queries))
                if isinstance(term, str)
            )
        )

    question_queries = config.get("rules", {}).get("questions") or []
    if not question_queries:
        question_queries = [
            f"What is the {feature_prompt_label}?",
            f"Extract the {feature_prompt_label} from the clinical notes.",
        ]
        if config.get("description"):
            question_queries.append(config["description"])
    question_queries = list(dict.fromkeys([q for q in question_queries if q]))
    retrieval_question = question_queries[0]

    log(f"Retrieval question: {retrieval_question}", level="INFO", debug=True)

    t_start = time.time()
    retrieval_result = multi_stage_graph_retrieval(
        retrieval_question,
        graph,
        collection,
        base_keywords=keywords,
        synonyms=synonyms,
        additional_terms=additional_terms if additional_terms else None,
        vector_top_k=max(DEFAULT_RERANK_TOP_K, 40),
        expansion_depth=DEFAULT_GRAPH_SEARCH_DEPTH,
    )

    vector_stage = next((stage for stage in retrieval_result.stages if stage.stage_name == "vector_retrieval"), None)
    stage_error = str(vector_stage.notes.get("error", "")).lower() if vector_stage and isinstance(vector_stage.notes, dict) else ""
    if vector_backend == "iris" and stage_error:
        raise RuntimeError("IRIS vector retrieval failed; check the database and embedding configuration")
    if vector_backend == "chroma" and "dimension" in stage_error:
        log(
            "Chroma index uses incompatible embeddings; rebuilding collection",
            level="WARNING",
        )
        try:
            chroma_client = getattr(collection, "_client", None)
            collection_name = getattr(collection, "name", None)
            if chroma_client and collection_name:
                chroma_client.delete_collection(name=collection_name)
                log("Deleted stale Chroma collection", level="INFO")
            collection = get_vector_collection(patient_id, vector_settings)
        except Exception as exc:
            log(f"Could not reset Chroma collection automatically: {exc}", level="ERROR")
        collection = index_graph_nodes(graph, collection, None, replace=True)
        retrieval_result = multi_stage_graph_retrieval(
            retrieval_question,
            graph,
            collection,
            base_keywords=keywords,
            synonyms=synonyms,
            additional_terms=additional_terms if additional_terms else None,
            vector_top_k=max(DEFAULT_RERANK_TOP_K, 40),
            expansion_depth=DEFAULT_GRAPH_SEARCH_DEPTH,
        )

    timing_info["retrieval_time"] = time.time() - t_start

    selected_nodes = retrieval_result.pruned_nodes or retrieval_result.start_nodes

    retrieval_info = {
        "pipeline": "multi_stage_v1",
        "vector_backend": vector_backend,
        "collection_namespace": vector_settings["collection_namespace"],
        "use_enrichment_terms": USE_ENRICHMENT_TERMS,
        "stages": [
            {
                "stage": stage.stage_name,
                "details": stage.notes,
            }
            for stage in retrieval_result.stages
        ],
        "seed_count": len(retrieval_result.start_nodes),
        "pruned_count": len(retrieval_result.pruned_nodes),
    }

    if not selected_nodes:
        log("No relevant entities found in patient data", level="WARNING")
        _, missing_label = _resolve_missing_option(config)
        result = {
            "feature": config['feature_name'],
            "value": missing_label,
            "reasoning": "No relevant entities found in the clinical notes.",
            "confidence": "High"
        }
        _attach_display_name(result)
        if args.gt_value is not None:
            result["gt_value"] = args.gt_value
        
        # Save minimal cache
        timing_info["total_time"] = time.time() - timing_info["start_time"]
        result["processing_time"] = round(timing_info["total_time"], 2)
        config_info = {
            "scispacy_models": config.get("scispacy_models", []),
            "entity_deduplication": config.get("entity_deduplication", {}),
            "context_filters": context_filters,
            "keywords": keywords,
            "graph_search_depth": DEFAULT_GRAPH_SEARCH_DEPTH,
            "rerank_top_k": DEFAULT_RERANK_TOP_K,
            "use_enrichment_terms": USE_ENRICHMENT_TERMS,
        }

        save_prompt_to_cache(
            "", "", config['feature_name'], patient_id, result, prompt_cache_dir,
            graph_stats=stats,
            retrieval_info=retrieval_info,
            config_info=config_info,
            timing_info=timing_info
        )
    else:
        # Get entity details for debugging
        retrieved_entities = get_entity_details_from_graph(graph, selected_nodes)
        
        if is_debug_mode():
            log("=" * 70, level="INFO")
            log("RETRIEVED ENTITIES:", level="INFO")
            log("=" * 70, level="INFO")
            for i, ent in enumerate(retrieved_entities[:10], 1):  # Show first 10
                print(f"{i}. {ent['original_text']:30} | {ent['label']:15} | Model: {ent['source_model']}")
            if len(retrieved_entities) > 10:
                print(f"... and {len(retrieved_entities) - 10} more entities")
            log("=" * 70, level="INFO")
        
        # Get context from graph
        raw_context, context_meta = get_context_from_graph_with_metadata(
            graph,
            selected_nodes,
            max_depth=DEFAULT_GRAPH_SEARCH_DEPTH,
        )

        raw_context = apply_temporal_filters(raw_context)

        if not raw_context.strip():
            log("No context sentences retrieved from graph", level="WARNING")
            _, missing_label = _resolve_missing_option(config)
            result = {
                "feature": config['feature_name'],
                "value": missing_label,
                "reasoning": "Entities found but no context sentences available.",
                "confidence": "Medium"
            }
            _attach_display_name(result)
            if args.gt_value is not None:
                result["gt_value"] = args.gt_value

            result["value"] = convert_units(config['feature_name'], result.get("value"))
            
            # Save cache
            timing_info["total_time"] = time.time() - timing_info["start_time"]
            result["processing_time"] = round(timing_info["total_time"], 2)
            config_info = {
                "scispacy_models": config.get("scispacy_models", []),
                "entity_deduplication": config.get("entity_deduplication", {}),
                "context_filters": context_filters,
                "keywords": keywords,
                "graph_search_depth": DEFAULT_GRAPH_SEARCH_DEPTH,
                "rerank_top_k": DEFAULT_RERANK_TOP_K,
                "use_enrichment_terms": USE_ENRICHMENT_TERMS,
            }
            
            save_prompt_to_cache(
                "", "", config['feature_name'], patient_id, result, prompt_cache_dir,
                retrieved_entities=retrieved_entities,
                graph_stats=stats,
                retrieval_info=retrieval_info,
                config_info=config_info,
                timing_info=timing_info
            )
        else:
            # Prepare keyword boosts (no longer embedded in the rerank query itself)
            keywords = [
                k.strip()
                for k in config.get("rules", {}).get("keywords", [config.get("feature_name", args.feature)])
                if str(k).strip()
            ]
            extra_terms = []
            fname_lower = str(config.get("feature_name", "")).lower()
            if "surgery" in fname_lower:
                extra_terms = [
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
                    "operative",
                    "operative note",
                    "was performed",
                    "date of surgery",
                    "lumpectomy/snb",
                ]
            boost_terms = [term for term in keywords + extra_terms if term]
            anchor_tokens = [
                token
                for token in re.split(r"[^a-z0-9]+", str(config.get("feature_name", "")).lower())
                if len(token) >= 4
            ]
            if anchor_tokens:
                boost_terms = [
                    term for term in boost_terms if any(anchor in term.lower() for anchor in anchor_tokens)
                ]
            boost_terms = list(dict.fromkeys(boost_terms))

            iter_question_queries = question_queries

            # Tracking containers
            timing_info["reranking_time"] = 0.0
            timing_info["llm_query_time"] = 0.0
            timing_info["validation_time"] = 0.0
            per_query_entries = []

            options = None
            missing_option_key = None
            missing_label_default = "Missing"
            if config.get("data_type") in ["boolean", "categorical", "ordinal"]:
                output_format = config.get("output_format") or {}
                if isinstance(output_format, dict):
                    options = output_format.get("options")
                    if isinstance(options, dict):
                        missing_option_key, missing_label_default = _resolve_missing_option(config)

            total_queries = len(iter_question_queries)
            for query_idx, question in enumerate(iter_question_queries, start=1):
                log(
                    f"Extraction target: patient={patient_id} | feature={feature_prompt_label} | query={query_idx}/{total_queries}",
                    level="INFO",
                )
                entry = {"question": question}
                t_rerank = time.time()
                # Pass config data for intelligent abbreviation normalization
                enrichment = config.get("enrichment", {}) or {}
                normalized_name = enrichment.get("normalized_name") or config.get("feature_name", "")
                synonyms = enrichment.get("synonyms", [])
                reranked_context, score, reranking_details = rerank_context(
                    raw_context,
                    question,
                    top_k=DEFAULT_RERANK_TOP_K,
                    keywords=boost_terms,
                    normalized_name=normalized_name,
                    synonyms=synonyms,
                    sentence_meta=context_meta,
                )
                timing_info["reranking_time"] += time.time() - t_rerank
                entry["retrieval_score"] = round(score, 4)
                entry["reranking_details"] = reranking_details
                entry["context"] = reranked_context
                sentences_used = len([s for s in reranked_context.split("\n") if s.strip()])

                # Fast path for date features: parse dates without LLM
                if str(config.get("data_type", "")).lower() == "date":
                    found = _extract_dates_iso(reranked_context) or _extract_dates_iso(raw_context)
                    if found:
                        iso_val, raw_match = found[0]
                        entry_result = {
                            "feature": config["feature_name"],
                            "value": iso_val,
                            "reasoning": f"Parsed date from context: '{raw_match}'",
                            "confidence": "High",
                            "evidence": raw_match,
                            "retrieval_score": entry["retrieval_score"],
                            "extraction_source": "regex",
                        }
                    else:
                        iso_val = missing_label_default
                        entry_result = {
                            "feature": config["feature_name"],
                            "value": iso_val,
                            "reasoning": "No date-like string found in context.",
                            "confidence": "Low",
                            "evidence": "",
                            "retrieval_score": entry["retrieval_score"],
                            "extraction_source": "regex",
                        }
                    entry_result["sentences_used"] = sentences_used
                    _attach_display_name(entry_result)
                    log(
                        f"Extracted value: {entry_result.get('value')!r} (source={entry_result.get('extraction_source', 'unknown')})",
                        level="INFO",
                    )
                    entry["result"] = entry_result
                    per_query_entries.append(entry)
                    break

                if str(config.get("feature_name", "")).lower() == "radiotherapy performed" and reranked_context.strip():
                    rad_terms = [
                        "radiation",
                        "radiotherapy",
                        "radiation therapy",
                        "radiation treatment",
                        "imrt",
                        "protons",
                        "brachytherapy",
                        "cgy",
                        "gray",
                        "radiation oncology",
                        "post-radiation",
                        "radiation dermatitis",
                        "radiation completed",
                        "irradiation",
                    ]
                    raw_sentences = [s for s in raw_context.split("\n") if s.strip()]
                    radiation_sentences = [
                        s for s in raw_sentences if any(term in s.lower() for term in rad_terms)
                    ]
                    if radiation_sentences:
                        max_extra = max(5, min(20, len(radiation_sentences)))
                        extra_snippets = radiation_sentences[:max_extra]
                        existing = reranked_context.split("\n") if reranked_context else []
                        combined = extra_snippets + [line for line in existing if line not in extra_snippets]
                        reranked_context = "\n".join(combined)
                        entry["context"] = reranked_context

                if not reranked_context.strip():
                    log(f"No context survived re-ranking for question: '{question}'", level="WARNING")
                    entry_result = {
                        "feature": config["feature_name"],
                        "value": missing_label_default,
                        "reasoning": "Context found but deemed irrelevant after re-ranking.",
                        "confidence": "Medium",
                        "retrieval_score": entry["retrieval_score"],
                        "extraction_source": "llm",
                    }
                    if args.gt_value is not None:
                        entry_result["gt_value"] = args.gt_value
                    log(
                        f"Extracted value: {entry_result.get('value')!r} (source={entry_result.get('extraction_source', 'unknown')})",
                        level="INFO",
                    )
                    entry["result"] = entry_result
                    per_query_entries.append(entry)
                    continue

                config_info = {
                    "scispacy_models": config.get("scispacy_models", []),
                    "entity_deduplication": config.get("entity_deduplication", {}),
                    "context_filters": context_filters,
                    "keywords": boost_terms,
                    "graph_search_depth": DEFAULT_GRAPH_SEARCH_DEPTH,
                    "rerank_top_k": DEFAULT_RERANK_TOP_K,
                    "use_enrichment_terms": USE_ENRICHMENT_TERMS,
                }
                entry["config_info"] = config_info

                planning_guard_enabled = False
                feature_name_lower = str(config.get("feature_name", "")).lower()
                description_lower = str(config.get("description", "")).lower()
                if "toxicity" in feature_name_lower or "side effect" in feature_name_lower:
                    planning_guard_enabled = True
                if "toxicity" in description_lower or "side effect" in description_lower:
                    planning_guard_enabled = True

                top_sentences = reranking_details.get("top_sentences") or [
                    s for s in reranked_context.split("\n") if s.strip()
                ]
                if not top_sentences:
                    top_sentences = [reranked_context]
                chunk_size = max(
                    1,
                    int(os.getenv("ONCORAGGRAPH_SENTENCE_CHUNK_SIZE", "5") or 5),
                )
                sentences_used = 0
                total_sentences = len(top_sentences)
                entry_result = None

                if is_debug_mode():
                    log("=" * 70, level="INFO")
                    log(f"QUESTION: {question}", level="INFO")
                    log("=" * 70, level="INFO")
                    log("RERANKED CONTEXT (sent to LLM):", level="INFO")
                    log("=" * 70, level="INFO")
                    print(reranked_context)
                    log("=" * 70, level="INFO")

                while sentences_used < total_sentences:
                    sentences_used = min(total_sentences, sentences_used + chunk_size)
                    chunk_context = "\n".join(top_sentences[:sentences_used])
                    entry["context"] = chunk_context
                    prompt = build_prompt(config, chunk_context)
                    entry["prompt"] = prompt

                    t_llm_start = time.time()
                    llm_response = query_llm_with_prompt(
                        prompt,
                        chunk_context,
                        config["feature_name"],
                        patient_id,
                        prompt_cache_dir,
                        raw_context=raw_context,
                        retrieved_entities=retrieved_entities,
                        graph_stats=stats,
                        retrieval_info=retrieval_info,
                        reranking_details=reranking_details,
                        config_info=config_info,
                        timing_info=None,
                    )
                    timing_info["llm_query_time"] += time.time() - t_llm_start
                    entry["llm_response"] = llm_response

                    val = llm_response.get("value")
                    log(
                        f"LLM raw value: {val!r} (confidence: {llm_response.get('confidence', 'Unknown')})",
                        level="INFO",
                    )
                    if config.get("data_type") in ["boolean", "categorical", "ordinal"] and not isinstance(options, dict):
                        log(
                            f"No categorical options defined for feature '{feature_label}'. Treating response as free-text.",
                            level="WARNING",
                        )

                    if config.get("data_type") in ["boolean", "categorical", "ordinal"] and isinstance(options, dict):
                        raw_val = val if isinstance(val, str) else ""
                        norm = raw_val.strip()
                        import re as _re

                        m = _re.match(r"^\s*([A-Za-z])\b", norm)
                        candidate_key = m.group(1).upper() if m else None
                        if candidate_key and candidate_key in options:
                            normalized_key = candidate_key
                        elif norm.upper() in options:
                            normalized_key = norm.upper()
                        else:
                            rev = {str(v).strip().lower(): k for k, v in options.items()}
                            normalized_key = rev.get(norm.lower())
                            if normalized_key is None:
                                # Heuristic recovery for malformed outputs (e.g., "one of: A, B, C")
                                # Try to find any option key in the raw value.
                                key_candidates = _re.findall(r"\b([A-Z])\b", norm.upper())
                                for key in key_candidates:
                                    if key in options:
                                        normalized_key = key
                                        break
                            if normalized_key is None:
                                # Map common textual variants to option keys when available.
                                treatment_status_aliases = {
                                    "actual": "actual_treatment_received",
                                    "actual_treatment": "actual_treatment_received",
                                    "actual_treatment_received": "actual_treatment_received",
                                    "received": "actual_treatment_received",
                                    "administered": "actual_treatment_received",
                                    "given": "actual_treatment_received",
                                    "completed": "actual_treatment_received",
                                    "active": "actual_treatment_received",
                                    "historical": "actual_treatment_received",
                                    "planned": "treatment_planned",
                                    "plan": "treatment_planned",
                                    "recommended": "treatment_planned",
                                    "indicated": "treatment_planned",
                                    "scheduled": "treatment_planned",
                                    "discussed": "treatment_discussed",
                                    "discussion": "treatment_discussed",
                                    "counseling": "treatment_discussed",
                                    "counselling": "treatment_discussed",
                                    "held": "treatment_held_or_stopped",
                                    "stopped": "treatment_held_or_stopped",
                                    "discontinued": "treatment_held_or_stopped",
                                    "interrupted": "treatment_held_or_stopped",
                                    "delayed": "treatment_held_or_stopped",
                                    "dose_reduced": "treatment_held_or_stopped",
                                    "supportive": "supportive_medication",
                                    "supportive_medication": "supportive_medication",
                                    "supportive_care": "supportive_medication",
                                }
                                normalized_text = norm.lower().replace("-", "_").replace(" ", "_")
                                for alias, label in treatment_status_aliases.items():
                                    if alias in normalized_text and label.lower() in rev:
                                        normalized_key = rev[label.lower()]
                                        break
                            if normalized_key is None:
                                if "yes" in norm.lower() and "B" in options:
                                    normalized_key = "B"
                                elif "no" in norm.lower() and "A" in options:
                                    normalized_key = "A"
                                elif "missing" in norm.lower() and "C" in options:
                                    normalized_key = "C"
                        mapped_val = options.get(normalized_key, "invalid_output")

                        if mapped_val == "invalid_output":
                            log(f"LLM returned invalid option: {val}", level="WARNING")
                            entry_result = {
                                "feature": config["feature_name"],
                                "value": missing_label_default,
                                "reasoning": f"LLM returned invalid option '{val}'. Options are: {list(options.keys())}",
                                "confidence": "Low",
                                "retrieval_score": entry["retrieval_score"],
                                "raw_llm_response": llm_response,
                                "extraction_source": "llm",
                            }
                        else:
                            entry_result = {
                                "feature": config["feature_name"],
                                "value": mapped_val,
                                "reasoning": llm_response.get("reasoning", ""),
                                "confidence": llm_response.get("confidence", "Unknown"),
                                "evidence": llm_response.get("evidence", ""),
                                "retrieval_score": entry["retrieval_score"],
                                "extraction_source": "llm",
                            }
                    else:
                        out_val = val
                        try:
                            is_date = (
                                str(config.get("data_type", "")).lower() == "date"
                                or str(config.get("output_format", {}).get("type", "")).lower() == "date"
                            )
                            if is_date and isinstance(out_val, str):
                                out_val = normalize_date_to_mmddyyyy(out_val)
                        except Exception:
                            pass
                        entry_result = {
                            "feature": config["feature_name"],
                            "value": out_val,
                            "reasoning": llm_response.get("reasoning", ""),
                            "confidence": llm_response.get("confidence", "Unknown"),
                            "evidence": llm_response.get("evidence", ""),
                            "retrieval_score": entry["retrieval_score"],
                            "extraction_source": "llm",
                        }

                    entry_result["sentences_used"] = sentences_used
                    _attach_display_name(entry_result)
                    if planning_guard_enabled:
                        evidence_text = llm_response.get("evidence") or entry_result.get("evidence", "")
                        reasoning_text = entry_result.get("reasoning", "")
                        if _evidence_suggests_planning(evidence_text) or _evidence_suggests_planning(reasoning_text):
                            entry_result["value"] = missing_label_default
                            entry_result["confidence"] = "Low"
                            entry_result["reasoning"] = (
                                (reasoning_text or "").strip()
                                + (" | " if reasoning_text else "")
                                + "Evidence references anticipated/educational side effects only."
                            ).strip(" |")
                            if evidence_text:
                                entry_result["evidence"] = evidence_text
                    # German laterality mapping fallback
                    if os.getenv("ONCORAGGRAPH_LANGUAGE", "").lower().startswith("ger") and isinstance(entry_result.get("value"), str):
                        mapped = map_german_laterality(entry_result.get("evidence", "") + " " + entry_result.get("reasoning", ""))
                        if mapped and mapped.lower() in (v.lower() for v in (options or {}).values()):
                            entry_result["value"] = mapped
                    log(
                        f"Extracted value: {entry_result.get('value')!r} (confidence: {entry_result.get('confidence', 'Unknown')})",
                        level="INFO",
                    )
                    entry["result"] = entry_result

                    if not _is_missing_value(entry_result.get("value"), missing_label_default):
                        break
                    if sentences_used >= total_sentences:
                        break
                    log(
                        f"No definitive value found yet; expanding context to {min(total_sentences, sentences_used + chunk_size)} sentences.",
                        level="INFO",
                        debug=True,
                    )

                    if str(config.get("feature_name", "")).lower() == "surgery date":
                        fallback_values = {
                            "",
                            "n/a",
                            "na",
                            "not available",
                            "not available in context",
                            "not provided",
                            "not provided in context",
                            "unknown",
                            "unknown - date not specified in context",
                        }
                        current_value = str(entry_result.get("value") or "").strip().lower()
                        if current_value in fallback_values:
                            fallback_raw = extract_date_from_note(raw_context)
                            if fallback_raw:
                                normalized = normalize_date_to_mmddyyyy(fallback_raw)
                                if normalized:
                                    log(
                                        f"Fallback regex extracted surgery date {normalized} for patient {patient_id}",
                                        level="INFO",
                                    )
                                    augmented_context = f"{fallback_raw}\n{entry['context']}"
                                    augmented_prompt = build_prompt(config, augmented_context)
                                    t_aug = time.time()
                                    augmented_response = query_llm_with_prompt(
                                        augmented_prompt,
                                        augmented_context,
                                        config["feature_name"],
                                        patient_id,
                                        prompt_cache_dir,
                                        raw_context=raw_context,
                                        retrieved_entities=retrieved_entities,
                                        graph_stats=stats,
                                        retrieval_info=retrieval_info,
                                        reranking_details=reranking_details,
                                        config_info=config_info,
                                        timing_info=None,
                                    )
                                    timing_info["llm_query_time"] += time.time() - t_aug

                                    aug_value = augmented_response.get("value", "")
                                    aug_reason = augmented_response.get("reasoning", "")
                                    aug_conf = augmented_response.get("confidence", "Unknown")
                                    aug_evidence = augmented_response.get("evidence", "")

                                    if aug_value:
                                        aug_norm = (
                                            normalize_date_to_mmddyyyy(aug_value)
                                            if isinstance(aug_value, str)
                                            else aug_value
                                        )
                                        if isinstance(aug_norm, str) and aug_norm:
                                            entry_result.update(
                                                {
                                                    "value": aug_norm,
                                                    "reasoning": aug_reason
                                                    or f"Date inferred from note header ('{fallback_raw}').",
                                                    "confidence": aug_conf,
                                                    "evidence": aug_evidence or fallback_raw,
                                                    "fallback_method": "regex_header",
                                                    "extraction_source": "regex_header_llm",
                                                }
                                            )
                                        else:
                                            entry_result.update(
                                                {
                                                    "value": normalized,
                                                    "reasoning": f"No explicit surgery date found by LLM; inferred from note header ('{fallback_raw}').",
                                                    "confidence": "Medium",
                                                    "evidence": fallback_raw,
                                                    "fallback_method": "regex_header",
                                                    "extraction_source": "regex_header",
                                                }
                                            )
                                    else:
                                        entry_result.update(
                                            {
                                                "value": normalized,
                                                "reasoning": f"No explicit surgery date found by LLM; inferred from note header ('{fallback_raw}').",
                                                "confidence": "Medium",
                                                "evidence": fallback_raw,
                                                "fallback_method": "regex_header",
                                                "extraction_source": "regex_header",
                                            }
                                        )
                                else:
                                    log("Fallback regex found date but normalization failed", level="WARNING")
                            else:
                                log("Fallback regex could not find surgery date", level="WARNING")
                                entry_result.setdefault("fallback_method", "regex_header_attempted")

                entry_result["value"] = convert_units(config["feature_name"], entry_result.get("value"))
                if args.gt_value is not None:
                    entry_result["gt_value"] = args.gt_value

                t_val = time.time()
                entry_result = validate_extraction(entry_result, config, reranked_context)
                if args.gt_value is not None:
                    entry_result.setdefault("gt_value", args.gt_value)
                timing_info["validation_time"] += time.time() - t_val

                validation_info = {}
                if entry_result.get("validation_warning"):
                    validation_info["warning"] = entry_result["validation_warning"]
                    validation_info["confidence_before"] = llm_response.get("confidence")
                    validation_info["confidence_after"] = entry_result.get("confidence")
                entry["validation_info"] = validation_info if validation_info else None
                entry["result"] = entry_result
                per_query_entries.append(entry)
                if EARLY_STOP_ON_VALUE and not _is_missing_value(entry_result.get("value"), missing_label_default):
                    log(
                        "Definitive value found; skipping remaining query variants for this feature.",
                        level="INFO",
                        debug=True,
                    )
                    break

            if not per_query_entries:
                result = {
                    "feature": config["feature_name"],
                    "value": missing_label_default,
                    "reasoning": "No valid context extracted for any query.",
                    "confidence": "Low",
                    "retrieval_score": 0.0,
                    "extraction_source": "llm",
                }
                _attach_display_name(result)
                if args.gt_value is not None:
                    result["gt_value"] = args.gt_value
                
                # Include raw context result if available
                per_query_results_list = []
                if raw_context_result:
                    per_query_results_list.append(raw_context_result)
                result["per_query_results"] = per_query_results_list
                
                selected_entry = {"prompt": "", "context": "", "reranking_details": {}, "validation_info": None}
            else:
                missing_tokens = {
                    "",
                    "missing",
                    "error_during_extraction",
                    "n/a",
                    "not available",
                    "unknown",
                }
                selected_entry = per_query_entries[0]
                for entry in per_query_entries:
                    value = entry["result"].get("value")
                    value_str = str(value).strip().lower() if isinstance(value, str) else value
                    if value is None or (isinstance(value_str, str) and value_str in missing_tokens):
                        continue
                    selected_entry = entry
                    break

                result = selected_entry["result"]
                per_query_results_list = [
                    {
                        "question": entry["question"],
                        "value": entry["result"].get("value"),
                        "confidence": entry["result"].get("confidence"),
                        "retrieval_score": entry["result"].get("retrieval_score"),
                        "reranked_context": entry.get("context", ""),
                        "prompt": entry.get("prompt", ""),
                        "reranking_details": entry.get("reranking_details", {}),
                    }
                    for entry in per_query_entries
                ]
                result["per_query_results"] = per_query_results_list
            log(
                f"Selected extraction: patient={patient_id} | feature={feature_prompt_label} | value={result.get('value')!r} | confidence={result.get('confidence', 'Unknown')}",
                level="INFO",
            )

            timing_info["total_time"] = time.time() - timing_info["start_time"]
            result["processing_time"] = round(timing_info["total_time"], 2)
            result["timing_breakdown"] = {
                key: round(value, 4)
                for key, value in timing_info.items()
                if isinstance(value, (int, float)) and value is not None
            }

            save_prompt_to_cache(
                selected_entry.get("prompt", ""),
                selected_entry.get("context", ""),
                config["feature_name"],
                patient_id,
                result,
                prompt_cache_dir,
                raw_context=raw_context,
                retrieved_entities=retrieved_entities,
                graph_stats=stats,
                retrieval_info=retrieval_info,
                reranking_details=selected_entry.get("reranking_details"),
                config_info=selected_entry.get(
                    "config_info",
                    {
                        "scispacy_models": config.get("scispacy_models", []),
                        "entity_deduplication": config.get("entity_deduplication", {}),
                        "context_filters": context_filters,
                        "keywords": boost_terms,
                        "graph_search_depth": DEFAULT_GRAPH_SEARCH_DEPTH,
                        "rerank_top_k": DEFAULT_RERANK_TOP_K,
                        "use_enrichment_terms": USE_ENRICHMENT_TERMS,
                    },
                ),
                timing_info=timing_info,
                validation_info=selected_entry.get("validation_info"),
            )

    # Final output
    log("EXTRACTION COMPLETE", level="SUBHEADER")
    print("\n" + "=" * 70)
    print("FINAL EXTRACTED RESULT")
    print("=" * 70)
    print(json.dumps(result, indent=2))
    print("=" * 70 + "\n")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="GraphRAG clinical extractor with multi-model extraction + embedding deduplication"
    )
    parser.add_argument("patient_dir", type=str, help="Directory with patient .txt files")
    parser.add_argument("feature", type=str, help="Feature to extract (must match a .json config file)")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild of graph and vector index")
    parser.add_argument("--vector-backend", choices=["chroma", "iris"], help="Override the vector backend")
    parser.add_argument("--vector-store-config", help="JSON/YAML file with vector_store settings")
    parser.add_argument("--cache-dir", help="Directory for extraction prompt caches")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debugging output")
    parser.add_argument("--gt-value", type=str, default=None, help="Ground truth value for this feature/patient")
    args = parser.parse_args()
    try:
        vector_settings = load_vector_store_config(args.vector_store_config, backend=args.vector_backend)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))

    return run_feature_extraction(
        patient_dir=args.patient_dir,
        feature=args.feature,
        gt_value=args.gt_value,
        force_rebuild=args.force_rebuild,
        debug=args.debug,
        prompt_cache_dir=args.cache_dir,
        vector_store_config=vector_settings,
    )


if __name__ == "__main__":
    main()
