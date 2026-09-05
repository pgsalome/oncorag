"""LLM querying and validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from ..utils.logging_utils import log
from ..utils.parsing_utils import parse_json_loose
from ..models import model_init
from .llm_adapter import get_llm_adapter


def query_llm_with_prompt(
    prompt: str,
    context: str,
    feature: str,
    pid: str,
    cache_dir: Path,
    raw_context: Optional[str] = None,
    retrieved_entities: Optional[list] = None,
    graph_stats: Optional[dict] = None,
    retrieval_info: Optional[dict] = None,
    reranking_details: Optional[dict] = None,
    config_info: Optional[dict] = None,
    timing_info: Optional[dict] = None,
) -> Dict:
    """Query LLM for extraction using configured adapter and return parsed JSON."""
    try:
        # Get the appropriate LLM adapter based on system configuration
        adapter = get_llm_adapter()
        
        # Query using the adapter (PHI removal handled automatically)
        llm = adapter.query(prompt, context)
        
        # Parse response if it's a string
        if isinstance(llm, str):
            llm = parse_json_loose(llm)

        log(
            f"LLM extraction complete (confidence: {llm.get('confidence', 'Unknown')})",
            level="SUCCESS",
        )
        return llm
    except Exception as exc:
        log(f"LLM error: {exc}", level="ERROR")
        return {
            "value": "error_during_extraction",
            "reasoning": str(exc),
            "confidence": "Low",
        }


def validate_extraction(result: Dict, config: Dict, context: str) -> Dict:
    """Validate extraction results for consistency using medspaCy."""
    log("Validating extraction result...", level="STEP")

    if result.get("evidence") and result["value"] not in [
        "missing",
        "error_during_extraction",
        "C",
    ]:
        model_init.initialize_models()
        doc = model_init.NLP_MED(result["evidence"])

        keywords = config["rules"]["keywords"]
        for ent in doc.ents:
            if any(keyword.lower() in ent.text.lower() for keyword in keywords):
                if getattr(ent._, "is_negated", False) and result["value"] == "A":
                    result["validation_warning"] = (
                        "Evidence contains negation but extraction is positive"
                    )
                    result["confidence"] = "Low"
                    log(
                        "Validation warning: Negation detected in positive result",
                        level="WARNING",
                    )

    log("Validation complete", level="SUCCESS")
    return result


__all__ = [
    "query_llm_with_prompt",
    "validate_extraction",
]
