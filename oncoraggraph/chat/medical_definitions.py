"""Medical term definitions and ontology linking.

Inspired by MedGraphRAG: Provides medical term definitions and links
to credible sources (UMLS, medical dictionaries) for evidence-based responses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from ..utils.logging_utils import log


def load_feature_configs(feature_config_dir: Path) -> Dict[str, Dict]:
    """Load all feature configs to access medical definitions."""
    configs: Dict[str, Dict] = {}

    if not feature_config_dir.exists():
        return configs

    for config_file in feature_config_dir.glob("*.json"):
        try:
            with config_file.open() as f:
                config = json.load(f)
                feature_name = config.get("feature", {}).get("name", "")
                if feature_name:
                    configs[feature_name] = config
        except Exception as e:
            log(f"Failed to load config {config_file}: {e}", level="WARNING")

    return configs


def extract_medical_terms_from_question(question: str) -> List[str]:
    """Extract medical terms from question that might have definitions."""
    # Common medical terms that might be in feature configs
    medical_terms = []
    question_lower = question.lower()

    # Check for common medical concepts
    medical_concepts = [
        "hypertension", "diabetes", "chemotherapy", "surgery", "radiation",
        "cancer", "tumor", "diagnosis", "treatment", "medication", "drug",
        "bmi", "weight", "height", "blood pressure", "mastectomy", "lumpectomy",
    ]

    for concept in medical_concepts:
        if concept in question_lower:
            medical_terms.append(concept)

    return medical_terms


def get_medical_definition(
    term: str,
    feature_configs: Dict[str, Dict],
) -> Optional[Dict]:
    """Get medical definition for a term from feature configs.

    Returns dict with:
    - description: Medical definition
    - normalized_name: Standard medical name
    - synonyms: List of synonyms
    - ontology_mappings: UMLS/BioPortal mappings
    - clinical_context: Clinical context
    """
    term_lower = term.lower()

    # Search through feature configs
    for feature_name, config in feature_configs.items():
        enrichment = config.get("enrichment", {})

        # Check normalized name
        normalized_name = enrichment.get("normalized_name", "").lower()
        if normalized_name and (term_lower in normalized_name or normalized_name in term_lower):
            return {
                "term": term,
                "normalized_name": enrichment.get("normalized_name"),
                "description": enrichment.get("description"),
                "synonyms": enrichment.get("synonyms", []),
                "ontology_mappings": enrichment.get("ontology_mappings", {}),
                "clinical_context": enrichment.get("clinical_context"),
                "category": enrichment.get("category"),
            }

        # Check synonyms
        synonyms = enrichment.get("synonyms", [])
        for synonym in synonyms:
            if synonym and (term_lower in synonym.lower() or synonym.lower() in term_lower):
                return {
                    "term": term,
                    "normalized_name": enrichment.get("normalized_name"),
                    "description": enrichment.get("description"),
                    "synonyms": enrichment.get("synonyms", []),
                    "ontology_mappings": enrichment.get("ontology_mappings", {}),
                    "clinical_context": enrichment.get("clinical_context"),
                    "category": enrichment.get("category"),
                }

    return None


def get_ontology_citations(ontology_mappings: Dict) -> List[Dict[str, str]]:
    """Format ontology mappings as citations.

    Returns list of citation dicts with source information.
    """
    citations = []

    # UMLS citations
    umls_entries = ontology_mappings.get("umls", [])
    for entry in umls_entries[:3]:  # Limit to top 3
        if isinstance(entry, dict):
            citations.append({
                "source": "UMLS",
                "cui": entry.get("cui", ""),
                "name": entry.get("name", ""),
                "url": f"https://uts.nlm.nih.gov/uts/umls/concept/{entry.get('cui', '')}" if entry.get("cui") else None,
            })

    # BioPortal citations
    bioportal_entries = ontology_mappings.get("bioportal", [])
    for entry in bioportal_entries[:2]:  # Limit to top 2
        if isinstance(entry, dict):
            citations.append({
                "source": "BioPortal",
                "ontology": entry.get("ontology", ""),
                "id": entry.get("id", ""),
                "name": entry.get("name", ""),
            })

    return citations


def format_medical_definitions_for_response(
    terms: List[str],
    feature_configs: Dict[str, Dict],
) -> Dict[str, Dict]:
    """Get medical definitions for multiple terms.

    Returns dict mapping term -> definition info.
    """
    definitions = {}

    for term in terms:
        definition = get_medical_definition(term, feature_configs)
        if definition:
            definitions[term] = definition

    return definitions


__all__ = [
    "load_feature_configs",
    "extract_medical_terms_from_question",
    "get_medical_definition",
    "get_ontology_citations",
    "format_medical_definitions_for_response",
]
