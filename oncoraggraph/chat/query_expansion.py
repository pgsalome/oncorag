"""Medical query expansion for chatbot QA.

Extracts medical terms from questions and expands them with synonyms,
abbreviations, and related clinical terminology.

Inspired by MedGraphRAG paper (https://arxiv.org/html/2408.04187v1):
- Medical tag-based query structuring
- Hierarchical knowledge linking
- Enhanced ontology integration
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

from .term_matching import fuzzy_match_phrases, normalize_question_variants

# Medical terminology mappings for common clinical concepts
MEDICAL_TERM_EXPANSIONS: Dict[str, List[str]] = {
    # Chemotherapy
    "chemotherapy": [
        "chemo",
        "chemotherapeutic",
        "antineoplastic",
        "cancer treatment",
        "oncology treatment",
        "cytotoxic therapy",
        "chemotherapy regimen",
        "chemotherapy treatment",
    ],
    "chemo": [
        "chemotherapy",
        "chemotherapeutic",
        "antineoplastic",
        "cancer treatment",
    ],
    # Common chemotherapy drugs
    "carboplatin": ["paraplatin", "carbo", "carboplatinum"],
    "paclitaxel": ["taxol", "abraxane"],
    "doxorubicin": ["adriamycin", "dox"],
    "cyclophosphamide": ["cytoxan", "neosar"],
    "docetaxel": ["taxotere"],
    # Treatment concepts
    "treatment": ["therapy", "therapeutic", "intervention", "management"],
    "medication": ["drug", "medicine", "pharmaceutical", "med", "meds"],
    "drug": ["medication", "medicine", "pharmaceutical", "agent"],
    # Response / pathology
    "pathologic complete response": [
        "pathological complete response",
        "complete pathologic response",
        "complete pathological response",
        "pcr",
        "no residual disease",
        "no residual carcinoma",
        "no residual invasive carcinoma",
        "no residual invasive cancer",
        "no residual tumor",
        "no residual tumour",
        "rcb 0",
        "rcb=0",
        "rcb-0",
        "rcb class 0",
        "ypt0",
        "ypn0",
        "ypt0n0",
        "yptis",
        "yptisn0",
    ],
    "pcr": [
        "pathologic complete response",
        "pathological complete response",
        "complete pathologic response",
        "complete pathological response",
    ],
    "residual disease": [
        "residual carcinoma",
        "residual tumor",
        "residual tumour",
        "residual invasive carcinoma",
        "residual invasive cancer",
        "residual cancer burden",
        "rcb",
    ],
    "residual cancer burden": [
        "rcb",
        "residual cancer burden",
        "residual cancer burden score",
        "residual cancer burden class",
    ],
    "pathology": [
        "pathology report",
        "surgical pathology",
        "surgical pathology report",
        "final pathology",
        "specimen",
        "surgical specimen",
    ],
    "neoadjuvant": [
        "preoperative",
        "pre-op",
        "preop",
        "induction",
        "neoadj",
    ],
    "adjuvant": [
        "postoperative",
        "post-op",
        "postop",
    ],
    "toxicity": [
        "adverse event",
        "adverse events",
        "side effect",
        "side effects",
        "complication",
        "treatment-related",
        "treatment held",
        "treatment delayed",
        "treatment discontinued",
        "dose reduced",
        "dose reduction",
        "grade 3",
        "grade 4",
        "severe",
        "significant toxicity",
        "hospitalized",
        "hospitalised",
    ],
    "neutropenia": [
        "neutropenic",
        "febrile neutropenia",
        "anc",
        "low neutrophils",
        "low neutrophil count",
    ],
    "febrile neutropenia": [
        "neutropenic fever",
        "fever with neutropenia",
    ],
    # Surgery
    "surgery": [
        "surgical",
        "operation",
        "procedure",
        "surgical procedure",
        "operative",
    ],
    "mastectomy": ["mastectomy", "breast removal", "breast surgery"],
    # Diagnosis
    "diagnosis": ["diagnosed", "diagnostic", "dx", "diagnosed with"],
    "stage": [
        "staging",
        "clinical stage",
        "pathologic stage",
        "ajcc",
    ],
    "cancer": [
        "malignancy",
        "malignant",
        "neoplasm",
        "tumor",
        "tumour",
        "carcinoma",
        "adenocarcinoma",
    ],
    "tumor": ["tumour", "tumor", "mass", "lesion", "neoplasm"],
    # Conditions
    "hypertension": ["high blood pressure", "htn", "hbp", "elevated blood pressure"],
    "blood pressure": [
        "bp",
        "systolic",
        "diastolic",
        "blood presuure",
        "blood presure",
        "blood presssure",
    ],
    "diabetes": ["diabetes mellitus", "dm", "diabetic"],
    # Procedures
    "biopsy": ["biopsy", "tissue sample", "sampling"],
    "radiation": [
        "radiotherapy",
        "radiation therapy",
        "rt",
        "xrt",
        "irradiation",
        "rad",
    ],
    # Allergies
    "allergy": [
        "allergies",
        "allergic",
        "drug allergy",
        "nkda",
        "no known allergies",
        "no known drug allergies",
    ],
    # General medical terms
    "patient": ["pt", "subject", "individual"],
    "given": ["administered", "received", "prescribed", "delivered", "provided"],
    "received": ["given", "administered", "prescribed", "delivered"],
    "administered": ["given", "received", "prescribed", "delivered"],
}

# Abbreviation expansions
MEDICAL_TERM_EXPANSIONS.update({
    "behandlung": ["treatment", "therapy", "therapie"],
    "therapie": ["therapy", "treatment", "behandlung"],
    "chemotherapie": ["chemotherapy", "chemo"],
    "diagnose": ["diagnosis", "diagnosed"],
    "haemoglobin": ["hemoglobin", "h\u00e4moglobin", "hb"],
    "h\u00e4moglobin": ["hemoglobin", "haemoglobin", "hb"],
    "gewicht": ["weight", "kg"],
    "blutdruck": ["blood pressure", "systolic", "diastolic"],
    "bestrahlung": ["radiation", "radiotherapy"],
})

MEDICAL_ABBREVIATIONS: Dict[str, List[str]] = {
    "chemo": ["chemotherapy"],
    "carbo": ["carboplatin"],
    "taxol": ["paclitaxel"],
    "taxotere": ["docetaxel"],
    "adriamycin": ["doxorubicin"],
    "cytoxan": ["cyclophosphamide"],
    "htn": ["hypertension"],
    "hbp": ["high blood pressure"],
    "dm": ["diabetes mellitus"],
    "rt": ["radiation therapy", "radiotherapy"],
    "xrt": ["radiation therapy", "radiotherapy"],
    "pcr": ["pathologic complete response", "pathological complete response"],
    "rcb": ["residual cancer burden"],
    "neoadj": ["neoadjuvant"],
}

# Stopwords to filter out
_STOPWORDS = {
    "the", "and", "with", "from", "that", "this", "these", "those",
    "are", "was", "were", "for", "have", "has", "had", "does", "did",
    "into", "onto", "per", "about", "patient", "clinical", "note", "notes",
    "history", "current", "past", "any", "case", "cases", "show", "tell",
    "list", "value", "status", "result", "results", "question", "answer",
    "what", "when", "where", "which", "who", "how", "was", "were", "is",
    "was", "given", "received", "administered", "prescribed",
}

# Medical tags for query structuring (inspired by MedGraphRAG)
MEDICAL_TAGS: Dict[str, List[str]] = {
    "treatment": ["treatment", "therapy", "medication", "drug", "medication", "therapeutic", "intervention"],
    "diagnosis": ["diagnosis", "condition", "disease", "disorder", "syndrome", "diagnosed"],
    "procedure": ["surgery", "procedure", "operation", "surgical", "operative", "biopsy"],
    "temporal": ["when", "date", "time", "history", "occurred", "started", "ended", "duration"],
    "entity": ["patient", "doctor", "physician", "hospital", "clinic"],
    "measurement": ["size", "weight", "height", "dose", "dose", "amount", "level", "count"],
    "location": ["where", "site", "location", "organ", "anatomy", "region"],
    "status": ["status", "positive", "negative", "present", "absent", "yes", "no"],
    "response": ["response", "pcr", "complete response", "pathologic complete response", "residual disease"],
    "toxicity": ["toxicity", "adverse event", "side effect", "complication", "tolerated", "intolerance"],
}


def _build_medical_lexicon(extra_terms: Optional[Iterable[str]] = None) -> List[str]:
    terms: Set[str] = set()
    for term, synonyms in MEDICAL_TERM_EXPANSIONS.items():
        terms.add(term)
        terms.update(synonyms)
    for abbrev, expansions in MEDICAL_ABBREVIATIONS.items():
        terms.add(abbrev)
        terms.update(expansions)
    if extra_terms:
        terms.update(extra_terms)
    return sorted({term for term in terms if term})


def _build_symspell_terms() -> List[str]:
    terms = set(_build_medical_lexicon())
    terms.update(_STOPWORDS)
    return sorted({term for term in terms if term})


_MEDICAL_LEXICON = _build_medical_lexicon()
_SYMSPELL_TERMS = _build_symspell_terms()


def _get_question_variants(question: str) -> List[str]:
    return normalize_question_variants(question, _SYMSPELL_TERMS)


@dataclass
class StructuredQuery:
    """Structured representation of a medical question."""

    original: str
    tags: List[str]
    entities: List[str]
    intent: str  # "what", "when", "who", "yes_no", "how_much", etc.
    medical_terms: List[str]


def expand_medical_term(term: str) -> Set[str]:
    """Expand a single medical term to include synonyms and related terms."""
    term_lower = term.lower().strip()
    expansions: Set[str] = {term_lower}

    # Direct mapping
    if term_lower in MEDICAL_TERM_EXPANSIONS:
        expansions.update(MEDICAL_TERM_EXPANSIONS[term_lower])

    # Abbreviation expansion
    if term_lower in MEDICAL_ABBREVIATIONS:
        expansions.update(MEDICAL_ABBREVIATIONS[term_lower])
        # Also expand the full terms
        for full_term in MEDICAL_ABBREVIATIONS[term_lower]:
            if full_term in MEDICAL_TERM_EXPANSIONS:
                expansions.update(MEDICAL_TERM_EXPANSIONS[full_term])

    # Reverse lookup - if term is a synonym, add the main term
    for main_term, synonyms in MEDICAL_TERM_EXPANSIONS.items():
        if term_lower in synonyms:
            expansions.add(main_term)
            expansions.update(MEDICAL_TERM_EXPANSIONS.get(main_term, []))

    # Reverse lookup for abbreviations
    for abbrev, full_terms in MEDICAL_ABBREVIATIONS.items():
        if term_lower in full_terms:
            expansions.add(abbrev)
            if abbrev in MEDICAL_TERM_EXPANSIONS:
                expansions.update(MEDICAL_TERM_EXPANSIONS[abbrev])

    return expansions


def extract_medical_keywords(question: str) -> List[str]:
    """Extract medical keywords from a question for reranking.

    Returns a list of medical terms that can be used to guide reranking.
    """
    if not question:
        return []

    variants = _get_question_variants(question)

    # Tokenize question variants
    keywords: List[str] = []
    for variant in variants:
        tokens = re.findall(r"\b[a-z]{4,}\b", variant)
        for token in tokens:
            if token in _STOPWORDS:
                continue
            if len(token) < 4:
                continue
            expansions = expand_medical_term(token)
            if len(expansions) > 1 or token in MEDICAL_TERM_EXPANSIONS:
                keywords.append(token)
            elif any(char.isdigit() for char in token):
                keywords.append(token)

        phrases = re.findall(r"\b[a-z]+(?:\s+[a-z]+){1,2}\b", variant)
        for phrase in phrases:
            if len(phrase.split()) <= 3:
                expansions = expand_medical_term(phrase)
                if len(expansions) > 1:
                    keywords.append(phrase)

    # Fuzzy match terms from both original and corrected variants
    for variant in variants:
        for matched in fuzzy_match_phrases(variant, _MEDICAL_LEXICON):
            keywords.append(matched)

    return list(dict.fromkeys(keywords))


def identify_medical_tags(question: str) -> List[str]:
    """Identify medical tags from question (inspired by MedGraphRAG tag structuring)."""
    question_lower = question.lower()
    identified_tags: List[str] = []

    for tag, keywords in MEDICAL_TAGS.items():
        if any(keyword in question_lower for keyword in keywords):
            identified_tags.append(tag)

    return identified_tags


def classify_question_intent(question: str) -> str:
    """Classify question intent type."""
    question_lower = question.lower().strip()

    if question_lower.startswith(("was", "were", "did", "does", "has", "have", "is", "are")):
        return "yes_no"
    if question_lower.startswith("when"):
        return "when"
    if question_lower.startswith("where"):
        return "where"
    if question_lower.startswith("who"):
        return "who"
    if question_lower.startswith("how much") or question_lower.startswith("how many"):
        return "how_much"
    if question_lower.startswith("what"):
        return "what"
    if question_lower.startswith("how"):
        return "how"

    return "general"


def structure_query_with_tags(question: str) -> StructuredQuery:
    """Structure query with medical tags (MedGraphRAG approach).

    Returns a structured representation that can guide retrieval.
    """
    tags = identify_medical_tags(question)
    intent = classify_question_intent(question)
    medical_terms = extract_medical_keywords(question)

    # Extract entities (medical terms that are likely entities)
    entities = [term for term in medical_terms if len(term) > 4]

    return StructuredQuery(
        original=question,
        tags=tags,
        entities=entities,
        intent=intent,
        medical_terms=medical_terms,
    )


def expand_medical_query(question: str, max_terms: int = 80) -> List[str]:
    """Expand a medical question into query terms for retrieval.

    Extracts medical concepts and expands them with synonyms and related terms.
    """
    if not question:
        return []

    variants = _get_question_variants(question)

    # Extract base terms from question variants
    expanded_terms: Set[str] = set()

    for variant in variants:
        tokens = re.findall(r"[\w+/]+", variant)
        filtered = [tok for tok in tokens if tok not in _STOPWORDS and len(tok) > 2]

        for token in filtered:
            expansions = expand_medical_term(token)
            expanded_terms.update(expansions)

        phrases = re.findall(r"\b[a-z]+(?:\s+[a-z]+)+\b", variant)
        for phrase in phrases:
            if len(phrase.split()) <= 3:
                expansions = expand_medical_term(phrase)
                expanded_terms.update(expansions)

    # Fuzzy match terms from both original and corrected variants
    fuzzy_terms: Set[str] = set()
    for variant in variants:
        fuzzy_terms.update(fuzzy_match_phrases(variant, _MEDICAL_LEXICON))
    for term in fuzzy_terms:
        expanded_terms.update(expand_medical_term(term))

    # If chemotherapy is mentioned, add common chemotherapy drugs
    if any(
        term in variant
        for variant in variants
        for term in ["chemotherapy", "chemo", "chemotherapeutic"]
    ):
        common_chemo_drugs = [
            "carboplatin", "paclitaxel", "doxorubicin", "cyclophosphamide",
            "docetaxel", "taxol", "taxotere", "adriamycin", "cytoxan",
            "paraplatin", "carbo", "tax", "carbo/tax", "carbo/taxol",
        ]
        expanded_terms.update(common_chemo_drugs)

    # Prioritize medical terms over generic words
    medical_indicators = {
        "chemo", "chemotherapy", "treatment", "drug", "medication",
        "surgery", "diagnosis", "cancer", "tumor", "radiation",
        "therapy", "disease", "condition", "procedure",
    }
    generic_actions = {
        "given", "givens", "received", "receiveds", "provided",
        "provideds", "administered", "prescribed",
    }

    # Sort: medical terms first, then others, then generic actions
    medical_terms = [t for t in expanded_terms if any(ind in t for ind in medical_indicators)]
    action_terms = [t for t in expanded_terms if any(act in t for act in generic_actions)]
    other_terms = [t for t in expanded_terms if t not in medical_terms and t not in action_terms]

    ranked = list(dict.fromkeys(sorted(medical_terms) + sorted(other_terms) + sorted(action_terms)))

    return ranked[:max_terms]


__all__ = [
    "expand_medical_term",
    "extract_medical_keywords",
    "expand_medical_query",
    "identify_medical_tags",
    "classify_question_intent",
    "structure_query_with_tags",
    "StructuredQuery",
    "MEDICAL_TERM_EXPANSIONS",
    "MEDICAL_ABBREVIATIONS",
    "MEDICAL_TAGS",
]
